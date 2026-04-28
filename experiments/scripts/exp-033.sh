#!/bin/bash
set -euo pipefail

# EXP-033: Features 修复验证 — EXP-031A Rerun with Correct Feature Injection
# Date: 2026-04-28
#
# EXP-031 Config A（features 模型 exp025 + full RL stack）出现异常退化：
#   clip=0.964（vs 正常 0.924），R@500=61.8%（vs 66.2% baseline）
#
# 本 session 修复了三处 features 注入 bug：
#   1. beam search 未传 ctx_time_gaps/ctx_action_levels
#   2. compute_sid_logprobs 绕过 embed_with_features 统一入口
#   3. context_pool 未存储 time_gaps/action_levels
#
# EXP-033 用完全相同参数重跑 EXP-031 Config A，验证 clip 率是否回落到 ~0.924。
# 无需新对照组——EXP-031A 的旧结果（clip=0.964）即为对照。

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
NTP_DATA="experiments/ntp_data/exp023-14d-features"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
BEHAVIOR_CACHE="/mnt/workspace/gr-demo-behavior-cache"
DATE_END="2026-03-31"
SFT_FEATURES="${CKPT_DIR}/exp025-beam-passes"

NAME="exp033-features-fix"
OUTPUT="${CKPT_DIR}/${NAME}"

echo "=========================================="
echo "EXP-033: Features 修复验证"
echo "=========================================="
echo "  GPUs:           ${N_GPUS}"
echo "  SFT checkpoint: ${SFT_FEATURES}"
echo "  NTP data:       ${NTP_DATA}"
echo "  Output:         ${OUTPUT}"
echo "  Baseline:       EXP-031A clip=0.964, R@500=61.8%"
echo "  Target:         clip ~0.924, R@500 > 61.8%"
echo ""

# Sanity checks
if [ ! -f "${SFT_FEATURES}/probe.pt" ]; then
    echo "ERROR: features SFT checkpoint not found at ${SFT_FEATURES}"
    exit 1
fi
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo "ERROR: NTP data not found at ${NTP_DATA}"
    exit 1
fi
if [ ! -d "${BEHAVIOR_CACHE}/2026-03-31" ]; then
    echo "ERROR: behavior cache not found at ${BEHAVIOR_CACHE}"
    exit 1
fi

# ── Smoke test ────────────────────────────────────────────────
if [ ! -f "${CKPT_DIR}/exp033-smoke/probe.pt" ]; then
    echo ">>> Smoke test (2 steps, G=16, batch=2)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_FEATURES}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT_DIR}/exp033-smoke" \
        --name exp033-smoke \
        --eps 0.2 --delta 0.1 \
        --grpo_weight 0.03 \
        --group_size 16 \
        --grpo_batch_size 2 \
        --rl_data_ratio 1.0 \
        --lr 1e-4 \
        --reward_behavior --behavior_weight 1.0 \
        --behavior_cache_dir "${BEHAVIOR_CACHE}" \
        --behavior_cache_eval_date "${DATE_END}" \
        --reward_format --format_weight 0.5 \
        --on_policy_beam \
        --rank_norm \
        --a2po --a2po_alpha 1.0 \
        --nll_reg 0.01 \
        --hepo_scales "0.1,0.5" \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${CKPT_DIR}/exp033-smoke"
    echo ""
fi

# ── Main run ──────────────────────────────────────────────────
if [ -f "${OUTPUT}/probe.pt" ]; then
    echo "  [${NAME}] Already exists, skipping training."
else
    echo ">>> Training: ${NAME}  (features + full RL stack, same as EXP-031A)"

    T0=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_FEATURES}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT}" \
        --name "${NAME}" \
        --eps 0.2 --delta 0.1 \
        --grpo_weight 0.03 \
        --group_size 512 \
        --grpo_batch_size 4 \
        --rl_data_ratio 1.0 \
        --lr 1e-4 \
        --reward_behavior --behavior_weight 1.0 \
        --behavior_cache_dir "${BEHAVIOR_CACHE}" \
        --behavior_cache_eval_date "${DATE_END}" \
        --reward_format --format_weight 0.5 \
        --on_policy_beam \
        --rank_norm \
        --a2po --a2po_alpha 1.0 \
        --nll_reg 0.01 \
        --hepo_scales "0.1,0.5"
    T1=$(date +%s)
    TRAIN_MIN=$(( (T1 - T0) / 60 ))
    echo "  [${NAME}] Training complete  (${TRAIN_MIN}min)"
fi

# ── Full eval ─────────────────────────────────────────────────
echo ""
echo ">>> Full eval (n_recall=1000)..."
T2=$(date +%s)
torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
    --checkpoint "${OUTPUT}" \
    --n_recall 1000
T3=$(date +%s)
EVAL_MIN=$(( (T3 - T2) / 60 ))
TOTAL_MIN=$(( (T3 - T0) / 60 ))
echo "  [${NAME}] Eval complete  (${EVAL_MIN}min)"
echo "  [${NAME}] Total: train=${TRAIN_MIN}min  eval=${EVAL_MIN}min  total=${TOTAL_MIN}min"

# ── Results summary ───────────────────────────────────────────
echo ""
echo ">>> Results vs EXP-031A baseline:"
python3 -c "
import json, os
path = 'experiments/ntp_checkpoints/${NAME}/train_meta.json'
try:
    m = json.load(open(path))
    e = m.get('eval', {})
    t = m.get('train', {})
    clip = t.get('avg_clip_fraction', float('nan'))
    adv_std = t.get('avg_advantage_std', float('nan'))
    wall = t.get('wall_time_s', 0)
    print(f'  EXP-033 (features fixed): R@10={e.get(\"item_recall@10\",float(\"nan\")):.3f}  R@500={e.get(\"item_recall@500\",float(\"nan\")):.3f}  clip={clip:.3f}  adv_std={adv_std:.3f}  train={int(wall)//60}min')
    print(f'  EXP-031A (features bug):  R@10=0.111                    R@500=0.618                    clip=0.964')
    print(f'  EXP-031B (no features):   R@10=0.125                    R@500=0.677                    clip=0.924')
    if clip < 0.94:
        print('  -> clip 率正常，features bug 已确认修复')
    else:
        print('  -> clip 率仍偏高，可能有其他因素')
except Exception as ex:
    print(f'  not available: {ex}')
" 2>/dev/null || true

# ── Timing summary ────────────────────────────────────────────
echo ""
echo ">>> Timing summary:"
python3 -c "
import json, os
path = 'experiments/ntp_checkpoints/${NAME}/train_meta.json'
if os.path.exists(path):
    m = json.load(open(path))
    w = m.get('train', {}).get('wall_time_s', 0)
    print(f'  ${NAME}: train={int(w)//60}min{int(w)%60:02d}s')
" 2>/dev/null || true

git add experiments/
git commit -m "EXP-033 complete: features fix validation results" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-033 complete!"
