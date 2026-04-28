#!/bin/bash
set -euo pipefail

# EXP-034: Ref Model Alignment — exp025 as ref_checkpoint
# Date: 2026-04-28
#
# 根因假设（EXP-033 证伪 features bug 后确认）：
#   exp031-features 和 exp033 clip 率高达 96% 的真正原因是
#   ref model (exp020) ≠ policy 起点 (exp025)。
#   从 exp025 出发做 RL 时，第一步就已经大量触发 clip，
#   不是因为更新过大，而是两个模型对同一 token 的 log-prob 系统性不同。
#
# 修法：令 ref_checkpoint = sft_checkpoint = exp025
#   这样 RL 开始时 KL=0，clip 只在真正更新过大时才触发。
#
# 对照：
#   exp031-baseline: policy=exp020, ref=exp020 → clip=92.4%, R@500=67.7%
#   exp031-features: policy=exp025, ref=exp020 → clip=96.4%, R@500=61.8%  ← bug
#   exp033:          policy=exp025, ref=exp020 → clip=96.2%, R@500=61.0%  ← same bug
#   EXP-034:         policy=exp025, ref=exp025 → clip≈92%?,  R@500=??    ← 本实验

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
NTP_DATA="experiments/ntp_data/exp023-14d-features"
BEHAVIOR_CACHE="/mnt/workspace/gr-demo-behavior-cache"
DATE_END="2026-03-31"
SFT_FEATURES="${CKPT_DIR}/exp025-beam-passes"  # policy 起点
REF_CKPT="${CKPT_DIR}/exp025-beam-passes"       # ref model = 同一个 checkpoint

NAME="exp034-ref-aligned"
OUTPUT="${CKPT_DIR}/${NAME}"

echo "=========================================="
echo "EXP-034: Ref Model Alignment (exp025 as ref)"
echo "=========================================="
echo "  GPUs:             ${N_GPUS}"
echo "  Policy (SFT):     ${SFT_FEATURES}"
echo "  Ref model:        ${REF_CKPT}  (same as policy → KL=0 at step 0)"
echo "  NTP data:         ${NTP_DATA}"
echo "  Output:           ${OUTPUT}"
echo "  Hypothesis:       clip 率应回落至 ~92%，R@500 应超过 66.2%"
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
if [ ! -f "${CKPT_DIR}/exp034-smoke/probe.pt" ]; then
    echo ">>> Smoke test (2 steps, G=16, batch=2)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_FEATURES}" \
        --ref_checkpoint "${REF_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT_DIR}/exp034-smoke" \
        --name exp034-smoke \
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
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${CKPT_DIR}/exp034-smoke"
    echo ""
fi

# ── Main run ──────────────────────────────────────────────────
if [ -f "${OUTPUT}/probe.pt" ]; then
    echo "  [${NAME}] Already exists, skipping training."
else
    echo ">>> Training: ${NAME}"
    echo "    (features SFT + ECPO, ref=exp025 aligned)"

    T0=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_FEATURES}" \
        --ref_checkpoint "${REF_CKPT}" \
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
echo ">>> Results vs reference experiments:"
python3 -c "
import json, os
path = 'experiments/ntp_checkpoints/${NAME}/train_meta.json'
try:
    m = json.load(open(path))
    e = m.get('eval', {})
    t = m.get('train', {})
    clip = t.get('avg_clip_fraction', float('nan'))
    adv_std = t.get('avg_advantage_std', float('nan'))
    r10  = e.get('item_recall@10', float('nan'))
    r500 = e.get('item_recall@500', float('nan'))
    wall = t.get('wall_time_s', 0)
    print(f'  EXP-034 (ref=exp025): R@10={r10:.3f}  R@500={r500:.3f}  clip={clip:.3f}  adv_std={adv_std:.3f}  train={int(wall)//60}min')
    print(f'  EXP-033 (ref=exp020): R@10=0.103         R@500=0.610         clip=0.962')
    print(f'  EXP-031B (baseline):  R@10=0.125         R@500=0.677         clip=0.924')
    print(f'  EXP-029 (SOTA):       R@10=0.130         R@500=0.678         clip=0.923')
    if clip < 0.94:
        print(f'  -> clip 率正常 ({clip:.3f} < 0.94)，ref/policy 对齐假设验证 ✓')
        if r500 > 0.677:
            print(f'  -> R@500={r500:.3f} > 0.677，features RL 设立新 SOTA！')
        else:
            print(f'  -> R@500={r500:.3f}，低于 SOTA 0.678；features 路线仍需调参')
    else:
        print(f'  -> clip 率仍偏高 ({clip:.3f})，可能有其他因素，建议扩大 epsilon 或降低 lr')
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

git add experiments/ rl/
git commit -m "EXP-034 complete: ref model alignment results" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-034 complete!"
echo ">>> Starting EXP-032..."
bash experiments/scripts/exp-032.sh
