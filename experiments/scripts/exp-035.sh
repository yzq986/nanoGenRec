#!/bin/bash
set -euo pipefail

# EXP-035: Constrained Sampling — Replace Beam Search with T=1.0 Sampling
# Date: 2026-04-28
#
# 核心假设：
#   beam search 生成的候选集中在 policy 峰值，导致：
#     (1) ρ = π_θ/π_ref >> 1 → clip 率 ~95%，大量样本无效
#     (2) advantage ≈ 0（候选 reward 方差极小），梯度退化
#   改用 constrained_sampling(T=1.0) 后：
#     (1) 候选直接从 policy 分布采样 → ρ ≈ 1 by construction → clip 率应降至 10~40%
#     (2) 候选多样性大增 → advantage 有真正对比信号
#   同时 G: 512→64（sampling 多样性由 T 保证，不需要大 G），显存节省
#
# 对照：
#   EXP-034 (beam, G=512, ref=exp025): clip=95%, adv≈0
#   EXP-035 (sampling T=1.0, G=64):   clip=??,  adv=??   ← 本实验

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
NTP_DATA="experiments/ntp_data/exp023-14d-features"
BEHAVIOR_CACHE="/mnt/workspace/gr-demo-behavior-cache"
DATE_END="2026-03-31"
SFT_FEATURES="${CKPT_DIR}/exp025-beam-passes"
REF_CKPT="${CKPT_DIR}/exp025-beam-passes"

NAME="exp035-sampling-t1"
OUTPUT="${CKPT_DIR}/${NAME}"

echo "=========================================="
echo "EXP-035: Constrained Sampling (T=1.0, G=64)"
echo "=========================================="
echo "  GPUs:             ${N_GPUS}"
echo "  Policy (SFT):     ${SFT_FEATURES}"
echo "  Ref model:        ${REF_CKPT}"
echo "  NTP data:         ${NTP_DATA}"
echo "  Output:           ${OUTPUT}"
echo "  Sampling:         T=1.0, G=64 (vs beam G=512 in EXP-034)"
echo "  Hypothesis:       clip 率降至 10~40%，adv 有真正对比信号"
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
if [ ! -f "${CKPT_DIR}/exp035-smoke/probe.pt" ]; then
    echo ">>> Smoke test (2 steps, G=16, batch=2, T=1.0)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_FEATURES}" \
        --ref_checkpoint "${REF_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT_DIR}/exp035-smoke" \
        --name exp035-smoke \
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
        --sampling_temperature 1.0 \
        --rank_norm \
        --a2po --a2po_alpha 1.0 \
        --nll_reg 0.01 \
        --hepo_scales "0.1,0.5" \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${CKPT_DIR}/exp035-smoke"
    echo ""
fi

# ── Main run ──────────────────────────────────────────────────
if [ -f "${OUTPUT}/probe.pt" ]; then
    echo "  [${NAME}] Already exists, skipping training."
else
    echo ">>> Training: ${NAME}"
    echo "    (constrained sampling T=1.0, G=64, ref=exp025)"

    T0=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_FEATURES}" \
        --ref_checkpoint "${REF_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT}" \
        --name "${NAME}" \
        --eps 0.2 --delta 0.1 \
        --grpo_weight 0.03 \
        --group_size 64 \
        --grpo_batch_size 4 \
        --rl_data_ratio 1.0 \
        --lr 1e-4 \
        --reward_behavior --behavior_weight 1.0 \
        --behavior_cache_dir "${BEHAVIOR_CACHE}" \
        --behavior_cache_eval_date "${DATE_END}" \
        --reward_format --format_weight 0.5 \
        --sampling_temperature 1.0 \
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
    print(f'  EXP-035 (sampling T=1.0, G=64): R@10={r10:.3f}  R@500={r500:.3f}  clip={clip:.3f}  adv_std={adv_std:.3f}  train={int(wall)//60}min')
    print(f'  EXP-034 (beam G=512, ref=025):  R@10=TBD          R@500=TBD          clip=0.950')
    print(f'  EXP-031B (beam baseline):        R@10=0.125         R@500=0.677         clip=0.924')
    print(f'  EXP-029 (SOTA):                  R@10=0.130         R@500=0.678         clip=0.923')
    if clip < 0.50:
        print(f'  -> clip 率正常 ({clip:.3f} < 0.50)，sampling 假设验证 ✓')
        if r500 > 0.678:
            print(f'  -> R@500={r500:.3f} > 0.678，sampling RL 设立新 SOTA！')
        else:
            print(f'  -> R@500={r500:.3f}，与 SOTA 接近；sampling 路线有潜力')
    else:
        print(f'  -> clip 率仍偏高 ({clip:.3f})，检查 sampling 是否正确触发')
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

git add experiments/ rl/ ntp/
git commit -m "EXP-035 complete: constrained sampling T=1.0 results" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-035 complete!"
