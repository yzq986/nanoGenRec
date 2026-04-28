#!/bin/bash
set -euo pipefail

# EXP-031: New SOTA — Features SFT + Full RL Stack
# Date: 2026-04-27
#
# 从 exp025-beam-passes（features 模型，R@500=63.6%）出发，
# 叠加完整 RL stack：
#   ECPO δ=0.1 + on-policy beam + rank_norm + A2PO(α=1.0) + NLL(0.01) + HEPO(0.1,0.5)
#
# Config A: exp025 (features) + full RL stack  → 目标新 SOTA > 66.2%
# Config B: exp020 (no features) + full RL stack → 与 EXP-030 对照，确认 features 增益
#
# 顺带修复：trainer 现已正确传 time_gaps_list/action_levels_list 给 UnifiedSequenceDataset

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
SFT_BASELINE="${CKPT_DIR}/exp020-hard-lam03"

echo "=========================================="
echo "EXP-031: New SOTA — Features SFT + Full RL Stack"
echo "=========================================="
echo "  GPUs:                  ${N_GPUS}"
echo "  NTP data:              ${NTP_DATA}"
echo "  SFT (features):        ${SFT_FEATURES}"
echo "  SFT (baseline, ablat): ${SFT_BASELINE}"
echo "  Behavior cache:        ${BEHAVIOR_CACHE}"
echo ""

# Sanity checks
if [ ! -f "${SFT_FEATURES}/probe.pt" ]; then
    echo "ERROR: features SFT checkpoint not found at ${SFT_FEATURES}"
    exit 1
fi
if [ ! -f "${SFT_BASELINE}/probe.pt" ]; then
    echo "ERROR: baseline SFT checkpoint not found at ${SFT_BASELINE}"
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
if [ ! -f "${CKPT_DIR}/exp031-smoke/probe.pt" ]; then
    echo ">>> Smoke test (2 steps, G=16, features model)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_FEATURES}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT_DIR}/exp031-smoke" \
        --name exp031-smoke \
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
    rm -rf "${CKPT_DIR}/exp031-smoke"
    echo ""
fi

# ── Config A: features SFT + full RL stack ───────────────────
NAME_A="exp031-features-full-stack"
OUTPUT_A="${CKPT_DIR}/${NAME_A}"

if [ -f "${OUTPUT_A}/probe.pt" ]; then
    echo "  [${NAME_A}] Already exists, skipping."
else
    echo ">>> Config A: ${NAME_A}"
    echo "    exp025 (features) + ECPO + on-policy beam + rank_norm + A2PO + NLL + HEPO"

    T0_A=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_FEATURES}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT_A}" \
        --name "${NAME_A}" \
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
    T1_A=$(date +%s)
    TRAIN_MIN_A=$(( (T1_A - T0_A) / 60 ))
    echo "  [${NAME_A}] Training complete  (${TRAIN_MIN_A}min)"

    echo "  [${NAME_A}] Running full eval (n_recall=1000)..."
    T2_A=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT_A}" \
        --n_recall 1000
    T3_A=$(date +%s)
    EVAL_MIN_A=$(( (T3_A - T2_A) / 60 ))
    TOTAL_MIN_A=$(( (T3_A - T0_A) / 60 ))
    echo "  [${NAME_A}] Eval complete  (${EVAL_MIN_A}min)"
    echo "  [${NAME_A}] Total: train=${TRAIN_MIN_A}min  eval=${EVAL_MIN_A}min  total=${TOTAL_MIN_A}min"
    echo ""

    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-031: ${NAME_A} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/exp031-git.lock
fi

# ── Config B: baseline SFT + full RL stack (ablation) ────────
NAME_B="exp031-baseline-full-stack"
OUTPUT_B="${CKPT_DIR}/${NAME_B}"

if [ -f "${OUTPUT_B}/probe.pt" ]; then
    echo "  [${NAME_B}] Already exists, skipping."
else
    echo ">>> Config B: ${NAME_B}"
    echo "    exp020 (no features) + ECPO + on-policy beam + rank_norm + A2PO + NLL + HEPO"
    echo "    (ablation: same RL stack as A but no features — quantifies features contribution)"

    T0_B=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_BASELINE}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT_B}" \
        --name "${NAME_B}" \
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
    T1_B=$(date +%s)
    TRAIN_MIN_B=$(( (T1_B - T0_B) / 60 ))
    echo "  [${NAME_B}] Training complete  (${TRAIN_MIN_B}min)"

    echo "  [${NAME_B}] Running full eval (n_recall=1000)..."
    T2_B=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT_B}" \
        --n_recall 1000
    T3_B=$(date +%s)
    EVAL_MIN_B=$(( (T3_B - T2_B) / 60 ))
    TOTAL_MIN_B=$(( (T3_B - T0_B) / 60 ))
    echo "  [${NAME_B}] Eval complete  (${EVAL_MIN_B}min)"
    echo "  [${NAME_B}] Total: train=${TRAIN_MIN_B}min  eval=${EVAL_MIN_B}min  total=${TOTAL_MIN_B}min"
    echo ""

    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-031: ${NAME_B} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/exp031-git.lock
fi

echo ""
echo ">>> Final results:"
for N in "${NAME_A}" "${NAME_B}"; do
    python -c "
import json, sys
try:
    m = json.load(open('${CKPT_DIR}/' + sys.argv[1] + '/train_meta.json'))
    e = m.get('eval', {})
    print(f'  {sys.argv[1]}: R@10={e.get(\"item_recall@10\",\"?\"):.3f}  R@500={e.get(\"item_recall@500\",\"?\"):.3f}')
except Exception as ex:
    print(f'  {sys.argv[1]}: eval not available ({ex})')
" "${N}" 2>/dev/null || echo "  ${N}: eval not available"
done

echo ""
echo "  Baseline (exp025-beam-passes): R@10=0.104  R@500=0.636"
echo "  Target SOTA (exp020-hard-lam03): R@10=0.141  R@500=0.662"

echo ""
echo ">>> Timing summary:"
python3 -c "
import json, os
for name in ['${NAME_A}', '${NAME_B}']:
    path = 'experiments/ntp_checkpoints/' + name + '/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        w = m.get('train', {}).get('wall_time_s', 0)
        print(f'  {name}: train={int(w)//60}min{int(w)%60}s')
    else:
        print(f'  {name}: not found')
" 2>/dev/null || true

git add experiments/
git commit -m "EXP-031 complete: features SFT + full RL stack results" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-031 complete!"
