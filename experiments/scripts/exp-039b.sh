#!/bin/bash
set -euo pipefail

# EXP-039B: ECPO on exp038b-ep1 (Features 路线第四步 — RL 链路终点)
# Date: 2026-04-28
#
# RL 对齐链路:
#   exp036-full-features (SFT)
#   → EXP-037 SP-DPO → exp037-medium
#   → EXP-038B RF-DPO → exp038b-hard-lam03-3ep-ep1 (best ep, R@500=62.1%)
#   → [本实验: ECPO δ=0.1]
#
# 对标: EXP-029 ECPO on-policy (非 features 版本，从 exp020 起跑 → 67.8%)
# EXP-038B ep1 R@500=62.1% (持平 SP-DPO ref), ep2/ep3 NTP 过拟合退化
# 验证 features + ECPO 能否超越 exp020 SOTA (R@500=66.2%)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
GRPO_BATCH="${GRPO_BATCH:-${N_GPUS}}"   # 1 batch/GPU by default
SFT_CKPT="experiments/ntp_checkpoints/exp038b-hard-lam03-3ep-ep1"   # RF-DPO ep1 (best: R@500=62.1%, matches ref)
NTP_DATA="experiments/ntp_data/exp023-14d-features"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
BEHAVIOR_CACHE="/mnt/workspace/gr-demo-behavior-cache"
CKPT_DIR="experiments/ntp_checkpoints"
DATE_END="2026-03-31"

FORCE=false
SKIP_SMOKE=false
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --no-smoke) SKIP_SMOKE=true ;;
    esac
done

echo "=========================================="
echo "EXP-039B: ECPO on exp037-medium"
echo "=========================================="
echo "  GPUs:           ${N_GPUS}"
echo "  SFT ckpt:       ${SFT_CKPT}"
echo "  NTP data:       ${NTP_DATA}"
echo "  Behavior cache: ${BEHAVIOR_CACHE}"
echo ""

# Sanity checks
if [ ! -f "${SFT_CKPT}/train_meta.json" ]; then
    echo "ERROR: exp038b-ep1 checkpoint not found at ${SFT_CKPT}"
    echo "Run exp-038b.sh first."
    exit 1
fi
if [ ! -d "${BEHAVIOR_CACHE}/${DATE_END}" ]; then
    echo "ERROR: behavior cache not found at ${BEHAVIOR_CACHE}/${DATE_END}"
    exit 1
fi

mkdir -p "${CKPT_DIR}"

# ── Smoke test ─────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" == false ]; then
    SMOKE_OUT="${CKPT_DIR}/exp039b-smoke"
    echo ""
    echo ">>> Smoke test (dry run)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_OUT}" \
        --name exp039b-smoke \
        --eps 0.2 --delta 0.1 \
        --grpo_weight 0.03 \
        --group_size 16 \
        --grpo_batch_size "${GRPO_BATCH}" \
        --rl_data_ratio 1.0 \
        --lr 1e-4 \
        --reward_behavior --behavior_weight 1.0 \
        --behavior_cache_dir "${BEHAVIOR_CACHE}" \
        --behavior_cache_eval_date "${DATE_END}" \
        --reward_format --format_weight 0.5 \
        --on_policy_beam \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${SMOKE_OUT}"
    echo ""
fi

# ── Main training: ECPO δ=0.1 ─────────────────────────────────
NAME="exp039b-ecpo-from-spdpo"
OUTPUT="${CKPT_DIR}/${NAME}"

T0=$(date +%s)
if [ -f "${OUTPUT}/probe.pt" ] && [ "${FORCE}" != true ]; then
    echo "  [${NAME}] Checkpoint found, skipping training (use --force to retrain)."
else
    echo ">>> Training: ${NAME}"
    echo "    ref=exp037-medium, ECPO δ=0.1, G=512, BehaviorReward+FormatReward, on-policy beam"
    echo ""
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT}" \
        --name "${NAME}" \
        --eps 0.2 --delta 0.1 \
        --grpo_weight 0.03 \
        --group_size 512 \
        --grpo_batch_size "${GRPO_BATCH}" \
        --rl_data_ratio 1.0 \
        --lr 1e-4 \
        --reward_behavior --behavior_weight 1.0 \
        --behavior_cache_dir "${BEHAVIOR_CACHE}" \
        --behavior_cache_eval_date "${DATE_END}" \
        --reward_format --format_weight 0.5 \
        --on_policy_beam \
        --wandb
fi
T1=$(date +%s)
TRAIN_MIN=$(( (T1 - T0) / 60 ))
echo "  Training complete  (${TRAIN_MIN}min)"

# ── Full eval ────────────────────────────────────────────────
echo ">>> Full eval (n_recall=1000)..."
T2=$(date +%s)
torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
    --checkpoint "${OUTPUT}" \
    --n_recall 1000
T3=$(date +%s)
EVAL_MIN=$(( (T3 - T2) / 60 ))
TOTAL_MIN=$(( (T3 - T0) / 60 ))
echo "  Eval complete  (${EVAL_MIN}min)"
echo "  Total: train=${TRAIN_MIN}min  eval=${EVAL_MIN}min  total=${TOTAL_MIN}min"

echo ""
echo ">>> Final comparison:"
python3 -c "
import json, os
results = [
    ('exp020-hard-lam03',                  'SOTA (no features)'),
    ('exp036-full-features',               'SFT w/ features'),
    ('exp037-medium',                      'SP-DPO'),
    ('exp038b-hard-lam03-3ep-ep1',         'RF-DPO ep1 (ref)'),
    ('${NAME}',                            'ECPO (this)'),
]
for name, desc in results:
    path = f'experiments/ntp_checkpoints/{name}/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        e = m.get('eval', {})
        r10 = e.get('item_recall@10', 0)
        r500 = e.get('item_recall@500', 0)
        ppl = e.get('ppl', 0)
        print(f'  {name:<30} ({desc:<22}): R@10={r10:.1%}  R@500={r500:.1%}  PPL={ppl:.2f}')
" 2>/dev/null || echo "  Results not available"

echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-039B results: ECPO features RL chain complete" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-039B complete! Features RL alignment chain finished."
