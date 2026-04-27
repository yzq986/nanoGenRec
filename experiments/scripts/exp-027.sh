#!/bin/bash
set -euo pipefail

# EXP-027: ECPO grpo_weight Sweep — Align with RF-DPO Training Structure
#
# EXP-026 结论：grpo_weight=0.5 导致 R@500 63%→19%，RL 严重损害 NTP。
# 根因：RF-DPO 用 λ=0.03 每步必触发；GRPO 用 weight=0.5 稀疏触发，
# 触发步 GRPO 梯度占比 ~22%，单步 gnorm spike 到 158。
#
# 本次 sweep 三个方向：
#   A: weight=0.03, ratio=1.0  — 完全对齐 RF-DPO（每步必触发，weight=0.03）
#   B: weight=0.03, ratio=0.5  — 每步 50% 触发，介于 A/C 之间
#   C: weight=0.03, ratio=0.02 — 保持稀疏触发，只降 weight
#
# 全部用 ECPO (δ=0.1)，已证明比 GRPO 稳定（EXP-026）
# lr=1e-4 对齐 RF-DPO，SFT=exp020-hard-lam03

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
NTP_DATA="experiments/ntp_data/exp023-14d-features"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
RF_FEEDBACK_DIR="experiments/rf_dpo_data/exp018/hard"
DATE_START="2026-03-18"
DATE_END="2026-03-31"
SFT_CKPT="${CKPT_DIR}/exp020-hard-lam03"

echo "=========================================="
echo "EXP-027: ECPO grpo_weight Sweep"
echo "=========================================="
echo "  GPUs:           ${N_GPUS}"
echo "  NTP data:       ${NTP_DATA}"
echo "  SFT checkpoint: ${SFT_CKPT}"
echo "  Feedback dir:   ${RF_FEEDBACK_DIR}"
echo ""

# Sanity checks
if [ ! -f "${SFT_CKPT}/probe.pt" ]; then
    echo "ERROR: SFT checkpoint not found at ${SFT_CKPT}"
    exit 1
fi
if [ ! -f "${RF_FEEDBACK_DIR}/meta.json" ]; then
    echo "ERROR: feedback dir not found at ${RF_FEEDBACK_DIR}"
    exit 1
fi
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo "ERROR: NTP data not found at ${NTP_DATA}"
    exit 1
fi

# ── Smoke test ────────────────────────────────────────────────
if [ ! -f "${CKPT_DIR}/exp027-smoke/probe.pt" ]; then
    echo ">>> Smoke test (2 steps, G=16)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT_DIR}/exp027-smoke" \
        --name exp027-smoke \
        --eps 0.2 --delta 0.1 \
        --grpo_weight 0.03 \
        --group_size 16 \
        --grpo_batch_size 2 \
        --rl_data_ratio 1.0 \
        --lr 1e-4 \
        --reward_behavior --behavior_weight 1.0 \
        --reward_format --format_weight 0.5 \
        --feedback_dir "${RF_FEEDBACK_DIR}" \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${CKPT_DIR}/exp027-smoke"
    echo ""
fi

# ── Training helper ───────────────────────────────────────────
run_config() {
    local NAME="$1"
    local WEIGHT="$2"
    local RATIO="$3"
    local OUTPUT="${CKPT_DIR}/${NAME}"

    if [ -f "${OUTPUT}/probe.pt" ]; then
        echo "  [${NAME}] Already exists, skipping."
        return
    fi

    echo ">>> Training: ${NAME}"
    echo "    grpo_weight=${WEIGHT}  rl_data_ratio=${RATIO}  (ECPO δ=0.1, lr=1e-4)"

    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT}" \
        --name "${NAME}" \
        --eps 0.2 --delta 0.1 \
        --grpo_weight "${WEIGHT}" \
        --group_size 512 \
        --grpo_batch_size 4 \
        --rl_data_ratio "${RATIO}" \
        --lr 1e-4 \
        --reward_behavior --behavior_weight 1.0 \
        --reward_format --format_weight 0.5 \
        --feedback_dir "${RF_FEEDBACK_DIR}"

    echo "  [${NAME}] Training complete"

    # Full eval (aligned with baseline)
    echo "  [${NAME}] Running full eval (n_recall=1000)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT}" \
        --n_recall 1000

    echo "  [${NAME}] Eval complete"
    echo ""

    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-027: ${NAME} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/exp027-git.lock
}

# ── Config A: weight=0.03, ratio=1.0 (RF-DPO aligned) ────────
run_config "exp027-ecpo-w003-r100" 0.03 1.0

# ── Config B: weight=0.03, ratio=0.5 ─────────────────────────
run_config "exp027-ecpo-w003-r050" 0.03 0.5

# ── Config C: weight=0.03, ratio=0.02 (sparse, low weight) ───
run_config "exp027-ecpo-w003-r002" 0.03 0.02

echo ""
echo ">>> All configs complete. Final results:"
for NAME in exp027-ecpo-w003-r100 exp027-ecpo-w003-r050 exp027-ecpo-w003-r002; do
    python -c "
import json
m = json.load(open('${CKPT_DIR}/${NAME}/train_meta.json'))
e = m.get('eval', {})
print(f'  ${NAME}: R@10={e.get(\"item_recall@10\",\"?\"):.3f}  R@500={e.get(\"item_recall@500\",\"?\"):.3f}')
" 2>/dev/null || echo "  ${NAME}: eval not available"
done

git add experiments/
git commit -m "EXP-027 complete: ECPO grpo_weight sweep results" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-027 complete!"
