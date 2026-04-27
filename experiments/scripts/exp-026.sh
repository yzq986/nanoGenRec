#!/bin/bash
set -euo pipefail

# EXP-026: GRPO+ECPO — Group Relative Policy Optimization + Pluggable Reward
# Date: 2026-04-27
#
# Phase 3: GRPO — group-normalized advantage (G=512), PPO clipped surrogate,
#          BehaviorReward + FormatReward, rl_data_ratio=2%
# Phase 4: ECPO — GRPO + early clip (delta=0.1) for negative-advantage stability
#
# SFT starting point (priority order):
#   1. exp020-hard-lam03  — best RF-DPO hard checkpoint (if available)
#   2. exp019-joint-hard-lam10 — RF-DPO joint hard (fallback)
#   3. exp016-B-14d-S     — NTP base (final fallback, always present)
#
# Configs:
#   1. grpo-behavior       GRPO, BehaviorReward only (baseline)
#   2. grpo-behavior-fmt   GRPO, BehaviorReward + FormatReward
#   3. ecpo-behavior-fmt   ECPO (delta=0.1), BehaviorReward + FormatReward

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

# ── Paths ──
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
NTP_DATA="experiments/ntp_data/exp023-14d-features"
RF_FEEDBACK_DIR="experiments/rf_dpo_data/hard"   # for BehaviorReward
CKPT_DIR="experiments/ntp_checkpoints"
DATE_START="2026-03-18"
DATE_END="2026-03-31"
N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
START_FROM="${START_FROM:-0}"
FORCE="${FORCE:-false}"

# ── Auto-select SFT checkpoint (best available RF-DPO → NTP base fallback) ──
if [ -f "${CKPT_DIR}/exp020-hard-lam03/probe.pt" ]; then
    SFT_CKPT="${CKPT_DIR}/exp020-hard-lam03"
    echo "  Using SFT: exp020-hard-lam03 (RF-DPO hard best)"
elif [ -f "${CKPT_DIR}/exp019-joint-hard-lam10/probe.pt" ]; then
    SFT_CKPT="${CKPT_DIR}/exp019-joint-hard-lam10"
    echo "  Using SFT: exp019-joint-hard-lam10 (RF-DPO fallback)"
elif [ -f "${CKPT_DIR}/exp016-B-14d-S/probe.pt" ]; then
    SFT_CKPT="${CKPT_DIR}/exp016-B-14d-S"
    echo "  Using SFT: exp016-B-14d-S (NTP base fallback)"
else
    echo "ERROR: No SFT checkpoint found. Run one of: exp016.sh, exp019.sh, exp020.sh first."
    exit 1
fi

echo "=========================================="
echo "EXP-026: GRPO+ECPO + Pluggable Reward"
echo "=========================================="
echo "  SFT checkpoint: ${SFT_CKPT}"
echo "  NTP data:       ${NTP_DATA}"
echo "  SID cache:      ${SID_CACHE}"
echo "  Checkpoints:    ${CKPT_DIR}/exp026-*"
echo "  GPUs:           ${N_GPUS}"
echo "  Date range:     ${DATE_START} ~ ${DATE_END}"
echo ""

# ── Preflight checks ──
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo "ERROR: NTP data not found at ${NTP_DATA}/meta.json"
    echo "  Run exp-023.sh first, or pull from public remote."
    exit 1
fi

# ──────────────────────────────────────────────────────────────
# Phase 0: Smoke test (dry run, 2 steps)
# ──────────────────────────────────────────────────────────────

if [ "$START_FROM" -le 0 ]; then
    echo ">>> Phase 0: Smoke test (dry run, 2 steps)"
    echo "============================================================"
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT_DIR}/exp026-smoke" \
        --name exp026-smoke \
        --grpo_weight 0.5 \
        --group_size 16 \
        --grpo_batch_size 2 \
        --rl_data_ratio 1.0 \
        --reward_format \
        --reward_format_k 5 \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${CKPT_DIR}/exp026-smoke"
    echo ""
fi

# ──────────────────────────────────────────────────────────────
# Training helper
# ──────────────────────────────────────────────────────────────

run_config() {
    local NAME="$1"
    local ALGO_FLAGS="$2"
    local REWARD_FLAGS="$3"
    local OUTPUT="${CKPT_DIR}/${NAME}"

    if [ "$FORCE" != "true" ] && [ -f "${OUTPUT}/train_meta.json" ]; then
        echo "  [${NAME}] Already exists, skipping."
        return 0
    fi

    echo ">>> Training: ${NAME}"
    echo "    algo:   ${ALGO_FLAGS}"
    echo "    reward: ${REWARD_FLAGS}"
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT}" \
        --name "${NAME}" \
        --grpo_weight 0.5 \
        --group_size 512 \
        --grpo_batch_size 4 \
        --rl_data_ratio 0.02 \
        --lr 5e-5 \
        ${ALGO_FLAGS} \
        ${REWARD_FLAGS}

    echo "  [${NAME}] Training complete, committing..."
    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-026: ${NAME} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/git-lock-exp026
    echo ""
}

echo ">>> Phase 1-3: Training configs (serial DDP, 8 GPUs each)"
echo "============================================================"
echo ""

# Config 1: GRPO, BehaviorReward only
if [ "$START_FROM" -le 1 ]; then
    run_config "exp026-grpo-behavior" \
        "--eps 0.2" \
        "--reward_behavior --behavior_weight 1.0 --feedback_dir ${RF_FEEDBACK_DIR}"
fi

# Config 2: GRPO, BehaviorReward + FormatReward
if [ "$START_FROM" -le 2 ]; then
    run_config "exp026-grpo-behavior-fmt" \
        "--eps 0.2" \
        "--reward_behavior --behavior_weight 1.0 --reward_format --format_weight 0.5 --feedback_dir ${RF_FEEDBACK_DIR}"
fi

# Config 3: ECPO (delta=0.1), BehaviorReward + FormatReward
if [ "$START_FROM" -le 3 ]; then
    run_config "exp026-ecpo-behavior-fmt" \
        "--eps 0.2 --delta 0.1" \
        "--reward_behavior --behavior_weight 1.0 --reward_format --format_weight 0.5 --feedback_dir ${RF_FEEDBACK_DIR}"
fi

# ──────────────────────────────────────────────────────────────
# Final commit
# ──────────────────────────────────────────────────────────────

echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-026 results: GRPO+ECPO + Pluggable Reward" || echo "Nothing to commit"
./push.sh

echo ""
echo "=========================================="
echo "EXP-026 complete!"
echo "  Results:"
echo "    ${CKPT_DIR}/exp026-grpo-behavior/train_meta.json"
echo "    ${CKPT_DIR}/exp026-grpo-behavior-fmt/train_meta.json"
echo "    ${CKPT_DIR}/exp026-ecpo-behavior-fmt/train_meta.json"
echo "=========================================="
