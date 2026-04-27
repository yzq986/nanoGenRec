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
BEHAVIOR_CACHE="/mnt/workspace/gr-demo-behavior-cache"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
NTP_DATA="experiments/ntp_data/exp023-14d-features"
RF_FEEDBACK_DIR="experiments/rf_dpo_data/exp018/hard"   # for BehaviorReward
CKPT_DIR="experiments/ntp_checkpoints"
SFT_BASE="${CKPT_DIR}/exp016-B-14d-S"
DATE_START="2026-03-18"
DATE_END="2026-03-31"
N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
START_FROM="${START_FROM:-0}"
FORCE="${FORCE:-false}"

echo "=========================================="
echo "EXP-026: GRPO+ECPO + Pluggable Reward"
echo "=========================================="
echo "  Behavior cache: ${BEHAVIOR_CACHE}"
echo "  SID cache:      ${SID_CACHE}"
echo "  NTP data:       ${NTP_DATA}"
echo "  SFT base:       ${SFT_BASE}"
echo "  Checkpoints:    ${CKPT_DIR}/exp026-*"
echo "  GPUs:           ${N_GPUS}"
echo "  Date range:     ${DATE_START} ~ ${DATE_END}"
echo ""

# ──────────────────────────────────────────────────────────────
# Phase -3: preprocess-sid (SID cache)
# ──────────────────────────────────────────────────────────────
if [ -f "${SID_CACHE}/semantic_ids.npy" ]; then
    echo ">>> Phase -3: SID cache found, skipping preprocess-sid"
else
    echo ">>> Phase -3: preprocess-sid (4096×3, FSQ [2]×12 binary)"
    echo "============================================================"
    python run.py preprocess-sid \
        --model qwen3-0.6b \
        --behavior_path "${BEHAVIOR_CACHE}" \
        --date_start "${DATE_START}" \
        --date_end "${DATE_END}" \
        --output_dir "${SID_CACHE}" \
        --num_clusters 4096 \
        --fsq_levels 12d_4096 \
        --fsq_projection mlp \
        --fsq_mlp_hidden 64 \
        --fsq_epochs 50
    echo "  preprocess-sid DONE"
    echo ""
fi

# ──────────────────────────────────────────────────────────────
# Phase -2: preprocess-ntp (NTP data shards)
# ──────────────────────────────────────────────────────────────
if [ -f "${NTP_DATA}/meta.json" ]; then
    echo ">>> Phase -2: NTP data found, skipping preprocess-ntp"
else
    echo ">>> Phase -2: preprocess-ntp (14d, ${DATE_START}~${DATE_END})"
    echo "============================================================"
    python run.py preprocess-ntp \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${NTP_DATA}" \
        --n_shards "${N_GPUS}" \
        --date_start "${DATE_START}" \
        --date_end "${DATE_END}"
    if [ ! -f "${NTP_DATA}/meta.json" ]; then
        echo "ERROR: preprocess-ntp did not produce meta.json"
        exit 1
    fi
    echo "  preprocess-ntp DONE"
    echo ""
fi

# ──────────────────────────────────────────────────────────────
# Phase -1: train-ntp (SFT base: exp016-B-14d-S)
# ──────────────────────────────────────────────────────────────
if [ -f "${SFT_BASE}/probe.pt" ]; then
    echo ">>> Phase -1: SFT base found at ${SFT_BASE}, skipping train-ntp"
else
    echo ">>> Phase -1: train-ntp (S-tier, 14d data) → exp016-B-14d-S"
    echo "============================================================"
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SFT_BASE}" \
        --model s-tier \
        --batch_size 128 \
        --lr 1e-3 \
        --embed_dim 256 \
        --n_heads 8 \
        --n_transformer_layers 6 \
        --n_experts 8 \
        --top_k 2 \
        --expert_dim 1024 \
        --name exp016-B-14d-S
    if [ ! -f "${SFT_BASE}/probe.pt" ]; then
        echo "ERROR: train-ntp did not produce probe.pt"
        exit 1
    fi
    echo "  train-ntp DONE"
    git add experiments/
    git commit -m "EXP-026 prereq: exp016-B-14d-S SFT base trained" || echo "Nothing to commit"
    ./push.sh
    echo ""
fi

# ── Auto-select best available RF-DPO checkpoint on top of SFT base ──
if [ -f "${CKPT_DIR}/exp020-hard-lam03/probe.pt" ]; then
    SFT_CKPT="${CKPT_DIR}/exp020-hard-lam03"
    echo "  Using SFT: exp020-hard-lam03 (RF-DPO hard best)"
elif [ -f "${CKPT_DIR}/exp019-joint-hard-lam10/probe.pt" ]; then
    SFT_CKPT="${CKPT_DIR}/exp019-joint-hard-lam10"
    echo "  Using SFT: exp019-joint-hard-lam10 (RF-DPO fallback)"
else
    SFT_CKPT="${SFT_BASE}"
    echo "  Using SFT: exp016-B-14d-S (NTP base)"
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
