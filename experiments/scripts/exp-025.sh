#!/bin/bash
set -euo pipefail

# EXP-025: Beam Search Feature Passing — 正确消除 side feature 训练-推理 gap
# Date: 2026-04-21
#
# EXP-024 证明 shift 方案失败。正确做法：
#   - 不 shift 训练数据
#   - 修复 beam search incremental path，传入真实 features
#
# Configs:
#   1. seg+all+beam_passes — segment + time_gap + action，beam search 传 features
#      (复用 EXP-023 数据 exp023-14d-features，不 shift)
#   2. seg+time+action_l2  — segment + time_gap(all) + action(L2-only)
#      (新 preprocess: --action_l2_only)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

# ── Paths ──
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
NTP_DATA_ALL="experiments/ntp_data/exp023-14d-features"
NTP_DATA_L2="experiments/ntp_data/exp025-14d-action-l2only"
CKPT_DIR="experiments/ntp_checkpoints"
DATE_START="2026-03-18"
DATE_END="2026-03-31"
N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
START_FROM="${START_FROM:-0}"
FORCE="${FORCE:-false}"

echo "=========================================="
echo "EXP-025: Beam Search Feature Passing"
echo "=========================================="
echo "  SID cache:  ${SID_CACHE}"
echo "  NTP data 1: ${NTP_DATA_ALL} (reuse EXP-023)"
echo "  NTP data 2: ${NTP_DATA_L2} (action L2-only)"
echo "  Checkpoints: ${CKPT_DIR}/exp025-*"
echo "  GPUs:       ${N_GPUS}"
echo "  Date range: ${DATE_START} ~ ${DATE_END}"
echo ""

# ──────────────────────────────────────────────────────────────
# Phase 0: Preprocess (action_l2_only data for Config 2)
# ──────────────────────────────────────────────────────────────

if [ ! -f "${NTP_DATA_L2}/meta.json" ]; then
    echo ">>> Phase 0: Preprocessing NTP data (action L2-only)"
    echo "============================================================"
    python run.py preprocess-ntp \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${NTP_DATA_L2}" \
        --n_shards "${N_GPUS}" \
        --n_items 10 \
        --max_seq_len 512 \
        --n_eval_target 50000 \
        --date_start "${DATE_START}" \
        --date_end "${DATE_END}" \
        --action_l2_only
    echo "  Preprocessing complete."
    echo ""
else
    echo ">>> Phase 0: NTP data already exists at ${NTP_DATA_L2}, skipping."
    echo ""
fi

# ──────────────────────────────────────────────────────────────
# Phase 1: Smoke test (Config 1 — all features + beam passes)
# ──────────────────────────────────────────────────────────────

if [ "$START_FROM" -le 0 ]; then
    SMOKE_OUTPUT="${CKPT_DIR}/exp025-smoke"
    if [ ! -f "${SMOKE_OUTPUT}/train_meta.json" ]; then
        echo ">>> Phase 1: Smoke test (dry run, all features)"
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA_ALL}" \
            --output_dir "${SMOKE_OUTPUT}" \
            --name exp025-smoke \
            --model s-tier \
            --use_time_gap \
            --use_action_level \
            --use_segment_emb \
            --dry_run
        echo "  Smoke test PASSED"
        rm -rf "${SMOKE_OUTPUT}"
    else
        echo "  Smoke test skipped (already passed)"
    fi
    echo ""
fi

# ──────────────────────────────────────────────────────────────
# Phase 2: Training configs (DDP, serial, 8 GPUs each)
# ──────────────────────────────────────────────────────────────

run_config() {
    local NAME="$1"
    local DATA_DIR="$2"
    local TIME_GAP="$3"
    local ACTION="$4"
    local OUTPUT="${CKPT_DIR}/${NAME}"

    if [ "$FORCE" != true ] && [ -f "${OUTPUT}/train_meta.json" ]; then
        echo "  [${NAME}] Already exists, skipping."
        return 0
    fi

    local FEATURE_FLAGS="--use_segment_emb"
    [ "$TIME_GAP" = "1" ] && FEATURE_FLAGS="${FEATURE_FLAGS} --use_time_gap"
    [ "$ACTION" = "1" ] && FEATURE_FLAGS="${FEATURE_FLAGS} --use_action_level"

    echo ">>> Training: ${NAME} (segment=1, time_gap=${TIME_GAP}, action=${ACTION})"
    echo "    data: ${DATA_DIR}"
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT}" \
        --name "${NAME}" \
        --model s-tier \
        ${FEATURE_FLAGS}

    echo "  [${NAME}] Training complete, committing..."
    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-025: ${NAME} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/git-lock-exp025
    echo ""
}

echo ">>> Phase 2: Training (2 configs, serial DDP)"
echo "============================================================"
echo ""

# Baseline: exp023-segment (PPL=25.94, R@500=61.2%) — no retraining needed

# Config 1: segment + time_gap + action (all positions) — beam search will pass features
if [ "$START_FROM" -le 1 ]; then
    run_config "exp025-beam-passes" "${NTP_DATA_ALL}" "1" "1"
fi

# Config 2: segment + time_gap(all) + action(L2-only)
if [ "$START_FROM" -le 2 ]; then
    run_config "exp025-action-l2only" "${NTP_DATA_L2}" "1" "1"
fi

# ──────────────────────────────────────────────────────────────
# Final commit
# ──────────────────────────────────────────────────────────────

echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-025 results: beam search feature passing" || echo "Nothing to commit"
./push.sh

echo ""
echo "=========================================="
echo "EXP-025 complete!"
echo ""
echo "Baseline (segment-only) from EXP-023: PPL=25.94, R@500=61.2%"
echo "Compare with: exp025-beam-passes, exp025-action-l2only"
echo "=========================================="
