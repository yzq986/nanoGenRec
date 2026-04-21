#!/bin/bash
set -euo pipefail

# EXP-024: Side Feature Shift — 消除 time_gap/action_level 信息泄漏
# Date: 2026-04-21
#
# EXP-023 发现 side features 复制到 target item token 位置导致训练-推理泄漏。
# 本实验将 features 延迟一个 item（每个 item 的 3 token 使用上一个 item 的 features）。
# Segment embedding 不受影响，作为 baseline。
#
# Configs (all include segment_emb):
#   1. segment-only   — baseline (复用 EXP-023 结果，此处跳过)
#   2. seg+timegap    — segment + shifted time_gap
#   3. seg+action     — segment + shifted action_level
#   4. seg+all        — segment + shifted time_gap + shifted action_level

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
cd "${REPO_ROOT}"

# ── Paths (grep'd from exp-023.sh) ──
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
NTP_DATA="experiments/ntp_data/exp024-14d-shifted"
CKPT_DIR="experiments/ntp_checkpoints"
DATE_START="2026-03-18"
DATE_END="2026-03-31"
N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
START_FROM="${START_FROM:-0}"
FORCE="${FORCE:-false}"

echo "=========================================="
echo "EXP-024: Side Feature Shift"
echo "=========================================="
echo "  SID cache:  ${SID_CACHE}"
echo "  NTP data:   ${NTP_DATA}"
echo "  Checkpoints: ${CKPT_DIR}/exp024-*"
echo "  GPUs:       ${N_GPUS}"
echo "  Date range: ${DATE_START} ~ ${DATE_END}"
echo ""

# ──────────────────────────────────────────────────────────────
# Phase 0: Preprocess (with shifted features)
# ──────────────────────────────────────────────────────────────

if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo ">>> Phase 0: Preprocessing NTP data (shifted features)"
    echo "============================================================"
    python run.py preprocess-ntp \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${NTP_DATA}" \
        --n_shards "${N_GPUS}" \
        --n_items 10 \
        --max_seq_len 512 \
        --n_eval_target 50000 \
        --date_start "${DATE_START}" \
        --date_end "${DATE_END}" \
        --shift_features
    echo "  Preprocessing complete."
    echo ""
else
    echo ">>> Phase 0: NTP data already exists at ${NTP_DATA}, skipping."
    echo ""
fi

# ──────────────────────────────────────────────────────────────
# Phase 1: Smoke test (all features on)
# ──────────────────────────────────────────────────────────────

if [ "$START_FROM" -le 0 ]; then
    SMOKE_OUTPUT="${CKPT_DIR}/exp024-smoke"
    if [ ! -f "${SMOKE_OUTPUT}/train_meta.json" ]; then
        echo ">>> Phase 1: Smoke test (dry run, all features)"
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${SMOKE_OUTPUT}" \
            --name exp024-smoke \
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
    local TIME_GAP="$2"
    local ACTION="$3"
    local OUTPUT="${CKPT_DIR}/${NAME}"

    if [ "$FORCE" != true ] && [ -f "${OUTPUT}/train_meta.json" ]; then
        echo "  [${NAME}] Already exists, skipping."
        return 0
    fi

    local FEATURE_FLAGS="--use_segment_emb"
    [ "$TIME_GAP" = "1" ] && FEATURE_FLAGS="${FEATURE_FLAGS} --use_time_gap"
    [ "$ACTION" = "1" ] && FEATURE_FLAGS="${FEATURE_FLAGS} --use_action_level"

    echo ">>> Training: ${NAME} (segment=1, time_gap=${TIME_GAP}, action=${ACTION})"
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT}" \
        --name "${NAME}" \
        --model s-tier \
        ${FEATURE_FLAGS}

    echo "  [${NAME}] Training complete, committing..."
    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-024: ${NAME} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/git-lock-exp024
    echo ""
}

echo ">>> Phase 2: Shifted feature training (3 configs, serial DDP)"
echo "============================================================"
echo ""

# Note: segment-only baseline reuses EXP-023 result (exp023-segment)
# PPL=25.94, R@500=61.2%

# Config 1: segment + shifted time_gap
if [ "$START_FROM" -le 1 ]; then
    run_config "exp024-seg-timegap" "1" "0"
fi

# Config 2: segment + shifted action_level
if [ "$START_FROM" -le 2 ]; then
    run_config "exp024-seg-action" "0" "1"
fi

# Config 3: segment + shifted time_gap + shifted action_level
if [ "$START_FROM" -le 3 ]; then
    run_config "exp024-seg-all" "1" "1"
fi

# ──────────────────────────────────────────────────────────────
# Final commit
# ──────────────────────────────────────────────────────────────

echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-024 results: side feature shift" || echo "Nothing to commit"
./push.sh

echo ""
echo "=========================================="
echo "EXP-024 complete!"
echo ""
echo "Baseline (segment-only) from EXP-023: PPL=25.94, R@500=61.2%"
echo "Compare with: exp024-seg-timegap, exp024-seg-action, exp024-seg-all"
echo "=========================================="
