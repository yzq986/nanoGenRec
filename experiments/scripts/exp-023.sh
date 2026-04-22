#!/usr/bin/env bash
# ============================================================
# EXP-023: NTP Side Information — Time Gap + Action Type + Segment Embedding
# Date: 2026-04-21
#
# Validate three P0 additive features (IDEA-feat-0/1/2) independently
# and in combination. All features are cheap additive embeddings that
# don't alter sequence structure.
#
# Variable: side feature combination (5 configs)
# Baseline: EXP-016 B-14d-S (PPL=27.05, R@500=58.5%)
#
# Prerequisites:
#   - SID cache: experiments/sid_cache/exp013-4096x3-12d-binary
#   - Behavior data: 2026-03-18 ~ 2026-03-31 (14d)
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SKIP_SMOKE=false
FORCE=false
START_FROM=0
for arg in "$@"; do
    case "$arg" in
        --no-smoke) SKIP_SMOKE=true ;;
        --force) FORCE=true ;;
        --start-from=*) START_FROM="${arg#*=}" ;;
    esac
done

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
NTP_DATA="experiments/ntp_data/exp023-14d-features"
CKPT_DIR="experiments/ntp_checkpoints"
DATE_START="2026-03-18"
DATE_END="2026-03-31"

echo "============================================================"
echo "EXP-023: NTP Side Information (Time Gap + Action + Segment)"
echo "============================================================"
echo "  GPUs: ${N_GPUS}"
echo "  SID cache: ${SID_CACHE}"
echo "  NTP data: ${NTP_DATA}"
echo "  Date range: ${DATE_START} ~ ${DATE_END}"
echo ""

# ── Verify SID cache ──
if [ ! -f "${SID_CACHE}/semantic_ids.npy" ]; then
    echo "ERROR: SID cache not found at ${SID_CACHE}"
    echo "Run exp-013.sh first to generate SIDs."
    exit 1
fi

# ──────────────────────────────────────────────────────────────
# Phase 0: Preprocess NTP data (with time_gaps + action_levels)
# ──────────────────────────────────────────────────────────────
if [ -f "${NTP_DATA}/meta.json" ] && [ "${FORCE}" != true ]; then
    echo ">>> Phase 0: NTP data found at ${NTP_DATA}, skipping preprocess"
else
    echo ">>> Phase 0: Preprocessing NTP data (with side features)"
    rm -rf "${NTP_DATA}"
    python run.py preprocess-ntp \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${NTP_DATA}" \
        --n_shards "${N_GPUS}" \
        --date_start "${DATE_START}" \
        --date_end "${DATE_END}"

    if [ ! -f "${NTP_DATA}/meta.json" ]; then
        echo "FAILED: preprocess-ntp did not produce meta.json"
        exit 1
    fi
    echo "  Preprocess complete!"
fi
echo ""

# ──────────────────────────────────────────────────────────────
# Phase 1: Smoke test (all features on)
# ──────────────────────────────────────────────────────────────
if [ "$SKIP_SMOKE" = false ]; then
    echo ">>> Phase 1: Smoke test (all features enabled)"
    SMOKE_OUTPUT="${CKPT_DIR}/exp023-smoke"
    if [ "$FORCE" = true ] || [ ! -f "${SMOKE_OUTPUT}/probe.pt" ]; then
        rm -rf "${SMOKE_OUTPUT}"
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${SMOKE_OUTPUT}" \
            --name exp023-smoke \
            --model s-tier \
            --use_time_gap \
            --use_action_level \
            --use_segment_emb \
            --dry_run
        echo "  Smoke test PASSED (all features forward/backward OK)"
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
    local SEGMENT="$4"
    local OUTPUT="${CKPT_DIR}/${NAME}"

    if [ "$FORCE" != true ] && [ -f "${OUTPUT}/train_meta.json" ]; then
        echo "  [${NAME}] Already exists, skipping."
        return 0
    fi

    local FEATURE_FLAGS=""
    [ "$TIME_GAP" = "1" ] && FEATURE_FLAGS="${FEATURE_FLAGS} --use_time_gap"
    [ "$ACTION" = "1" ] && FEATURE_FLAGS="${FEATURE_FLAGS} --use_action_level"
    [ "$SEGMENT" = "1" ] && FEATURE_FLAGS="${FEATURE_FLAGS} --use_segment_emb"

    echo ">>> Training: ${NAME} (time_gap=${TIME_GAP}, action=${ACTION}, segment=${SEGMENT})"
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
        git commit -m "EXP-023: ${NAME} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/git-lock-exp023
    echo ""
}

echo ">>> Phase 2: Feature ablation training (5 configs, serial DDP)"
echo "============================================================"
echo ""

# Config 1: Baseline (no new features) — reproduce EXP-016
if [ "$START_FROM" -le 1 ]; then
    run_config "exp023-baseline" "0" "0" "0"
fi

# Config 2: Time gap only
if [ "$START_FROM" -le 2 ]; then
    run_config "exp023-timegap" "1" "0" "0"
fi

# Config 3: Action level only
if [ "$START_FROM" -le 3 ]; then
    run_config "exp023-action" "0" "1" "0"
fi

# Config 4: Segment embedding only
if [ "$START_FROM" -le 4 ]; then
    run_config "exp023-segment" "0" "0" "1"
fi

# Config 5: All features combined
if [ "$START_FROM" -le 5 ]; then
    run_config "exp023-all" "1" "1" "1"
fi

# ──────────────────────────────────────────────────────────────
# Final commit
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-023 results: NTP Side Information (Time Gap + Action + Segment)" || echo "Nothing to commit"
./push.sh

echo ""
echo "============================================================"
echo "EXP-023 complete!"
echo "============================================================"
