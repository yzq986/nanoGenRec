#!/usr/bin/env bash
# ============================================================
# EXP-022: NTP In-Batch Contrastive Loss (IDEA-onemall-0)
# Date: 2026-04-20
#
# Add InfoNCE contrastive auxiliary loss on s₃ hidden state,
# aligned with target item embedding (OneMall §3.2 Eq.7).
#
# Variable: α (contrastive weight), temperature, projection dim
# Baseline: EXP-016 14d-S (PPL=27.05, R@500=58.5%)
#
# Prerequisites:
#   - NTP data: experiments/ntp_data/exp016-14d (8 shards)
#   - Embedding cache: EFS shard files (qwen3-0.6b)
#   - SID cache: experiments/sid_cache/exp013-4096x3-12d-binary
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

# Fix CUDA memory fragmentation (2.78 GiB reserved-but-unallocated → OOM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SKIP_SMOKE=false
FORCE=false
START_FROM=1
for arg in "$@"; do
    case "$arg" in
        --no-smoke) SKIP_SMOKE=true ;;
        --force) FORCE=true ;;
        --start-from=*) START_FROM="${arg#*=}" ;;
    esac
done

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
NTP_DATA="experiments/ntp_data/exp016-14d"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
CKPT_DIR="experiments/ntp_checkpoints"

echo "============================================================"
echo "EXP-022: NTP In-Batch Contrastive Loss (IDEA-onemall-0)"
echo "============================================================"
echo "  GPUs: ${N_GPUS}"
echo "  NTP data: ${NTP_DATA}"
echo "  SID cache: ${SID_CACHE}"
echo ""

# ──────────────────────────────────────────────────────────────
# Phase 0: Smoke test
# ──────────────────────────────────────────────────────────────
if [ "$SKIP_SMOKE" = false ]; then
    echo ">>> Phase 0: Smoke test (contrastive loss forward/backward)"
    SMOKE_OUTPUT="${CKPT_DIR}/exp022-smoke"
    if [ "$FORCE" = true ] || [ ! -f "${SMOKE_OUTPUT}/probe.pt" ]; then
        rm -rf "${SMOKE_OUTPUT}"
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${SMOKE_OUTPUT}" \
            --name exp022-smoke \
            --model s-tier \
            --contrastive_weight 0.1 \
            --contrastive_temp 0.07 \
            --contrastive_dim 128 \
            --dry_run
        echo "  Smoke test PASSED"
        rm -rf "${SMOKE_OUTPUT}"
    else
        echo "  Smoke test skipped (already passed)"
    fi
    echo ""
fi

# ──────────────────────────────────────────────────────────────
# Phase 1: α sweep (fixed τ=0.07, dim=128)
# ──────────────────────────────────────────────────────────────
# DDP training: each config uses all 8 GPUs, run serially

run_config() {
    local NAME="$1"
    local ALPHA="$2"
    local TEMP="$3"
    local DIM="$4"
    local OUTPUT="${CKPT_DIR}/${NAME}"

    if [ "$FORCE" != true ] && [ -f "${OUTPUT}/train_meta.json" ]; then
        echo "  [${NAME}] Already exists, skipping."
        return 0
    fi

    echo ">>> Training: ${NAME} (α=${ALPHA}, τ=${TEMP}, dim=${DIM})"
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT}" \
        --name "${NAME}" \
        --model s-tier \
        --contrastive_weight "${ALPHA}" \
        --contrastive_temp "${TEMP}" \
        --contrastive_dim "${DIM}"

    echo "  [${NAME}] Training complete, committing..."
    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-022: ${NAME} (α=${ALPHA}, τ=${TEMP}, dim=${DIM})" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/git-lock-exp022
    echo ""
}

echo ""
echo ">>> Phase 1: α sweep (τ=0.07, dim=128)"
echo "============================================================"

if [ "$START_FROM" -le 1 ]; then
    run_config "exp022-alpha001" "0.01" "0.07" "128"
fi
if [ "$START_FROM" -le 2 ]; then
    run_config "exp022-alpha01" "0.1" "0.07" "128"
fi
if [ "$START_FROM" -le 3 ]; then
    run_config "exp022-alpha05" "0.5" "0.07" "128"
fi

# ──────────────────────────────────────────────────────────────
# Phase 2: temperature sweep (best α from Phase 1, dim=128)
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Phase 2: temperature sweep (τ=0.05, best α)"
echo "============================================================"

if [ "$START_FROM" -le 4 ]; then
    run_config "exp022-temp005" "0.1" "0.05" "128"
fi

# ──────────────────────────────────────────────────────────────
# Phase 3: projection dim sweep (best α + τ)
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Phase 3: projection dim sweep (dim=256)"
echo "============================================================"

if [ "$START_FROM" -le 5 ]; then
    run_config "exp022-dim256" "0.1" "0.07" "256"
fi

# ──────────────────────────────────────────────────────────────
# Final commit
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-022 results: NTP In-Batch Contrastive Loss" || echo "Nothing to commit"
./push.sh

echo ""
echo "============================================================"
echo "EXP-022 complete!"
echo "============================================================"
