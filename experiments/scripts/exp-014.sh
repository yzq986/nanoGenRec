#!/usr/bin/env bash
# ============================================================
# EXP-014: ENTP-Loss — Exposure-Aware Hard Negatives for L0
# Date: 2026-04-16
#
# DualGR (WWW 2026) ENTP-Loss: penalize L0 probability of
# unclicked exposures via -alpha * log(1 - p_L0).
#
# Variable: alpha in {0, 0.05, 0.1, 0.2}, K=5 negatives/position
# Fixed: S-tier 6L MoE (EXP-013 config), 4096x3 binary SID
#
# Pipeline:
#   1. preprocess-sid  — reuse EXP-013 SID cache
#   2. preprocess-ntp  — build shards with neg_l0 data (once)
#   3. train-ntp x4    — DDP training from shards, vary --entp_weight
#
# neg_l0 data is baked into shards (independent of alpha).
# Alpha only controls loss weight at training time.
# So: preprocess once → 4 configs share the same shards.
# ============================================================
set -euo pipefail

SKIP_SMOKE=false
FORCE=false
EVAL_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --no-smoke) SKIP_SMOKE=true ;;
        --force) FORCE=true ;;
        --eval-only) EVAL_ONLY=true; SKIP_SMOKE=true ;;
    esac
done

# Ensure wandb is available
pip install -q wandb 2>/dev/null || true

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
NTP_DATA="experiments/ntp_data/exp014"
NTP_DATA_SMOKE="experiments/ntp_data/exp014-smoke"
DATE_START="2026-03-01"
DATE_END="2026-03-31"

echo "============================================================"
echo "EXP-014: ENTP-Loss — Exposure-Aware Hard Negatives for L0"
echo "  SID:       4096x3 + FSQ [2]x12 binary (reuse EXP-013)"
echo "  GPUs:      ${N_GPUS}"
echo "  SID cache: ${SID_CACHE}"
echo "  NTP data:  ${NTP_DATA}"
echo "  Data:      ${DATE_START} ~ ${DATE_END}"
echo "============================================================"

# ── Verify SID cache exists ──
if [ ! -f "${SID_CACHE}/semantic_ids.npy" ]; then
    echo "ERROR: SID cache not found at ${SID_CACHE}"
    echo "Run exp-013.sh first or set SID_CACHE to an existing cache."
    exit 1
fi

# ── Phase 0: Smoke test (preprocess 1 day + train from shard) ──
if [ "${SKIP_SMOKE}" = true ]; then
    echo ""
    echo "[Phase 0] Skipping smoke test (--no-smoke)"
else
echo ""
echo "============================================================"
echo "[Phase 0] Smoke test — preprocess + train end-to-end"
echo "============================================================"
SMOKE_CKPT="experiments/ntp_checkpoints/exp014-smoke"
rm -rf "${SMOKE_CKPT}" "${NTP_DATA_SMOKE}"

echo "[Phase 0a] Preprocess smoke data (1 day, ${N_GPUS} shards, with ENTP neg)..."
python run.py preprocess-ntp \
    --sid_cache "${SID_CACHE}" \
    --output_dir "${NTP_DATA_SMOKE}" \
    --n_shards "${N_GPUS}" \
    --date_start 2026-03-31 --date_end 2026-03-31 \
    --entp_weight 0.1 --entp_k 5

echo "[Phase 0b] Train from smoke shards..."
if [ "${N_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA_SMOKE}" \
        --output_dir "${SMOKE_CKPT}" \
        --model s-tier \
        --batch_size 64 \
        --entp_weight 0.1 \
        --name exp014-smoke
else
    python run.py train-ntp \
        --preprocessed_dir "${NTP_DATA_SMOKE}" \
        --output_dir "${SMOKE_CKPT}" \
        --model s-tier \
        --batch_size 64 \
        --entp_weight 0.1 \
        --name exp014-smoke
fi

if [ ! -f "${SMOKE_CKPT}/probe.pt" ]; then
    echo "SMOKE TEST FAILED: no checkpoint saved"
    exit 1
fi
echo "[Phase 0] Smoke test passed!"
rm -rf "${SMOKE_CKPT}" "${NTP_DATA_SMOKE}"
fi  # end SKIP_SMOKE

# ── Step 1: Preprocess full data (once, with ENTP neg) ──
if [ "${FORCE}" = true ]; then rm -rf "${NTP_DATA}"; fi
if [ -f "${NTP_DATA}/meta.json" ]; then
    echo ""
    echo "[Step 1] NTP data found at ${NTP_DATA}, skipping preprocess (use --force to rebuild)"
else
    echo ""
    echo "============================================================"
    echo "[Step 1] Preprocess NTP data (${N_GPUS} shards, with ENTP neg K=5)"
    echo "============================================================"
    python run.py preprocess-ntp \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${NTP_DATA}" \
        --n_shards "${N_GPUS}" \
        --date_start "${DATE_START}" --date_end "${DATE_END}" \
        --entp_weight 0.1 --entp_k 5
fi

# ── Helper: train + eval from preprocessed shards ──
train_entp_config() {
    local NAME=$1
    local ALPHA=$2
    local DESC=$3
    local NTP_CKPT="experiments/ntp_checkpoints/${NAME}"
    local EXTRA_FLAGS=""

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "  alpha=${ALPHA}"
    echo "============================================================"

    if [ "${EVAL_ONLY}" = true ]; then
        if [ ! -f "${NTP_CKPT}/probe.pt" ]; then
            echo "[${NAME}] ERROR: --eval-only but no checkpoint at ${NTP_CKPT}"
            return 1
        fi
        echo "[${NAME}] Eval only (loading checkpoint)..."
        EXTRA_FLAGS="--eval_only"
    elif [ -f "${NTP_CKPT}/probe.pt" ]; then
        echo "[${NAME}] Checkpoint found, skipping training (use --force to retrain)"
        return 0
    else
        echo "[${NAME}] Training (${N_GPUS} GPUs, preprocessed shards)..."
    fi

    if [ "${N_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${NTP_CKPT}" \
            --model s-tier \
            --batch_size 128 \
            --entp_weight "${ALPHA}" \
            --name "${NAME}" \
            ${EXTRA_FLAGS}
    else
        python run.py train-ntp \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${NTP_CKPT}" \
            --model s-tier \
            --batch_size 128 \
            --entp_weight "${ALPHA}" \
            --name "${NAME}" \
            ${EXTRA_FLAGS}
    fi

    # Commit after each config
    echo ""
    echo ">>> Committing ${NAME} results..."
    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-014 ${NAME}: ENTP alpha=${ALPHA}" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/exp014-git.lock
}

# ── Config A: Baseline (alpha=0) — skip, reuse EXP-013 s-tier results ──
echo ""
echo "[exp014-A-baseline] Skipped — reuse EXP-013 s-tier results as baseline"
echo "  PPL=29.6, L0=344.8, L1=13.3, L2=5.7, recall@500=59.5%"

# ── Round 2: with L0 collision filter (preprocess filters neg sharing L0 with pos) ──
# Round 1 (B/C) regressed due to gradient conflict from L0 collision.
# Round 2 uses same alpha sweep but with filtered negatives.

# ── Config E: Conservative (alpha=0.05) — with L0 filter ──
train_entp_config "exp014-E-a005" 0.05 \
    "ENTP alpha=0.05, K=5 — with L0 collision filter"

# ── Config F: Paper default (alpha=0.1) — with L0 filter ──
train_entp_config "exp014-F-a010" 0.1 \
    "ENTP alpha=0.1, K=5 — with L0 collision filter"

# ── Config G: Aggressive (alpha=0.2) — with L0 filter ──
train_entp_config "exp014-G-a020" 0.2 \
    "ENTP alpha=0.2, K=5 — with L0 collision filter"

# ── Final commit ──
echo ""
echo ">>> Final commit..."
git add experiments/
git commit -m "EXP-014 results: ENTP-Loss round 2 with L0 collision filter" || echo "Nothing to commit"
./push.sh

echo ""
echo "============================================================"
echo "EXP-014 complete!"
echo "  Results: experiments/ntp_checkpoints/exp014-*/"
echo "  Compare: L0 PPL, recall@500 across alpha values"
echo "============================================================"
