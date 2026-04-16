#!/usr/bin/env bash
# ============================================================
# EXP-013: S-tier NTP Model — 6L MoE + Loss-Free Balancing
# Date: 2026-04-15
#
# SID config: 4096×3 + FSQ [2]×12 binary (EXP-011-H / EXP-012 best)
# Compare NTPProbe (2L dense, ~5M) vs NTPModel (6L MoE, ~42M).
#
# Pipeline:
#   1. preprocess-sid  — generate 4096×3-12d SID cache (skip if found)
#   2. preprocess-ntp  — build packed sequences + save 8 shards
#   3. train-ntp       — DDP training (8 GPUs, each rank loads 1 shard)
#   4. hyperparam      — eval NTP from checkpoint
# ============================================================
set -euo pipefail

SKIP_SMOKE=false
for arg in "$@"; do
    case "$arg" in
        --no-smoke) SKIP_SMOKE=true ;;
    esac
done

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
NTP_DATA="experiments/ntp_data/exp013"

echo "============================================================"
echo "EXP-013: S-tier NTP Model — 6L MoE + Loss-Free Balancing"
echo "  SID:       4096×3 + FSQ [2]×12 binary"
echo "  GPUs:      ${N_GPUS}"
echo "  SID cache: ${SID_CACHE}"
echo "  NTP data:  ${NTP_DATA}"
echo "============================================================"

# ── Step 1: preprocess-sid (4096×3 + FSQ 12d_4096 binary) ──
if [ -f "${SID_CACHE}/semantic_ids.npy" ]; then
    echo "[Step 1] SID cache found, skipping preprocess-sid"
else
    echo "[Step 1] Running preprocess-sid (4096×3, FSQ [2]×12 binary)..."
    python run.py preprocess-sid \
        --model qwen3-0.6b \
        --behavior_path auto \
        --output_dir "${SID_CACHE}" \
        --num_clusters 4096 \
        --fsq_levels 12d_4096 \
        --fsq_projection mlp \
        --fsq_mlp_hidden 64 \
        --fsq_epochs 50
fi

# ── Step 2: preprocess-ntp (build shards, single process) ──
if [ -f "${NTP_DATA}/meta.json" ]; then
    echo "[Step 2] NTP data shards found, skipping preprocess-ntp"
else
    echo "[Step 2] Running preprocess-ntp (${N_GPUS} shards)..."
    python run.py preprocess-ntp \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${NTP_DATA}" \
        --n_shards "${N_GPUS}" \
        --date_start 2026-03-24 --date_end 2026-03-31
fi

# ── Phase 0: Smoke test (s-tier, small batch) ──
if [ "${SKIP_SMOKE}" = true ]; then
    echo ""
    echo "[Phase 0] Skipping smoke test (--no-smoke)"
else
echo ""
echo "============================================================"
echo "[Phase 0] Smoke test — s-tier model quick sanity check"
echo "============================================================"
SMOKE_CKPT="experiments/ntp_checkpoints/exp013-smoke"
rm -rf "${SMOKE_CKPT}"

if [ "${N_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${SMOKE_CKPT}" \
        --model s-tier \
        --batch_size 64 \
        --date_start 2026-03-31 --date_end 2026-03-31 \
        --name exp013-smoke
else
    python run.py train-ntp \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${SMOKE_CKPT}" \
        --model s-tier \
        --batch_size 64 \
        --date_start 2026-03-31 --date_end 2026-03-31 \
        --name exp013-smoke
fi

if [ ! -f "${SMOKE_CKPT}/probe.pt" ]; then
    echo "SMOKE TEST FAILED: no checkpoint saved"
    exit 1
fi
echo "[Phase 0] Smoke test passed!"
rm -rf "${SMOKE_CKPT}"
fi  # end SKIP_SMOKE

# ── Helper: train + eval ──
train_and_eval() {
    local NAME=$1
    local MODEL=$2
    local DESC=$3
    local BATCH=$4
    local NTP_CKPT="experiments/ntp_checkpoints/${NAME}"

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "============================================================"

    if [ -f "${NTP_CKPT}/probe.pt" ]; then
        echo "[${NAME}] Checkpoint found, skipping training"
    else
        echo "[${NAME}] Training (${N_GPUS} GPUs, pre-cached shards)..."
        if [ "${N_GPUS}" -gt 1 ]; then
            torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
                --sid_cache "${SID_CACHE}" \
                --preprocessed_dir "${NTP_DATA}" \
                --output_dir "${NTP_CKPT}" \
                --model "${MODEL}" \
                --batch_size "${BATCH}" \
                --name "${NAME}"
        else
            python run.py train-ntp \
                --sid_cache "${SID_CACHE}" \
                --preprocessed_dir "${NTP_DATA}" \
                --output_dir "${NTP_CKPT}" \
                --model "${MODEL}" \
                --batch_size "${BATCH}" \
                --name "${NAME}"
        fi
    fi

    echo "[${NAME}] Running eval..."
    python run.py hyperparam \
        --skip_embedding \
        --sid_cache "${SID_CACHE}" \
        --ntp_checkpoint "${NTP_CKPT}" \
        --run_ntp \
        --recall_beam_size 500 \
        --eval_sample_size 50000 \
        --name "${NAME}"
}

# ── Config A: Baseline (NTPProbe, 2L dense, packed training) ──
train_and_eval "exp013-probe" "probe" "NTPProbe — 2L dense, ~5M params, packed training" 4096

# ── Config B: S-tier (NTPModel, 6L MoE, packed sequences) ──
train_and_eval "exp013-s-tier" "s-tier" "NTPModel — 6L MoE (8E top-2), packed training" 128

# ── Commit results ──
echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-013 results: S-tier NTP (6L MoE) vs Probe (2L dense)" || echo "Nothing to commit"
./push.sh

echo ""
echo "============================================================"
echo "EXP-013 complete!"
echo "  Probe results:  experiments/hyperparam/*exp013-probe/"
echo "  S-tier results: experiments/hyperparam/*exp013-s-tier/"
echo "============================================================"
