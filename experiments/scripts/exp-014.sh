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
# Pipeline per config:
#   1. preprocess-sid  — reuse EXP-013 SID cache
#   2. train-ntp       — DDP training (8 GPUs) with --entp_weight
#   3. inline eval     — PPL + beam search recall
#
# NOTE: Cannot use preprocessed shards for ENTP configs because
# neg_l0 data must be built per entp_weight. Config A (baseline)
# uses slow path with entp_weight=0 for fair comparison.
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

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
DATE_START="2026-03-01"
DATE_END="2026-03-31"

echo "============================================================"
echo "EXP-014: ENTP-Loss — Exposure-Aware Hard Negatives for L0"
echo "  SID:       4096x3 + FSQ [2]x12 binary (reuse EXP-013)"
echo "  GPUs:      ${N_GPUS}"
echo "  SID cache: ${SID_CACHE}"
echo "  Data:      ${DATE_START} ~ ${DATE_END}"
echo "============================================================"

# ── Verify SID cache exists ──
if [ ! -f "${SID_CACHE}/semantic_ids.npy" ]; then
    echo "ERROR: SID cache not found at ${SID_CACHE}"
    echo "Run exp-013.sh first or set SID_CACHE to an existing cache."
    exit 1
fi

# ── Phase 0: Smoke test (ENTP with 1 day of data) ──
if [ "${SKIP_SMOKE}" = true ]; then
    echo ""
    echo "[Phase 0] Skipping smoke test (--no-smoke)"
else
echo ""
echo "============================================================"
echo "[Phase 0] Smoke test — ENTP pipeline end-to-end"
echo "============================================================"
SMOKE_CKPT="experiments/ntp_checkpoints/exp014-smoke"
rm -rf "${SMOKE_CKPT}"

if [ "${N_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${SMOKE_CKPT}" \
        --model s-tier \
        --batch_size 64 \
        --date_start 2026-03-31 --date_end 2026-03-31 \
        --entp_weight 0.1 --entp_k 5 \
        --name exp014-smoke
else
    python run.py train-ntp \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${SMOKE_CKPT}" \
        --model s-tier \
        --batch_size 64 \
        --date_start 2026-03-31 --date_end 2026-03-31 \
        --entp_weight 0.1 --entp_k 5 \
        --name exp014-smoke
fi

if [ ! -f "${SMOKE_CKPT}/probe.pt" ]; then
    echo "SMOKE TEST FAILED: no checkpoint saved"
    exit 1
fi
echo "[Phase 0] Smoke test passed!"
rm -rf "${SMOKE_CKPT}"
fi  # end SKIP_SMOKE

# ── Helper: train + eval for a single ENTP config ──
train_entp_config() {
    local NAME=$1
    local ALPHA=$2
    local K=$3
    local DESC=$4
    local NTP_CKPT="experiments/ntp_checkpoints/${NAME}"
    local EXTRA_FLAGS=""
    local ENTP_FLAGS=""

    if [ "$(echo "${ALPHA} > 0" | bc -l)" -eq 1 ]; then
        ENTP_FLAGS="--entp_weight ${ALPHA} --entp_k ${K}"
    fi

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "  alpha=${ALPHA}, K=${K}"
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
        echo "[${NAME}] Training (${N_GPUS} GPUs, slow path)..."
    fi

    if [ "${N_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --sid_cache "${SID_CACHE}" \
            --output_dir "${NTP_CKPT}" \
            --model s-tier \
            --batch_size 128 \
            --date_start "${DATE_START}" --date_end "${DATE_END}" \
            --name "${NAME}" \
            ${ENTP_FLAGS} \
            ${EXTRA_FLAGS}
    else
        python run.py train-ntp \
            --sid_cache "${SID_CACHE}" \
            --output_dir "${NTP_CKPT}" \
            --model s-tier \
            --batch_size 128 \
            --date_start "${DATE_START}" --date_end "${DATE_END}" \
            --name "${NAME}" \
            ${ENTP_FLAGS} \
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

# ── Config A: Baseline (alpha=0, EXP-013 reproduction) ──
train_entp_config "exp014-A-baseline" 0 0 \
    "Baseline (alpha=0) — EXP-013 reproduction, no ENTP"

# ── Config B: Conservative (alpha=0.05) ──
train_entp_config "exp014-B-a005" 0.05 5 \
    "ENTP alpha=0.05, K=5 — conservative"

# ── Config C: Paper default (alpha=0.1) ──
train_entp_config "exp014-C-a010" 0.1 5 \
    "ENTP alpha=0.1, K=5 — DualGR paper default"

# ── Config D: Aggressive (alpha=0.2) ──
train_entp_config "exp014-D-a020" 0.2 5 \
    "ENTP alpha=0.2, K=5 — aggressive"

# ── Final commit ──
echo ""
echo ">>> Final commit..."
git add experiments/
git commit -m "EXP-014 results: ENTP-Loss alpha sweep (0/0.05/0.1/0.2)" || echo "Nothing to commit"
./push.sh

echo ""
echo "============================================================"
echo "EXP-014 complete!"
echo "  Results: experiments/ntp_checkpoints/exp014-*/"
echo "  Compare: L0 PPL, recall@500 across alpha values"
echo "============================================================"
