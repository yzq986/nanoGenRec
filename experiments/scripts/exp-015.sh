#!/usr/bin/env bash
# ============================================================
# EXP-015: NTP Scaling Law — Sweep model size from ~1M to ~100M active params
# Date: 2026-04-16
#
# Fixed data: exp013 NTP data (4096×3 binary SID, 31 days behavior)
# Variable: model architecture (embed_dim, layers, MoE config)
#
# Goal: Fit L̂(N) = a + b / N^α (Chinchilla / OneRec-V2 style)
#
# Configs (7 points, log-spaced active params):
#   scale-01-1M    64d   2L  dense         ~1.7M active
#   scale-02-3M   128d   2L  dense         ~3.6M active
#   scale-03-5M   128d   4L  MoE 4E top-2  ~5.0M active
#   scale-04-11M  256d   6L  MoE 8E top-2  ~17M active (= S)
#   scale-05-25M  384d   6L  MoE 8E top-2  ~34M active
#   scale-06-55M  512d   8L  MoE 8E top-2  ~71M active
#   scale-07-100M 512d  12L  MoE 16E top-2 ~101M active
#
# Prerequisites: exp013 NTP data already preprocessed
# ============================================================
set -euo pipefail

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
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
NTP_DATA="experiments/ntp_data/exp013"

echo "============================================================"
echo "EXP-015: NTP Scaling Law"
echo "  SID:       4096×3 + FSQ [2]×12 binary"
echo "  Data:      ${NTP_DATA} (reuse exp013)"
echo "  GPUs:      ${N_GPUS}"
echo "  Start from: config #${START_FROM}"
echo "============================================================"

# ── Verify prerequisites ──
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo "ERROR: NTP data not found at ${NTP_DATA}"
    echo "Run exp-013.sh first to preprocess data."
    exit 1
fi
if [ ! -f "${SID_CACHE}/semantic_ids.npy" ]; then
    echo "ERROR: SID cache not found at ${SID_CACHE}"
    echo "Run exp-013.sh first to generate SIDs."
    exit 1
fi

# ── Helper: train + eval a single config ──
train_config() {
    local NAME=$1
    local MODEL=$2    # probe or s-tier
    local BATCH=$3
    local LR=$4
    local EMBED=$5
    local HEADS=$6
    local LAYERS=$7
    local EXPERTS=$8
    local TOPK=$9
    local EXPERT_DIM=${10}
    local DESC=${11}

    local NTP_CKPT="experiments/ntp_checkpoints/${NAME}"

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "============================================================"

    if [ -f "${NTP_CKPT}/probe.pt" ] && [ "${FORCE}" != true ]; then
        echo "[${NAME}] Checkpoint found, skipping (use --force to re-run)"
        return 0
    fi

    rm -rf "${NTP_CKPT}"

    local CMD_ARGS=(
        --preprocessed_dir "${NTP_DATA}"
        --output_dir "${NTP_CKPT}"
        --model "${MODEL}"
        --batch_size "${BATCH}"
        --lr "${LR}"
        --embed_dim "${EMBED}"
        --n_heads "${HEADS}"
        --n_transformer_layers "${LAYERS}"
        --n_experts "${EXPERTS}"
        --top_k "${TOPK}"
        --expert_dim "${EXPERT_DIM}"
        --name "${NAME}"
    )

    if [ "${N_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp "${CMD_ARGS[@]}"
    else
        python run.py train-ntp "${CMD_ARGS[@]}"
    fi

    if [ ! -f "${NTP_CKPT}/probe.pt" ]; then
        echo "[${NAME}] FAILED: no checkpoint saved"
        return 1
    fi
    echo "[${NAME}] Done!"
}

# ── Smoke test (uses preprocessed exp013 data, tiny model) ──
if [ "${SKIP_SMOKE}" != true ] && [ "${START_FROM}" -le 1 ]; then
    echo ""
    echo "[Smoke] Quick sanity check — dense 64d 2L, small batch"
    SMOKE_CKPT="experiments/ntp_checkpoints/exp015-smoke"
    rm -rf "${SMOKE_CKPT}"
    if [ "${N_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${SMOKE_CKPT}" \
            --model s-tier \
            --batch_size 64 \
            --embed_dim 64 --n_heads 2 --n_transformer_layers 2 \
            --n_experts 0 --top_k 1 --expert_dim 256 \
            --name exp015-smoke
    else
        python run.py train-ntp \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${SMOKE_CKPT}" \
            --model s-tier \
            --batch_size 64 \
            --embed_dim 64 --n_heads 2 --n_transformer_layers 2 \
            --n_experts 0 --top_k 1 --expert_dim 256 \
            --name exp015-smoke
    fi
    if [ -f "${SMOKE_CKPT}/probe.pt" ]; then
        echo "[Smoke] Passed!"
        rm -rf "${SMOKE_CKPT}"
    else
        echo "[Smoke] FAILED"
        exit 1
    fi
fi

# ============================================================
# Scaling Law Sweep — 7 configs, small → large
# ============================================================

#        name             model  batch   lr    embed heads layers experts topk exp_dim  description
[ "${START_FROM}" -le 1 ] && \
train_config "exp015-scale-01-1M"   s-tier 4096 3e-3   64    2     2      0    1    256   "Dense 64d 2L  — ~1.7M active"

[ "${START_FROM}" -le 2 ] && \
train_config "exp015-scale-02-3M"   s-tier 4096 3e-3  128    4     2      0    1    512   "Dense 128d 2L — ~3.6M active"

[ "${START_FROM}" -le 3 ] && \
train_config "exp015-scale-03-5M"   s-tier 2048 2e-3  128    4     4      4    2    512   "MoE 128d 4L 4E top-2 — ~5.0M active"

[ "${START_FROM}" -le 4 ] && \
train_config "exp015-scale-04-11M"  s-tier  128 1e-3  256    8     6      8    2   1024   "MoE 256d 6L 8E top-2 — ~17M active (=S)"

[ "${START_FROM}" -le 5 ] && \
train_config "exp015-scale-05-25M"  s-tier   64 5e-4  384    8     6      8    2   1536   "MoE 384d 6L 8E top-2 — ~34M active"

[ "${START_FROM}" -le 6 ] && \
train_config "exp015-scale-06-55M"  s-tier   32 3e-4  512    8     8      8    2   2048   "MoE 512d 8L 8E top-2 — ~71M active"

[ "${START_FROM}" -le 7 ] && \
train_config "exp015-scale-07-100M" s-tier   16 2e-4  512    8    12     16    2   2048   "MoE 512d 12L 16E top-2 — ~101M active"

# ── Commit results ──
echo ""
echo ">>> Committing results..."
git add experiments/results/
git commit -m "EXP-015 results: NTP scaling law (7 configs, 1.7M-101M active params)" || echo "Nothing to commit"
./push.sh

echo ""
echo "============================================================"
echo "EXP-015 complete! Run scaling law analysis:"
echo "  python experiments/scripts/exp015_scaling_law.py"
echo "============================================================"
