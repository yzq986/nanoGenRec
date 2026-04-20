#!/usr/bin/env bash
# ============================================================
# EXP-021: Qwen3-4B vs 0.6B Embedding Quality for SID Tokenizer
# Date: 2026-04-20
#
# Compare SID tokenizer quality and downstream NTP recall
# between Qwen3-Embedding-0.6B (dim=1024) and Qwen3-Embedding-4B (dim=2560).
#
# Prerequisites:
#   - EFS embedding cache for qwen3-4b
#   - Baseline: experiments/sid_cache/exp013-4096x3-12d-binary (0.6B, 4096×3, FSQ 12d, MLP h=64)
#
# Tokenizer config (same as baseline EXP-013):
#   4096 clusters × 2 KMeans layers, FSQ 12d_4096, MLP projection
# NTP data window: 14d (2026-03-18 ~ 2026-03-31), same as EXP-016 B-14d
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
cd "${REPO_ROOT}"

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

echo "============================================================"
echo "EXP-021: Qwen3-4B vs 0.6B Embedding Quality"
echo "  GPUs:        ${N_GPUS}"
echo "  Start from:  config #${START_FROM}"
echo "============================================================"

# ============================================================
# Phase 0: Smoke Test — verify 4B embedding cache is accessible
# ============================================================
if [ "${SKIP_SMOKE}" != true ] && [ "${START_FROM}" -le 1 ]; then
    echo ""
    echo "============================================================"
    echo "[Smoke] Verify Qwen3-4B embedding cache exists"
    echo "============================================================"

    python -c "
from gr_demo.config import EFS_EMBEDDING_CACHE, MODEL_CONFIGS
import os, glob

model_key = 'qwen3-4b'
cfg = MODEL_CONFIGS[model_key]
cache_dir = os.path.join(EFS_EMBEDDING_CACHE, model_key)
shards = glob.glob(os.path.join(cache_dir, 'shard_*.npy'))
assert len(shards) > 0, f'No embedding shards found at {cache_dir}'
print(f'[Smoke] Found {len(shards)} shards for {model_key} (dim={cfg[1]})')

import numpy as np
sample = np.load(shards[0], allow_pickle=True).item()
n_items = len(sample)
first_key = next(iter(sample))
dim = len(sample[first_key])
assert dim == cfg[1], f'Dim mismatch: got {dim}, expected {cfg[1]}'
print(f'[Smoke] Shard 0: {n_items} items, dim={dim}. OK!')
"

    echo "[Smoke] Passed!"
fi

# ============================================================
# Baseline reference (no retraining needed):
#   Qwen3-0.6B: PPL=17.5, R@10=15.4%, R@500=68.3%
#   SID cache: experiments/sid_cache/exp013-4096x3-12d-binary
#   Checkpoint: experiments/ntp_checkpoints/exp017-fixed-medium
# ============================================================

# ============================================================
# Config 1: Qwen3-4B — same FSQ hidden (64)
# ============================================================
SID_CACHE_4B="experiments/sid_cache/qwen3-4b"
NTP_DATA_4B="experiments/ntp_data/exp021-4b"
CKPT_4B="experiments/ntp_checkpoints/exp021-4b"

if [ "${START_FROM}" -le 1 ]; then
    echo ""
    echo "============================================================"
    echo "[Config 1] Qwen3-4B — train SID tokenizer + NTP (fsq_hidden=64)"
    echo "============================================================"

    # Train SID tokenizer for 4B (same config as 0.6B baseline: 4096×3, FSQ 12d_4096, MLP h=64)
    python run.py preprocess-sid \
        --model qwen3-4b \
        --output_dir "${SID_CACHE_4B}" \
        --behavior_path auto \
        --num_clusters 4096 \
        --fsq_levels 12d_4096 \
        --fsq_projection mlp \
        --fsq_mlp_hidden 64 \
        --fsq_epochs 50

    # Preprocess NTP data (14d window, same as EXP-016 B-14d)
    python run.py preprocess-ntp \
        --sid_cache "${SID_CACHE_4B}" \
        --output_dir "${NTP_DATA_4B}" \
        --n_shards "${N_GPUS}" \
        --date_start 2026-03-18 \
        --date_end 2026-03-31

    # Train NTP probe
    if [ "${N_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA_4B}" \
            --output_dir "${CKPT_4B}" \
            --model probe \
            --lr 3e-4 \
            --epochs 3 \
            --wandb \
            --name exp021-4b
    else
        python run.py train-ntp \
            --preprocessed_dir "${NTP_DATA_4B}" \
            --output_dir "${CKPT_4B}" \
            --model probe \
            --lr 3e-4 \
            --epochs 3 \
            --wandb \
            --name exp021-4b
    fi

    echo ""
    echo ">>> Committing 4B results..."
    git add experiments/
    git commit -m "EXP-021: Qwen3-4B SID + NTP (fsq_hidden=64)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 2: Qwen3-4B — larger FSQ hidden (128)
# ============================================================
SID_CACHE_4B_H128="experiments/sid_cache/qwen3-4b-h128"
NTP_DATA_4B_H128="experiments/ntp_data/exp021-4b-h128"
CKPT_4B_H128="experiments/ntp_checkpoints/exp021-4b-h128"

if [ "${START_FROM}" -le 2 ]; then
    echo ""
    echo "============================================================"
    echo "[Config 2] Qwen3-4B — larger FSQ hidden (128)"
    echo "============================================================"

    # Train SID tokenizer for 4B with larger FSQ hidden (128 vs baseline 64)
    python run.py preprocess-sid \
        --model qwen3-4b \
        --output_dir "${SID_CACHE_4B_H128}" \
        --behavior_path auto \
        --num_clusters 4096 \
        --fsq_levels 12d_4096 \
        --fsq_projection mlp \
        --fsq_mlp_hidden 128 \
        --fsq_epochs 50

    # Preprocess NTP data (14d window, same as EXP-016 B-14d)
    python run.py preprocess-ntp \
        --sid_cache "${SID_CACHE_4B_H128}" \
        --output_dir "${NTP_DATA_4B_H128}" \
        --n_shards "${N_GPUS}" \
        --date_start 2026-03-18 \
        --date_end 2026-03-31

    # Train NTP probe
    if [ "${N_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA_4B_H128}" \
            --output_dir "${CKPT_4B_H128}" \
            --model probe \
            --lr 3e-4 \
            --epochs 3 \
            --wandb \
            --name exp021-4b-h128
    else
        python run.py train-ntp \
            --preprocessed_dir "${NTP_DATA_4B_H128}" \
            --output_dir "${CKPT_4B_H128}" \
            --model probe \
            --lr 3e-4 \
            --epochs 3 \
            --wandb \
            --name exp021-4b-h128
    fi

    echo ""
    echo ">>> Committing 4B h128 results..."
    git add experiments/
    git commit -m "EXP-021: Qwen3-4B SID + NTP (fsq_hidden=128)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Final
# ============================================================
echo ""
echo "============================================================"
echo "EXP-021 complete!"
echo "============================================================"

echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-021 results: Qwen3-4B vs 0.6B embedding quality" || echo "Nothing to commit"
./push.sh

echo ""
echo "Compare checkpoints:"
echo "  0.6B baseline:     ${CKPT_06B}"
echo "  4B (h=64):         ${CKPT_4B}"
echo "  4B (h=128):        ${CKPT_4B_H128}"
echo ""
echo "Key metrics to compare: PPL, R@10, R@500, collision_rate"
