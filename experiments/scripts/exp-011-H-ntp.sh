#!/usr/bin/env bash
# ============================================================
# EXP-011-H NTP: Config H (4096x3, FSQ [2]×12 binary) full pipeline
#
# 3-stage pipeline:
#   1. preprocess-sid  — train tokenizer + cache SID assignments
#   2. train-ntp       — train NTP probe (DDP)
#   3. hyperparam      — eval NTP from checkpoint
# ============================================================
set -euo pipefail

# ── Config ──
NAME="exp011-H-4096x3-12d-binary-30d"
N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"

SID_CACHE="experiments/sid_cache/${NAME}"
NTP_CKPT="experiments/ntp_checkpoints/${NAME}"

echo "============================================================"
echo "EXP-011-H NTP Pipeline"
echo "  Config:    4096x3 + FSQ 12d_4096 [2]×12 (binary)"
echo "  GPUs:      ${N_GPUS}"
echo "  SID cache: ${SID_CACHE}"
echo "  NTP ckpt:  ${NTP_CKPT}"
echo "============================================================"

# ── Step 1: preprocess-sid ──
if [ -f "${SID_CACHE}/semantic_ids.npy" ]; then
    echo "[Step 1] SID cache found, skipping preprocess-sid"
else
    echo "[Step 1] Running preprocess-sid..."
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

# ── Step 2: train-ntp ──
if [ -f "${NTP_CKPT}/probe.pt" ]; then
    echo "[Step 2] NTP checkpoint found, skipping train-ntp"
else
    echo "[Step 2] Training NTP probe (${N_GPUS} GPUs)..."
    if [ "${N_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --sid_cache "${SID_CACHE}" \
            --output_dir "${NTP_CKPT}" \
            --batch_size 4096
    else
        python run.py train-ntp \
            --sid_cache "${SID_CACHE}" \
            --output_dir "${NTP_CKPT}" \
            --batch_size 4096
    fi
fi

# ── Step 3: NTP eval ──
echo "[Step 3] Running NTP eval..."
python run.py hyperparam \
    --skip_embedding \
    --sid_cache "${SID_CACHE}" \
    --ntp_checkpoint "${NTP_CKPT}" \
    --run_ntp \
    --recall_beam_size 500 \
    --eval_sample_size 50000 \
    --name "${NAME}"

echo ""
echo "============================================================"
echo "Done! Results: experiments/hyperparam/*${NAME}/"
echo "============================================================"
