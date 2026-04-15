#!/bin/bash
set -e

# EXP-010: NTP Baseline — MLP-FSQ SID 端到端 Recall
# Date: 2026-04-15
# Step 1: preprocess-sid (训练 tokenizer + 缓存 SID，一次性)
# Step 2: hyperparam --sid_cache (加载缓存，只跑 NTP eval)

echo "=========================================="
echo "EXP-010: NTP Baseline — MLP-FSQ + Probe"
echo "=========================================="

SID_CACHE="experiments/sid_cache/qwen3-0.6b"

# ── Step 1: Preprocess SID (skip if already cached) ──
if [ -f "$SID_CACHE/semantic_ids.npy" ]; then
    echo ""
    echo ">>> Step 1: SID cache found at $SID_CACHE, skipping tokenizer training"
    cat "$SID_CACHE/config.json"
else
    echo ""
    echo ">>> Step 1: Training tokenizer + caching SIDs"
    CUDA_VISIBLE_DEVICES=0 python run.py preprocess-sid \
        --model qwen3-0.6b --behavior_path auto
fi

# ── Step 2: NTP eval from cache ──
echo ""
echo ">>> Step 2: NTP baseline (eval_sample_size=50000, beam=50)"
CUDA_VISIBLE_DEVICES=0 python run.py hyperparam --skip_embedding \
    --sid_cache "$SID_CACHE" \
    --run_ntp --recall_beam_size 50 --eval_sample_size 50000 \
    --name exp010-ntp-baseline

echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-010 result: NTP baseline Recall with MLP-FSQ" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-010 complete!"
