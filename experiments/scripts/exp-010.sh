#!/bin/bash
set -e

# EXP-010: NTP Baseline — MLP-FSQ SID 端到端 Recall
# Date: 2026-04-15
# MLP-FSQ h=64 tokenizer + 2-layer NTP probe, single config baseline

echo "=========================================="
echo "EXP-010: NTP Baseline — MLP-FSQ + Probe"
echo "=========================================="

BEHAVIOR_PATH="auto"

# ── Phase 0: Smoke test ──
echo ""
echo ">>> Phase 0: Smoke test (eval_sample_size=100, 验证 pipeline 完整)"
CUDA_VISIBLE_DEVICES=0 python run.py hyperparam --skip_embedding \
    --quantizer rkmeans_fsq --clusters 1024 \
    --fsq_levels 6d_4096 --fsq_projection mlp --fsq_mlp_hidden 64 \
    --behavior_path "$BEHAVIOR_PATH" \
    --run_ntp --recall_beam_size 50 --eval_sample_size 100 \
    --name exp010-smoke
echo ">>> Smoke test passed!"

# ── Phase 1: Full baseline ──
echo ""
echo ">>> Phase 1: Full NTP baseline (eval_sample_size=50000)"
CUDA_VISIBLE_DEVICES=0 python run.py hyperparam --skip_embedding \
    --quantizer rkmeans_fsq --clusters 1024 \
    --fsq_levels 6d_4096 --fsq_projection mlp --fsq_mlp_hidden 64 \
    --behavior_path "$BEHAVIOR_PATH" \
    --run_ntp --recall_beam_size 50 --eval_sample_size 50000 \
    --name exp010-ntp-baseline

echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-010 result: NTP baseline Recall with MLP-FSQ" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-010 complete!"
