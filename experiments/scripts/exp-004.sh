#!/bin/bash
set -e

# EXP-004: OPQ Parallel Semantic IDs — Intrinsic Metrics
# Date: 2026-04-13
# Reference: Meta RPG (KDD'25, arxiv 2506.05781)
# Variable: n_subvectors (8, 16, 32), n_clusters_per_sub=256

echo "=========================================="
echo "EXP-004: OPQ Parallel Semantic IDs"
echo "=========================================="

echo ""
echo ">>> Running OPQ sweep: m={8,16,32}, M=256"
python run.py hyperparam --skip_embedding \
    --quantizer opq \
    --n_subvectors 8 16 32 \
    --n_clusters_per_sub 256 \
    --skip_ntp \
    --name exp004-opq

echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-004 results: OPQ parallel semantic IDs (m=8,16,32 x M=256)" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-004 complete!"
