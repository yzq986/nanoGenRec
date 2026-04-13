#!/bin/bash
set -e

# EXP-002: ResKmeansFSQ — 2 layers RKMeans + 1 layer FSQ
# Date: 2026-04-13
# Variable: Layer 3 quantizer (KMeans baseline vs FSQ configs)
# Fixed: 2 KMeans layers x 1024 clusters, niter=25, nredo=3

echo "=========================================="
echo "EXP-002: ResKmeansFSQ vs RKMeans Baseline"
echo "=========================================="

echo ""
echo ">>> [1/2] Baseline: pure RKMeans 3 layers x 1024 clusters"
python run.py hyperparam --skip_embedding \
    --clusters 1024 \
    --name exp002-baseline

echo ""
echo ">>> [2/2] FSQ: 2 KMeans + 1 FSQ (4d_4096, 5d_4375, 6d_4096)"
python run.py hyperparam --skip_embedding \
    --quantizer rkmeans_fsq \
    --clusters 1024 \
    --fsq_levels 4d_4096 5d_4375 6d_4096 \
    --name exp002-fsq

echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-002 results: ResKmeansFSQ vs RKMeans baseline" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-002 complete!"
