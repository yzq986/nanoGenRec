#!/bin/bash
set -e

# EXP-003: Learned FSQ — MLP projection + straight-through training
# Date: 2026-04-13
# Variable: L3 projection (PCA vs MLP with hidden=64/128/256)
# Fixed: 2 KMeans x 1024, FSQ 6d_4096 [4,4,4,4,4,4]

echo "=========================================="
echo "EXP-003: Learned FSQ (MLP projection)"
echo "=========================================="

echo ""
echo ">>> [1/3] MLP hidden=64"
python run.py hyperparam --skip_embedding \
    --quantizer rkmeans_fsq \
    --clusters 1024 \
    --fsq_levels 6d_4096 \
    --fsq_projection mlp --fsq_mlp_hidden 64 --fsq_epochs 50 \
    --name exp003-mlp64

echo ""
echo ">>> [2/3] MLP hidden=128"
python run.py hyperparam --skip_embedding \
    --quantizer rkmeans_fsq \
    --clusters 1024 \
    --fsq_levels 6d_4096 \
    --fsq_projection mlp --fsq_mlp_hidden 128 --fsq_epochs 50 \
    --name exp003-mlp128

echo ""
echo ">>> [3/3] MLP hidden=256"
python run.py hyperparam --skip_embedding \
    --quantizer rkmeans_fsq \
    --clusters 1024 \
    --fsq_levels 6d_4096 \
    --fsq_projection mlp --fsq_mlp_hidden 256 --fsq_epochs 50 \
    --name exp003-mlp256

echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-003 results: Learned FSQ MLP projection" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-003 complete!"
