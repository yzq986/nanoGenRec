#!/bin/bash
set -e

# EXP-008: FORGE Proxy 对比 — MLP-FSQ vs OPQ 最优解
# Date: 2026-04-14
# 3 configs 串行跑（每个需要 behavior_data 加载），每个完成后立即 commit+push

echo "=========================================="
echo "EXP-008: FORGE Proxy — MLP-FSQ vs OPQ"
echo "=========================================="

BEHAVIOR_PATH="auto"
LOCK=/tmp/exp008-git.lock

commit_and_push() {
    local label=$1
    flock "$LOCK" bash -c "
        git add experiments/ &&
        git commit -m 'EXP-008 result: ${label}' &&
        ./push.sh
    " || echo "Nothing to commit for ${label}"
}

# Config A: MLP-FSQ h=64 (6d_4096), 2 KMeans layers + 1 FSQ layer
echo ""
echo ">>> Config A: MLP-FSQ h=64 (6d_4096)"
CUDA_VISIBLE_DEVICES=0 python run.py hyperparam --skip_embedding \
    --quantizer rkmeans_fsq --clusters 1024 \
    --fsq_levels 6d_4096 --fsq_projection mlp --fsq_mlp_hidden 64 \
    --behavior_path "$BEHAVIOR_PATH" --skip_ntp \
    --name exp008-mlpfsq-h64
commit_and_push "MLP-FSQ h=64 (6d_4096)"

# Config B: OPQ 4×256 (等 bits 对照, 32 bits)
echo ""
echo ">>> Config B: OPQ 4×256 (等 bits)"
CUDA_VISIBLE_DEVICES=0 python run.py hyperparam --skip_embedding \
    --quantizer opq --n_subvectors 4 --n_clusters_per_sub 256 \
    --behavior_path "$BEHAVIOR_PATH" --skip_ntp \
    --name exp008-opq-m4
commit_and_push "OPQ 4x256"

# Config C: OPQ 8×256 (最优, 64 bits)
echo ""
echo ">>> Config C: OPQ 8×256 (最优)"
CUDA_VISIBLE_DEVICES=0 python run.py hyperparam --skip_embedding \
    --quantizer opq --n_subvectors 8 --n_clusters_per_sub 256 \
    --behavior_path "$BEHAVIOR_PATH" --skip_ntp \
    --name exp008-opq-m8
commit_and_push "OPQ 8x256"

echo ""
echo "EXP-008 complete! All 3 configs done."
