#!/bin/bash
set -e

# EXP-004: OPQ Parallel Semantic IDs — Intrinsic Metrics
# Date: 2026-04-13
# Reference: Meta RPG (KDD'25, arxiv 2506.05781)
# Environment: 8xA100 — 4 configs in parallel, each auto-commits on completion

echo "=========================================="
echo "EXP-004: OPQ Parallel Semantic IDs"
echo "=========================================="

LOCK=/tmp/exp004-git.lock

# Run one OPQ config, then immediately commit+push (serialized via flock)
run_and_push() {
    local gpu=$1 m=$2
    echo "[GPU $gpu] Starting OPQ m=$m ..."
    CUDA_VISIBLE_DEVICES=$gpu python run.py hyperparam --skip_embedding \
        --quantizer opq --n_subvectors $m --n_clusters_per_sub 256 \
        --skip_ntp --name exp004-opq-m${m}
    echo "[GPU $gpu] OPQ m=$m done, committing..."
    flock "$LOCK" bash -c "
        git add experiments/ &&
        git commit -m 'EXP-004 result: OPQ m=${m} x M=256' &&
        ./push.sh
    " || echo "[GPU $gpu] Nothing to commit for m=$m"
}

echo ""
echo ">>> Launching 4 configs on GPU 0-3..."
run_and_push 0 4  &
run_and_push 1 8  &
run_and_push 2 16 &
run_and_push 3 32 &

wait
echo ""
echo "EXP-004 complete! All 4 configs done."
