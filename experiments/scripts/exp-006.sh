#!/bin/bash
set -e

# EXP-006: OPQ Parallel NTP + Graph-Constrained Decoding
# Date: 2026-04-13
# 目的: ParallelNTPModel (双向 attention + 8 独立 MLP head)
#       + code-graph constrained decoding → 与 RKMeans beam search 对比
# Environment: 8xA100

echo "=========================================="
echo "EXP-006: OPQ Parallel NTP + GCD"
echo "=========================================="

LOCK=/tmp/exp006-git.lock

python run.py hyperparam --skip_embedding --quantizer opq \
    --n_subvectors 8 --n_clusters_per_sub 256 \
    --run_ntp --name exp006-opq-parallel-gcd

echo ""
echo ">>> Committing results..."
flock "$LOCK" bash -c "
    git add experiments/ &&
    git commit -m 'EXP-006 result: OPQ parallel NTP + graph-constrained decoding' &&
    ./push.sh
" || echo "Nothing to commit"

echo ""
echo "EXP-006 complete!"
echo "Compare with EXP-001 (RKMeans baseline) recall numbers."
echo "Check: experiments/hyperparam/*exp006*/report.md"
