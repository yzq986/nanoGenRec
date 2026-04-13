#!/bin/bash
set -e

# EXP-005: OPQ AR Baseline — 验证 beam search 在 OPQ 8-token SIDs 上失败
# Date: 2026-04-13
# 目的: 零模型改动，用现有 AutoregressiveNTPModel 跑 OPQ m=8×256
#       确认 beam search recall ≈ 0，建立对比基线
# Environment: 8xA100

echo "=========================================="
echo "EXP-005: OPQ AR Baseline (beam search failure)"
echo "=========================================="

LOCK=/tmp/exp005-git.lock

python run.py hyperparam --skip_embedding --quantizer opq \
    --n_subvectors 8 --n_clusters_per_sub 256 \
    --run_ntp --force_ar --name exp005-opq-ar-baseline

echo ""
echo ">>> Committing results..."
flock "$LOCK" bash -c "
    git add experiments/ &&
    git commit -m 'EXP-005 result: OPQ AR baseline — beam search on 8-token SIDs' &&
    ./push.sh
" || echo "Nothing to commit"

echo ""
echo "EXP-005 complete!"
echo "Expected: teacher-forcing per-digit hit OK, beam recall ≈ 0"
echo "Check: experiments/hyperparam/*exp005*/report.md"
