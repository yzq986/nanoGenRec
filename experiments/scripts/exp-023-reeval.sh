#!/bin/bash
set -euo pipefail

# EXP-023 re-eval: fix train-eval mismatch for side features
# Only re-runs eval (training is skipped automatically)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
cd "${REPO_ROOT}"

NTP_DATA="experiments/ntp_data/exp023-14d-features"
CKPT_DIR="experiments/ntp_checkpoints"
N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"

CONFIGS="exp023-timegap exp023-action exp023-all"

echo "=========================================="
echo "EXP-023 Re-eval (side feature fix)"
echo "=========================================="

# Step 1: Strip eval from train_meta.json
echo ""
echo ">>> Stripping eval results from train_meta.json..."
for d in $CONFIGS; do
    python -c "
import json, sys
p = sys.argv[1]
m = json.load(open(p))
if 'eval' in m:
    m.pop('eval')
    json.dump(m, open(p, 'w'), indent=2)
    print(f'  {p}: eval removed')
else:
    print(f'  {p}: no eval found')
" "${CKPT_DIR}/${d}/train_meta.json"
done

# Step 2: Re-run eval (training auto-skipped via probe.pt check)
echo ""
for d in $CONFIGS; do
    echo ">>> Re-eval: ${d}"
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT_DIR}/${d}" \
        --name "${d}" \
        --model s-tier
    echo "  [${d}] eval complete"
    echo ""
done

# Step 3: Commit + push
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-023 re-eval: fix side feature train-eval mismatch" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-023 re-eval complete!"
