#!/bin/bash
set -euo pipefail

# EXP-026 re-eval: full eval for GRPO + ECPO checkpoints
# Aligns with baseline (exp020-hard-lam03) using same NTP data + full beam search
# Training is auto-skipped (probe.pt already exists)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

NTP_DATA="experiments/ntp_data/exp023-14d-features"
CKPT_DIR="experiments/ntp_checkpoints"
N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"

CONFIGS=(
    "exp026-grpo-behavior-fmt"
    "exp026-ecpo-behavior-fmt"
)

echo "=========================================="
echo "EXP-026 Re-eval (full, baseline-aligned)"
echo "=========================================="
echo "  NTP data: ${NTP_DATA}"
echo "  GPUs: ${N_GPUS}"
echo ""

# Step 1: Strip stale inline eval results from train_meta.json
echo ">>> Stripping inline eval results..."
for d in "${CONFIGS[@]}"; do
    python -c "
import json, sys
p = sys.argv[1]
try:
    m = json.load(open(p))
except Exception:
    print(f'  {p}: not found, skipping')
    sys.exit(0)
if 'eval' in m:
    m.pop('eval')
    json.dump(m, open(p, 'w'), indent=2)
    print(f'  stripped eval from {p}')
else:
    print(f'  {p}: no eval key, ok')
" "${CKPT_DIR}/${d}/train_meta.json"
done
echo ""

# Step 2: Full eval for each checkpoint
for d in "${CONFIGS[@]}"; do
    echo ">>> Full eval: ${d}"
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT_DIR}/${d}" \
        --name "${d}" \
        --model s-tier
    echo "  [${d}] done"
    echo ""
done

# Step 3: Commit + push
echo ">>> Committing results..."
git add experiments/ntp_checkpoints/exp026-*/
git commit -m "EXP-026 re-eval: full baseline-aligned eval for GRPO + ECPO" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-026 re-eval complete!"
