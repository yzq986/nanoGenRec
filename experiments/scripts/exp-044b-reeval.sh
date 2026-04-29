#!/bin/bash
set -euo pipefail

# EXP-044B re-eval: re-run eval-ntp on existing TO-RoPE checkpoints
# after fixing train-infer mismatch (timestamps now injected during beam search).
# No retraining — just eval the 3 existing checkpoints.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

GR_PYTHON="/home/dev/.conda/envs/gr/bin/python"
[ -f "${GR_PYTHON}" ] && export PATH="/home/dev/.conda/envs/gr/bin:${PATH}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"

echo "=========================================="
echo "EXP-044B re-eval (fixed beam search timestamps)"
echo "  GPUs: ${N_GPUS}"
echo "=========================================="

eval_ckpt() {
    local NAME=$1
    local CKPT="experiments/ntp_checkpoints/${NAME}"
    if [ ! -d "${CKPT}" ]; then
        echo "  [skip] ${NAME}: checkpoint not found"
        return
    fi
    echo ""
    echo ">>> Evaluating ${NAME}..."
    T0=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${CKPT}" \
        --n_recall 1000
    T1=$(date +%s)
    echo "  done ($(( (T1 - T0) / 60 ))min)"
}

eval_ckpt "exp044b-torope-ts05"
eval_ckpt "exp044b-torope-ts025"
eval_ckpt "exp044b-torope-ts05-notg"

echo ""
echo ">>> Results summary:"
for NAME in exp044b-torope-ts05 exp044b-torope-ts025 exp044b-torope-ts05-notg; do
    CKPT="experiments/ntp_checkpoints/${NAME}"
    META="${CKPT}/train_meta.json"
    if [ -f "${META}" ]; then
        python3 -c "
import json
m = json.load(open('${META}'))
r10  = m.get('item_recall@10',  m.get('item_recall_10',  '?'))
r500 = m.get('item_recall@500', m.get('item_recall_500', '?'))
print(f'  ${NAME}: R@10={r10}  R@500={r500}')
" 2>/dev/null || echo "  ${NAME}: (parse error)"
    fi
done

echo ""
echo "EXP-044B-REEVAL complete!"
