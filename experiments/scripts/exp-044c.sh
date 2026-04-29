#!/bin/bash
set -euo pipefail

# EXP-044C: TO-RoPE position fix + 3-dim RoPE (order, time, layer)
# Date: 2026-04-29
#
# EXP-044B best: exp044b-torope-ts025, R@500=63.6% (+2.4pp vs baseline 61.2%)
# Two hypotheses for remaining improvement:
#   1. position-RoPE used token-level indices (0,1,2,...) but time-RoPE treats
#      all tokens within an item as simultaneous → conflicting signals.
#      Fix: use item-level positions (pos//L). Explains high PPL (467-480).
#   2. SID layer index (0/1/2) currently in segment_emb; can be a 3rd RoPE dim
#      so attention directly knows layer distance.
#
# NTP data: reuse exp044b-0.6b-14d (timestamps already correct)
#
# Configs:
#   A: 2-dim TO-RoPE ts=0.25 + item-pos fix   (baseline for this exp)
#   B: 2-dim TO-RoPE ts=0.5  + item-pos fix
#   C: 3-dim RoPE ts=0.25 layer=0.15          (new: layer as 3rd dim)
#   D: 3-dim RoPE ts=0.25 layer=0.25

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

GR_PYTHON="/home/dev/.conda/envs/gr/bin/python"
[ -f "${GR_PYTHON}" ] && export PATH="/home/dev/.conda/envs/gr/bin:${PATH}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
NTP_DATA="experiments/ntp_data/exp044b-0.6b-14d"

FORCE=false
SKIP_SMOKE=false
for arg in "$@"; do
    case "$arg" in
        --force)    FORCE=true ;;
        --no-smoke) SKIP_SMOKE=true ;;
    esac
done

echo "=========================================="
echo "EXP-044C: TO-RoPE item-pos fix + 3-dim RoPE"
echo "  GPUs:     ${N_GPUS}"
echo "  NTP data: ${NTP_DATA} (reused from 044B)"
echo "=========================================="

if [ ! -d "${NTP_DATA}" ]; then
    echo "ERROR: NTP data not found: ${NTP_DATA}"; exit 1
fi

# ── Smoke test ────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" == false ]; then
    echo ""
    echo ">>> Smoke test..."
    SMOKE_OUT="${CKPT_DIR}/exp044c-smoke"
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_OUT}" \
        --name exp044c-smoke \
        --model s-tier \
        --use_torope \
        --torope_time_split 0.25 \
        --use_segment_emb \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${SMOKE_OUT}"
fi

# ── Helper ────────────────────────────────────────────────────
train_eval() {
    local NAME=$1
    local DESC=$2
    shift 2
    local EXTRA_FLAGS="$@"
    local OUTPUT="${CKPT_DIR}/${NAME}"

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "============================================================"

    T0=$(date +%s)
    if [ -f "${OUTPUT}/train_meta.json" ] && [ "${FORCE}" != true ]; then
        echo "  Checkpoint found, skipping training."
    else
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${OUTPUT}" \
            --name "${NAME}" \
            --model s-tier \
            ${EXTRA_FLAGS}
    fi
    T1=$(date +%s)
    echo "  Training done ($(( (T1 - T0) / 60 ))min)"

    echo ">>> Full eval (n_recall=1000)..."
    T2=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT}" \
        --n_recall 1000
    T3=$(date +%s)
    echo "  Eval done ($(( (T3 - T2) / 60 ))min)  total=$(( (T3 - T0) / 60 ))min"

    git add experiments/
    git commit -m "EXP-044C ${NAME}: ${DESC}" || echo "Nothing to commit"
    ./push.sh
}

# Config A: 2-dim ts=0.25 + item-pos fix (replicate 044B best w/ fix)
train_eval \
    "exp044c-torope-ts025" \
    "2-dim TO-RoPE ts=0.25 item-pos-fix" \
    "--use_torope --torope_time_split 0.25 --use_segment_emb"

# Config B: 2-dim ts=0.5 + item-pos fix
train_eval \
    "exp044c-torope-ts05" \
    "2-dim TO-RoPE ts=0.5 item-pos-fix" \
    "--use_torope --torope_time_split 0.5 --use_segment_emb"

# Config C: 3-dim ts=0.25 layer=0.15
train_eval \
    "exp044c-torope-3d-ts025-l015" \
    "3-dim TO-RoPE ts=0.25 layer=0.15" \
    "--use_torope --torope_time_split 0.25 --torope_layer_split 0.15 --use_segment_emb"

# Config D: 3-dim ts=0.25 layer=0.25
train_eval \
    "exp044c-torope-3d-ts025-l025" \
    "3-dim TO-RoPE ts=0.25 layer=0.25" \
    "--use_torope --torope_time_split 0.25 --torope_layer_split 0.25 --use_segment_emb"

# ── Summary ──────────────────────────────────────────────────
echo ""
echo ">>> Results summary:"
python3 - <<'PYEOF'
import json, os
configs = [
    ("exp043-s-0.6b",              "Baseline (abs pos)"),
    ("exp044b-torope-ts025",       "EXP-044B: 2-dim ts=0.25 (no pos fix)"),
    ("exp044c-torope-ts025",       "EXP-044C-A: 2-dim ts=0.25 + pos fix"),
    ("exp044c-torope-ts05",        "EXP-044C-B: 2-dim ts=0.5  + pos fix"),
    ("exp044c-torope-3d-ts025-l015", "EXP-044C-C: 3-dim ts=0.25 layer=0.15"),
    ("exp044c-torope-3d-ts025-l025", "EXP-044C-D: 3-dim ts=0.25 layer=0.25"),
]
print(f"  {'Config':<42} {'R@10':>6} {'R@500':>7} {'PPL':>8}")
print(f"  {'-'*42} {'-'*6} {'-'*7} {'-'*8}")
for name, desc in configs:
    path = f"experiments/ntp_checkpoints/{name}/train_meta.json"
    if not os.path.exists(path):
        print(f"  {desc:<42}  (not found)")
        continue
    m = json.load(open(path))
    e = m.get('eval', m)
    r10  = e.get('item_recall@10',  '?')
    r500 = e.get('item_recall@500', '?')
    ppl  = e.get('ppl', '?')
    print(f"  {desc:<42} {r10 if isinstance(r10,str) else f'{r10:.3f}':>6} "
          f"{r500 if isinstance(r500,str) else f'{r500:.3f}':>7} "
          f"{ppl if isinstance(ppl,str) else f'{ppl:.1f}':>8}")
PYEOF

echo ""
echo "EXP-044C complete!"
