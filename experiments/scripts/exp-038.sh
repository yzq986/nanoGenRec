#!/bin/bash
set -euo pipefail

# EXP-038: RF-DPO on exp037-medium (Features 路线第三步)
# Date: 2026-04-28
#
# RL 对齐链路:
#   exp036-full-features (SFT)
#   → EXP-037 SP-DPO → exp037-medium   (ref for this exp)
#   → [本实验: RF-DPO]  → exp038-hard-lam03
#   → EXP-039 ECPO
#
# 对标: EXP-020 (exp017-fixed-medium → exp020-hard-lam03 SOTA 66.2%)
# 本实验用 exp037-medium 作为 ref，复用 exp018 真实反馈数据 (2026-03-18~03-31)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
SFT_CKPT="experiments/ntp_checkpoints/exp037-medium"   # SP-DPO output from EXP-037
NTP_DATA="experiments/ntp_data/exp023-14d-features"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
PREF_DIR="experiments/rf_dpo_data/exp018"              # Reuse EXP-018 real feedback data
CKPT_DIR="experiments/ntp_checkpoints"
DATE_START="2026-03-18"
DATE_END="2026-03-31"

FORCE=false
SKIP_SMOKE=false
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --no-smoke) SKIP_SMOKE=true ;;
    esac
done

echo "=========================================="
echo "EXP-038: RF-DPO on exp037-medium"
echo "=========================================="
echo "  GPUs:       ${N_GPUS}"
echo "  SFT ckpt:   ${SFT_CKPT}"
echo "  Pref dir:   ${PREF_DIR}"
echo "  NTP data:   ${NTP_DATA}"
echo "  Date range: ${DATE_START} ~ ${DATE_END}"
echo ""

# Sanity checks
if [ ! -f "${SFT_CKPT}/train_meta.json" ]; then
    echo "ERROR: exp037-medium checkpoint not found at ${SFT_CKPT}"
    echo "Wait for EXP-037 Medium stage to complete before running this."
    exit 1
fi
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo "ERROR: NTP data not found at ${NTP_DATA}"
    exit 1
fi

# ── Generate RF-DPO preference pairs if not already done ──────
if [ ! -f "${PREF_DIR}/hard/meta.json" ] || [ "${FORCE}" == true ]; then
    echo ">>> Generating RF-DPO preference pairs (date: ${DATE_START}~${DATE_END})..."
    python run.py rf-dpo-prepare \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${PREF_DIR}" \
        --date_start "${DATE_START}" \
        --date_end "${DATE_END}" \
        --n_rejected 20 \
        --difficulty all
    echo "  RF-DPO pairs generated."
else
    echo "  [pref] Already exists at ${PREF_DIR}, skipping generation."
fi

mkdir -p "${CKPT_DIR}"

# ── Smoke test ─────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" == false ]; then
    SMOKE_OUT="${CKPT_DIR}/exp038-smoke"
    echo ""
    echo ">>> Smoke test (1 step)..."
    python run.py sp-dpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preference_dir "${PREF_DIR}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_OUT}" \
        --dpo_weight 0.03 \
        --dpo_beta 0.1 \
        --lr 1e-4 \
        --max_steps 1 \
        --difficulty hard \
        --name exp038-smoke \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${SMOKE_OUT}"
    echo ""
fi

# ── Main training: Joint NTP+DPO λ=0.03 ──────────────────────
NAME="exp038-hard-lam03"
OUTPUT="${CKPT_DIR}/${NAME}"

T0=$(date +%s)
if [ -f "${OUTPUT}/probe.pt" ] && [ "${FORCE}" != true ]; then
    echo "  [${NAME}] Checkpoint found, skipping training (use --force to retrain)."
else
    echo ">>> Training: ${NAME}"
    echo "    ref=exp037-medium, RF-DPO hard pairs, λ=0.03, β=0.1, Joint NTP+DPO"
    echo ""
    torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preference_dir "${PREF_DIR}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT}" \
        --dpo_weight 0.03 \
        --dpo_beta 0.1 \
        --lr 1e-4 \
        --difficulty hard \
        --name "${NAME}" \
        --wandb
fi
T1=$(date +%s)
TRAIN_MIN=$(( (T1 - T0) / 60 ))
echo "  Training complete  (${TRAIN_MIN}min)"

# ── Full eval ────────────────────────────────────────────────
echo ">>> Full eval (n_recall=1000)..."
T2=$(date +%s)
torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
    --checkpoint "${OUTPUT}" \
    --n_recall 1000
T3=$(date +%s)
EVAL_MIN=$(( (T3 - T2) / 60 ))
TOTAL_MIN=$(( (T3 - T0) / 60 ))
echo "  Eval complete  (${EVAL_MIN}min)"
echo "  Total: train=${TRAIN_MIN}min  eval=${EVAL_MIN}min  total=${TOTAL_MIN}min"

echo ""
echo ">>> Results:"
python3 -c "
import json, os
for name in ['${NAME}']:
    path = 'experiments/ntp_checkpoints/' + name + '/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        e = m.get('eval', {})
        w = m.get('train', {}).get('wall_time_s', 0)
        r10 = e.get('item_recall@10', 0)
        r500 = e.get('item_recall@500', 0)
        ppl = e.get('ppl', 0)
        print(f'  {name}: R@10={r10:.1%}  R@500={r500:.1%}  PPL={ppl:.2f}  (train={int(w)//60}min{int(w)%60}s)')
" 2>/dev/null || echo "  Eval not available"

echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-038 results: RF-DPO on exp037-medium (features RL chain)" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-038 complete!"
echo "Next: EXP-039 ECPO (ref=exp038-hard-lam03, δ=0.1)"
