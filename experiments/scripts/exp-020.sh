#!/usr/bin/env bash
# ============================================================
# EXP-020: RF-DPO Hard λ Sweep — Finding Optimal DPO Weight
# Date: 2026-04-20
#
# EXP-019 showed λ=0.01 too weak (pref_acc=50%), λ=0.1 too strong (PPL=23.6).
# This experiment sweeps λ=0.03/0.05/0.07 to find the sweet spot.
# Also tests Easy multi-epoch (20 epochs = 100 steps) to check if
# more DPO passes help when data is scarce.
#
# Prerequisites:
#   - EXP-018 preference data (experiments/rf_dpo_data/exp018/)
#   - SP-DPO fixed-medium checkpoint (EXP-017)
#   - NTP data (experiments/ntp_data/exp016-14d)
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

SKIP_SMOKE=false
FORCE=false
START_FROM=1
for arg in "$@"; do
    case "$arg" in
        --no-smoke) SKIP_SMOKE=true ;;
        --force) FORCE=true ;;
        --start-from=*) START_FROM="${arg#*=}" ;;
    esac
done

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
NTP_DATA="experiments/ntp_data/exp016-14d"
PREF_DIR="experiments/rf_dpo_data/exp018"
CKPT_DIR="experiments/ntp_checkpoints"

# Use best SP-DPO output as reference
REF_CKPT="${CKPT_DIR}/exp017-fixed-medium"
if [ ! -f "${REF_CKPT}/probe.pt" ]; then
    echo "ERROR: Reference checkpoint not found at ${REF_CKPT}"
    exit 1
fi

echo "============================================================"
echo "EXP-020: RF-DPO Hard λ Sweep"
echo "  Reference:      ${REF_CKPT}"
echo "  NTP data:       ${NTP_DATA}"
echo "  Pref data:      ${PREF_DIR}"
echo "  GPUs:           ${N_GPUS}"
echo "  Start from:     config #${START_FROM}"
echo "============================================================"

# ── Verify prerequisites ──
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo "ERROR: NTP data not found at ${NTP_DATA}"
    exit 1
fi
if [ ! -f "${PREF_DIR}/hard/meta.json" ]; then
    echo "ERROR: Hard preference data not found. Run EXP-018 first."
    exit 1
fi
if [ ! -f "${PREF_DIR}/easy/meta.json" ]; then
    echo "ERROR: Easy preference data not found. Run EXP-018 first."
    exit 1
fi

# ============================================================
# Phase 0: Smoke Test
# ============================================================
if [ "${SKIP_SMOKE}" != true ] && [ "${START_FROM}" -le 1 ]; then
    echo ""
    echo "============================================================"
    echo "[Smoke] Joint NTP+DPO pipeline sanity check"
    echo "============================================================"

    SMOKE_CKPT="${CKPT_DIR}/exp020-smoke"

    python run.py sp-dpo-train \
        --sft_checkpoint "${REF_CKPT}" \
        --preference_dir "${PREF_DIR}/hard" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_CKPT}" \
        --dpo_weight 0.05 \
        --dpo_beta 0.1 \
        --lr 1e-4 \
        --max_steps 5 \
        --difficulty hard \
        --name exp020-smoke

    if [ -f "${SMOKE_CKPT}/probe.pt" ]; then
        echo "[Smoke] Passed!"
        rm -rf "${SMOKE_CKPT}"
    else
        echo "[Smoke] FAILED"
        exit 1
    fi
fi

# ============================================================
# Training helper
# ============================================================
train_joint_rf_dpo() {
    local NAME=$1
    local DIFFICULTY=$2
    local DPO_WEIGHT=$3
    local DPO_BETA=$4
    local LR=$5
    local MAX_STEPS=$6
    local THIS_REF=$7
    local PREF_PATH=$8
    local DESC=$9

    local OUTPUT="${CKPT_DIR}/exp020-${NAME}"

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "  REF:  ${THIS_REF}"
    echo "  PREF: ${PREF_PATH}"
    echo "  λ=${DPO_WEIGHT}, β=${DPO_BETA}, lr=${LR}, max_steps=${MAX_STEPS}"
    echo "  Mode: Joint NTP+DPO"
    echo "============================================================"

    if [ -f "${OUTPUT}/probe.pt" ] && [ "${FORCE}" != true ]; then
        echo "[${NAME}] Checkpoint found, running eval-only (use --force to re-train)"
    fi

    local CMD_ARGS=(
        --sft_checkpoint "${THIS_REF}"
        --preference_dir "${PREF_PATH}"
        --preprocessed_dir "${NTP_DATA}"
        --output_dir "${OUTPUT}"
        --dpo_weight "${DPO_WEIGHT}"
        --dpo_beta "${DPO_BETA}"
        --lr "${LR}"
        --max_steps "${MAX_STEPS}"
        --difficulty "${DIFFICULTY}"
        --name "exp020-${NAME}"
        --wandb
    )

    if [ "${N_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-train "${CMD_ARGS[@]}"
    else
        python run.py sp-dpo-train "${CMD_ARGS[@]}"
    fi

    if [ ! -f "${OUTPUT}/probe.pt" ]; then
        echo "[${NAME}] FAILED: no checkpoint saved"
        return 1
    fi
    echo "[${NAME}] Done!"
}

# ============================================================
# Config 1: Hard λ=0.03
# ============================================================
if [ "${START_FROM}" -le 1 ]; then
    train_joint_rf_dpo "hard-lam03" "hard" 0.03 0.1 1e-4 807 \
        "${REF_CKPT}" "${PREF_DIR}/hard" \
        "Joint NTP+DPO Hard, λ=0.03"

    echo ""
    echo ">>> Committing Hard λ=0.03 results..."
    git add experiments/
    git commit -m "EXP-020: Joint RF-DPO Hard λ=0.03" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 2: Hard λ=0.05
# ============================================================
if [ "${START_FROM}" -le 2 ]; then
    train_joint_rf_dpo "hard-lam05" "hard" 0.05 0.1 1e-4 807 \
        "${REF_CKPT}" "${PREF_DIR}/hard" \
        "Joint NTP+DPO Hard, λ=0.05"

    echo ""
    echo ">>> Committing Hard λ=0.05 results..."
    git add experiments/
    git commit -m "EXP-020: Joint RF-DPO Hard λ=0.05" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 3: Hard λ=0.07
# ============================================================
if [ "${START_FROM}" -le 3 ]; then
    train_joint_rf_dpo "hard-lam07" "hard" 0.07 0.1 1e-4 807 \
        "${REF_CKPT}" "${PREF_DIR}/hard" \
        "Joint NTP+DPO Hard, λ=0.07"

    echo ""
    echo ">>> Committing Hard λ=0.07 results..."
    git add experiments/
    git commit -m "EXP-020: Joint RF-DPO Hard λ=0.07" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 4: Easy multi-epoch (20 epochs = 100 steps)
# ============================================================
if [ "${START_FROM}" -le 4 ]; then
    train_joint_rf_dpo "easy-multi" "easy" 0.1 0.1 1e-4 100 \
        "${REF_CKPT}" "${PREF_DIR}/easy" \
        "Joint NTP+DPO Easy, λ=0.1, 20 epochs (100 steps)"

    echo ""
    echo ">>> Committing Easy multi-epoch results..."
    git add experiments/
    git commit -m "EXP-020: Joint RF-DPO Easy multi-epoch (100 steps)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Final
# ============================================================
echo ""
echo "============================================================"
echo "EXP-020 complete!"
echo "============================================================"

echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-020 results: RF-DPO Hard λ sweep" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-020 done! Compare checkpoints:"
echo "  Reference (SP-DPO):   ${REF_CKPT} (PPL=17.5, R@10=15.4%)"
echo "  EXP-019 Hard λ=0.01:  ${CKPT_DIR}/exp019-joint-hard-lam01 (PPL=14.4, R@10=13.5%, acc=50%)"
echo "  Hard λ=0.03:           ${CKPT_DIR}/exp020-hard-lam03"
echo "  Hard λ=0.05:           ${CKPT_DIR}/exp020-hard-lam05"
echo "  Hard λ=0.07:           ${CKPT_DIR}/exp020-hard-lam07"
echo "  EXP-019 Hard λ=0.1:   ${CKPT_DIR}/exp019-joint-hard-lam10 (PPL=23.6, R@10=13.6%, acc=94%)"
echo "  Easy multi-epoch:      ${CKPT_DIR}/exp020-easy-multi"
