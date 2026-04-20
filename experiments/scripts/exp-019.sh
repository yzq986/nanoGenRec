#!/usr/bin/env bash
# ============================================================
# EXP-019: RF-DPO Joint NTP+DPO — Step-Matched Training
# Date: 2026-04-20
#
# Key insight: EXP-018 (pure DPO) caused catastrophic forgetting.
# Solution: Joint NTP+DPO but cap max_steps to match DPO data volume.
# NTP regularizes to prevent forgetting; DPO provides alignment signal.
#
# Reuses RF-DPO preference pairs from EXP-018.
#
# Prerequisites:
#   - EXP-018 preference data (experiments/rf_dpo_data/exp018/)
#   - SP-DPO fixed-medium checkpoint (EXP-017)
#   - NTP data (experiments/ntp_data/exp016-14d)
# ============================================================
set -euo pipefail

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
SFT_CKPT="experiments/ntp_checkpoints/exp016-B-14d-S"
PREF_DIR="experiments/rf_dpo_data/exp018"
CKPT_DIR="experiments/ntp_checkpoints"

# Use best SP-DPO output as reference
if [ -f "${CKPT_DIR}/exp017-fixed-medium/probe.pt" ]; then
    REF_CKPT="${CKPT_DIR}/exp017-fixed-medium"
    echo "  Using SP-DPO fixed-medium as reference (R@10=15.4%, R@500=68.3%)"
elif [ -f "${CKPT_DIR}/exp017-fixed-hard/probe.pt" ]; then
    REF_CKPT="${CKPT_DIR}/exp017-fixed-hard"
    echo "  Using SP-DPO fixed-hard as reference"
else
    REF_CKPT="${SFT_CKPT}"
    echo "  No SP-DPO checkpoint found, using SFT baseline as reference"
fi

echo "============================================================"
echo "EXP-019: RF-DPO Joint NTP+DPO — Step-Matched Training"
echo "  Reference:      ${REF_CKPT}"
echo "  NTP data:       ${NTP_DATA}"
echo "  Pref data:      ${PREF_DIR}"
echo "  SID cache:      ${SID_CACHE}"
echo "  GPUs:           ${N_GPUS}"
echo "  Start from:     config #${START_FROM}"
echo "============================================================"

# ── Verify prerequisites ──
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo "ERROR: NTP data not found at ${NTP_DATA}"
    exit 1
fi
if [ ! -f "${REF_CKPT}/probe.pt" ]; then
    echo "ERROR: Reference checkpoint not found at ${REF_CKPT}"
    exit 1
fi
if [ ! -f "${PREF_DIR}/easy/meta.json" ]; then
    echo "ERROR: Easy preference data not found. Run EXP-018 first."
    exit 1
fi
if [ ! -f "${PREF_DIR}/hard/meta.json" ]; then
    echo "ERROR: Hard preference data not found. Run EXP-018 first."
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

    SMOKE_CKPT="${CKPT_DIR}/exp019-smoke"

    echo "[Smoke] Training joint NTP+DPO (5 steps, step-matched)..."
    python run.py sp-dpo-train \
        --sft_checkpoint "${REF_CKPT}" \
        --preference_dir "${PREF_DIR}/easy" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_CKPT}" \
        --dpo_weight 0.1 \
        --dpo_beta 0.1 \
        --lr 1e-4 \
        --max_steps 5 \
        --name exp019-smoke

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

    local OUTPUT="${CKPT_DIR}/exp019-${NAME}"

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "  REF:  ${THIS_REF}"
    echo "  PREF: ${PREF_PATH}"
    echo "  λ=${DPO_WEIGHT}, β=${DPO_BETA}, lr=${LR}, max_steps=${MAX_STEPS}"
    echo "  Mode: Joint NTP+DPO (step-matched to DPO data)"
    echo "============================================================"

    if [ -f "${OUTPUT}/probe.pt" ] && [ "${FORCE}" != true ]; then
        echo "[${NAME}] Checkpoint found, running eval-only (use --force to re-train)"
    fi

    # archive_if_exists is handled by trainer — no rm -rf needed

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
        --name "exp019-${NAME}"
    )

    CMD_ARGS+=(--wandb)

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
# Config 1: Joint Hard λ=0.1 (807 steps) — baseline λ
# ============================================================
if [ "${START_FROM}" -le 1 ]; then
    train_joint_rf_dpo "joint-hard-lam10" "hard" 0.1 0.1 1e-4 807 \
        "${REF_CKPT}" "${PREF_DIR}/hard" \
        "Joint NTP+DPO Hard, λ=0.1 (807 steps)"

    echo ""
    echo ">>> Committing Hard λ=0.1 results..."
    git add experiments/
    git commit -m "EXP-019: Joint RF-DPO Hard λ=0.1 (fixed λ scaling)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 2: Joint Hard λ=0.5 (stronger DPO signal)
# ============================================================
if [ "${START_FROM}" -le 2 ]; then
    train_joint_rf_dpo "joint-hard-lam50" "hard" 0.5 0.1 1e-4 807 \
        "${REF_CKPT}" "${PREF_DIR}/hard" \
        "Joint NTP+DPO Hard, λ=0.5 (stronger DPO)"

    echo ""
    echo ">>> Committing Hard λ=0.5 results..."
    git add experiments/
    git commit -m "EXP-019: Joint RF-DPO Hard λ=0.5 (fixed λ scaling)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 3: Joint Hard λ=0.01 (weaker DPO, more NTP regularization)
# ============================================================
if [ "${START_FROM}" -le 3 ]; then
    train_joint_rf_dpo "joint-hard-lam01" "hard" 0.01 0.1 1e-4 807 \
        "${REF_CKPT}" "${PREF_DIR}/hard" \
        "Joint NTP+DPO Hard, λ=0.01 (more NTP regularization)"

    echo ""
    echo ">>> Committing Hard λ=0.01 results..."
    git add experiments/
    git commit -m "EXP-019: Joint RF-DPO Hard λ=0.01 (fixed λ scaling)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 4: Joint Easy λ=0.1 (15 steps)
# ============================================================
if [ "${START_FROM}" -le 4 ]; then
    train_joint_rf_dpo "joint-easy-lam10" "easy" 0.1 0.1 1e-4 15 \
        "${REF_CKPT}" "${PREF_DIR}/easy" \
        "Joint NTP+DPO Easy, λ=0.1 step-matched (15 steps)"

    echo ""
    echo ">>> Committing Easy λ=0.1 results..."
    git add experiments/
    git commit -m "EXP-019: Joint RF-DPO Easy λ=0.1 (fixed λ scaling)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 5: Progressive Easy → Hard λ=0.1 (joint mode)
# ============================================================
if [ "${START_FROM}" -le 5 ]; then
    echo ""
    echo "============================================================"
    echo "Progressive Joint RF-DPO: Easy → Hard"
    echo "============================================================"

    # Stage 1: Easy (reuse Config 4 if available)
    EASY_CKPT="${CKPT_DIR}/exp019-joint-easy-lam10"
    if [ ! -f "${EASY_CKPT}/probe.pt" ]; then
        train_joint_rf_dpo "joint-easy-lam10" "easy" 0.1 0.1 1e-4 15 \
            "${REF_CKPT}" "${PREF_DIR}/easy" \
            "Progressive Stage 1/2: Easy"
    else
        echo "[Progressive] Reusing joint-easy-lam10 checkpoint"
    fi

    # Stage 2: Hard (reference = Easy output)
    train_joint_rf_dpo "joint-prog" "hard" 0.1 0.1 1e-4 807 \
        "${EASY_CKPT}" "${PREF_DIR}/hard" \
        "Progressive Stage 2/2: Hard (ref=Easy output)"

    echo ""
    echo ">>> Committing Progressive results..."
    git add experiments/
    git commit -m "EXP-019: Joint RF-DPO Progressive Easy→Hard (fixed λ scaling)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Final
# ============================================================
echo ""
echo "============================================================"
echo "EXP-019 complete!"
echo "============================================================"

echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-019 results: RF-DPO Joint NTP+DPO (fixed λ scaling)" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-019 done! Compare checkpoints:"
echo "  Reference (SP-DPO):   ${REF_CKPT}"
echo "  Hard λ=0.1:           ${CKPT_DIR}/exp019-joint-hard-lam10"
echo "  Hard λ=0.5:           ${CKPT_DIR}/exp019-joint-hard-lam50"
echo "  Hard λ=0.01:          ${CKPT_DIR}/exp019-joint-hard-lam01"
echo "  Easy λ=0.1:           ${CKPT_DIR}/exp019-joint-easy-lam10"
echo "  Progressive:          ${CKPT_DIR}/exp019-joint-prog"
