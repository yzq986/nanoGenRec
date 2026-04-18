#!/usr/bin/env bash
# ============================================================
# EXP-018: RF-DPO — Real Feedback DPO Alignment
# Date: 2026-04-18
#
# RF-DPO (Align³GR Phase 2):
#   - Real user feedback as preference signal
#   - Strong positive (like/share/follow/comment/trade) as chosen
#   - Negative feedback (report/dislike) as Easy rejected
#   - Weak positive (click-only) as Hard rejected
#   - Same-user pairing only
#
# Baseline: SP-DPO output (EXP-017) or SFT (EXP-016 14d-S)
# Data: 14d behavior data (same window as EXP-016/017)
#
# Prerequisites:
#   - EXP-016 14d-S checkpoint (SFT baseline)
#   - EXP-017 SP-DPO output (optional, preferred as π_ref)
#   - SID cache
#   - rl/ module (feedback.py, dpo.py, trainer.py)
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

# Use SP-DPO output as reference if available, otherwise SFT
if [ -f "${CKPT_DIR}/exp017-spdpo-prog/probe.pt" ]; then
    REF_CKPT="${CKPT_DIR}/exp017-spdpo-prog"
    echo "  Using SP-DPO progressive output as reference model"
elif [ -f "${CKPT_DIR}/exp017-spdpo-easy/probe.pt" ]; then
    REF_CKPT="${CKPT_DIR}/exp017-spdpo-easy"
    echo "  Using SP-DPO easy output as reference model"
else
    REF_CKPT="${SFT_CKPT}"
    echo "  No SP-DPO checkpoint found, using SFT baseline as reference"
fi

echo "============================================================"
echo "EXP-018: RF-DPO — Real Feedback DPO Alignment"
echo "  SFT baseline:  ${SFT_CKPT}"
echo "  Reference:      ${REF_CKPT}"
echo "  NTP data:       ${NTP_DATA}"
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

# ============================================================
# Phase 0: Smoke Test
# ============================================================
if [ "${SKIP_SMOKE}" != true ] && [ "${START_FROM}" -le 1 ]; then
    echo ""
    echo "============================================================"
    echo "[Smoke] RF-DPO pipeline sanity check"
    echo "============================================================"

    SMOKE_PREF="experiments/rf_dpo_data/exp018-smoke"
    SMOKE_CKPT="${CKPT_DIR}/exp018-smoke"

    # 1) Generate RF-DPO preference pairs (tiny subset)
    echo "[Smoke] Generating RF-DPO preference pairs (tiny)..."
    python run.py rf-dpo-prepare \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${SMOKE_PREF}" \
        --n_rejected 5 \
        --max_samples 100 \
        --difficulty all

    # 2) Train one DPO step (reuse sp-dpo-train)
    echo "[Smoke] Training RF-DPO (1 step)..."
    python run.py sp-dpo-train \
        --sft_checkpoint "${REF_CKPT}" \
        --preference_dir "${SMOKE_PREF}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_CKPT}" \
        --dpo_weight 0.1 \
        --dpo_beta 0.1 \
        --lr 1e-4 \
        --batch_size 64 \
        --max_steps 5 \
        --name exp018-smoke

    if [ -f "${SMOKE_CKPT}/probe.pt" ]; then
        echo "[Smoke] Passed!"
        rm -rf "${SMOKE_CKPT}" "${SMOKE_PREF}"
    else
        echo "[Smoke] FAILED"
        exit 1
    fi
fi

# ============================================================
# Phase 1: Generate RF-DPO Preference Pairs
# ============================================================
generate_rf_preferences() {
    local DIFFICULTY=$1
    local OUTPUT="${PREF_DIR}/${DIFFICULTY}"

    echo ""
    echo "============================================================"
    echo "[RF-Preference] Generating ${DIFFICULTY} pairs"
    echo "============================================================"

    if [ -f "${OUTPUT}/meta.json" ] && [ "${FORCE}" != true ]; then
        echo "[RF-Preference] ${DIFFICULTY} pairs found, skipping (use --force to re-run)"
        return 0
    fi

    python run.py rf-dpo-prepare \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${OUTPUT}" \
        --n_rejected 20 \
        --difficulty "${DIFFICULTY}"

    echo "[RF-Preference] ${DIFFICULTY} done → ${OUTPUT}"
}

# ============================================================
# Phase 2: DPO Training (reuse sp-dpo-train)
# ============================================================
train_rf_dpo() {
    local NAME=$1
    local DIFFICULTY=$2
    local DPO_WEIGHT=$3
    local DPO_BETA=$4
    local LR=$5
    local THIS_REF=$6
    local PREF_PATH=$7
    local DESC=$8

    local OUTPUT="${CKPT_DIR}/exp018-${NAME}"

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "  REF:  ${THIS_REF}"
    echo "  PREF: ${PREF_PATH}"
    echo "  λ=${DPO_WEIGHT}, β=${DPO_BETA}, lr=${LR}"
    echo "============================================================"

    if [ -f "${OUTPUT}/probe.pt" ] && [ "${FORCE}" != true ]; then
        echo "[${NAME}] Checkpoint found, skipping (use --force to re-run)"
        return 0
    fi

    rm -rf "${OUTPUT}"

    local CMD_ARGS=(
        --sft_checkpoint "${THIS_REF}"
        --preference_dir "${PREF_PATH}"
        --preprocessed_dir "${NTP_DATA}"
        --output_dir "${OUTPUT}"
        --dpo_weight "${DPO_WEIGHT}"
        --dpo_beta "${DPO_BETA}"
        --lr "${LR}"
        --batch_size 2048
        --difficulty "${DIFFICULTY}"
        --name "exp018-${NAME}"
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
# Config 1: RF-DPO Easy (negative feedback as rejected)
# ============================================================
if [ "${START_FROM}" -le 1 ]; then
    generate_rf_preferences "easy"
    train_rf_dpo "rfdpo-easy" "easy" 0.1 0.1 1e-4 \
        "${REF_CKPT}" "${PREF_DIR}/easy" \
        "RF-DPO Easy only: negative feedback rejected"

    echo ""
    echo ">>> Committing Easy results..."
    git add experiments/
    git commit -m "EXP-018 partial: RF-DPO Easy stage" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 2: RF-DPO Hard (weak positive as rejected)
# ============================================================
if [ "${START_FROM}" -le 2 ]; then
    generate_rf_preferences "hard"
    train_rf_dpo "rfdpo-hard" "hard" 0.1 0.1 1e-4 \
        "${REF_CKPT}" "${PREF_DIR}/hard" \
        "RF-DPO Hard only: weak positive rejected"

    echo ""
    echo ">>> Committing Hard results..."
    git add experiments/
    git commit -m "EXP-018 partial: RF-DPO Hard stage" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 3: RF-DPO Progressive Easy → Hard
# ============================================================
if [ "${START_FROM}" -le 3 ]; then
    echo ""
    echo "============================================================"
    echo "Progressive RF-DPO: Easy → Hard"
    echo "============================================================"

    # Stage 1: Easy (reuse if done in Config 1)
    if [ ! -f "${CKPT_DIR}/exp018-rfdpo-easy/probe.pt" ]; then
        generate_rf_preferences "easy"
        train_rf_dpo "rfdpo-easy" "easy" 0.1 0.1 1e-4 \
            "${REF_CKPT}" "${PREF_DIR}/easy" \
            "Progressive Stage 1/2: Easy"
    else
        echo "[Progressive] Reusing rfdpo-easy checkpoint"
    fi

    # Stage 2: Hard (reference = Easy output)
    generate_rf_preferences "hard"
    train_rf_dpo "rfdpo-prog" "hard" 0.1 0.1 1e-4 \
        "${CKPT_DIR}/exp018-rfdpo-easy" "${PREF_DIR}/hard" \
        "Progressive Stage 2/2: Hard (ref=Easy output)"

    echo ""
    echo ">>> Committing Progressive results..."
    git add experiments/
    git commit -m "EXP-018 partial: RF-DPO Progressive (Easy→Hard)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 4-5: λ ablation on Progressive
# ============================================================
if [ "${START_FROM}" -le 4 ]; then
    train_rf_dpo "rfdpo-prog-lam05" "hard" 0.05 0.1 1e-4 \
        "${CKPT_DIR}/exp018-rfdpo-easy" "${PREF_DIR}/hard" \
        "Progressive Hard, λ=0.05 (ablation)"
fi

if [ "${START_FROM}" -le 5 ]; then
    train_rf_dpo "rfdpo-prog-lam50" "hard" 0.5 0.1 1e-4 \
        "${CKPT_DIR}/exp018-rfdpo-easy" "${PREF_DIR}/hard" \
        "Progressive Hard, λ=0.5 (ablation)"
fi

# ============================================================
# Final
# ============================================================
echo ""
echo "============================================================"
echo "EXP-018 complete!"
echo "============================================================"

echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-018 results: RF-DPO Real Feedback DPO Alignment" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-018 done! Compare checkpoints:"
echo "  SFT baseline:         ${SFT_CKPT}"
echo "  SP-DPO reference:     ${REF_CKPT}"
echo "  RF-DPO Easy:          ${CKPT_DIR}/exp018-rfdpo-easy"
echo "  RF-DPO Hard:          ${CKPT_DIR}/exp018-rfdpo-hard"
echo "  RF-DPO Progressive:   ${CKPT_DIR}/exp018-rfdpo-prog"
echo "  RF-DPO λ=0.05:        ${CKPT_DIR}/exp018-rfdpo-prog-lam05"
echo "  RF-DPO λ=0.5:         ${CKPT_DIR}/exp018-rfdpo-prog-lam50"
