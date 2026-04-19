#!/usr/bin/env bash
# ============================================================
# EXP-017: SP-DPO — Self-Play DPO Alignment for NTP Model
# Date: 2026-04-17
#
# Self-Play DPO (Align³GR, AAAI 2026 Oral):
#   - Beam search 生成 rejected candidates
#   - Ground truth as chosen
#   - Prefix n-gram match 定义难度 (Easy/Medium/Hard)
#   - Progressive training: Easy → Medium → Hard
#
# Baseline: EXP-016 14d-S (S-tier 17.5M, PPL=27.05, R@500=58.5%)
# Data: EXP-016 preprocessed NTP data (4096×3, 14 days, 130M tokens)
#
# Prerequisites:
#   - EXP-016 14d-S checkpoint (optimal data window per EXP-016)
#   - EXP-016 14d preprocessed NTP data
#   - rl/ module implemented (preference.py, dpo.py, trainer.py)
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
NTP_DATA="experiments/ntp_data/exp016-14d"
SFT_CKPT="experiments/ntp_checkpoints/exp016-B-14d-S"
PREF_DIR="experiments/sp_dpo_data/exp017"
CKPT_DIR="experiments/ntp_checkpoints"

echo "============================================================"
echo "EXP-017: SP-DPO — Self-Play DPO Alignment"
echo "  SFT baseline: ${SFT_CKPT}"
echo "  NTP data:     ${NTP_DATA}"
echo "  GPUs:         ${N_GPUS}"
echo "  Start from:   config #${START_FROM}"
echo "============================================================"

# ── Verify prerequisites ──
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo "ERROR: NTP data not found at ${NTP_DATA}"
    echo "Run exp-013.sh first to preprocess data."
    exit 1
fi
if [ ! -f "${SFT_CKPT}/probe.pt" ]; then
    echo "ERROR: SFT checkpoint not found at ${SFT_CKPT}"
    echo "Run exp-015.sh first (scale-04 config)."
    exit 1
fi

# ============================================================
# Phase 0: Smoke Test
# ============================================================
if [ "${SKIP_SMOKE}" != true ] && [ "${START_FROM}" -le 1 ]; then
    echo ""
    echo "============================================================"
    echo "[Smoke] SP-DPO pipeline sanity check"
    echo "============================================================"

    SMOKE_PREF="experiments/sp_dpo_data/exp017-smoke"
    SMOKE_CKPT="${CKPT_DIR}/exp017-smoke"

    # 1) Generate preference pairs (tiny subset)
    echo "[Smoke] Generating preference pairs (tiny)..."
    python run.py sp-dpo-prepare \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_PREF}" \
        --beam_size 10 \
        --n_rejected 5 \
        --max_samples 100 \
        --difficulty easy

    # 2) Train one DPO step
    echo "[Smoke] Training SP-DPO (1 step)..."
    python run.py sp-dpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preference_dir "${SMOKE_PREF}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_CKPT}" \
        --dpo_weight 0.1 \
        --dpo_beta 0.1 \
        --lr 1e-4 \
        --batch_size 64 \
        --max_steps 5 \
        --name exp017-smoke

    if [ -f "${SMOKE_CKPT}/probe.pt" ]; then
        echo "[Smoke] Passed!"
        rm -rf "${SMOKE_CKPT}" "${SMOKE_PREF}"
    else
        echo "[Smoke] FAILED"
        exit 1
    fi
fi

# ============================================================
# Phase 1: Generate Preference Pairs
# ============================================================
# Beam search all eval items once, then filter by difficulty per stage.
# This avoids redundant beam search across stages.

generate_preferences() {
    local DIFFICULTY=$1
    local OUTPUT="${PREF_DIR}/${DIFFICULTY}"

    echo ""
    echo "============================================================"
    echo "[Preference] Generating ${DIFFICULTY} pairs"
    echo "============================================================"

    if [ -f "${OUTPUT}/meta.json" ] && [ "${FORCE}" != true ]; then
        echo "[Preference] ${DIFFICULTY} pairs found, skipping (use --force to re-run)"
        return 0
    fi

    # For progressive training, reference model changes per stage.
    # Easy: use SFT checkpoint. Medium/Hard: use previous stage output.
    local REF_CKPT="${SFT_CKPT}"
    if [ "${DIFFICULTY}" = "medium" ] && [ -f "${CKPT_DIR}/exp017-spdpo-easy/probe.pt" ]; then
        REF_CKPT="${CKPT_DIR}/exp017-spdpo-easy"
    elif [ "${DIFFICULTY}" = "hard" ] && [ -f "${CKPT_DIR}/exp017-spdpo-medium/probe.pt" ]; then
        REF_CKPT="${CKPT_DIR}/exp017-spdpo-medium"
    fi

    if [ "${N_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-prepare \
            --sft_checkpoint "${REF_CKPT}" \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${OUTPUT}" \
            --beam_size 50 \
            --n_rejected 20 \
            --difficulty "${DIFFICULTY}"
    else
        python run.py sp-dpo-prepare \
            --sft_checkpoint "${REF_CKPT}" \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${OUTPUT}" \
            --beam_size 50 \
            --n_rejected 20 \
            --difficulty "${DIFFICULTY}"
    fi

    echo "[Preference] ${DIFFICULTY} done → ${OUTPUT}"
}

# ============================================================
# Phase 2: DPO Training
# ============================================================

train_dpo() {
    local NAME=$1
    local DIFFICULTY=$2      # easy, medium, hard, or "progressive"
    local DPO_WEIGHT=$3
    local DPO_BETA=$4
    local LR=$5
    local REF_CKPT=$6        # reference model checkpoint
    local PREF_PATH=$7        # preference data dir
    local DESC=$8

    local OUTPUT="${CKPT_DIR}/exp017-${NAME}"

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "  REF:  ${REF_CKPT}"
    echo "  PREF: ${PREF_PATH}"
    echo "  λ=${DPO_WEIGHT}, β=${DPO_BETA}, lr=${LR}"
    echo "============================================================"

    if [ -f "${OUTPUT}/probe.pt" ] && [ "${FORCE}" != true ]; then
        echo "[${NAME}] Checkpoint found, skipping (use --force to re-run)"
        return 0
    fi

    rm -rf "${OUTPUT}"

    local CMD_ARGS=(
        --sft_checkpoint "${REF_CKPT}"
        --preference_dir "${PREF_PATH}"
        --preprocessed_dir "${NTP_DATA}"
        --output_dir "${OUTPUT}"
        --dpo_weight "${DPO_WEIGHT}"
        --dpo_beta "${DPO_BETA}"
        --lr "${LR}"
        --batch_size 2048
        --name "exp017-${NAME}"
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
# Config 1: Single-stage Easy (sanity check)
# ============================================================
if [ "${START_FROM}" -le 1 ]; then
    generate_preferences "easy"
    train_dpo "spdpo-easy" "easy" 0.1 0.1 1e-4 \
        "${SFT_CKPT}" "${PREF_DIR}/easy" \
        "SP-DPO Easy only: λ=0.1, β=0.1"

    echo ""
    echo ">>> Committing Easy results..."
    git add experiments/
    git commit -m "EXP-017 partial: SP-DPO Easy stage" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 2: Single-stage Hard (skip Easy/Medium)
# ============================================================
if [ "${START_FROM}" -le 2 ]; then
    generate_preferences "hard"
    train_dpo "spdpo-hard" "hard" 0.1 0.1 1e-4 \
        "${SFT_CKPT}" "${PREF_DIR}/hard" \
        "SP-DPO Hard only (no progressive): λ=0.1, β=0.1"

    echo ""
    echo ">>> Committing Hard results..."
    git add experiments/
    git commit -m "EXP-017 partial: SP-DPO Hard stage (non-progressive)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 3: Progressive Easy → Medium → Hard (core experiment)
# ============================================================
if [ "${START_FROM}" -le 3 ]; then
    echo ""
    echo "============================================================"
    echo "Progressive SP-DPO: Easy → Medium → Hard"
    echo "============================================================"

    # Stage 1: Easy (reuse if already done in Config 1)
    if [ ! -f "${CKPT_DIR}/exp017-spdpo-easy/probe.pt" ]; then
        generate_preferences "easy"
        train_dpo "spdpo-easy" "easy" 0.1 0.1 1e-4 \
            "${SFT_CKPT}" "${PREF_DIR}/easy" \
            "Progressive Stage 1/3: Easy"
    else
        echo "[Progressive] Reusing spdpo-easy checkpoint"
    fi

    # Stage 2: Medium (reference = Easy output)
    # Re-generate preferences with the Easy-trained model
    generate_preferences "medium"
    train_dpo "spdpo-medium" "medium" 0.1 0.1 1e-4 \
        "${CKPT_DIR}/exp017-spdpo-easy" "${PREF_DIR}/medium" \
        "Progressive Stage 2/3: Medium (ref=Easy output)"

    # Stage 3: Hard (reference = Medium output)
    # Re-generate preferences with the Medium-trained model
    generate_preferences "hard"
    train_dpo "spdpo-prog" "hard" 0.1 0.1 1e-4 \
        "${CKPT_DIR}/exp017-spdpo-medium" "${PREF_DIR}/hard" \
        "Progressive Stage 3/3: Hard (ref=Medium output)"

    echo ""
    echo ">>> Committing Progressive results..."
    git add experiments/
    git commit -m "EXP-017 partial: SP-DPO Progressive (Easy→Medium→Hard)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 3b: Self-play ablation — Medium with SFT candidates
#   Reuses Config 1 Easy checkpoint (same starting point as Config 3).
#   Only difference: Medium beam search from SFT vs from Easy model.
#   Compare spdpo-fixed-med vs spdpo-medium to isolate self-play effect.
# ============================================================
if [ "${START_FROM}" -le 4 ]; then
    echo ""
    echo "============================================================"
    echo "Self-play ablation: Medium with fixed SFT candidates"
    echo "============================================================"

    # Generate Medium pairs from SFT model (not from Easy-trained model)
    FIXED_MED="${PREF_DIR}/fixed-sft/medium"
    if [ -f "${FIXED_MED}/meta.json" ] && [ "${FORCE}" != true ]; then
        echo "[Fixed-SFT] Medium pairs found, skipping"
    else
        if [ "${N_GPUS}" -gt 1 ]; then
            torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-prepare \
                --sft_checkpoint "${SFT_CKPT}" \
                --preprocessed_dir "${NTP_DATA}" \
                --output_dir "${FIXED_MED}" \
                --beam_size 50 \
                --n_rejected 20 \
                --difficulty medium
        else
            python run.py sp-dpo-prepare \
                --sft_checkpoint "${SFT_CKPT}" \
                --preprocessed_dir "${NTP_DATA}" \
                --output_dir "${FIXED_MED}" \
                --beam_size 50 \
                --n_rejected 20 \
                --difficulty medium
        fi
    fi

    # Train Medium from Easy checkpoint, using SFT-generated candidates
    # (Easy checkpoint is same as Config 3 — reuse spdpo-easy)
    train_dpo "spdpo-fixed-med" "medium" 0.1 0.1 1e-4 \
        "${CKPT_DIR}/exp017-spdpo-easy" "${FIXED_MED}" \
        "Ablation: Medium (ref=Easy, SFT candidates vs self-play)"

    echo ""
    echo ">>> Committing self-play ablation results..."
    git add experiments/
    git commit -m "EXP-017 partial: self-play ablation (Medium w/ SFT candidates)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 5-6: λ ablation on Progressive
# ============================================================
if [ "${START_FROM}" -le 5 ]; then
    echo ""
    echo "============================================================"
    echo "λ ablation: Progressive SP-DPO with λ=0.05"
    echo "============================================================"

    # Reuse easy→medium checkpoints, only re-run hard with different λ
    train_dpo "spdpo-prog-lam05" "hard" 0.05 0.1 1e-4 \
        "${CKPT_DIR}/exp017-spdpo-medium" "${PREF_DIR}/hard" \
        "Progressive Hard, λ=0.05 (ablation)"
fi

if [ "${START_FROM}" -le 6 ]; then
    echo ""
    echo "============================================================"
    echo "λ ablation: Progressive SP-DPO with λ=0.5"
    echo "============================================================"

    train_dpo "spdpo-prog-lam50" "hard" 0.5 0.1 1e-4 \
        "${CKPT_DIR}/exp017-spdpo-medium" "${PREF_DIR}/hard" \
        "Progressive Hard, λ=0.5 (ablation)"
fi

# ============================================================
# Final: Commit all results
# ============================================================
echo ""
echo "============================================================"
echo "EXP-017 complete!"
echo "============================================================"

echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-017 results: SP-DPO Self-Play DPO Alignment" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-017 done! Compare checkpoints:"
echo "  Baseline (SFT):           ${SFT_CKPT}"
echo "  SP-DPO Easy:              ${CKPT_DIR}/exp017-spdpo-easy"
echo "  SP-DPO Hard (direct):     ${CKPT_DIR}/exp017-spdpo-hard"
echo "  SP-DPO Progressive:       ${CKPT_DIR}/exp017-spdpo-prog"
echo "  SP-DPO Med (SFT cands):   ${CKPT_DIR}/exp017-spdpo-fixed-med"
echo "  SP-DPO λ=0.05:            ${CKPT_DIR}/exp017-spdpo-prog-lam05"
echo "  SP-DPO λ=0.5:             ${CKPT_DIR}/exp017-spdpo-prog-lam50"
echo ""
echo "Self-play ablation: compare spdpo-medium vs spdpo-fixed-med"
echo "  (same Easy init, different Medium candidates: self-play vs SFT)"
