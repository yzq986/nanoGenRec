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
# Two-config ablation design:
#   Config 1 (Fixed): SFT beam search → Easy→eval→Medium→eval→Hard→eval
#   Config 2 (Self-play): SFT → Easy → Easy beam search → Medium→eval→Hard→eval
#   Shared Easy stage. Difference: candidates for Medium/Hard from SFT vs Easy model.
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
# Helper: Generate Preference Pairs
# ============================================================
# beam_search_model: which model to run beam search with
# output_dir: where to save preference pairs
# difficulty: easy, medium, hard

generate_preferences() {
    local BEAM_MODEL=$1
    local OUTPUT=$2
    local DIFFICULTY=$3

    echo ""
    echo "============================================================"
    echo "[Preference] Generating ${DIFFICULTY} pairs"
    echo "  Model: ${BEAM_MODEL}"
    echo "  Output: ${OUTPUT}"
    echo "============================================================"

    if [ -f "${OUTPUT}/meta.json" ] && [ "${FORCE}" != true ]; then
        echo "[Preference] ${DIFFICULTY} pairs found at ${OUTPUT}, skipping (use --force)"
        return 0
    fi

    if [ "${N_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-prepare \
            --sft_checkpoint "${BEAM_MODEL}" \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${OUTPUT}" \
            --beam_size 50 \
            --n_rejected 20 \
            --difficulty "${DIFFICULTY}"
    else
        python run.py sp-dpo-prepare \
            --sft_checkpoint "${BEAM_MODEL}" \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${OUTPUT}" \
            --beam_size 50 \
            --n_rejected 20 \
            --difficulty "${DIFFICULTY}"
    fi

    echo "[Preference] ${DIFFICULTY} done → ${OUTPUT}"
}

# ============================================================
# Helper: DPO Training
# ============================================================

train_dpo() {
    local NAME=$1
    local DIFFICULTY=$2      # easy, medium, hard
    local DPO_WEIGHT=$3
    local DPO_BETA=$4
    local LR=$5
    local REF_CKPT=$6        # reference model checkpoint (π_ref for training)
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
        --difficulty "${DIFFICULTY}"
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
# Shared: SFT beam search (once) + Easy stage
#   One beam search with --difficulty all produces easy/medium/hard
#   pairs in the same npz shards. No redundant beam search.
# ============================================================
if [ "${START_FROM}" -le 1 ]; then
    # SFT beam search — all difficulties in one pass
    generate_preferences "${SFT_CKPT}" "${PREF_DIR}/sft" "all"

    # Easy training (shared by both configs)
    train_dpo "spdpo-easy" "easy" 0.1 0.1 1e-4 \
        "${SFT_CKPT}" "${PREF_DIR}/sft" \
        "Shared Easy: SFT beam search, λ=0.1, β=0.1"

    echo ""
    echo ">>> Committing Easy results..."
    git add experiments/
    git commit -m "EXP-017 partial: SP-DPO Easy stage (shared)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 1: Fixed SFT candidates → Easy → Medium → Hard
#   SFT beam search already done above (--difficulty all).
#   Progressive training using SFT candidates for all stages.
#   Isolates curriculum effect (no self-play re-generation).
# ============================================================
if [ "${START_FROM}" -le 2 ]; then
    echo ""
    echo "============================================================"
    echo "Config 1: Fixed SFT candidates (Easy → Medium → Hard)"
    echo "============================================================"

    # Reuse Easy from shared stage
    if [ ! -f "${CKPT_DIR}/exp017-spdpo-easy/probe.pt" ]; then
        echo "ERROR: Easy checkpoint missing. Run from --start-from=1"
        exit 1
    fi

    # SFT candidates already in ${PREF_DIR}/sft (from shared stage)

    # Medium (ref = Easy output, candidates from SFT)
    train_dpo "fixed-medium" "medium" 0.1 0.1 1e-4 \
        "${CKPT_DIR}/exp017-spdpo-easy" "${PREF_DIR}/sft" \
        "Config1 Medium: ref=Easy, SFT candidates"

    # Hard (ref = fixed-Medium output, candidates from SFT)
    train_dpo "fixed-hard" "hard" 0.1 0.1 1e-4 \
        "${CKPT_DIR}/exp017-fixed-medium" "${PREF_DIR}/sft" \
        "Config1 Hard: ref=fixed-Medium, SFT candidates"

    echo ""
    echo ">>> Committing Config 1 results..."
    git add experiments/
    git commit -m "EXP-017 partial: Config 1 Fixed SFT candidates (E→M→H)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 2: Self-play from Easy → Medium → Hard
#   Easy model beam search once (--difficulty all) → all difficulties.
#   Same npz, trainer picks the right difficulty column.
# ============================================================
if [ "${START_FROM}" -le 3 ]; then
    echo ""
    echo "============================================================"
    echo "Config 2: Self-play from Easy (Easy → Medium → Hard)"
    echo "============================================================"

    # Reuse Easy from shared stage
    if [ ! -f "${CKPT_DIR}/exp017-spdpo-easy/probe.pt" ]; then
        echo "ERROR: Easy checkpoint missing. Run from --start-from=1"
        exit 1
    fi

    # Easy model beam search — one pass, all difficulties
    generate_preferences "${CKPT_DIR}/exp017-spdpo-easy" "${PREF_DIR}/sp-easy" "all"

    # Medium (ref = Easy output, candidates from Easy model)
    train_dpo "sp-medium" "medium" 0.1 0.1 1e-4 \
        "${CKPT_DIR}/exp017-spdpo-easy" "${PREF_DIR}/sp-easy" \
        "Config2 Medium: ref=Easy, Easy-model candidates"

    # Hard (ref = sp-Medium output, candidates from Easy model)
    train_dpo "sp-hard" "hard" 0.1 0.1 1e-4 \
        "${CKPT_DIR}/exp017-sp-medium" "${PREF_DIR}/sp-easy" \
        "Config2 Hard: ref=sp-Medium, Easy-model candidates"

    echo ""
    echo ">>> Committing Config 2 results..."
    git add experiments/
    git commit -m "EXP-017 partial: Config 2 Self-play from Easy (E→M→H)" || echo "Nothing to commit"
    ./push.sh
fi

# ============================================================
# Config 3-4: λ ablation on the better config
#   (Run after comparing Config 1 vs Config 2)
# ============================================================
if [ "${START_FROM}" -le 4 ]; then
    echo ""
    echo "============================================================"
    echo "λ ablation: λ=0.05 on self-play Hard"
    echo "============================================================"

    # Uses self-play (Config 2) Medium checkpoint + Easy-model candidates
    train_dpo "sp-hard-lam05" "hard" 0.05 0.1 1e-4 \
        "${CKPT_DIR}/exp017-sp-medium" "${PREF_DIR}/sp-easy" \
        "λ ablation: Hard, λ=0.05 (self-play)"
fi

if [ "${START_FROM}" -le 5 ]; then
    echo ""
    echo "============================================================"
    echo "λ ablation: λ=0.5 on self-play Hard"
    echo "============================================================"

    train_dpo "sp-hard-lam50" "hard" 0.5 0.1 1e-4 \
        "${CKPT_DIR}/exp017-sp-medium" "${PREF_DIR}/sp-easy" \
        "λ ablation: Hard, λ=0.5 (self-play)"
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
echo "  SP-DPO Easy (shared):     ${CKPT_DIR}/exp017-spdpo-easy"
echo ""
echo "  Config 1 (Fixed SFT candidates):"
echo "    Medium:                  ${CKPT_DIR}/exp017-fixed-medium"
echo "    Hard:                    ${CKPT_DIR}/exp017-fixed-hard"
echo ""
echo "  Config 2 (Self-play from Easy):"
echo "    Medium:                  ${CKPT_DIR}/exp017-sp-medium"
echo "    Hard:                    ${CKPT_DIR}/exp017-sp-hard"
echo ""
echo "  λ ablation (self-play):"
echo "    λ=0.05:                  ${CKPT_DIR}/exp017-sp-hard-lam05"
echo "    λ=0.5:                   ${CKPT_DIR}/exp017-sp-hard-lam50"
echo ""
echo "Key comparison:"
echo "  Config 1 vs Config 2 → isolates self-play re-generation effect"
echo "  fixed-medium vs sp-medium → Medium stage comparison"
echo "  fixed-hard vs sp-hard → Hard stage comparison"
