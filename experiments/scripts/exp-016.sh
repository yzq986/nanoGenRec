#!/usr/bin/env bash
# ============================================================
# EXP-016: Data Scaling Law — Fix model, sweep data size
# Date: 2026-04-17
#
# Goal: Fit Chinchilla dual-variable law L(N,D) = E + A/N^α + B/D^β
#       by sweeping data D ∈ {7d, 14d, 31d, 62d, 66d} for two models:
#       - S  (17.5M active): 256d 6L 8E top-2
#       - M+ (101M active):  512d 12L 16E top-2
#
# C-31d reuses EXP-015 results (scale-04 for S, scale-07 for M+).
# New training runs: 4×2 = 8 (minus C-31d reuse).
#
# Prerequisites: SID cache from exp013
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
NTP_DATA_BASE="experiments/ntp_data"
CKPT_BASE="experiments/ntp_checkpoints"
RESULTS_DIR="experiments/results/ntp"

echo "============================================================"
echo "EXP-016: Data Scaling Law"
echo "  SID:       4096×3 + FSQ [2]×12 binary"
echo "  Models:    S (17.5M active), M+ (101M active)"
echo "  Data:      7d / 14d / 31d / 62d / 66d"
echo "  GPUs:      ${N_GPUS}"
echo "  Start from: config #${START_FROM}"
echo "============================================================"

# ── Verify SID cache ──
if [ ! -f "${SID_CACHE}/semantic_ids.npy" ]; then
    echo "ERROR: SID cache not found at ${SID_CACHE}"
    echo "Run exp-013.sh first to generate SIDs."
    exit 1
fi

# ── Date ranges ──
# Each config has a name, date_start, date_end
# C-31d reuses exp013 data, so no new preprocessing needed
declare -A DATE_STARTS DATE_ENDS NTP_DIRS TOKENS_APPROX
DATE_STARTS=( [A-7d]="2026-03-25" [B-14d]="2026-03-18" [C-31d]="2026-03-01" [D-62d]="2026-02-01" [E-66d]="2026-01-25" )
DATE_ENDS=(   [A-7d]="2026-03-31" [B-14d]="2026-03-31" [C-31d]="2026-03-31" [D-62d]="2026-03-31" [E-66d]="2026-03-31" )
NTP_DIRS=(    [A-7d]="${NTP_DATA_BASE}/exp016-7d" [B-14d]="${NTP_DATA_BASE}/exp016-14d" [C-31d]="${NTP_DATA_BASE}/exp013" [D-62d]="${NTP_DATA_BASE}/exp016-62d" [E-66d]="${NTP_DATA_BASE}/exp016-66d" )
TOKENS_APPROX=( [A-7d]="~61M" [B-14d]="~119M" [C-31d]="~238M" [D-62d]="~404M" [E-66d]="~445M" )

DATA_KEYS=("A-7d" "B-14d" "C-31d" "D-62d" "E-66d")

# ── Helper: preprocess NTP data for a date range ──
preprocess_data() {
    local KEY=$1
    local DATA_DIR="${NTP_DIRS[$KEY]}"

    if [ -f "${DATA_DIR}/meta.json" ] && [ "${FORCE}" != true ]; then
        echo "[Preprocess ${KEY}] Data found at ${DATA_DIR}, skipping"
        return 0
    fi

    echo ""
    echo "============================================================"
    echo "[Preprocess ${KEY}] ${DATE_STARTS[$KEY]} ~ ${DATE_ENDS[$KEY]} (${TOKENS_APPROX[$KEY]} tokens)"
    echo "============================================================"

    rm -rf "${DATA_DIR}"
    python run.py preprocess-ntp \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${DATA_DIR}" \
        --n_shards "${N_GPUS}" \
        --date_start "${DATE_STARTS[$KEY]}" \
        --date_end "${DATE_ENDS[$KEY]}"

    if [ ! -f "${DATA_DIR}/meta.json" ]; then
        echo "[Preprocess ${KEY}] FAILED"
        exit 1
    fi
    echo "[Preprocess ${KEY}] Done!"
}

# ── Helper: train a single config ──
train_config() {
    local NAME=$1
    local DATA_DIR=$2
    local BATCH=$3
    local LR=$4
    local EMBED=$5
    local HEADS=$6
    local LAYERS=$7
    local EXPERTS=$8
    local TOPK=$9
    local EXPERT_DIM=${10}
    local DESC=${11}

    local NTP_CKPT="${CKPT_BASE}/${NAME}"

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "============================================================"

    if [ -f "${NTP_CKPT}/probe.pt" ] && [ "${FORCE}" != true ]; then
        echo "[${NAME}] Checkpoint found, skipping (use --force to re-run)"
        return 0
    fi

    rm -rf "${NTP_CKPT}"

    local CMD_ARGS=(
        --preprocessed_dir "${DATA_DIR}"
        --output_dir "${NTP_CKPT}"
        --model s-tier
        --batch_size "${BATCH}"
        --lr "${LR}"
        --embed_dim "${EMBED}"
        --n_heads "${HEADS}"
        --n_transformer_layers "${LAYERS}"
        --n_experts "${EXPERTS}"
        --top_k "${TOPK}"
        --expert_dim "${EXPERT_DIM}"
        --name "${NAME}"
    )

    if [ "${N_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp "${CMD_ARGS[@]}"
    else
        python run.py train-ntp "${CMD_ARGS[@]}"
    fi

    if [ ! -f "${NTP_CKPT}/probe.pt" ]; then
        echo "[${NAME}] FAILED: no checkpoint saved"
        return 1
    fi
    echo "[${NAME}] Done!"
}

# ── Helper: commit + push results after each config ──
commit_results() {
    local MSG=$1
    (
        flock -n 9 || { echo "  Waiting for git lock..."; flock 9; }
        git add experiments/results/ experiments/ntp_checkpoints/*/train_meta.json 2>/dev/null || true
        git commit -m "${MSG}" 2>/dev/null || echo "  Nothing to commit"
        ./push.sh --main-only 2>/dev/null || echo "  Push warning (non-fatal)"
    ) 9>/tmp/exp016-git.lock
}

# ============================================================
# Phase 0: Smoke test
# ============================================================
if [ "${SKIP_SMOKE}" != true ] && [ "${START_FROM}" -le 1 ]; then
    echo ""
    echo "[Smoke] Quick sanity check — dense 64d 2L, 7d data"

    # Preprocess 7d data first (needed for smoke test)
    preprocess_data "A-7d"

    SMOKE_CKPT="${CKPT_BASE}/exp016-smoke"
    rm -rf "${SMOKE_CKPT}"
    if [ "${N_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DIRS[A-7d]}" \
            --output_dir "${SMOKE_CKPT}" \
            --model s-tier \
            --batch_size 64 \
            --embed_dim 64 --n_heads 2 --n_transformer_layers 2 \
            --n_experts 0 --top_k 1 --expert_dim 256 \
            --name exp016-smoke
    else
        python run.py train-ntp \
            --preprocessed_dir "${NTP_DIRS[A-7d]}" \
            --output_dir "${SMOKE_CKPT}" \
            --model s-tier \
            --batch_size 64 \
            --embed_dim 64 --n_heads 2 --n_transformer_layers 2 \
            --n_experts 0 --top_k 1 --expert_dim 256 \
            --name exp016-smoke
    fi
    if [ -f "${SMOKE_CKPT}/probe.pt" ]; then
        echo "[Smoke] Passed!"
        rm -rf "${SMOKE_CKPT}"
    else
        echo "[Smoke] FAILED"
        exit 1
    fi
fi

# ============================================================
# Phase 1: Preprocess all date ranges
# ============================================================
echo ""
echo "============================================================"
echo "Phase 1: Preprocess NTP data for all date ranges"
echo "============================================================"

for KEY in "${DATA_KEYS[@]}"; do
    preprocess_data "${KEY}"
done

# ============================================================
# Phase 2: Training — 5 data sizes × 2 models
# ============================================================
# Config map (step → data_key, model_tier)
# Steps 1-10, S models first at each data size, then M+
# C-31d (steps 5,6) reuse EXP-015 results

echo ""
echo "============================================================"
echo "Phase 2: Training sweep (5 data sizes × 2 models)"
echo "============================================================"

# ── S-tier configs (17.5M active): embed=256, 6L, 8E top-2, expert_dim=1024 ──
S_BATCH=128
S_LR=1e-3
S_EMBED=256
S_HEADS=8
S_LAYERS=6
S_EXPERTS=8
S_TOPK=2
S_EXPERT_DIM=1024

# ── M+-tier configs (101M active): embed=512, 12L, 16E top-2, expert_dim=2048 ──
M_BATCH=16
M_LR=2e-4
M_EMBED=512
M_HEADS=8
M_LAYERS=12
M_EXPERTS=16
M_TOPK=2
M_EXPERT_DIM=2048

# Step 1: A-7d S
if [ "${START_FROM}" -le 1 ]; then
    train_config "exp016-A-7d-S" "${NTP_DIRS[A-7d]}" \
        $S_BATCH $S_LR $S_EMBED $S_HEADS $S_LAYERS $S_EXPERTS $S_TOPK $S_EXPERT_DIM \
        "S (17.5M) on 7d data (~61M tokens, 3.5 tok/param)"
    commit_results "EXP-016: A-7d S-tier (17.5M) result"
fi

# Step 2: A-7d M+
if [ "${START_FROM}" -le 2 ]; then
    train_config "exp016-A-7d-M" "${NTP_DIRS[A-7d]}" \
        $M_BATCH $M_LR $M_EMBED $M_HEADS $M_LAYERS $M_EXPERTS $M_TOPK $M_EXPERT_DIM \
        "M+ (101M) on 7d data (~61M tokens, 0.6 tok/param)"
    commit_results "EXP-016: A-7d M+-tier (101M) result"
fi

# Step 3: B-14d S
if [ "${START_FROM}" -le 3 ]; then
    train_config "exp016-B-14d-S" "${NTP_DIRS[B-14d]}" \
        $S_BATCH $S_LR $S_EMBED $S_HEADS $S_LAYERS $S_EXPERTS $S_TOPK $S_EXPERT_DIM \
        "S (17.5M) on 14d data (~119M tokens, 6.8 tok/param)"
    commit_results "EXP-016: B-14d S-tier (17.5M) result"
fi

# Step 4: B-14d M+
if [ "${START_FROM}" -le 4 ]; then
    train_config "exp016-B-14d-M" "${NTP_DIRS[B-14d]}" \
        $M_BATCH $M_LR $M_EMBED $M_HEADS $M_LAYERS $M_EXPERTS $M_TOPK $M_EXPERT_DIM \
        "M+ (101M) on 14d data (~119M tokens, 1.2 tok/param)"
    commit_results "EXP-016: B-14d M+-tier (101M) result"
fi

# Step 5: C-31d S — REUSE EXP-015 scale-04
if [ "${START_FROM}" -le 5 ]; then
    echo ""
    echo "============================================================"
    echo "[C-31d S] Reusing EXP-015 scale-04 (17.5M, 31d, ~262M tokens)"
    echo "============================================================"
    SRC_015="${RESULTS_DIR}/exp015-scale-04-11M.json"
    DST_016="${RESULTS_DIR}/exp016-C-31d-S.json"
    if [ -f "${SRC_015}" ]; then
        cp "${SRC_015}" "${DST_016}"
        # Update name in the copied file
        python -c "
import json
with open('${DST_016}') as f: d = json.load(f)
d['name'] = 'exp016-C-31d-S'
d['reused_from'] = 'exp015-scale-04-11M'
with open('${DST_016}', 'w') as f: json.dump(d, f, indent=2)
"
        echo "[C-31d S] Copied from ${SRC_015}"
    else
        echo "[C-31d S] WARNING: ${SRC_015} not found, training from scratch..."
        train_config "exp016-C-31d-S" "${NTP_DIRS[C-31d]}" \
            $S_BATCH $S_LR $S_EMBED $S_HEADS $S_LAYERS $S_EXPERTS $S_TOPK $S_EXPERT_DIM \
            "S (17.5M) on 31d data (~238M tokens, 13.6 tok/param) [fallback]"
    fi
    commit_results "EXP-016: C-31d S-tier (reuse EXP-015 scale-04)"
fi

# Step 6: C-31d M+ — REUSE EXP-015 scale-07
if [ "${START_FROM}" -le 6 ]; then
    echo ""
    echo "============================================================"
    echo "[C-31d M+] Reusing EXP-015 scale-07 (101M, 31d, ~262M tokens)"
    echo "============================================================"
    SRC_015="${RESULTS_DIR}/exp015-scale-07-100M.json"
    DST_016="${RESULTS_DIR}/exp016-C-31d-M.json"
    if [ -f "${SRC_015}" ]; then
        cp "${SRC_015}" "${DST_016}"
        python -c "
import json
with open('${DST_016}') as f: d = json.load(f)
d['name'] = 'exp016-C-31d-M'
d['reused_from'] = 'exp015-scale-07-100M'
with open('${DST_016}', 'w') as f: json.dump(d, f, indent=2)
"
        echo "[C-31d M+] Copied from ${SRC_015}"
    else
        echo "[C-31d M+] WARNING: ${SRC_015} not found, training from scratch..."
        train_config "exp016-C-31d-M" "${NTP_DIRS[C-31d]}" \
            $M_BATCH $M_LR $M_EMBED $M_HEADS $M_LAYERS $M_EXPERTS $M_TOPK $M_EXPERT_DIM \
            "M+ (101M) on 31d data (~238M tokens, 2.4 tok/param) [fallback]"
    fi
    commit_results "EXP-016: C-31d M+-tier (reuse EXP-015 scale-07)"
fi

# Step 7: D-62d S
if [ "${START_FROM}" -le 7 ]; then
    train_config "exp016-D-62d-S" "${NTP_DIRS[D-62d]}" \
        $S_BATCH $S_LR $S_EMBED $S_HEADS $S_LAYERS $S_EXPERTS $S_TOPK $S_EXPERT_DIM \
        "S (17.5M) on 62d data (~404M tokens, 23.1 tok/param)"
    commit_results "EXP-016: D-62d S-tier (17.5M) result"
fi

# Step 8: D-62d M+
if [ "${START_FROM}" -le 8 ]; then
    train_config "exp016-D-62d-M" "${NTP_DIRS[D-62d]}" \
        $M_BATCH $M_LR $M_EMBED $M_HEADS $M_LAYERS $M_EXPERTS $M_TOPK $M_EXPERT_DIM \
        "M+ (101M) on 62d data (~404M tokens, 4.0 tok/param)"
    commit_results "EXP-016: D-62d M+-tier (101M) result"
fi

# Step 9: E-66d S
if [ "${START_FROM}" -le 9 ]; then
    train_config "exp016-E-66d-S" "${NTP_DIRS[E-66d]}" \
        $S_BATCH $S_LR $S_EMBED $S_HEADS $S_LAYERS $S_EXPERTS $S_TOPK $S_EXPERT_DIM \
        "S (17.5M) on 66d data (~445M tokens, 25.4 tok/param)"
    commit_results "EXP-016: E-66d S-tier (17.5M) result"
fi

# Step 10: E-66d M+
if [ "${START_FROM}" -le 10 ]; then
    train_config "exp016-E-66d-M" "${NTP_DIRS[E-66d]}" \
        $M_BATCH $M_LR $M_EMBED $M_HEADS $M_LAYERS $M_EXPERTS $M_TOPK $M_EXPERT_DIM \
        "M+ (101M) on 66d data (~445M tokens, 4.4 tok/param)"
    commit_results "EXP-016: E-66d M+-tier (101M) result"
fi

# ── Final commit ──
echo ""
echo ">>> Final commit..."
git add experiments/
git commit -m "EXP-016 results: Data scaling law (5 data sizes × 2 models)" || echo "Nothing to commit"
./push.sh

echo ""
echo "============================================================"
echo "EXP-016 complete! Run analysis:"
echo "  python experiments/scripts/exp016_data_scaling.py"
echo "============================================================"
