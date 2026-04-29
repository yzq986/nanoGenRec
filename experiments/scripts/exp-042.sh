#!/bin/bash
set -euo pipefail

# EXP-042: M-tier Full Pipeline — SFT → SP-DPO → RF-DPO → ECPO
# Date: 2026-04-29
#
# M-tier 规格: embed_dim=512, n_transformer_layers=8, n_experts=8, expert_dim=2048
#   ~222M total params / ~89M active (top-2 of 8)  vs S-tier: 45.7M / 17.4M
#
# 完整链路 (复用已有数据，不重新生成):
#   Stage 1: SFT          → exp042-m-sft
#   Stage 2: SP-DPO Easy  → exp042-m-spdpo-easy
#            SP-DPO Medium → exp042-m-spdpo-medium
#   Stage 3: RF-DPO 3ep   → exp042-m-rfdpo-3ep  (mid-ckpts ep1/ep2/ep3)
#   Stage 4: ECPO δ=0.1   → exp042-m-ecpo
#
# 早停条件 (每阶段完成后检查):
#   Stage 1 SFT:    R@500 < 0.55 → 停止 (S-tier 59.0%, 期望 M-tier >= 62%)
#   Stage 2 SP-DPO: R@500 < SFT  → 停止 (DPO 没有提升则终止)
#   Stage 3 RF-DPO: 选最优 ep (ep1/ep2/ep3)，R@500 < SP-DPO + 1pp → 停止
#   Stage 4 ECPO:   (always run if RF-DPO passes)
#
# 复用数据:
#   SID cache:    experiments/sid_cache/exp013-4096x3-12d-binary
#   NTP data:     experiments/ntp_data/exp023-14d-features
#   SP-DPO pairs: experiments/sp_dpo_data/exp037/ (M-tier 重新生成, 分布不同)
#   RF-DPO pairs: experiments/rf_dpo_data/exp018/hard (真实反馈, 与模型无关)
#   ECPO context: experiments/ntp_data/exp023-14d-features
#   Behavior:     /mnt/workspace/gr-demo-behavior-cache

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
GRPO_BATCH="${GRPO_BATCH:-${N_GPUS}}"

SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
NTP_DATA="experiments/ntp_data/exp023-14d-features"
RF_PREF_DIR="experiments/rf_dpo_data/exp018/hard"
SP_PREF_DIR="experiments/sp_dpo_data/exp042"
CKPT_DIR="experiments/ntp_checkpoints"
BEHAVIOR_CACHE="/mnt/workspace/gr-demo-behavior-cache"
DATE_END="2026-03-31"

# M-tier model config
M_EMBED=512
M_HEADS=8
M_LAYERS=8
M_EXPERTS=8
M_EXPERT_DIM=2048

FORCE=false
SKIP_SMOKE=false
START_FROM=1
for arg in "$@"; do
    case "$arg" in
        --force)         FORCE=true ;;
        --no-smoke)      SKIP_SMOKE=true ;;
        --start-from=*)  START_FROM="${arg#*=}" ;;
    esac
done

echo "============================================================"
echo "EXP-042: M-tier Full Pipeline (SFT → SP-DPO → RF-DPO → ECPO)"
echo "============================================================"
echo "  GPUs:        ${N_GPUS}"
echo "  M-tier:      ${M_EMBED}d ${M_LAYERS}L ${M_EXPERTS}E (expert_dim=${M_EXPERT_DIM})"
echo "  NTP data:    ${NTP_DATA}"
echo "  RF pairs:    ${RF_PREF_DIR}"
echo "  Start from:  stage ${START_FROM}"
echo ""

# ── Sanity checks ─────────────────────────────────────────────
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo "ERROR: NTP data not found at ${NTP_DATA}"
    exit 1
fi
if [ ! -f "${RF_PREF_DIR}/meta.json" ]; then
    echo "ERROR: RF-DPO pairs not found at ${RF_PREF_DIR}"
    exit 1
fi
if [ ! -d "${BEHAVIOR_CACHE}/${DATE_END}" ]; then
    echo "ERROR: behavior cache not found at ${BEHAVIOR_CACHE}/${DATE_END}"
    exit 1
fi

# ── Helper: check R@500 and maybe early stop ──────────────────
check_r500() {
    local CKPT=$1
    local THRESHOLD=$2
    local STAGE_NAME=$3
    local R500
    R500=$(python3 -c "
import json, sys
m = json.load(open('${CKPT}/train_meta.json'))
r = m.get('eval', {}).get('item_recall@500', 0)
print(f'{r:.4f}')
" 2>/dev/null || echo "0")
    echo "  [early-stop check] ${STAGE_NAME}: R@500=${R500} (threshold=${THRESHOLD})"
    python3 -c "
import sys
r = float('${R500}')
t = float('${THRESHOLD}')
sys.exit(0 if r >= t else 1)
"
}

# ── Smoke test ─────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" == false ] && [ "${START_FROM}" -le 1 ]; then
    echo ">>> Smoke test (M-tier SFT dry run)..."
    SMOKE_OUT="${CKPT_DIR}/exp042-smoke"
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_OUT}" \
        --name exp042-smoke \
        --model s-tier \
        --embed_dim "${M_EMBED}" \
        --n_heads "${M_HEADS}" \
        --n_transformer_layers "${M_LAYERS}" \
        --n_experts "${M_EXPERTS}" \
        --expert_dim "${M_EXPERT_DIM}" \
        --use_segment_emb \
        --use_time_gap \
        --use_action_level \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${SMOKE_OUT}"
    echo ""
fi

# ════════════════════════════════════════════════════════════════
# Stage 1: SFT
# ════════════════════════════════════════════════════════════════
SFT_NAME="exp042-m-sft"
SFT_OUT="${CKPT_DIR}/${SFT_NAME}"

if [ "${START_FROM}" -le 1 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "Stage 1: M-tier SFT"
    echo "════════════════════════════════════════════════════════════════"
    T0=$(date +%s)
    if [ -f "${SFT_OUT}/train_meta.json" ] && [ "${FORCE}" != true ]; then
        echo "  Checkpoint found, skipping."
    else
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${SFT_OUT}" \
            --name "${SFT_NAME}" \
            --model s-tier \
            --embed_dim "${M_EMBED}" \
            --n_heads "${M_HEADS}" \
            --n_transformer_layers "${M_LAYERS}" \
            --n_experts "${M_EXPERTS}" \
            --expert_dim "${M_EXPERT_DIM}" \
            --use_segment_emb \
            --use_time_gap \
            --use_action_level
    fi
    T1=$(date +%s)
    echo "  Training complete  ($(( (T1 - T0) / 60 ))min)"

    echo ">>> Stage 1 full eval..."
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${SFT_OUT}" \
        --n_recall 1000
    T2=$(date +%s)
    echo "  Eval complete  ($(( (T2 - T1) / 60 ))min)"

    git add experiments/
    git commit -m "EXP-042 Stage1 SFT complete: M-tier baseline" || echo "Nothing to commit"
    ./push.sh

    # Early stop check: SFT R@500 >= 0.55
    if ! check_r500 "${SFT_OUT}" "0.55" "SFT"; then
        echo "  !! EARLY STOP: SFT R@500 < 55%, M-tier not promising. Stopping."
        echo "EXP-042 stopped at Stage 1 (SFT early stop)."
        exit 0
    fi
    echo "  Early stop check PASSED — continuing to SP-DPO."
fi

# ════════════════════════════════════════════════════════════════
# Stage 2: SP-DPO
# ════════════════════════════════════════════════════════════════
SPDPO_NAME="exp042-m-spdpo-medium"
SPDPO_OUT="${CKPT_DIR}/${SPDPO_NAME}"

if [ "${START_FROM}" -le 2 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "Stage 2: SP-DPO (Easy → Medium, M-tier beam)"
    echo "════════════════════════════════════════════════════════════════"
    mkdir -p "${SP_PREF_DIR}"

    # Step 2a: generate SFT beam candidates
    if [ ! -f "${SP_PREF_DIR}/sft/meta.json" ] || [ "${FORCE}" == true ]; then
        echo ">>> Generating SP-DPO pairs (SFT beam, all difficulties)..."
        torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-prepare \
            --sft_checkpoint "${SFT_OUT}" \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${SP_PREF_DIR}/sft" \
            --beam_size 50 \
            --n_rejected 20 \
            --difficulty all
    else
        echo "  [pref/sft] Found, skipping."
    fi

    # Step 2b: Easy stage
    EASY_OUT="${CKPT_DIR}/exp042-m-spdpo-easy"
    if [ ! -f "${EASY_OUT}/probe.pt" ] || [ "${FORCE}" == true ]; then
        echo ">>> SP-DPO Easy..."
        T0=$(date +%s)
        torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-train \
            --sft_checkpoint "${SFT_OUT}" \
            --preference_dir "${SP_PREF_DIR}/sft" \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${EASY_OUT}" \
            --difficulty easy \
            --dpo_weight 0.1 \
            --dpo_beta 0.1 \
            --lr 1e-4 \
            --batch_size 2048 \
            --name exp042-m-spdpo-easy
        echo "  Easy complete  ($(( ($(date +%s) - T0) / 60 ))min)"
    else
        echo "  [easy] Found, skipping."
    fi

    # Step 2c: Easy-model prefix-locked beam search for Medium
    if [ ! -f "${SP_PREF_DIR}/easy-pfx/meta.json" ] || [ "${FORCE}" == true ]; then
        echo ">>> Generating prefix-locked pairs (Easy-model beam)..."
        torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-prepare \
            --sft_checkpoint "${EASY_OUT}" \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${SP_PREF_DIR}/easy-pfx" \
            --beam_size 50 \
            --n_rejected 20 \
            --difficulty all \
            --prefix_locked
    else
        echo "  [pref/easy-pfx] Found, skipping."
    fi

    # Step 2d: Medium stage
    if [ ! -f "${SPDPO_OUT}/probe.pt" ] || [ "${FORCE}" == true ]; then
        echo ">>> SP-DPO Medium..."
        T0=$(date +%s)
        torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-train \
            --sft_checkpoint "${EASY_OUT}" \
            --preference_dir "${SP_PREF_DIR}/easy-pfx" \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${SPDPO_OUT}" \
            --difficulty medium \
            --dpo_weight 0.1 \
            --dpo_beta 0.1 \
            --lr 1e-4 \
            --batch_size 2048 \
            --name "${SPDPO_NAME}"
        echo "  Medium complete  ($(( ($(date +%s) - T0) / 60 ))min)"
    else
        echo "  [medium] Found, skipping."
    fi

    echo ">>> Stage 2 full eval..."
    T0=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${SPDPO_OUT}" \
        --n_recall 1000
    echo "  Eval complete  ($(( ($(date +%s) - T0) / 60 ))min)"

    git add experiments/
    git commit -m "EXP-042 Stage2 SP-DPO complete: M-tier medium" || echo "Nothing to commit"
    ./push.sh

    # Early stop: SP-DPO must improve over SFT
    SFT_R500=$(python3 -c "
import json
m = json.load(open('${SFT_OUT}/train_meta.json'))
print(f\"{m['eval']['item_recall@500']:.4f}\")
" 2>/dev/null || echo "0")
    if ! check_r500 "${SPDPO_OUT}" "${SFT_R500}" "SP-DPO vs SFT"; then
        echo "  !! EARLY STOP: SP-DPO did not improve over SFT (R@500 < ${SFT_R500}). Stopping."
        echo "EXP-042 stopped at Stage 2 (SP-DPO regression)."
        exit 0
    fi
    echo "  Early stop check PASSED — continuing to RF-DPO."
fi

# ════════════════════════════════════════════════════════════════
# Stage 3: RF-DPO (3 epochs, pick best)
# ════════════════════════════════════════════════════════════════
RFDPO_NAME="exp042-m-rfdpo-3ep"
RFDPO_OUT="${CKPT_DIR}/${RFDPO_NAME}"

if [ "${START_FROM}" -le 3 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "Stage 3: RF-DPO λ=0.03, 3 epochs (mid-checkpoints at ep1/ep2)"
    echo "════════════════════════════════════════════════════════════════"
    T0=$(date +%s)
    if [ -f "${RFDPO_OUT}/probe.pt" ] && [ "${FORCE}" != true ]; then
        echo "  Checkpoint found, skipping."
    else
        torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-train \
            --sft_checkpoint "${SPDPO_OUT}" \
            --preference_dir "${RF_PREF_DIR}" \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${RFDPO_OUT}" \
            --dpo_weight 0.03 \
            --dpo_beta 0.1 \
            --lr 1e-4 \
            --difficulty hard \
            --ntp_epochs 3 \
            --name "${RFDPO_NAME}"
    fi
    T1=$(date +%s)
    echo "  Training complete  ($(( (T1 - T0) / 60 ))min)"

    # Eval all mid-checkpoints + final, pick best
    BEST_EP=""
    BEST_R500=0
    for EP in ep1 ep2 ep3 ""; do
        if [ -z "${EP}" ]; then
            CKPT_PATH="${RFDPO_OUT}"
            EP_LABEL="final"
        else
            CKPT_PATH="${RFDPO_OUT}-${EP}"
            EP_LABEL="${EP}"
        fi
        if [ ! -f "${CKPT_PATH}/probe.pt" ] && [ ! -f "${CKPT_PATH}/train_meta.json" ]; then
            echo "  [${EP_LABEL}] Not found, skipping."
            continue
        fi
        if [ ! -f "${CKPT_PATH}/train_meta.json" ] || \
           ! python3 -c "import json; m=json.load(open('${CKPT_PATH}/train_meta.json')); m['eval']['item_recall@500']" 2>/dev/null; then
            echo ">>> Eval ${EP_LABEL}..."
            torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
                --checkpoint "${CKPT_PATH}" \
                --n_recall 1000
        fi
        EP_R500=$(python3 -c "
import json
m = json.load(open('${CKPT_PATH}/train_meta.json'))
print(f\"{m.get('eval',{}).get('item_recall@500',0):.4f}\")
" 2>/dev/null || echo "0")
        echo "  [${EP_LABEL}] R@500=${EP_R500}"
        IS_BEST=$(python3 -c "
import sys
sys.exit(0 if float('${EP_R500}') > float('${BEST_R500}') else 1)
" && echo "true" || echo "false")
        if [ "${IS_BEST}" == "true" ]; then
            BEST_R500="${EP_R500}"
            BEST_EP="${EP_LABEL}"
            BEST_CKPT="${CKPT_PATH}"
        fi
    done
    echo "  Best checkpoint: ${BEST_EP} (R@500=${BEST_R500}) → ${BEST_CKPT}"

    git add experiments/
    git commit -m "EXP-042 Stage3 RF-DPO complete: best=${BEST_EP} R@500=${BEST_R500}" || echo "Nothing to commit"
    ./push.sh

    # Early stop: RF-DPO best must beat SP-DPO by >= 1pp
    SPDPO_R500=$(python3 -c "
import json
m = json.load(open('${SPDPO_OUT}/train_meta.json'))
print(f\"{m['eval']['item_recall@500']:.4f}\")
" 2>/dev/null || echo "0")
    THRESHOLD=$(python3 -c "print(f'{float(\"${SPDPO_R500}\") + 0.01:.4f}')")
    if ! check_r500 "${BEST_CKPT}" "${THRESHOLD}" "RF-DPO vs SP-DPO+1pp"; then
        echo "  !! EARLY STOP: RF-DPO best R@500=${BEST_R500} < SP-DPO+1pp (${THRESHOLD}). Stopping."
        echo "EXP-042 stopped at Stage 3 (RF-DPO insufficient gain)."
        exit 0
    fi
    echo "  Early stop check PASSED — continuing to ECPO."
fi

# Determine best RF-DPO checkpoint for ECPO (needed even when start_from=4)
BEST_CKPT=""
BEST_R500=0
for EP in ep1 ep2 ep3 ""; do
    if [ -z "${EP}" ]; then
        CKPT_PATH="${RFDPO_OUT}"
        EP_LABEL="final"
    else
        CKPT_PATH="${RFDPO_OUT}-${EP}"
        EP_LABEL="${EP}"
    fi
    if [ ! -f "${CKPT_PATH}/train_meta.json" ]; then
        continue
    fi
    EP_R500=$(python3 -c "
import json
m = json.load(open('${CKPT_PATH}/train_meta.json'))
print(f\"{m.get('eval',{}).get('item_recall@500',0):.4f}\")
" 2>/dev/null || echo "0")
    IS_BEST=$(python3 -c "
import sys
sys.exit(0 if float('${EP_R500}') > float('${BEST_R500}') else 1)
" && echo "true" || echo "false")
    if [ "${IS_BEST}" == "true" ]; then
        BEST_R500="${EP_R500}"
        BEST_CKPT="${CKPT_PATH}"
    fi
done

if [ -z "${BEST_CKPT}" ]; then
    echo "ERROR: No RF-DPO checkpoint found. Run stage 3 first."
    exit 1
fi
echo "  RF-DPO best for ECPO: ${BEST_CKPT} (R@500=${BEST_R500})"

# ════════════════════════════════════════════════════════════════
# Stage 4: ECPO δ=0.1
# ════════════════════════════════════════════════════════════════
ECPO_NAME="exp042-m-ecpo"
ECPO_OUT="${CKPT_DIR}/${ECPO_NAME}"

if [ "${START_FROM}" -le 4 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "Stage 4: ECPO δ=0.1 (from RF-DPO best)"
    echo "════════════════════════════════════════════════════════════════"
    T0=$(date +%s)
    if [ -f "${ECPO_OUT}/probe.pt" ] && [ "${FORCE}" != true ]; then
        echo "  Checkpoint found, skipping."
    else
        torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
            --sft_checkpoint "${BEST_CKPT}" \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${ECPO_OUT}" \
            --name "${ECPO_NAME}" \
            --eps 0.2 --delta 0.1 \
            --grpo_weight 0.03 \
            --group_size 512 \
            --grpo_batch_size "${GRPO_BATCH}" \
            --rl_data_ratio 1.0 \
            --lr 1e-4 \
            --reward_behavior --behavior_weight 1.0 \
            --behavior_cache_dir "${BEHAVIOR_CACHE}" \
            --behavior_cache_eval_date "${DATE_END}" \
            --reward_format --format_weight 0.5 \
            --on_policy_beam
    fi
    T1=$(date +%s)
    echo "  Training complete  ($(( (T1 - T0) / 60 ))min)"

    echo ">>> Stage 4 full eval..."
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${ECPO_OUT}" \
        --n_recall 1000
    T2=$(date +%s)
    echo "  Eval complete  ($(( (T2 - T1) / 60 ))min)"

    git add experiments/
    git commit -m "EXP-042 Stage4 ECPO complete: M-tier full pipeline" || echo "Nothing to commit"
    ./push.sh
fi

# ── Final summary ─────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "EXP-042 Final Results"
echo "════════════════════════════════════════════════════════════════"
python3 -c "
import json, os
checkpoints = [
    ('exp036-full-features',         'S-tier SFT (baseline)'),
    ('exp039b-ecpo-from-spdpo',      'S-tier ECPO (SOTA features)'),
    ('exp042-m-sft',                 'M-tier SFT'),
    ('exp042-m-spdpo-medium',        'M-tier SP-DPO'),
    ('exp042-m-rfdpo-3ep-ep1',       'M-tier RF-DPO ep1'),
    ('exp042-m-rfdpo-3ep-ep2',       'M-tier RF-DPO ep2'),
    ('exp042-m-rfdpo-3ep',           'M-tier RF-DPO final'),
    ('exp042-m-ecpo',                'M-tier ECPO (this)'),
]
print(f'  {\"Config\":<30}  {\"R@10\":>6}  {\"R@500\":>7}  {\"PPL\":>7}')
print(f'  {\"-\"*30}  {\"-\"*6}  {\"-\"*7}  {\"-\"*7}')
for name, desc in checkpoints:
    path = f'experiments/ntp_checkpoints/{name}/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        e = m.get('eval', {})
        r10  = e.get('item_recall@10', 0)
        r500 = e.get('item_recall@500', 0)
        ppl  = e.get('ppl', 0)
        w    = m.get('train', {}).get('wall_time_s', 0)
        print(f'  {name:<30}  {r10:>6.1%}  {r500:>7.1%}  {ppl:>7.2f}  ({int(w)//60}min)')
    else:
        print(f'  {name:<30}: not available')
" 2>/dev/null || echo "  Results not available"

echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-042 complete: M-tier full pipeline results" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-042 complete!"
