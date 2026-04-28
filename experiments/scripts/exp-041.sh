#!/bin/bash
set -euo pipefail

# EXP-041: ENTP-Loss — Exposure-Aware Hard Negatives for L0 (with Features)
# Date: 2026-04-28
#
# IDEA-dualgr-0: DualGR (Kuaishou WWW 2026)
# 在 NTP loss 中对 L0 层的未点击曝光 item 加 -α*log(1-p_L0) 惩罚
#
# 相比 EXP-014 (基于 exp013 无特征模型), 本实验:
#   1. 使用 exp023-14d-features 的 SID + features 管线
#   2. 加入 time_gap + action_level + segment_emb
#   3. 更新日期窗口 2026-03-18~03-31
#
# 实验设计:
#   Config A (baseline): entp_weight=0 (等价于 exp036-full-features, 直接引用)
#   Config B: entp_weight=0.05
#   Config C: entp_weight=0.1  (DualGR 推荐值)
#   Config D: entp_weight=0.2
#
# 注: neg_l0 baked into shards once, alpha 仅在 train-ntp 时变化 → preprocess 一次

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
CKPT_DIR="experiments/ntp_checkpoints"
NTP_DATA_ENTP="experiments/ntp_data/exp041-entp-features"
DATE_START="2026-03-18"
DATE_END="2026-03-31"
EXPOSURE_NEG_PATH="/mnt/workspace/gr-demo-exposure-neg/2026-03-01_2026-03-31"

FORCE=false
SKIP_SMOKE=false
START_FROM=1
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --no-smoke) SKIP_SMOKE=true ;;
        --start-from=*) START_FROM="${arg#*=}" ;;
    esac
done

echo "=========================================="
echo "EXP-041: ENTP-Loss (Exposure Hard Negatives)"
echo "=========================================="
echo "  GPUs:      ${N_GPUS}"
echo "  SID cache: ${SID_CACHE}"
echo "  Data:      ${NTP_DATA_ENTP}"
echo "  Dates:     ${DATE_START} ~ ${DATE_END}"
echo "  Baseline:  exp036-full-features (already done, no entp)"
echo ""

# Sanity checks
if [ ! -d "${SID_CACHE}" ]; then
    echo "ERROR: SID cache not found at ${SID_CACHE}"
    exit 1
fi

# ── Step 1: Preprocess with ENTP neg (once for all alpha configs) ───────────
if [ ! -f "${NTP_DATA_ENTP}/meta.json" ] || [ "${FORCE}" == true ]; then
    echo ""
    echo ">>> Step 1: Preprocessing NTP data with ENTP negatives (K=5)..."
    echo "    exposure_neg_path: ${EXPOSURE_NEG_PATH}"
    torchrun --nproc_per_node="${N_GPUS}" run.py preprocess-ntp \
        --sid_cache "${SID_CACHE}" \
        --output_dir "${NTP_DATA_ENTP}" \
        --n_shards "${N_GPUS}" \
        --date_start "${DATE_START}" \
        --date_end "${DATE_END}" \
        --shift_features \
        --entp_weight 0.1 \
        --entp_k 5 \
        --exposure_neg_path "${EXPOSURE_NEG_PATH}"
    echo "  Preprocessing complete."
    python3 -c "
import json
m = json.load(open('${NTP_DATA_ENTP}/meta.json'))
print(f'  n_seqs={m[\"n_seqs\"]:,}  has_neg_l0={m[\"has_neg_l0\"]}  entp_k={m[\"entp_k\"]}')
" 2>/dev/null || true
else
    echo "  [data] ENTP data already exists at ${NTP_DATA_ENTP}, skipping preprocess."
    python3 -c "
import json
m = json.load(open('${NTP_DATA_ENTP}/meta.json'))
print(f'  n_seqs={m[\"n_seqs\"]:,}  has_neg_l0={m[\"has_neg_l0\"]}  entp_k={m[\"entp_k\"]}')
" 2>/dev/null || true
fi

# ── Smoke test ─────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" == false ]; then
    echo ""
    echo ">>> Smoke test (dry run, alpha=0.1)..."
    SMOKE_CKPT="${CKPT_DIR}/exp041-smoke"
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA_ENTP}" \
        --name exp041-smoke \
        --use_time_gap \
        --use_action_level \
        --use_segment_emb \
        --entp_weight 0.1 \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${SMOKE_CKPT}"
    echo ""
fi

# ── Helper: train one alpha config ────────────────────────────
train_entp() {
    local ALPHA=$1
    local NAME="exp041-entp${ALPHA//./}"   # e.g. 0.05 → entp005
    local DESC="ENTP α=${ALPHA}"
    local OUTPUT="${CKPT_DIR}/${NAME}"

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "============================================================"

    T0=$(date +%s)
    if [ -f "${OUTPUT}/probe.pt" ] && [ "${FORCE}" != true ]; then
        echo "  Checkpoint found, skipping (use --force to retrain)."
    else
        echo ">>> Training ${NAME} (entp_weight=${ALPHA})..."
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA_ENTP}" \
            --name "${NAME}" \
            --use_time_gap \
            --use_action_level \
            --use_segment_emb \
            --entp_weight "${ALPHA}" \
            --wandb
    fi
    T1=$(date +%s)
    TRAIN_MIN=$(( (T1 - T0) / 60 ))
    echo "  Training complete  (${TRAIN_MIN}min)"

    echo ">>> Full eval (n_recall=1000)..."
    T2=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT}" \
        --n_recall 1000
    T3=$(date +%s)
    EVAL_MIN=$(( (T3 - T2) / 60 ))
    TOTAL_MIN=$(( (T3 - T0) / 60 ))
    echo "  Total: train=${TRAIN_MIN}min  eval=${EVAL_MIN}min  total=${TOTAL_MIN}min"

    git add experiments/
    git commit -m "EXP-041 ${NAME}: ENTP alpha=${ALPHA}" || echo "Nothing to commit"
    ./push.sh
}

# ── Alpha sweep ────────────────────────────────────────────────
if [ "${START_FROM}" -le 1 ]; then
    train_entp "0.05"
fi
if [ "${START_FROM}" -le 2 ]; then
    train_entp "0.1"
fi
if [ "${START_FROM}" -le 3 ]; then
    train_entp "0.2"
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo ">>> EXP-041 Results Summary:"
python3 -c "
import json, os
configs = [
    ('exp036-full-features', 'α=0 (baseline, EXP-036)'),
    ('exp041-entp005',       'α=0.05'),
    ('exp041-entp01',        'α=0.1'),
    ('exp041-entp02',        'α=0.2'),
]
print(f'  {\"Config\":<25}  {\"R@10\":>6}  {\"R@500\":>7}  {\"PPL\":>7}  {\"L0 PPL\":>9}')
print(f'  {\"-\"*25}  {\"-\"*6}  {\"-\"*7}  {\"-\"*7}  {\"-\"*9}')
for name, desc in configs:
    path = f'experiments/ntp_checkpoints/{name}/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        e = m.get('eval', {})
        r10 = e.get('item_recall@10', 0)
        r500 = e.get('item_recall@500', 0)
        ppl = e.get('ppl', 0)
        # per-layer PPL if available
        layer_ppl = e.get('layer_ppl', {})
        l0_ppl = layer_ppl.get('L0', e.get('ppl_l0', '-'))
        w = m.get('train', {}).get('wall_time_s', 0)
        print(f'  {name:<25}  {r10:>6.1%}  {r500:>7.1%}  {ppl:>7.2f}  {str(l0_ppl):>9}  ({int(w)//60}min{int(w)%60}s)')
    else:
        print(f'  {name:<25}: not available')
" 2>/dev/null || echo "  Results not available"

echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-041 results: ENTP-Loss alpha sweep" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-041 complete!"
