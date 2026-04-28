#!/bin/bash
set -euo pipefail

# EXP-040: RSFT — Reject Sampling Fine-Tuning (Training Data Quality Filter)
# Date: 2026-04-28
#
# IDEA-onerec-1: OneRec post-training 阶段 — 按行为质量过滤训练数据
# 对比: 全量行为数据 vs 仅强行为(like/fav/share/purchase) 训练
#
# 实验设计:
#   Config A (baseline): min_action_level=1 (全量, 等价于 exp036-full-features)
#   Config B (RSFT-2):   min_action_level=2 (仅 strong+trade, 约 20-30% 数据)
#   Config C (RSFT-3):   min_action_level=3 (仅 trade, 约 5-10% 数据)
#
# 注: Config A 已有 exp036-full-features 结果 (R@500=59.0%)，直接引用，不重训

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
CKPT_DIR="experiments/ntp_checkpoints"
DATA_BASE="experiments/ntp_data"
DATE_START="2026-03-18"
DATE_END="2026-03-31"
BEHAVIOR_CACHE="/mnt/workspace/gr-demo-behavior-cache"

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
echo "EXP-040: RSFT — Action Quality Filter"
echo "=========================================="
echo "  GPUs:  ${N_GPUS}"
echo "  Dates: ${DATE_START} ~ ${DATE_END}"
echo "  Baseline (A): exp036-full-features (min_action_level=1, already done)"
echo "  Config B: RSFT-2 (min_action_level=2, strong+trade)"
echo "  Config C: RSFT-3 (min_action_level=3, trade only)"
echo ""

# Check baseline exists
if [ ! -f "${CKPT_DIR}/exp036-full-features/train_meta.json" ]; then
    echo "WARNING: exp036-full-features not found; baseline numbers will be missing."
fi

# ── Helper: preprocess + train one config ────────────────────
run_rsft_config() {
    local MIN_LEVEL=$1
    local NAME=$2
    local DESC=$3
    local DATA_DIR="${DATA_BASE}/exp040-rsft${MIN_LEVEL}"
    local OUTPUT="${CKPT_DIR}/${NAME}"

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "  min_action_level=${MIN_LEVEL}"
    echo "  data: ${DATA_DIR}"
    echo "============================================================"

    # Step 1: Preprocess data with quality filter
    if [ ! -f "${DATA_DIR}/meta.json" ] || [ "${FORCE}" == true ]; then
        echo ">>> Preprocessing data (min_action_level=${MIN_LEVEL})..."
        torchrun --nproc_per_node="${N_GPUS}" run.py preprocess-ntp \
            --sid_cache "${SID_CACHE}" \
            --output_dir "${DATA_DIR}" \
            --n_shards "${N_GPUS}" \
            --date_start "${DATE_START}" \
            --date_end "${DATE_END}" \
            --shift_features \
            --min_action_level "${MIN_LEVEL}" \
            --behavior_path "${BEHAVIOR_CACHE}"
        echo "  Preprocessing complete."
    else
        echo "  [data] Already exists at ${DATA_DIR}, skipping preprocessing."
        python3 -c "
import json
m = json.load(open('${DATA_DIR}/meta.json'))
print(f'  n_seqs={m[\"n_seqs\"]:,}  min_action_level={m.get(\"min_action_level\",1)}')
" 2>/dev/null || true
    fi

    # Step 2: Train
    T0=$(date +%s)
    if [ -f "${OUTPUT}/probe.pt" ] && [ "${FORCE}" != true ]; then
        echo "  [${NAME}] Checkpoint found, skipping training (use --force to retrain)."
    else
        echo ">>> Training: ${NAME}..."
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${DATA_DIR}" \
            --name "${NAME}" \
            --use_time_gap \
            --use_action_level \
            --use_segment_emb
    fi
    T1=$(date +%s)
    TRAIN_MIN=$(( (T1 - T0) / 60 ))
    echo "  Training complete  (${TRAIN_MIN}min)"

    # Step 3: Full eval
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
    git commit -m "EXP-040 ${NAME}: RSFT min_action_level=${MIN_LEVEL}" || echo "Nothing to commit"
    ./push.sh
}

# ── Smoke test ────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" == false ]; then
    echo ""
    echo ">>> Smoke test (dry run, min_action_level=2)..."
    SMOKE_DATA="${DATA_BASE}/exp040-smoke"
    SMOKE_CKPT="${CKPT_DIR}/exp040-smoke"
    if [ ! -f "${SMOKE_DATA}/meta.json" ]; then
        torchrun --nproc_per_node="${N_GPUS}" run.py preprocess-ntp \
            --sid_cache "${SID_CACHE}" \
            --output_dir "${SMOKE_DATA}" \
            --n_shards "${N_GPUS}" \
            --date_start "${DATE_END}" \
            --date_end "${DATE_END}" \
            --shift_features \
            --min_action_level 2 \
            --behavior_path "${BEHAVIOR_CACHE}"
    fi
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${SMOKE_DATA}" \
        --name exp040-smoke \
        --use_time_gap \
        --use_action_level \
        --use_segment_emb \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${SMOKE_CKPT}" "${SMOKE_DATA}"
    echo ""
fi

# ── Config B: min_action_level=2 (strong+trade) ───────────────
if [ "${START_FROM}" -le 1 ]; then
    run_rsft_config 2 "exp040-rsft2" "RSFT-2: strong+trade interactions only"
fi

# ── Config C: min_action_level=3 (trade only) ────────────────
if [ "${START_FROM}" -le 2 ]; then
    run_rsft_config 3 "exp040-rsft3" "RSFT-3: trade (purchase) interactions only"
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo ">>> EXP-040 Results Summary:"
python3 -c "
import json, os
configs = [
    ('exp036-full-features', 'A: all (baseline, EXP-036)'),
    ('exp040-rsft2',         'B: min_level=2 (strong+trade)'),
    ('exp040-rsft3',         'C: min_level=3 (trade only)'),
]
for name, desc in configs:
    path = f'experiments/ntp_checkpoints/{name}/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        e = m.get('eval', {})
        r10 = e.get('item_recall@10', 0)
        r500 = e.get('item_recall@500', 0)
        ppl = e.get('ppl', 0)
        w = m.get('train', {}).get('wall_time_s', 0)
        print(f'  {name:<25} ({desc:<35}): R@10={r10:.1%}  R@500={r500:.1%}  PPL={ppl:.2f}  ({int(w)//60}min{int(w)%60}s)')
    else:
        print(f'  {name:<25}: not available')
" 2>/dev/null || echo "  Results not available"

echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-040 results: RSFT action quality filter ablation" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-040 complete!"
