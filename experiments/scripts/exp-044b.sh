#!/bin/bash
set -euo pipefail

# EXP-044B: TO-RoPE 重跑 — 真实 timestamps + time_gap_emb 共存
# Date: 2026-04-29
#
# EXP-044 无效原因：
#   1. timestamps 全为 0（preprocess-ntp pipeline 未接通）
#   2. time_gap_emb 被 use_torope=True 的条件块屏蔽（已修复）
#
# 修复后重跑，使用真实 rel_hours timestamps:
#   rel_hours[i] = (first_ts[i] - first_ts[0]) / 3600
#
# 本次 NTP 数据必须重新 preprocess（--use_torope 触发 timestamps 写入）
# NTP data: experiments/ntp_data/exp044b-0.6b-14d（新目录，带 timestamps）
#
# Configs:
#   A (baseline): abs pos + time_gap + action + segment  → 引用 exp043-s-0.6b
#   B: TO-RoPE ts=0.5 + time_gap + action + segment     (time_gap 共存)
#   C: TO-RoPE ts=0.25 + time_gap + action + segment
#   D: TO-RoPE ts=0.5 + action + segment only           (无 time_gap，消融)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
SID_CACHE="experiments/sid_cache/exp026-0.6b-14d"
NTP_DATA="experiments/ntp_data/exp044b-0.6b-14d"
BEHAVIOR_CACHE="/mnt/workspace/gr-demo-behavior-cache"
DATE_START="2026-03-18"
DATE_END="2026-03-31"

FORCE=false
SKIP_SMOKE=false
START_FROM=1
for arg in "$@"; do
    case "$arg" in
        --force)        FORCE=true ;;
        --no-smoke)     SKIP_SMOKE=true ;;
        --start-from=*) START_FROM="${arg#*=}" ;;
    esac
done

echo "=========================================="
echo "EXP-044B: TO-RoPE with Real Timestamps"
echo "=========================================="
echo "  GPUs:      ${N_GPUS}"
echo "  SID cache: ${SID_CACHE}"
echo "  NTP data:  ${NTP_DATA}"
echo ""

if [ ! -f "${SID_CACHE}/config.json" ]; then
    echo "ERROR: SID cache not found at ${SID_CACHE}"
    exit 1
fi

# ── Step 1: Preprocess NTP data WITH timestamps ───────────────
if [ "${START_FROM}" -le 1 ]; then
    if [ ! -f "${NTP_DATA}/meta.json" ] || [ "${FORCE}" == true ]; then
        echo ">>> Preprocessing NTP data with timestamps (exp044b-0.6b-14d)..."
        torchrun --nproc_per_node="${N_GPUS}" run.py preprocess-ntp \
            --sid_cache "${SID_CACHE}" \
            --output_dir "${NTP_DATA}" \
            --n_shards "${N_GPUS}" \
            --date_start "${DATE_START}" \
            --date_end "${DATE_END}" \
            --behavior_path "${BEHAVIOR_CACHE}" \
            --shift_features
    else
        echo "  [data] ${NTP_DATA} found, skipping preprocess."
    fi
    # Verify timestamps are present
    python3 -c "
import numpy as np
s = np.load('${NTP_DATA}/train_shard_0.npz', allow_pickle=True)
keys = list(s.keys())
print(f'  shard keys: {keys}')
if 'timestamps' in keys:
    ts = s['timestamps']
    print(f'  timestamps: shape={ts.shape}, non-zero={int((ts != 0).sum()):,}/{ts.size:,}')
else:
    print('  ERROR: timestamps field missing!')
    exit(1)
" || { echo "ERROR: timestamps not in NTP data"; exit 1; }
fi

# ── Smoke test ────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" == false ] && [ "${START_FROM}" -le 1 ]; then
    echo ""
    echo ">>> Smoke test..."
    SMOKE_OUT="${CKPT_DIR}/exp044b-smoke"
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_OUT}" \
        --name exp044b-smoke \
        --model s-tier \
        --use_torope \
        --use_segment_emb \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${SMOKE_OUT}"
    echo ""
fi

# ── Helper ────────────────────────────────────────────────────
train_eval() {
    local NAME=$1
    local DESC=$2
    shift 2
    local EXTRA_FLAGS="$@"
    local OUTPUT="${CKPT_DIR}/${NAME}"

    echo ""
    echo "============================================================"
    echo "[${NAME}] ${DESC}"
    echo "============================================================"

    T0=$(date +%s)
    if [ -f "${OUTPUT}/train_meta.json" ] && [ "${FORCE}" != true ]; then
        echo "  Checkpoint found, skipping training."
    else
        echo ">>> Training..."
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${OUTPUT}" \
            --name "${NAME}" \
            --model s-tier \
            ${EXTRA_FLAGS}
    fi
    T1=$(date +%s)
    echo "  Training complete  ($(( (T1 - T0) / 60 ))min)"

    echo ">>> Full eval (n_recall=1000)..."
    T2=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT}" \
        --n_recall 1000
    T3=$(date +%s)
    echo "  Eval complete  ($(( (T3 - T2) / 60 ))min)  total=$(( (T3 - T0) / 60 ))min"

    git add experiments/
    git commit -m "EXP-044B ${NAME}: ${DESC}" || echo "Nothing to commit"
    ./push.sh
}

# ── Configs ───────────────────────────────────────────────────
# Config A: baseline → 引用 exp043-s-0.6b，不重训
echo ""
echo "  [Config A] Baseline: exp043-s-0.6b (abs pos + time_gap + action + seg)"

# Config B: TO-RoPE ts=0.5 + time_gap + action + segment
[ "${START_FROM}" -le 2 ] && train_eval \
    "exp044b-torope-ts05" \
    "TO-RoPE ts=0.5 + time_gap + action + segment" \
    "--use_torope --torope_time_split 0.5 --use_segment_emb"

# Config C: TO-RoPE ts=0.25 + time_gap + action + segment
[ "${START_FROM}" -le 3 ] && train_eval \
    "exp044b-torope-ts025" \
    "TO-RoPE ts=0.25 + time_gap + action + segment" \
    "--use_torope --torope_time_split 0.25 --use_segment_emb"

# Config D: TO-RoPE ts=0.5 + action + segment (无 time_gap，消融)
[ "${START_FROM}" -le 4 ] && train_eval \
    "exp044b-torope-ts05-notg" \
    "TO-RoPE ts=0.5 + action + segment (no time_gap)" \
    "--use_torope --torope_time_split 0.5 --use_segment_emb"

# ── Summary ───────────────────────────────────────────────────
echo ""
echo ">>> EXP-044B Results Summary:"
python3 -c "
import json, os

configs = [
    ('exp043-s-0.6b',              'Baseline (abs pos + time_gap + action + seg)'),
    ('exp044b-torope-ts05',        'TO-RoPE ts=0.5 + time_gap + action + seg'),
    ('exp044b-torope-ts025',       'TO-RoPE ts=0.25 + time_gap + action + seg'),
    ('exp044b-torope-ts05-notg',   'TO-RoPE ts=0.5 + action + seg (no time_gap)'),
]
print(f'  {\"Config\":<32}  {\"R@10\":>6}  {\"R@500\":>7}  {\"PPL\":>7}  {\"L0 PPL\":>8}')
print(f'  {\"-\"*32}  {\"-\"*6}  {\"-\"*7}  {\"-\"*7}  {\"-\"*8}')
for name, desc in configs:
    path = f'experiments/ntp_checkpoints/{name}/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        e = m.get('eval', {})
        r10  = e.get('item_recall@10', 0)
        r500 = e.get('item_recall@500', 0)
        ppl  = e.get('ppl', 0)
        lppl = e.get('layer_ppl', [])
        l0   = lppl[0] if lppl else '-'
        w    = m.get('train', {}).get('wall_time_s', 0)
        print(f'  {name:<32}  {r10:>6.1%}  {r500:>7.1%}  {ppl:>7.2f}  {str(l0):>8}  ({int(w)//60}min)')
    else:
        print(f'  {name:<32}: not available')
" 2>/dev/null || echo "  Results not available"

echo ""
git add experiments/
git commit -m "EXP-044B complete: TO-RoPE with real timestamps" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-044B complete!"
