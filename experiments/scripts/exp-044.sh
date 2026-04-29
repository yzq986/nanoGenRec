#!/bin/bash
set -euo pipefail

# EXP-044: TO-RoPE vs Absolute Pos Emb — Time-and-Order RoPE (IDEA-feat-5)
# Date: 2026-04-29
# arxiv 2510.20455 (Roblox split-by-dim variant)
#
# TO-RoPE 将 order index + wall-clock time 统一编码到 Q/K 旋转中，
# 替换当前的 learnable pos_emb + time_gap bucket embedding。
# 优势：时间信息自然进入 KV cache，beam search 无需手动传递 time_gap。
#
# 实验设计:
#   Config A (baseline): --use_time_gap --use_action_level --use_segment_emb
#                        → 直接引用 exp043-s-0.6b（如已完成）
#   Config B: TO-RoPE (time_split=0.5) + action_level + segment_emb
#             (去掉 time_gap，改用 TO-RoPE 编码时间)
#   Config C: TO-RoPE (time_split=0.25) + action_level + segment_emb
#             (较少维度给时间，更多给 order)
#   Config D: TO-RoPE (time_split=0.5) + segment_emb only
#             (消融：去掉 action_level，看 TO-RoPE 本身贡献)
#
# SID cache: exp026-0.6b-14d (与 EXP-043 S-tier 0.6b 对齐)
# NTP data: 复用 experiments/ntp_data/exp043-0.6b-14d（EXP-043 已生成）

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
SID_CACHE="experiments/sid_cache/exp026-0.6b-14d"
NTP_DATA="experiments/ntp_data/exp043-0.6b-14d"
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
echo "EXP-044: TO-RoPE vs Absolute Pos Emb"
echo "=========================================="
echo "  GPUs:      ${N_GPUS}"
echo "  SID cache: ${SID_CACHE}"
echo "  NTP data:  ${NTP_DATA}"
echo ""

# Sanity checks
if [ ! -f "${SID_CACHE}/config.json" ]; then
    echo "ERROR: SID cache not found at ${SID_CACHE}"
    exit 1
fi

# ── Ensure NTP data exists (reuse from EXP-043 if available) ─
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo ">>> NTP data not found, preprocessing (exp043-0.6b-14d)..."
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
python3 -c "
import json
m = json.load(open('${NTP_DATA}/meta.json'))
print(f'  n_seqs={m[\"n_seqs\"]:,}  n_eval_items={m[\"n_eval_items\"]:,}')
" 2>/dev/null || true

# ── Smoke test ────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" == false ] && [ "${START_FROM}" -le 1 ]; then
    echo ""
    echo ">>> Smoke test (TO-RoPE dry run)..."
    SMOKE_OUT="${CKPT_DIR}/exp044-smoke"
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_OUT}" \
        --name exp044-smoke \
        --model s-tier \
        --use_torope \
        --use_action_level \
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
    git commit -m "EXP-044 ${NAME}: ${DESC}" || echo "Nothing to commit"
    ./push.sh
}

# ── Configs ───────────────────────────────────────────────────
# Config A: baseline (absolute pos + time_gap + action + segment)
# → 直接引用 exp043-s-0.6b，不重训
if [ "${START_FROM}" -le 1 ]; then
    if [ -f "${CKPT_DIR}/exp043-s-0.6b/train_meta.json" ]; then
        echo ""
        echo "  [Config A] exp043-s-0.6b already exists, using as baseline."
    else
        train_eval "exp044-baseline" \
            "Baseline: abs pos + time_gap + action + segment" \
            "--use_time_gap --use_action_level --use_segment_emb"
    fi
fi

# Config B: TO-RoPE time_split=0.5 + action + segment
[ "${START_FROM}" -le 2 ] && train_eval \
    "exp044-torope-ts05" \
    "TO-RoPE time_split=0.5 + action + segment" \
    "--use_torope --torope_time_split 0.5 --use_action_level --use_segment_emb"

# Config C: TO-RoPE time_split=0.25 + action + segment
[ "${START_FROM}" -le 3 ] && train_eval \
    "exp044-torope-ts025" \
    "TO-RoPE time_split=0.25 + action + segment" \
    "--use_torope --torope_time_split 0.25 --use_action_level --use_segment_emb"

# Config D: TO-RoPE time_split=0.5 + segment only (ablation)
[ "${START_FROM}" -le 4 ] && train_eval \
    "exp044-torope-ts05-noseg" \
    "TO-RoPE time_split=0.5 + segment only (no action)" \
    "--use_torope --torope_time_split 0.5 --use_segment_emb"

# ── Summary ───────────────────────────────────────────────────
echo ""
echo ">>> EXP-044 Results Summary:"
python3 -c "
import json, os

configs = [
    ('exp043-s-0.6b',         'Baseline (abs pos + time_gap + action + seg)'),
    ('exp044-baseline',       'Baseline (reproduced, if 043 not done)'),
    ('exp044-torope-ts05',    'TO-RoPE ts=0.5 + action + seg'),
    ('exp044-torope-ts025',   'TO-RoPE ts=0.25 + action + seg'),
    ('exp044-torope-ts05-noseg', 'TO-RoPE ts=0.5 + seg only'),
]
print(f'  {\"Config\":<30}  {\"R@10\":>6}  {\"R@500\":>7}  {\"PPL\":>7}  {\"L0 PPL\":>8}')
print(f'  {\"-\"*30}  {\"-\"*6}  {\"-\"*7}  {\"-\"*7}  {\"-\"*8}')
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
        print(f'  {name:<30}  {r10:>6.1%}  {r500:>7.1%}  {ppl:>7.2f}  {str(l0):>8}  ({int(w)//60}min)')
    else:
        print(f'  {name:<30}: not available')
" 2>/dev/null || echo "  Results not available"

echo ""
git add experiments/
git commit -m "EXP-044 complete: TO-RoPE vs abs pos emb results" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-044 complete!"
