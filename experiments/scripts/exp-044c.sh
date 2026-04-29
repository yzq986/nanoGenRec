#!/bin/bash
set -euo pipefail

# EXP-044C: TO-RoPE with item-level positions (pos//L fix)
# Date: 2026-04-29
#
# EXP-044B (修复 timestamps 注入后): R@500=63.6% (ts=0.25+tg), 63.5% (ts=0.5 notg)
# 但 PPL 仍高达 467-480，怀疑原因：
#   position-RoPE 用 token 级别连续索引 (0,1,2,3,...)
#   time-RoPE 把同 item 内 tokens 视为同时刻 (ts 相同)
#   → 两个 RoPE 信号在 item 内产生矛盾
#
# 本次修复：position 改为 item-level (pos//L)，即 0,0,0,1,1,1,2,2,2,...
# NTP data 复用 exp044b-0.6b-14d（timestamps 已接通，无需重跑 preprocess）
#
# Configs:
#   A: TO-RoPE ts=0.25 + time_gap (exp044b 最佳 config，重跑验证)
#   B: TO-RoPE ts=0.5  + time_gap
#   C: TO-RoPE ts=0.25 无 time_gap (消融)
#
# Baseline: exp043-s-0.6b (R@500=61.2%)
# EXP-044B best: exp044b-torope-ts025 (R@500=63.6%)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export EFS_BASE="${EFS_BASE:-/mnt/workspace}"
cd "${REPO_ROOT}"

GR_PYTHON="/home/dev/.conda/envs/gr/bin/python"
[ -f "${GR_PYTHON}" ] && export PATH="/home/dev/.conda/envs/gr/bin:${PATH}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
SID_CACHE="experiments/sid_cache/exp026-0.6b-14d"
NTP_DATA="experiments/ntp_data/exp044b-0.6b-14d"   # reuse — timestamps already computed
OUTPUT="experiments/ntp_checkpoints"

echo "=========================================="
echo "EXP-044C: TO-RoPE item-level position fix"
echo "  GPUs:      ${N_GPUS}"
echo "  SID cache: ${SID_CACHE}"
echo "  NTP data:  ${NTP_DATA} (reused from 044B)"
echo "=========================================="

# Sanity checks
if [ ! -f "${SID_CACHE}/config.json" ]; then
    echo "ERROR: SID cache not found: ${SID_CACHE}"; exit 1
fi
if [ ! -d "${NTP_DATA}" ]; then
    echo "ERROR: NTP data not found: ${NTP_DATA}"; exit 1
fi

# ── Smoke test ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Smoke test..."
SMOKE_OUT="${OUTPUT}/exp044c-smoke"
torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
    --preprocessed_dir "${NTP_DATA}" \
    --output_dir "${SMOKE_OUT}" \
    --name exp044c-smoke \
    --model s-tier \
    --use_torope \
    --torope_time_split 0.25 \
    --use_segment_emb \
    --active_features time_gaps action_levels timestamps \
    --smoke_test
echo "  Smoke test passed."

# ── Train helper ─────────────────────────────────────────────────────────────
train_config() {
    local NAME=$1
    shift
    local EXTRA_FLAGS="$@"

    local CKPT="${OUTPUT}/${NAME}"
    if [ -f "${CKPT}/train_meta.json" ]; then
        echo "  [skip] ${NAME} already trained"
        return
    fi

    echo ""
    echo ">>> Training ${NAME}..."
    T0=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT}" \
        --name "${NAME}" \
        --model s-tier \
        --use_torope \
        --use_segment_emb \
        ${EXTRA_FLAGS}
    T1=$(date +%s)
    echo "  Training done ($(( (T1 - T0) / 60 ))min)"

    echo ">>> Evaluating ${NAME}..."
    T2=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${CKPT}" \
        --n_recall 1000
    T3=$(date +%s)
    echo "  Eval done ($(( (T3 - T2) / 60 ))min)  total=$(( (T3 - T0) / 60 ))min"
}

# Config A: ts=0.25 + time_gap (replicate 044B best with pos fix)
train_config "exp044c-torope-ts025" \
    --torope_time_split 0.25 \
    --active_features time_gaps action_levels timestamps

# Config B: ts=0.5 + time_gap
train_config "exp044c-torope-ts05" \
    --torope_time_split 0.5 \
    --active_features time_gaps action_levels timestamps

# Config C: ts=0.25 no time_gap (ablation)
train_config "exp044c-torope-ts025-notg" \
    --torope_time_split 0.25 \
    --active_features action_levels timestamps

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo ">>> Results summary:"
for NAME in exp044c-torope-ts025 exp044c-torope-ts05 exp044c-torope-ts025-notg; do
    CKPT="${OUTPUT}/${NAME}"
    META="${CKPT}/train_meta.json"
    if [ -f "${META}" ]; then
        python3 -c "
import json
m = json.load(open('${META}'))
e = m.get('eval', m)
r10  = e.get('item_recall@10',  '?')
r500 = e.get('item_recall@500', '?')
ppl  = e.get('ppl', '?')
print(f'  ${NAME}: R@10={r10}  R@500={r500}  PPL={ppl}')
" 2>/dev/null || echo "  ${NAME}: (parse error)"
    fi
done

# ── Commit ────────────────────────────────────────────────────────────────────
echo ""
echo ">>> Committing results..."
git add experiments/ntp_checkpoints/exp044c-*/ 2>/dev/null || true
git commit -m "EXP-044C complete: TO-RoPE item-level pos fix" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-044C complete!"
