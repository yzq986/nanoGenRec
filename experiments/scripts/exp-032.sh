#!/bin/bash
set -euo pipefail

# EXP-032: GRPO Group Size vs Context Diversity — G × batch_size Sweep
# Date: 2026-04-28
#
# 验证假设：在总 candidate 预算相同（G × grpo_batch ≈ 2048）的前提下，
# 更多 context（小 G + 大 batch）是否优于更多 per-context candidates（大 G + 小 batch）。
#
# Config A (control): G=512, batch=4   → 复现 EXP-029 baseline
# Config B:           G=128, batch=16  → 4× context 多样性
# Config C:           G=32,  batch=64  → 16× context 多样性
#
# 所有 config 保持其他参数与 EXP-029 完全一致（on-policy ECPO，无额外 reward shaping）

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
NTP_DATA="experiments/ntp_data/exp023-14d-features"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
BEHAVIOR_CACHE="/mnt/workspace/gr-demo-behavior-cache"
DATE_END="2026-03-31"
SFT_CKPT="${CKPT_DIR}/exp020-hard-lam03"

echo "=========================================="
echo "EXP-032: GRPO Group Size vs Context Diversity"
echo "=========================================="
echo "  GPUs:             ${N_GPUS}"
echo "  NTP data:         ${NTP_DATA}"
echo "  SFT checkpoint:   ${SFT_CKPT}"
echo "  Behavior cache:   ${BEHAVIOR_CACHE}"
echo "  Configs: A(G=512,b=4)  B(G=128,b=16)  C(G=32,b=64)"
echo ""

# Sanity checks
if [ ! -f "${SFT_CKPT}/probe.pt" ]; then
    echo "ERROR: SFT checkpoint not found at ${SFT_CKPT}"
    exit 1
fi
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo "ERROR: NTP data not found at ${NTP_DATA}"
    exit 1
fi
if [ ! -d "${BEHAVIOR_CACHE}/2026-03-31" ]; then
    echo "ERROR: behavior cache not found at ${BEHAVIOR_CACHE}"
    exit 1
fi

# ── Smoke test ────────────────────────────────────────────────
if [ ! -f "${CKPT_DIR}/exp032-smoke/probe.pt" ]; then
    echo ">>> Smoke test (2 steps, G=16, batch=2)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT_DIR}/exp032-smoke" \
        --name exp032-smoke \
        --eps 0.2 --delta 0.1 \
        --grpo_weight 0.03 \
        --group_size 16 \
        --grpo_batch_size 2 \
        --rl_data_ratio 1.0 \
        --lr 1e-4 \
        --reward_behavior --behavior_weight 1.0 \
        --behavior_cache_dir "${BEHAVIOR_CACHE}" \
        --behavior_cache_eval_date "${DATE_END}" \
        --reward_format --format_weight 0.5 \
        --on_policy_beam \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${CKPT_DIR}/exp032-smoke"
    echo ""
fi

# ── Config A: G=512, batch=4 (EXP-029 control) ───────────────
NAME_A="exp032-G512-b4"
OUTPUT_A="${CKPT_DIR}/${NAME_A}"

if [ -f "${OUTPUT_A}/probe.pt" ]; then
    echo "  [${NAME_A}] Already exists, skipping."
else
    echo ">>> Config A: ${NAME_A}  (G=512, grpo_batch=4 — control, reproduces EXP-029)"

    T0_A=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT_A}" \
        --name "${NAME_A}" \
        --eps 0.2 --delta 0.1 \
        --grpo_weight 0.03 \
        --group_size 512 \
        --grpo_batch_size 4 \
        --rl_data_ratio 1.0 \
        --lr 1e-4 \
        --reward_behavior --behavior_weight 1.0 \
        --behavior_cache_dir "${BEHAVIOR_CACHE}" \
        --behavior_cache_eval_date "${DATE_END}" \
        --reward_format --format_weight 0.5 \
        --on_policy_beam
    T1_A=$(date +%s)
    TRAIN_MIN_A=$(( (T1_A - T0_A) / 60 ))
    echo "  [${NAME_A}] Training complete  (${TRAIN_MIN_A}min)"

    echo "  [${NAME_A}] Running full eval (n_recall=1000)..."
    T2_A=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT_A}" \
        --n_recall 1000
    T3_A=$(date +%s)
    EVAL_MIN_A=$(( (T3_A - T2_A) / 60 ))
    TOTAL_MIN_A=$(( (T3_A - T0_A) / 60 ))
    echo "  [${NAME_A}] Eval complete  (${EVAL_MIN_A}min)"
    echo "  [${NAME_A}] Total: train=${TRAIN_MIN_A}min  eval=${EVAL_MIN_A}min  total=${TOTAL_MIN_A}min"
    echo ""

    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-032: ${NAME_A} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/exp032-git.lock
fi

# ── Config B: G=128, batch=16 ────────────────────────────────
NAME_B="exp032-G128-b16"
OUTPUT_B="${CKPT_DIR}/${NAME_B}"

if [ -f "${OUTPUT_B}/probe.pt" ]; then
    echo "  [${NAME_B}] Already exists, skipping."
else
    echo ">>> Config B: ${NAME_B}  (G=128, grpo_batch=16 — 4× context diversity)"

    T0_B=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT_B}" \
        --name "${NAME_B}" \
        --eps 0.2 --delta 0.1 \
        --grpo_weight 0.03 \
        --group_size 128 \
        --grpo_batch_size 16 \
        --rl_data_ratio 1.0 \
        --lr 1e-4 \
        --reward_behavior --behavior_weight 1.0 \
        --behavior_cache_dir "${BEHAVIOR_CACHE}" \
        --behavior_cache_eval_date "${DATE_END}" \
        --reward_format --format_weight 0.5 \
        --on_policy_beam
    T1_B=$(date +%s)
    TRAIN_MIN_B=$(( (T1_B - T0_B) / 60 ))
    echo "  [${NAME_B}] Training complete  (${TRAIN_MIN_B}min)"

    echo "  [${NAME_B}] Running full eval (n_recall=1000)..."
    T2_B=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT_B}" \
        --n_recall 1000
    T3_B=$(date +%s)
    EVAL_MIN_B=$(( (T3_B - T2_B) / 60 ))
    TOTAL_MIN_B=$(( (T3_B - T0_B) / 60 ))
    echo "  [${NAME_B}] Eval complete  (${EVAL_MIN_B}min)"
    echo "  [${NAME_B}] Total: train=${TRAIN_MIN_B}min  eval=${EVAL_MIN_B}min  total=${TOTAL_MIN_B}min"
    echo ""

    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-032: ${NAME_B} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/exp032-git.lock
fi

# ── Config C: G=32, batch=64 ─────────────────────────────────
NAME_C="exp032-G32-b64"
OUTPUT_C="${CKPT_DIR}/${NAME_C}"

if [ -f "${OUTPUT_C}/probe.pt" ]; then
    echo "  [${NAME_C}] Already exists, skipping."
else
    echo ">>> Config C: ${NAME_C}  (G=32, grpo_batch=64 — 16× context diversity)"

    T0_C=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT_C}" \
        --name "${NAME_C}" \
        --eps 0.2 --delta 0.1 \
        --grpo_weight 0.03 \
        --group_size 32 \
        --grpo_batch_size 64 \
        --rl_data_ratio 1.0 \
        --lr 1e-4 \
        --reward_behavior --behavior_weight 1.0 \
        --behavior_cache_dir "${BEHAVIOR_CACHE}" \
        --behavior_cache_eval_date "${DATE_END}" \
        --reward_format --format_weight 0.5 \
        --on_policy_beam
    T1_C=$(date +%s)
    TRAIN_MIN_C=$(( (T1_C - T0_C) / 60 ))
    echo "  [${NAME_C}] Training complete  (${TRAIN_MIN_C}min)"

    echo "  [${NAME_C}] Running full eval (n_recall=1000)..."
    T2_C=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT_C}" \
        --n_recall 1000
    T3_C=$(date +%s)
    EVAL_MIN_C=$(( (T3_C - T2_C) / 60 ))
    TOTAL_MIN_C=$(( (T3_C - T0_C) / 60 ))
    echo "  [${NAME_C}] Eval complete  (${EVAL_MIN_C}min)"
    echo "  [${NAME_C}] Total: train=${TRAIN_MIN_C}min  eval=${EVAL_MIN_C}min  total=${TOTAL_MIN_C}min"
    echo ""

    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-032: ${NAME_C} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/exp032-git.lock
fi

echo ""
echo ">>> Final results:"
python3 -c "
import json, sys, os
configs = [
    ('${NAME_A}', 'G=512 batch=4  (control)'),
    ('${NAME_B}', 'G=128 batch=16 (4x ctx) '),
    ('${NAME_C}', 'G=32  batch=64 (16x ctx)'),
]
for name, label in configs:
    path = 'experiments/ntp_checkpoints/' + name + '/train_meta.json'
    try:
        m = json.load(open(path))
        e = m.get('eval', {})
        t = m.get('train', {})
        clip = t.get('avg_clip_fraction', float('nan'))
        adv_std = t.get('avg_advantage_std', float('nan'))
        print(f'  {label}: R@10={e.get(\"item_recall@10\",float(\"nan\")):.3f}  R@500={e.get(\"item_recall@500\",float(\"nan\")):.3f}  clip={clip:.3f}  adv_std={adv_std:.3f}')
    except Exception as ex:
        print(f'  {label}: not available ({ex})')
" 2>/dev/null || true

echo ""
echo "  Baseline (EXP-029, G=512 b=4): R@500=0.678  clip=0.923"

echo ""
echo ">>> Timing summary:"
python3 -c "
import json, os
for name in ['${NAME_A}', '${NAME_B}', '${NAME_C}']:
    path = 'experiments/ntp_checkpoints/' + name + '/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        w = m.get('train', {}).get('wall_time_s', 0)
        print(f'  {name}: train={int(w)//60}min{int(w)%60:02d}s')
    else:
        print(f'  {name}: not found')
" 2>/dev/null || true

git add experiments/
git commit -m "EXP-032 complete: G×batch context diversity sweep results" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-032 complete!"
