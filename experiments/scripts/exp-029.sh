#!/bin/bash
set -euo pipefail

# EXP-029: ECPO + On-Policy Beam Search
#
# EXP-028 对照：相同超参，但用 policy model（非 ref model）生成 beam candidates。
# 解决 off-policy 偏差：policy 和 ref 偏离后，off-policy candidates 代表性下降。
#
# --on_policy_beam: beam search 改用 raw_policy（eval mode + no_grad），
#   ref model 仍用于计算参考 log-probs。
#
# 预期：importance ratio 更接近 1，clip 率下降，advantage 信号更有效。

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
echo "EXP-029: ECPO + On-Policy Beam Search"
echo "=========================================="
echo "  GPUs:             ${N_GPUS}"
echo "  NTP data:         ${NTP_DATA}"
echo "  SFT checkpoint:   ${SFT_CKPT}"
echo "  Behavior cache:   ${BEHAVIOR_CACHE}"
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
if [ ! -f "${CKPT_DIR}/exp029-smoke/probe.pt" ]; then
    echo ">>> Smoke test (2 steps, G=16)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT_DIR}/exp029-smoke" \
        --name exp029-smoke \
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
    rm -rf "${CKPT_DIR}/exp029-smoke"
    echo ""
fi

# ── Training ──────────────────────────────────────────────────
NAME="exp029-ecpo-onpolicy-w003-r100"
OUTPUT="${CKPT_DIR}/${NAME}"

if [ -f "${OUTPUT}/probe.pt" ]; then
    echo "  [${NAME}] Already exists, skipping."
else
    echo ">>> Training: ${NAME}"
    echo "    On-policy beam, WeightedBehaviorReward, grpo_weight=0.03, ratio=1.0, ECPO δ=0.1, lr=1e-4"

    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT}" \
        --name "${NAME}" \
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

    echo "  [${NAME}] Training complete"

    echo "  [${NAME}] Running full eval (n_recall=1000)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT}" \
        --n_recall 1000

    echo "  [${NAME}] Eval complete"
fi

echo ""
echo ">>> Final results:"
python -c "
import json
m = json.load(open('${CKPT_DIR}/${NAME}/train_meta.json'))
e = m.get('eval', {})
print(f'  ${NAME}: R@10={e.get(\"item_recall@10\",\"?\"):.3f}  R@500={e.get(\"item_recall@500\",\"?\"):.3f}')
" 2>/dev/null || echo "  ${NAME}: eval not available"

git add experiments/
git commit -m "EXP-029 complete: ECPO on-policy beam search results" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-029 complete!"
