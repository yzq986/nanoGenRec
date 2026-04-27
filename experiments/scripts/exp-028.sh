#!/bin/bash
set -euo pipefail

# EXP-028: ECPO + WeightedBehaviorReward — Continuous Quality×Freshness Reward
#
# EXP-027 结论（pending）：BehaviorReward 仍稀疏（97.5% SID reward=0），clip=98%。
#
# 本次改进：WeightedBehaviorReward
#   - 质量分：action_bitmap v0420 生产权重，log10(1+Σw)
#     place_order/follow=4000, comment=2000, share=3, like=1, click=0.1
#   - 新鲜度：exp(-age_hours/24)，τ=1天，与线上3d截止策略对齐
#   - 覆盖率：100%（SID cache 内所有 item 都有行为记录）
#
# 对照：EXP-027 最佳 config（w003-r100），完全相同超参，只换 reward 构建方式
#
# 预期：within-group reward 方差提升，clip 率下降，RL 梯度真正起作用

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
NTP_DATA="experiments/ntp_data/exp023-14d-features"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
BEHAVIOR_CACHE="/mnt/workspace/gr-demo-behavior-cache"
DATE_START="2026-03-18"
DATE_END="2026-03-31"
SFT_CKPT="${CKPT_DIR}/exp020-hard-lam03"

echo "=========================================="
echo "EXP-028: ECPO + WeightedBehaviorReward"
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
if [ ! -f "${CKPT_DIR}/exp028-smoke/probe.pt" ]; then
    echo ">>> Smoke test (2 steps, G=16)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT_DIR}/exp028-smoke" \
        --name exp028-smoke \
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
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${CKPT_DIR}/exp028-smoke"
    echo ""
fi

# ── Config A: WeightedBehaviorReward (identical hyperparams to EXP-027-A) ────
NAME="exp028-ecpo-weighted-w003-r100"
OUTPUT="${CKPT_DIR}/${NAME}"

if [ -f "${OUTPUT}/probe.pt" ]; then
    echo "  [${NAME}] Already exists, skipping."
else
    echo ">>> Training: ${NAME}"
    echo "    WeightedBehaviorReward, grpo_weight=0.03, rl_data_ratio=1.0, ECPO δ=0.1, lr=1e-4"

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
        --reward_format --format_weight 0.5

    echo "  [${NAME}] Training complete"

    echo "  [${NAME}] Running full eval (n_recall=1000)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT}" \
        --n_recall 1000

    echo "  [${NAME}] Eval complete"
    echo ""

    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-028: ${NAME} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/exp028-git.lock
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
git commit -m "EXP-028 complete: ECPO WeightedBehaviorReward results" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-028 complete!"

# ── Auto-chain: EXP-029 ───────────────────────────────────────
echo ""
echo ">>> Chaining into EXP-029..."
bash "$(dirname "$0")/exp-029.sh"
