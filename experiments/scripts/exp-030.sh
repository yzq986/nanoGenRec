#!/bin/bash
set -euo pipefail

# EXP-030: A2PO + NLL Regularization + HEPO Prefix Scoring
#
# 三项改进同时引入（均为 grpo.py/trainer.py/reward.py 的小改动）：
#
# A. HEPO（Hierarchical Evidence Policy Optimization）：
#    WeightedBehaviorReward 加 hepo_scales=[0.1, 0.5]
#    L0 prefix match → ×0.1，L0L1 → ×0.5，full match → ×1.0
#    使 reward 梯度反映 SID 层级语义，不同层的匹配质量分开处理
#
# B. A2PO（Asymmetric Advantage Policy Optimization）：
#    对 negative-advantage candidates 按 SID prefix overlap with best 放大惩罚
#    gate = shared_prefix_length / n_layers，neg_adv *= (1 + alpha * gate)
#    语义上与 best 相近但 reward 低的候选（hard negatives）受到更强惩罚
#
# C. NLL 正则化：
#    在 GRPO loss 上加 -nll_reg * log p_policy(best)
#    防止 reward hacking：不仅鼓励 best candidate 的概率相对提升，
#    也直接推高它的绝对概率，防止 RL 收缩到 degenerate 解
#
# 基线：EXP-028 最佳 config（w003-r100，WeightedBehaviorReward，ECPO δ=0.1，lr=1e-4）
# 增量对照：A+B+C 联合 vs EXP-028 baseline
#
# 关键超参：
#   --a2po --a2po_alpha 1.0  (B)
#   --nll_reg 0.01           (C，轻权重，避免主导 loss)
#   hepo_scales 通过 reward.py 传参，exp-030 专用 --hepo_scales "0.1,0.5" (A)

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
echo "EXP-030: A2PO + NLL Reg + HEPO Prefix Scoring"
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
if [ ! -f "${CKPT_DIR}/exp030-smoke/probe.pt" ]; then
    echo ">>> Smoke test (2 steps, G=16)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${CKPT_DIR}/exp030-smoke" \
        --name exp030-smoke \
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
        --a2po --a2po_alpha 1.0 \
        --nll_reg 0.01 \
        --hepo_scales "0.1,0.5" \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${CKPT_DIR}/exp030-smoke"
    echo ""
fi

# ── Config A: A+B+C all-in ────────────────────────────────────
NAME="exp030-a2po-nll-hepo-w003-r100"
OUTPUT="${CKPT_DIR}/${NAME}"

if [ -f "${OUTPUT}/probe.pt" ]; then
    echo "  [${NAME}] Already exists, skipping."
else
    echo ">>> Training: ${NAME}"
    echo "    A2PO α=1.0, NLL reg=0.01, HEPO scales=0.1/0.5, ECPO δ=0.1, lr=1e-4"

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
        --a2po --a2po_alpha 1.0 \
        --nll_reg 0.01 \
        --hepo_scales "0.1,0.5"

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
        git commit -m "EXP-030: ${NAME} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/exp030-git.lock
fi

# ── Config B: A2PO only (ablation) ───────────────────────────
NAME_B="exp030-a2po-only-w003-r100"
OUTPUT_B="${CKPT_DIR}/${NAME_B}"

if [ -f "${OUTPUT_B}/probe.pt" ]; then
    echo "  [${NAME_B}] Already exists, skipping."
else
    echo ">>> Training: ${NAME_B}"
    echo "    A2PO α=1.0 only (ablation: no NLL reg, no HEPO), ECPO δ=0.1, lr=1e-4"

    torchrun --nproc_per_node="${N_GPUS}" run.py grpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT_B}" \
        --name "${NAME_B}" \
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
        --a2po --a2po_alpha 1.0

    echo "  [${NAME_B}] Training complete"

    echo "  [${NAME_B}] Running full eval (n_recall=1000)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT_B}" \
        --n_recall 1000

    echo "  [${NAME_B}] Eval complete"
    echo ""

    (
        flock -x 200
        git add experiments/
        git commit -m "EXP-030: ${NAME_B} results" || echo "Nothing to commit"
        ./push.sh
    ) 200>/tmp/exp030-git.lock
fi

echo ""
echo ">>> Final results:"
for N in "${NAME}" "${NAME_B}"; do
    python -c "
import json, sys
try:
    m = json.load(open('${CKPT_DIR}/' + sys.argv[1] + '/train_meta.json'))
    e = m.get('eval', {})
    print(f'  {sys.argv[1]}: R@10={e.get(\"item_recall@10\",\"?\"):.3f}  R@500={e.get(\"item_recall@500\",\"?\"):.3f}')
except Exception as ex:
    print(f'  {sys.argv[1]}: eval not available ({ex})')
" "${N}" 2>/dev/null || echo "  ${N}: eval not available"
done

git add experiments/
git commit -m "EXP-030 complete: A2PO+NLL+HEPO results" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-030 complete!"
