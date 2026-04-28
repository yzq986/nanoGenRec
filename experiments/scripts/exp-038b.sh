#!/bin/bash
set -euo pipefail

# EXP-038B: RF-DPO on exp037-medium — 对齐 exp020 配置 (807 steps = ratio-matching)
# Date: 2026-04-28
#
# EXP-038 失败原因：
#   1. PREF_DIR 指向根目录，DPO 完全未激活 (n_dpo_pairs=0)
#   2. 修复后仍只有 406 步 (NTP loader 长度)，而 exp019/020 有 807 步
#
# 807 步的设计逻辑 (ratio-matching)：
#   target_dpo_batches = (4312 pairs / 16 batch) × 3 epochs = 807
#   NTP 步数也要跑到 807，维持 NTP:DPO ≈ 1:1 配比
#   exp023 数据集每 epoch 仅 406 步 → 需要 ntp_epochs=2 循环，再用 --max_steps 807 截断
#   实际: min(406×2=812, 807) = 807 步
#
# 本实验完全对齐 exp020 配置：
#   ref=exp037-medium, λ=0.03, β=0.1, difficulty=hard, ntp_epochs=3 (1218 steps)
#   每个 epoch 结束保存中间 checkpoint (ep1=406, ep2=812, ep3=1218)，对比哪个最优
#
# RL 对齐链路:
#   exp036-full-features (SFT)
#   → EXP-037 SP-DPO → exp037-medium   (ref)
#   → [本实验: RF-DPO 3 epochs]
#   → EXP-039B ECPO

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
SFT_CKPT="experiments/ntp_checkpoints/exp037-medium"
NTP_DATA="experiments/ntp_data/exp023-14d-features"
PREF_DIR="experiments/rf_dpo_data/exp018/hard"
CKPT_DIR="experiments/ntp_checkpoints"

FORCE=false
SKIP_SMOKE=false
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --no-smoke) SKIP_SMOKE=true ;;
    esac
done

echo "=========================================="
echo "EXP-038B: RF-DPO on exp037-medium (3 epochs)"
echo "=========================================="
echo "  GPUs:       ${N_GPUS}"
echo "  SFT ckpt:   ${SFT_CKPT}"
echo "  Pref dir:   ${PREF_DIR}"
echo "  NTP data:   ${NTP_DATA}"
echo "  Config:     λ=0.03, β=0.1, ntp_epochs=3 (1218 steps, mid-checkpoints at ep1/ep2)"
echo ""

if [ ! -f "${SFT_CKPT}/train_meta.json" ]; then
    echo "ERROR: exp037-medium not found at ${SFT_CKPT}"
    exit 1
fi
if [ ! -f "${PREF_DIR}/meta.json" ]; then
    echo "ERROR: preference pairs not found at ${PREF_DIR}"
    exit 1
fi

mkdir -p "${CKPT_DIR}"

# ── Smoke test ─────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" == false ]; then
    SMOKE_OUT="${CKPT_DIR}/exp038b-smoke"
    echo ">>> Smoke test (1 step)..."
    python run.py sp-dpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preference_dir "${PREF_DIR}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_OUT}" \
        --dpo_weight 0.03 \
        --dpo_beta 0.1 \
        --lr 1e-4 \
        --ntp_epochs 1 \
        --max_steps 1 \
        --difficulty hard \
        --name exp038b-smoke
    echo "  Smoke test PASSED"
    rm -rf "${SMOKE_OUT}"
    echo ""
fi

# ── Main training: RF-DPO λ=0.03, 3 epochs (~807 steps) ───────
NAME="exp038b-hard-lam03-3ep"
OUTPUT="${CKPT_DIR}/${NAME}"

T0=$(date +%s)
if [ -f "${OUTPUT}/probe.pt" ] && [ "${FORCE}" != true ]; then
    echo "  [${NAME}] Checkpoint found, skipping (use --force to retrain)."
else
    echo ">>> Training: ${NAME}"
    echo "    ref=exp037-medium, RF-DPO hard, λ=0.03, β=0.1, ntp_epochs=2, max_steps=807"
    echo ""
    torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-train \
        --sft_checkpoint "${SFT_CKPT}" \
        --preference_dir "${PREF_DIR}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT}" \
        --dpo_weight 0.03 \
        --dpo_beta 0.1 \
        --lr 1e-4 \
        --difficulty hard \
        --ntp_epochs 3 \
        --name "${NAME}" \
        --wandb
fi
T1=$(date +%s)
TRAIN_MIN=$(( (T1 - T0) / 60 ))
echo "  Training complete  (${TRAIN_MIN}min)"

# ── Full eval ────────────────────────────────────────────────
echo ">>> Full eval (n_recall=1000)..."
T2=$(date +%s)
torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
    --checkpoint "${OUTPUT}" \
    --n_recall 1000
T3=$(date +%s)
EVAL_MIN=$(( (T3 - T2) / 60 ))
TOTAL_MIN=$(( (T3 - T0) / 60 ))
echo "  Total: train=${TRAIN_MIN}min  eval=${EVAL_MIN}min  total=${TOTAL_MIN}min"

echo ""
echo ">>> Results summary:"
python3 -c "
import json, os
checkpoints = [
    ('exp037-medium',           'SP-DPO ref'),
    ('exp038-hard-lam03',       'RF-DPO 1ep (failed)'),
    ('${NAME}',                 'RF-DPO 3ep (this)'),
    ('exp020-hard-lam03',       'SOTA no-features'),
]
for name, desc in checkpoints:
    path = f'experiments/ntp_checkpoints/{name}/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        e = m.get('eval', {})
        print(f'  {name:<35} ({desc:<22}): R@10={e.get(\"item_recall@10\",0):.1%}  R@500={e.get(\"item_recall@500\",0):.1%}  PPL={e.get(\"ppl\",0):.2f}')
" 2>/dev/null || true

echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-038B complete: RF-DPO 3ep features R@500=$(python3 -c "
import json
m=json.load(open('${OUTPUT}/train_meta.json'))
print(f\"{m['eval']['item_recall@500']:.1%}\")
" 2>/dev/null || echo 'TBD')" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-038B complete!"
echo "Next: EXP-039B ECPO from exp038b-hard-lam03-3ep"
