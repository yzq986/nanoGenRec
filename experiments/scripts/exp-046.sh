#!/bin/bash
set -euo pipefail

# EXP-046: GateAttention 变体实验
# Date: 2026-04-29
#
# 目标: 测试 sigmoid gate on attention output 是否对推荐序列建模有收益。
#
# GateAttention 设计:
#   attn_out = attn_out * sigmoid(W_g * x_norm)
#   W_g ∈ R^{D×D}，per-position sigmoid gate，抑制噪声 token 的注意力影响。
#   新增参数: n_transformer_layers * D^2 ≈ 6 * 256^2 = 393K params（约 +1%）
#
# Configs:
#   A (baseline): 引用 exp043-s-0.6b，不重训
#   B: +gate_attn，其他参数完全相同（same data, same hyperparams）
#
# 复用 exp043 NTP 数据（无需重新 preprocess）
#
# 对照标准:
#   baseline (exp043-s-0.6b): PPL=?, R@10=?, R@500=?（全量 eval 补充）
#   判断: R@500 提升 > 0.5pp 视为有效

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
SID_CACHE="experiments/sid_cache/exp026-0.6b-14d"
NTP_DATA="experiments/ntp_data/exp043-0.6b-14d"
DATE_START="2026-03-18"
DATE_END="2026-03-31"

echo "=========================================="
echo "EXP-046: GateAttention"
echo "=========================================="
echo "  N_GPUS: ${N_GPUS}"
echo "  NTP data: ${NTP_DATA}"
echo ""

# ── Smoke test ────────────────────────────────────────────────
if [ "${1:-}" != "--no-smoke" ]; then
    echo "[Smoke test]"
    SMOKE_OUT="/tmp/exp046-smoke"
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_OUT}" \
        --name exp046-smoke \
        --model s-tier \
        --use_segment_emb \
        --use_gate_attn \
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

    if [ -f "${OUTPUT}/train_meta.json" ]; then
        echo "  Already exists, skipping training."
    else
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${OUTPUT}" \
            --name "${NAME}" \
            --model s-tier \
            ${EXTRA_FLAGS}
    fi

    echo ""
    echo "  [Eval] ${NAME}"
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT}" \
        --n_recall 1000
}

# ── Config A: baseline ────────────────────────────────────────
echo ""
echo "============================================================"
echo "[Config A] Baseline: exp043-s-0.6b (abs pos + time_gap + action + seg)"
echo "============================================================"
echo "  参考 EXP-043 结果，不重训。"
torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
    --checkpoint "${CKPT_DIR}/exp043-s-0.6b" \
    --n_recall 1000

# ── Config B: +gate_attn ─────────────────────────────────────
train_eval \
    "exp046-gate-attn" \
    "+gate_attn (same as baseline + sigmoid gate on attn output)" \
    "--use_segment_emb --use_gate_attn"

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "EXP-046 Summary"
echo "============================================================"
python3 - <<'PYEOF'
import json, os, sys

CKPT_DIR = "experiments/ntp_checkpoints"
configs = [
    ('exp043-s-0.6b',   'Baseline (abs pos + time_gap + action + seg)'),
    ('exp046-gate-attn', '+GateAttention'),
]

print(f'  {"Config":<32}  {"R@10":>6}  {"R@500":>7}  {"PPL":>7}')
print(f'  {"-"*32}  {"-"*6}  {"-"*7}  {"-"*7}')
for name, desc in configs:
    meta_path = os.path.join(CKPT_DIR, name, 'train_meta.json')
    if not os.path.exists(meta_path):
        print(f'  {desc:<32}  {"N/A":>6}  {"N/A":>7}  {"N/A":>7}')
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    r10  = meta.get('item_recall@10',  meta.get('item_recall_10',  0)) * 100
    r500 = meta.get('item_recall@500', meta.get('item_recall_500', 0)) * 100
    ppl  = meta.get('eval_ppl', 0)
    print(f'  {desc:<32}  {r10:>5.1f}%  {r500:>6.1f}%  {ppl:>7.2f}')
PYEOF

echo ""
echo "EXP-046 complete!"
