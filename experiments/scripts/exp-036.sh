#!/bin/bash
set -euo pipefail

# EXP-036: Clean Features NTP — From-Scratch Training with time_gap + action_level
# Date: 2026-04-28
#
# 目标：干净对照实验
#   Config A: 无 features（复现 exp020，验证数据集一致性）
#   Config B: time_gap + action_level + segment（从头训练，与 A 唯一差异是 features）
#
# exp025 不是干净对照：beam-passes SFT + 不同数据集
# 本实验用相同数据（exp023-14d-features）、相同超参，唯一变量是 features on/off

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
NTP_DATA="experiments/ntp_data/exp023-14d-features"

echo "=========================================="
echo "EXP-036: Clean Features NTP (from scratch)"
echo "=========================================="
echo "  GPUs:      ${N_GPUS}"
echo "  SID cache: ${SID_CACHE}"
echo "  NTP data:  ${NTP_DATA}"
echo "  Config A:  no features (exp020 reproduction)"
echo "  Config B:  time_gap + action_level + segment_emb"
echo ""

# Sanity checks
if [ ! -f "${SID_CACHE}/meta.json" ] && [ ! -d "${SID_CACHE}" ]; then
    echo "ERROR: SID cache not found at ${SID_CACHE}"
    exit 1
fi
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo "ERROR: NTP data not found at ${NTP_DATA}"
    exit 1
fi

# ── Smoke test ────────────────────────────────────────────────
SMOKE_DIR="${CKPT_DIR}/exp036-smoke"
if [ ! -f "${SMOKE_DIR}/train_meta.json" ]; then
    echo ">>> Smoke test (Config B, 2 steps)..."
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${SMOKE_DIR}" \
        --name exp036-smoke \
        --model s-tier \
        --use_segment_emb \
        --use_time_gap \
        --use_action_level \
        --max_seq_len 512 \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${SMOKE_DIR}"
    echo ""
fi

# ── Config A: no features ─────────────────────────────────────
NAME_A="exp036-no-features"
OUTPUT_A="${CKPT_DIR}/${NAME_A}"

if [ -f "${OUTPUT_A}/train_meta.json" ]; then
    echo "  [${NAME_A}] Already exists, skipping."
else
    echo ">>> Config A: no features (exp020 reproduction on exp023 data)"
    T0=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT_A}" \
        --name "${NAME_A}" \
        --model s-tier \
        --max_seq_len 512
    T1=$(date +%s)
    TRAIN_MIN_A=$(( (T1 - T0) / 60 ))
    echo "  [${NAME_A}] Training complete  (${TRAIN_MIN_A}min)"

    echo ">>> Config A: full eval..."
    T2=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT_A}" \
        --n_recall 1000
    T3=$(date +%s)
    EVAL_MIN_A=$(( (T3 - T2) / 60 ))
    echo "  [${NAME_A}] Eval complete  (${EVAL_MIN_A}min)"
    echo "  [${NAME_A}] Total: train=${TRAIN_MIN_A}min  eval=${EVAL_MIN_A}min"

    git add experiments/
    git commit -m "EXP-036 Config A complete: no-features baseline" || echo "Nothing to commit"
    ./push.sh
fi

# ── Config B: full features ───────────────────────────────────
NAME_B="exp036-full-features"
OUTPUT_B="${CKPT_DIR}/${NAME_B}"

if [ -f "${OUTPUT_B}/train_meta.json" ]; then
    echo "  [${NAME_B}] Already exists, skipping."
else
    echo ">>> Config B: full features (segment + time_gap + action_level)"
    T0=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT_B}" \
        --name "${NAME_B}" \
        --model s-tier \
        --use_segment_emb \
        --use_time_gap \
        --use_action_level \
        --max_seq_len 512
    T1=$(date +%s)
    TRAIN_MIN_B=$(( (T1 - T0) / 60 ))
    echo "  [${NAME_B}] Training complete  (${TRAIN_MIN_B}min)"

    echo ">>> Config B: full eval..."
    T2=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT_B}" \
        --n_recall 1000
    T3=$(date +%s)
    EVAL_MIN_B=$(( (T3 - T2) / 60 ))
    echo "  [${NAME_B}] Eval complete  (${EVAL_MIN_B}min)"
    echo "  [${NAME_B}] Total: train=${TRAIN_MIN_B}min  eval=${EVAL_MIN_B}min"

    git add experiments/
    git commit -m "EXP-036 Config B complete: full features" || echo "Nothing to commit"
    ./push.sh
fi

# ── Results summary ───────────────────────────────────────────
echo ""
echo ">>> Results comparison:"
python3 -c "
import json, os

def read(name):
    path = f'experiments/ntp_checkpoints/{name}/train_meta.json'
    if not os.path.exists(path):
        return None
    return json.load(open(path))

a = read('${NAME_A}')
b = read('${NAME_B}')
ref_exp020 = {'r10': 0.141, 'r500': 0.662, 'label': 'exp020 (no feat, RF-DPO data)'}
ref_exp025 = {'r10': 0.104, 'r500': 0.636, 'label': 'exp025 (beam-passes SFT)'}

print(f'  {\"Model\":<40} {\"R@10\":>6} {\"R@500\":>7} {\"PPL\":>7}')
print(f'  {\"-\"*40} {\"-\"*6} {\"-\"*7} {\"-\"*7}')
print(f'  {ref_exp020[\"label\"]:<40} {ref_exp020[\"r10\"]:>6.3f} {ref_exp020[\"r500\"]:>7.3f}  {\"N/A\":>7}')
print(f'  {ref_exp025[\"label\"]:<40} {ref_exp025[\"r10\"]:>6.3f} {ref_exp025[\"r500\"]:>7.3f}  {\"N/A\":>7}')

for name, m in [('${NAME_A}', a), ('${NAME_B}', b)]:
    if m:
        e = m.get('eval', {})
        r10  = e.get('item_recall@10', float('nan'))
        r500 = e.get('item_recall@500', float('nan'))
        t = m.get('train', {})
        loss = t.get('avg_ntp_loss', float('nan'))
        print(f'  {name:<40} {r10:>6.3f} {r500:>7.3f}  {loss:>7.3f}')
    else:
        print(f'  {name:<40} {\"N/A\":>6} {\"N/A\":>7}  {\"N/A\":>7}')

if a and b:
    ea = a.get('eval', {})
    eb = b.get('eval', {})
    delta = eb.get('item_recall@500', 0) - ea.get('item_recall@500', 0)
    print()
    if delta > 0:
        print(f'  -> features 有效: R@500 +{delta:.3f} ({delta*100:.1f}pp)')
        if eb.get('item_recall@500', 0) > 0.662:
            print(f'  -> Config B 超越 exp020 (0.662)，可作为新 RL 起点！')
    else:
        print(f'  -> features 无效: R@500 {delta:.3f} ({delta*100:.1f}pp)')
        print(f'  -> 建议继续以 exp020 路线为主')
" 2>/dev/null || true

# ── Timing summary ────────────────────────────────────────────
echo ""
echo ">>> Timing summary:"
python3 -c "
import json, os
for name in ['${NAME_A}', '${NAME_B}']:
    path = f'experiments/ntp_checkpoints/{name}/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        w = m.get('train', {}).get('wall_time_s', 0)
        print(f'  {name}: train={int(w)//60}min{int(w)%60:02d}s')
" 2>/dev/null || true

git add experiments/
git commit -m "EXP-036 complete: clean features NTP comparison" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-036 complete!"
