#!/bin/bash
set -euo pipefail

# EXP-037: SP-DPO on exp036-full-features (Features 路线第二步)
# Date: 2026-04-28
#
# 完整 features 对齐链路：
#   exp036-B (NTP+feat) → [本实验: SP-DPO] → EXP-038 RF-DPO → EXP-039 ECPO
#
# 对标：EXP-017 SP-DPO（相同结构，不同 SFT 起点和数据集）
# 只做 Easy + Medium（Hard 阶段 exp017 已验证退化，跳过）
# SP-DPO pairs 必须重新生成（beam candidates 依赖模型分布）

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
SFT_CKPT="experiments/ntp_checkpoints/exp036-full-features"
NTP_DATA="experiments/ntp_data/exp023-14d-features"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"
PREF_DIR="experiments/sp_dpo_data/exp037"
CKPT_DIR="experiments/ntp_checkpoints"

SKIP_SMOKE=false
FORCE=false
START_FROM=1
for arg in "$@"; do
    case "$arg" in
        --no-smoke) SKIP_SMOKE=true ;;
        --force) FORCE=true ;;
        --start-from=*) START_FROM="${arg#*=}" ;;
    esac
done

echo "=========================================="
echo "EXP-037: SP-DPO on exp036-full-features"
echo "=========================================="
echo "  GPUs:       ${N_GPUS}"
echo "  SFT ckpt:   ${SFT_CKPT}"
echo "  NTP data:   ${NTP_DATA}"
echo "  Pref dir:   ${PREF_DIR}"
echo "  Start from: config #${START_FROM}"
echo ""

# Sanity checks
if [ ! -f "${SFT_CKPT}/train_meta.json" ]; then
    echo "ERROR: SFT checkpoint not found at ${SFT_CKPT}"
    echo "Run exp-036.sh first."
    exit 1
fi
if [ ! -f "${NTP_DATA}/meta.json" ]; then
    echo "ERROR: NTP data not found at ${NTP_DATA}"
    exit 1
fi

mkdir -p "${PREF_DIR}"

# ── Helper: generate preference pairs ────────────────────────
generate_preferences() {
    local BEAM_MODEL=$1
    local OUTPUT=$2
    local DIFFICULTY=$3
    local BEAM_SIZE=${4:-50}
    local EXTRA_ARGS=${5:-}

    if [ -f "${OUTPUT}/meta.json" ] && [ "${FORCE}" != true ]; then
        echo "  [pref/${DIFFICULTY}] Found at ${OUTPUT}, skipping."
        return 0
    fi

    echo ">>> Generating ${DIFFICULTY} pairs (beam_size=${BEAM_SIZE}${EXTRA_ARGS:+ $EXTRA_ARGS})..."
    torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-prepare \
        --sft_checkpoint "${BEAM_MODEL}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT}" \
        --beam_size "${BEAM_SIZE}" \
        --n_rejected 20 \
        --difficulty "${DIFFICULTY}" \
        ${EXTRA_ARGS}
    echo "  [pref/${DIFFICULTY}] Done → ${OUTPUT}"
}

# ── Helper: SP-DPO training ───────────────────────────────────
train_spdpo() {
    local NAME=$1
    local DIFFICULTY=$2
    local REF_CKPT=$3
    local PREF_PATH=$4
    local DESC=$5

    local OUTPUT="${CKPT_DIR}/exp037-${NAME}"

    if [ -f "${OUTPUT}/probe.pt" ] && [ "${FORCE}" != true ]; then
        echo "  [${NAME}] Already exists, skipping."
        return 0
    fi

    echo ">>> [${NAME}] ${DESC}"
    T0=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py sp-dpo-train \
        --sft_checkpoint "${REF_CKPT}" \
        --preference_dir "${PREF_PATH}" \
        --preprocessed_dir "${NTP_DATA}" \
        --output_dir "${OUTPUT}" \
        --difficulty "${DIFFICULTY}" \
        --dpo_weight 0.1 \
        --dpo_beta 0.1 \
        --lr 1e-4 \
        --batch_size 2048 \
        --name "exp037-${NAME}"
    T1=$(date +%s)
    echo "  [${NAME}] Training complete  ($(( (T1 - T0) / 60 ))min)"

    if [ ! -f "${OUTPUT}/probe.pt" ]; then
        echo "  [${NAME}] FAILED: no checkpoint saved"
        exit 1
    fi
}

# ── Phase 0: Smoke test ───────────────────────────────────────
if [ "${SKIP_SMOKE}" != true ] && [ "${START_FROM}" -le 1 ]; then
    SMOKE_PREF="${PREF_DIR}/smoke"
    SMOKE_CKPT="${CKPT_DIR}/exp037-smoke"

    if [ ! -f "${SMOKE_CKPT}/probe.pt" ]; then
        echo ">>> Smoke test (Easy, 5 steps)..."
        python run.py sp-dpo-prepare \
            --sft_checkpoint "${SFT_CKPT}" \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${SMOKE_PREF}" \
            --beam_size 10 \
            --n_rejected 5 \
            --max_samples 100 \
            --difficulty easy

        python run.py sp-dpo-train \
            --sft_checkpoint "${SFT_CKPT}" \
            --preference_dir "${SMOKE_PREF}" \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${SMOKE_CKPT}" \
            --dpo_weight 0.1 \
            --dpo_beta 0.1 \
            --lr 1e-4 \
            --batch_size 64 \
            --max_steps 5 \
            --name exp037-smoke

        if [ -f "${SMOKE_CKPT}/probe.pt" ]; then
            echo "  Smoke test PASSED"
            rm -rf "${SMOKE_CKPT}" "${SMOKE_PREF}"
        else
            echo "  Smoke test FAILED"
            exit 1
        fi
        echo ""
    fi
fi

# ── Phase 1: Easy stage ───────────────────────────────────────
# SFT beam search (all difficulties in one pass)
if [ "${START_FROM}" -le 1 ]; then
    generate_preferences "${SFT_CKPT}" "${PREF_DIR}/sft" "all"

    train_spdpo "easy" "easy" \
        "${SFT_CKPT}" "${PREF_DIR}/sft" \
        "Easy: SFT beam search, λ=0.1, β=0.1"

    echo ">>> Config Easy: full eval..."
    T0=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${CKPT_DIR}/exp037-easy" \
        --n_recall 1000
    T1=$(date +%s)
    echo "  [easy] Eval complete  ($(( (T1 - T0) / 60 ))min)"

    git add experiments/
    git commit -m "EXP-037 Easy stage complete" || echo "Nothing to commit"
    ./push.sh
fi

# ── Phase 2: Medium stage (prefix-locked, Easy-model beam) ───
# 对标 exp017 Config 2 (C2 Medium 全面最优)
if [ "${START_FROM}" -le 2 ]; then
    if [ ! -f "${CKPT_DIR}/exp037-easy/probe.pt" ]; then
        echo "ERROR: Easy checkpoint missing. Run from --start-from=1"
        exit 1
    fi

    # Easy-model prefix-locked beam search for M/H candidates
    generate_preferences "${CKPT_DIR}/exp037-easy" \
        "${PREF_DIR}/easy-pfx" "all" 50 "--prefix_locked"

    train_spdpo "medium" "medium" \
        "${CKPT_DIR}/exp037-easy" "${PREF_DIR}/easy-pfx" \
        "Medium: ref=Easy, Easy-model prefix-locked candidates"

    echo ">>> Config Medium: full eval..."
    T0=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${CKPT_DIR}/exp037-medium" \
        --n_recall 1000
    T1=$(date +%s)
    echo "  [medium] Eval complete  ($(( (T1 - T0) / 60 ))min)"

    git add experiments/
    git commit -m "EXP-037 Medium stage complete" || echo "Nothing to commit"
    ./push.sh
fi

# ── Results summary ───────────────────────────────────────────
echo ""
echo ">>> Results comparison:"
python3 -c "
import json, os

def read_eval(name):
    path = f'experiments/ntp_checkpoints/{name}/train_meta.json'
    if not os.path.exists(path):
        return None
    return json.load(open(path))

refs = [
    ('exp036-full-features (SFT起点)', 0.109, 0.590, 27.3),
    ('exp017-spdpo-easy (EXP-017 参考)', 0.125, 0.550, 28.5),
    ('exp017-fixed-medium (EXP-017 参考)', 0.154, 0.683, 14.5),
]
print(f'  {\"Model\":<42} {\"R@10\":>6} {\"R@500\":>7} {\"PPL\":>7}')
print(f'  {\"-\"*42} {\"-\"*6} {\"-\"*7} {\"-\"*7}')
for label, r10, r500, ppl in refs:
    print(f'  {label:<42} {r10:>6.3f} {r500:>7.3f} {ppl:>7.1f}')
print()
for name in ['exp037-easy', 'exp037-medium']:
    m = read_eval(name)
    if m:
        e = m.get('eval', {})
        r10  = e.get('item_recall@10', float('nan'))
        r500 = e.get('item_recall@500', float('nan'))
        ppl  = e.get('ppl', float('nan'))
        print(f'  {name:<42} {r10:>6.3f} {r500:>7.3f} {ppl:>7.1f}')
    else:
        print(f'  {name:<42} {\"N/A\":>6} {\"N/A\":>7} {\"N/A\":>7}')
" 2>/dev/null || true

# ── Timing summary ────────────────────────────────────────────
echo ""
echo ">>> Timing summary:"
python3 -c "
import json, os
for name in ['exp037-easy', 'exp037-medium']:
    path = f'experiments/ntp_checkpoints/{name}/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        w = m.get('train', {}).get('wall_time_s', 0)
        print(f'  {name}: {int(w)//60}min{int(w)%60:02d}s')
" 2>/dev/null || true

git add experiments/
git commit -m "EXP-037 complete: SP-DPO on exp036-full-features" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-037 complete!"
echo "Next: EXP-038 RF-DPO (ref=exp037-medium, Joint NTP+DPO λ=0.03)"
