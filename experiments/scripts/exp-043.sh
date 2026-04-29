#!/bin/bash
set -euo pipefail

# EXP-043: Embedding Model Size Comparison — 0.6B vs 4B vs 8B SID Tokenizer
# Date: 2026-04-29
#
# 目标: 对比三种 embedding model 生成的 SID tokenizer 对 NTP 性能的影响
# 两组模型 × 三个 SID = 6 configs:
#
#   S-tier (256d 6L 8E, ~17.5M active):
#     Config S1: Qwen3-0.6B SID — exp043-s-0.6b
#     Config S2: Qwen3-4B  SID — exp043-s-4b
#     Config S3: Qwen3-8B  SID — exp043-s-8b
#
#   M-tier (512d 8L 8E, ~71.6M active):
#     Config M1: Qwen3-0.6B SID — exp043-m-0.6b
#     Config M2: Qwen3-4B  SID — exp043-m-4b
#     Config M3: Qwen3-8B  SID — exp043-m-8b
#
# 固定条件: 14d 数据, full features (time_gap+action_level+segment_emb)
# NTP 数据三个 SID 各一份，S/M 两个模型共用
# 对标: exp036-full-features (exp013 SID, 旧 0.6B) R@500=59.0%
#       exp042-m-sft          (exp013 SID, M-tier)  R@500=TBD

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
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
echo "EXP-043: Embedding Model Size Comparison"
echo "=========================================="
echo "  GPUs:      ${N_GPUS}"
echo "  Dates:     ${DATE_START} ~ ${DATE_END}"
echo "  Features:  time_gap + action_level + segment_emb"
echo ""

# Sanity checks
for model in 0.6b 4b 8b; do
    cache="experiments/sid_cache/exp026-${model}-14d"
    if [ ! -f "${cache}/config.json" ]; then
        echo "ERROR: SID cache not found at ${cache}"
        echo "Run: bash experiments/scripts/download_s3_data.sh sid-${model}"
        exit 1
    fi
done

# ── Smoke test ────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" == false ] && [ "${START_FROM}" -le 1 ]; then
    echo ">>> Smoke test (0.6b SID, dry run)..."
    SMOKE_OUT="${CKPT_DIR}/exp043-smoke"
    # preprocess smoke (1 shard, 100 seqs)
    python run.py preprocess-ntp \
        --sid_cache "experiments/sid_cache/exp026-0.6b-14d" \
        --output_dir "${SMOKE_OUT}-data" \
        --n_shards 1 \
        --date_start "${DATE_START}" \
        --date_end "${DATE_END}" \
        --behavior_path "${BEHAVIOR_CACHE}" \
        --shift_features \
        --max_seqs 200
    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${SMOKE_OUT}-data" \
        --output_dir "${SMOKE_OUT}" \
        --name exp043-smoke \
        --model s-tier \
        --use_time_gap \
        --use_action_level \
        --use_segment_emb \
        --dry_run
    echo "  Smoke test PASSED"
    rm -rf "${SMOKE_OUT}" "${SMOKE_OUT}-data"
    echo ""
fi

# M-tier model config
M_EMBED=512
M_HEADS=8
M_LAYERS=8
M_EXPERTS=8
M_EXPERT_DIM=2048

# ── Helper: preprocess NTP data for one SID ──────────────────
preprocess_sid() {
    local MODEL_KEY=$1    # 0.6b / 4b / 8b
    local SID_CACHE="experiments/sid_cache/exp026-${MODEL_KEY}-14d"
    local NTP_DATA="experiments/ntp_data/exp043-${MODEL_KEY}-14d"

    if [ ! -f "${NTP_DATA}/meta.json" ] || [ "${FORCE}" == true ]; then
        echo ">>> Preprocessing NTP data (${MODEL_KEY} SID)..."
        torchrun --nproc_per_node="${N_GPUS}" run.py preprocess-ntp \
            --sid_cache "${SID_CACHE}" \
            --output_dir "${NTP_DATA}" \
            --n_shards "${N_GPUS}" \
            --date_start "${DATE_START}" \
            --date_end "${DATE_END}" \
            --behavior_path "${BEHAVIOR_CACHE}" \
            --shift_features
    else
        echo "  [data] ${NTP_DATA} already exists, skipping preprocess."
    fi
    python3 -c "
import json
m = json.load(open('${NTP_DATA}/meta.json'))
print(f'  n_seqs={m[\"n_seqs\"]:,}  n_eval_items={m[\"n_eval_items\"]:,}')
" 2>/dev/null || true
}

# ── Helper: train + eval one config ──────────────────────────
run_config() {
    local TIER=$1         # s / m
    local MODEL_KEY=$2    # 0.6b / 4b / 8b
    local CONFIG_NUM=$3

    local NTP_DATA="experiments/ntp_data/exp043-${MODEL_KEY}-14d"
    local NAME="exp043-${TIER}-${MODEL_KEY}"
    local OUTPUT="${CKPT_DIR}/${NAME}"

    echo ""
    echo "============================================================"
    echo "Config ${CONFIG_NUM}: ${TIER^^}-tier × Qwen3-${MODEL_KEY^^} SID"
    echo "============================================================"

    # Train
    T0=$(date +%s)
    if [ -f "${OUTPUT}/train_meta.json" ] && [ "${FORCE}" != true ]; then
        echo "  [train] Checkpoint found, skipping."
    else
        echo ">>> Training ${NAME}..."
        if [ "${TIER}" == "s" ]; then
            torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
                --preprocessed_dir "${NTP_DATA}" \
                --output_dir "${OUTPUT}" \
                --name "${NAME}" \
                --model s-tier \
                --use_time_gap \
                --use_action_level \
                --use_segment_emb
        else
            torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
                --preprocessed_dir "${NTP_DATA}" \
                --output_dir "${OUTPUT}" \
                --name "${NAME}" \
                --model s-tier \
                --embed_dim "${M_EMBED}" \
                --n_heads "${M_HEADS}" \
                --n_transformer_layers "${M_LAYERS}" \
                --n_experts "${M_EXPERTS}" \
                --expert_dim "${M_EXPERT_DIM}" \
                --use_time_gap \
                --use_action_level \
                --use_segment_emb
        fi
    fi
    T1=$(date +%s)
    echo "  Training complete  ($(( (T1 - T0) / 60 ))min)"

    # Full eval
    echo ">>> Full eval (n_recall=1000)..."
    T2=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT}" \
        --n_recall 1000
    T3=$(date +%s)
    echo "  Eval complete  ($(( (T3 - T2) / 60 ))min)  total=$(( (T3 - T0) / 60 ))min"

    git add experiments/
    git commit -m "EXP-043 ${NAME}: ${TIER^^}-tier NTP with Qwen3-${MODEL_KEY} SID" || echo "Nothing to commit"
    ./push.sh
}

# ── Step 1: Preprocess NTP data (3 SID caches, shared by S+M) ─
echo ">>> Preprocessing NTP data for all 3 SID caches..."
[ "${START_FROM}" -le 1 ] && preprocess_sid "0.6b"
[ "${START_FROM}" -le 1 ] && preprocess_sid "4b"
[ "${START_FROM}" -le 1 ] && preprocess_sid "8b"

# ── Step 2: S-tier configs ────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "S-tier × 3 SID"
echo "════════════════════════════════════════════════════════════════"
[ "${START_FROM}" -le 1 ] && run_config "s" "0.6b" "S1"
[ "${START_FROM}" -le 2 ] && run_config "s" "4b"   "S2"
[ "${START_FROM}" -le 3 ] && run_config "s" "8b"   "S3"

# ── Step 3: M-tier configs ────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "M-tier × 3 SID"
echo "════════════════════════════════════════════════════════════════"
[ "${START_FROM}" -le 4 ] && run_config "m" "0.6b" "M1"
[ "${START_FROM}" -le 5 ] && run_config "m" "4b"   "M2"
[ "${START_FROM}" -le 6 ] && run_config "m" "8b"   "M3"

# ── Summary ───────────────────────────────────────────────────
echo ""
echo ">>> EXP-043 Results Summary:"
python3 -c "
import json, os

configs = [
    ('exp036-full-features',  'S-tier, exp013 SID (old 0.6B)'),
    ('exp042-m-sft',          'M-tier, exp013 SID (old 0.6B)'),
    ('exp043-s-0.6b',         'S-tier, Qwen3-0.6B SID'),
    ('exp043-s-4b',           'S-tier, Qwen3-4B  SID'),
    ('exp043-s-8b',           'S-tier, Qwen3-8B  SID'),
    ('exp043-m-0.6b',         'M-tier, Qwen3-0.6B SID'),
    ('exp043-m-4b',           'M-tier, Qwen3-4B  SID'),
    ('exp043-m-8b',           'M-tier, Qwen3-8B  SID'),
]

# also show SID cache stats
sid_configs = [
    ('experiments/sid_cache/exp013-4096x3-12d-binary/config.json', 'exp013'),
    ('experiments/sid_cache/exp026-0.6b-14d/config.json',          'exp026-0.6b'),
    ('experiments/sid_cache/exp026-4b-14d/config.json',            'exp026-4b'),
    ('experiments/sid_cache/exp026-8b-14d/config.json',            'exp026-8b'),
]
print(f'  SID Cache Stats:')
print(f'  {\"Cache\":<18}  {\"Emb Model\":<14}  {\"Dim\":>5}  {\"N Items\":>9}  {\"Collision\":>10}')
print(f'  {\"-\"*18}  {\"-\"*14}  {\"-\"*5}  {\"-\"*9}  {\"-\"*10}')
for path, label in sid_configs:
    if os.path.exists(path):
        c = json.load(open(path))
        print(f'  {label:<18}  {c.get(\"model_key\",\"?\"):<14}  {c.get(\"embedding_dim\",0):>5}  {c.get(\"n_items\",0):>9,}  {c.get(\"collision_rate\",0):>10.3%}')

print()
print(f'  {\"Config\":<25}  {\"R@10\":>6}  {\"R@500\":>7}  {\"PPL\":>7}  {\"L0 PPL\":>8}')
print(f'  {\"-\"*25}  {\"-\"*6}  {\"-\"*7}  {\"-\"*7}  {\"-\"*8}')
for name, desc in configs:
    path = f'experiments/ntp_checkpoints/{name}/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        e = m.get('eval', {})
        r10   = e.get('item_recall@10', 0)
        r500  = e.get('item_recall@500', 0)
        ppl   = e.get('ppl', 0)
        lppl  = e.get('layer_ppl', [])
        l0    = lppl[0] if lppl else '-'
        w     = m.get('train', {}).get('wall_time_s', 0)
        print(f'  {name:<25}  {r10:>6.1%}  {r500:>7.1%}  {ppl:>7.2f}  {str(l0):>8}  ({int(w)//60}min)')
    else:
        print(f'  {name:<25}: not available')
" 2>/dev/null || echo "  Results not available"

echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-043 complete: S+M tier × 0.6B/4B/8B SID comparison" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-043 complete!"
