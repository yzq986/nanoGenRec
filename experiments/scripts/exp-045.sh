#!/bin/bash
set -euo pipefail

# EXP-045: FSQ Hidden Dim Fix — 4B and 8B Embedding Models
# Date: 2026-04-29
#
# 问题: exp026 的 fsq_mlp_hidden=64 是为 0.6B (dim=1024) 设计的。
# 对于 4B (dim=2560) 和 8B (dim=4096)，MLP bottleneck 太小，导致 L2 entropy 坍缩，
# 大量 item 被映射到相同 SID (collision rate 高)。
#
# exp026 collision 对比:
#   0.6b: embedding_dim=1024, h=64 → collision=0.49%   (可接受)
#   4b:   embedding_dim=2560, h=64 → collision=2.76%   (过高)
#   8b:   embedding_dim=4096, h=64 → collision=5.44%   (严重)
#
# 修复方案: 将 h 设为 embedding_dim/8:
#   4b: h=256  (2560/8 ≈ 320, round down to 256 for clean power of 2)
#   8b: h=512  (4096/8 = 512)
#
# 每个 model 跑 S-tier NTP（与 exp043 相同模型大小，方便对比）
# 0.6b-h64 结果直接引用 exp043-s-0.6b，不重跑
#
# 预期 configs:
#   exp045-4b-h256:  Qwen3-4B SID h=256   → NTP S-tier
#   exp045-8b-h512:  Qwen3-8B SID h=512   → NTP S-tier
#
# 对标:
#   exp043-s-4b: 4B SID h=64,  R@500=?
#   exp043-s-8b: 8B SID h=64,  R@500=?
#   exp043-s-0.6b: 0.6B SID baseline

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
CKPT_DIR="experiments/ntp_checkpoints"
BEHAVIOR_CACHE="/mnt/workspace/gr-demo-behavior-cache"
DATE_START="2026-03-18"
DATE_END="2026-03-31"
EMB_CACHE_ROOT="${HOME}/gr_demo_cache/embedding_cache"

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
echo "EXP-045: FSQ Hidden Dim Fix (4B h=256, 8B h=512)"
echo "=========================================="
echo "  GPUs:        ${N_GPUS}"
echo "  Dates:       ${DATE_START} ~ ${DATE_END}"
echo "  EMB cache:   ${EMB_CACHE_ROOT}"
echo ""

# Sanity checks: embedding caches must exist
for model in 4b 8b; do
    cache="${EMB_CACHE_ROOT}/qwen3-${model}"
    if [ ! -f "${cache}/content_ids.npy" ]; then
        echo "ERROR: Embedding cache not found at ${cache}/content_ids.npy"
        echo "Run: bash experiments/scripts/download_s3_data.sh emb-cache-${model}"
        exit 1
    fi
done

# ── Config table ──────────────────────────────────────────────
# model_key  fsq_hidden  sid_cache_name
CONFIGS=(
    "qwen3-4b  256  exp045-4b-h256"
    "qwen3-8b  512  exp045-8b-h512"
)

# ── Smoke test ────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" == false ] && [ "${START_FROM}" -le 1 ]; then
    echo ">>> Smoke test (4b SID dry run)..."
    SMOKE_SID="${REPO_ROOT}/experiments/sid_cache/exp045-smoke-4b"
    SMOKE_NTP="${CKPT_DIR}/exp045-smoke"

    python run.py preprocess-sid \
        --model qwen3-4b \
        --output_dir "${SMOKE_SID}" \
        --fsq_mlp_hidden 256 \
        --fsq_levels 12d_4096 \
        --fsq_projection mlp \
        --max_items 2000

    torchrun --nproc_per_node="${N_GPUS}" run.py preprocess-ntp \
        --sid_cache "${SMOKE_SID}" \
        --output_dir "${SMOKE_NTP}-data" \
        --n_shards 1 \
        --date_start "${DATE_START}" \
        --date_end "${DATE_END}" \
        --behavior_path "${BEHAVIOR_CACHE}" \
        --shift_features \
        --max_seqs 200

    torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
        --preprocessed_dir "${SMOKE_NTP}-data" \
        --output_dir "${SMOKE_NTP}" \
        --name exp045-smoke \
        --model s-tier \
        --use_time_gap \
        --use_action_level \
        --use_segment_emb \
        --dry_run

    echo "  Smoke test PASSED"
    rm -rf "${SMOKE_SID}" "${SMOKE_NTP}" "${SMOKE_NTP}-data"
    echo ""
fi

# ── Helper: train FSQ SID for one config ─────────────────────
preprocess_sid() {
    local MODEL_KEY=$1     # qwen3-4b / qwen3-8b
    local FSQ_HIDDEN=$2
    local SID_NAME=$3
    local SID_CACHE="experiments/sid_cache/${SID_NAME}"

    if [ -f "${SID_CACHE}/config.json" ] && [ "${FORCE}" != true ]; then
        echo "  [sid] ${SID_CACHE} already exists, skipping."
    else
        echo ">>> Training FSQ SID (model=${MODEL_KEY}, h=${FSQ_HIDDEN})..."
        python run.py preprocess-sid \
            --model "${MODEL_KEY}" \
            --output_dir "${SID_CACHE}" \
            --fsq_mlp_hidden "${FSQ_HIDDEN}" \
            --fsq_levels 12d_4096 \
            --fsq_projection mlp
    fi

    python3 -c "
import json
cfg = json.load(open('${SID_CACHE}/config.json'))
print(f'  model={cfg[\"model_key\"]}  dim={cfg[\"embedding_dim\"]}  h={cfg[\"fsq_mlp_hidden\"]}  n_items={cfg[\"n_items\"]:,}  collision={cfg[\"collision_rate\"]:.3%}')
" 2>/dev/null || true
}

# ── Helper: preprocess NTP for one SID ───────────────────────
preprocess_ntp() {
    local SID_NAME=$1
    local SID_CACHE="experiments/sid_cache/${SID_NAME}"
    local NTP_DATA="experiments/ntp_data/${SID_NAME}"

    if [ -f "${NTP_DATA}/meta.json" ] && [ "${FORCE}" != true ]; then
        echo "  [ntp-data] ${NTP_DATA} already exists, skipping."
    else
        echo ">>> Preprocessing NTP data (${SID_NAME})..."
        torchrun --nproc_per_node="${N_GPUS}" run.py preprocess-ntp \
            --sid_cache "${SID_CACHE}" \
            --output_dir "${NTP_DATA}" \
            --n_shards "${N_GPUS}" \
            --date_start "${DATE_START}" \
            --date_end "${DATE_END}" \
            --behavior_path "${BEHAVIOR_CACHE}" \
            --shift_features
    fi

    python3 -c "
import json
m = json.load(open('${NTP_DATA}/meta.json'))
print(f'  n_seqs={m[\"n_seqs\"]:,}  n_eval_items={m[\"n_eval_items\"]:,}')
" 2>/dev/null || true
}

# ── Helper: train + eval NTP for one config ───────────────────
run_ntp() {
    local SID_NAME=$1
    local CONFIG_NUM=$2
    local NTP_DATA="experiments/ntp_data/${SID_NAME}"
    local NAME="exp045-${SID_NAME#exp045-}"
    local OUTPUT="${CKPT_DIR}/${NAME}"

    echo ""
    echo "============================================================"
    echo "Config ${CONFIG_NUM}: NTP S-tier × ${SID_NAME}"
    echo "============================================================"

    T0=$(date +%s)
    if [ -f "${OUTPUT}/train_meta.json" ] && [ "${FORCE}" != true ]; then
        echo "  [train] Checkpoint found, skipping."
    else
        echo ">>> Training ${NAME}..."
        torchrun --nproc_per_node="${N_GPUS}" run.py train-ntp \
            --preprocessed_dir "${NTP_DATA}" \
            --output_dir "${OUTPUT}" \
            --name "${NAME}" \
            --model s-tier \
            --use_time_gap \
            --use_action_level \
            --use_segment_emb
    fi
    T1=$(date +%s)
    echo "  Training complete  ($(( (T1 - T0) / 60 ))min)"

    echo ">>> Full eval (n_recall=1000)..."
    T2=$(date +%s)
    torchrun --nproc_per_node="${N_GPUS}" run.py eval-ntp \
        --checkpoint "${OUTPUT}" \
        --n_recall 1000
    T3=$(date +%s)
    echo "  Eval complete  ($(( (T3 - T2) / 60 ))min)  total=$(( (T3 - T0) / 60 ))min"

    git add experiments/
    git commit -m "EXP-045 ${NAME}: S-tier NTP with ${SID_NAME} FSQ" || echo "Nothing to commit"
    ./push.sh
}

# ── Step 1: Train FSQ SID for 4b and 8b ──────────────────────
if [ "${START_FROM}" -le 1 ]; then
    echo ">>> Step 1: Train FSQ SID caches..."
    echo ""
    for cfg in "${CONFIGS[@]}"; do
        read -r model_key fsq_hidden sid_name <<< "$cfg"
        preprocess_sid "${model_key}" "${fsq_hidden}" "${sid_name}"
        echo ""
    done
fi

# ── Step 2: Preprocess NTP data ───────────────────────────────
if [ "${START_FROM}" -le 2 ]; then
    echo ">>> Step 2: Preprocess NTP data..."
    echo ""
    for cfg in "${CONFIGS[@]}"; do
        read -r model_key fsq_hidden sid_name <<< "$cfg"
        preprocess_ntp "${sid_name}"
        echo ""
    done
fi

# ── Step 3: Train + Eval NTP ──────────────────────────────────
echo ">>> Step 3: Train + Eval NTP..."
CONFIG_I=1
for cfg in "${CONFIGS[@]}"; do
    read -r model_key fsq_hidden sid_name <<< "$cfg"
    SID_SHORT="${sid_name#exp045-}"
    [ "${START_FROM}" -le $((CONFIG_I + 2)) ] && run_ntp "${sid_name}" "${CONFIG_I}"
    CONFIG_I=$((CONFIG_I + 1))
done

# ── Summary ───────────────────────────────────────────────────
echo ""
echo ">>> EXP-045 Results Summary:"
python3 -c "
import json, os

print('  === FSQ SID Cache Stats ===')
sid_configs = [
    ('experiments/sid_cache/exp026-0.6b-14d/config.json', 'exp026-0.6b (h=64,  baseline)'),
    ('experiments/sid_cache/exp026-4b-14d/config.json',   'exp026-4b   (h=64,  old)'),
    ('experiments/sid_cache/exp026-8b-14d/config.json',   'exp026-8b   (h=64,  old)'),
    ('experiments/sid_cache/exp045-4b-h256/config.json',  'exp045-4b   (h=256, new)'),
    ('experiments/sid_cache/exp045-8b-h512/config.json',  'exp045-8b   (h=512, new)'),
]
print(f'  {\"Cache\":<35}  {\"Dim\":>5}  {\"h\":>5}  {\"N Items\":>9}  {\"Collision\":>10}  {\"Time\":>6}')
print(f'  {\"-\"*35}  {\"-\"*5}  {\"-\"*5}  {\"-\"*9}  {\"-\"*10}  {\"-\"*6}')
for path, label in sid_configs:
    if os.path.exists(path):
        c = json.load(open(path))
        t = int(c.get('train_time_seconds', 0))
        print(f'  {label:<35}  {c.get(\"embedding_dim\",0):>5}  {c.get(\"fsq_mlp_hidden\",0):>5}  {c.get(\"n_items\",0):>9,}  {c.get(\"collision_rate\",0):>10.3%}  {t//60:>5}m')

print()
print('  === NTP Results ===')
ntp_configs = [
    ('exp043-s-0.6b', '0.6B SID h=64  (ref)'),
    ('exp043-s-4b',   '4B  SID h=64  (old)'),
    ('exp043-s-8b',   '8B  SID h=64  (old)'),
    ('exp045-4b-h256','4B  SID h=256 (new)'),
    ('exp045-8b-h512','8B  SID h=512 (new)'),
]
print(f'  {\"Config\":<25}  {\"R@10\":>6}  {\"R@500\":>7}  {\"PPL\":>7}')
print(f'  {\"-\"*25}  {\"-\"*6}  {\"-\"*7}  {\"-\"*7}')
for name, desc in ntp_configs:
    path = f'experiments/ntp_checkpoints/{name}/train_meta.json'
    if os.path.exists(path):
        m = json.load(open(path))
        e = m.get('eval', {})
        r10   = e.get('item_recall@10', 0)
        r500  = e.get('item_recall@500', 0)
        ppl   = e.get('ppl', 0)
        print(f'  {name:<25}  {r10:>6.1%}  {r500:>7.1%}  {ppl:>7.2f}  # {desc}')
    else:
        print(f'  {name:<25}: not available  # {desc}')
" 2>/dev/null || echo "  Results not available"

echo ""
echo ">>> Committing final results..."
git add experiments/
git commit -m "EXP-045 complete: FSQ h=256(4b) h=512(8b) vs baseline h=64" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-045 complete!"
