#!/bin/bash
set -e

# EXP-009: QFormer Tokenizer — 冻结 Qwen3 + Cross-Attention 压缩
# Date: 2026-04-14
# IDEA: IDEA-onerec-3 — freeze Qwen3, train QFormer (cross-attention) on top
# Hardware: 8 x A100 40GB
# Prereq: model/qformer.py implemented, --use_qformer flag added to contrastive_finetune.py

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOCKFILE="/tmp/exp009_git.lock"
EXP_DIR="experiments/hyperparam/2026-04-14_exp009-qformer"
mkdir -p "$EXP_DIR"

# Options: --no-smoke to skip smoke test, then config names (A B C)
SKIP_SMOKE=false
if [[ "${1:-}" == "--no-smoke" ]]; then
    SKIP_SMOKE=true
    shift
fi
CONFIGS="${@:-A B C}"
run_config() { [[ " $CONFIGS " == *" $1 "* ]]; }
echo "Selected configs: $CONFIGS (smoke: $( $SKIP_SMOKE && echo skip || echo run ))"

commit_result() {
    local msg="$1"
    (
        flock -x 200
        git add experiments/ || true
        git commit -m "$msg" || echo "Nothing to commit"
        ./push.sh
    ) 200>"$LOCKFILE"
}

echo "=========================================="
echo "EXP-009: QFormer Tokenizer"
echo "=========================================="

# ──────────────────────────────────────────────
# Phase 0: Smoke test — 1% data, 10 steps, verify QFormer pipeline
# ──────────────────────────────────────────────
if $SKIP_SMOKE; then
    echo ">>> Phase 0: Skipped (--no-smoke)"
else
    echo ""
    echo ">>> Phase 0: Smoke test (QFormer, 1% data, 10 steps)"
    torchrun --nproc_per_node=8 \
        model/contrastive_finetune.py \
        --use_qformer \
        --qformer_layers 2 \
        --qformer_queries 4 \
        --dry_run \
        --temperature 0.05 \
        --batch_size 32 \
        --grad_accum 8 \
        --lr 1e-4 \
        --output_dir "$EXP_DIR/smoke_test"
    echo ">>> Smoke test PASSED"
    rm -rf "$EXP_DIR/smoke_test"
fi

# ──────────────────────────────────────────────
# Phase 1: QFormer training (8 GPU DDP per config, sequential)
# Freeze Qwen3, train QFormer only
# ──────────────────────────────────────────────
echo ""
echo ">>> Phase 1: QFormer contrastive training (8 GPU DDP, sequential)"

# Config A: 2-layer QFormer, M=4, lr=1e-4
if run_config A; then
echo "[Config A] Starting: 2-layer QFormer, M=4, lr=1e-4"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --use_qformer \
    --qformer_layers 2 \
    --qformer_queries 4 \
    --temperature 0.05 \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 1e-4 \
    --max_pairs 2000000 \
    --output_dir "$EXP_DIR/config_a_L2_M4_lr1e4" \
    --experiment_name "qformer_a"
echo "[Config A] Done"
commit_result "EXP-009 config A done: QFormer L=2, M=4, lr=1e-4"
fi

# Config B: 2-layer QFormer, M=4, lr=5e-4
if run_config B; then
echo "[Config B] Starting: 2-layer QFormer, M=4, lr=5e-4"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --use_qformer \
    --qformer_layers 2 \
    --qformer_queries 4 \
    --temperature 0.05 \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 5e-4 \
    --max_pairs 2000000 \
    --output_dir "$EXP_DIR/config_b_L2_M4_lr5e4" \
    --experiment_name "qformer_b"
echo "[Config B] Done"
commit_result "EXP-009 config B done: QFormer L=2, M=4, lr=5e-4"
fi

# Config C: 4-layer QFormer, M=4, lr=1e-4
if run_config C; then
echo "[Config C] Starting: 4-layer QFormer, M=4, lr=1e-4"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --use_qformer \
    --qformer_layers 4 \
    --qformer_queries 4 \
    --temperature 0.05 \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 1e-4 \
    --max_pairs 2000000 \
    --output_dir "$EXP_DIR/config_c_L4_M4_lr1e4" \
    --experiment_name "qformer_c"
echo "[Config C] Done"
commit_result "EXP-009 config C done: QFormer L=4, M=4, lr=1e-4"
fi

echo ""
echo ">>> Phase 1 complete: QFormer training done"

# Config name → directory mapping
declare -A CONFIG_DIRS=(
    [A]=config_a_L2_M4_lr1e4
    [B]=config_b_L2_M4_lr5e4
    [C]=config_c_L4_M4_lr1e4
)

# ──────────────────────────────────────────────
# Phase 2: Generate embeddings from QFormer models (parallel on GPUs)
# ──────────────────────────────────────────────
echo ""
echo ">>> Phase 2: Generate QFormer embeddings"

for key in A B C; do
    run_config "$key" || continue
    config="${CONFIG_DIRS[$key]}"
    echo "  Generating embeddings for $config ..."
    python run.py encode \
        --model_path "$EXP_DIR/$config/model" \
        --output_dir "$EXP_DIR/$config/embeddings" \
        --model_type qwen3-text \
        --use_qformer &
done
wait
echo ">>> Phase 2 complete: embeddings generated"

# ──────────────────────────────────────────────
# Phase 3: Evaluate — embedding_hit_rate + OPQ intrinsic (parallel)
# ──────────────────────────────────────────────
echo ""
echo ">>> Phase 3: Evaluate all configs"

for key in A B C; do
    run_config "$key" || continue
    config="${CONFIG_DIRS[$key]}"
    (
        echo "[$config] Evaluating..."
        python run.py hyperparam --skip_embedding \
            --quantizer opq --n_subvectors 8 \
            --embedding_cache "$EXP_DIR/$config/embeddings" \
            --behavior_path auto \
            --name "exp009-$config" \
            --append
        commit_result "EXP-009 $config eval done"
    ) &
done
wait

echo ""
echo ">>> Phase 3 complete: all evaluations done"

# ──────────────────────────────────────────────
# Final commit
# ──────────────────────────────────────────────
echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-009 results: QFormer Tokenizer" || echo "Nothing to commit"
./push.sh

echo ""
echo "=========================================="
echo "EXP-009 complete!"
echo "=========================================="
