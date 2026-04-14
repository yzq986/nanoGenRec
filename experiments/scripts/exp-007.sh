#!/bin/bash
set -e

# EXP-007: Collaborative Signal Enhanced Embedding (Qwen3-0.6B Full Fine-tune)
# Date: 2026-04-13
# IDEA: sid-1 — I2I contrastive learning to inject collaborative signals into embedding
# Hardware: 8 x A100 40GB

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOCKFILE="/tmp/exp007_git.lock"
EXP_DIR="experiments/hyperparam/2026-04-13_exp007-collab-embed"
mkdir -p "$EXP_DIR"

commit_result() {
    local msg="$1"
    (
        flock -x 200
        git add experiments/ model/contrastive_finetune.py || true
        git commit -m "$msg" || echo "Nothing to commit"
        ./push.sh
    ) 200>"$LOCKFILE"
}

echo "=========================================="
echo "EXP-007: Collaborative Signal Enhanced Embedding"
echo "=========================================="

# ──────────────────────────────────────────────
# Phase 1: Contrastive fine-tune (all 8 GPUs per config, sequential)
# 8 GPU DDP: 2x throughput + 2x negatives vs 4 GPU
# ──────────────────────────────────────────────
echo ""
echo ">>> Phase 1: Contrastive fine-tune Qwen3-0.6B (8 GPU DDP, sequential)"

# Config A: τ=0.05, 3 epochs
echo "[Config A] Starting: τ=0.05, 3 epochs, 8 GPU"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --temperature 0.05 \
    --epochs 3 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 1e-5 \
    --output_dir "$EXP_DIR/config_a_t005_ep3" \
    --experiment_name "config_a"
echo "[Config A] Done"
commit_result "EXP-007 config A done: τ=0.05, 3ep"

# Config B: τ=0.07, 3 epochs
echo "[Config B] Starting: τ=0.07, 3 epochs, 8 GPU"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --temperature 0.07 \
    --epochs 3 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 1e-5 \
    --output_dir "$EXP_DIR/config_b_t007_ep3" \
    --experiment_name "config_b"
echo "[Config B] Done"
commit_result "EXP-007 config B done: τ=0.07, 3ep"

# Config C: τ=0.05, 5 epochs
echo "[Config C] Starting: τ=0.05, 5 epochs, 8 GPU"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --temperature 0.05 \
    --epochs 5 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 1e-5 \
    --output_dir "$EXP_DIR/config_c_t005_ep5" \
    --experiment_name "config_c"
echo "[Config C] Done"
commit_result "EXP-007 config C done: τ=0.05, 5ep"
echo ""
echo ">>> Phase 1 complete: all 3 configs trained"

# ──────────────────────────────────────────────
# Phase 2: Generate embeddings from fine-tuned models (parallel)
# ──────────────────────────────────────────────
echo ""
echo ">>> Phase 2: Generate embeddings from fine-tuned models"

for config in config_a_t005_ep3 config_b_t007_ep3 config_c_t005_ep5; do
    echo "  Generating embeddings for $config ..."
    python run.py encode \
        --model_path "$EXP_DIR/$config/model" \
        --output_dir "$EXP_DIR/$config/embeddings" \
        --model_type qwen3-text &
done
wait
echo ">>> Phase 2 complete: all embeddings generated"

# ──────────────────────────────────────────────
# Phase 3: Evaluate — embedding_hit_rate + OPQ intrinsic (parallel)
# ──────────────────────────────────────────────
echo ""
echo ">>> Phase 3: Evaluate all configs"

# Baseline (original Qwen3 embedding, already cached)
(
    echo "[Baseline] Evaluating original Qwen3-0.6b embedding..."
    python run.py hyperparam --skip_embedding \
        --quantizer opq --n_subvectors 8 \
        --behavior_path auto \
        --name exp007-baseline \
        --append
    commit_result "EXP-007 baseline eval done"
) &

# Fine-tuned configs
for config in config_a_t005_ep3 config_b_t007_ep3 config_c_t005_ep5; do
    (
        echo "[$config] Evaluating..."
        python run.py hyperparam --skip_embedding \
            --quantizer opq --n_subvectors 8 \
            --embedding_cache "$EXP_DIR/$config/embeddings" \
            --behavior_path auto \
            --name "exp007-$config" \
            --append
        commit_result "EXP-007 $config eval done"
    ) &
done
wait

echo ""
echo ">>> Phase 3 complete: all evaluations done"

# ──────────────────────────────────────────────
# Final commit
# ──────────────────────────────────────────────
echo ""
echo ">>> Final commit..."
git add experiments/
git commit -m "EXP-007 results: Collaborative Signal Enhanced Embedding" || echo "Nothing to commit"
./push.sh

echo ""
echo "=========================================="
echo "EXP-007 complete!"
echo "=========================================="
