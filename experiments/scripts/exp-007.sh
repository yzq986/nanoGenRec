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

# Options: --no-smoke to skip smoke test, then config names (A B C)
# Examples: ./exp-007.sh --no-smoke B C
#           ./exp-007.sh C
#           ./exp-007.sh  (runs all with smoke test)
SKIP_SMOKE=false
if [[ "${1:-}" == "--no-smoke" ]]; then
    SKIP_SMOKE=true
    shift
fi
CONFIGS="${@:-A B C D E F G H I}"
run_config() { [[ " $CONFIGS " == *" $1 "* ]]; }
echo "Selected configs: $CONFIGS (smoke: $( $SKIP_SMOKE && echo skip || echo run ))"

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
# Phase 0: Smoke test — 1% data, 10 steps, verify pipeline
# ──────────────────────────────────────────────
if $SKIP_SMOKE; then
    echo ">>> Phase 0: Skipped (--no-smoke)"
else
    echo ""
    echo ">>> Phase 0: Smoke test (1% data, 10 steps)"
    torchrun --nproc_per_node=8 \
        model/contrastive_finetune.py \
        --dry_run \
        --temperature 0.05 \
        --batch_size 32 \
        --grad_accum 8 \
        --lr 1e-5 \
        --output_dir "$EXP_DIR/smoke_test"
    echo ">>> Smoke test PASSED"
    rm -rf "$EXP_DIR/smoke_test"
fi

# ──────────────────────────────────────────────
# Baseline: evaluate original Qwen3 embedding (no fine-tune)
# ──────────────────────────────────────────────
if run_config BL; then
echo ">>> Baseline: evaluating original Qwen3-0.6B embedding (no fine-tune)"
python run.py hyperparam --skip_embedding \
    --quantizer opq --n_subvectors 8 \
    --name exp007-baseline \
    --append
commit_result "EXP-007 baseline eval done"
fi

# ──────────────────────────────────────────────
# Phase 1: Contrastive fine-tune (all 8 GPUs per config, sequential)
# 8 GPU DDP: 2x throughput + 2x negatives vs 4 GPU
# ──────────────────────────────────────────────
echo ""
echo ">>> Phase 1: Contrastive fine-tune Qwen3-0.6B (8 GPU DDP, sequential)"

# Config A: τ=0.05, 1 epoch
if run_config A; then
echo "[Config A] Starting: τ=0.05, 1 epoch, 8 GPU"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --temperature 0.05 \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 1e-5 \
    --output_dir "$EXP_DIR/config_a_t005_ep1" \
    --experiment_name "config_a"
echo "[Config A] Done"
commit_result "EXP-007 config A done: τ=0.05, 1ep"
fi

# Config B: τ=0.07, 1 epoch
if run_config B; then
echo "[Config B] Starting: τ=0.07, 1 epoch, 8 GPU"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --temperature 0.07 \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 1e-5 \
    --output_dir "$EXP_DIR/config_b_t007_ep1" \
    --experiment_name "config_b"
echo "[Config B] Done"
commit_result "EXP-007 config B done: τ=0.07, 1ep"
fi

# Config C: τ=0.05, 1 epoch, lr=3e-5 (higher lr to see if faster convergence)
if run_config C; then
echo "[Config C] Starting: τ=0.05, 1 epoch, lr=3e-5, 8 GPU"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --temperature 0.05 \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 3e-5 \
    --output_dir "$EXP_DIR/config_c_t005_lr3e5" \
    --experiment_name "config_c"
echo "[Config C] Done"
commit_result "EXP-007 config C done: τ=0.05, lr=3e-5"
fi

# ── Round 2: aggressive lr (cap_loss flat in A/B/C → model not learning) ──

# Config D: lr=1e-4 (10x of A)
if run_config D; then
echo "[Config D] Starting: τ=0.05, 1 epoch, lr=1e-4, 8 GPU"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --temperature 0.05 \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 1e-4 \
    --output_dir "$EXP_DIR/config_d_lr1e4" \
    --experiment_name "config_d"
echo "[Config D] Done"
commit_result "EXP-007 config D done: τ=0.05, lr=1e-4"
fi

# Config E: lr=3e-4 (30x of A)
if run_config E; then
echo "[Config E] Starting: τ=0.05, 1 epoch, lr=3e-4, 8 GPU"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --temperature 0.05 \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 3e-4 \
    --output_dir "$EXP_DIR/config_e_lr3e4" \
    --experiment_name "config_e"
echo "[Config E] Done"
commit_result "EXP-007 config E done: τ=0.05, lr=3e-4"
fi

# Config F: lr=1e-3 (100x of A)
if run_config F; then
echo "[Config F] Starting: τ=0.05, 1 epoch, lr=1e-3, 8 GPU"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --temperature 0.05 \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 1e-3 \
    --output_dir "$EXP_DIR/config_f_lr1e3" \
    --experiment_name "config_f"
echo "[Config F] Done"
commit_result "EXP-007 config F done: τ=0.05, lr=1e-3"
fi

# ── Round 3: LoRA (freeze base, concentrate gradient on adapter) ──

# Config G: LoRA r=16, lr=1e-4
if run_config G; then
echo "[Config G] Starting: LoRA r=16, τ=0.05, lr=1e-4, 8 GPU"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --lora --lora_rank 16 \
    --temperature 0.05 \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 1e-4 \
    --output_dir "$EXP_DIR/config_g_lora_r16" \
    --experiment_name "config_g"
echo "[Config G] Done"
commit_result "EXP-007 config G done: LoRA r=16, lr=1e-4"
fi

# Config H: LoRA r=16, lr=5e-4
if run_config H; then
echo "[Config H] Starting: LoRA r=16, τ=0.05, lr=5e-4, 8 GPU"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --lora --lora_rank 16 \
    --temperature 0.05 \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 5e-4 \
    --output_dir "$EXP_DIR/config_h_lora_lr5e4" \
    --experiment_name "config_h"
echo "[Config H] Done"
commit_result "EXP-007 config H done: LoRA r=16, lr=5e-4"
fi

# Config I: LoRA r=64, lr=1e-4 (more capacity)
if run_config I; then
echo "[Config I] Starting: LoRA r=64, τ=0.05, lr=1e-4, 8 GPU"
torchrun --nproc_per_node=8 \
    model/contrastive_finetune.py \
    --lora --lora_rank 64 \
    --temperature 0.05 \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 1e-4 \
    --output_dir "$EXP_DIR/config_i_lora_r64" \
    --experiment_name "config_i"
echo "[Config I] Done"
commit_result "EXP-007 config I done: LoRA r=64, lr=1e-4"
fi

echo ""
echo ">>> Phase 1 complete: selected configs trained"

# Config name → directory mapping
declare -A CONFIG_DIRS=(
    [A]=config_a_t005_ep1
    [B]=config_b_t007_ep1
    [C]=config_c_t005_lr3e5
    [D]=config_d_lr1e4
    [E]=config_e_lr3e4
    [F]=config_f_lr1e3
    [G]=config_g_lora_r16
    [H]=config_h_lora_lr5e4
    [I]=config_i_lora_r64
)

# ──────────────────────────────────────────────
# Phase 2: Generate embeddings from fine-tuned models (parallel)
# ──────────────────────────────────────────────
echo ""
echo ">>> Phase 2: Generate embeddings from fine-tuned models"

for key in A B C D E F G H I; do
    run_config "$key" || continue
    config="${CONFIG_DIRS[$key]}"
    echo "  Generating embeddings for $config ..."
    python run.py encode \
        --model_path "$EXP_DIR/$config/model" \
        --output_dir "$EXP_DIR/$config/embeddings" \
        --model_type qwen3-text &
done
wait
echo ">>> Phase 2 complete: embeddings generated"

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
for key in A B C D E F G H I; do
    run_config "$key" || continue
    config="${CONFIG_DIRS[$key]}"
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
