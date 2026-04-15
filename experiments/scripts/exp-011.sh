#!/bin/bash
set -e

# EXP-011: Codebook Size 消融 — FSQ 等大 1024/4096 + OPQ 等 bits 对照
# Date: 2026-04-15
# 背景: EXP-010 NTP baseline 效果极差，根因之一是 L1=1024, L2=1024, L3=4096 不等大。
#        OneMall 实际用三层等大 4096×4096×4096。需要确定最优 codebook 配置。
#
# 已有 baseline (EXP-008):
#   A: MLP-FSQ 1024×1024×4096 [4,4,4,4,4,4]  → semantic_neighbor_HR=0.078, collision=10.7%
#   B: OPQ 4×256 (32 bit)                      → semantic_neighbor_HR=0.050, collision=3.5%
#   C: OPQ 8×256 (64 bit)                      → semantic_neighbor_HR=0.033, collision=0.06%
#
# 新增 configs:
#   E: 等大 1024×3, FSQ [4,4,4,4,4]            (30 bit, multi-level)
#   F: 等大 1024×3, FSQ [2,...,2] ×10           (30 bit, binary)
#   G: 等大 4096×3, FSQ [4,4,4,4,4,4]          (36 bit, OneMall 配置)
#   H: 等大 4096×3, FSQ [2,...,2] ×12           (36 bit, OneMall binary 解读)
#   I: OPQ 3×1024 (30 bit, 等 bits 对照 E/F)
#   J: OPQ 3×4096 (36 bit, 等 bits 对照 G/H)

echo "=========================================="
echo "EXP-011: Codebook Size Ablation"
echo "=========================================="

BEHAVIOR_PATH="auto"
LOCK=/tmp/exp011-git.lock

commit_and_push() {
    local label=$1
    flock "$LOCK" bash -c "
        git add experiments/ &&
        git commit -m 'EXP-011 result: ${label}' &&
        ./push.sh
    " || echo "Nothing to commit for ${label}"
}

# ── Config E: 等大 1024×3, FSQ 5d_1024 [4,4,4,4,4] ──
echo ""
echo ">>> Config E: 1024×1024×1024, FSQ [4,4,4,4,4] (30 bit)"
CUDA_VISIBLE_DEVICES=0 python run.py hyperparam --skip_embedding \
    --quantizer rkmeans_fsq --clusters 1024 \
    --fsq_levels 5d_1024 --fsq_projection mlp --fsq_mlp_hidden 64 \
    --behavior_path "$BEHAVIOR_PATH" --skip_ntp \
    --name exp011-1024x3-5d
commit_and_push "1024x3 FSQ 5d_1024"

# ── Config F: 等大 1024×3, FSQ 10d_1024 [2,2,...,2] binary ──
echo ""
echo ">>> Config F: 1024×1024×1024, FSQ [2]×10 binary (30 bit)"
CUDA_VISIBLE_DEVICES=0 python run.py hyperparam --skip_embedding \
    --quantizer rkmeans_fsq --clusters 1024 \
    --fsq_levels 10d_1024 --fsq_projection mlp --fsq_mlp_hidden 64 \
    --behavior_path "$BEHAVIOR_PATH" --skip_ntp \
    --name exp011-1024x3-10d-binary
commit_and_push "1024x3 FSQ 10d_1024 binary"

# ── Config G: 等大 4096×3, FSQ 6d_4096 [4,4,4,4,4,4] (OneMall) ──
echo ""
echo ">>> Config G: 4096×4096×4096, FSQ [4,4,4,4,4,4] (36 bit, OneMall)"
CUDA_VISIBLE_DEVICES=0 python run.py hyperparam --skip_embedding \
    --quantizer rkmeans_fsq --clusters 4096 \
    --fsq_levels 6d_4096 --fsq_projection mlp --fsq_mlp_hidden 64 \
    --behavior_path "$BEHAVIOR_PATH" --skip_ntp \
    --name exp011-4096x3-6d
commit_and_push "4096x3 FSQ 6d_4096 (OneMall)"

# ── Config H: 等大 4096×3, FSQ 12d_4096 [2,...,2] binary (OneMall binary) ──
echo ""
echo ">>> Config H: 4096×4096×4096, FSQ [2]×12 binary (36 bit)"
CUDA_VISIBLE_DEVICES=0 python run.py hyperparam --skip_embedding \
    --quantizer rkmeans_fsq --clusters 4096 \
    --fsq_levels 12d_4096 --fsq_projection mlp --fsq_mlp_hidden 64 \
    --behavior_path "$BEHAVIOR_PATH" --skip_ntp \
    --name exp011-4096x3-12d-binary
commit_and_push "4096x3 FSQ 12d_4096 binary"

# ── Config I: OPQ 3×1024 (30 bit, 等 bits 对照 E/F) ──
echo ""
echo ">>> Config I: OPQ 3×1024 (30 bit, 等 bits 对照)"
CUDA_VISIBLE_DEVICES=0 python run.py hyperparam --skip_embedding \
    --quantizer opq --n_subvectors 3 --n_clusters_per_sub 1024 \
    --behavior_path "$BEHAVIOR_PATH" --skip_ntp \
    --name exp011-opq-3x1024
commit_and_push "OPQ 3x1024 (30 bit)"

# ── Config J: OPQ 3×4096 (36 bit, 等 bits 对照 G/H) ──
echo ""
echo ">>> Config J: OPQ 3×4096 (36 bit, 等 bits 对照)"
CUDA_VISIBLE_DEVICES=0 python run.py hyperparam --skip_embedding \
    --quantizer opq --n_subvectors 3 --n_clusters_per_sub 4096 \
    --behavior_path "$BEHAVIOR_PATH" --skip_ntp \
    --name exp011-opq-3x4096
commit_and_push "OPQ 3x4096 (36 bit)"

echo ""
echo "=========================================="
echo "EXP-011 complete! 6 configs done."
echo "=========================================="
echo ""
echo "Compare with EXP-008 baselines:"
echo "  A: MLP-FSQ 1024×1024×4096 → semantic_neighbor_HR=0.078"
echo "  B: OPQ 4×256 (32 bit)     → semantic_neighbor_HR=0.050"
echo "  C: OPQ 8×256 (64 bit)     → semantic_neighbor_HR=0.033"
