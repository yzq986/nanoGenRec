#!/bin/bash
# 通用 S3 数据下载脚本
# Usage: bash experiments/scripts/download_s3_data.sh <dataset>
#
# Datasets:
#   behavior-v2        feed_user_behavior_v2 (正+负样本, 2026-03-18~03-31)
#   exposure-neg       feed_user_exposure_neg (ENTP 负样本, 2026-03-01~03-31)
#   behavior           feed_user_behavior (原始行为数据, 2026-03-18~03-31)
#   qwen3-emb-0.6b     Qwen3-Embedding-0.6B  → ~/.cache/huggingface/hub/...
#   qwen3-emb-4b       Qwen3-Embedding-4B    → ~/.cache/huggingface/hub/...
#   qwen3-emb-8b       Qwen3-Embedding-8B    → ~/.cache/huggingface/hub/...
#   qwen3-vl-emb-8b    Qwen3-VL-Embedding-8B → ~/.cache/huggingface/hub/...
#   qwen3-vl-emb-2b    Qwen3-VL-Embedding-2B → ~/.cache/huggingface/hub/...

set -euo pipefail

IDAAS="/mnt/workspace/alibaba-cloud-idaas/alibaba-cloud-idaas"
S3_BASE="s3://example-bucket/zachery"

usage() {
    echo "Usage: bash $0 <dataset>"
    echo ""
    echo "Available datasets:"
    echo "  behavior-v2    feed_user_behavior_v2 (正+内联负样本, 2026-03-18~03-31)"
    echo "  exposure-neg   feed_user_exposure_neg (ENTP 负样本, 2026-03-01~03-31)"
    echo "  behavior       feed_user_behavior (原始行为数据, 2026-03-18~03-31)"
    echo "  sid-0.6b           SID cache (Qwen3-0.6B, 14d) → experiments/sid_cache/exp026-0.6b-14d"
    echo "  sid-4b             SID cache (Qwen3-4B,  14d) → experiments/sid_cache/exp026-4b-14d"
    echo "  sid-8b             SID cache (Qwen3-8B,  14d) → experiments/sid_cache/exp026-8b-14d"
    echo "  emb-cache-0.6b     Embedding cache (qwen3-0.6b) → EFS embedding_cache/qwen3-0.6b/"
    echo "  emb-cache-4b       Embedding cache (qwen3-4b)   → EFS embedding_cache/qwen3-4b/"
    echo "  emb-cache-8b       Embedding cache (qwen3-8b)   → EFS embedding_cache/qwen3-8b/"
    echo "  qwen3-emb-0.6b     Qwen3-Embedding-0.6B  (~1.2GB)"
    echo "  qwen3-emb-4b       Qwen3-Embedding-4B    (~7.8GB)"
    echo "  qwen3-emb-8b       Qwen3-Embedding-8B    (~15GB)"
    echo "  qwen3-vl-emb-8b    Qwen3-VL-Embedding-8B (~16.3GB)"
    echo "  qwen3-vl-emb-2b    Qwen3-VL-Embedding-2B (~4.27GB)"
    exit 1
}

[ $# -lt 1 ] && usage

DATASET="$1"

case "${DATASET}" in
    behavior-v2)
        S3_PATH="${S3_BASE}/feed_user_behavior_v2"
        LOCAL_DIR="/mnt/workspace/gr-demo-behavior-v2"
        ;;
    exposure-neg)
        S3_PATH="${S3_BASE}/feed_user_exposure_neg/2026-03-01_2026-03-31"
        LOCAL_DIR="/mnt/workspace/gr-demo-exposure-neg/2026-03-01_2026-03-31"
        ;;
    behavior)
        S3_PATH="${S3_BASE}/feed_user_behavior"
        LOCAL_DIR="/mnt/workspace/gr-demo-behavior-cache"
        ;;
    sid-0.6b)
        S3_PATH="${S3_BASE}/sid_cache/exp026-0.6b-14d"
        LOCAL_DIR="experiments/sid_cache/exp026-0.6b-14d"
        ;;
    sid-4b)
        S3_PATH="${S3_BASE}/sid_cache/exp026-4b-14d"
        LOCAL_DIR="experiments/sid_cache/exp026-4b-14d"
        ;;
    sid-8b)
        S3_PATH="${S3_BASE}/sid_cache/exp026-8b-14d"
        LOCAL_DIR="experiments/sid_cache/exp026-8b-14d"
        ;;
    emb-cache-0.6b)
        S3_PATH="s3://example-bucket/gr-demo/embedding_cache_backup/qwen3-0.6b"
        LOCAL_DIR="${HOME}/gr_demo_cache/embedding_cache/qwen3-0.6b"
        ;;
    emb-cache-4b)
        S3_PATH="s3://example-bucket/gr-demo/embedding_cache_backup/qwen3-4b"
        LOCAL_DIR="${HOME}/gr_demo_cache/embedding_cache/qwen3-4b"
        ;;
    emb-cache-8b)
        S3_PATH="s3://example-bucket/gr-demo/embedding_cache_backup/qwen3-8b"
        LOCAL_DIR="${HOME}/gr_demo_cache/embedding_cache/qwen3-8b"
        ;;
    qwen3-emb-0.6b)
        S3_PATH="${S3_BASE}/models/Qwen3-Embedding-0.6B"
        LOCAL_DIR="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B"
        ;;
    qwen3-emb-4b)
        S3_PATH="${S3_BASE}/models/Qwen3-Embedding-4B"
        LOCAL_DIR="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B"
        ;;
    qwen3-emb-8b)
        S3_PATH="${S3_BASE}/models/Qwen3-Embedding-8B"
        LOCAL_DIR="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B"
        ;;
    qwen3-vl-emb-8b)
        S3_PATH="${S3_BASE}/models/Qwen3-VL-Embedding-8B"
        LOCAL_DIR="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-8B"
        ;;
    qwen3-vl-emb-2b)
        S3_PATH="${S3_BASE}/models/Qwen3-VL-Embedding-2B"
        LOCAL_DIR="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B"
        ;;
    *)
        echo "Unknown dataset: ${DATASET}"
        usage
        ;;
esac

mkdir -p "${LOCAL_DIR}"

echo "========================================"
echo "Dataset:  ${DATASET}"
echo "S3:       ${S3_PATH}"
echo "Local:    ${LOCAL_DIR}"
echo "========================================"

echo ""
echo "Listing S3 path..."
"${IDAAS}" exec aws s3 ls "${S3_PATH}/"

echo ""
echo "Downloading..."
"${IDAAS}" exec aws s3 sync "${S3_PATH}/" "${LOCAL_DIR}/" --no-progress

echo ""
echo "Done. Contents:"
ls -lh "${LOCAL_DIR}/" | head -20
echo ""
echo "Total size: $(du -sh ${LOCAL_DIR} | cut -f1)"
