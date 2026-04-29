#!/bin/bash
# 通用 S3 数据下载脚本
# Usage: bash experiments/scripts/download_s3_data.sh <dataset>
#
# Datasets:
#   behavior-v2      feed_user_behavior_v2 (正+负样本, 2026-03-18~03-31)
#   exposure-neg     feed_user_exposure_neg (ENTP 负样本, 2026-03-01~03-31)
#   behavior         feed_user_behavior (原始行为数据, 2026-03-18~03-31)

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
    echo "  sid-0.6b       SID cache (Qwen3-0.6B, 14d) → experiments/sid_cache/exp026-0.6b-14d"
    echo "  sid-4b         SID cache (Qwen3-4B,  14d) → experiments/sid_cache/exp026-4b-14d"
    echo "  sid-8b         SID cache (Qwen3-8B,  14d) → experiments/sid_cache/exp026-8b-14d"
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
