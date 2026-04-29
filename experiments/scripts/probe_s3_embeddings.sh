#!/bin/bash
# 探查 S3 上的 embedding cache 目录结构
# Usage: bash experiments/scripts/probe_s3_embeddings.sh

set -euo pipefail

IDAAS="/mnt/workspace/alibaba-cloud-idaas/alibaba-cloud-idaas"
S3_BASE="s3://example-bucket/zachery"

echo "========================================"
echo "S3 Embedding Cache Probe"
echo "S3 base: ${S3_BASE}"
echo "========================================"

echo ""
echo ">>> Top-level directories under ${S3_BASE}:"
"${IDAAS}" exec aws s3 ls "${S3_BASE}/" 2>/dev/null || echo "  (failed)"

echo ""
echo ">>> Probing known embedding paths..."

CANDIDATES=(
    "embedding_cache"
    "embeddings"
    "emb_cache"
    "sid_cache"
    "qwen3-embedding"
    "qwen3_embedding"
    "content_embeddings"
    "item_embeddings"
)

for path in "${CANDIDATES[@]}"; do
    FULL="${S3_BASE}/${path}"
    result=$("${IDAAS}" exec aws s3 ls "${FULL}/" 2>/dev/null || echo "")
    if [ -n "${result}" ]; then
        echo "  FOUND: ${FULL}/"
        echo "${result}" | head -10 | sed 's/^/    /'
    fi
done

echo ""
echo ">>> Probing qwen3 model-specific paths..."

MODELS=(
    "qwen3-0.6b"
    "qwen3-2b"
    "qwen3-4b"
    "qwen3-7b"
    "qwen3-8b"
    "qwen3-vl-2b"
    "qwen3-vl-8b"
    "qwen3-embedding-0.6b"
    "qwen3-embedding-2b"
    "qwen3-embedding-8b"
)

for model in "${MODELS[@]}"; do
    for prefix in "" "embedding_cache/" "embeddings/" "sid_cache/"; do
        FULL="${S3_BASE}/${prefix}${model}"
        result=$("${IDAAS}" exec aws s3 ls "${FULL}/" 2>/dev/null || echo "")
        if [ -n "${result}" ]; then
            echo "  FOUND: ${FULL}/"
            echo "${result}" | head -5 | sed 's/^/    /'
            break
        fi
    done
done

echo ""
echo ">>> Probing sid_cache paths (may differ by experiment)..."
SID_PATHS=(
    "sid_cache"
    "sid_caches"
    "ntp_sid"
    "gr-demo/sid_cache"
    "gr_demo/sid_cache"
)
for path in "${SID_PATHS[@]}"; do
    FULL="${S3_BASE}/${path}"
    result=$("${IDAAS}" exec aws s3 ls "${FULL}/" 2>/dev/null || echo "")
    if [ -n "${result}" ]; then
        echo "  FOUND: ${FULL}/"
        echo "${result}" | head -10 | sed 's/^/    /'
    fi
done

echo ""
echo "Done."
