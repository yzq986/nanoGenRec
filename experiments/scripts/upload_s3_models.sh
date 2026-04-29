#!/bin/bash
# 上传 HuggingFace 模型到 S3
# Usage: bash experiments/scripts/upload_s3_models.sh <model>
#
# Models:
#   qwen3-vl-emb-8b    Qwen3-VL-Embedding-8B → s3://.../models/Qwen3-VL-Embedding-8B
#   qwen3-vl-emb-2b    Qwen3-VL-Embedding-2B → s3://.../models/Qwen3-VL-Embedding-2B
#   all                上传以上全部

set -euo pipefail

IDAAS="/mnt/workspace/alibaba-cloud-idaas/alibaba-cloud-idaas"
S3_BASE="s3://example-bucket/zachery"
HF_CACHE="/home/dev/.cache/huggingface/hub"

usage() {
    echo "Usage: bash $0 <model>"
    echo ""
    echo "Available models:"
    echo "  qwen3-vl-emb-8b    Qwen3-VL-Embedding-8B (16.3GB)"
    echo "  qwen3-vl-emb-2b    Qwen3-VL-Embedding-2B (4.27GB)"
    echo "  all                上传以上全部"
    exit 1
}

[ $# -lt 1 ] && usage

upload_model() {
    local MODEL_KEY=$1
    local HF_REPO=$2       # e.g. models--Qwen--Qwen3-VL-Embedding-8B
    local S3_NAME=$3       # e.g. Qwen3-VL-Embedding-8B

    local BLOBS_DIR="${HF_CACHE}/${HF_REPO}/blobs"
    local REFS_DIR="${HF_CACHE}/${HF_REPO}/refs"
    local SNAPSHOTS_DIR="${HF_CACHE}/${HF_REPO}/snapshots"
    local S3_PATH="${S3_BASE}/models/${S3_NAME}"

    echo ""
    echo "========================================"
    echo "Model:  ${MODEL_KEY}"
    echo "S3:     ${S3_PATH}"
    echo "========================================"

    if [ ! -d "${BLOBS_DIR}" ]; then
        echo "ERROR: blobs dir not found: ${BLOBS_DIR}"
        echo "  Run: hf download Qwen/${S3_NAME}"
        return 1
    fi

    echo "Local:  ${BLOBS_DIR}"
    echo "Size:   $(du -sh "${BLOBS_DIR}" | cut -f1)"
    echo ""

    # 上传 blobs（真正的模型文件）
    echo ">>> Uploading blobs/..."
    "${IDAAS}" exec aws s3 sync "${BLOBS_DIR}/" "${S3_PATH}/blobs/" --no-progress

    # 上传 refs 和 snapshots（含 symlink 的目录结构，用于还原 HF cache）
    echo ">>> Uploading refs/ and snapshots/..."
    "${IDAAS}" exec aws s3 sync "${REFS_DIR}/" "${S3_PATH}/refs/" --no-progress
    "${IDAAS}" exec aws s3 sync "${SNAPSHOTS_DIR}/" "${S3_PATH}/snapshots/" \
        --no-progress --follow-symlinks

    echo ""
    echo ">>> Verifying..."
    "${IDAAS}" exec aws s3 ls "${S3_PATH}/" | head -10
    echo ""
    echo "  Upload complete: ${S3_PATH}"
}

MODEL="$1"

case "${MODEL}" in
    qwen3-vl-emb-8b)
        upload_model "qwen3-vl-emb-8b" \
            "models--Qwen--Qwen3-VL-Embedding-8B" \
            "Qwen3-VL-Embedding-8B"
        ;;
    qwen3-vl-emb-2b)
        upload_model "qwen3-vl-emb-2b" \
            "models--Qwen--Qwen3-VL-Embedding-2B" \
            "Qwen3-VL-Embedding-2B"
        ;;
    all)
        upload_model "qwen3-vl-emb-8b" \
            "models--Qwen--Qwen3-VL-Embedding-8B" \
            "Qwen3-VL-Embedding-8B"
        upload_model "qwen3-vl-emb-2b" \
            "models--Qwen--Qwen3-VL-Embedding-2B" \
            "Qwen3-VL-Embedding-2B"
        ;;
    *)
        echo "Unknown model: ${MODEL}"
        usage
        ;;
esac

echo ""
echo "All done!"
