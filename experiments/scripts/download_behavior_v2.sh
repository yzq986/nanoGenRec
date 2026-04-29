#!/bin/bash
set -euo pipefail

IDAAS="/mnt/workspace/alibaba-cloud-idaas/alibaba-cloud-idaas"
S3_PATH="s3://example-bucket/gr-demo/feed_user_behavior_v2"
LOCAL_DIR="/mnt/workspace/gr-demo-behavior-v2"

mkdir -p "${LOCAL_DIR}"

echo "Listing S3 path..."
"${IDAAS}" exec aws s3 ls "${S3_PATH}/"

echo ""
echo "Downloading to ${LOCAL_DIR} ..."
"${IDAAS}" exec aws s3 sync "${S3_PATH}/" "${LOCAL_DIR}/" --no-progress

echo ""
echo "Done. Files:"
ls -lh "${LOCAL_DIR}/"
