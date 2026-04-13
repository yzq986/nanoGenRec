#!/bin/bash
# Example Shell Configuration Template
# Copy to config/config.sh and fill in actual values.

S3_BUCKET="your-s3-bucket"
S3_PREFIX="your-prefix"
S3_BASE="s3://${S3_BUCKET}/${S3_PREFIX}"

S3_CONTENT_TEXT="${S3_BASE}/feed_content_text"
S3_CONTENT_TEXT_EXPOSED="${S3_BASE}/feed_content_text_exposed"
S3_USER_BEHAVIOR="${S3_BASE}/feed_user_behavior"
S3_RKMEANS_BASE="${S3_BASE}/feed_rkmeans"
S3_RKMEANS_QWEN="${S3_BASE}/feed_rkmeans_qwen"

RAVEN_ENDPOINT="http://your-internal-endpoint:10271"

EFS_BASE="~/.cache/gr_demo"
EFS_EMBEDDING_CACHE="${EFS_BASE}/embedding_cache"

DEFAULT_DATE="2026-03-31"
DEFAULT_DATE_START="2026-03-24"
DEFAULT_DATE_END="2026-03-31"
