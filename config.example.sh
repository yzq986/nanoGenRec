#!/bin/bash
# Example shell configuration for public/open-source deployments.
# Source this file or copy the exports into your job environment.

export GR_S3_BUCKET="your-public-or-private-bucket"
export GR_S3_PREFIX="gr-demo"
export GR_S3_BASE="s3://${GR_S3_BUCKET}/${GR_S3_PREFIX}"

export GR_S3_CONTENT_TEXT_EXPOSED="${GR_S3_BASE}/feed_content_text_exposed"
export GR_S3_CONTENT_TEXT_EXPOSED_S3="${GR_S3_BASE}/feed_content_text_exposed_s3"
export GR_S3_USER_BEHAVIOR="${GR_S3_BASE}/feed_user_behavior"
export GR_S3_RKMEANS_BASE="${GR_S3_BASE}/feed_rkmeans"
export GR_S3_RKMEANS_QWEN="${GR_S3_BASE}/feed_rkmeans_qwen"
export GR_S3_OLD_EMBEDDINGS="${GR_S3_BASE}/feed_embeddings"
export GR_S3_EMBEDDING_CACHE_BACKUP="${GR_S3_BASE}/embedding_cache_backup"

export GR_HIVE_BEHAVIOR_TABLE="your_db.behavior_table"
export GR_HIVE_CONTENTS_TABLE="your_db.contents_table"
export GR_HIVE_COMMENT_TABLE="your_db.comment_table"

export GR_EFS_BASE="${HOME}/.cache/gr_demo"
export GR_EFS_EMBEDDING_CACHE="${GR_EFS_BASE}/embedding_cache"
export GR_EFS_HF_CACHE="${GR_EFS_BASE}/huggingface_cache"
export GR_EFS_IMAGE_CACHE="${GR_EFS_BASE}/image_cache"
export GR_EFS_MODEL_CACHE="${GR_EFS_BASE}/model_cache/qwen3_emb"
export GR_EFS_DEFAULT_OUTPUT="${GR_EFS_BASE}/feed_content_embedding_v4.tar.gz"
