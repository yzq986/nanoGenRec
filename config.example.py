"""
Example Configuration Template

Copy this to config/config.py and fill in your actual values.
See README for setup instructions.
"""

# ============================================================
# S3 Configuration
# ============================================================

S3_BUCKET = "your-s3-bucket"
S3_PREFIX = "your-prefix"
S3_BASE = f"s3://{S3_BUCKET}/{S3_PREFIX}"

S3_CONTENT_TEXT_EXPOSED = f"{S3_BASE}/feed_content_text_exposed"
S3_USER_BEHAVIOR = f"{S3_BASE}/feed_user_behavior"
S3_RKMEANS_BASE = f"{S3_BASE}/feed_rkmeans"
S3_RKMEANS_QWEN = f"{S3_BASE}/feed_rkmeans_qwen"
S3_OLD_EMBEDDINGS = f"{S3_BASE}/feed_embeddings"
S3_EMBEDDING_CACHE_BACKUP = f"{S3_BASE}/embedding_cache_backup"


# ============================================================
# Internal Endpoints
# ============================================================

RAVEN_ENDPOINT = "http://your-internal-endpoint:10271"


# ============================================================
# Hive Table Names
# ============================================================

HIVE_BEHAVIOR_TABLE = "your_db.behavior_table"
HIVE_CONTENTS_TABLE = "your_db.contents_table"
HIVE_COMMENT_TABLE = "your_db.comment_table"


# ============================================================
# EFS / cloud notebook Paths
# ============================================================

EFS_BASE = "~/.cache/gr_demo"
EFS_EMBEDDING_CACHE = f"{EFS_BASE}/embedding_cache"
EFS_HF_CACHE = f"{EFS_BASE}/huggingface_cache"
EFS_IMAGE_CACHE = f"{EFS_BASE}/image_cache"
EFS_MODEL_CACHE = f"{EFS_BASE}/.model_cache/qwen3_emb"
EFS_DEFAULT_OUTPUT = f"{EFS_BASE}/feed_content_embedding_v4.tar.gz"


# ============================================================
# Default Dates
# ============================================================

DEFAULT_DATE = "2026-03-31"
DEFAULT_DATE_START = "2026-03-24"
DEFAULT_DATE_END = "2026-03-31"
