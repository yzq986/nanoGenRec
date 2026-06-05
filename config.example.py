"""
Example environment configuration template.

`config.py` reads these values from environment variables. You can source a
`.env` file, export them in your shell, or map them in your job launcher.
"""

# ============================================================
# S3 Configuration
# ============================================================

S3_BUCKET = "your-public-or-private-bucket"
S3_PREFIX = "gr-demo"
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

HIVE_BEHAVIOR_TABLE = "your_db.behavior_table"
HIVE_CONTENTS_TABLE = "your_db.contents_table"
HIVE_COMMENT_TABLE = "your_db.comment_table"


# ============================================================
# Local/cache paths
# ============================================================

EFS_BASE = "~/.cache/gr_demo"
EFS_EMBEDDING_CACHE = f"{EFS_BASE}/embedding_cache"
EFS_HF_CACHE = f"{EFS_BASE}/huggingface_cache"
EFS_IMAGE_CACHE = f"{EFS_BASE}/image_cache"
EFS_MODEL_CACHE = f"{EFS_BASE}/.model_cache/qwen3_emb"
EFS_DEFAULT_OUTPUT = f"{EFS_BASE}/feed_content_embedding_v4.tar.gz"


# ============================================================
# Default Dates — defined in config.py
# ============================================================
# DEFAULT_DATE, DEFAULT_DATE_START, DEFAULT_DATE_END are now in config.py
