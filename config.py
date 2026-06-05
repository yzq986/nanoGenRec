"""Model configurations and public project defaults.

Private deployments can override paths and table names through environment
variables. The repository intentionally has no dependency on private config
packages.
"""

import os
from dataclasses import dataclass

try:
    import torch
except ImportError:  # Allow lightweight config imports before ML deps are installed.
    torch = None


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


S3_BUCKET = _env("GR_S3_BUCKET", "your-s3-bucket")
S3_PREFIX = _env("GR_S3_PREFIX", "gr-demo")
S3_BASE = _env("GR_S3_BASE", f"s3://{S3_BUCKET}/{S3_PREFIX}")

S3_CONTENT_TEXT_EXPOSED = _env("GR_S3_CONTENT_TEXT_EXPOSED", f"{S3_BASE}/feed_content_text_exposed")
S3_CONTENT_TEXT_EXPOSED_S3 = _env("GR_S3_CONTENT_TEXT_EXPOSED_S3", f"{S3_BASE}/feed_content_text_exposed_s3")
S3_USER_BEHAVIOR = _env("GR_S3_USER_BEHAVIOR", f"{S3_BASE}/feed_user_behavior")
S3_RKMEANS_BASE = _env("GR_S3_RKMEANS_BASE", f"{S3_BASE}/feed_rkmeans")
S3_RKMEANS_QWEN = _env("GR_S3_RKMEANS_QWEN", f"{S3_BASE}/feed_rkmeans_qwen")
S3_OLD_EMBEDDINGS = _env("GR_S3_OLD_EMBEDDINGS", f"{S3_BASE}/feed_embeddings")
S3_EMBEDDING_CACHE_BACKUP = _env("GR_S3_EMBEDDING_CACHE_BACKUP", f"{S3_BASE}/embedding_cache_backup")

HIVE_BEHAVIOR_TABLE = _env("GR_HIVE_BEHAVIOR_TABLE", "your_db.behavior_table")
HIVE_CONTENTS_TABLE = _env("GR_HIVE_CONTENTS_TABLE", "your_db.contents_table")
HIVE_COMMENT_TABLE = _env("GR_HIVE_COMMENT_TABLE", "your_db.comment_table")

EFS_BASE = _env("GR_EFS_BASE", os.path.expanduser("~/.cache/gr_demo"))
EFS_EMBEDDING_CACHE = _env("GR_EFS_EMBEDDING_CACHE", f"{EFS_BASE}/embedding_cache")
EFS_HF_CACHE = _env("GR_EFS_HF_CACHE", f"{EFS_BASE}/huggingface_cache")
EFS_IMAGE_CACHE = _env("GR_EFS_IMAGE_CACHE", f"{EFS_BASE}/image_cache")
EFS_MODEL_CACHE = _env("GR_EFS_MODEL_CACHE", f"{EFS_BASE}/model_cache/qwen3_emb")
EFS_DEFAULT_OUTPUT = _env("GR_EFS_DEFAULT_OUTPUT", f"{EFS_BASE}/feed_content_embedding_v4.tar.gz")

# ── Default dates (public, version-controlled) ──
DEFAULT_DATE = "2026-03-31"
DEFAULT_DATE_START = "2026-03-01"
DEFAULT_DATE_END = "2026-03-31"

# Set HuggingFace cache directories when the configured cache path is writable.
HF_CACHE_DIR = EFS_HF_CACHE
try:
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR
except OSError:
    pass

# 模型配置映射: model_key -> (hf_model_name, embedding_dim, is_multimodal, batch_size)
# batch_size 基于 8xA100 40GB 测试，保守设置避免 OOM
# 图文多模态: Qwen3-VL-Embedding-2B/8B
# 纯文本: Qwen3-Embedding-0.6B/4B/8B
MODEL_CONFIGS = {
    # 图文多模态 (图片占用额外显存)
    "qwen3-vl-8b": ("Qwen/Qwen3-VL-Embedding-8B", 4096, True, 8),
    "qwen3-vl-2b": ("Qwen/Qwen3-VL-Embedding-2B", 2048, True, 16),
    # 纯文本
    "qwen3-8b": ("Qwen/Qwen3-Embedding-8B", 4096, False, 16),
    "qwen3-4b": ("Qwen/Qwen3-Embedding-4B", 2560, False, 32),
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", 1024, False, 64),
}


@dataclass
class Config:
    # 数据路径
    INPUT_PATH: str = f"{S3_CONTENT_TEXT_EXPOSED}/{DEFAULT_DATE}"
    INPUT_PATH_OLD_EMB: str = f"{S3_OLD_EMBEDDINGS}/{DEFAULT_DATE}"  # 旧 embedding
    OUTPUT_PATH_BASE: str = S3_RKMEANS_BASE

    # Embedding 模型 (通过 --model 参数选择)
    EMBEDDING_BATCH_SIZE: int = 16  # 8xA100 40GB
    EMBEDDING_MAX_LENGTH: int = 2048

    # RKMeans 参数
    NUM_LAYERS: int = 3
    NUM_CLUSTERS: int = 1024
    NORMALIZE_RESIDUALS: bool = True  # 只对 layer 0 输入做 L2 normalize，残差不 normalize

    # FAISS KMeans 训练参数
    NITER: int = 25       # Lloyd's iterations per layer
    NREDO: int = 3        # number of restarts, keep best

    # 多 GPU 配置
    NUM_GPUS: int = torch.cuda.device_count() if torch is not None and torch.cuda.is_available() else 0
    DEVICE: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
