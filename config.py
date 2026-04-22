"""Model configurations and project-level Config dataclass."""

import os
from dataclasses import dataclass

import torch

from config import (
    S3_CONTENT_TEXT_EXPOSED,
    S3_OLD_EMBEDDINGS,
    S3_RKMEANS_BASE,
    S3_EMBEDDING_CACHE_BACKUP,
    EFS_HF_CACHE,
    EFS_IMAGE_CACHE,
    EFS_EMBEDDING_CACHE,
)

# ── Default dates (non-sensitive, version-controlled) ──
DEFAULT_DATE = "2026-03-31"
DEFAULT_DATE_START = "2026-03-01"
DEFAULT_DATE_END = "2026-03-31"

# 设置 HuggingFace 缓存目录到大容量 EFS（仅在 cloud notebook 环境下生效）
HF_CACHE_DIR = EFS_HF_CACHE
try:
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR
except OSError:
    pass  # 非 cloud notebook 环境，跳过

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
    NUM_GPUS: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
