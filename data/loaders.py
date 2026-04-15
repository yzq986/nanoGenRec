"""所有 S3/本地数据加载函数。

合并自:
- rkmeans_stage2_train_v2.py (load_text_from_s3, load_old_embeddings_from_s3, _load_exposed_iids, export_results_to_s3)
- eval_metrics.py (load_results_from_s3, load_model_from_s3, load_local_results, load_local_model)
"""

import os
import tempfile
from typing import Any, List, Tuple

import numpy as np
import torch

from gr_demo.s3_utils import upload_to_s3


# ============================================================
# Training data loaders (from rkmeans_stage2_train_v2.py)
# ============================================================

def load_text_from_s3(s3_path: str, max_partitions: int = 0) -> Tuple[np.ndarray, List[str], List[List[str]], np.ndarray]:
    """Load text and images data from S3 parquet files

    Args:
        s3_path: S3 path to parquet files
        max_partitions: 只读取前 N 个 partition，0 表示读取全部
                        数据已按 create_date DESC 排序，所以前面的 partition 包含更新的内容
    """
    import pandas as pd
    import s3fs

    print(f"Loading data from {s3_path}...")

    fs = s3fs.S3FileSystem()
    s3_path_clean = s3_path.replace('s3://', '')
    files = fs.glob(f"{s3_path_clean}/*.parquet")
    files = sorted(files)  # 按文件名排序，保证读取顺序一致
    total_files = len(files)

    # 只读取前 N 个 partition (包含更新的内容)
    if max_partitions > 0:
        files = files[:max_partitions]
        print(f"Reading {len(files)}/{total_files} partitions (--max_partitions {max_partitions})")
    else:
        print(f"Reading all {total_files} partitions")

    dfs = []
    for i, f in enumerate(files):
        with fs.open(f, 'rb') as file:
            df = pd.read_parquet(file)
            dfs.append(df)
        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  Loaded {i + 1}/{len(files)} files...")

    combined_df = pd.concat(dfs, ignore_index=True)

    # 如果有 create_date 字段，按日期降序排序（新的在前）
    if 'create_date' in combined_df.columns:
        combined_df = combined_df.sort_values('create_date', ascending=False).reset_index(drop=True)
        print(f"Sorted by create_date DESC (newest first)")

    print(f"Combined {len(combined_df):,} rows")

    content_ids = combined_df['content_id'].values
    texts = combined_df['full_text'].tolist() if 'full_text' in combined_df.columns else None
    languages = combined_df['content_language'].values if 'content_language' in combined_df.columns else None

    # 加载图片列表 (转成 Python list，避免 numpy array 布尔值问题)
    images_list = None
    if 'images' in combined_df.columns:
        images_list = [list(imgs) if imgs is not None else [] for imgs in combined_df['images']]
        n_with_images = sum(1 for imgs in images_list if len(imgs) > 0)
        print(f"  Samples with images: {n_with_images:,} ({n_with_images/len(content_ids):.1%})")

    print(f"Loaded {len(content_ids):,} samples")
    return content_ids, texts, images_list, languages


def load_content_texts(s3_path: str = None) -> dict:
    """Load content_id -> text mapping from S3 parquet.

    Returns:
        dict: {content_id_str: text_string}
    """
    if s3_path is None:
        from gr_demo.config import S3_CONTENT_TEXT_EXPOSED, DEFAULT_DATE
        s3_path = f'{S3_CONTENT_TEXT_EXPOSED}/{DEFAULT_DATE}'

    content_ids, texts, _, _ = load_text_from_s3(s3_path)
    mapping = {str(cid): txt for cid, txt in zip(content_ids, texts) if txt}
    print(f"Content text mapping: {len(mapping):,} items with text")
    return mapping


def load_old_embeddings_from_s3(s3_path: str, max_partitions: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Load old Sentence-BERT embeddings from S3 parquet files

    Args:
        s3_path: S3 path to parquet files
        max_partitions: 只读取前 N 个 partition，0 表示读取全部
    """
    import pandas as pd
    import s3fs

    print(f"Loading OLD embeddings from {s3_path}...")

    fs = s3fs.S3FileSystem()
    s3_path_clean = s3_path.replace('s3://', '')
    files = fs.glob(f"{s3_path_clean}/*.parquet")
    total_files = len(files)

    if max_partitions > 0:
        files = files[:max_partitions]
        print(f"Reading {len(files)}/{total_files} partitions")
    else:
        print(f"Reading all {total_files} partitions")

    dfs = []
    for i, f in enumerate(files):
        with fs.open(f, 'rb') as file:
            df = pd.read_parquet(file)
            dfs.append(df)
        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  Loaded {i + 1}/{len(files)} files...")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined {len(combined_df):,} rows")

    content_ids = combined_df['content_id'].values
    embeddings = np.array(combined_df['embeddings'].tolist(), dtype=np.float32)

    print(f"Loaded {len(content_ids):,} samples, embedding dim: {embeddings.shape[1]}")
    return content_ids, embeddings


def resolve_behavior_paths(behavior_path: str) -> list:
    """将 behavior_path 解析为 S3 路径列表。
    - "auto": DEFAULT_DATE_START ~ DEFAULT_DATE_END 每日增量路径
    - 具体 S3 路径: 原样返回
    """
    if behavior_path == "auto":
        from gr_demo.config import S3_USER_BEHAVIOR, DEFAULT_DATE_START, DEFAULT_DATE_END
        from datetime import datetime, timedelta
        start = datetime.strptime(DEFAULT_DATE_START, "%Y-%m-%d")
        end = datetime.strptime(DEFAULT_DATE_END, "%Y-%m-%d")
        paths = []
        d = start
        while d <= end:
            paths.append(f"{S3_USER_BEHAVIOR}/{d.strftime('%Y-%m-%d')}")
            d += timedelta(days=1)
        print(f"  Resolved behavior_path='auto' → {len(paths)} days ({DEFAULT_DATE_START} ~ {DEFAULT_DATE_END})")
        return paths
    return [behavior_path]


def load_exposed_iids(behavior_path: str) -> set:
    """从 behavior parquet 加载去重后的曝光 iid 集合（支持 'auto' 和 S3 路径）"""
    import pandas as pd
    import s3fs

    paths = resolve_behavior_paths(behavior_path)
    fs = s3fs.S3FileSystem()

    files = []
    for bp in paths:
        path_clean = bp.replace('s3://', '')
        files.extend(fs.glob(f"{path_clean}/*.parquet"))
    print(f"  Found {len(files)} behavior files")

    iid_set = set()
    for i, f in enumerate(files):
        with fs.open(f, 'rb') as file:
            df = pd.read_parquet(file, columns=['iid'])
        iid_set.update(df['iid'].dropna().astype(str).values)
        if (i + 1) % 5 == 0 or i == len(files) - 1:
            print(f"  Loaded {i + 1}/{len(files)} files, unique iids so far: {len(iid_set):,}")

    return iid_set


def export_results_to_s3(
    content_ids: np.ndarray,
    semantic_ids: List[str],
    embeddings: np.ndarray,
    output_path: str,
    embedding_type: str
):
    """导出 content_id, semantic_id, embedding 到 S3 (parquet 格式)"""
    import pandas as pd

    print(f"Exporting results to S3...")

    # 构建 DataFrame
    df = pd.DataFrame({
        'content_id': content_ids,
        'semantic_id': semantic_ids,
        'embedding': list(embeddings)  # 每行是一个 embedding 向量
    })

    # 保存到本地临时文件
    local_path = f'/tmp/results_{embedding_type}.parquet'
    df.to_parquet(local_path, index=False)
    print(f"  Saved to {local_path}, size: {os.path.getsize(local_path) / 1024 / 1024:.1f} MB")

    # 上传到 S3
    upload_to_s3(local_path, f"{output_path}/results_{embedding_type}.parquet")

    # 统计 semantic_id 分布
    unique_sids = df['semantic_id'].nunique()
    print(f"  Unique semantic_ids: {unique_sids:,}")

    # 显示 top semantic_ids
    top_sids = df['semantic_id'].value_counts().head(10)
    print(f"  Top 10 semantic_ids:")
    for sid, count in top_sids.items():
        print(f"    {sid}: {count:,}")


# ============================================================
# Eval data loaders (from eval_metrics.py)
# ============================================================

def load_results_from_s3(s3_path: str, sample_size: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load results parquet from S3

    Args:
        s3_path: S3 path to results parquet file
        sample_size: If > 0, randomly sample this many rows

    Returns:
        content_ids, embeddings, semantic_ids
    """
    import pandas as pd
    import s3fs

    print(f"Loading results from {s3_path}...")

    fs = s3fs.S3FileSystem()
    s3_path_clean = s3_path.replace('s3://', '')

    with fs.open(s3_path_clean, 'rb') as f:
        df = pd.read_parquet(f)

    print(f"Loaded {len(df):,} rows")

    # Sample if requested
    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled to {len(df):,} rows")

    content_ids = df['content_id'].values
    semantic_ids = df['semantic_id'].tolist()

    # Load embeddings
    if 'embedding' in df.columns:
        embeddings = np.array(df['embedding'].tolist(), dtype=np.float32)
    else:
        embeddings = None
        print("Warning: No 'embedding' column found")

    print(f"Embeddings shape: {embeddings.shape if embeddings is not None else 'N/A'}")
    print(f"Unique semantic IDs: {len(set(semantic_ids)):,}")

    return content_ids, embeddings, semantic_ids


def load_model_from_s3(s3_path: str) -> Any:
    """Load RKMeans model from S3

    Args:
        s3_path: S3 path to .pt model file

    Returns:
        Loaded model data dict
    """
    import boto3

    print(f"Loading model from {s3_path}...")

    # Download to temp file
    s3 = boto3.client('s3')
    s3_path_clean = s3_path.replace('s3://', '')
    bucket = s3_path_clean.split('/')[0]
    key = '/'.join(s3_path_clean.split('/')[1:])

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
        s3.download_file(bucket, key, temp_path)

    model_data = torch.load(temp_path, map_location='cpu')
    os.unlink(temp_path)

    print(f"Model loaded: {model_data.get('n_layers')} layers x {model_data.get('n_clusters')} clusters")
    return model_data


def load_local_results(path: str, sample_size: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load results from local parquet file"""
    import pandas as pd

    print(f"Loading results from {path}...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows")

    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled to {len(df):,} rows")

    content_ids = df['content_id'].values
    semantic_ids = df['semantic_id'].tolist()

    if 'embedding' in df.columns:
        embeddings = np.array(df['embedding'].tolist(), dtype=np.float32)
    else:
        embeddings = None

    return content_ids, embeddings, semantic_ids


def load_local_model(path: str) -> Any:
    """Load RKMeans model from local file"""
    print(f"Loading model from {path}...")
    model_data = torch.load(path, map_location='cpu')
    print(f"Model loaded: {model_data.get('n_layers')} layers x {model_data.get('n_clusters')} clusters")
    return model_data
