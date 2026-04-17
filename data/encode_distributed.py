"""torchrun 多卡编码（原 encode_multiprocess.py）。

用法:
    # 8 卡并行
    torchrun --nproc_per_node=8 -m gr_demo.data.encode_distributed --model qwen3-0.6b

    # 指定卡数
    torchrun --nproc_per_node=4 -m gr_demo.data.encode_distributed --model qwen3-0.6b

每个进程处理 1/N 的数据，结果保存到各自的缓存文件，最后合并。
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
from typing import List
from transformers import AutoModel, AutoTokenizer

from gr_demo.model.embedders import Qwen3TextEmbedder
from gr_demo.config import S3_CONTENT_TEXT_EXPOSED, EFS_EMBEDDING_CACHE
from gr_demo.config import DEFAULT_DATE, DEFAULT_DATE_START, DEFAULT_DATE_END


# ============================================================
# 分布式工具
# ============================================================

def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
# Model Download
# ============================================================

def download_model_if_needed(model_name: str, rank: int, world_size: int):
    """Rank 0 先下载模型，其他 rank 等待"""
    if rank == 0:
        print(f"[Rank 0] Downloading model {model_name}...")
        # 触发下载
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"[Rank 0] Model downloaded")

    # 等待 rank 0 下载完成
    if world_size > 1:
        dist.barrier()


# ============================================================
# 缓存工具
# ============================================================

def load_cached_ids(output_dir: str, world_size: int) -> set:
    """
    快速加载已缓存的 content_id 集合 (不加载 embedding)

    优先从 cached_ids.txt 加载 (最快)，否则从 .npy 文件提取 keys
    """
    cached_ids = set()

    # 方案 1: 从轻量索引文件加载 (最快)
    ids_file = f"{output_dir}/cached_ids.txt"
    if os.path.exists(ids_file):
        with open(ids_file, 'r') as f:
            cached_ids = set(line.strip() for line in f if line.strip())
        return cached_ids

    # 方案 2: 从 .npy 文件提取 keys (较慢，但兼容旧缓存)
    merged_cache_file = f"{output_dir}/incremental_cache.npy"
    if os.path.exists(merged_cache_file):
        try:
            data = np.load(merged_cache_file, allow_pickle=True).item()
            cached_ids.update(data.keys())
        except:
            pass

    for r in range(world_size):
        shard_file = f"{output_dir}/cache_rank{r}.npy"
        if os.path.exists(shard_file):
            try:
                data = np.load(shard_file, allow_pickle=True).item()
                cached_ids.update(data.keys())
            except:
                pass

    return cached_ids


def save_cached_ids(output_dir: str, cached_ids: set):
    """保存已缓存的 content_id 集合到轻量索引文件"""
    ids_file = f"{output_dir}/cached_ids.txt"
    with open(ids_file, 'w') as f:
        for cid in cached_ids:
            f.write(f"{cid}\n")


# ============================================================
# 数据加载
# ============================================================

def load_data_shard(input_paths, rank: int, world_size: int, cached_ids: set = None):
    """
    每个 rank 加载自己的数据分片 (Row-level 均匀分配)

    Args:
        input_paths: str 或 list[str]，S3 路径（支持单路径或多天路径列表）

    策略: 所有 rank 读取全部文件，但只保留 row_idx % world_size == rank 的行
    这样保证每个 rank 处理的数据量完全均匀

    如果提供 cached_ids (set)，还会进一步做 cache-aware 负载均衡:
    - 先过滤掉已缓存的 content_id
    - 再对剩余待处理的数据做均匀分配
    """
    import pandas as pd
    import s3fs

    if isinstance(input_paths, str):
        input_paths = [input_paths]

    fs = s3fs.S3FileSystem()
    files = []
    for input_path in input_paths:
        path_clean = input_path.replace('s3://', '')
        files.extend(sorted(fs.glob(f"{path_clean}/*.parquet")))

    if rank == 0:
        print(f"Found {len(files)} parquet files")

    # 每个 rank 读取全部文件
    dfs = []
    for f in files:
        with fs.open(f, 'rb') as file:
            dfs.append(pd.read_parquet(file))

    if not dfs:
        return np.array([]), []

    df = pd.concat(dfs, ignore_index=True)
    total_rows = len(df)

    if rank == 0:
        print(f"Total rows: {total_rows:,}")

    # 尝试不同的文本列名
    text_col = None
    for col in ['full_text', 'text', 'content']:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(f"No text column found. Available: {list(df.columns)}")

    content_ids_all = df['content_id'].values
    texts_all = df[text_col].fillna('').values

    # Cache-aware 负载均衡: 只对未缓存的数据做均匀分配
    if cached_ids is not None and len(cached_ids) > 0:
        # Vectorized: 用 numpy 批量判断 (比 Python for loop 快 10x+)
        is_cached = np.array([cid in cached_ids for cid in content_ids_all])
        uncached_indices = np.where(~is_cached)[0]
        n_uncached = len(uncached_indices)

        if rank == 0:
            print(f"Uncached rows: {n_uncached:,} ({100*n_uncached/total_rows:.1f}%)")

        # 对未缓存的数据做均匀分配
        my_mask = np.arange(n_uncached) % world_size == rank
        my_indices = uncached_indices[my_mask]
    else:
        # 普通 row-level 分配
        my_indices = np.arange(total_rows)[np.arange(total_rows) % world_size == rank]

    content_ids = content_ids_all[my_indices]
    texts = texts_all[my_indices].tolist()

    print(f"[Rank {rank}] Got {len(content_ids):,} rows to process")

    return content_ids, texts


# ============================================================
# Text-only Model Configs (分布式场景)
# ============================================================

DISTRIBUTED_MODEL_CONFIGS = {
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", 1024, 64),
    "qwen3-4b": ("Qwen/Qwen3-Embedding-4B", 2560, 32),
    "qwen3-8b": ("Qwen/Qwen3-Embedding-8B", 4096, 16),
}


# ============================================================
# Main
# ============================================================

def _resolve_content_paths(date_start, date_end):
    """Resolve date range to S3 content paths (one per day)."""
    from datetime import datetime, timedelta
    ds = date_start or DEFAULT_DATE_START
    de = date_end or DEFAULT_DATE_END
    paths = []
    d = datetime.strptime(ds, "%Y-%m-%d")
    end = datetime.strptime(de, "%Y-%m-%d")
    while d <= end:
        paths.append(f"{S3_CONTENT_TEXT_EXPOSED}/{d.strftime('%Y-%m-%d')}")
        d += timedelta(days=1)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-0.6b', choices=DISTRIBUTED_MODEL_CONFIGS.keys())
    parser.add_argument('--date_start', type=str, default=None,
                        help=f'Content date range start (default: {DEFAULT_DATE_START})')
    parser.add_argument('--date_end', type=str, default=None,
                        help=f'Content date range end (default: {DEFAULT_DATE_END})')
    parser.add_argument('--output_dir', type=str, default=EFS_EMBEDDING_CACHE)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--checkpoint_every', type=int, default=100)
    args = parser.parse_args()

    # 分布式设置
    rank, world_size, local_rank = setup_distributed()
    device = f'cuda:{local_rank}'

    model_name, embedding_dim, default_batch_size = DISTRIBUTED_MODEL_CONFIGS[args.model]
    batch_size = args.batch_size or default_batch_size

    output_dir = f"{args.output_dir}/{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Rank {rank}] Starting, device={device}")

    # Resolve input paths
    input_paths = _resolve_content_paths(args.date_start, args.date_end)

    if rank == 0:
        print("=" * 60)
        print(f"Multi-GPU Embedding Generation (Cache-aware Load Balancing)")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Input: {len(input_paths)} path(s)")
        if len(input_paths) > 1:
            print(f"  {input_paths[0]} ~ {input_paths[-1]}")
        else:
            print(f"  {input_paths[0]}")
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Output: {output_dir}")
        print("=" * 60)

    # Step 1: Rank 0 加载 cached_ids 并广播给其他 ranks
    cache_file = f"{output_dir}/cache_rank{rank}.npy"
    cached_ids = None

    if rank == 0:
        print("Loading cached IDs...")
        t0 = time.time()
        cached_ids = load_cached_ids(output_dir, world_size)
        print(f"Loaded {len(cached_ids):,} cached IDs in {time.time()-t0:.1f}s")

    # 广播 cached_ids 给所有 ranks
    if world_size > 1:
        if rank == 0:
            # Rank 0 序列化并广播
            cached_ids_list = list(cached_ids)
            size_tensor = torch.tensor([len(cached_ids_list)], dtype=torch.long, device=device)
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)

        dist.broadcast(size_tensor, src=0)
        size = size_tensor.item()

        if size > 0:
            if rank == 0:
                # 将 string ids 编码为 bytes 并广播
                import pickle
                data_bytes = pickle.dumps(cached_ids_list)
                data_tensor = torch.ByteTensor(list(data_bytes)).to(device)
                len_tensor = torch.tensor([len(data_bytes)], dtype=torch.long, device=device)
            else:
                len_tensor = torch.tensor([0], dtype=torch.long, device=device)

            dist.broadcast(len_tensor, src=0)
            data_len = len_tensor.item()

            if rank != 0:
                data_tensor = torch.empty(data_len, dtype=torch.uint8, device=device)

            dist.broadcast(data_tensor, src=0)

            if rank != 0:
                import pickle
                cached_ids_list = pickle.loads(bytes(data_tensor.cpu().tolist()))
                cached_ids = set(cached_ids_list)
        else:
            cached_ids = set()
    else:
        if cached_ids is None:
            cached_ids = set()

    if rank == 0:
        print(f"All ranks have {len(cached_ids):,} cached IDs")

    # Step 2: 用 cache-aware 方式加载数据分片 (只分配未缓存的数据)
    print(f"[Rank {rank}] Loading data with cache-aware balancing...")
    my_content_ids, my_texts = load_data_shard(input_paths, rank, world_size, cached_ids=cached_ids)

    # 此时 my_content_ids 已经是未缓存的数据，直接作为 to_process
    to_process = [(i, my_content_ids[i], my_texts[i]) for i in range(len(my_content_ids))]
    print(f"[Rank {rank}] To process: {len(to_process):,}")

    if len(to_process) == 0:
        print(f"[Rank {rank}] All cached, skipping")
    else:
        # Rank 0 先下载模型，其他 rank 等待
        download_model_if_needed(model_name, rank, world_size)

        # 加载模型到各自的 GPU（使用统一的 Qwen3TextEmbedder，分布式模式传 device）
        print(f"[Rank {rank}] Loading model on {device}...")
        embedder = Qwen3TextEmbedder(model_name, device=device)

        # 编码
        new_embeddings = {}
        start_time = time.time()

        batch_idx = 0
        while batch_idx < len(to_process):
            batch = to_process[batch_idx:batch_idx + batch_size]
            batch_ids = [b[1] for b in batch]
            batch_texts = [b[2] for b in batch]

            # OOM 重试逻辑
            retry_size = len(batch_texts)
            success = False

            while retry_size >= 1 and not success:
                try:
                    chunk_embs = []
                    for i in range(0, len(batch_texts), retry_size):
                        chunk = batch_texts[i:i + retry_size]
                        emb = embedder.encode(chunk)
                        chunk_embs.append(emb.cpu().float().numpy())

                    emb_np = np.concatenate(chunk_embs, axis=0)
                    for cid, e in zip(batch_ids, emb_np):
                        new_embeddings[cid] = e
                    success = True

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    retry_size = retry_size // 2
                    if retry_size >= 1:
                        print(f"[Rank {rank}] OOM, retry with chunk_size={retry_size}")

            if not success:
                print(f"[Rank {rank}] Failed batch {batch_idx}, chunk_size too small")

            batch_idx += batch_size

            # Checkpoint (只保存本次新生成的，避免重复写入巨大的 cached)
            batch_num = batch_idx // batch_size
            if batch_num % args.checkpoint_every == 0:
                np.save(cache_file, new_embeddings)

            # Progress
            if batch_num % 10 == 0:
                elapsed = time.time() - start_time
                done = min(batch_idx, len(to_process))
                speed = done / elapsed if elapsed > 0 else 0
                remaining = len(to_process) - done
                eta = remaining / speed if speed > 0 else 0
                print(f"[Rank {rank}] {done:,}/{len(to_process):,} | {speed:.1f}/s | ETA: {eta/3600:.1f}h")

        # 保存本次新生成的结果
        np.save(cache_file, new_embeddings)
        print(f"[Rank {rank}] Saved {len(new_embeddings):,} new embeddings to {cache_file}")

    # 等待所有进程完成
    if world_size > 1:
        dist.barrier()

    # Rank 0 合并结果
    if rank == 0:
        print("\nMerging results from all ranks...")

        # 先加载已有的合并缓存
        merged_cache = f"{output_dir}/incremental_cache.npy"
        all_embeddings = {}
        if os.path.exists(merged_cache):
            try:
                all_embeddings = np.load(merged_cache, allow_pickle=True).item()
                print(f"  Existing merged cache: {len(all_embeddings):,} embeddings")
            except:
                pass

        # 再合并所有 rank 的新缓存
        new_count = 0
        for r in range(world_size):
            cache_file_r = f"{output_dir}/cache_rank{r}.npy"
            if os.path.exists(cache_file_r):
                try:
                    data = np.load(cache_file_r, allow_pickle=True).item()
                    all_embeddings.update(data)
                    new_count += len(data)
                    print(f"  Rank {r}: {len(data):,} new embeddings")
                except:
                    pass

        # 保存合并后的缓存
        np.save(merged_cache, all_embeddings)
        print(f"Merged total: {len(all_embeddings):,} embeddings (new: {new_count:,})")

        # 保存轻量索引文件 (下次启动时快速加载)
        save_cached_ids(output_dir, set(all_embeddings.keys()))
        print(f"Saved cached_ids.txt for fast loading")

        # 不再生成完整数组，让 rkmeans 脚本用 incremental_cache 加载
        print("Note: Use --skip_embedding with 'python -m gr_demo train' to load from cache")

    cleanup_distributed()
    print(f"[Rank {rank}] Done!")


if __name__ == '__main__':
    main()
