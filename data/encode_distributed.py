"""torchrun 多卡编码 — 8-shard 存储。

用法:
    # 8 卡并行 (默认日期范围)
    torchrun --nproc_per_node=8 -m gr_demo.data.encode_distributed --model qwen3-0.6b

    # 指定日期范围
    torchrun --nproc_per_node=8 -m gr_demo.data.encode_distributed \
        --model qwen3-0.6b --date_start 2026-01-01 --date_end 2026-04-15

每个 rank 处理 hash(content_id) % world_size == rank 的数据，
结果直接写入对应的 shard_{rank}.npy，无需全量 merge。
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

NUM_SHARDS = 8  # 固定 shard 数，与 8xA100 对齐


def cid_to_shard(cid, n_shards=NUM_SHARDS) -> int:
    """Deterministic shard assignment by content_id hash."""
    return hash(str(cid)) % n_shards


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
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"[Rank 0] Model downloaded")

    if world_size > 1:
        dist.barrier()


# ============================================================
# 缓存工具
# ============================================================

def load_cached_ids(output_dir: str) -> set:
    """快速加载已缓存的 content_id 集合 (不加载 embedding)

    优先从 cached_ids.txt 加载 (最快)，否则从 shard/旧缓存文件提取 keys
    """
    cached_ids = set()

    ids_file = f"{output_dir}/cached_ids.txt"
    if os.path.exists(ids_file):
        with open(ids_file, 'r') as f:
            cached_ids = set(line.strip() for line in f if line.strip())
        return cached_ids

    # Fallback: scan shard files
    for i in range(NUM_SHARDS):
        shard_file = f"{output_dir}/shard_{i}.npy"
        if os.path.exists(shard_file):
            try:
                data = np.load(shard_file, allow_pickle=True).item()
                cached_ids.update(str(k) for k in data.keys())
            except:
                pass

    # Legacy: incremental_cache.npy
    legacy_file = f"{output_dir}/incremental_cache.npy"
    if os.path.exists(legacy_file):
        try:
            data = np.load(legacy_file, allow_pickle=True).item()
            cached_ids.update(str(k) for k in data.keys())
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
    """每个 rank 加载 hash(cid) % world_size == rank 的数据。

    两阶段加载:
    1. 只读 content_id 列，按 ID 去重，跳过全部已缓存的文件
    2. 对有未缓存行的文件，读全量数据，逐行去重 + hash 分配

    数据按 hash(content_id) % world_size 分配，保证同一 cid 始终归同一 rank/shard。
    """
    import pandas as pd
    import s3fs

    if isinstance(input_paths, str):
        input_paths = [input_paths]
    if cached_ids is None:
        cached_ids = set()

    fs = s3fs.S3FileSystem()
    files = []
    for input_path in input_paths:
        path_clean = input_path.replace('s3://', '')
        files.extend(sorted(fs.glob(f"{path_clean}/*.parquet")))

    if rank == 0:
        print(f"Found {len(files)} parquet files")

    # ── Phase 1: scan content_id column, dedup, find files with new rows ──
    files_with_new = []
    total_rows = 0
    skipped_files = 0
    seen_cids = set()        # dedup across files
    my_new_cids = set()      # uncached unique cids hashing to this rank

    for fi, f in enumerate(files):
        with fs.open(f, 'rb') as file:
            df_ids = pd.read_parquet(file, columns=['content_id'])
        total_rows += len(df_ids)
        file_has_new = False
        for cid in df_ids['content_id'].values:
            cid_str = str(cid)
            if cid_str in seen_cids:
                continue
            seen_cids.add(cid_str)
            if cid_str not in cached_ids and cid_to_shard(cid, world_size) == rank:
                my_new_cids.add(cid_str)
                file_has_new = True
        if file_has_new:
            files_with_new.append(f)
        else:
            skipped_files += 1
        if rank == 0 and ((fi + 1) % 50 == 0 or fi == len(files) - 1):
            print(f"  Phase 1: scanned {fi+1}/{len(files)} files...")

    n_unique = len(seen_cids)
    n_cached = len(seen_cids & cached_ids)
    if rank == 0:
        print(f"  Phase 1 summary: {total_rows:,} total rows, "
              f"{n_unique:,} unique content_ids")
        print(f"    Cached: {n_cached:,}, New (all ranks): {n_unique - n_cached:,}")
        print(f"    Files to read: {len(files_with_new)}/{len(files)} "
              f"(skipped {skipped_files})")
    print(f"  [Rank {rank}] New items: {len(my_new_cids):,}")

    if not files_with_new:
        return np.array([]), []

    # ── Phase 2: read files with new rows, collect this rank's data ──
    seen_cids = set(cached_ids)  # reset: treat cached as seen
    my_content_ids = []
    my_texts = []

    for f in files_with_new:
        with fs.open(f, 'rb') as file:
            df = pd.read_parquet(file)

        text_col = None
        for col in ['full_text', 'text', 'content']:
            if col in df.columns:
                text_col = col
                break
        if text_col is None:
            raise ValueError(f"No text column found. Available: {list(df.columns)}")

        for cid, text in zip(df['content_id'].values, df[text_col].fillna('').values):
            cid_str = str(cid)
            if cid_str in seen_cids:
                continue
            seen_cids.add(cid_str)
            if cid_to_shard(cid, world_size) == rank:
                my_content_ids.append(cid)
                my_texts.append(text)

    if not my_content_ids:
        return np.array([]), []

    content_ids = np.array(my_content_ids)
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

    shard_file = f"{output_dir}/shard_{rank}.npy"

    print(f"[Rank {rank}] Starting, device={device}")

    # Resolve input paths
    input_paths = _resolve_content_paths(args.date_start, args.date_end)

    if rank == 0:
        print("=" * 60)
        print(f"Multi-GPU Embedding Generation (8-shard, hash-partitioned)")
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
    cached_ids = None

    if rank == 0:
        print("Loading cached IDs...")
        t0 = time.time()
        cached_ids = load_cached_ids(output_dir)
        print(f"Loaded {len(cached_ids):,} cached IDs in {time.time()-t0:.1f}s")

    # 广播 cached_ids 给所有 ranks
    if world_size > 1:
        if rank == 0:
            cached_ids_list = list(cached_ids)
            size_tensor = torch.tensor([len(cached_ids_list)], dtype=torch.long, device=device)
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)

        dist.broadcast(size_tensor, src=0)
        size = size_tensor.item()

        if size > 0:
            if rank == 0:
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

    # Step 2: 加载数据 (hash-partitioned, 每个 rank 只处理自己 shard 的数据)
    print(f"[Rank {rank}] Loading data (hash-partitioned)...")
    my_content_ids, my_texts = load_data_shard(input_paths, rank, world_size, cached_ids=cached_ids)

    to_process = [(i, my_content_ids[i], my_texts[i]) for i in range(len(my_content_ids))]
    print(f"[Rank {rank}] To process: {len(to_process):,}")

    if len(to_process) == 0:
        print(f"[Rank {rank}] All cached, skipping")
        new_embeddings = {}
    else:
        # Rank 0 先下载模型，其他 rank 等待
        download_model_if_needed(model_name, rank, world_size)

        # 加载模型
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

            # Checkpoint
            batch_num = batch_idx // batch_size
            if batch_num % args.checkpoint_every == 0:
                np.save(f"{output_dir}/.tmp_rank{rank}.npy", new_embeddings)

            # Progress
            if batch_num % 10 == 0:
                elapsed = time.time() - start_time
                done = min(batch_idx, len(to_process))
                speed = done / elapsed if elapsed > 0 else 0
                remaining = len(to_process) - done
                eta = remaining / speed if speed > 0 else 0
                print(f"[Rank {rank}] {done:,}/{len(to_process):,} | {speed:.1f}/s | ETA: {eta/3600:.1f}h")

    # Step 3: 每个 rank 合并新数据到自己的 shard 文件
    existing_shard = {}
    if os.path.exists(shard_file):
        try:
            existing_shard = np.load(shard_file, allow_pickle=True).item()
        except:
            pass

    existing_shard.update(new_embeddings)
    np.save(shard_file, existing_shard)
    new_count = len(new_embeddings)
    total_count = len(existing_shard)
    print(f"[Rank {rank}] Shard {rank}: {total_count:,} total (+{new_count:,} new) -> {shard_file}")

    # 清理临时 checkpoint
    tmp_file = f"{output_dir}/.tmp_rank{rank}.npy"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    # 等待所有 rank 写完 shard
    if world_size > 1:
        dist.barrier()

    # Rank 0 更新 cached_ids.txt
    if rank == 0:
        print("\nUpdating cached_ids.txt...")
        all_ids = set()
        for i in range(world_size):
            sf = f"{output_dir}/shard_{i}.npy"
            if os.path.exists(sf):
                try:
                    data = np.load(sf, allow_pickle=True).item()
                    all_ids.update(str(k) for k in data.keys())
                except:
                    pass
        save_cached_ids(output_dir, all_ids)
        print(f"cached_ids.txt: {len(all_ids):,} total IDs")

    cleanup_distributed()
    print(f"[Rank {rank}] Done!")


if __name__ == '__main__':
    main()
