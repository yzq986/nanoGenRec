"""torchrun 多卡编码 — 8-shard 存储，逐日期处理。

用法:
    # 8 卡并行 (默认日期范围)
    torchrun --nproc_per_node=8 -m gr_demo.data.encode_distributed --model qwen3-0.6b

    # 指定日期范围
    torchrun --nproc_per_node=8 -m gr_demo.data.encode_distributed \
        --model qwen3-0.6b --date_start 2026-01-01 --date_end 2026-04-15

逐日期从新到旧处理。每个日期内:
  1. 逐文件读 content_id，按 ID 级别判断是否已缓存，跳过已缓存的 cid
  2. hash(content_id) % world_size 分配到对应 rank
  3. 编码后每个 rank 追加到自己的 shard_{rank}.npy
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer

from gr_demo.model.embedders import Qwen3TextEmbedder
from gr_demo.config import S3_CONTENT_TEXT_EXPOSED, EFS_EMBEDDING_CACHE
from gr_demo.config import DEFAULT_DATE_START, DEFAULT_DATE_END

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
    ids_file = f"{output_dir}/cached_ids.txt"
    if os.path.exists(ids_file):
        with open(ids_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())

    cached_ids = set()
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
# 单日期数据加载
# ============================================================

def load_date_data(date_path, rank, world_size, cached_ids):
    """加载单个日期的数据，返回本 rank 需要编码的 (content_ids, texts)。

    逐文件读取，按 content_id 级别去重 + 缓存判断:
    - cid 已在 cached_ids 中 → 跳过
    - cid 不属于本 rank (hash % world_size != rank) → 跳过
    """
    import pandas as pd
    import s3fs

    fs = s3fs.S3FileSystem()
    path_clean = date_path.replace('s3://', '')
    files = sorted(fs.glob(f"{path_clean}/*.parquet"))

    if not files:
        return np.array([]), []

    my_content_ids = []
    my_texts = []
    total_rows = 0
    n_cached = 0

    for f in files:
        with fs.open(f, 'rb') as file:
            df = pd.read_parquet(file)
        total_rows += len(df)

        text_col = None
        for col in ['full_text', 'text', 'content']:
            if col in df.columns:
                text_col = col
                break
        if text_col is None:
            raise ValueError(f"No text column found. Available: {list(df.columns)}")

        for cid, text in zip(df['content_id'].values, df[text_col].fillna('').values):
            if str(cid) in cached_ids:
                n_cached += 1
                continue
            if cid_to_shard(cid, world_size) == rank:
                my_content_ids.append(cid)
                my_texts.append(text)

    n_new = len(my_content_ids)
    if rank == 0:
        print(f"    {total_rows:,} rows, {n_cached:,} cached, "
              f"{total_rows - n_cached:,} new")
    if n_new > 0:
        print(f"    [Rank {rank}] {n_new:,} items to encode")

    if not my_content_ids:
        return np.array([]), []

    return np.array(my_content_ids), my_texts


# ============================================================
# 编码
# ============================================================

TEXT_CACHE_MAX_LEN = 256   # 短于此长度的文本缓存 text→embedding
TEXT_CACHE_MAX_SIZE = 500_000  # LRU 最大条目数


class LRUTextCache:
    """简单 LRU: OrderedDict, 超过 max_size 淘汰最久未用的。"""

    def __init__(self, max_size=TEXT_CACHE_MAX_SIZE):
        from collections import OrderedDict
        self._cache = OrderedDict()
        self._max_size = max_size
        self.hits = 0

    def get(self, text):
        if text in self._cache:
            self._cache.move_to_end(text)
            self.hits += 1
            return self._cache[text]
        return None

    def put(self, text, embedding):
        if text in self._cache:
            self._cache.move_to_end(text)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[text] = embedding

    def __len__(self):
        return len(self._cache)


def encode_batch(embedder, content_ids, texts, batch_size, rank, text_cache=None):
    """编码一批文本，返回 {cid: embedding} dict。

    text_cache: LRUTextCache, 短文本 → embedding 缓存 (跨日期复用)。
    相同短文本直接复用 embedding，不重复过模型。
    """
    if text_cache is None:
        text_cache = LRUTextCache()

    new_embeddings = {}
    start_time = time.time()

    # 分离: 可以从缓存命中的 vs 需要过模型的
    to_encode_idx = []
    n_text_hits = 0
    for i, (cid, text) in enumerate(zip(content_ids, texts)):
        cached_emb = text_cache.get(text) if len(text) <= TEXT_CACHE_MAX_LEN else None
        if cached_emb is not None:
            new_embeddings[cid] = cached_emb
            n_text_hits += 1
        else:
            to_encode_idx.append(i)

    if n_text_hits > 0:
        print(f"    [Rank {rank}] Text cache hit: {n_text_hits:,}, "
              f"to encode: {len(to_encode_idx):,}, "
              f"cache size: {len(text_cache):,}")

    # 编码需要过模型的
    batch_idx = 0
    total = len(to_encode_idx)
    while batch_idx < total:
        batch_indices = to_encode_idx[batch_idx:batch_idx + batch_size]
        batch_cids = [content_ids[i] for i in batch_indices]
        batch_texts = [texts[i] for i in batch_indices]

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
                for cid, text, e in zip(batch_cids, batch_texts, emb_np):
                    new_embeddings[cid] = e
                    if len(text) <= TEXT_CACHE_MAX_LEN:
                        text_cache.put(text, e)
                success = True

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                retry_size = retry_size // 2
                if retry_size >= 1:
                    print(f"    [Rank {rank}] OOM, retry chunk_size={retry_size}")

        if not success:
            print(f"    [Rank {rank}] Failed batch {batch_idx}")

        batch_idx += batch_size

        # Progress
        batch_num = batch_idx // batch_size
        if batch_num % 10 == 0:
            elapsed = time.time() - start_time
            done = min(batch_idx, total)
            speed = done / elapsed if elapsed > 0 else 0
            remaining = total - done
            eta = remaining / speed if speed > 0 else 0
            print(f"    [Rank {rank}] {done:,}/{total:,} | {speed:.1f}/s | ETA: {eta/60:.1f}min")

    return new_embeddings


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

def _resolve_dates(date_start, date_end):
    """Resolve date range to list of date strings, newest first."""
    from datetime import datetime, timedelta
    ds = date_start or DEFAULT_DATE_START
    de = date_end or DEFAULT_DATE_END
    dates = []
    d = datetime.strptime(ds, "%Y-%m-%d")
    end = datetime.strptime(de, "%Y-%m-%d")
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    dates.reverse()  # 新日期优先
    return dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-0.6b', choices=DISTRIBUTED_MODEL_CONFIGS.keys())
    parser.add_argument('--date_start', type=str, default=None,
                        help=f'Content date range start (default: {DEFAULT_DATE_START})')
    parser.add_argument('--date_end', type=str, default=None,
                        help=f'Content date range end (default: {DEFAULT_DATE_END})')
    parser.add_argument('--output_dir', type=str, default=EFS_EMBEDDING_CACHE)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    # 分布式设置
    rank, world_size, local_rank = setup_distributed()
    device = f'cuda:{local_rank}'

    model_name, embedding_dim, default_batch_size = DISTRIBUTED_MODEL_CONFIGS[args.model]
    batch_size = args.batch_size or default_batch_size

    output_dir = f"{args.output_dir}/{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    shard_file = f"{output_dir}/shard_{rank}.npy"

    # Resolve dates (newest first)
    dates = _resolve_dates(args.date_start, args.date_end)

    if rank == 0:
        print("=" * 60)
        print(f"Multi-GPU Embedding Generation (8-shard, per-date)")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Dates: {dates[0]} ~ {dates[-1]} ({len(dates)} days, newest first)")
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Output: {output_dir}")
        print("=" * 60)

    # 加载 cached_ids (Rank 0 加载，广播给其他 ranks)
    cached_ids = None
    if rank == 0:
        print("Loading cached IDs...")
        t0 = time.time()
        cached_ids = load_cached_ids(output_dir)
        print(f"Loaded {len(cached_ids):,} cached IDs in {time.time()-t0:.1f}s")

    if world_size > 1:
        if rank == 0:
            import pickle
            cached_ids_list = list(cached_ids)
            data_bytes = pickle.dumps(cached_ids_list)
            len_tensor = torch.tensor([len(data_bytes)], dtype=torch.long, device=device)
        else:
            len_tensor = torch.tensor([0], dtype=torch.long, device=device)

        dist.broadcast(len_tensor, src=0)
        data_len = len_tensor.item()

        if data_len > 0:
            if rank == 0:
                data_tensor = torch.ByteTensor(list(data_bytes)).to(device)
            else:
                data_tensor = torch.empty(data_len, dtype=torch.uint8, device=device)

            dist.broadcast(data_tensor, src=0)

            if rank != 0:
                import pickle
                cached_ids = set(pickle.loads(bytes(data_tensor.cpu().tolist())))
        else:
            if rank != 0:
                cached_ids = set()
    else:
        if cached_ids is None:
            cached_ids = set()

    if rank == 0:
        print(f"All ranks have {len(cached_ids):,} cached IDs\n")

    # 加载已有 shard
    existing_shard = {}
    if os.path.exists(shard_file):
        try:
            existing_shard = np.load(shard_file, allow_pickle=True).item()
        except:
            pass

    embedder = None  # 惰性加载模型
    text_cache = LRUTextCache()  # 短文本 → embedding LRU 缓存 (跨日期复用)
    total_new = 0

    # ── 逐日期处理 (新 → 旧) ──
    for di, date_str in enumerate(dates):
        date_path = f"{S3_CONTENT_TEXT_EXPOSED}/{date_str}"
        if rank == 0:
            print(f"[{di+1}/{len(dates)}] {date_str}")

        # 加载本日期数据
        my_cids, my_texts = load_date_data(date_path, rank, world_size, cached_ids)

        if len(my_cids) == 0:
            if rank == 0:
                print(f"    All cached, skipping")
            # 所有 rank 需要同步，避免有些 rank 有数据有些没有导致 barrier 死锁
            if world_size > 1:
                dist.barrier()
            continue

        # 惰性加载模型 (第一次遇到有数据时)
        if embedder is None:
            download_model_if_needed(model_name, rank, world_size)
            print(f"  [Rank {rank}] Loading model on {device}...")
            embedder = Qwen3TextEmbedder(model_name, device=device)

        # 编码 (text_cache 跨日期复用，相同短文本不重复过模型)
        new_embeddings = encode_batch(
            embedder, my_cids, my_texts, batch_size, rank, text_cache=text_cache)

        # 追加到 shard
        existing_shard.update(new_embeddings)
        # 更新 cached_ids (这样后续日期的重复 cid 会被跳过)
        cached_ids.update(str(cid) for cid in new_embeddings.keys())

        n_new = len(new_embeddings)
        total_new += n_new
        print(f"    [Rank {rank}] +{n_new:,} embeddings")

        # 每个日期处理完后保存 shard (断点续跑)
        np.save(shard_file, existing_shard)

        if world_size > 1:
            dist.barrier()

    # ── 最终汇总 ──
    print(f"\n[Rank {rank}] Shard {rank}: {len(existing_shard):,} total "
          f"(+{total_new:,} new) -> {shard_file}")

    if world_size > 1:
        dist.barrier()

    # Rank 0 更新 cached_ids.txt
    if rank == 0:
        print("Updating cached_ids.txt...")
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
