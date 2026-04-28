"""torchrun 多卡编码 — 8-shard 存储，逐日期处理。

用法:
    # 纯文本 (默认日期范围)
    torchrun --nproc_per_node=8 -m data.encode_distributed --model qwen3-0.6b

    # 图文 VL (qwen3-vl-2b)
    torchrun --nproc_per_node=8 -m data.encode_distributed --model qwen3-vl-2b

    # 指定日期范围
    torchrun --nproc_per_node=8 -m data.encode_distributed \
        --model qwen3-0.6b --date_start 2026-01-01 --date_end 2026-04-15

逐日期从新到旧处理。每个日期内:
  1. 逐文件读 content_id，按 ID 级别判断是否已缓存，跳过已缓存的 cid
  2. sha256(content_id) % world_size 分配到对应 rank
  3. 编码后每个 rank 追加到自己的 shard_{rank}.npy

VL 模型额外加载 parquet 的 images 列 (ARRAY<STRING>)，每条最多取
VL_MAX_IMAGES 张；下载失败的 URL 会重试 VL_IMAGE_RETRIES 次。
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer

from model.embedders import Qwen3TextEmbedder, Qwen3VLEmbedder
from config import S3_CONTENT_TEXT_EXPOSED, S3_CONTENT_TEXT_EXPOSED_S3, EFS_EMBEDDING_CACHE, EFS_IMAGE_CACHE
from config import DEFAULT_DATE_START, DEFAULT_DATE_END

NUM_SHARDS = 8  # 固定 shard 数，与 8xA100 对齐

# VL 编码默认参数
# QWEN_MAX_LENGTH=8192 是整个序列 (text+vision) 的 budget。
# 1:1 split → vision 4096 tokens + text 4096 tokens。
# vision tokens = pixels / IMAGE_FACTOR² (=1024)
# max_images=1 → per-image 4096 tokens → max_pixels = 4096 * 1024 = 4,194,304
VL_MAX_IMAGES = 1
VL_MAX_PIXELS = 4_194_304  # ≈ 2048x2048, 4096 vision tokens 上限
VL_IMAGE_TIMEOUT = 3
VL_IMAGE_RETRIES = 2
VL_IMAGE_WORKERS = 16


def cid_to_shard(cid, n_shards=NUM_SHARDS) -> int:
    """Deterministic shard assignment by content_id hash.

    Uses hashlib instead of hash() because Python randomizes hash() seeds
    across processes (PYTHONHASHSEED), which breaks distributed shard routing.
    """
    import hashlib
    return int(hashlib.sha256(str(cid).encode()).hexdigest(), 16) % n_shards


# ============================================================
# 分布式工具
# ============================================================

def setup_distributed():
    """初始化分布式环境。

    NCCL watchdog 默认 10min 超时会在 rank 0 慢操作 (下模型/读大量 parquet)
    时把其他 rank 炸掉，放宽到 30min。
    """
    from datetime import timedelta

    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
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

def download_model_if_needed(model_name: str, rank: int, world_size: int, is_vl: bool = False):
    """Rank 0 先下载模型，其他 rank 等待"""
    if rank == 0:
        print(f"[Rank 0] Downloading model {model_name}...")
        if is_vl:
            from huggingface_hub import snapshot_download
            snapshot_download(model_name)
        else:
            AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"[Rank 0] Model downloaded")

    if world_size > 1:
        dist.barrier()


# ============================================================
# 图片下载 (带重试)
# ============================================================

def _download_one_image(url: str, timeout: int, cache_dir: str, s3_client=None):
    """下载单张图片，带磁盘缓存。支持 s3:// 和 http(s):// 两种路径。"""
    import hashlib
    from io import BytesIO
    from PIL import Image

    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{url_hash}.jpg")
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert('RGB')
        except Exception:
            pass

    try:
        if url.startswith('s3://'):
            from urllib.parse import urlparse
            parsed = urlparse(url)
            client = s3_client or __import__('boto3').client('s3')
            resp = client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip('/'))
            data = resp['Body'].read()
        else:
            import requests
            resp = requests.get(url, timeout=(timeout, timeout))
            resp.raise_for_status()
            data = resp.content
        img = Image.open(BytesIO(data)).convert('RGB')
        try:
            img.save(cache_path, 'JPEG', quality=85)
        except Exception:
            pass
        return img
    except Exception:
        return None


def download_images_with_retry(
    urls,
    cache_dir: str,
    max_workers: int = VL_IMAGE_WORKERS,
    timeout: int = VL_IMAGE_TIMEOUT,
    max_retries: int = VL_IMAGE_RETRIES,
):
    """并发下载 URL 列表，失败的 URL 在后续 attempt 中重试。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(cache_dir, exist_ok=True)
    results = [None] * len(urls)
    pending = [i for i, u in enumerate(urls) if u]

    # boto3 low-level client is thread-safe
    has_s3 = any(u and u.startswith('s3://') for u in urls)
    s3_client = __import__('boto3').client('s3') if has_s3 else None

    for attempt in range(max_retries + 1):
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as pool:
            futures = {pool.submit(_download_one_image, urls[i], timeout, cache_dir, s3_client): i
                       for i in pending}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    img = fut.result()
                except Exception:
                    img = None
                if img is not None:
                    results[i] = img
        pending = [i for i in pending if results[i] is None]

    return results


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

def load_date_data(date_path, rank, world_size, cached_ids, load_images: bool = False):
    """加载单个日期的数据，返回本 rank 需要编码的 (content_ids, texts, images)。

    逐文件读取，按 content_id 级别去重 + 缓存判断:
    - cid 已在 cached_ids 中 → 跳过
    - cid 不属于本 rank (hash % world_size != rank) → 跳过

    load_images=True 时同时返回 images (List[List[str]])，否则为空列表。
    """
    import pandas as pd
    import s3fs

    fs = s3fs.S3FileSystem()
    path_clean = date_path.replace('s3://', '')
    files = sorted(fs.glob(f"{path_clean}/*.parquet"))

    if not files:
        return np.array([]), [], []

    my_content_ids = []
    my_texts = []
    my_images = []
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

        # image column: 优先 image_s3_path (S3 缓存), 回退原始 URL
        image_col = None
        if load_images:
            for col in ['image_s3_path', 'images', 'image', 'image_url', 'image_urls']:
                if col in df.columns:
                    image_col = col
                    break
        images_iter = df[image_col].values if image_col else [None] * len(df)

        for cid, text, imgs in zip(df['content_id'].values,
                                   df[text_col].fillna('').values,
                                   images_iter):
            if str(cid) in cached_ids:
                n_cached += 1
                continue
            if cid_to_shard(cid) % world_size == rank:
                my_content_ids.append(cid)
                my_texts.append(text)
                if load_images:
                    # 统一为 list[str]: None/空 → []; string → [string]; array → list
                    if imgs is None:
                        my_images.append([])
                    elif isinstance(imgs, str):
                        my_images.append([imgs] if imgs else [])
                    else:
                        try:
                            my_images.append([u for u in list(imgs) if u])
                        except TypeError:
                            my_images.append([])

    n_new = len(my_content_ids)
    if rank == 0:
        print(f"    {total_rows:,} rows, {n_cached:,} cached, "
              f"{total_rows - n_cached:,} new")
    if n_new > 0:
        print(f"    [Rank {rank}] {n_new:,} items to encode")

    if not my_content_ids:
        return np.array([]), [], []

    return np.array(my_content_ids), my_texts, my_images


# ============================================================
# 编码
# ============================================================

TEXT_CACHE_MAX_LEN = 16        # 短于此长度的文本缓存 text→embedding
TEXT_CACHE_MAX_SIZE = 500_000  # 最大条目数


class LFUCache:
    """O(1) LFU cache: 频率桶 + min_freq 追踪。"""

    def __init__(self, max_size=TEXT_CACHE_MAX_SIZE):
        from collections import OrderedDict, defaultdict
        self._data = {}                          # text → embedding
        self._freq = {}                          # text → int
        self._buckets = defaultdict(OrderedDict)  # freq → {text: None}
        self._min_freq = 0
        self._max_size = max_size
        self.hits = 0

    def _touch(self, text):
        """将 text 从当前频率桶移到 freq+1 桶。"""
        f = self._freq[text]
        del self._buckets[f][text]
        if not self._buckets[f]:
            del self._buckets[f]
            if self._min_freq == f:
                self._min_freq = f + 1
        self._freq[text] = f + 1
        self._buckets[f + 1][text] = None

    def get(self, text):
        if text in self._data:
            self._touch(text)
            self.hits += 1
            return self._data[text]
        return None

    def put(self, text, embedding):
        if text in self._data:
            return
        if len(self._data) >= self._max_size:
            # 淘汰 min_freq 桶中最早插入的
            bucket = self._buckets[self._min_freq]
            victim, _ = bucket.popitem(last=False)
            if not bucket:
                del self._buckets[self._min_freq]
            del self._data[victim]
            del self._freq[victim]
        self._data[text] = embedding
        self._freq[text] = 1
        self._buckets[1][text] = None
        self._min_freq = 1

    def __len__(self):
        return len(self._data)


def encode_batch(embedder, content_ids, texts, batch_size, rank, text_cache=None):
    """编码一批文本，返回 {cid: embedding} dict。

    text_cache: LFUCache, 短文本 → embedding 缓存 (跨日期复用)。
    相同短文本直接复用 embedding，不重复过模型。
    """
    if text_cache is None:
        text_cache = LFUCache()

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
    # (hf_name, dim, default_batch_size, is_vl)
    "qwen3-0.6b":   ("Qwen/Qwen3-Embedding-0.6B", 1024, 64, False),
    "qwen3-4b":     ("Qwen/Qwen3-Embedding-4B",   2560, 32, False),
    "qwen3-8b":     ("Qwen/Qwen3-Embedding-8B",   4096, 16, False),
    "qwen3-vl-2b":  ("Qwen/Qwen3-VL-Embedding-2B", 2048, 8, True),
}


def _prepare_vl_batch(content_ids, texts, images, batch_idx, batch_size, max_images):
    """CPU-side: download images + build inputs dict. Runs in prefetch thread."""
    batch_cids = content_ids[batch_idx:batch_idx + batch_size]
    batch_texts = texts[batch_idx:batch_idx + batch_size]
    batch_imgs = images[batch_idx:batch_idx + batch_size]

    all_urls = []
    for si, urls in enumerate(batch_imgs):
        if not urls or max_images <= 0:
            continue
        for ii, u in enumerate(urls[:max_images]):
            if u:
                all_urls.append((si, ii, u))

    t_dl0 = time.time()
    downloaded = {}
    n_ok = 0
    if all_urls:
        urls_only = [u[2] for u in all_urls]
        pil_imgs = download_images_with_retry(urls_only, cache_dir=EFS_IMAGE_CACHE)
        for (si, ii, _), img in zip(all_urls, pil_imgs):
            if img is not None:
                downloaded[(si, ii)] = img
                n_ok += 1
    t_dl = time.time() - t_dl0

    inputs = []
    for si, (text, urls) in enumerate(zip(batch_texts, batch_imgs)):
        item = {"text": text or ""}
        if urls and max_images > 0:
            imgs = [downloaded.get((si, ii)) for ii in range(min(len(urls), max_images))]
            imgs = [p for p in imgs if p is not None]
            if imgs:
                item["image"] = imgs
        inputs.append(item)

    return batch_cids, inputs, t_dl, n_ok, len(all_urls)


def encode_batch_vl(embedder, content_ids, texts, images, batch_size, rank,
                    max_images: int = VL_MAX_IMAGES, prefetch_depth: int = 3):
    """VL 编码: prefetch 多个 batch 的图片下载 + 输入构造，与 GPU 推理重叠。"""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    new_embeddings = {}
    start_time = time.time()
    total = len(content_ids)

    prefetch = ThreadPoolExecutor(max_workers=prefetch_depth)
    queue = deque()
    # seed the prefetch queue
    for k in range(prefetch_depth):
        idx = k * batch_size
        if idx < total:
            queue.append((idx, prefetch.submit(
                _prepare_vl_batch, content_ids, texts, images, idx, batch_size, max_images)))

    batch_idx = 0
    while batch_idx < total:
        _, future = queue.popleft()
        batch_cids, inputs, t_dl, n_ok, n_urls = future.result()

        # enqueue next batch
        next_idx = batch_idx + prefetch_depth * batch_size
        if next_idx < total:
            queue.append((next_idx, prefetch.submit(
                _prepare_vl_batch, content_ids, texts, images, next_idx, batch_size, max_images)))

        # 推理 + OOM retry: chunk_size 减半；到 1 仍 OOM 时只 skip 那 1 条
        n = len(inputs)
        chunk_size = n
        i = 0
        n_skipped = 0
        while i < n:
            end = min(i + chunk_size, n)
            sub_cids = batch_cids[i:end]
            sub_inputs = inputs[i:end]
            try:
                emb = embedder.process(sub_inputs).cpu().float().numpy()
                for cid, e in zip(sub_cids, emb):
                    new_embeddings[cid] = e
                i = end
            except torch.cuda.OutOfMemoryError:
                import gc
                del sub_inputs
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if chunk_size == 1:
                    bad_cid = sub_cids[0]
                    bad_text = texts[batch_idx + i] if (batch_idx + i) < len(texts) else ""
                    bad_imgs = images[batch_idx + i] if (batch_idx + i) < len(images) else []
                    mem_alloc = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"    [Rank {rank}] Skip cid={bad_cid} "
                          f"(text_len={len(bad_text)}, n_imgs={len(bad_imgs) if bad_imgs else 0}, "
                          f"mem={mem_alloc:.1f}/{mem_reserved:.1f}/{mem_total:.1f}GB alloc/reserved/total)")
                    n_skipped += 1
                    i += 1
                else:
                    chunk_size = max(1, chunk_size // 2)
                    print(f"    [Rank {rank}] OOM, chunk_size -> {chunk_size}")

        if n_skipped:
            print(f"    [Rank {rank}] batch@{batch_idx}: skipped {n_skipped}/{n}")

        batch_idx += batch_size
        batch_num = batch_idx // batch_size
        if batch_num % 5 == 0 or batch_idx >= total:
            elapsed = time.time() - start_time
            done = min(batch_idx, total)
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / speed if speed > 0 else 0
            print(f"    [Rank {rank}] {done:,}/{total:,} | dl: {t_dl:.1f}s "
                  f"({n_ok}/{n_urls}) | {speed:.1f}/s | ETA: {eta/60:.1f}min")

    prefetch.shutdown(wait=False)
    return new_embeddings


# ============================================================
# Main
# ============================================================

def _resolve_dates(date_start, date_end, oldest_first=False):
    """Resolve date range to list of date strings."""
    from datetime import datetime, timedelta
    ds = date_start or DEFAULT_DATE_START
    de = date_end or DEFAULT_DATE_END
    dates = []
    d = datetime.strptime(ds, "%Y-%m-%d")
    end = datetime.strptime(de, "%Y-%m-%d")
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    if not oldest_first:
        dates.reverse()
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
    parser.add_argument('--max_images', type=int, default=VL_MAX_IMAGES,
                        help='(VL) max images per content')
    parser.add_argument('--max_pixels', type=int, default=VL_MAX_PIXELS,
                        help='(VL) per-image pixel cap; each IMAGE_FACTOR² (=1024) pixels → 1 vision token')
    parser.add_argument('--oldest_first', action='store_true',
                        help='Process dates oldest→newest (default: newest first)')
    parser.add_argument('--prefetch', type=int, default=3,
                        help='(VL) number of batches to prefetch (default: 3)')
    args = parser.parse_args()

    # 分布式设置
    rank, world_size, local_rank = setup_distributed()
    device = f'cuda:{local_rank}'

    model_name, embedding_dim, default_batch_size, is_vl = DISTRIBUTED_MODEL_CONFIGS[args.model]
    batch_size = args.batch_size or default_batch_size

    output_dir = f"{args.output_dir}/{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    my_shard_ids = [i for i in range(NUM_SHARDS) if i % world_size == rank]

    # Resolve dates (newest first)
    dates = _resolve_dates(args.date_start, args.date_end, args.oldest_first)

    if rank == 0:
        print("=" * 60)
        print(f"Multi-GPU Embedding Generation ({NUM_SHARDS}-shard, per-date)")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Dates: {dates[0]} ~ {dates[-1]} ({len(dates)} days, newest first)")
        print(f"World size: {world_size}, shards per rank: {NUM_SHARDS // world_size}")
        print(f"Batch size per GPU: {batch_size}")
        if is_vl:
            print(f"VL: max_images={args.max_images}, max_pixels={args.max_pixels:,} "
                  f"(~{args.max_pixels // 1024} vision tokens/image)")
        s3_src = S3_CONTENT_TEXT_EXPOSED_S3 if is_vl else S3_CONTENT_TEXT_EXPOSED
        print(f"Source: {s3_src}")
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

    # 加载已有 shard (每个 rank 可能管多个 shard)
    existing_shards = {}
    for si in my_shard_ids:
        sf = f"{output_dir}/shard_{si}.npy"
        if os.path.exists(sf):
            try:
                existing_shards[si] = np.load(sf, allow_pickle=True).item()
            except:
                existing_shards[si] = {}
        else:
            existing_shards[si] = {}
    print(f"  [Rank {rank}] Managing shards {my_shard_ids}, "
          f"{sum(len(s) for s in existing_shards.values()):,} existing items")

    # 模型加载前置 — 避免惰性加载导致各 rank barrier 序列不一致
    # (有数据的 rank 走 download_model_if_needed 内的 barrier + 末尾 barrier,
    #  无数据的 rank 只走 empty-skip barrier, 多日期叠加后 SeqNum 错位 → NCCL timeout)
    download_model_if_needed(model_name, rank, world_size, is_vl=is_vl)
    print(f"  [Rank {rank}] Loading model on {device}...")
    if is_vl:
        embedder = Qwen3VLEmbedder(
            model_name, device=device, max_pixels=args.max_pixels)
    else:
        embedder = Qwen3TextEmbedder(model_name, device=device)

    text_cache = LFUCache()  # 短文本 → embedding LRU 缓存 (跨日期复用)
    total_new = 0

    # ── 逐日期处理 (新 → 旧) ──
    for di, date_str in enumerate(dates):
        s3_base = S3_CONTENT_TEXT_EXPOSED_S3 if is_vl else S3_CONTENT_TEXT_EXPOSED
        date_path = f"{s3_base}/{date_str}"
        if rank == 0:
            print(f"[{di+1}/{len(dates)}] {date_str}")

        # 加载本日期数据 (VL 模式额外加载 images 列)
        my_cids, my_texts, my_images = load_date_data(
            date_path, rank, world_size, cached_ids, load_images=is_vl)

        if len(my_cids) == 0:
            if rank == 0:
                print(f"    All cached, skipping")
            # 所有 rank 需要同步, 与有数据路径末尾的 barrier 对齐
            if world_size > 1:
                dist.barrier()
            continue

        # 编码
        if is_vl:
            new_embeddings = encode_batch_vl(
                embedder, my_cids, my_texts, my_images, batch_size, rank,
                max_images=args.max_images, prefetch_depth=args.prefetch)
        else:
            new_embeddings = encode_batch(
                embedder, my_cids, my_texts, batch_size, rank, text_cache=text_cache)

        # 按 cid_to_shard 分配到对应 shard
        for cid, emb in new_embeddings.items():
            si = cid_to_shard(cid)
            existing_shards[si][cid] = emb
        cached_ids.update(str(cid) for cid in new_embeddings.keys())

        n_new = len(new_embeddings)
        total_new += n_new
        print(f"    [Rank {rank}] +{n_new:,} embeddings")

        # 每个日期处理完后保存所有管辖的 shard (断点续跑)
        for si in my_shard_ids:
            np.save(f"{output_dir}/shard_{si}.npy", existing_shards[si])

        if world_size > 1:
            dist.barrier()

    # ── 最终汇总 ──
    total_lookups = text_cache.hits + total_new  # hits + misses (encoded)
    if not is_vl and total_lookups > 0:
        hit_rate = text_cache.hits / total_lookups
        print(f"\n[Rank {rank}] Text cache: {text_cache.hits:,} hits / "
              f"{total_lookups:,} lookups ({hit_rate:.1%}), "
              f"size: {len(text_cache):,}")
    total_items = sum(len(s) for s in existing_shards.values())
    print(f"[Rank {rank}] Shards {my_shard_ids}: {total_items:,} total "
          f"(+{total_new:,} new)")

    if world_size > 1:
        dist.barrier()

    # Rank 0 更新 cached_ids.txt
    if rank == 0:
        print("Updating cached_ids.txt...")
        all_ids = set()
        for i in range(NUM_SHARDS):
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
