"""encode_with_qwen3() / encode_with_qwen3_text() — 含缓存逻辑。"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from config import EFS_IMAGE_CACHE, S3_EMBEDDING_CACHE_BACKUP

IMAGE_CACHE_DIR = EFS_IMAGE_CACHE


def download_image(url: str, timeout: int = 3, use_cache: bool = True) -> 'Optional[Image.Image]':
    """下载单张图片，带超时和磁盘缓存"""
    import hashlib
    import requests
    from io import BytesIO
    from PIL import Image

    # 生成缓存文件路径
    if use_cache:
        os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = os.path.join(IMAGE_CACHE_DIR, f"{url_hash}.jpg")

        # 如果缓存存在，直接读取
        if os.path.exists(cache_path):
            try:
                return Image.open(cache_path).convert('RGB')
            except Exception:
                pass  # 缓存损坏，重新下载

    try:
        # timeout=(connect_timeout, read_timeout)，两个都设严格一点
        resp = requests.get(url, timeout=(timeout, timeout), stream=False)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert('RGB')

        # 保存到缓存
        if use_cache:
            try:
                img.save(cache_path, 'JPEG', quality=85)
            except Exception:
                pass  # 保存失败不影响返回

        return img
    except Exception:
        # 下载失败，返回 None
        return None


def download_images_async(
    urls: List[str],
    max_workers: int = 8,
    timeout_per_image: int = 3,
    timeout_total: int = 5
) -> 'List[Optional[Image.Image]]':
    """并行下载多张图片，带总超时控制（使用进程池可强制终止）"""
    from multiprocessing import Pool, TimeoutError as MPTimeoutError
    import time as _time

    if not urls:
        return []

    results = [None] * len(urls)
    start = _time.time()

    # 使用进程池，可以强制 terminate
    with Pool(processes=min(max_workers, len(urls))) as pool:
        async_results = [
            (idx, pool.apply_async(download_image, (url, timeout_per_image)))
            for idx, url in enumerate(urls)
        ]

        for idx, async_res in async_results:
            remaining = timeout_total - (_time.time() - start)
            if remaining <= 0:
                break  # 总超时，跳过剩余
            try:
                results[idx] = async_res.get(timeout=max(0.1, remaining))
            except (MPTimeoutError, Exception):
                results[idx] = None

        pool.terminate()  # 强制终止所有子进程

    return results


def encode_with_qwen3(
    embedder: 'Qwen3VLEmbedder',
    content_ids: np.ndarray,
    texts: List[str],
    images_list: List[List[str]] = None,
    batch_size: int = 32,
    max_images: int = 0,
    image_download_workers: int = 16,
    show_progress: bool = True,
    cache_dir: str = None,
    checkpoint_every: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用 Qwen3VLEmbedder 编码，支持增量缓存

    流程:
    1. 加载缓存 {id -> embedding}
    2. 过滤出未缓存的样本
    3. 只对未缓存的 infer
    4. 每 checkpoint_every 个 batch 保存缓存
    5. 最后合并返回
    """
    import time

    n = len(texts)
    start_time = time.time()

    # 1. 加载已有缓存 {id -> embedding}
    cache_file = f"{cache_dir}/embedding_cache.npy" if cache_dir else None
    cached = {}
    if cache_file and os.path.exists(cache_file):
        cached = np.load(cache_file, allow_pickle=True).item()
        print(f"  Loaded cache: {len(cached):,} embeddings")

    # 2. 过滤出未缓存的样本
    to_process = [(i, content_ids[i], texts[i], images_list[i] if images_list else [])
                  for i in range(n) if content_ids[i] not in cached]
    print(f"  Total: {n:,}, Cached: {n - len(to_process):,}, To process: {len(to_process):,}")

    if len(to_process) == 0:
        embeddings = np.array([cached[cid] for cid in content_ids])
        return content_ids, embeddings

    # 3. 只对未缓存的 infer
    new_embeddings = {}
    processed = 0

    for batch_idx in range(0, len(to_process), batch_size):
        batch = to_process[batch_idx:batch_idx + batch_size]
        batch_ids = [b[1] for b in batch]
        batch_texts = [b[2] for b in batch]
        batch_images = [b[3] for b in batch]

        batch_start = time.time()

        # 下载图片
        all_urls = []
        for sample_idx, images in enumerate(batch_images):
            if max_images > 0 and images and len(images) > 0:
                for img_idx, url in enumerate(images[:max_images]):
                    if url:
                        all_urls.append((sample_idx, img_idx, url))

        t_dl_start = time.time()
        downloaded = {}
        if all_urls:
            urls_only = [u[2] for u in all_urls]
            pil_images = download_images_async(urls_only, max_workers=image_download_workers)
            for (si, ii, _), img in zip(all_urls, pil_images):
                if img:
                    downloaded[(si, ii)] = img
        t_dl_end = time.time()

        # 构建输入
        inputs = []
        for si, (text, images) in enumerate(zip(batch_texts, batch_images)):
            item = {"text": text or ""}
            if max_images > 0 and images:
                pil_imgs = [downloaded.get((si, ii)) for ii in range(min(len(images), max_images))]
                pil_imgs = [p for p in pil_imgs if p]
                if pil_imgs:
                    item["image"] = pil_imgs
            inputs.append(item)

        if show_progress and processed == 0:
            print(f"  Processing first batch ({len(inputs)} samples)...")

        # Infer with OOM retry
        batch_emb = None
        retry_chunk_size = len(inputs)

        while batch_emb is None and retry_chunk_size >= 1:
            try:
                chunk_embeddings = []
                for chunk_start in range(0, len(inputs), retry_chunk_size):
                    chunk_inputs = inputs[chunk_start:chunk_start + retry_chunk_size]
                    chunk_emb = embedder.process(chunk_inputs).cpu().float().numpy()
                    chunk_embeddings.append(chunk_emb)
                batch_emb = np.concatenate(chunk_embeddings, axis=0)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                retry_chunk_size = retry_chunk_size // 2
                print(f"  [OOM] Retrying with chunk_size={retry_chunk_size}...")
                if retry_chunk_size < 1:
                    print(f"  [OOM] Skipping batch")
                    break

        if batch_emb is None:
            processed += len(batch)
            continue

        for cid, emb in zip(batch_ids, batch_emb):
            new_embeddings[cid] = emb

        processed += len(batch)
        batch_time = time.time() - batch_start
        dl_time = t_dl_end - t_dl_start

        if show_progress:
            elapsed = time.time() - start_time
            remaining = len(to_process) - processed
            speed = processed / elapsed if elapsed > 0 else 0
            eta = remaining / speed if speed > 0 else 0
            eta_str = f"{int(eta//60)}m{int(eta%60)}s" if eta < 3600 else f"{eta/3600:.1f}h"
            pct = processed / len(to_process) * 100
            batch_num = batch_idx // batch_size + 1
            print(f"  Batch {batch_num}: {processed:,}/{len(to_process):,} ({pct:.1f}%) | "
                  f"dl: {dl_time:.1f}s ({len(downloaded)}/{len(all_urls)}) | "
                  f"{speed:.1f}/s | ETA: {eta_str}")

        # 4. 增量保存
        batch_num = batch_idx // batch_size + 1
        if cache_dir and batch_num % checkpoint_every == 0:
            merged = {**cached, **new_embeddings}
            np.save(cache_file, merged)
            print(f"  [Checkpoint: {len(merged):,} embeddings]")

    # 最终保存
    if cache_dir and new_embeddings:
        merged = {**cached, **new_embeddings}
        np.save(cache_file, merged)
        print(f"  [Final: {len(merged):,} embeddings]")

    # 5. 合并返回
    all_emb = {**cached, **new_embeddings}
    embeddings = np.array([all_emb[cid] for cid in content_ids])
    return content_ids, embeddings


def encode_with_qwen3_text(
    embedder: 'Qwen3TextEmbedder',
    content_ids: np.ndarray,
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = True,
    cache_dir: str = None,
    checkpoint_every: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用 Qwen3TextEmbedder 编码纯文本，支持增量缓存（同 encode_with_qwen3）"""
    import time

    n = len(texts)
    start_time = time.time()

    # 加载缓存 (增量缓存，按 content_id 为 key)
    cache_file = f"{cache_dir}/incremental_cache.npy" if cache_dir else None
    s3_cache_backup = f"{S3_EMBEDDING_CACHE_BACKUP}/{os.path.basename(cache_dir)}/incremental_cache.npy" if cache_dir else None
    cached = {}

    def load_cache_from_s3(s3_path: str, local_path: str) -> dict:
        """从 S3 下载缓存"""
        try:
            import boto3
            print(f"  Downloading cache from S3: {s3_path}")
            s3 = boto3.client('s3')
            bucket = s3_path.replace('s3://', '').split('/')[0]
            key = '/'.join(s3_path.replace('s3://', '').split('/')[1:])
            s3.download_file(bucket, key, local_path)
            data = np.load(local_path, allow_pickle=True).item()
            print(f"  Downloaded and loaded {len(data):,} embeddings from S3")
            return data
        except Exception as e:
            print(f"  Warning: Failed to load from S3 ({e})")
            return {}

    if cache_file and os.path.exists(cache_file):
        try:
            cached = np.load(cache_file, allow_pickle=True).item()
            print(f"  Loaded incremental cache: {len(cached):,} embeddings")
        except (EOFError, Exception) as e:
            print(f"  Warning: Local cache corrupted ({e}), trying S3 backup...")
            cached = load_cache_from_s3(s3_cache_backup, cache_file) if s3_cache_backup else {}
    elif cache_file and s3_cache_backup:
        # 本地不存在，尝试从 S3 下载
        print(f"  Local cache not found, trying S3 backup...")
        cached = load_cache_from_s3(s3_cache_backup, cache_file)

    # 过滤未缓存
    to_process = [(i, content_ids[i], texts[i]) for i in range(n) if content_ids[i] not in cached]
    print(f"  Total: {n:,}, Cached: {n - len(to_process):,}, To process: {len(to_process):,}")

    if len(to_process) == 0:
        return content_ids, np.array([cached[cid] for cid in content_ids])

    new_embeddings = {}
    total_to_process = len(to_process)

    for batch_idx in range(0, total_to_process, batch_size):
        batch = to_process[batch_idx:batch_idx + batch_size]
        batch_ids = [b[1] for b in batch]
        batch_texts = [b[2] for b in batch]
        batch_start = time.time()

        if show_progress and batch_idx == 0:
            print(f"  Processing first batch ({len(batch_texts)} samples)...")

        # OOM 重试逻辑：遇到 OOM 时清理缓存、减半 batch size 重试
        batch_emb_np = None
        current_batch_ids = batch_ids
        current_batch_texts = batch_texts
        retry_batch_size = len(current_batch_texts)

        while batch_emb_np is None and retry_batch_size >= 1:
            try:
                # 分块处理
                chunk_embeddings = []
                for chunk_start in range(0, len(current_batch_texts), retry_batch_size):
                    chunk_texts = current_batch_texts[chunk_start:chunk_start + retry_batch_size]
                    chunk_emb = embedder.encode(chunk_texts)
                    chunk_embeddings.append(chunk_emb.cpu().float().numpy())
                batch_emb_np = np.concatenate(chunk_embeddings, axis=0)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                retry_batch_size = retry_batch_size // 2
                print(f"  [OOM] Retrying with batch_size={retry_batch_size}...")
                if retry_batch_size < 1:
                    print(f"  [OOM] Skipping batch, batch_size too small")
                    batch_emb_np = None
                    break

        if batch_emb_np is None:
            continue  # 跳过这个 batch

        # 存入缓存字典
        for cid, emb in zip(current_batch_ids, batch_emb_np):
            new_embeddings[cid] = emb

        # 定期保存缓存 (每 checkpoint_every 个 batch)
        batch_num = batch_idx // batch_size + 1
        if cache_file and batch_num % checkpoint_every == 0:
            cached.update(new_embeddings)
            np.save(cache_file, cached)
            print(f"  [Checkpoint] Saved {len(cached):,} embeddings to EFS")

            # 同时备份到 S3 (每 5 次 checkpoint 备份一次)
            if batch_num % (checkpoint_every * 5) == 0 and s3_cache_backup:
                try:
                    import boto3
                    s3 = boto3.client('s3')
                    bucket = s3_cache_backup.replace('s3://', '').split('/')[0]
                    key = '/'.join(s3_cache_backup.replace('s3://', '').split('/')[1:])
                    s3.upload_file(cache_file, bucket, key, ExtraArgs={'ACL': 'bucket-owner-full-control'})
                    print(f"  [Checkpoint] Backed up to S3: {s3_cache_backup}")
                except Exception as e:
                    print(f"  [Warning] S3 backup failed: {e}")

        batch_time = time.time() - batch_start
        if show_progress:
            elapsed = time.time() - start_time
            samples_done = min(batch_idx + batch_size, total_to_process)
            samples_remaining = total_to_process - samples_done
            speed = samples_done / elapsed if elapsed > 0 else 0
            eta_seconds = samples_remaining / speed if speed > 0 else 0
            eta_str = f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f}h"
            pct = samples_done / total_to_process * 100
            print(f"  Batch {batch_idx//batch_size + 1}: {samples_done:,}/{total_to_process:,} ({pct:.1f}%) | "
                  f"{speed:.1f} samples/s | ETA: {eta_str}")

    # 合并缓存和新生成的 embedding
    cached.update(new_embeddings)

    # 最终保存缓存
    if cache_file:
        np.save(cache_file, cached)
        print(f"  [Final] Saved {len(cached):,} embeddings to EFS")

        # 备份到 S3
        if s3_cache_backup:
            try:
                import boto3
                s3 = boto3.client('s3')
                bucket = s3_cache_backup.replace('s3://', '').split('/')[0]
                key = '/'.join(s3_cache_backup.replace('s3://', '').split('/')[1:])
                s3.upload_file(cache_file, bucket, key, ExtraArgs={'ACL': 'bucket-owner-full-control'})
                print(f"  [Final] Backed up to S3: {s3_cache_backup}")
            except Exception as e:
                print(f"  [Warning] S3 backup failed: {e}")

    result_embeddings = np.array([cached[cid] for cid in content_ids])
    return content_ids, result_embeddings
