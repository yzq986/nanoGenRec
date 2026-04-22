"""一次性迁移: incremental_cache.npy → 8 个 shard 文件。

用法:
    python -m gr_demo.data.migrate_shards --model qwen3-0.6b
    python -m gr_demo.data.migrate_shards --model qwen3-0.6b --dry_run

将 {EFS_EMBEDDING_CACHE}/{model}/incremental_cache.npy 按
sha256(str(content_id)) % 8 拆分为 shard_0.npy ~ shard_7.npy，
并生成 cached_ids.txt 索引。
"""

import argparse
import os
import time

import numpy as np

from config import EFS_EMBEDDING_CACHE

NUM_SHARDS = 8


def cid_to_shard(cid, n_shards=NUM_SHARDS) -> int:
    import hashlib
    return int(hashlib.sha256(str(cid).encode()).hexdigest(), 16) % n_shards


def main():
    parser = argparse.ArgumentParser(description='Migrate incremental_cache.npy to 8 shards')
    parser.add_argument('--model', type=str, default='qwen3-0.6b')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help=f'Override cache directory (default: {EFS_EMBEDDING_CACHE}/{{model}})')
    parser.add_argument('--dry_run', action='store_true', help='Only print stats, do not write')
    args = parser.parse_args()

    cache_dir = args.cache_dir or f'{EFS_EMBEDDING_CACHE}/{args.model}'
    legacy_path = f'{cache_dir}/incremental_cache.npy'

    if not os.path.exists(legacy_path):
        print(f"Legacy cache not found: {legacy_path}")
        print("Nothing to migrate.")
        return

    print(f"Loading {legacy_path}...")
    t0 = time.time()
    cache = np.load(legacy_path, allow_pickle=True).item()
    load_time = time.time() - t0
    print(f"Loaded {len(cache):,} entries in {load_time:.1f}s")

    # Split by hash
    shards = [{} for _ in range(NUM_SHARDS)]
    for cid, emb in cache.items():
        shards[cid_to_shard(cid)][cid] = emb

    print(f"\nShard distribution:")
    for i, shard in enumerate(shards):
        print(f"  shard_{i}: {len(shard):,} entries")
    total = sum(len(s) for s in shards)
    print(f"  Total: {total:,} (original: {len(cache):,})")
    assert total == len(cache), "Entry count mismatch!"

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Write shards
    print(f"\nWriting shards to {cache_dir}/...")
    for i, shard in enumerate(shards):
        shard_path = f'{cache_dir}/shard_{i}.npy'
        np.save(shard_path, shard)
        size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        print(f"  shard_{i}.npy: {len(shard):,} entries, {size_mb:.1f} MB")

    # Write cached_ids.txt
    ids_path = f'{cache_dir}/cached_ids.txt'
    with open(ids_path, 'w') as f:
        for cid in cache.keys():
            f.write(f"{cid}\n")
    print(f"  cached_ids.txt: {len(cache):,} IDs")

    total_time = time.time() - t0
    print(f"\nMigration complete! ({total_time:.1f}s)")
    print(f"You can now delete the legacy file:")
    print(f"  rm {legacy_path}")


if __name__ == '__main__':
    main()
