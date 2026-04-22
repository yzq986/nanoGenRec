"""重新分配已有 shard 中的 embedding 到正确的 shard (基于 sha256 确定性 hash)。

修复 hash() 非确定性 bug 后，已有 shard 中的 item 路由是乱的。
此脚本将所有 shard 合并后按新 hash 重新分配，并更新 cached_ids.txt。

用法:
    python -m gr_demo.data.rebalance_shards --model qwen3-4b
    python -m gr_demo.data.rebalance_shards --model qwen3-4b --dry_run
"""

import argparse
import os
import time

import numpy as np

from gr_demo.data.encode_distributed import cid_to_shard, NUM_SHARDS
from gr_demo.config import EFS_EMBEDDING_CACHE


def main():
    parser = argparse.ArgumentParser(description='Rebalance shard files using deterministic hash')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    cache_dir = args.cache_dir or f'{EFS_EMBEDDING_CACHE}/{args.model}'

    print(f"Rebalancing shards in {cache_dir}")
    print(f"Shard count: {NUM_SHARDS}")

    # 1. Load all shards
    merged = {}
    for i in range(NUM_SHARDS):
        shard_file = f'{cache_dir}/shard_{i}.npy'
        if not os.path.exists(shard_file):
            continue
        t0 = time.time()
        data = np.load(shard_file, allow_pickle=True).item()
        print(f"  shard_{i}: {len(data):,} items ({time.time()-t0:.1f}s)")
        merged.update(data)

    print(f"\nTotal unique items: {len(merged):,}")

    if not merged:
        print("Nothing to rebalance.")
        return

    # 2. Re-route by new deterministic hash
    new_shards = [{} for _ in range(NUM_SHARDS)]
    for cid, emb in merged.items():
        new_shards[cid_to_shard(cid)][cid] = emb

    print(f"\nNew shard distribution:")
    for i, shard in enumerate(new_shards):
        print(f"  shard_{i}: {len(shard):,} items")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # 3. Write new shards
    print(f"\nWriting shards...")
    for i, shard in enumerate(new_shards):
        shard_file = f'{cache_dir}/shard_{i}.npy'
        np.save(shard_file, shard)
        size_mb = os.path.getsize(shard_file) / (1024 * 1024)
        print(f"  shard_{i}.npy: {len(shard):,} items, {size_mb:.1f} MB")

    # 4. Update cached_ids.txt
    ids_file = f'{cache_dir}/cached_ids.txt'
    with open(ids_file, 'w') as f:
        for cid in merged.keys():
            f.write(f"{str(cid)}\n")
    print(f"  cached_ids.txt: {len(merged):,} IDs")

    print(f"\nDone! Rebalanced {len(merged):,} items across {NUM_SHARDS} shards.")


if __name__ == '__main__':
    main()
