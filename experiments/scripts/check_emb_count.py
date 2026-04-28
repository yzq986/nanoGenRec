import numpy as np
import glob
import os
from gr_demo.config import EFS_EMBEDDING_CACHE

for model in ['qwen3-0.6b', 'qwen3-4b']:
    cache_dir = os.path.join(EFS_EMBEDDING_CACHE, model)
    shards = sorted(glob.glob(os.path.join(cache_dir, 'shard_*.npy')))
    total = 0
    for s in shards:
        d = np.load(s, allow_pickle=True).item()
        total += len(d)
    print(f'{model}: {len(shards)} shards, {total:,} items')
