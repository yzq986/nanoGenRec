import numpy as np
import glob
import os
from gr_demo.config import EFS_EMBEDDING_CACHE
from gr_demo.data.loaders import load_exposed_iids

exposed = load_exposed_iids('auto', date_start='2026-03-01', date_end='2026-03-31')
print(f'Exposed IIDs: {len(exposed):,}')

for model in ['qwen3-0.6b', 'qwen3-4b']:
    cache_dir = os.path.join(EFS_EMBEDDING_CACHE, model)
    shards = sorted(glob.glob(os.path.join(cache_dir, 'shard_*.npy')))
    cache_ids = set()
    for s in shards:
        d = np.load(s, allow_pickle=True).item()
        cache_ids.update(d.keys())
    overlap = cache_ids & exposed
    missing = exposed - cache_ids
    print(f'{model}: {len(cache_ids):,} cached, {len(overlap):,} exposed covered ({len(overlap)/len(exposed)*100:.1f}%), {len(missing):,} exposed missing')
