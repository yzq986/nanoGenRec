#!/usr/bin/env python3
"""Batch snHR eval for EXP-049 sid_caches.

Run from repo root:
    python -m experiments.scripts.run_snhr

Reads: sid_cache/*/semantic_ids.npy + local behavior parquet files.
Uses metrics.behavior.build_content_user_sets for fast vectorized loading.
"""
import glob
import os
import random
from collections import defaultdict

import numpy as np

from eval.batch import load_all_behavior_data
from metrics.behavior import build_content_user_sets

BEHAVIOR_DIR = '/mnt/workspace/gr-demo-behavior-cache'
SID_CACHE_ROOT = 'experiments/sid_cache'
PREFIX_LAYERS = 2
MAX_ITEMS = 5000
RANDOM_SEED = 42


def compute_snhr(semantic_ids_path, content_users):
    sid_dict = np.load(semantic_ids_path, allow_pickle=True).item()
    content_ids = list(sid_dict.keys())

    prefix_to_cids = defaultdict(list)
    for cid in content_ids:
        prefix = '_'.join(str(sid_dict[cid]).split('_')[:PREFIX_LAYERS])
        prefix_to_cids[prefix].append(cid)

    valid = [cid for cid in content_ids
             if cid in content_users and len(content_users[cid]) >= 2]
    if not valid:
        return None, 0

    random.seed(RANDOM_SEED)
    if len(valid) > MAX_ITEMS:
        valid = random.sample(valid, MAX_ITEMS)

    hit_rates = []
    for cid in valid:
        prefix = '_'.join(str(sid_dict[cid]).split('_')[:PREFIX_LAYERS])
        neighbors = [c for c in prefix_to_cids[prefix] if c != cid]
        if not neighbors:
            continue
        pos_users = content_users[cid]
        hits = sum(1 for nb in neighbors if content_users.get(nb, frozenset()) & pos_users)
        hit_rates.append(hits / len(neighbors))

    if not hit_rates:
        return None, len(valid)
    return float(np.mean(hit_rates)), len(hit_rates)


def quality(v):
    if v >= 0.15: return 'excellent'
    if v >= 0.10: return 'good'
    if v >= 0.05: return 'acceptable'
    return 'poor'


print("Loading behavior data...")
behavior_data = load_all_behavior_data(
    behavior_path=BEHAVIOR_DIR,
    date_start='2026-03-18',
    date_end='2026-03-31',
)
print(f"  {len(behavior_data['uid']):,} interactions loaded")

print("Building content->users index (vectorized)...")
content_users = build_content_user_sets(behavior_data)
print(f"  {len(content_users):,} unique content IDs with positive interactions\n")

caches = sorted(glob.glob(os.path.join(SID_CACHE_ROOT, 'exp049-*')))
print(f"{'Name':<40} {'snHR':>7}  {'n':>6}  Quality")
print('-' * 65)
for cache in caches:
    name = os.path.basename(cache)
    sid_path = os.path.join(cache, 'semantic_ids.npy')
    if not os.path.exists(sid_path):
        print(f"{name:<40}  MISSING")
        continue
    snhr, n = compute_snhr(sid_path, content_users)
    if snhr is None:
        print(f"{name:<40}  N/A    {n:>6}")
    else:
        print(f"{name:<40} {snhr:>7.4f}  {n:>6}  {quality(snhr)}")
