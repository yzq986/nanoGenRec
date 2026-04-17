#!/usr/bin/env python3
"""Analyze user behavior data distribution across date ranges.

Shows items/user distribution and truncation stats for EXP-016 planning.

Usage:
    python experiments/scripts/analyze_data_distribution.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gr_demo.eval.batch import load_all_behavior_data

# Date ranges matching EXP-016 design
DATE_RANGES = [
    ("A-7d",  "2026-03-25", "2026-03-31"),
    ("B-14d", "2026-03-18", "2026-03-31"),
    ("C-31d", "2026-03-01", "2026-03-31"),
    ("D-62d", "2026-02-01", "2026-03-31"),
    ("E-66d", "2026-01-25", "2026-03-31"),
]

MAX_ITEMS = 170  # current cap: max_seq_len=512 / n_layers=3

print("=" * 80)
print("Data Distribution Analysis for EXP-016")
print(f"max_items cap = {MAX_ITEMS} (max_seq_len=512, n_layers=3)")
print("=" * 80)

results = []
for name, ds, de in DATE_RANGES:
    print(f"\n--- {name}: {ds} ~ {de} ---")
    try:
        d = load_all_behavior_data(date_start=ds, date_end=de)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    uids = d['uid']
    actions = d.get('action_bitmap', np.ones(len(uids), dtype=np.int32))
    # Filter to positive actions (strip view_exit bit 128)
    VIEW_EXIT_BIT = 128
    mask = (actions & ~VIEW_EXIT_BIT) > 0
    uids_pos = uids[mask]

    _, counts = np.unique(uids_pos, return_counts=True)
    n_users = len(counts)
    n_interactions = int(counts.sum())
    pcts = np.percentile(counts, [25, 50, 75, 90, 95, 99, 99.9])
    n_trunc = int((counts > MAX_ITEMS).sum())
    trunc_pct = 100.0 * n_trunc / n_users if n_users > 0 else 0
    # Items lost to truncation
    items_lost = int((counts[counts > MAX_ITEMS] - MAX_ITEMS).sum())
    items_total = int(counts.sum())
    items_lost_pct = 100.0 * items_lost / items_total if items_total > 0 else 0

    print(f"  Interactions:    {n_interactions:>12,} (positive actions)")
    print(f"  Users:           {n_users:>12,}")
    print(f"  Items/user:      mean={counts.mean():.1f}")
    print(f"    p25={pcts[0]:.0f}  p50={pcts[1]:.0f}  p75={pcts[2]:.0f}  "
          f"p90={pcts[3]:.0f}  p95={pcts[4]:.0f}  p99={pcts[5]:.0f}  "
          f"p99.9={pcts[6]:.0f}  max={counts.max()}")
    print(f"  Truncated users: {n_trunc:,} / {n_users:,} ({trunc_pct:.2f}%)")
    print(f"  Items lost:      {items_lost:,} / {items_total:,} ({items_lost_pct:.2f}%)")

    # Users with >= 2 items (minimum for training)
    n_trainable = int((counts >= 2).sum())
    print(f"  Trainable users: {n_trainable:,} (>=2 items)")

    results.append({
        'name': name, 'ds': ds, 'de': de,
        'n_users': n_users, 'n_interactions': n_interactions,
        'mean': round(counts.mean(), 1),
        'p50': int(pcts[1]), 'p90': int(pcts[3]),
        'p95': int(pcts[4]), 'p99': int(pcts[5]),
        'max': int(counts.max()),
        'n_truncated': n_trunc, 'truncated_pct': round(trunc_pct, 2),
        'items_lost': items_lost, 'items_lost_pct': round(items_lost_pct, 2),
        'n_trainable': n_trainable,
    })

# Summary table
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"{'Config':<8} {'Users':>10} {'Interactions':>14} {'Mean':>6} {'P50':>5} "
      f"{'P95':>5} {'P99':>5} {'Max':>6} {'Trunc%':>7} {'ItemsLost%':>10}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<8} {r['n_users']:>10,} {r['n_interactions']:>14,} "
          f"{r['mean']:>6.1f} {r['p50']:>5} {r['p95']:>5} {r['p99']:>5} "
          f"{r['max']:>6} {r['truncated_pct']:>6.2f}% {r['items_lost_pct']:>9.2f}%")
