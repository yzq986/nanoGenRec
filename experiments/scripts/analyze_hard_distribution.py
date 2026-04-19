#!/usr/bin/env python3
"""Analyze distribution of hard/medium/easy rejected candidates per preference pair.

Helps diagnose why Hard DPO training sees avg 5.9 rejected per pair (vs 20 requested).

Usage (from repo root):
    python experiments/scripts/analyze_hard_distribution.py

Self-contained: only requires numpy (no gr_demo import needed).
"""

import os
import sys
import numpy as np
from collections import Counter


def load_preference_shard(path):
    """Load preference pairs from npz shard (inlined from rl/preference.py)."""
    data = np.load(path, allow_pickle=True)
    n_layers = int(data['n_layers'])

    context_flat = data['context_flat']
    context_offsets = data['context_offsets']
    chosen_sids = data['chosen_sids']

    rej_easy_flat = data['rejected_easy_flat']
    rej_easy_offsets = data['rejected_easy_offsets']
    rej_medium_flat = data['rejected_medium_flat']
    rej_medium_offsets = data['rejected_medium_offsets']
    rej_hard_flat = data['rejected_hard_flat']
    rej_hard_offsets = data['rejected_hard_offsets']

    pairs = []
    n = len(context_offsets) - 1
    for i in range(n):
        ctx = context_flat[context_offsets[i]:context_offsets[i + 1]].tolist()
        chosen = chosen_sids[i].tolist()

        def _read_rejected(flat, offsets, idx):
            start = int(offsets[idx])
            end = int(offsets[idx + 1])
            count = end - start
            flat_start = start * n_layers
            flat_end = end * n_layers
            chunk = flat[flat_start:flat_end].tolist()
            return [chunk[j * n_layers:(j + 1) * n_layers] for j in range(count)]

        pairs.append({
            'context': ctx,
            'chosen': chosen,
            'rejected_easy': _read_rejected(rej_easy_flat, rej_easy_offsets, i),
            'rejected_medium': _read_rejected(rej_medium_flat, rej_medium_offsets, i),
            'rejected_hard': _read_rejected(rej_hard_flat, rej_hard_offsets, i),
        })

    return pairs


def bucket_label(count):
    if count == 0:
        return "0"
    elif count <= 5:
        return "1-5"
    elif count <= 10:
        return "6-10"
    elif count <= 15:
        return "11-15"
    elif count <= 20:
        return "16-20"
    else:
        return "21+"


def analyze_difficulty(pairs, key, label):
    """Print distribution stats for a given difficulty key."""
    counts = [len(pair[key]) for pair in pairs]
    total = len(counts)
    nonzero = [c for c in counts if c > 0]

    print(f"\n{'='*60}")
    print(f"  {label} (key={key})")
    print(f"{'='*60}")
    print(f"  Total pairs:          {total:,}")
    print(f"  Pairs with >0:        {len(nonzero):,} ({100*len(nonzero)/total:.1f}%)")
    print(f"  Pairs with 0:         {total - len(nonzero):,} ({100*(total-len(nonzero))/total:.1f}%)")

    if nonzero:
        arr = np.array(nonzero)
        print(f"  Mean (non-zero):      {arr.mean():.2f}")
        print(f"  Median (non-zero):    {np.median(arr):.1f}")
        print(f"  Min (non-zero):       {arr.min()}")
        print(f"  Max (non-zero):       {arr.max()}")
    else:
        print(f"  (no non-zero entries)")

    # Overall mean (including zeros) - this is what training sees
    all_arr = np.array(counts)
    print(f"  Mean (all pairs):     {all_arr.mean():.2f}")

    # Histogram
    buckets = Counter(bucket_label(c) for c in counts)
    bucket_order = ["0", "1-5", "6-10", "11-15", "16-20", "21+"]
    print(f"\n  Distribution:")
    for b in bucket_order:
        n = buckets.get(b, 0)
        pct = 100 * n / total
        bar = '#' * int(pct / 2)
        print(f"    {b:>5s}: {n:>6,} ({pct:5.1f}%) {bar}")

    return counts


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pref_dir', type=str, default=None,
                        help='Override preference shard directory')
    args = parser.parse_args()

    # Load preference shards
    if args.pref_dir:
        pref_dir = args.pref_dir
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pref_dir = os.path.join(repo_root, 'experiments', 'sp_dpo_data', 'exp017', 'sft-pfx')

    # Check how many shards
    meta_path = os.path.join(pref_dir, 'meta.json')
    n_shards = 1
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        n_shards = meta.get('n_shards', 1)
        print(f"Meta: {json.dumps(meta, indent=2)}")

    # Load all available shards
    pairs = []
    for si in range(n_shards):
        sp = os.path.join(pref_dir, f'preference_shard_{si}.npz')
        if os.path.exists(sp):
            shard_pairs = load_preference_shard(sp)
            pairs.extend(shard_pairs)
            print(f"  Shard {si}: {len(shard_pairs):,} pairs")
    if not pairs:
        print(f"ERROR: No preference shards found in {pref_dir}")
        print(f"  Expected files like preference_shard_0.npz")
        print(f"  Available files: {os.listdir(pref_dir)}")
        sys.exit(1)
    print(f"\nLoaded {len(pairs):,} total pairs from {len([1 for si in range(n_shards) if os.path.exists(os.path.join(pref_dir, f'preference_shard_{si}.npz'))])} shards")

    # Analyze each difficulty
    easy_counts = analyze_difficulty(pairs, 'rejected_easy', 'EASY rejected')
    med_counts = analyze_difficulty(pairs, 'rejected_medium', 'MEDIUM rejected')
    hard_counts = analyze_difficulty(pairs, 'rejected_hard', 'HARD rejected')

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*60}")
    for label, counts in [('Easy', easy_counts), ('Medium', med_counts), ('Hard', hard_counts)]:
        arr = np.array(counts)
        nonzero = arr[arr > 0]
        print(f"  {label:>6s}: "
              f"mean_all={arr.mean():.2f}, "
              f"mean_nz={nonzero.mean() if len(nonzero) > 0 else 0:.2f}, "
              f"pairs_with_any={len(nonzero):,}/{len(counts):,} "
              f"({100*len(nonzero)/len(counts):.1f}%)")

    # What training actually sees (PreferencePairDataset filters)
    print(f"\n{'='*60}")
    print(f"  TRAINING IMPACT (PreferencePairDataset filter)")
    print(f"{'='*60}")
    for diff_name, key in [('easy', 'rejected_easy'), ('medium', 'rejected_medium'), ('hard', 'rejected_hard')]:
        valid = [p for p in pairs if len(p[key]) > 0]
        if valid:
            rej_counts = [min(len(p[key]), 20) for p in valid]
            arr = np.array(rej_counts)
            print(f"  --difficulty {diff_name}: "
                  f"{len(valid):,} pairs survive filter, "
                  f"avg {arr.mean():.1f} rejected/pair, "
                  f"median {np.median(arr):.0f}")
        else:
            print(f"  --difficulty {diff_name}: 0 pairs survive filter")


if __name__ == '__main__':
    main()
