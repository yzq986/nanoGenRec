"""NTP data preprocessing — build unified sequences and save as shards.

Single-process, multi-worker command that prepares data for DDP training.
Each shard contains unified per-user sequences (tokens + split_pos + eval_cids).
During training, each rank loads only its own shard.

Usage:
    python run.py preprocess-ntp \\
        --sid_cache experiments/sid_cache/exp049-0.6b-nc8192-h128 \\
        --output_dir experiments/ntp_data/exp049-0.6b-nc8192-h128 \\
        --n_workers 64

Output:
    {output_dir}/
        train_shard_0.npz ... train_shard_{N-1}.npz   (per-rank data)
        meta.json                                       (n_layers, n_clusters, etc.)
"""

import argparse
import json
import multiprocessing
import os
import time

import numpy as np
import pandas as pd

from ntp.train import (
    _parse_sid_dict, _build_user_items,
    _VIEW_EXIT_BIT, _STRONG_POSITIVE_MASK, _TRADE_MASK,
    _TIME_GAP_BOUNDARIES,
)

# ── Vectorized helpers ─────────────────────────────────────────────────────────

_BOUNDARIES_ARR = np.array(_TIME_GAP_BOUNDARIES, dtype=np.float64)


def _time_gap_buckets_vec(ts_arr):
    """ts_arr: 1-D float array for one user → bucket int array, same length."""
    deltas = np.diff(ts_arr.astype(np.float64))
    deltas = np.maximum(deltas, 0.0)
    buckets = np.searchsorted(_BOUNDARIES_ARR, deltas, side='right').astype(np.int8)
    return np.concatenate([[0], buckets])   # first item = BOS = 0


def _bitmap_to_level_vec(bm_arr):
    """bm_arr: int32 array → level int8 array (1=weak, 2=strong, 3=trade)."""
    bm = bm_arr.astype(np.int32) & ~_VIEW_EXIT_BIT
    level = np.ones(len(bm), dtype=np.int8)
    level[(bm & _STRONG_POSITIVE_MASK) != 0] = 2
    level[(bm & _TRADE_MASK) != 0] = 3
    return level


# ── Per-chunk worker ───────────────────────────────────────────────────────────

def _process_chunk(args):
    """Build sequences for a slice of users. Called in worker processes."""
    (user_indices, starts, ends,
     iid_idx_s, ts_s, actions_s,
     token_matrix, n_layers,
     split_ts, max_items,
     shift_features, action_l2_only) = args

    sequences = []
    n_truncated = 0
    raw_items_per_user = []

    for u in user_indices:
        s, e = int(starts[u]), int(ends[u])
        n = e - s
        if n < 2:
            continue

        raw_items_per_user.append(n)
        u_idx = iid_idx_s[s:e]   # int32 indices into token_matrix
        u_ts  = ts_s[s:e].astype(np.float64)
        u_act = actions_s[s:e]

        if n > max_items:
            n_truncated += 1
            off = n - max_items
            u_idx = u_idx[off:]
            u_ts  = u_ts[off:]
            u_act = u_act[off:]
            n = max_items

        # token flat — numpy take, C-layer, no Python loop
        flat = token_matrix[u_idx].flatten().tolist()   # list[int], len = n*n_layers

        # split point
        split_item_idx = int(np.searchsorted(u_ts, split_ts, side='right'))
        split_item_idx = min(split_item_idx, n)
        split_token_pos = split_item_idx * n_layers

        # eval_cids — store as Python list of iid index (decoded later? no — we need original iids)
        # u_idx contains token_matrix row indices; we need original iid strings
        # They're passed in via iids_s (not passed here to save memory).
        # We store u_idx slice for eval_cids — caller resolves back to strings.
        eval_idx = u_idx[split_item_idx:].tolist()   # int32 indices

        # time_gaps and action_levels — vectorized
        tg = _time_gap_buckets_vec(u_ts)                  # (n,) int8
        al = _bitmap_to_level_vec(u_act.astype(np.int32)) # (n,) int8

        # repeat n_layers times per item
        tg_tok = np.repeat(tg, n_layers).tolist()
        al_tok = np.repeat(al, n_layers).tolist()

        # relative timestamps (hours from first item), repeated per layer
        rel_h = ((u_ts - u_ts[0]) / 3600.0).astype(np.float32)
        ts_tok = np.repeat(rel_h, n_layers).tolist()

        if action_l2_only:
            # zero out action_level at non-L2 positions
            for i in range(len(al_tok)):
                if (i + 1) % n_layers != 0:
                    al_tok[i] = 0

        if shift_features:
            L = n_layers
            tg_tok  = [0]  * L + tg_tok[:-L]
            al_tok  = [0]  * L + al_tok[:-L]
            ts_tok  = [0.0]* L + ts_tok[:-L]

        sequences.append({
            'flat':            flat,
            'split_token_pos': split_token_pos,
            'eval_idx':        eval_idx,
            'time_gaps':       tg_tok,
            'action_levels':   al_tok,
            'timestamps':      ts_tok,
        })

    return sequences, n_truncated, raw_items_per_user


# ── save / load helpers (unchanged interface) ─────────────────────────────────

def save_shard(sequences, path):
    """Save unified sequences (tokens + split_pos + eval_cids) as .npz."""
    if not sequences:
        np.savez_compressed(
            path,
            tokens=np.array([], dtype=np.int32),
            offsets=np.array([0], dtype=np.int64),
            split_pos=np.array([], dtype=np.int32),
            eval_cids_flat=np.array([], dtype='<U1'),
            eval_cids_offsets=np.array([0], dtype=np.int64),
        )
        return

    all_tokens = []
    offsets = [0]
    split_pos = []
    eval_cids_flat = []
    eval_cids_offsets = [0]

    has_neg = 'neg_l0' in sequences[0]
    neg_l0_flat = [] if has_neg else None
    neg_l0_offsets = [0] if has_neg else None
    neg_l0_k = None

    has_features = 'time_gaps' in sequences[0]
    all_time_gaps = [] if has_features else None
    all_action_levels = [] if has_features else None
    has_timestamps = 'timestamps' in sequences[0]
    all_timestamps = [] if has_timestamps else None

    for seq in sequences:
        all_tokens.extend(seq['tokens'])
        offsets.append(offsets[-1] + len(seq['tokens']))
        split_pos.append(seq['split_pos'])
        eval_cids_flat.extend(seq['eval_cids'])
        eval_cids_offsets.append(eval_cids_offsets[-1] + len(seq['eval_cids']))
        if has_neg:
            neg = seq['neg_l0']
            if neg_l0_k is None and len(neg) > 0:
                neg_l0_k = len(neg[0])
            for row in neg:
                neg_l0_flat.extend(row)
            neg_l0_offsets.append(neg_l0_offsets[-1] + len(neg))
        if has_features:
            all_time_gaps.extend(seq['time_gaps'])
            all_action_levels.extend(seq['action_levels'])
        if has_timestamps:
            all_timestamps.extend(seq['timestamps'])

    arrays = dict(
        tokens=np.array(all_tokens, dtype=np.int32),
        offsets=np.array(offsets, dtype=np.int64),
        split_pos=np.array(split_pos, dtype=np.int32),
        eval_cids_flat=np.array(eval_cids_flat),
        eval_cids_offsets=np.array(eval_cids_offsets, dtype=np.int64),
    )
    if has_neg:
        arrays['neg_l0_flat'] = np.array(neg_l0_flat, dtype=np.int16)
        arrays['neg_l0_offsets'] = np.array(neg_l0_offsets, dtype=np.int64)
        arrays['neg_l0_k'] = np.array(neg_l0_k or 0, dtype=np.int32)
    if has_features:
        arrays['time_gaps'] = np.array(all_time_gaps, dtype=np.int8)
        arrays['action_levels'] = np.array(all_action_levels, dtype=np.int8)
    if has_timestamps:
        arrays['timestamps'] = np.array(all_timestamps, dtype=np.float32)

    np.savez_compressed(path, **arrays)


def load_shard(path):
    """Load shard → dict with keys: tokens_list, split_pos_list, and optionals."""
    data = np.load(path, allow_pickle=True)
    tokens = data['tokens']
    offsets = data['offsets']
    split_pos = data['split_pos']

    has_neg = 'neg_l0_flat' in data
    has_features = 'time_gaps' in data.files
    has_timestamps = 'timestamps' in data.files

    if has_neg:
        neg_flat = data['neg_l0_flat']
        neg_offsets = data['neg_l0_offsets']
        neg_k = int(data['neg_l0_k'])

    time_gaps_all = data['time_gaps'] if has_features else None
    action_levels_all = data['action_levels'] if has_features else None
    timestamps_all = data['timestamps'] if has_timestamps else None

    tokens_list = []
    split_pos_list = []
    neg_l0_list = [] if has_neg else None
    time_gaps_list = [] if has_features else None
    action_levels_list = [] if has_features else None
    timestamps_list = [] if has_timestamps else None

    for i in range(len(offsets) - 1):
        start, end = int(offsets[i]), int(offsets[i + 1])
        tokens_list.append(tokens[start:end].tolist())
        split_pos_list.append(int(split_pos[i]))
        if has_neg:
            flat_start = int(neg_offsets[i]) * neg_k
            flat_end = int(neg_offsets[i + 1]) * neg_k
            chunk = neg_flat[flat_start:flat_end].astype(int).tolist()
            neg_l0_list.append([chunk[j * neg_k:(j + 1) * neg_k]
                                for j in range(int(neg_offsets[i+1]-neg_offsets[i]))])
        if has_features:
            time_gaps_list.append(time_gaps_all[start:end].tolist())
            action_levels_list.append(action_levels_all[start:end].tolist())
        if has_timestamps:
            timestamps_list.append(timestamps_all[start:end].tolist())

    result = {'tokens_list': tokens_list, 'split_pos_list': split_pos_list}
    if has_neg:
        result['neg_l0_list'] = neg_l0_list
    if has_features:
        result['time_gaps_list'] = time_gaps_list
        result['action_levels_list'] = action_levels_list
    if has_timestamps:
        result['timestamps_list'] = timestamps_list
    return result


def load_shard_full(path):
    """Load full shard data → list of dicts with tokens, split_pos, eval_cids."""
    data = np.load(path, allow_pickle=True)
    tokens = data['tokens']
    offsets = data['offsets']
    split_pos = data['split_pos']
    eval_cids_flat = data['eval_cids_flat']
    eval_cids_offsets = data['eval_cids_offsets']
    has_features = 'time_gaps' in data.files
    has_timestamps = 'timestamps' in data.files
    time_gaps_all = data['time_gaps'] if has_features else None
    action_levels_all = data['action_levels'] if has_features else None
    timestamps_all = data['timestamps'] if has_timestamps else None

    sequences = []
    for i in range(len(offsets) - 1):
        start, end = int(offsets[i]), int(offsets[i + 1])
        seq_tokens = tokens[start:end].tolist()
        cids_start = eval_cids_offsets[i]
        cids_end = eval_cids_offsets[i + 1]
        eval_cids = eval_cids_flat[cids_start:cids_end].tolist()
        seq = {
            'tokens': seq_tokens,
            'split_pos': int(split_pos[i]),
            'eval_cids': eval_cids,
        }
        if has_features:
            seq['time_gaps'] = time_gaps_all[start:end].tolist()
            seq['action_levels'] = action_levels_all[start:end].tolist()
        if has_timestamps:
            seq['timestamps'] = timestamps_all[start:end].tolist()
        sequences.append(seq)
    return sequences


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess NTP data into shards')
    parser.add_argument('--sid_cache', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_shards', type=int, default=8,
                        help='Number of output shards (should match DDP world_size)')
    parser.add_argument('--n_workers', type=int, default=32,
                        help='Worker processes for parallel sequence building')
    parser.add_argument('--n_items', type=int, default=10)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--n_eval_target', type=int, default=50000)
    parser.add_argument('--date_start', type=str, default=None)
    parser.add_argument('--date_end', type=str, default=None)
    parser.add_argument('--behavior_path', type=str, default='auto')
    parser.add_argument('--entp_weight', type=float, default=0.0)
    parser.add_argument('--exposure_neg_path', type=str, default=None)
    parser.add_argument('--behavior_v2_path', type=str, default=None)
    parser.add_argument('--entp_k', type=int, default=5)
    parser.add_argument('--shift_features', action='store_true', default=False)
    parser.add_argument('--action_l2_only', action='store_true', default=False)
    parser.add_argument('--min_action_level', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    out_meta = os.path.join(args.output_dir, 'meta.json')
    if os.path.exists(out_meta):
        print(f"Output already exists at {args.output_dir}, skipping.")
        return

    t0 = time.time()
    print("=" * 60)
    print("NTP Data Preprocessing")
    print("=" * 60)

    # ── Step 1: Load SID cache ─────────────────────────────────────────────────
    print(f"\nStep 1: Loading SID cache from {args.sid_cache}")
    sid_dict = np.load(
        os.path.join(args.sid_cache, 'semantic_ids.npy'), allow_pickle=True
    ).item()
    print(f"  SID assignments: {len(sid_dict):,}")

    content_to_tokens, n_layers, n_clusters_per_layer, _sid_to_items = \
        _parse_sid_dict(sid_dict)
    print(f"  SID: {n_layers} layers, codebooks={n_clusters_per_layer}")
    print(f"  Unique SIDs: {len(_sid_to_items):,}")

    # Build vectorized token lookup
    all_cids = list(content_to_tokens.keys())
    token_matrix = np.array([content_to_tokens[c] for c in all_cids], dtype=np.int32)
    # shape: (N_items, n_layers)
    del sid_dict, _sid_to_items

    # ── Step 2: Load behavior data ─────────────────────────────────────────────
    if args.behavior_v2_path is not None:
        print("\nStep 2: Loading behavior_v2 data")
        from eval.batch import load_behavior_v2_data
        behavior_v2_data = load_behavior_v2_data(
            local_path=args.behavior_v2_path,
            date_start=args.date_start, date_end=args.date_end)
        behavior_data = behavior_v2_data['positives']
        print(f"  Positives: {len(behavior_data['uid']):,}")
    elif args.entp_weight > 0:
        raise NotImplementedError("ENTP mode not yet ported to parallel preprocess. Use legacy torchrun path.")
    else:
        print("\nStep 2: Loading behavior data")
        from eval.batch import load_all_behavior_data
        behavior_data = load_all_behavior_data(
            date_start=args.date_start, date_end=args.date_end,
            behavior_path=args.behavior_path)
        behavior_v2_data = None
        print(f"  Interactions: {len(behavior_data['uid']):,}")

    # ── Step 3: Group + sort users ─────────────────────────────────────────────
    print("\nStep 3: Grouping and sorting users")
    uids_s, iids_s, ts_s, actions_s, starts, ends, _ = \
        _build_user_items(behavior_data, content_to_tokens,
                          min_action_level=args.min_action_level)
    del behavior_data, behavior_v2_data, content_to_tokens

    # Vectorized iid → token_matrix row index
    cat = pd.Categorical(iids_s, categories=all_cids)
    iid_idx_s = cat.codes.astype(np.int32)
    del cat, all_cids
    print(f"  iid index built, {(iid_idx_s < 0).sum()} unknown iids (should be 0)")

    # Compute split_ts
    sorted_ts = np.sort(ts_s)
    total_items = len(sorted_ts)
    split_idx = max(0, min(total_items - 1, total_items - args.n_eval_target))
    split_ts = float(sorted_ts[split_idx])
    actual_eval = int((sorted_ts > split_ts).sum())
    pct = 100.0 * split_idx / total_items if total_items > 0 else 0
    print(f"  Time split: {actual_eval:,} eval items (~{pct:.1f}th percentile, split_ts={split_ts:.0f})")
    del sorted_ts

    max_items = args.max_seq_len // n_layers
    n_users = len(starts)
    print(f"  Users: {n_users:,}, max_items_per_user={max_items}")

    # ── Step 4: Parallel sequence building ────────────────────────────────────
    n_workers = min(args.n_workers, os.cpu_count() or 1, n_users)
    print(f"\nStep 4: Building sequences ({n_workers} workers)")

    user_chunks = np.array_split(np.arange(n_users), n_workers)
    chunk_args = [
        (chunk.tolist(), starts, ends,
         iid_idx_s, ts_s, actions_s,
         token_matrix, n_layers,
         split_ts, max_items,
         args.shift_features, args.action_l2_only)
        for chunk in user_chunks
    ]

    t_seq = time.time()
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_process_chunk, chunk_args)
    print(f"  Pool.map done in {time.time() - t_seq:.1f}s")

    # Flatten results; resolve eval_idx → original iid strings
    # iid_idx_s maps position → token_matrix row; we need reverse: row → iid string
    # Rebuild idx_to_cid from iids_s (original string array, same order as iid_idx_s)
    # Use unique iid_idx values to reconstruct — faster: build once from iids_s
    idx_to_cid = {}
    for cid_str, idx in zip(iids_s, iid_idx_s):
        if idx not in idx_to_cid:
            idx_to_cid[int(idx)] = cid_str
    del iids_s, iid_idx_s, ts_s, actions_s, token_matrix

    sequences = []
    total_truncated = 0
    raw_items_all = []
    n_train_only = n_eval_only = n_both = 0

    for chunk_seqs, n_trunc, raw_items in results:
        total_truncated += n_trunc
        raw_items_all.extend(raw_items)
        for s in chunk_seqs:
            eval_cids = [idx_to_cid[i] for i in s['eval_idx']]
            split_item_idx = s['split_token_pos'] // n_layers
            n_seq = (len(s['flat']) // n_layers)
            if split_item_idx == n_seq:
                n_train_only += 1
            elif split_item_idx == 0:
                n_eval_only += 1
            else:
                n_both += 1
            sequences.append({
                'tokens':        s['flat'],
                'split_pos':     s['split_token_pos'],
                'eval_cids':     eval_cids,
                'time_gaps':     s['time_gaps'],
                'action_levels': s['action_levels'],
                'timestamps':    s['timestamps'],
            })

    del results, idx_to_cid

    total_tokens = sum(len(s['tokens']) for s in sequences)
    avg_len = total_tokens / max(len(sequences), 1)
    n_eval_items = sum(len(s['eval_cids']) for s in sequences)
    raw_arr = np.array(raw_items_all)
    pcts = np.percentile(raw_arr, [25, 50, 75, 90, 95, 99, 99.9])
    seq_stats = {
        'n_users': len(raw_arr),
        'max_items': max_items,
        'items_per_user_mean': round(float(raw_arr.mean()), 1),
        'items_per_user_p50': int(pcts[1]),
        'items_per_user_p90': int(pcts[3]),
        'items_per_user_p99': int(pcts[5]),
        'items_per_user_max': int(raw_arr.max()),
        'n_truncated': total_truncated,
        'truncated_pct': round(100.0 * total_truncated / max(len(raw_arr), 1), 2),
    }
    print(f"  Sequences: {len(sequences):,}, {total_tokens:,} tokens, avg {avg_len:.0f} tok/seq")
    print(f"  Items/user: mean={seq_stats['items_per_user_mean']}, "
          f"p50={seq_stats['items_per_user_p50']}, p90={seq_stats['items_per_user_p90']}, "
          f"p99={seq_stats['items_per_user_p99']}, max={seq_stats['items_per_user_max']}")
    print(f"  Truncated: {total_truncated:,} ({seq_stats['truncated_pct']:.2f}%)")
    print(f"  Split: {n_both:,} train+eval, {n_train_only:,} train-only, {n_eval_only:,} eval-only")
    print(f"  Eval items: {n_eval_items:,}")

    # ── Step 5: Save shards ────────────────────────────────────────────────────
    n_total = len(sequences)
    shard_size = n_total // args.n_shards
    print(f"\nStep 5: Saving {args.n_shards} shards")
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.n_shards):
        start = i * shard_size
        end = start + shard_size if i < args.n_shards - 1 else n_total
        shard_path = os.path.join(args.output_dir, f'train_shard_{i}.npz')
        save_shard(sequences[start:end], shard_path)
        file_size = os.path.getsize(shard_path) / 1e6
        print(f"  shard {i}: {end - start:,} seqs → {shard_path} ({file_size:.1f}MB)")

    del sequences

    # ── Step 6: Save metadata ─────────────────────────────────────────────────
    meta = {
        'n_layers': n_layers,
        'n_clusters_per_layer': n_clusters_per_layer,
        'n_seqs': n_total,
        'n_eval_items': n_eval_items,
        'n_shards': args.n_shards,
        'n_items': args.n_items,
        'max_seq_len': args.max_seq_len,
        'split_ts': float(split_ts),
        'sid_cache': args.sid_cache,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'has_neg_l0': False,
        'entp_k': 0,
        'min_action_level': args.min_action_level,
        'seq_stats': seq_stats,
    }
    with open(out_meta, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json saved")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Preprocessing complete! ({elapsed:.1f}s)")
    print(f"  Output: {args.output_dir}/")
    print(f"  Shards: {args.n_shards} x ~{shard_size:,} seqs")
    print(f"  Eval items: {n_eval_items:,}")
    print(f"{'=' * 60}")
    print(f"\nNext: torchrun --nproc_per_node={args.n_shards} run.py train-ntp "
          f"--preprocessed_dir {args.output_dir} --model probe")


if __name__ == '__main__':
    main()
