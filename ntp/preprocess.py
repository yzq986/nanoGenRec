"""NTP data preprocessing — build unified sequences and save as shards.

Single-process command that prepares data for DDP training.
Each shard contains unified per-user sequences (tokens + split_pos + eval_cids).
During training, each rank loads only its own shard.

Usage:
    python run.py preprocess-ntp \\
        --sid_cache experiments/sid_cache/qwen3-0.6b \\
        --output_dir experiments/ntp_data/exp013 \\
        --n_shards 8

Output:
    {output_dir}/
        train_shard_0.npz ... train_shard_{N-1}.npz   (per-rank data)
        meta.json                                       (n_layers, n_clusters, etc.)
"""

import argparse
import json
import os
import time

import numpy as np

from gr_demo.ntp.train import build_unified_sequences


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess NTP data into shards')
    parser.add_argument('--sid_cache', type=str, required=True,
                        help='Path to preprocess-sid cache dir')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for shards')
    parser.add_argument('--n_shards', type=int, default=8,
                        help='Number of training shards (should match world_size)')
    parser.add_argument('--n_items', type=int, default=10,
                        help='Number of history items per sequence')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Max packed sequence length in tokens')
    parser.add_argument('--n_eval_target', type=int, default=50000,
                        help='Target number of eval items (determines time split)')
    parser.add_argument('--date_start', type=str, default=None,
                        help='Behavior data start date (YYYY-MM-DD)')
    parser.add_argument('--date_end', type=str, default=None,
                        help='Behavior data end date (YYYY-MM-DD)')
    parser.add_argument('--entp_weight', type=float, default=0.0,
                        help='If > 0, load exposure data and build neg_l0 for ENTP loss')
    parser.add_argument('--entp_k', type=int, default=5,
                        help='Max negative L0 tokens per position for ENTP')
    return parser.parse_args()


def save_shard(sequences, path):
    """Save unified sequences (tokens + split_pos + eval_cids) as .npz.

    Args:
        sequences: list of dicts with 'tokens', 'split_pos', 'eval_cids',
                   and optionally 'neg_l0' (list of K ints per item).
    """
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

    for seq in sequences:
        all_tokens.extend(seq['tokens'])
        offsets.append(offsets[-1] + len(seq['tokens']))
        split_pos.append(seq['split_pos'])
        eval_cids_flat.extend(seq['eval_cids'])
        eval_cids_offsets.append(eval_cids_offsets[-1] + len(seq['eval_cids']))
        if has_neg:
            neg = seq['neg_l0']  # list of lists, shape (n_items, K)
            if neg_l0_k is None and len(neg) > 0:
                neg_l0_k = len(neg[0])
            for row in neg:
                neg_l0_flat.extend(row)
            neg_l0_offsets.append(neg_l0_offsets[-1] + len(neg))

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

    np.savez_compressed(path, **arrays)


def load_shard(path):
    """Load shard → (tokens_list, split_pos_list[, neg_l0_list]).

    Returns only what training needs. eval_cids loaded separately via load_shard_eval_cids.
    Returns 3-tuple if shard contains neg_l0 data, 2-tuple otherwise.
    """
    data = np.load(path, allow_pickle=True)
    tokens = data['tokens']
    offsets = data['offsets']
    split_pos = data['split_pos']

    has_neg = 'neg_l0_flat' in data
    if has_neg:
        neg_flat = data['neg_l0_flat']
        neg_offsets = data['neg_l0_offsets']
        neg_k = int(data['neg_l0_k'])

    tokens_list = []
    split_pos_list = []
    neg_l0_list = [] if has_neg else None
    for i in range(len(offsets) - 1):
        tokens_list.append(tokens[offsets[i]:offsets[i + 1]].tolist())
        split_pos_list.append(int(split_pos[i]))
        if has_neg:
            n_items = int(neg_offsets[i + 1] - neg_offsets[i])
            flat_start = int(neg_offsets[i]) * neg_k
            flat_end = int(neg_offsets[i + 1]) * neg_k
            chunk = neg_flat[flat_start:flat_end].astype(int).tolist()
            neg_l0_list.append([chunk[j * neg_k:(j + 1) * neg_k]
                                for j in range(n_items)])

    if has_neg:
        return tokens_list, split_pos_list, neg_l0_list
    return tokens_list, split_pos_list


def load_shard_full(path):
    """Load full shard data → list of dicts with tokens, split_pos, eval_cids."""
    data = np.load(path, allow_pickle=True)
    tokens = data['tokens']
    offsets = data['offsets']
    split_pos = data['split_pos']
    eval_cids_flat = data['eval_cids_flat']
    eval_cids_offsets = data['eval_cids_offsets']

    sequences = []
    for i in range(len(offsets) - 1):
        seq_tokens = tokens[offsets[i]:offsets[i + 1]].tolist()
        cids_start = eval_cids_offsets[i]
        cids_end = eval_cids_offsets[i + 1]
        eval_cids = eval_cids_flat[cids_start:cids_end].tolist()
        sequences.append({
            'tokens': seq_tokens,
            'split_pos': int(split_pos[i]),
            'eval_cids': eval_cids,
        })
    return sequences


def main():
    args = parse_args()
    t0 = time.time()

    print("=" * 60)
    print("NTP Data Preprocessing")
    print("=" * 60)

    # ── Load SID cache ──
    print(f"\nStep 1: Loading SID cache from {args.sid_cache}")
    sid_dict = np.load(
        os.path.join(args.sid_cache, 'semantic_ids.npy'), allow_pickle=True
    ).item()
    print(f"  SID assignments: {len(sid_dict):,}")

    # ── Load behavior data ──
    print("\nStep 2: Loading behavior data")
    from gr_demo.eval.batch import load_all_behavior_data
    behavior_data = load_all_behavior_data(
        date_start=args.date_start, date_end=args.date_end)
    print(f"  Interactions: {len(behavior_data['uid']):,}")

    # ── Load exposure data (ENTP) ──
    exposure_data = None
    if args.entp_weight > 0:
        print("\nStep 2b: Loading exposure data (ENTP)")
        from gr_demo.eval.batch import load_all_exposure_data
        exposure_data = load_all_exposure_data(
            date_start=args.date_start, date_end=args.date_end)

    # ── Build unified sequences ──
    print("\nStep 3: Building unified sequences")
    sequences, n_layers, n_clusters_per_layer, split_ts = \
        build_unified_sequences(
            sid_dict, behavior_data,
            n_items=args.n_items, max_seq_len=args.max_seq_len,
            n_eval_target=args.n_eval_target,
            exposure_data=exposure_data, entp_k=args.entp_k)

    del sid_dict, behavior_data, exposure_data  # free memory

    # ── Save shards ──
    n_total = len(sequences)
    n_eval_items = sum(len(s['eval_cids']) for s in sequences)
    shard_size = n_total // args.n_shards

    print(f"\nStep 4: Saving {args.n_shards} shards")
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.n_shards):
        start = i * shard_size
        end = start + shard_size if i < args.n_shards - 1 else n_total
        shard_path = os.path.join(args.output_dir, f'train_shard_{i}.npz')
        save_shard(sequences[start:end], shard_path)
        file_size = os.path.getsize(shard_path) / 1e6
        print(f"  shard {i}: {end - start:,} seqs -> {shard_path} ({file_size:.1f}MB)")

    has_neg = args.entp_weight > 0
    del sequences

    # ── Save metadata ──
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
        'has_neg_l0': has_neg,
        'entp_k': args.entp_k if has_neg else 0,
    }
    meta_path = os.path.join(args.output_dir, 'meta.json')
    with open(meta_path, 'w') as f:
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
