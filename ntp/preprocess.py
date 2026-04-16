"""NTP data preprocessing — build packed sequences and save as shards.

Single-process command that prepares data for DDP training.
Each shard contains a portion of the training data; during training,
each rank loads only its own shard (rank k loads train_shard_k.npz).

Usage:
    python run.py preprocess-ntp \
        --sid_cache experiments/sid_cache/qwen3-0.6b \
        --output_dir experiments/ntp_data/exp013 \
        --n_shards 8

Output:
    {output_dir}/
        train_shard_0.npz ... train_shard_{N-1}.npz   (per-rank training data)
        eval_data.pt                                    (eval sequences + cids + sid_to_items)
        meta.json                                       (n_layers, n_clusters, n_train, etc.)
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from gr_demo.ntp.train import build_packed_sequences


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
    parser.add_argument('--date_start', type=str, default=None,
                        help='Behavior data start date (YYYY-MM-DD)')
    parser.add_argument('--date_end', type=str, default=None,
                        help='Behavior data end date (YYYY-MM-DD)')
    return parser.parse_args()


def save_shard(sequences, path):
    """Save variable-length sequences as concatenated tokens + offsets."""
    if not sequences:
        np.savez_compressed(path, tokens=np.array([], dtype=np.int32),
                            offsets=np.array([0], dtype=np.int64))
        return

    all_tokens = []
    offsets = [0]
    for seq in sequences:
        all_tokens.extend(seq)
        offsets.append(offsets[-1] + len(seq))

    np.savez_compressed(
        path,
        tokens=np.array(all_tokens, dtype=np.int32),
        offsets=np.array(offsets, dtype=np.int64),
    )


def load_shard(path):
    """Load shard back into list of token lists."""
    data = np.load(path)
    tokens = data['tokens']
    offsets = data['offsets']
    sequences = []
    for i in range(len(offsets) - 1):
        sequences.append(tokens[offsets[i]:offsets[i+1]].tolist())
    return sequences


def save_eval_data(output_dir, eval_data, eval_cids):
    """Save eval sequences as numpy arrays. No sid_to_items — rebuilt from SID cache at eval."""
    eval_inputs_flat, eval_inputs_offsets = [], [0]
    eval_targets = []
    for inp, tgt in eval_data:
        eval_inputs_flat.extend(inp)
        eval_inputs_offsets.append(eval_inputs_offsets[-1] + len(inp))
        eval_targets.append(tgt)
    path = os.path.join(output_dir, 'eval_sequences.npz')
    np.savez_compressed(
        path,
        inputs=np.array(eval_inputs_flat, dtype=np.int32),
        offsets=np.array(eval_inputs_offsets, dtype=np.int64),
        targets=np.array(eval_targets, dtype=np.int32),
        cids=np.array(eval_cids),
    )
    size = os.path.getsize(path) / 1e6
    print(f"  eval_sequences.npz: {len(eval_data):,} samples ({size:.1f}MB)")


def load_eval_sequences(prep_dir):
    """Load eval sequences from numpy. Returns (eval_data, eval_cids)."""
    seq_data = np.load(os.path.join(prep_dir, 'eval_sequences.npz'), allow_pickle=True)
    inputs_flat = seq_data['inputs']
    offsets = seq_data['offsets']
    targets = seq_data['targets']
    cids = seq_data['cids'].tolist()

    eval_data = []
    for i in range(len(offsets) - 1):
        inp = inputs_flat[offsets[i]:offsets[i + 1]].tolist()
        tgt = targets[i].tolist()
        eval_data.append((inp, tgt))

    return eval_data, cids


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

    # ── Build packed sequences ──
    print("\nStep 3: Building packed sequences")
    train_seqs, eval_data, eval_cids, sid_to_items, n_layers, n_clusters_per_layer = \
        build_packed_sequences(
            sid_dict, behavior_data,
            n_items=args.n_items, max_seq_len=args.max_seq_len)

    del sid_dict, behavior_data  # free memory

    # ── Save shards ──
    print(f"\nStep 4: Saving {args.n_shards} training shards")
    os.makedirs(args.output_dir, exist_ok=True)

    n_total = len(train_seqs)
    shard_size = n_total // args.n_shards

    for i in range(args.n_shards):
        start = i * shard_size
        end = start + shard_size if i < args.n_shards - 1 else n_total
        shard_path = os.path.join(args.output_dir, f'train_shard_{i}.npz')
        save_shard(train_seqs[start:end], shard_path)
        file_size = os.path.getsize(shard_path) / 1e6
        print(f"  shard {i}: {end - start:,} seqs → {shard_path} ({file_size:.1f}MB)")

    del train_seqs

    # ── Save eval data (sequences only; sid_to_items rebuilt from SID cache at eval) ──
    print("\nStep 5: Saving eval data")
    save_eval_data(args.output_dir, eval_data, eval_cids)

    # ── Save metadata ──
    meta = {
        'n_layers': n_layers,
        'n_clusters_per_layer': n_clusters_per_layer,
        'n_train': n_total,
        'n_eval': len(eval_data),
        'n_shards': args.n_shards,
        'n_items': args.n_items,
        'max_seq_len': args.max_seq_len,
        'sid_cache': args.sid_cache,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    meta_path = os.path.join(args.output_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json saved")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Preprocessing complete! ({elapsed:.1f}s)")
    print(f"  Output: {args.output_dir}/")
    print(f"  Shards: {args.n_shards} × ~{shard_size:,} seqs")
    print(f"  Eval:   {len(eval_data):,} samples")
    print(f"{'=' * 60}")
    print(f"\nNext: torchrun --nproc_per_node={args.n_shards} run.py train-ntp "
          f"--preprocessed_dir {args.output_dir} --model probe")


if __name__ == '__main__':
    main()
