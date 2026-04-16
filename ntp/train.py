"""NTP Training — DDP support with unified sequences.

Each user → one complete SID token sequence with split_pos (global 80th-pctl ts).
Training loss computed on positions before split_pos; eval on positions after.

Usage:
    # 单卡 (builds data on the fly)
    python run.py train-ntp --sid_cache experiments/sid_cache/qwen3-0.6b

    # 8卡 DDP with pre-cached shards
    torchrun --nproc_per_node=8 run.py train-ntp --preprocessed_dir experiments/ntp_data/exp013

输出目录: {output_dir}/
    - probe.pt          model state_dict + config
    - train_meta.json   训练元信息 (loss, eval PPL/recall, etc.)
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from gr_demo.ntp.baseline import NTPProbe
from gr_demo.ntp.model import NTPModel


# ============================================================
# DDP helpers (borrowed from contrastive_finetune.py)
# ============================================================

def setup_ddp():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        from datetime import timedelta
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    is_main = (local_rank == 0)
    return local_rank, world_size, device, is_main


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def log(is_main, msg):
    if is_main:
        print(msg)


# ============================================================
# Data preparation
# ============================================================

def _parse_sid_dict(sid_dict):
    """Parse SID dict → content_to_tokens, n_layers, n_clusters_per_layer, sid_to_items."""
    content_to_tokens = {}
    for cid, sid_str in sid_dict.items():
        if isinstance(sid_str, str):
            content_to_tokens[cid] = [int(t) for t in sid_str.split('_')]
        else:
            content_to_tokens[cid] = [int(t) for t in sid_str]

    n_layers = len(next(iter(content_to_tokens.values())))
    n_clusters_per_layer = []
    for l in range(n_layers):
        max_token = max(tokens[l] for tokens in content_to_tokens.values())
        n_clusters_per_layer.append(max_token + 1)

    sid_to_items = defaultdict(set)
    for cid, tokens in content_to_tokens.items():
        sid_to_items['_'.join(str(t) for t in tokens)].add(cid)

    return content_to_tokens, n_layers, n_clusters_per_layer, sid_to_items


def _build_user_items(behavior_data, content_to_tokens, verbose_fn=print):
    """Vectorized user interaction grouping using numpy. Returns sorted per-user item lists."""
    import pandas as pd

    uids = behavior_data['uid']
    iids = behavior_data['iid']
    actions = behavior_data['action_bitmap']
    timestamps = behavior_data.get('first_ts')
    if timestamps is None:
        timestamps = np.arange(len(uids))

    verbose_fn(f"  Total interactions: {len(uids):,}")

    # Vectorized filter: action > 0
    action_mask = actions > 0
    uids_f = uids[action_mask]
    iids_f = iids[action_mask]
    ts_f = timestamps[action_mask]

    # Filter: iid in SID dict (vectorized via pandas isin)
    valid_iids = set(content_to_tokens.keys())
    iid_mask = pd.Index(iids_f).isin(valid_iids)
    uids_f = uids_f[iid_mask]
    iids_f = iids_f[iid_mask]
    ts_f = ts_f[iid_mask]

    verbose_fn(f"  Valid interactions: {len(uids_f):,}")

    # Sort by (uid, ts) using numpy lexsort (secondary key first)
    sort_idx = np.lexsort((ts_f, uids_f))
    uids_s = uids_f[sort_idx]
    iids_s = iids_f[sort_idx]
    ts_s = ts_f[sort_idx]

    # Group boundaries
    boundaries = np.where(uids_s[1:] != uids_s[:-1])[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(uids_s)]])

    verbose_fn(f"  Users with valid interactions: {len(starts):,}")
    return uids_s, iids_s, ts_s, starts, ends


def build_unified_sequences(sid_dict, behavior_data, n_items=10, max_seq_len=512,
                            verbose_fn=print):
    """Build unified per-user sequences with split_pos for train/eval masking.

    Each user → one complete SID token sequence. split_pos marks the boundary
    between train positions (< split_pos) and eval positions (>= split_pos),
    derived from global 80th-percentile timestamp.

    Returns:
        sequences: list of dicts with keys:
            'tokens': List[int]  — full SID token sequence (n_items * n_layers)
            'split_pos': int     — token-level split point
            'eval_cids': List    — content_ids for items after split
        n_layers: int
        n_clusters_per_layer: list
    """
    content_to_tokens, n_layers, n_clusters_per_layer, _sid_to_items = \
        _parse_sid_dict(sid_dict)
    verbose_fn(f"  SID: {n_layers} layers, codebooks={n_clusters_per_layer}")
    verbose_fn(f"  Unique SIDs: {len(_sid_to_items):,}")

    uids_s, iids_s, ts_s, starts, ends = \
        _build_user_items(behavior_data, content_to_tokens, verbose_fn)

    # Global 80th percentile timestamp for train/eval split
    split_ts = np.percentile(ts_s, 80)
    verbose_fn(f"  Time split at 80th percentile: {split_ts}")

    max_items = max_seq_len // n_layers
    sequences = []
    n_train_only = 0
    n_eval_only = 0
    n_both = 0

    for u in range(len(starts)):
        s, e = starts[u], ends[u]
        n = e - s
        if n < 2:
            continue

        user_iids = iids_s[s:e]
        user_ts = ts_s[s:e]
        user_tokens = [content_to_tokens[iid] for iid in user_iids]

        # Truncate to most recent max_items
        if n > max_items:
            offset = n - max_items
            user_iids = user_iids[offset:]
            user_ts = user_ts[offset:]
            user_tokens = user_tokens[offset:]
            n = max_items

        # Find split_item_idx: first item where ts > split_ts
        split_item_idx = n  # default: all items are train
        for i in range(n):
            if user_ts[i] > split_ts:
                split_item_idx = i
                break

        split_token_pos = split_item_idx * n_layers

        # eval_cids: content_ids for items at/after split
        eval_cids = [user_iids[i] for i in range(split_item_idx, n)]

        # Flatten tokens
        flat = []
        for toks in user_tokens:
            flat.extend(toks)

        sequences.append({
            'tokens': flat,
            'split_pos': split_token_pos,
            'eval_cids': eval_cids,
        })

        # Stats
        if split_item_idx == n:
            n_train_only += 1
        elif split_item_idx == 0:
            n_eval_only += 1
        else:
            n_both += 1

    if not sequences:
        raise ValueError("No valid sequences")

    total_tokens = sum(len(s['tokens']) for s in sequences)
    avg_len = total_tokens / len(sequences)
    n_eval_items = sum(len(s['eval_cids']) for s in sequences)
    verbose_fn(f"  Unified sequences: {len(sequences):,}, "
               f"{total_tokens:,} tokens, avg {avg_len:.0f} tok/seq")
    verbose_fn(f"  Split: {n_both:,} train+eval, "
               f"{n_train_only:,} train-only, {n_eval_only:,} eval-only")
    verbose_fn(f"  Eval items: {n_eval_items:,}")

    return sequences, n_layers, n_clusters_per_layer, split_ts


# ============================================================
# Packed sequence dataset + collate
# ============================================================

class UnifiedSequenceDataset(torch.utils.data.Dataset):
    """Dataset of unified per-user sequences with split_pos."""

    def __init__(self, tokens_list, split_pos_list):
        """
        Args:
            tokens_list: list of 1D token lists (variable length)
            split_pos_list: list of int split positions
        """
        self.tokens_list = tokens_list
        self.split_pos_list = split_pos_list

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        return (torch.tensor(self.tokens_list[idx], dtype=torch.long),
                self.split_pos_list[idx])


def unified_collate_fn(batch):
    """Right-pad variable-length sequences, return split_pos tensor."""
    seqs, split_positions = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    split_pos = torch.tensor(split_positions, dtype=torch.long)
    max_len = lengths.max().item()
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded, lengths, split_pos


# ============================================================
# Training
# ============================================================

def train_packed(
    tokens_list,
    split_pos_list,
    n_clusters_per_layer,
    n_layers,
    n_items,
    local_rank,
    world_size,
    device,
    is_main,
    batch_size=4096,
    lr=1e-3,
    embed_dim=256,
    n_heads=8,
    n_transformer_layers=6,
    max_seq_len=512,
    model_type='s-tier',
    ffn_dim=512,
    pre_sharded=False,
):
    """Train NTPModel or NTPProbe with unified sequences (causal LM style).

    Loss is computed only on train positions (pos < split_pos) per sequence.

    Args:
        tokens_list: list of 1D token lists (variable length per user)
        split_pos_list: list of int split positions per sequence
        pre_sharded: if True, data is already this rank's shard (from preprocess-ntp).
    """

    if model_type == 's-tier':
        model = NTPModel(
            n_clusters_per_layer=n_clusters_per_layer,
            n_sid_layers=n_layers,
            n_items=n_items,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            use_moe=True,
            n_experts=8,
            top_k=2,
            expert_dim=1024,
            parallel=False,  # packed = always causal AR
            max_seq_len=max_seq_len,
        ).to(device)
    else:
        model = NTPProbe(
            n_clusters_per_layer=n_clusters_per_layer,
            n_sid_layers=n_layers,
            n_items=n_items,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            ffn_dim=ffn_dim,
            parallel=False,  # packed = always causal AR
            max_seq_len=max_seq_len,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(is_main, f"  {model_type} (packed): {n_params / 1e6:.1f}M params, max_seq={max_seq_len}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Shard data per rank to save memory (each rank only holds 1/N)
    if pre_sharded:
        tokens_shard = tokens_list
        split_pos_shard = split_pos_list
    elif world_size > 1:
        n_total = len(tokens_list)
        shard_size = n_total // world_size
        shard_start = local_rank * shard_size
        shard_end = shard_start + shard_size if local_rank < world_size - 1 else n_total
        tokens_shard = tokens_list[shard_start:shard_end]
        split_pos_shard = split_pos_list[shard_start:shard_end]
        del tokens_list, split_pos_list
        log(is_main, f"  Rank {local_rank}: shard {shard_start}..{shard_end} "
                      f"({len(tokens_shard):,} seqs)")
    else:
        tokens_shard = tokens_list
        split_pos_shard = split_pos_list

    dataset = UnifiedSequenceDataset(tokens_shard, split_pos_shard)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=unified_collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
    n_batches = len(train_loader)

    log(is_main, f"  Training: {len(tokens_shard):,} seqs/rank, "
                 f"{n_batches} batches/epoch, batch_size={batch_size}, "
                 f"world_size={world_size}")

    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (padded, lengths, split_positions) in enumerate(train_loader):
        padded = padded.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        split_positions = split_positions.to(device, non_blocking=True)
        B, T = padded.shape

        # LM-style: input = tokens[:-1], target = tokens[1:]
        input_tokens = padded[:, :-1]
        target_tokens = padded[:, 1:]

        # Valid mask: position i is valid if i+1 < length
        arange = torch.arange(T - 1, device=device).unsqueeze(0)
        valid_mask = arange < (lengths.unsqueeze(1) - 1)

        # Train mask: position i predicts token[i+1]. Token[i+1] is train when
        # i+1 < split_pos, i.e. i < split_pos - 1.
        train_mask = valid_mask & (arange < (split_positions.unsqueeze(1) - 1))

        loss = model(input_tokens, packed_targets=target_tokens, packed_mask=train_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if is_main and (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            seqs_per_sec = (step + 1) * batch_size * world_size / elapsed
            remaining = (n_batches - step - 1) / ((step + 1) / elapsed)
            eta = format_eta(remaining)
            print(f"    step {step+1}/{n_batches}: "
                  f"loss={total_loss/(step+1):.4f}, "
                  f"{seqs_per_sec:.0f} seqs/s, ETA {eta}")

    avg_loss = total_loss / n_batches
    elapsed = time.time() - t0
    log(is_main, f"  Train done: loss={avg_loss:.4f} ({elapsed:.1f}s)")

    raw_model = model.module if isinstance(model, DDP) else model
    return raw_model.cpu(), avg_loss, n_params, model_type


def format_eta(seconds):
    """Format seconds into human-readable ETA string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m{seconds%60:.0f}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"



# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(output_dir, probe, n_clusters_per_layer, n_layers, n_items,
                    avg_loss, n_params, sid_cache_dir, preprocessed_dir,
                    model_type='probe', n_train=0, n_eval=0):
    """Save probe checkpoint + train_meta.json (rank 0 only).

    Eval data lives in preprocessed shards — not duplicated here.
    sid_to_items is rebuilt from SID cache at eval time.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Model checkpoint
    if model_type == 's-tier':
        probe_config = {
            'model_type': 's-tier',
            'n_clusters_per_layer': n_clusters_per_layer,
            'n_sid_layers': n_layers,
            'n_items': n_items,
            'embed_dim': probe.embed_dim,
            'n_heads': probe.layers[0].attn.num_heads,
            'n_transformer_layers': len(probe.layers),
            'use_moe': True,
            'n_experts': 8,
            'top_k': 2,
            'expert_dim': 1024,
            'parallel': probe.parallel,
            'max_seq_len': probe.max_seq_len,
        }
    else:
        probe_config = {
            'model_type': 'probe',
            'n_clusters_per_layer': n_clusters_per_layer,
            'n_sid_layers': n_layers,
            'n_items': n_items,
            'embed_dim': probe.embed_dim,
            'n_heads': probe.decoder.layers[0].self_attn.num_heads,
            'n_transformer_layers': len(probe.decoder.layers),
            'ffn_dim': probe.decoder.layers[0].linear1.out_features,
            'parallel': probe.parallel,
            'max_seq_len': probe.max_seq_len,
        }
    torch.save({
        'model_state_dict': probe.state_dict(),
        'config': probe_config,
    }, os.path.join(output_dir, 'probe.pt'))

    # 2. Train metadata
    meta = {
        'n_train': n_train,
        'n_eval': n_eval,
        'n_clusters_per_layer': n_clusters_per_layer,
        'n_layers': n_layers,
        'n_items': n_items,
        'n_params': n_params,
        'train_loss': round(avg_loss, 6),
        'sid_cache': sid_cache_dir,
        'preprocessed_dir': preprocessed_dir,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(output_dir, 'train_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {output_dir}/")
    print(f"    probe.pt      ({os.path.getsize(os.path.join(output_dir, 'probe.pt')) / 1e6:.1f}MB)")
    print(f"    train_meta.json")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train NTP Probe (DDP)')
    parser.add_argument('--sid_cache', type=str, default=None,
                        help='Path to preprocess-sid cache dir (required unless --preprocessed_dir)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir (default: experiments/ntp_checkpoints/{name})')
    parser.add_argument('--name', type=str, default='default',
                        help='Experiment name for output subdir')
    parser.add_argument('--n_items', type=int, default=10,
                        help='Number of history items per sequence')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_transformer_layers', type=int, default=2)
    parser.add_argument('--ffn_dim', type=int, default=512)
    parser.add_argument('--model', type=str, default='probe',
                        choices=['probe', 's-tier'],
                        help='Model: probe (2L dense, ~5M) or s-tier (6L MoE, ~39.5M)')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Max packed sequence length in tokens (s-tier only)')
    parser.add_argument('--date_start', type=str, default=None,
                        help='Behavior data start date (YYYY-MM-DD). Default: config DEFAULT_DATE_START')
    parser.add_argument('--date_end', type=str, default=None,
                        help='Behavior data end date (YYYY-MM-DD). Default: config DEFAULT_DATE_END')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                        help='Pre-cached shard directory from preprocess-ntp. '
                             'If set, skips data building and loads per-rank shard directly.')
    return parser.parse_args()


def _run_inline_eval(probe, sid_cache_dir, preprocessed_dir, n_layers,
                     n_clusters_per_layer, device, batch_size=2048):
    """Run eval immediately after training — model + data already in memory.

    Returns dict with PPL, depth_hit@10, recall@K, etc.
    """
    from gr_demo.ntp.eval import (
        _batched_teacher_forced_eval, _beam_search_recall,
        _build_sid_to_items, _load_eval_sequences,
    )
    from gr_demo.ntp.model import SIDTrie

    # Load eval sequences from preprocessed shards (all shards, with eval_cids)
    if preprocessed_dir and os.path.isdir(preprocessed_dir):
        prep_meta_path = os.path.join(preprocessed_dir, 'meta.json')
        with open(prep_meta_path) as f:
            prep_meta = json.load(f)
        eval_sequences = _load_eval_sequences(preprocessed_dir, prep_meta['n_shards'])
    else:
        print("  Warning: no preprocessed_dir, skipping inline eval")
        return {}

    print(f"  Eval sequences: {len(eval_sequences):,}")

    probe = probe.to(device)
    probe.eval()

    # Teacher-forced (PPL + hit@10)
    print("  Running teacher-forced eval...")
    with torch.no_grad():
        tf_results = _batched_teacher_forced_eval(
            probe, eval_sequences, n_layers, device,
            batch_size=batch_size, verbose=True)

    print(f"  PPL: {tf_results['ppl']:.2f}")
    print(f"  Depth hit@10: {[f'{h:.3f}' for h in tf_results['depth_hit_10']]}")

    # Beam search recall (5K subsample)
    print(f"  Building sid_to_items from {sid_cache_dir}")
    sid_to_items = _build_sid_to_items(sid_cache_dir)
    sid_trie = SIDTrie(sid_to_items, n_layers)

    print("  Running beam search recall (5K)...")
    with torch.no_grad():
        beam_results = _beam_search_recall(
            probe, eval_sequences, sid_trie, sid_to_items, n_layers,
            device, recall_beam_size=500, n_recall_samples=5000, verbose=True)

    # Move probe back to CPU for checkpoint saving
    probe.cpu()

    # Combine results
    results = {
        'ppl': round(tf_results['ppl'], 4),
        'avg_loss': round(tf_results['avg_loss'], 6),
        'depth_hit@10': [round(h, 4) for h in tf_results['depth_hit_10']],
        'n_eval_positions': tf_results['n_eval_positions'],
        'n_eval_sequences': len(eval_sequences),
    }
    results.update(beam_results)

    for k in ['item_recall@10', 'item_recall@50', 'item_recall@100', 'item_recall@500']:
        if k in results:
            print(f"  {k}: {results[k]:.4f}")

    return results


def main():
    args = parse_args()
    local_rank, world_size, device, is_main = setup_ddp()
    model_type = args.model

    if not args.preprocessed_dir and not args.sid_cache:
        raise ValueError("Either --sid_cache or --preprocessed_dir is required")

    log(is_main, "=" * 60)
    log(is_main, f"NTP Training — {model_type}" +
                 (f" (DDP x{world_size})" if world_size > 1 else ""))
    log(is_main, "=" * 60)

    # ── Fast path: load pre-cached shards from preprocess-ntp ──
    if args.preprocessed_dir:
        log(is_main, f"\nLoading pre-cached data from {args.preprocessed_dir}")
        meta_path = os.path.join(args.preprocessed_dir, 'meta.json')
        with open(meta_path) as f:
            prep_meta = json.load(f)

        n_layers = prep_meta['n_layers']
        n_clusters_per_layer = prep_meta['n_clusters_per_layer']
        n_items = prep_meta['n_items']
        max_seq_len = prep_meta['max_seq_len']
        sid_cache_dir = prep_meta['sid_cache']
        preprocessed_dir = args.preprocessed_dir

        # Each rank loads only its own shard
        shard_path = os.path.join(args.preprocessed_dir, f'train_shard_{local_rank}.npz')
        if not os.path.exists(shard_path):
            raise FileNotFoundError(
                f"Shard {shard_path} not found. "
                f"Expected {prep_meta['n_shards']} shards but world_size={world_size}. "
                f"Re-run preprocess-ntp with --n_shards {world_size}.")
        from gr_demo.ntp.preprocess import load_shard
        tokens_list, split_pos_list = load_shard(shard_path)
        log(is_main, f"  Rank {local_rank}: loaded {len(tokens_list):,} seqs from shard")
        log(is_main, f"  Layers: {n_layers}, n_items: {n_items}, max_seq_len: {max_seq_len}")

        n_train = prep_meta['n_seqs']
        n_eval = prep_meta['n_eval_items']

    # ── Slow path: build data on rank 0, share to other ranks ──
    else:
        log(is_main, f"\nStep 1: Loading SID cache from {args.sid_cache}")
        sid_cache_dir = args.sid_cache
        n_items = args.n_items
        max_seq_len = args.max_seq_len
        preprocessed_dir = None

        if is_main:
            sid_dict = np.load(
                os.path.join(args.sid_cache, 'semantic_ids.npy'), allow_pickle=True
            ).item()
            log(is_main, f"  SID assignments: {len(sid_dict):,}")

            log(is_main, "\nStep 2: Loading behavior data")
            from gr_demo.eval.batch import load_all_behavior_data
            behavior_data = load_all_behavior_data(
                date_start=args.date_start, date_end=args.date_end)
            log(is_main, f"  Interactions: {len(behavior_data['uid']):,}")

            log(is_main, "\nStep 3: Building unified sequences")
            sequences, n_layers, n_clusters_per_layer, _split_ts = \
                build_unified_sequences(
                    sid_dict, behavior_data,
                    n_items=n_items, max_seq_len=max_seq_len)

            del sid_dict, behavior_data

            tokens_list = [s['tokens'] for s in sequences]
            split_pos_list = [s['split_pos'] for s in sequences]
            n_eval = sum(len(s['eval_cids']) for s in sequences)
            n_train = len(sequences)
            del sequences

            if world_size > 1:
                shared_dir = os.path.join(args.sid_cache, '_train_tmp')
                os.makedirs(shared_dir, exist_ok=True)
                np.save(os.path.join(shared_dir, 'tokens_list.npy'),
                        tokens_list, allow_pickle=True)
                np.save(os.path.join(shared_dir, 'split_pos_list.npy'),
                        split_pos_list, allow_pickle=True)
            meta = (n_layers, n_clusters_per_layer, n_train, n_eval)
        else:
            tokens_list = split_pos_list = None
            meta = None

        if world_size > 1:
            meta_list = [meta]
            dist.broadcast_object_list(meta_list, src=0)
            meta = meta_list[0]

            if is_main:
                shared_path_list = [os.path.join(args.sid_cache, '_train_tmp')]
            else:
                shared_path_list = [None]
            dist.broadcast_object_list(shared_path_list, src=0)
            shared_dir = shared_path_list[0]

            if not is_main:
                tokens_list = np.load(
                    os.path.join(shared_dir, 'tokens_list.npy'),
                    allow_pickle=True).tolist()
                split_pos_list = np.load(
                    os.path.join(shared_dir, 'split_pos_list.npy'),
                    allow_pickle=True).tolist()

            dist.barrier()
            if is_main:
                os.remove(os.path.join(shared_dir, 'tokens_list.npy'))
                os.remove(os.path.join(shared_dir, 'split_pos_list.npy'))
                try:
                    os.rmdir(shared_dir)
                except OSError:
                    pass

        n_layers, n_clusters_per_layer, n_train, n_eval = meta
        log(is_main, f"  Seqs: {len(tokens_list):,}, Eval items: {n_eval:,}, Layers: {n_layers}")

    # ── Train (both probe and s-tier use packed sequences) ──
    if model_type == 's-tier':
        embed_dim, n_heads, n_transformer_layers = 256, 8, 6
        lr = 1e-3
        ffn_dim = 512
    else:
        embed_dim = args.embed_dim
        n_heads = args.n_heads
        n_transformer_layers = args.n_transformer_layers
        ffn_dim = args.ffn_dim
        lr = args.lr

    log(is_main, f"\nStep 4: Training ({model_type}, packed)")
    probe, avg_loss, n_params, model_type = train_packed(
        tokens_list=tokens_list,
        split_pos_list=split_pos_list,
        n_clusters_per_layer=n_clusters_per_layer,
        n_layers=n_layers,
        n_items=n_items,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main=is_main,
        batch_size=args.batch_size,
        lr=lr,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
        max_seq_len=max_seq_len,
        model_type=model_type,
        ffn_dim=ffn_dim,
        pre_sharded=bool(args.preprocessed_dir),
    )

    # ── Save + Eval (rank 0 only) ──
    if is_main:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = args.output_dir or os.path.join(
            repo_root, 'experiments', 'ntp_checkpoints', args.name)

        log(is_main, f"\nStep 5: Saving checkpoint")
        save_checkpoint(
            output_dir=output_dir,
            probe=probe,
            n_clusters_per_layer=n_clusters_per_layer,
            n_layers=n_layers,
            n_items=n_items,
            avg_loss=avg_loss,
            n_params=n_params,
            sid_cache_dir=sid_cache_dir,
            preprocessed_dir=preprocessed_dir or '',
            model_type=model_type,
            n_train=n_train,
            n_eval=n_eval,
        )

        # ── Inline eval on eval positions ──
        log(is_main, f"\nStep 6: Eval (teacher-forced + beam search)")
        eval_results = _run_inline_eval(
            probe, sid_cache_dir, preprocessed_dir, n_layers,
            n_clusters_per_layer, device, args.batch_size)

        # Append eval results to train_meta.json
        meta_path = os.path.join(output_dir, 'train_meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        meta['eval'] = eval_results
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        log(is_main, f"\n{'=' * 60}")
        log(is_main, "Training + eval complete!")
        log(is_main, f"  Checkpoint: {output_dir}/")
        log(is_main, f"  PPL: {eval_results.get('ppl', 'N/A')}")
        for k in ['item_recall@10', 'item_recall@50', 'item_recall@100', 'item_recall@500']:
            if k in eval_results:
                log(is_main, f"  {k}: {eval_results[k]:.4f}")
        log(is_main, f"{'=' * 60}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
