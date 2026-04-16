"""NTP Probe 训练 — DDP 支持。

将 NTP probe 的数据准备 + 训练从 metric 中拆出，支持多卡 DDP。
训练产物保存到 checkpoint 目录，eval 阶段只加载 checkpoint 做推理。

Usage:
    # 单卡
    python run.py train-ntp --sid_cache experiments/sid_cache/qwen3-0.6b

    # 8卡 DDP
    torchrun --nproc_per_node=8 run.py train-ntp --sid_cache experiments/sid_cache/qwen3-0.6b

输出目录: {output_dir}/
    - probe.pt          NTPProbe state_dict + model config
    - eval_data.pt      eval sequences + eval_cids + sid_to_items
    - train_meta.json   训练元信息 (loss, n_train, n_eval, etc.)
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
from torch.utils.data.distributed import DistributedSampler

from gr_demo.config import MODEL_CONFIGS, EFS_EMBEDDING_CACHE
from gr_demo.ntp.baseline import NTPProbe, SIDSequenceDataset
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


def build_packed_sequences(sid_dict, behavior_data, n_items=10, max_seq_len=512,
                           verbose_fn=print):
    """Build packed per-user sequences for training + sliding windows for eval.

    Training: each user → one long SID token sequence (causal mask training).
    Eval: sliding windows of n_items → target (same format as old, for beam search).

    Returns:
        train_seqs: list of 1D lists (variable-length SID token sequences)
        eval_data: list of (input_tokens, target_tokens)
        eval_cids: list of target content_ids
        sid_to_items: dict
        n_layers: int
        n_clusters_per_layer: list
    """
    content_to_tokens, n_layers, n_clusters_per_layer, sid_to_items = \
        _parse_sid_dict(sid_dict)
    verbose_fn(f"  SID: {n_layers} layers, codebooks={n_clusters_per_layer}")
    verbose_fn(f"  Unique SIDs: {len(sid_to_items):,}")

    uids_s, iids_s, ts_s, starts, ends = \
        _build_user_items(behavior_data, content_to_tokens, verbose_fn)

    # Global 80th percentile timestamp for train/eval split
    split_ts = np.percentile(ts_s, 80)
    verbose_fn(f"  Time split at 80th percentile: {split_ts}")

    max_items = max_seq_len // n_layers
    train_seqs = []
    eval_data = []
    eval_cids = []

    for u in range(len(starts)):
        s, e = starts[u], ends[u]
        n = e - s
        if n < 2:
            continue

        user_iids = iids_s[s:e]
        user_ts = ts_s[s:e]

        # Token lists for each item
        user_tokens = [content_to_tokens[iid] for iid in user_iids]

        # Train: items with ts <= split_ts → packed sequence
        train_mask = user_ts <= split_ts
        n_train = int(train_mask.sum())
        if n_train >= 2:
            # Keep most recent max_items
            train_items = [user_tokens[i] for i in range(n) if train_mask[i]]
            if len(train_items) > max_items:
                train_items = train_items[-max_items:]
            flat = []
            for toks in train_items:
                flat.extend(toks)
            train_seqs.append(flat)

        # Eval: items with ts > split_ts, using full preceding history as context
        max_context_items = max_seq_len // n_layers  # match training context window
        for i in range(n):
            if user_ts[i] <= split_ts:
                continue
            if i < 2:  # need at least 2 preceding items
                continue
            # Full history context: all items before position i (up to max_context_items)
            ctx_start = max(0, i - max_context_items)
            input_tokens = []
            for j in range(ctx_start, i):
                input_tokens.extend(user_tokens[j])
            target_tokens = user_tokens[i]
            eval_data.append((input_tokens, target_tokens))
            eval_cids.append(user_iids[i])

    if not train_seqs:
        raise ValueError("No valid training sequences")

    total_tokens = sum(len(s) for s in train_seqs)
    avg_len = total_tokens / len(train_seqs)
    verbose_fn(f"  Packed train: {len(train_seqs):,} seqs, "
               f"{total_tokens:,} tokens, avg {avg_len:.0f} tok/seq")
    verbose_fn(f"  Eval windows: {len(eval_data):,}")
    verbose_fn(f"  Unique SIDs with items: {len(sid_to_items):,}")

    return train_seqs, eval_data, eval_cids, sid_to_items, n_layers, n_clusters_per_layer


def build_sequences(sid_dict, behavior_data, n_items=10, verbose_fn=print):
    """Legacy sliding window builder for NTPProbe. Kept for backward compat."""
    content_to_tokens, n_layers, n_clusters_per_layer, sid_to_items = \
        _parse_sid_dict(sid_dict)
    verbose_fn(f"  SID: {n_layers} layers, codebooks={n_clusters_per_layer}")

    uids_s, iids_s, ts_s, starts, ends = \
        _build_user_items(behavior_data, content_to_tokens, verbose_fn)

    all_samples = []
    for u in range(len(starts)):
        s, e = starts[u], ends[u]
        n = e - s
        if n < n_items + 1:
            continue
        user_iids = iids_s[s:e]
        user_ts = ts_s[s:e]
        user_tokens = [content_to_tokens[iid] for iid in user_iids]

        for i in range(n - n_items):
            input_tokens = []
            for j in range(n_items):
                input_tokens.extend(user_tokens[i + j])
            target_tokens = user_tokens[i + n_items]
            target_cid = user_iids[i + n_items]
            target_ts = user_ts[i + n_items]
            all_samples.append((input_tokens, target_tokens, target_cid, target_ts))

    if not all_samples:
        raise ValueError("No valid sequences generated")

    all_samples.sort(key=lambda x: x[3])
    split_idx = int(len(all_samples) * 0.8)

    train_data = [(s[0], s[1]) for s in all_samples[:split_idx]]
    eval_data = [(s[0], s[1]) for s in all_samples[split_idx:]]
    eval_cids = [s[2] for s in all_samples[split_idx:]]

    verbose_fn(f"  Total samples: {len(all_samples):,} "
               f"(train={len(train_data):,}, eval={len(eval_data):,})")
    verbose_fn(f"  Unique SIDs with items: {len(sid_to_items):,}")

    return train_data, eval_data, eval_cids, sid_to_items, n_layers, n_clusters_per_layer


# ============================================================
# Packed sequence dataset + collate
# ============================================================

class PackedSequenceDataset(torch.utils.data.Dataset):
    """Dataset of variable-length packed SID token sequences."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def packed_collate_fn(batch):
    """Right-pad variable-length sequences to max length in batch."""
    lengths = torch.tensor([len(seq) for seq in batch], dtype=torch.long)
    max_len = lengths.max().item()
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded, lengths


class EvalSequenceDataset(torch.utils.data.Dataset):
    """Variable-length eval dataset: (input_tokens, target_tokens) where inputs vary in length."""

    def __init__(self, samples):
        self.inputs = [torch.tensor(s[0], dtype=torch.long) for s in samples]
        self.targets = [torch.tensor(s[1], dtype=torch.long) for s in samples]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def eval_collate_fn(batch):
    """Collate variable-length inputs with right-padding. Targets are fixed-size."""
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(inp) for inp in inputs], dtype=torch.long)
    max_len = lengths.max().item()
    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    for i, inp in enumerate(inputs):
        padded_inputs[i, :len(inp)] = inp
    targets = torch.stack(targets)
    return padded_inputs, targets, lengths


# ============================================================
# Training
# ============================================================

def train_packed(
    train_seqs,
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
    """Train NTPModel or NTPProbe with packed user sequences (causal LM style).

    Args:
        pre_sharded: if True, train_seqs is already this rank's shard (from preprocess-ntp).
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
        train_seqs_shard = train_seqs  # already per-rank from preprocess-ntp
    elif world_size > 1:
        n_total = len(train_seqs)
        shard_size = n_total // world_size
        shard_start = local_rank * shard_size
        shard_end = shard_start + shard_size if local_rank < world_size - 1 else n_total
        train_seqs_shard = train_seqs[shard_start:shard_end]
        del train_seqs  # free full copy
        log(is_main, f"  Rank {local_rank}: shard {shard_start}..{shard_end} "
                      f"({len(train_seqs_shard):,} seqs)")
    else:
        train_seqs_shard = train_seqs

    dataset = PackedSequenceDataset(train_seqs_shard)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=packed_collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
    n_batches = len(train_loader)

    log(is_main, f"  Training: {len(train_seqs_shard):,} seqs/rank, "
                 f"{n_batches} batches/epoch, batch_size={batch_size}, "
                 f"world_size={world_size}")

    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (padded, lengths) in enumerate(train_loader):
        padded = padded.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        B, T = padded.shape

        # LM-style: input = tokens[:-1], target = tokens[1:]
        input_tokens = padded[:, :-1]
        target_tokens = padded[:, 1:]

        # Valid mask: position i is valid if i+1 < length
        arange = torch.arange(T - 1, device=device).unsqueeze(0)
        target_mask = arange < (lengths.unsqueeze(1) - 1)

        loss = model(input_tokens, packed_targets=target_tokens, packed_mask=target_mask)

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


def train_probe(
    train_data,
    n_clusters_per_layer,
    n_layers,
    n_items,
    local_rank,
    world_size,
    device,
    is_main,
    batch_size=4096,
    lr=3e-3,
    embed_dim=256,
    n_heads=4,
    n_transformer_layers=2,
    ffn_dim=512,
    model_type='probe',
):
    """Train NTPProbe or NTPModel with optional DDP. Returns (model, loss, n_params, model_type)."""

    use_parallel = n_layers >= 5

    if model_type == 's-tier':
        probe = NTPModel(
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
            parallel=use_parallel,
        ).to(device)
    else:
        probe = NTPProbe(
            n_clusters_per_layer=n_clusters_per_layer,
            n_sid_layers=n_layers,
            n_items=n_items,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            ffn_dim=ffn_dim,
            parallel=use_parallel,
        ).to(device)

    n_params = sum(p.numel() for p in probe.parameters())
    mode_str = "parallel (MTP)" if use_parallel else "autoregressive"
    log(is_main, f"  {model_type}: {n_params / 1e6:.1f}M params, {mode_str}")

    # DDP wrap
    if world_size > 1:
        probe = DDP(probe, device_ids=[local_rank])

    # Shard data per rank to save memory
    if world_size > 1:
        n_total = len(train_data)
        shard_size = n_total // world_size
        shard_start = local_rank * shard_size
        shard_end = shard_start + shard_size if local_rank < world_size - 1 else n_total
        train_shard = train_data[shard_start:shard_end]
        del train_data
    else:
        train_shard = train_data

    dataset = SIDSequenceDataset(train_shard)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
    n_batches = len(train_loader)

    log(is_main, f"  Training: {len(train_shard):,} samples/rank, "
                 f"{n_batches} batches/epoch, batch_size={batch_size}, "
                 f"world_size={world_size}")

    probe.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (input_batch, target_batch) in enumerate(train_loader):
        input_batch = input_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)

        if use_parallel:
            logits_list = probe(input_batch)  # [(B, C_l), ...]
        else:
            teacher_input = torch.cat([input_batch, target_batch[:, :-1]], dim=1)
            logits_list = probe(teacher_input, return_last_n=n_layers)  # [(B, C_l), ...]

        # Per-layer CE loss (different codebook sizes)
        loss = sum(
            F.cross_entropy(logits_list[l], target_batch[:, l])
            for l in range(n_layers)
        ) / n_layers

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if is_main and (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            samples_per_sec = (step + 1) * batch_size * world_size / elapsed
            print(f"    step {step+1}/{n_batches}: "
                  f"loss={total_loss/(step+1):.4f}, "
                  f"{samples_per_sec:.0f} samples/s")

    avg_loss = total_loss / n_batches
    elapsed = time.time() - t0
    log(is_main, f"  Train done: loss={avg_loss:.4f} ({elapsed:.1f}s)")

    # Unwrap DDP
    raw_probe = probe.module if isinstance(probe, DDP) else probe
    return raw_probe.cpu(), avg_loss, n_params, model_type


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(output_dir, probe, train_data, eval_data, eval_cids,
                    sid_to_items, n_clusters_per_layer, n_layers, n_items,
                    avg_loss, n_params, sid_cache_dir, model_type='probe',
                    n_train_total=None):
    """Save probe checkpoint + eval data (rank 0 only)."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Model checkpoint — config varies by model_type
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

    # 2. Eval data
    torch.save({
        'eval_data': eval_data,
        'eval_cids': eval_cids,
        'sid_to_items': dict(sid_to_items),  # defaultdict -> dict for serialization
    }, os.path.join(output_dir, 'eval_data.pt'))

    # 3. Train metadata
    meta = {
        'n_train': n_train_total if n_train_total else len(train_data),
        'n_eval': len(eval_data),
        'n_clusters_per_layer': n_clusters_per_layer,
        'n_layers': n_layers,
        'n_items': n_items,
        'n_params': n_params,
        'train_loss': round(avg_loss, 6),
        'sid_cache': sid_cache_dir,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(output_dir, 'train_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {output_dir}/")
    print(f"    probe.pt      ({os.path.getsize(os.path.join(output_dir, 'probe.pt')) / 1e6:.1f}MB)")
    print(f"    eval_data.pt  ({os.path.getsize(os.path.join(output_dir, 'eval_data.pt')) / 1e6:.1f}MB)")
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

        # Each rank loads only its own shard
        shard_path = os.path.join(args.preprocessed_dir, f'train_shard_{local_rank}.npz')
        if not os.path.exists(shard_path):
            raise FileNotFoundError(
                f"Shard {shard_path} not found. "
                f"Expected {prep_meta['n_shards']} shards but world_size={world_size}. "
                f"Re-run preprocess-ntp with --n_shards {world_size}.")
        from gr_demo.ntp.preprocess import load_shard
        train_data = load_shard(shard_path)
        log(is_main, f"  Rank {local_rank}: loaded {len(train_data):,} seqs from shard")
        log(is_main, f"  Layers: {n_layers}, n_items: {n_items}, max_seq_len: {max_seq_len}")

        # Eval data only needed on rank 0 for saving
        if is_main:
            eval_ckpt = torch.load(
                os.path.join(args.preprocessed_dir, 'eval_data.pt'),
                map_location='cpu', weights_only=False)
            eval_data = eval_ckpt['eval_data']
            eval_cids = eval_ckpt['eval_cids']
            sid_to_items = eval_ckpt['sid_to_items']
            n_eval = len(eval_data)
        else:
            eval_data = eval_cids = sid_to_items = None
            n_eval = prep_meta['n_eval']

    # ── Slow path: build data on rank 0, share to other ranks ──
    else:
        log(is_main, f"\nStep 1: Loading SID cache from {args.sid_cache}")
        cache_config_path = os.path.join(args.sid_cache, 'config.json')
        with open(cache_config_path) as f:
            cache_config = json.load(f)

        sid_cache_dir = args.sid_cache
        n_items = args.n_items
        max_seq_len = args.max_seq_len

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

            log(is_main, "\nStep 3: Building packed sequences")
            train_data, eval_data, eval_cids, sid_to_items, n_layers, n_clusters_per_layer = \
                build_packed_sequences(
                    sid_dict, behavior_data,
                    n_items=n_items, max_seq_len=max_seq_len)

            del sid_dict, behavior_data

            if world_size > 1:
                shared_dir = os.path.join(args.sid_cache, '_train_tmp')
                os.makedirs(shared_dir, exist_ok=True)
                np.save(os.path.join(shared_dir, 'train_data.npy'),
                        train_data, allow_pickle=True)
            meta = (n_layers, n_clusters_per_layer, len(train_data), len(eval_data))
        else:
            train_data = eval_data = eval_cids = sid_to_items = None
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
                train_data = np.load(
                    os.path.join(shared_dir, 'train_data.npy'),
                    allow_pickle=True).tolist()

            dist.barrier()
            if is_main:
                os.remove(os.path.join(shared_dir, 'train_data.npy'))
                try:
                    os.rmdir(shared_dir)
                except OSError:
                    pass

        n_layers, n_clusters_per_layer, n_train, n_eval = meta
        log(is_main, f"  Train: {len(train_data):,}, Eval: {n_eval:,}, Layers: {n_layers}")

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
        train_seqs=train_data,
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

    # ── Save (rank 0 only) ──
    if is_main:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = args.output_dir or os.path.join(
            repo_root, 'experiments', 'ntp_checkpoints', args.name)

        log(is_main, f"\nStep 5: Saving checkpoint")
        n_train_total = prep_meta['n_train'] if args.preprocessed_dir else None
        save_checkpoint(
            output_dir=output_dir,
            probe=probe,
            train_data=train_data,
            eval_data=eval_data,
            eval_cids=eval_cids,
            sid_to_items=sid_to_items,
            n_clusters_per_layer=n_clusters_per_layer,
            n_layers=n_layers,
            n_items=n_items,
            avg_loss=avg_loss,
            n_params=n_params,
            sid_cache_dir=sid_cache_dir,
            model_type=model_type,
            n_train_total=n_train_total,
        )

        log(is_main, f"\n{'=' * 60}")
        log(is_main, "Training complete!")
        log(is_main, f"{'=' * 60}")
        log(is_main, f"\nNext: python run.py hyperparam --sid_cache {sid_cache_dir} "
                      f"--ntp_checkpoint {output_dir} --run_ntp")

    cleanup_ddp()


if __name__ == '__main__':
    main()
