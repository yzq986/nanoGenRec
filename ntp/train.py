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


def _build_user_neg_index(exposure_data, content_to_tokens, verbose_fn=print):
    """Build per-user index of unclicked exposure L0 tokens, sorted by timestamp.

    Returns:
        user_neg_index: dict {uid: np.ndarray of shape (N, 2)} where
            col 0 = exposure_ts, col 1 = L0 token.
            Sorted by exposure_ts. Only includes iids in content_to_tokens.
    """
    import pandas as pd

    uids = exposure_data['uid']
    iids = exposure_data['iid']
    ts = exposure_data['exposure_ts']

    verbose_fn(f"  ENTP: {len(uids):,} unclicked exposures")

    # Filter: iid must be in SID dict
    valid_iids = set(content_to_tokens.keys())
    iid_mask = pd.Index(iids).isin(valid_iids)
    uids_f = uids[iid_mask]
    iids_f = iids[iid_mask]
    ts_f = ts[iid_mask]
    verbose_fn(f"  ENTP: {len(uids_f):,} with valid SID")

    # Map iid → L0 token (first layer only)
    l0_tokens = np.array([content_to_tokens[iid][0] for iid in iids_f], dtype=np.int32)

    # Sort by (uid, ts)
    sort_idx = np.lexsort((ts_f, uids_f))
    uids_s = uids_f[sort_idx]
    ts_s = ts_f[sort_idx]
    l0_s = l0_tokens[sort_idx]

    # Group by uid
    boundaries = np.where(uids_s[1:] != uids_s[:-1])[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(uids_s)]])

    user_neg_index = {}
    for u in range(len(starts)):
        s, e = starts[u], ends[u]
        uid = uids_s[s]
        # Stack (ts, l0_token) pairs
        user_neg_index[uid] = np.column_stack([ts_s[s:e], l0_s[s:e]])

    verbose_fn(f"  ENTP: {len(user_neg_index):,} users with unclicked exposures")
    return user_neg_index


def build_unified_sequences(sid_dict, behavior_data, n_items=10, max_seq_len=512,
                            n_eval_target=50000, verbose_fn=print,
                            exposure_data=None, entp_k=5):
    """Build unified per-user sequences with split_pos for train/eval masking.

    Each user → one complete SID token sequence. split_pos marks the boundary
    between train positions (< split_pos) and eval positions (>= split_pos).

    The split timestamp is chosen so that the total number of eval items across
    all users is approximately n_eval_target.

    Args:
        exposure_data: if provided, builds per-position negative L0 tokens for
            ENTP-Loss (DualGR). Must be the output of load_all_exposure_data().
        entp_k: max negatives per item position.

    Returns:
        sequences: list of dicts with keys:
            'tokens': List[int]  — full SID token sequence (n_items * n_layers)
            'split_pos': int     — token-level split point
            'eval_cids': List    — content_ids for items after split
            'neg_l0': List[List[int]] (optional) — per-item K negative L0 tokens,
                padded with -1. Length = n_user_items, inner length = entp_k.
        n_layers: int
        n_clusters_per_layer: list
    """
    content_to_tokens, n_layers, n_clusters_per_layer, _sid_to_items = \
        _parse_sid_dict(sid_dict)
    verbose_fn(f"  SID: {n_layers} layers, codebooks={n_clusters_per_layer}")
    verbose_fn(f"  Unique SIDs: {len(_sid_to_items):,}")

    uids_s, iids_s, ts_s, starts, ends = \
        _build_user_items(behavior_data, content_to_tokens, verbose_fn)

    # Build per-user negative index (ENTP)
    user_neg_index = None
    if exposure_data is not None:
        user_neg_index = _build_user_neg_index(
            exposure_data, content_to_tokens, verbose_fn)

    # Find split_ts so that total eval items ≈ n_eval_target
    # Items with ts > split_ts become eval items
    sorted_ts = np.sort(ts_s)
    total_items = len(sorted_ts)
    # We want n_eval_target items after the split, so split at (total - n_eval_target)
    split_idx = max(0, min(total_items - 1, total_items - n_eval_target))
    split_ts = float(sorted_ts[split_idx])
    actual_eval = int((sorted_ts > split_ts).sum())
    pct = 100.0 * split_idx / total_items if total_items > 0 else 0
    verbose_fn(f"  Time split: {actual_eval:,} eval items targeted "
               f"(~{pct:.1f}th percentile, split_ts={split_ts:.0f})")

    max_items = max_seq_len // n_layers
    sequences = []
    n_train_only = 0
    n_eval_only = 0
    n_both = 0
    n_neg_total = 0

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

        seq_dict = {
            'tokens': flat,
            'split_pos': split_token_pos,
            'eval_cids': eval_cids,
        }

        # ── Per-item negative L0 tokens (ENTP) ──
        if user_neg_index is not None:
            uid = uids_s[s]
            neg_arr = user_neg_index.get(uid)  # (M, 2): [ts, l0_token] sorted
            neg_l0_per_item = []
            for i in range(n):
                item_negs = [-1] * entp_k
                if neg_arr is not None and len(neg_arr) > 0:
                    t_lo = float(user_ts[i - 1]) if i > 0 else 0.0
                    t_hi = float(user_ts[i])
                    # Binary search for window (t_lo, t_hi]
                    lo = np.searchsorted(neg_arr[:, 0], t_lo, side='right')
                    hi = np.searchsorted(neg_arr[:, 0], t_hi, side='right')
                    candidates = neg_arr[lo:hi, 1]
                    if len(candidates) > 0:
                        if len(candidates) > entp_k:
                            chosen = np.random.choice(candidates, entp_k, replace=False)
                        else:
                            chosen = candidates
                        for j, tok in enumerate(chosen):
                            item_negs[j] = int(tok)
                        n_neg_total += len(chosen)
                neg_l0_per_item.append(item_negs)
            seq_dict['neg_l0'] = neg_l0_per_item

        sequences.append(seq_dict)

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
    if user_neg_index is not None:
        avg_neg = n_neg_total / max(len(sequences), 1)
        verbose_fn(f"  ENTP negatives: {n_neg_total:,} total, "
                   f"avg {avg_neg:.1f}/seq (K={entp_k})")

    return sequences, n_layers, n_clusters_per_layer, split_ts


# ============================================================
# Packed sequence dataset + collate
# ============================================================

class UnifiedSequenceDataset(torch.utils.data.Dataset):
    """Dataset of unified per-user sequences with split_pos."""

    def __init__(self, tokens_list, split_pos_list, neg_l0_list=None):
        """
        Args:
            tokens_list: list of 1D token lists (variable length)
            split_pos_list: list of int split positions
            neg_l0_list: optional list of 2D lists (n_items, K) neg L0 tokens
        """
        self.tokens_list = tokens_list
        self.split_pos_list = split_pos_list
        self.neg_l0_list = neg_l0_list

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        item = (torch.tensor(self.tokens_list[idx], dtype=torch.long),
                self.split_pos_list[idx])
        if self.neg_l0_list is not None:
            return item + (torch.tensor(self.neg_l0_list[idx], dtype=torch.long),)
        return item


def unified_collate_fn(batch):
    """Right-pad variable-length sequences, return split_pos tensor.

    When neg_l0 data is present (3-tuples), also pads and returns neg tensors.
    """
    has_neg = len(batch[0]) == 3
    if has_neg:
        seqs, split_positions, neg_l0s = zip(*batch)
    else:
        seqs, split_positions = zip(*batch)

    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    split_pos = torch.tensor(split_positions, dtype=torch.long)
    max_len = lengths.max().item()
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq

    if not has_neg:
        return padded, lengths, split_pos

    # Pad neg_l0: (B, max_items, K) → expand to (B, max_len_tokens, K)
    # neg_l0s[i] shape: (n_items_i, K) — one row per item
    # We need to broadcast item-level negs to token-level positions.
    # Each item spans n_layers tokens; neg only applies at L0 prediction positions.
    K = neg_l0s[0].size(1)
    n_layers = max_len // neg_l0s[0].size(0) if neg_l0s[0].size(0) > 0 else 3
    # Infer n_layers from first sequence: tokens / items
    for i, nl in enumerate(neg_l0s):
        if nl.size(0) > 0:
            n_layers = lengths[i].item() // nl.size(0)
            break

    # Build (B, max_len-1, K) tensor aligned with input_tokens positions.
    # Position j in input predicts token j+1. If (j+1) % n_layers == 0,
    # it's an L0 prediction → use neg from item index (j+1) // n_layers.
    S = max_len - 1  # same as input_tokens length
    neg_padded = torch.full((len(seqs), S, K), -1, dtype=torch.long)
    neg_mask = torch.zeros(len(seqs), S, K, dtype=torch.bool)

    for i, nl in enumerate(neg_l0s):
        n_items_i = nl.size(0)
        for item_idx in range(n_items_i):
            # This item's L0 prediction is at input position: item_idx * n_layers - 1
            # because position j predicts token j+1, and token item_idx*n_layers
            # is the L0 token of item_idx. So j+1 = item_idx*n_layers → j = item_idx*n_layers - 1.
            # But item_idx=0 means j=-1 (no preceding context) → skip.
            if item_idx == 0:
                continue
            pos = item_idx * n_layers - 1
            if pos >= S:
                break
            neg_padded[i, pos] = nl[item_idx]
            neg_mask[i, pos] = (nl[item_idx] >= 0)

    return padded, lengths, split_pos, neg_padded, neg_mask


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
    neg_l0_list=None,
    entp_weight=0.0,
):
    """Train NTPModel or NTPProbe with unified sequences (causal LM style).

    Loss is computed only on train positions (pos < split_pos) per sequence.

    Args:
        tokens_list: list of 1D token lists (variable length per user)
        split_pos_list: list of int split positions per sequence
        pre_sharded: if True, data is already this rank's shard (from preprocess-ntp).
        neg_l0_list: optional list of 2D lists (n_items, K) neg L0 tokens for ENTP.
        entp_weight: α weight for ENTP loss (0 = disabled).
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

    # Auto-cap batch_size to fit GPU memory.
    # Attention is O(B × S²); cap so peak activation fits in ~30GB.
    mem_safe_bs = max(64, 40_000_000 // (max_seq_len * max_seq_len))
    if batch_size > mem_safe_bs:
        log(is_main, f"  batch_size {batch_size} too large for seq_len={max_seq_len}, "
                      f"capping to {mem_safe_bs}")
        batch_size = mem_safe_bs

    log(is_main, f"  {model_type} (packed): {n_params / 1e6:.1f}M params, "
                 f"max_seq={max_seq_len}, batch_size={batch_size}")

    if world_size > 1:
        # MoE: not all experts active every batch → unused params expected
        ddp_kwargs = {}
        if model_type == 's-tier':
            ddp_kwargs['find_unused_parameters'] = True
            import warnings
            warnings.filterwarnings('ignore', message='.*find_unused_parameters.*')
        model = DDP(model, device_ids=[local_rank], **ddp_kwargs)

    # Shard data per rank to save memory (each rank only holds 1/N)
    neg_shard = neg_l0_list
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
        if neg_l0_list is not None:
            neg_shard = neg_l0_list[shard_start:shard_end]
        del tokens_list, split_pos_list, neg_l0_list
        log(is_main, f"  Rank {local_rank}: shard {shard_start}..{shard_end} "
                      f"({len(tokens_shard):,} seqs)")
    else:
        tokens_shard = tokens_list
        split_pos_shard = split_pos_list

    dataset = UnifiedSequenceDataset(tokens_shard, split_pos_shard, neg_shard)
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
    n_batches = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_batches)

    log(is_main, f"  Training: {len(tokens_shard):,} seqs/rank, "
                 f"{n_batches} batches/epoch, batch_size={batch_size}, "
                 f"world_size={world_size}")

    use_entp = entp_weight > 0 and neg_shard is not None
    if use_entp:
        log(is_main, f"  ENTP-Loss enabled: α={entp_weight}")

    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        if use_entp:
            padded, lengths, split_positions, neg_padded, neg_mask_batch = batch
            neg_padded = neg_padded.to(device, non_blocking=True)
            neg_mask_batch = neg_mask_batch.to(device, non_blocking=True)
        else:
            padded, lengths, split_positions = batch
            neg_padded = None
            neg_mask_batch = None

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

        loss = model(
            input_tokens,
            packed_targets=target_tokens,
            packed_mask=train_mask,
            neg_l0_tokens=neg_padded,
            neg_l0_mask=neg_mask_batch,
            entp_weight=entp_weight,
        )

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
            'n_heads': probe.encoder.layers[0].self_attn.num_heads,
            'n_transformer_layers': len(probe.encoder.layers),
            'ffn_dim': probe.encoder.layers[0].linear1.out_features,
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
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training, load checkpoint and run eval only')
    # ENTP-Loss (DualGR, WWW 2026)
    parser.add_argument('--entp_weight', type=float, default=0.0,
                        help='ENTP-Loss weight α (0=disabled). Paper default: 0.1')
    parser.add_argument('--entp_k', type=int, default=5,
                        help='Max negative L0 tokens per item position for ENTP')
    return parser.parse_args()


def _run_inline_eval(probe, sid_cache_dir, preprocessed_dir, n_layers,
                     n_clusters_per_layer, local_rank, world_size, device,
                     is_main, batch_size=2048, n_recall_total=1000):
    """Run eval on ALL ranks in parallel, all-reduce results.

    Each rank loads its own shard's eval data. Teacher-forced and beam search
    run in parallel across GPUs. Results are reduced to rank 0.

    Returns dict with PPL, depth_hit@10, recall@K, etc. (meaningful only on rank 0).
    """
    from gr_demo.ntp.eval import (
        _batched_teacher_forced_eval, _beam_search_recall,
        _build_sid_to_items,
    )
    from gr_demo.ntp.preprocess import load_shard_full
    from gr_demo.ntp.model import SIDTrie

    if not (preprocessed_dir and os.path.isdir(preprocessed_dir)):
        log(is_main, "  Warning: no preprocessed_dir, skipping inline eval")
        return {}

    # Each rank loads its own shard's eval sequences
    shard_path = os.path.join(preprocessed_dir, f'train_shard_{local_rank}.npz')
    all_seqs = load_shard_full(shard_path)
    eval_sequences = [s for s in all_seqs
                      if s['split_pos'] < len(s['tokens']) and len(s['eval_cids']) > 0]
    del all_seqs

    log(is_main, f"  Eval sequences per rank: {len(eval_sequences):,} (rank {local_rank})")

    probe = probe.to(device)
    probe.eval()

    # ── Teacher-forced (each rank on its own shard) ──
    log(is_main, "  Running teacher-forced eval (all ranks)...")
    with torch.no_grad():
        tf_results = _batched_teacher_forced_eval(
            probe, eval_sequences, n_layers, device,
            batch_size=batch_size, verbose=is_main)

    # All-reduce teacher-forced stats
    # Layout: [total_loss, n_positions, n_eval_items,
    #          prefix_hit_0..L-1, indep_hit_0..L-1, per_layer_loss_0..L-1]
    local_n_pos = tf_results['n_eval_positions']
    local_n_items = tf_results.get('n_eval_items', 0)
    local_n_per_layer = local_n_pos / n_layers if n_layers > 0 else 1
    local_layer_ppl = tf_results.get('layer_ppl', [1.0] * n_layers)
    # Reconstruct per-layer total loss from layer_ppl: loss_li = ln(ppl_li) * n_per_layer
    local_layer_loss = [np.log(max(p, 1e-8)) * local_n_per_layer for p in local_layer_ppl]

    local_stats = torch.tensor(
        [tf_results['avg_loss'] * local_n_pos,  # total loss
         local_n_pos,                             # total positions
         local_n_items,                           # total eval items
         ] + [h * local_n_items for h in tf_results['depth_hit_10']         # prefix hit counts
         ] + [h * local_n_per_layer for h in tf_results.get('depth_hit_10_indep',
                                                             tf_results['depth_hit_10'])
         ] + local_layer_loss,                    # per-layer total loss
        device=device)

    if world_size > 1:
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

    total_loss = local_stats[0].item()
    total_positions = local_stats[1].item()
    total_eval_items = local_stats[2].item()
    n_per_layer = total_positions / n_layers if n_layers > 0 else 1

    global_avg_loss = total_loss / max(total_positions, 1)
    global_ppl = np.exp(global_avg_loss)
    global_depth_h10 = [local_stats[3 + li].item() / max(total_eval_items, 1)
                        for li in range(n_layers)]
    global_depth_h10_indep = [local_stats[3 + n_layers + li].item() / max(n_per_layer, 1)
                              for li in range(n_layers)]
    off = 3 + 2 * n_layers
    global_layer_ppl = [np.exp(local_stats[off + li].item() / max(n_per_layer, 1))
                        for li in range(n_layers)]

    if is_main:
        print(f"  PPL: {global_ppl:.2f}  (per-layer: {[f'L{i}={p:.2f}' for i, p in enumerate(global_layer_ppl)]})")
        print(f"    L0=cross-item (hard), L1..L{n_layers-1}=intra-item (easy)")
        print(f"  Depth hit@10 (prefix): {[f'{h:.3f}' for h in global_depth_h10]}")
        print(f"  Depth hit@10 (indep):  {[f'{h:.3f}' for h in global_depth_h10_indep]}")
        print(f"  Eval items: {int(total_eval_items):,}, "
              f"positions: {int(total_positions):,} (across {world_size} ranks)")

    # ── Beam search recall (split 5K across ranks) ──
    log(is_main, f"  Building sid_to_items from {sid_cache_dir}")
    sid_to_items = _build_sid_to_items(sid_cache_dir)
    sid_trie = SIDTrie(sid_to_items, n_layers)

    n_recall_per_rank = max(1, n_recall_total // world_size)
    log(is_main, f"  Beam search: {n_recall_per_rank} items/rank × {world_size} ranks "
                 f"= {n_recall_per_rank * world_size} total")

    with torch.no_grad():
        beam_results = _beam_search_recall(
            probe, eval_sequences, sid_trie, sid_to_items, n_layers,
            device, recall_beam_size=500, n_recall_samples=n_recall_per_rank,
            verbose=is_main)

    # All-reduce beam search recall counts
    recall_ks = [10, 50, 100, 500]
    n_local = beam_results.get('n_recall_samples', 0)
    local_sid_found = beam_results.get('target_sid_found_rate', 0) * n_local
    local_beam_stats = torch.tensor(
        [n_local, local_sid_found] +
        [beam_results.get(f'item_recall@{k}', 0) * n_local for k in recall_ks] +
        [beam_results.get('depth_acc_beam', [0] * n_layers)[li] * n_local
         for li in range(n_layers)],
        device=device)

    if world_size > 1:
        dist.all_reduce(local_beam_stats, op=dist.ReduceOp.SUM)

    total_recall_n = local_beam_stats[0].item()
    global_sid_found_rate = local_beam_stats[1].item() / max(total_recall_n, 1)
    global_recall = {}
    for ki, k in enumerate(recall_ks):
        global_recall[f'item_recall@{k}'] = (
            local_beam_stats[2 + ki].item() / max(total_recall_n, 1))
    global_depth_acc = [local_beam_stats[2 + len(recall_ks) + li].item() / max(total_recall_n, 1)
                        for li in range(n_layers)]

    probe.cpu()

    # Combine results (meaningful on rank 0)
    results = {
        'ppl': round(global_ppl, 4),
        'layer_ppl': [round(p, 4) for p in global_layer_ppl],
        'avg_loss': round(global_avg_loss, 6),
        'depth_hit@10': [round(h, 4) for h in global_depth_h10],
        'depth_hit@10_indep': [round(h, 4) for h in global_depth_h10_indep],
        'depth_acc_beam': [round(a, 4) for a in global_depth_acc],
        'target_sid_found_rate': round(global_sid_found_rate, 4),
        'n_eval_positions': int(total_positions),
        'n_eval_items': int(total_eval_items),
        'n_recall_samples': int(total_recall_n),
    }
    results.update(global_recall)

    if is_main:
        print(f"  target_sid_found_rate: {global_sid_found_rate:.4f}")
        for k in recall_ks:
            print(f"  item_recall@{k}: {results[f'item_recall@{k}']:.4f}")

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
        shard_data = load_shard(shard_path)
        tokens_list, split_pos_list = shard_data[0], shard_data[1]
        neg_l0_list = shard_data[2] if len(shard_data) > 2 else None
        log(is_main, f"  Rank {local_rank}: loaded {len(tokens_list):,} seqs from shard"
                     + (f" (with ENTP neg data)" if neg_l0_list is not None else ""))
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

            # ENTP: load exposure data if enabled
            exposure_data = None
            if args.entp_weight > 0:
                log(is_main, "\nStep 2b: Loading exposure data (ENTP)")
                from gr_demo.eval.batch import load_all_exposure_data
                exposure_data = load_all_exposure_data(
                    date_start=args.date_start, date_end=args.date_end)

            log(is_main, "\nStep 3: Building unified sequences")
            sequences, n_layers, n_clusters_per_layer, _split_ts = \
                build_unified_sequences(
                    sid_dict, behavior_data,
                    n_items=n_items, max_seq_len=max_seq_len,
                    exposure_data=exposure_data, entp_k=args.entp_k)

            del sid_dict, behavior_data, exposure_data

            tokens_list = [s['tokens'] for s in sequences]
            split_pos_list = [s['split_pos'] for s in sequences]
            neg_l0_list = [s['neg_l0'] for s in sequences] if 'neg_l0' in sequences[0] else None
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
                if neg_l0_list is not None:
                    np.save(os.path.join(shared_dir, 'neg_l0_list.npy'),
                            neg_l0_list, allow_pickle=True)
            meta = (n_layers, n_clusters_per_layer, n_train, n_eval,
                    neg_l0_list is not None)
        else:
            tokens_list = split_pos_list = neg_l0_list = None
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
                neg_l0_path = os.path.join(shared_dir, 'neg_l0_list.npy')
                if os.path.exists(neg_l0_path):
                    neg_l0_list = np.load(neg_l0_path, allow_pickle=True).tolist()

            dist.barrier()
            if is_main:
                os.remove(os.path.join(shared_dir, 'tokens_list.npy'))
                os.remove(os.path.join(shared_dir, 'split_pos_list.npy'))
                neg_l0_path = os.path.join(shared_dir, 'neg_l0_list.npy')
                if os.path.exists(neg_l0_path):
                    os.remove(neg_l0_path)
                try:
                    os.rmdir(shared_dir)
                except OSError:
                    pass

        n_layers, n_clusters_per_layer, n_train, n_eval, _has_neg = meta
        log(is_main, f"  Seqs: {len(tokens_list):,}, Eval items: {n_eval:,}, Layers: {n_layers}")

    # ── Data statistics ──
    if is_main:
        seq_lens = np.array([len(t) for t in tokens_list])
        split_pos_arr = np.array(split_pos_list)
        n_items_per_user = seq_lens // n_layers
        train_items = split_pos_arr // n_layers
        eval_items = n_items_per_user - train_items

        def _pct_str(arr):
            p = np.percentile(arr, [25, 50, 75, 90, 95, 99])
            return (f"p25={p[0]:.0f} p50={p[1]:.0f} p75={p[2]:.0f} "
                    f"p90={p[3]:.0f} p95={p[4]:.0f} p99={p[5]:.0f}")

        log(is_main, f"\n  Data stats (rank 0, {len(seq_lens):,} seqs):")
        log(is_main, f"    seq_len (tokens):  min={seq_lens.min()} mean={seq_lens.mean():.0f} "
                      f"max={seq_lens.max()}")
        log(is_main, f"      {_pct_str(seq_lens)}")
        log(is_main, f"    items/user:        min={n_items_per_user.min()} "
                      f"mean={n_items_per_user.mean():.1f} max={n_items_per_user.max()}")
        log(is_main, f"      {_pct_str(n_items_per_user)}")
        log(is_main, f"    train items/user:  mean={train_items.mean():.1f} "
                      f"| eval items/user: mean={eval_items.mean():.1f}")
        log(is_main, f"    train-only users:  {(eval_items == 0).sum():,} "
                      f"| eval-only users: {(train_items == 0).sum():,} "
                      f"| both: {((train_items > 0) & (eval_items > 0)).sum():,}")
        log(is_main, f"    total tokens: {seq_lens.sum():,} "
                      f"(train: {split_pos_arr.sum():,}, "
                      f"eval: {(seq_lens - split_pos_arr).sum():,})")

        # Sample sequences for inspection (only items <= 10 for mask verification)
        if is_main:
            import random as _rng
            _rng.seed(0)
            short_idxs = [i for i in range(len(tokens_list))
                          if len(tokens_list[i]) // n_layers <= 10]
            if short_idxs:
                sample_idxs = _rng.sample(short_idxs, min(5, len(short_idxs)))
                print(f"\n  Sample sequences (items<=10, {len(sample_idxs)} of "
                      f"{len(short_idxs):,} short / {len(tokens_list):,} total):")
                for si, idx in enumerate(sample_idxs):
                    toks = tokens_list[idx]
                    sp = split_pos_list[idx]
                    n_tok = len(toks)
                    n_user_items = n_tok // n_layers
                    split_item = sp // n_layers
                    L = n_layers
                    print(f"    [{si}] {n_user_items} items, split@item{split_item}")
                    # Causal attention matrix
                    # T=visible+has_train_loss, -=visible+no_train_loss, F=masked
                    row = ['  '] + [f'{jj:>2}' for jj in range(n_user_items)]
                    print('      ' + ' '.join(row))
                    for ii in range(n_user_items):
                        # Row has train loss if next item is in train range
                        has_loss = ii < split_item - 1 and ii < n_user_items - 1
                        vis = 'T ' if has_loss else '- '
                        cells = [f'{ii:>2}'] + [vis if jj <= ii else 'F ' for jj in range(n_user_items)]
                        role = 'T' if ii < split_item else 'E'
                        sid = '_'.join(str(toks[ii*L+li]) for li in range(L))
                        print(f"    {role} {' '.join(cells)}  {sid}")
            else:
                print(f"\n  No sequences with <=10 items to sample.")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(
        repo_root, 'experiments', 'ntp_checkpoints', args.name)

    if args.eval_only:
        # ── Eval-only: load checkpoint, skip training ──
        ckpt_path = os.path.join(output_dir, 'probe.pt')
        log(is_main, f"\n--eval_only: loading checkpoint from {output_dir}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt['config']
        model_type = cfg.get('model_type', 'probe')

        if model_type == 's-tier':
            from gr_demo.ntp.model import NTPModel
            probe = NTPModel(
                n_clusters_per_layer=cfg['n_clusters_per_layer'],
                n_sid_layers=cfg['n_sid_layers'],
                n_items=cfg.get('n_items', n_items),
                embed_dim=cfg.get('embed_dim', 256),
                n_heads=cfg.get('n_heads', 8),
                n_transformer_layers=cfg.get('n_transformer_layers', 6),
                n_experts=cfg.get('n_experts', 8),
                top_k=cfg.get('top_k', 2),
                expert_dim=cfg.get('expert_dim', 1024),
                max_seq_len=cfg.get('max_seq_len', max_seq_len),
            )
        else:
            probe = NTPProbe(
                n_clusters_per_layer=cfg['n_clusters_per_layer'],
                n_sid_layers=cfg['n_sid_layers'],
                n_items=cfg.get('n_items', n_items),
                embed_dim=cfg.get('embed_dim', 256),
                n_heads=cfg.get('n_heads', 4),
                n_transformer_layers=cfg.get('n_transformer_layers', 2),
                ffn_dim=cfg.get('ffn_dim', 512),
                max_seq_len=cfg.get('max_seq_len', max_seq_len),
            )
        probe.load_state_dict(ckpt['model_state_dict'])
        probe.to(device)
        n_params = sum(p.numel() for p in probe.parameters())
        avg_loss = 0.0
        log(is_main, f"  {model_type}: {n_params / 1e6:.1f}M params")
    else:
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
            neg_l0_list=neg_l0_list,
            entp_weight=args.entp_weight,
        )

        # ── Save checkpoint (rank 0 only) ──
        if is_main:
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

    # ── Inline eval (ALL ranks participate, all-reduce results) ──
    log(is_main, f"\nStep 6: Eval (teacher-forced + beam search, {world_size} GPUs)")
    eval_results = _run_inline_eval(
        probe, sid_cache_dir, preprocessed_dir, n_layers,
        n_clusters_per_layer, local_rank, world_size, device,
        is_main, args.batch_size)

    # Save eval results (rank 0 only)
    if is_main and eval_results:
        meta_path = os.path.join(output_dir, 'train_meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        meta['eval'] = eval_results
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # Save to experiments/results/ntp/ (git-tracked)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(repo_root, 'experiments', 'results', 'ntp')
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f'{args.name}.json')
        with open(results_path, 'w') as f:
            json.dump({
                'name': args.name,
                'model_type': model_type,
                'n_params': n_params,
                'sid_cache': sid_cache_dir,
                'eval': eval_results,
            }, f, indent=2)
        log(is_main, f"\n  Results saved to {results_path}")

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
