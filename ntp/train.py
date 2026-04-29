"""NTP Training — DDP support with unified sequences.

Each user → one complete SID token sequence with split_pos (global 80th-pctl ts).
Training loss computed on positions before split_pos; eval on positions after.

Requires pre-cached shards from `python run.py preprocess-ntp`.

Usage:
    # 8卡 DDP
    torchrun --nproc_per_node=8 run.py train-ntp --preprocessed_dir experiments/ntp_data/exp013

    # 单卡
    python run.py train-ntp --preprocessed_dir experiments/ntp_data/exp013

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
from ntp.baseline import NTPProbe
from ntp.model import NTPModel


# ============================================================
# DDP helpers (borrowed from contrastive_finetune.py)
# ============================================================

def setup_ddp():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        from datetime import timedelta
        dist.init_process_group('nccl', timeout=timedelta(minutes=120))
        torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    is_main = (local_rank == 0)
    return local_rank, world_size, device, is_main


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def log(is_main, msg):
    if is_main:
        print(msg, flush=True)


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


_VIEW_EXIT_BIT = 4096  # bit 12: view_exit is not a positive signal

# ── Side information: time gap buckets + action levels (EXP-023) ──
_TIME_GAP_BOUNDARIES = [0, 30, 120, 300, 900, 1800, 3600, 10800, 21600, 43200,
                        86400, 172800, 345600, 604800, 1209600]

_STRONG_POSITIVE_MASK = (2 | 4 | 8 | 256 | 512 | 1024 | 2048 |
                         131072 | 262144 | 524288 | 1048576)
_TRADE_MASK = 131072 | 262144


def _compute_time_gap_bucket(delta_seconds):
    """Map time gap in seconds → bucket index 0..15. Bucket 0 = BOS (first item)."""
    if delta_seconds <= 0:
        return 0
    for i, boundary in enumerate(_TIME_GAP_BOUNDARIES):
        if delta_seconds <= boundary:
            return i
    return 15


def _action_bitmap_to_level(bm):
    """Map action_bitmap → discrete level: 0=pad, 1=weak, 2=strong, 3=trade."""
    bm = bm & ~_VIEW_EXIT_BIT
    if bm & _TRADE_MASK:
        return 3
    if bm & _STRONG_POSITIVE_MASK:
        return 2
    return 1


def _build_user_items(behavior_data, content_to_tokens, verbose_fn=print, min_action_level=1):
    """Vectorized user interaction grouping using numpy. Returns sorted per-user item lists.

    Returns:
        uids_s, iids_s, ts_s, actions_s: sorted arrays
        starts, ends: per-user group boundaries
        sorted_orig_indices: original row indices (into behavior_data) for each sorted position
    """
    import pandas as pd

    uids = behavior_data['uid']
    iids = behavior_data['iid']
    actions = behavior_data['action_bitmap']
    timestamps = behavior_data.get('first_ts')
    if timestamps is None:
        timestamps = np.arange(len(uids))

    verbose_fn(f"  Total interactions: {len(uids):,}")

    # Vectorized filter: strip view_exit bit, then check for real positive actions
    action_mask = (actions & ~_VIEW_EXIT_BIT) > 0
    orig_indices = np.where(action_mask)[0]
    uids_f = uids[orig_indices]
    iids_f = iids[orig_indices]
    ts_f = timestamps[orig_indices]
    actions_f = actions[orig_indices]

    # Filter: iid in SID dict (vectorized via pandas isin)
    valid_iids = set(content_to_tokens.keys())
    iid_mask = pd.Index(iids_f).isin(valid_iids)
    orig_indices = orig_indices[iid_mask]
    uids_f = uids_f[iid_mask]
    iids_f = iids_f[iid_mask]
    ts_f = ts_f[iid_mask]
    actions_f = actions_f[iid_mask]

    # RSFT: filter by minimum action quality level (min_action_level >= 2 keeps only strong/trade)
    if min_action_level > 1:
        level_mask = np.array([_action_bitmap_to_level(int(bm)) >= min_action_level
                               for bm in actions_f])
        orig_indices = orig_indices[level_mask]
        uids_f = uids_f[level_mask]
        iids_f = iids_f[level_mask]
        ts_f = ts_f[level_mask]
        actions_f = actions_f[level_mask]
        verbose_fn(f"  RSFT (min_action_level={min_action_level}): {len(uids_f):,} interactions kept")

    verbose_fn(f"  Valid interactions: {len(uids_f):,}")

    # Sort by (uid, ts) using numpy lexsort (secondary key first)
    sort_idx = np.lexsort((ts_f, uids_f))
    uids_s = uids_f[sort_idx]
    iids_s = iids_f[sort_idx]
    ts_s = ts_f[sort_idx]
    actions_s = actions_f[sort_idx]
    sorted_orig_indices = orig_indices[sort_idx]

    # Group boundaries
    boundaries = np.where(uids_s[1:] != uids_s[:-1])[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(uids_s)]])

    verbose_fn(f"  Users with valid interactions: {len(starts):,}")
    return uids_s, iids_s, ts_s, actions_s, starts, ends, sorted_orig_indices



def build_unified_sequences(sid_dict, behavior_data=None, n_items=10, max_seq_len=512,
                            n_eval_target=50000, verbose_fn=print,
                            exposure_neg_data=None, entp_k=5,
                            behavior_v2_data=None,
                            shift_features=False, action_l2_only=False,
                            min_action_level=1):
    """Build unified per-user sequences with split_pos for train/eval masking.

    Each user → one complete SID token sequence. split_pos marks the boundary
    between train positions (< split_pos) and eval positions (>= split_pos).

    Two modes:
    - **exposure_neg_data provided** (ENTP mode): compact positive interactions
      with pre-attached neg_iids from PySpark export. Each row has uid, iid,
      first_ts, neg_iids (list of K iid strings).
    - **behavior_data only** (legacy mode): build positive sequences from behavior
      data. No ENTP negatives.

    Returns:
        sequences: list of dicts with keys:
            'tokens': List[int]  — full SID token sequence (n_items * n_layers)
            'split_pos': int     — token-level split point
            'eval_cids': List    — content_ids for items after split
            'neg_l0': List[List[int]] (optional) — per-item K negative L0 tokens,
                padded with -1. Length = n_user_items, inner length = entp_k.
        n_layers: int
        n_clusters_per_layer: list
        split_ts: float
    """
    content_to_tokens, n_layers, n_clusters_per_layer, _sid_to_items = \
        _parse_sid_dict(sid_dict)
    verbose_fn(f"  SID: {n_layers} layers, codebooks={n_clusters_per_layer}")
    verbose_fn(f"  Unique SIDs: {len(_sid_to_items):,}")

    if behavior_v2_data is not None:
        # ── V2 mode: behavior positives + inline session negatives ──
        return _build_sequences_from_behavior_v2(
            behavior_v2_data, content_to_tokens, n_layers, n_clusters_per_layer,
            entp_k, max_seq_len, n_eval_target, verbose_fn,
            shift_features=shift_features, action_l2_only=action_l2_only,
            min_action_level=min_action_level)
    elif exposure_neg_data is not None:
        # ── ENTP mode: compact positive + neg_iids from PySpark ──
        return _build_sequences_from_exposure(
            exposure_neg_data, content_to_tokens, n_layers, n_clusters_per_layer,
            entp_k, max_seq_len, n_eval_target, verbose_fn,
            shift_features=shift_features)
    else:
        # ── Legacy mode: behavior data only ──
        if behavior_data is None:
            raise ValueError("behavior_data, behavior_v2_data, or exposure_neg_data must be provided")
        return _build_sequences_from_behavior(
            behavior_data, content_to_tokens, n_layers, n_clusters_per_layer,
            max_seq_len, n_eval_target, verbose_fn,
            shift_features=shift_features, action_l2_only=action_l2_only,
            min_action_level=min_action_level)


def _build_sequences_from_exposure(exposure_neg_data, content_to_tokens,
                                   n_layers, n_clusters_per_layer,
                                   entp_k, max_seq_len, n_eval_target,
                                   verbose_fn, shift_features=False):
    """Build sequences from compact ENTP negative data (PySpark output).

    Input: exposure_neg_data dict with uid, iid, first_ts, neg_iids.
    Each row is a positive interaction with pre-attached negative iids.
    Reuses _build_user_items() for vectorized groupby + sort.
    """
    t0 = time.time()
    max_items = max_seq_len // n_layers

    # ── Phase 1: vectorized user grouping (reuse _build_user_items) ──
    # _build_user_items expects action_bitmap > 0; our data is already all positives.
    # Fake action_bitmap = 1 so the filter passes.
    fake_behavior = {
        'uid': exposure_neg_data['uid'],
        'iid': exposure_neg_data['iid'],
        'action_bitmap': np.ones(len(exposure_neg_data['uid']), dtype=np.int32),
        'first_ts': exposure_neg_data['first_ts'],
    }
    uids_s, iids_s, ts_s, _actions_s, starts, ends, sorted_orig_indices = \
        _build_user_items(fake_behavior, content_to_tokens, verbose_fn)

    orig_neg_iids = exposure_neg_data['neg_iids']

    # ── Phase 2: compute split_ts ──
    sorted_ts = np.sort(ts_s)
    total_items = len(sorted_ts)
    split_idx = max(0, min(total_items - 1, total_items - n_eval_target))
    split_ts = float(sorted_ts[split_idx])
    actual_eval = int((sorted_ts > split_ts).sum())
    pct = 100.0 * split_idx / total_items if total_items > 0 else 0
    verbose_fn(f"  Time split: {actual_eval:,} eval items targeted "
               f"(~{pct:.1f}th percentile, split_ts={split_ts:.0f})")

    # ── Phase 3: build sequences ──
    sequences = []
    n_train_only = 0
    n_eval_only = 0
    n_both = 0
    n_neg_total = 0
    n_users_with_negs = 0
    n_truncated = 0
    raw_items_per_user = []

    for u in range(len(starts)):
        s, e = starts[u], ends[u]
        n = e - s
        if n < 2:
            continue

        raw_items_per_user.append(n)
        user_iids = iids_s[s:e]
        user_ts = ts_s[s:e]
        user_orig_idx = sorted_orig_indices[s:e]

        if n > max_items:
            n_truncated += 1
            offset = n - max_items
            user_iids = user_iids[offset:]
            user_ts = user_ts[offset:]
            user_orig_idx = user_orig_idx[offset:]
            n = max_items

        # Map iid → SID tokens, neg_iids → L0 tokens
        user_tokens = [content_to_tokens[iid] for iid in user_iids]
        user_neg_l0 = []
        user_has_negs = False
        for oi in user_orig_idx:
            neg_iid_list = orig_neg_iids[oi]
            neg_l0 = [-1] * entp_k
            j = 0
            for neg_iid in neg_iid_list:
                neg_toks = content_to_tokens.get(neg_iid)
                if neg_toks is not None:
                    neg_l0[j] = int(neg_toks[0])
                    j += 1
                    if j >= entp_k:
                        break
            n_valid = j
            if n_valid > 0:
                n_neg_total += n_valid
                user_has_negs = True
            user_neg_l0.append(neg_l0)

        if user_has_negs:
            n_users_with_negs += 1

        # Find split_item_idx
        split_item_idx = n
        for i in range(n):
            if user_ts[i] > split_ts:
                split_item_idx = i
                break

        split_token_pos = split_item_idx * n_layers
        eval_cids = list(user_iids[split_item_idx:]) if split_item_idx < n else []

        flat = []
        for toks in user_tokens:
            flat.extend(toks)

        # Compute time_gaps, action_levels, and continuous timestamps (relative hours)
        time_gaps = []
        action_levels = []
        timestamps = []  # relative hours from first item, one value per token position
        t0_seq = float(user_ts[0])
        for i in range(n):
            if i == 0:
                bucket = 0
            else:
                delta = float(user_ts[i] - user_ts[i - 1])
                bucket = _compute_time_gap_bucket(delta)
            rel_hours = (float(user_ts[i]) - t0_seq) / 3600.0
            for _ in range(n_layers):
                time_gaps.append(bucket)
                action_levels.append(1)
                timestamps.append(rel_hours)

        if shift_features:
            L = n_layers
            time_gaps = [0] * L + time_gaps[:-L]
            action_levels = [0] * L + action_levels[:-L]
            timestamps = [0.0] * L + timestamps[:-L]

        sequences.append({
            'tokens': flat,
            'split_pos': split_token_pos,
            'eval_cids': eval_cids,
            'neg_l0': user_neg_l0,
            'time_gaps': time_gaps,
            'action_levels': action_levels,
            'timestamps': timestamps,
        })

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
    elapsed = time.time() - t0

    # Items/user distribution stats
    raw_arr = np.array(raw_items_per_user)
    pcts = np.percentile(raw_arr, [25, 50, 75, 90, 95, 99, 99.9])
    seq_stats = {
        'n_users': len(raw_arr),
        'max_items': max_items,
        'items_per_user_mean': round(float(raw_arr.mean()), 1),
        'items_per_user_p25': int(pcts[0]),
        'items_per_user_p50': int(pcts[1]),
        'items_per_user_p75': int(pcts[2]),
        'items_per_user_p90': int(pcts[3]),
        'items_per_user_p95': int(pcts[4]),
        'items_per_user_p99': int(pcts[5]),
        'items_per_user_p999': int(pcts[6]),
        'items_per_user_max': int(raw_arr.max()),
        'n_truncated': n_truncated,
        'truncated_pct': round(100.0 * n_truncated / len(raw_arr), 2),
    }

    verbose_fn(f"  Unified sequences: {len(sequences):,}, "
               f"{total_tokens:,} tokens, avg {avg_len:.0f} tok/seq ({elapsed:.1f}s)")
    verbose_fn(f"  Items/user: mean={seq_stats['items_per_user_mean']}, "
               f"p50={seq_stats['items_per_user_p50']}, "
               f"p90={seq_stats['items_per_user_p90']}, "
               f"p95={seq_stats['items_per_user_p95']}, "
               f"p99={seq_stats['items_per_user_p99']}, "
               f"max={seq_stats['items_per_user_max']}")
    verbose_fn(f"  Truncated at {max_items} items: {n_truncated:,} / {len(raw_arr):,} "
               f"({seq_stats['truncated_pct']:.2f}%)")
    verbose_fn(f"  Split: {n_both:,} train+eval, "
               f"{n_train_only:,} train-only, {n_eval_only:,} eval-only")
    verbose_fn(f"  Eval items: {n_eval_items:,}")
    avg_neg = n_neg_total / max(len(sequences), 1)
    verbose_fn(f"  ENTP negatives: {n_neg_total:,} total, "
               f"avg {avg_neg:.1f}/seq (K={entp_k}), "
               f"{n_users_with_negs:,}/{len(sequences):,} users with negs")

    return sequences, n_layers, n_clusters_per_layer, split_ts, seq_stats


def _build_sequences_from_behavior_v2(behavior_v2_data, content_to_tokens,
                                      n_layers, n_clusters_per_layer,
                                      entp_k, max_seq_len, n_eval_target,
                                      verbose_fn, shift_features=False,
                                      action_l2_only=False, min_action_level=1):
    """Build sequences from behavior_v2 data (positives + inline session negatives).

    behavior_v2_data: output of load_behavior_v2_data()
        positives: dict with uid, iid, action_bitmap, first_ts, session_id
        neg_lookup: dict[(uid, session_id)] -> List[str] neg iids
    """
    positives = behavior_v2_data['positives']
    neg_lookup = behavior_v2_data['neg_lookup']

    uids_s, iids_s, ts_s, actions_s, starts, ends, sorted_orig_indices = \
        _build_user_items(positives, content_to_tokens, verbose_fn,
                          min_action_level=min_action_level)

    session_ids_all = positives['session_id']

    sorted_ts = np.sort(ts_s)
    total_items = len(sorted_ts)
    split_idx = max(0, min(total_items - 1, total_items - n_eval_target))
    split_ts = float(sorted_ts[split_idx])
    actual_eval = int((sorted_ts > split_ts).sum())
    pct = 100.0 * split_idx / total_items if total_items > 0 else 0
    verbose_fn(f"  Time split: {actual_eval:,} eval items targeted "
               f"(~{pct:.1f}th percentile, split_ts={split_ts:.0f})")

    max_items = max_seq_len // n_layers
    sequences = []
    n_train_only = n_eval_only = n_both = n_truncated = 0
    n_neg_total = n_users_with_negs = 0
    raw_items_per_user = []

    for u in range(len(starts)):
        s, e = starts[u], ends[u]
        n = e - s
        if n < 2:
            continue

        raw_items_per_user.append(n)
        user_iids = iids_s[s:e]
        user_ts = ts_s[s:e]
        user_actions = actions_s[s:e]
        user_orig_idx = sorted_orig_indices[s:e]

        if n > max_items:
            n_truncated += 1
            offset = n - max_items
            user_iids = user_iids[offset:]
            user_ts = user_ts[offset:]
            user_actions = user_actions[offset:]
            user_orig_idx = user_orig_idx[offset:]
            n = max_items

        user_tokens = [content_to_tokens[iid] for iid in user_iids]

        # Build neg_l0 per item from session-level neg_lookup
        uid = positives['uid'][user_orig_idx[0]]
        user_neg_l0 = []
        user_has_negs = False
        for oi in user_orig_idx:
            iid = positives['iid'][oi]
            sess = session_ids_all[oi]
            neg_iids = neg_lookup.get((uid, sess), [])
            neg_l0 = [-1] * entp_k
            j = 0
            for neg_iid in neg_iids:
                if neg_iid == iid:
                    continue
                neg_toks = content_to_tokens.get(neg_iid)
                if neg_toks is not None:
                    neg_l0[j] = int(neg_toks[0])
                    j += 1
                    if j >= entp_k:
                        break
            if j > 0:
                n_neg_total += j
                user_has_negs = True
            user_neg_l0.append(neg_l0)
        if user_has_negs:
            n_users_with_negs += 1

        split_item_idx = n
        for i in range(n):
            if user_ts[i] > split_ts:
                split_item_idx = i
                break

        split_token_pos = split_item_idx * n_layers
        eval_cids = [user_iids[i] for i in range(split_item_idx, n)]

        flat = []
        for toks in user_tokens:
            flat.extend(toks)

        time_gaps = []
        action_levels = []
        timestamps = []  # relative hours from first item, one value per token position
        t0_seq = float(user_ts[0])
        for i in range(n):
            if i == 0:
                bucket = 0
            else:
                delta = float(user_ts[i] - user_ts[i - 1])
                bucket = _compute_time_gap_bucket(delta)
            level = _action_bitmap_to_level(int(user_actions[i]))
            rel_hours = (float(user_ts[i]) - t0_seq) / 3600.0
            for _ in range(n_layers):
                time_gaps.append(bucket)
                action_levels.append(level)
                timestamps.append(rel_hours)

        if action_l2_only:
            for i in range(len(action_levels)):
                if (i + 1) % n_layers != 0:
                    action_levels[i] = 0

        if shift_features:
            L = n_layers
            time_gaps = [0] * L + time_gaps[:-L]
            action_levels = [0] * L + action_levels[:-L]
            timestamps = [0.0] * L + timestamps[:-L]

        sequences.append({
            'tokens': flat,
            'split_pos': split_token_pos,
            'eval_cids': eval_cids,
            'neg_l0': user_neg_l0,
            'time_gaps': time_gaps,
            'action_levels': action_levels,
            'timestamps': timestamps,
        })

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
    raw_arr = np.array(raw_items_per_user)
    pcts = np.percentile(raw_arr, [25, 50, 75, 90, 95, 99, 99.9])
    seq_stats = {
        'n_users': len(raw_arr),
        'max_items': max_items,
        'items_per_user_mean': round(float(raw_arr.mean()), 1),
        'items_per_user_p25': int(pcts[0]),
        'items_per_user_p50': int(pcts[1]),
        'items_per_user_p75': int(pcts[2]),
        'items_per_user_p90': int(pcts[3]),
        'items_per_user_p95': int(pcts[4]),
        'items_per_user_p99': int(pcts[5]),
        'items_per_user_p999': int(pcts[6]),
        'items_per_user_max': int(raw_arr.max()),
        'n_truncated': n_truncated,
        'truncated_pct': round(100.0 * n_truncated / len(raw_arr), 2),
    }
    verbose_fn(f"  Unified sequences: {len(sequences):,}, "
               f"{total_tokens:,} tokens, avg {avg_len:.0f} tok/seq")
    verbose_fn(f"  Items/user: mean={seq_stats['items_per_user_mean']}, "
               f"p50={seq_stats['items_per_user_p50']}, "
               f"p90={seq_stats['items_per_user_p90']}, "
               f"p95={seq_stats['items_per_user_p95']}, "
               f"p99={seq_stats['items_per_user_p99']}, "
               f"max={seq_stats['items_per_user_max']}")
    verbose_fn(f"  Truncated at {max_items} items: {n_truncated:,} / {len(raw_arr):,} "
               f"({seq_stats['truncated_pct']:.2f}%)")
    verbose_fn(f"  Split: {n_both:,} train+eval, "
               f"{n_train_only:,} train-only, {n_eval_only:,} eval-only")
    verbose_fn(f"  Eval items: {n_eval_items:,}")
    avg_neg = n_neg_total / max(len(sequences), 1)
    verbose_fn(f"  ENTP negatives (session): {n_neg_total:,} total, "
               f"avg {avg_neg:.1f}/seq (K={entp_k}), "
               f"{n_users_with_negs:,}/{len(sequences):,} users with negs")
    return sequences, n_layers, n_clusters_per_layer, split_ts, seq_stats


def _build_sequences_from_behavior(behavior_data, content_to_tokens,
                                   n_layers, n_clusters_per_layer,
                                   max_seq_len, n_eval_target, verbose_fn,
                                   shift_features=False,
                                   action_l2_only=False,
                                   min_action_level=1):
    """Build sequences from behavior data only (no ENTP negatives)."""
    uids_s, iids_s, ts_s, actions_s, starts, ends, _ = \
        _build_user_items(behavior_data, content_to_tokens, verbose_fn,
                          min_action_level=min_action_level)

    # Find split_ts
    sorted_ts = np.sort(ts_s)
    total_items = len(sorted_ts)
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
    n_truncated = 0
    raw_items_per_user = []

    for u in range(len(starts)):
        s, e = starts[u], ends[u]
        n = e - s
        if n < 2:
            continue

        raw_items_per_user.append(n)
        user_iids = iids_s[s:e]
        user_ts = ts_s[s:e]
        user_actions = actions_s[s:e]
        user_tokens = [content_to_tokens[iid] for iid in user_iids]

        if n > max_items:
            n_truncated += 1
            offset = n - max_items
            user_iids = user_iids[offset:]
            user_ts = user_ts[offset:]
            user_actions = user_actions[offset:]
            user_tokens = user_tokens[offset:]
            n = max_items

        split_item_idx = n
        for i in range(n):
            if user_ts[i] > split_ts:
                split_item_idx = i
                break

        split_token_pos = split_item_idx * n_layers
        eval_cids = [user_iids[i] for i in range(split_item_idx, n)]

        flat = []
        for toks in user_tokens:
            flat.extend(toks)

        # Compute time_gaps and action_levels (replicated across n_layers tokens per item)
        time_gaps = []
        action_levels = []
        for i in range(n):
            if i == 0:
                bucket = 0  # BOS
            else:
                delta = float(user_ts[i] - user_ts[i - 1])
                bucket = _compute_time_gap_bucket(delta)
            level = _action_bitmap_to_level(int(user_actions[i]))
            for _ in range(n_layers):
                time_gaps.append(bucket)
                action_levels.append(level)

        if action_l2_only:
            for i in range(len(action_levels)):
                if (i + 1) % n_layers != 0:
                    action_levels[i] = 0

        if shift_features:
            L = n_layers
            time_gaps = [0] * L + time_gaps[:-L]
            action_levels = [0] * L + action_levels[:-L]

        sequences.append({
            'tokens': flat,
            'split_pos': split_token_pos,
            'eval_cids': eval_cids,
            'time_gaps': time_gaps,
            'action_levels': action_levels,
        })

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

    # Items/user distribution stats
    raw_arr = np.array(raw_items_per_user)
    pcts = np.percentile(raw_arr, [25, 50, 75, 90, 95, 99, 99.9])
    seq_stats = {
        'n_users': len(raw_arr),
        'max_items': max_items,
        'items_per_user_mean': round(float(raw_arr.mean()), 1),
        'items_per_user_p25': int(pcts[0]),
        'items_per_user_p50': int(pcts[1]),
        'items_per_user_p75': int(pcts[2]),
        'items_per_user_p90': int(pcts[3]),
        'items_per_user_p95': int(pcts[4]),
        'items_per_user_p99': int(pcts[5]),
        'items_per_user_p999': int(pcts[6]),
        'items_per_user_max': int(raw_arr.max()),
        'n_truncated': n_truncated,
        'truncated_pct': round(100.0 * n_truncated / len(raw_arr), 2),
    }

    verbose_fn(f"  Unified sequences: {len(sequences):,}, "
               f"{total_tokens:,} tokens, avg {avg_len:.0f} tok/seq")
    verbose_fn(f"  Items/user: mean={seq_stats['items_per_user_mean']}, "
               f"p50={seq_stats['items_per_user_p50']}, "
               f"p90={seq_stats['items_per_user_p90']}, "
               f"p95={seq_stats['items_per_user_p95']}, "
               f"p99={seq_stats['items_per_user_p99']}, "
               f"max={seq_stats['items_per_user_max']}")
    verbose_fn(f"  Truncated at {max_items} items: {n_truncated:,} / {len(raw_arr):,} "
               f"({seq_stats['truncated_pct']:.2f}%)")
    verbose_fn(f"  Split: {n_both:,} train+eval, "
               f"{n_train_only:,} train-only, {n_eval_only:,} eval-only")
    verbose_fn(f"  Eval items: {n_eval_items:,}")

    return sequences, n_layers, n_clusters_per_layer, split_ts, seq_stats


# ============================================================
# Packed sequence dataset + collate
# ============================================================

class UnifiedSequenceDataset(torch.utils.data.Dataset):
    """Dataset of unified per-user sequences with split_pos."""

    # Keys that should be loaded as float32; all others default to long.
    _FLOAT_KEYS = {'timestamps'}

    def __init__(self, tokens_list, split_pos_list, neg_l0_list=None,
                 sid_to_embedding=None, n_sid_layers=None,
                 side_features_lists=None):
        """
        Args:
            tokens_list: list of 1D token lists (variable length)
            split_pos_list: list of int split positions
            neg_l0_list: optional list of 2D lists (n_items, K) neg L0 tokens
            sid_to_embedding: optional dict mapping SID tuple → mean embedding (np array)
            n_sid_layers: number of SID layers (needed to chunk tokens into SID tuples)
            side_features_lists: dict[str, list[list]] of per-token side features.
                Known keys: "time_gaps" (long), "action_levels" (long),
                "timestamps" (float32 relative hours).
        """
        self.tokens_list = tokens_list
        self.split_pos_list = split_pos_list
        self.neg_l0_list = neg_l0_list
        self.sid_to_embedding = sid_to_embedding
        self.n_sid_layers = n_sid_layers
        self.side_features_lists = side_features_lists or {}
        # Preserve insertion order so collate can reconstruct keys
        self._sf_keys = list(self.side_features_lists.keys())

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        item = (torch.tensor(tokens, dtype=torch.long),
                self.split_pos_list[idx])
        if self.neg_l0_list is not None:
            item = item + (torch.tensor(self.neg_l0_list[idx], dtype=torch.long),)

        # Look up per-item embeddings from SID tokens
        if self.sid_to_embedding is not None and self.n_sid_layers is not None:
            L = self.n_sid_layers
            n_items = len(tokens) // L
            emb_dim = next(iter(self.sid_to_embedding.values())).shape[0]
            embs = np.zeros((n_items, emb_dim), dtype=np.float32)
            for i in range(n_items):
                sid_tuple = tuple(tokens[i * L:(i + 1) * L])
                emb = self.sid_to_embedding.get(sid_tuple)
                if emb is not None:
                    embs[i] = emb
            item = item + (torch.from_numpy(embs),)

        # Side features in declared order (collate reconstructs by position)
        for key in self._sf_keys:
            dtype = torch.float32 if key in self._FLOAT_KEYS else torch.long
            item = item + (torch.tensor(self.side_features_lists[key][idx], dtype=dtype),)

        return item


def unified_collate_fn(batch):
    """Right-pad variable-length sequences, return split_pos tensor.

    Tuple elements after (tokens, split_pos) are detected by type:
    - ndim==2 + dtype==long → neg_l0 (n_items, K)
    - ndim==2 + dtype==float32 → item_embs (n_items, E)
    - ndim==1 + any dtype → side feature (in declaration order from dataset)

    The last element of the returned tuple is always a dict[str, Tensor] of
    side features if any were declared, otherwise an empty dict.
    The dict is keyed by the sf_keys injected via ``make_collate_fn``.
    When called as a bare function (legacy), side features are unnamed.
    """
    n_elems = len(batch[0])
    seqs = [b[0] for b in batch]
    split_positions = [b[1] for b in batch]

    # Classify remaining elements
    neg_l0s = None
    item_embs_list = None
    sf_tensors = []   # list of (feat_list, dtype) in appearance order

    for elem_idx in range(2, n_elems):
        sample = batch[0][elem_idx]
        if sample.ndim == 2 and sample.dtype == torch.long:
            neg_l0s = [b[elem_idx] for b in batch]
        elif sample.ndim == 2 and sample.dtype == torch.float32:
            item_embs_list = [b[elem_idx] for b in batch]
        elif sample.ndim == 1:
            sf_tensors.append(([b[elem_idx] for b in batch], sample.dtype))

    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    split_pos = torch.tensor(split_positions, dtype=torch.long)
    max_len = lengths.max().item()
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq

    result = [padded, lengths, split_pos]

    if neg_l0s is not None:
        K = neg_l0s[0].size(1)
        n_layers = 3
        for i, nl in enumerate(neg_l0s):
            if nl.size(0) > 0:
                n_layers = lengths[i].item() // nl.size(0)
                break

        S = max_len - 1
        neg_padded = torch.full((len(seqs), S, K), -1, dtype=torch.long)
        neg_mask = torch.zeros(len(seqs), S, K, dtype=torch.bool)

        for i, nl in enumerate(neg_l0s):
            n_items_i = nl.size(0)
            for item_idx in range(n_items_i):
                if item_idx == 0:
                    continue
                pos = item_idx * n_layers - 1
                if pos >= S:
                    break
                neg_padded[i, pos] = nl[item_idx]
                neg_mask[i, pos] = (nl[item_idx] >= 0)

        result.extend([neg_padded, neg_mask])

    if item_embs_list is not None:
        E = item_embs_list[0].size(1)
        max_items = max(e.size(0) for e in item_embs_list)
        item_embs_padded = torch.zeros(len(seqs), max_items, E, dtype=torch.float32)
        for i, emb in enumerate(item_embs_list):
            item_embs_padded[i, :emb.size(0)] = emb
        result.append(item_embs_padded)

    # Pad and collect side features — appended as individual tensors in order
    for feat_list, dtype in sf_tensors:
        feat_padded = torch.zeros(len(seqs), max_len, dtype=dtype)
        for i, feat in enumerate(feat_list):
            feat_padded[i, :len(feat)] = feat
        result.append(feat_padded)

    return tuple(result)


# ============================================================
# Contrastive embedding helpers
# ============================================================

def _build_sid_to_embedding(sid_cache_dir):
    """Build mapping from SID tuple → mean item embedding.

    Loads the SID dict (semantic_ids.npy) and embedding cache for the model_key
    specified in config.json. For each SID tuple, computes the mean embedding
    across all content items that share that SID.

    Caches result to sid_cache_dir/sid_to_embedding.npy for fast subsequent loads.

    Returns:
        sid_to_embedding: dict mapping tuple(int,...) → np.ndarray of shape (emb_dim,)
        emb_dim: int
    """
    cache_path = os.path.join(sid_cache_dir, 'sid_to_embedding.npz')
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        sid_keys = cached['sid_keys']   # (N, n_layers), int32
        sid_embs = cached['sid_embs']   # (N, emb_dim), float32
        sid_to_embedding = {tuple(int(x) for x in k): e for k, e in zip(sid_keys, sid_embs)}
        emb_dim = sid_embs.shape[1]
        print(f"  _build_sid_to_embedding: loaded from cache ({len(sid_to_embedding):,} SID tuples, dim={emb_dim})")
        return sid_to_embedding, emb_dim

    from eval.preprocess_sid import _load_embedding_cache

    # Load SID dict
    sid_path = os.path.join(sid_cache_dir, 'semantic_ids.npy')
    sid_dict = np.load(sid_path, allow_pickle=True).item()

    # Load model_key from config
    config_path = os.path.join(sid_cache_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    model_key = config['model_key']

    # Load embedding cache
    emb_cache, emb_dim = _load_embedding_cache(model_key)

    # Parse SID dict → content_to_tokens
    content_to_tokens, n_layers, _, _ = _parse_sid_dict(sid_dict)

    # Group content items by SID tuple, compute mean embedding
    from collections import defaultdict
    sid_items = defaultdict(list)
    for cid, tokens in content_to_tokens.items():
        emb = emb_cache.get(cid)
        if emb is not None:
            sid_tuple = tuple(tokens)
            sid_items[sid_tuple].append(emb)

    sid_to_embedding = {}
    for sid_tuple, embs in sid_items.items():
        sid_to_embedding[sid_tuple] = np.mean(embs, axis=0).astype(np.float32)

    print(f"  _build_sid_to_embedding: {len(sid_to_embedding):,} SID tuples "
          f"(from {len(content_to_tokens):,} items), emb_dim={emb_dim}")

    # Cache as aligned arrays for fast subsequent loads
    keys_arr = np.array(list(sid_to_embedding.keys()), dtype=np.int32)
    embs_arr = np.array(list(sid_to_embedding.values()), dtype=np.float32)
    np.savez(cache_path, sid_keys=keys_arr, sid_embs=embs_arr)
    print(f"  Cached to {cache_path} ({keys_arr.shape[0]:,} entries)")

    return sid_to_embedding, emb_dim


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
    n_experts=8,
    top_k=2,
    expert_dim=None,
    neg_l0_list=None,
    entp_weight=0.0,
    wandb_run=None,
    contrastive_weight=0.0,
    contrastive_temp=0.07,
    contrastive_dim=0,
    sid_to_embedding=None,
    dry_run=False,
    side_features_lists=None,
    use_segment_emb=False,
    use_torope=False,
    torope_time_split=0.5,
    use_gate_attn=False,
):
    """Train NTPModel or NTPProbe with unified sequences (causal LM style).

    Loss is computed only on train positions (pos < split_pos) per sequence.

    Args:
        tokens_list: list of 1D token lists (variable length per user)
        split_pos_list: list of int split positions per sequence
        neg_l0_list: optional list of 2D lists (n_items, K) neg L0 tokens for ENTP.
        entp_weight: α weight for ENTP loss (0 = disabled).
        wandb_run: optional wandb run object for logging (rank 0 only).
        contrastive_weight: α weight for in-batch contrastive loss (0 = disabled).
        contrastive_temp: InfoNCE temperature τ.
        contrastive_dim: projection dimension for contrastive head.
        sid_to_embedding: dict mapping SID tuple → mean embedding (for contrastive).
        side_features_lists: dict[str, list[list]] of per-token side features.
            Known keys: "time_gaps" (long), "action_levels" (long),
            "timestamps" (float32 relative hours).
    """
    sf_lists = side_features_lists or {}

    # Determine contrastive item embedding dimension from sid_to_embedding
    _contrastive_item_dim = 0
    if contrastive_weight > 0 and sid_to_embedding is not None:
        _contrastive_item_dim = next(iter(sid_to_embedding.values())).shape[0]

    if model_type == 's-tier':
        use_moe = n_experts >= 2
        _expert_dim = expert_dim if expert_dim is not None else embed_dim * 4
        _extra_kwargs = {}
        if contrastive_weight > 0:
            _extra_kwargs['contrastive_dim'] = contrastive_dim
            _extra_kwargs['contrastive_item_dim'] = _contrastive_item_dim
        if use_segment_emb:
            _extra_kwargs['use_segment_emb'] = True
        if use_torope:
            _extra_kwargs['use_torope'] = True
            _extra_kwargs['torope_time_split'] = torope_time_split
        if use_gate_attn:
            _extra_kwargs['use_gate_attn'] = True
        model = NTPModel(
            n_clusters_per_layer=n_clusters_per_layer,
            n_sid_layers=n_layers,
            n_items=n_items,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            use_moe=use_moe,
            n_experts=max(n_experts, 1),
            top_k=top_k,
            expert_dim=_expert_dim,
            parallel=False,
            max_seq_len=max_seq_len,
            active_features=list(sf_lists.keys()),
            **_extra_kwargs,
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

    # Compute active params for FLOPs estimation (MoE: only top_k/n_experts fraction active)
    if model_type == 's-tier' and hasattr(model, 'layers'):
        _m = model
        expert_params = 0
        non_expert_params = 0
        for name, p in _m.named_parameters():
            if '.experts.' in name:
                expert_params += p.numel()
            else:
                non_expert_params += p.numel()
        # top_k out of n_experts active per token
        _top_k = _m.layers[0].ffn.top_k if hasattr(_m.layers[0].ffn, 'top_k') else 2
        _n_experts = _m.layers[0].ffn.n_experts if hasattr(_m.layers[0].ffn, 'n_experts') else 8
        n_active_params = non_expert_params + int(expert_params * _top_k / _n_experts)
    else:
        n_active_params = n_params

    # Auto-cap batch_size to fit GPU memory.
    # Attention is O(B × S²); cap so peak activation fits in ~30GB.
    mem_safe_bs = max(64, 40_000_000 // (max_seq_len * max_seq_len))
    if batch_size > mem_safe_bs:
        log(is_main, f"  batch_size {batch_size} too large for seq_len={max_seq_len}, "
                      f"capping to {mem_safe_bs}")
        batch_size = mem_safe_bs

    log(is_main, f"  {model_type} (packed): {n_params / 1e6:.1f}M params "
                 f"({n_active_params / 1e6:.1f}M active), "
                 f"max_seq={max_seq_len}, batch_size={batch_size}")

    if world_size > 1:
        # MoE: not all experts active every batch → unused params expected
        ddp_kwargs = {}
        if model_type == 's-tier' and n_experts >= 2:
            ddp_kwargs['find_unused_parameters'] = True
            import warnings
            warnings.filterwarnings('ignore', message='.*find_unused_parameters.*')
        model = DDP(model, device_ids=[local_rank], **ddp_kwargs)

    if sf_lists:
        log(is_main, f"  Side features: {list(sf_lists.keys())}, segment_emb={use_segment_emb}")

    # Data is already pre-sharded (one shard per rank from preprocess-ntp)
    dataset = UnifiedSequenceDataset(
        tokens_list, split_pos_list, neg_l0_list,
        sid_to_embedding=sid_to_embedding if contrastive_weight > 0 else None,
        n_sid_layers=n_layers if contrastive_weight > 0 else None,
        side_features_lists=sf_lists,
    )
    sf_keys = dataset._sf_keys  # preserve key order for batch unpacking
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

    log(is_main, f"  Training: {len(tokens_list):,} seqs/rank, "
                 f"{n_batches} batches/epoch, batch_size={batch_size}, "
                 f"world_size={world_size}")

    use_entp = entp_weight > 0 and neg_l0_list is not None
    if use_entp:
        log(is_main, f"  ENTP-Loss enabled: α={entp_weight}")

    use_contrastive = contrastive_weight > 0 and sid_to_embedding is not None
    if use_contrastive:
        log(is_main, f"  Contrastive-Loss enabled: α={contrastive_weight}, "
                      f"τ={contrastive_temp}, dim={contrastive_dim}")

    model.train()
    total_loss = 0.0
    total_tokens = 0
    train_log = []  # per-step metrics for scaling law analysis
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        # Unpack batch tuple — layout depends on which optional data is present.
        # Base: (padded, lengths, split_pos)
        # +ENTP: ... + (neg_padded, neg_mask)
        # +contrastive: ... + (item_embs,)
        # +side features (long): ... + (time_gaps,) + (action_levels,)
        # +timestamps (float32): ... + (timestamps,)
        batch = list(batch)
        padded, lengths, split_positions = batch[0], batch[1], batch[2]
        idx = 3

        neg_padded = None
        neg_mask_batch = None
        batch_item_embs = None

        if use_entp:
            neg_padded = batch[idx].to(device, non_blocking=True)
            neg_mask_batch = batch[idx + 1].to(device, non_blocking=True)
            idx += 2

        if use_contrastive:
            batch_item_embs = batch[idx].to(device, non_blocking=True)
            idx += 1

        # Unpack side features in key order (same order dataset emits them)
        batch_sf = {}
        for key in sf_keys:
            batch_sf[key] = batch[idx].to(device, non_blocking=True)
            idx += 1

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

        # Shift side features by 1 (input = tokens[:-1])
        model_sf = {k: v[:, :-1] for k, v in batch_sf.items()}

        loss = model(
            input_tokens,
            packed_targets=target_tokens,
            packed_mask=train_mask,
            neg_l0_tokens=neg_padded,
            neg_l0_mask=neg_mask_batch,
            entp_weight=entp_weight,
            item_embeddings=batch_item_embs,
            contrastive_weight=contrastive_weight,
            contrastive_temp=contrastive_temp,
            side_features=model_sf or None,
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        scheduler.step()

        step_loss = loss.item()
        total_loss += step_loss
        # tokens processed this step (across all ranks)
        step_tokens = int(lengths.sum().item()) * world_size
        total_tokens += step_tokens

        # Sanity print on first step: show what positions/timestamps actually enter the model
        if step == 0 and is_main:
            S = input_tokens.size(1)
            train_pos = torch.arange(S, device=input_tokens.device)
            log(is_main, f"  [sanity] train  positions[:8]  = {train_pos[:8].tolist()}")
            log(is_main, f"  [sanity] train  positions[-4:]  = {train_pos[-4:].tolist()}")
            if 'time_gaps' in batch_sf:
                tg_sample = batch_sf['time_gaps'][0, :8].tolist()
                log(is_main, f"  [sanity] train  time_gaps[0,:8] = {tg_sample}")
            if 'timestamps' in batch_sf:
                ts_sample = batch_sf['timestamps'][0, :8].tolist()
                log(is_main, f"  [sanity] train  timestamps[0,:8] = {ts_sample}")
            elif use_torope:
                log(is_main, f"  [sanity] train  timestamps = zeros (not in pipeline)")

        if dry_run and step >= 1:
            log(is_main, f"  Dry run complete (2 steps, loss={total_loss/2:.4f})")
            break

        # Record per-step metrics (rank 0 only to avoid duplicate I/O)
        if is_main:
            cur_lr = scheduler.get_last_lr()[0]
            cur_flops = 6 * n_active_params * total_tokens
            train_log.append({
                'step': step,
                'loss': round(step_loss, 6),
                'lr': round(cur_lr, 8),
                'grad_norm': round(grad_norm, 4),
                'tokens': total_tokens,
                'flops': cur_flops,
                'wall_s': round(time.time() - t0, 2),
            })
            if wandb_run is not None:
                wandb_run.log({
                    'train/loss': step_loss,
                    'train/lr': cur_lr,
                    'train/grad_norm': grad_norm,
                    'train/ppl': np.exp(step_loss),
                    'tokens': total_tokens,
                    'flops': cur_flops,
                }, step=step)

        if is_main and (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            seqs_per_sec = (step + 1) * batch_size * world_size / elapsed
            toks_per_sec = total_tokens / elapsed
            remaining = (n_batches - step - 1) / ((step + 1) / elapsed)
            eta = format_eta(remaining)
            print(f"    step {step+1}/{n_batches}: "
                  f"loss={total_loss/(step+1):.4f}, "
                  f"lr={scheduler.get_last_lr()[0]:.2e}, "
                  f"gnorm={grad_norm:.2f}, "
                  f"{toks_per_sec:.0f} tok/s, ETA {eta}")

    avg_loss = total_loss / n_batches
    elapsed = time.time() - t0
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    # FLOPs ≈ 6 * N_active * D (forward + backward per token)
    total_flops = 6 * n_active_params * total_tokens
    log(is_main, f"  Train done: loss={avg_loss:.4f}, "
                 f"{total_tokens:,} tokens, {throughput:.0f} tok/s, "
                 f"{total_flops / 1e12:.2f} TFLOPs ({elapsed:.1f}s)")

    raw_model = model.module if isinstance(model, DDP) else model

    train_summary = {
        'n_params': n_params,
        'n_active_params': n_active_params,
        'total_tokens': total_tokens,
        'total_flops': total_flops,
        'throughput_tok_per_s': round(throughput, 1),
        'wall_time_s': round(elapsed, 1),
        'batch_size': batch_size,
        'world_size': world_size,
        'n_batches': n_batches,
        'max_seq_len': max_seq_len,
    }
    return raw_model.cpu(), avg_loss, n_params, model_type, train_log, train_summary


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
                    model_type='probe', n_train=0, n_eval=0,
                    train_log=None, train_summary=None):
    """Save probe checkpoint + train_meta.json (rank 0 only).

    Eval data lives in preprocessed shards — not duplicated here.
    sid_to_items is rebuilt from SID cache at eval time.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Model checkpoint
    if model_type == 's-tier':
        # Read actual MoE config from model (not hardcoded)
        _ffn0 = probe.layers[0].ffn
        if hasattr(_ffn0, 'n_experts'):
            _use_moe = True
            _n_experts = _ffn0.n_experts
            _top_k = _ffn0.top_k
            _expert_dim = _ffn0.experts[0].w1.out_features
        else:
            _use_moe = False
            _n_experts = 1
            _top_k = 1
            _expert_dim = _ffn0[0].out_features  # nn.Sequential dense FFN
        probe_config = {
            'model_type': 's-tier',
            'n_clusters_per_layer': n_clusters_per_layer,
            'n_sid_layers': n_layers,
            'n_items': n_items,
            'embed_dim': probe.embed_dim,
            'n_heads': probe.layers[0].attn.num_heads,
            'n_transformer_layers': len(probe.layers),
            'use_moe': _use_moe,
            'n_experts': _n_experts,
            'top_k': _top_k,
            'expert_dim': _expert_dim,
            'parallel': probe.parallel,
            'max_seq_len': probe.max_seq_len,
        }
        if hasattr(probe, 'use_segment_emb') and probe.use_segment_emb:
            probe_config['use_segment_emb'] = True
        if hasattr(probe, 'active_features') and probe.active_features:
            probe_config['active_features'] = probe.active_features
        if hasattr(probe, 'use_torope') and probe.use_torope:
            probe_config['use_torope'] = True
            probe_config['torope_time_split'] = probe.torope_time_split
        if getattr(probe.layers[0], 'attn_gate', None) is not None:
            probe_config['use_gate_attn'] = True
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

    # 2. Train metadata (summary-level, always saved)
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
    if train_summary:
        meta['train'] = train_summary
    with open(os.path.join(output_dir, 'train_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # 3. Per-step train log (JSONL, for scaling law analysis)
    #    Fields: step, loss, lr, grad_norm, tokens (cumulative), wall_s
    if train_log:
        log_path = os.path.join(output_dir, 'train_log.jsonl')
        with open(log_path, 'w') as f:
            for entry in train_log:
                f.write(json.dumps(entry) + '\n')

    print(f"  Saved to {output_dir}/")
    print(f"    probe.pt        ({os.path.getsize(os.path.join(output_dir, 'probe.pt')) / 1e6:.1f}MB)")
    print(f"    train_meta.json")
    if train_log:
        print(f"    train_log.jsonl ({len(train_log)} steps)")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train NTP Probe (DDP)')
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                        help='Pre-cached shard directory from preprocess-ntp.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir (default: experiments/ntp_checkpoints/{name})')
    parser.add_argument('--name', type=str, default='default',
                        help='Experiment name for output subdir')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 1e-3 for s-tier, 3e-3 for probe)')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=None,
                        help='Attention heads (default: 8 for s-tier, 4 for probe)')
    parser.add_argument('--n_transformer_layers', type=int, default=None,
                        help='Transformer layers (default: 6 for s-tier, 2 for probe)')
    parser.add_argument('--ffn_dim', type=int, default=512,
                        help='FFN hidden dim (probe only)')
    parser.add_argument('--model', type=str, default='probe',
                        choices=['probe', 's-tier'],
                        help='Model: probe (2L dense) or s-tier (MoE)')
    # MoE config (s-tier only)
    parser.add_argument('--n_experts', type=int, default=8,
                        help='Number of MoE experts (s-tier only, 0 or 1 = dense)')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Top-k expert routing (s-tier only)')
    parser.add_argument('--expert_dim', type=int, default=None,
                        help='Expert FFN hidden dim (s-tier only, default: 4*embed_dim)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training, load checkpoint and run eval only')
    # ENTP-Loss (DualGR, WWW 2026)
    parser.add_argument('--entp_weight', type=float, default=0.0,
                        help='ENTP-Loss weight α (0=disabled). Paper default: 0.1')
    # In-Batch Contrastive Loss (OneMall §3.2, IDEA-onemall-0)
    parser.add_argument('--contrastive_weight', type=float, default=0.0,
                        help='Contrastive loss weight α (0=disabled)')
    parser.add_argument('--contrastive_temp', type=float, default=0.07,
                        help='InfoNCE temperature τ')
    parser.add_argument('--contrastive_dim', type=int, default=128,
                        help='Contrastive projection dimension')
    parser.add_argument('--dry_run', action='store_true',
                        help='Run 2 steps only (smoke test)')
    # Side information features (EXP-023)
    parser.add_argument('--use_segment_emb', action='store_true', default=False,
                        help='Enable segment embedding (item_pos + layer_pos)')
    # TO-RoPE (feat-5, arxiv 2510.20455) — architecture flag, not a data feature
    parser.add_argument('--use_torope', action='store_true', default=False,
                        help='Enable TO-RoPE (Time-and-Order RoPE). Replaces learnable pos_emb '
                             'with split-by-dim rotary encoding.')
    parser.add_argument('--torope_time_split', type=float, default=0.5,
                        help='Fraction of RoPE planes for time encoding (default 0.5)')
    parser.add_argument('--use_gate_attn', action='store_true', default=False,
                        help='Enable GateAttention: sigmoid gate on attention output.')
    return parser.parse_args()


def _run_inline_eval(probe, sid_cache_dir, preprocessed_dir, n_layers,
                     n_clusters_per_layer, local_rank, world_size, device,
                     is_main, batch_size=2048, n_recall_total=1000):
    """Run eval on ALL ranks in parallel, all-reduce results.

    Each rank loads its own shard's eval data. Teacher-forced and beam search
    run in parallel across GPUs. Results are reduced to rank 0.

    Returns dict with PPL, depth_hit@10, recall@K, etc. (meaningful only on rank 0).
    """
    from ntp.eval import (
        _batched_teacher_forced_eval, _beam_search_recall,
        _build_sid_to_items,
    )
    from ntp.preprocess import load_shard_full
    from ntp.model import SIDTrie

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
              f"positions: {int(total_positions):,} (across {world_size} ranks)",
              flush=True)

    # ── Beam search recall (split 5K across ranks) ──
    # Sanity: show the first eval context's positions so we can verify train/infer alignment
    if is_main and eval_sequences:
        _s0 = eval_sequences[0]
        _ctx_len = _s0['split_pos']
        _ctx_pos = list(range(_ctx_len))
        print(f"  [sanity] eval   ctx positions[:8]  = {_ctx_pos[:8]}")
        print(f"  [sanity] eval   ctx positions[-4:] = {_ctx_pos[max(0,_ctx_len-4):]}")
        print(f"  [sanity] eval   ctx len={_ctx_len}, gen positions start at {_ctx_len}")
        if probe.use_torope:
            print(f"  [sanity] eval   timestamps = zeros (ctx_timestamps not passed to beam search)")

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

    # Barrier: beam search time varies per rank (different eval data + context lengths).
    # Without this, fast ranks hit all_reduce while slow ranks are still searching → NCCL timeout.
    if world_size > 1:
        dist.barrier()

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

    if not args.preprocessed_dir:
        raise ValueError(
            "--preprocessed_dir is required. "
            "Run `python run.py preprocess-ntp` first to build shards.")

    log(is_main, "=" * 60)
    log(is_main, f"NTP Training — {model_type}" +
                 (f" (DDP x{world_size})" if world_size > 1 else ""))
    log(is_main, "=" * 60)

    # ── Load pre-cached shards from preprocess-ntp ──
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
    from ntp.preprocess import load_shard
    shard_data = load_shard(shard_path)
    tokens_list = shard_data['tokens_list']
    split_pos_list = shard_data['split_pos_list']
    neg_l0_list = shard_data.get('neg_l0_list')
    side_features_lists = {}
    from ntp.features import REGISTRY as _FEAT_REG
    for key, fdef in _FEAT_REG.items():
        list_key = f'{key}_list'
        if list_key in shard_data:
            side_features_lists[key] = shard_data[list_key]
    sf_desc = ', '.join(side_features_lists.keys()) if side_features_lists else 'none'
    log(is_main, f"  Rank {local_rank}: loaded {len(tokens_list):,} seqs from shard"
                 + (f" (with ENTP neg data)" if neg_l0_list is not None else "")
                 + (f" (side_features: {sf_desc})" if side_features_lists else ""))
    log(is_main, f"  Layers: {n_layers}, n_items: {n_items}, max_seq_len: {max_seq_len}")

    n_train = prep_meta['n_seqs']
    n_eval = prep_meta['n_eval_items']

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

    # ── W&B init (rank 0 only) ──
    wandb_run = None
    if is_main:
        try:
            import wandb
            os.environ.setdefault('WANDB_DIR', os.path.join(repo_root, 'experiments'))
            wandb_run = wandb.init(
                project="gr-demo-ntp",
                name=args.name,
                dir=os.path.join(repo_root, 'experiments'),
                config={
                    'model_type': model_type,
                    'n_layers': n_layers,
                    'n_clusters_per_layer': n_clusters_per_layer,
                    'n_items': n_items,
                    'max_seq_len': max_seq_len,
                    'batch_size': args.batch_size,
                    'world_size': world_size,
                    'n_seqs': prep_meta['n_seqs'],
                    'n_eval_items': prep_meta['n_eval_items'],
                    'n_shards': prep_meta['n_shards'],
                    'sid_cache': sid_cache_dir,
                    'preprocessed_dir': preprocessed_dir,
                    'entp_weight': args.entp_weight,
                    'contrastive_weight': args.contrastive_weight,
                    'contrastive_temp': args.contrastive_temp,
                    'contrastive_dim': args.contrastive_dim,
                    'eval_only': args.eval_only,
                },
            )
            # Custom x-axes: train/* uses tokens, eval/* uses default step
            wandb_run.define_metric('train/*', step_metric='tokens')
            log(is_main, "  W&B initialized")
        except Exception as e:
            wandb_run = None
            log(is_main, f"  W&B not available: {e}")

    ckpt_path = os.path.join(output_dir, 'probe.pt')
    skip_train = args.eval_only or os.path.exists(ckpt_path)

    if skip_train:
        # ── Load existing checkpoint, skip training ──
        if args.eval_only:
            log(is_main, f"\n--eval_only: loading checkpoint from {output_dir}")
        else:
            log(is_main, f"\n  Checkpoint found at {output_dir}, skipping training.")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt['config']
        model_type = cfg.get('model_type', 'probe')

        if model_type == 's-tier':
            from ntp.model import NTPModel
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
                active_features=cfg.get('active_features', []),
                use_segment_emb=cfg.get('use_segment_emb', False),
                use_torope=cfg.get('use_torope', False),
                torope_time_split=cfg.get('torope_time_split', 0.5),
                use_gate_attn=cfg.get('use_gate_attn', False),
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
        probe.load_state_dict(ckpt['model_state_dict'], strict=False)
        probe.to(device)
        n_params = sum(p.numel() for p in probe.parameters())
        avg_loss = 0.0
        train_log = None
        train_summary = None
        log(is_main, f"  {model_type}: {n_params / 1e6:.1f}M params")
    else:
        # ── Train (both probe and s-tier use packed sequences) ──
        embed_dim = args.embed_dim
        ffn_dim = args.ffn_dim
        if model_type == 's-tier':
            n_heads = args.n_heads if args.n_heads is not None else 8
            n_transformer_layers = args.n_transformer_layers if args.n_transformer_layers is not None else 6
            lr = args.lr if args.lr is not None else 1e-3
        else:
            n_heads = args.n_heads if args.n_heads is not None else 4
            n_transformer_layers = args.n_transformer_layers if args.n_transformer_layers is not None else 2
            lr = args.lr if args.lr is not None else 3e-3

        # ── Load contrastive embeddings if needed ──
        sid_to_embedding = None
        if args.contrastive_weight > 0:
            log(is_main, f"\n  Loading SID→embedding for contrastive loss...")
            sid_to_embedding, _emb_dim = _build_sid_to_embedding(sid_cache_dir)
            log(is_main, f"  Contrastive item dim: {_emb_dim}")

        log(is_main, f"\nStep 4: Training ({model_type}, packed)")
        probe, avg_loss, n_params, model_type, train_log, train_summary = train_packed(
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
            n_experts=args.n_experts,
            top_k=args.top_k,
            expert_dim=args.expert_dim,
            neg_l0_list=neg_l0_list,
            entp_weight=args.entp_weight,
            wandb_run=wandb_run,
            contrastive_weight=args.contrastive_weight,
            contrastive_temp=args.contrastive_temp,
            contrastive_dim=args.contrastive_dim,
            sid_to_embedding=sid_to_embedding,
            dry_run=args.dry_run,
            side_features_lists=side_features_lists,
            use_segment_emb=args.use_segment_emb,
            use_torope=args.use_torope,
            torope_time_split=args.torope_time_split,
            use_gate_attn=args.use_gate_attn,
        )

        # Update W&B config with model-specific info discovered during training
        if wandb_run is not None and train_summary:
            wandb_run.config.update({
                'n_params': train_summary['n_params'],
                'n_active_params': train_summary['n_active_params'],
                'embed_dim': embed_dim,
                'n_heads': n_heads,
                'n_transformer_layers': n_transformer_layers,
                'n_experts': args.n_experts,
                'top_k': args.top_k,
                'expert_dim': args.expert_dim or embed_dim * 4,
                'lr': lr,
            }, allow_val_change=True)

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
                train_log=train_log,
                train_summary=train_summary,
            )

    # ── Inline eval (ALL ranks participate, all-reduce results) ──
    meta_path = os.path.join(output_dir, 'train_meta.json')
    has_eval = False
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            has_eval = 'eval' in json.load(f)

    if has_eval and skip_train:
        log(is_main, "\n  Eval already present, nothing to do.")
        eval_results = None
    else:
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
        _n_active = train_summary['n_active_params'] if train_summary else n_params
        _total_tokens = train_summary['total_tokens'] if train_summary else 0
        _total_flops = train_summary['total_flops'] if train_summary else 0
        with open(results_path, 'w') as f:
            json.dump({
                'name': args.name,
                'model_type': model_type,
                'n_params': n_params,
                'n_active_params': _n_active,
                'total_tokens': _total_tokens,
                'total_flops': _total_flops,
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

    # ── W&B: log eval summary + finish ──
    if wandb_run is not None:
        if eval_results:
            # Summary metrics for cross-run comparison (scaling law tables)
            wandb_run.summary['eval/ppl'] = eval_results.get('ppl')
            wandb_run.summary['eval/avg_loss'] = eval_results.get('avg_loss')
            for k in ['item_recall@10', 'item_recall@50', 'item_recall@100', 'item_recall@500']:
                if k in eval_results:
                    wandb_run.summary[f'eval/{k}'] = eval_results[k]
            for li, ppl in enumerate(eval_results.get('layer_ppl', [])):
                wandb_run.summary[f'eval/layer_ppl_L{li}'] = ppl
            wandb_run.summary['eval/target_sid_found_rate'] = eval_results.get(
                'target_sid_found_rate')
        if not args.eval_only and train_summary:
            wandb_run.summary['total_tokens'] = train_summary['total_tokens']
            wandb_run.summary['total_flops'] = train_summary['total_flops']
            wandb_run.summary['n_active_params'] = train_summary['n_active_params']
            wandb_run.summary['train_loss'] = avg_loss
        wandb_run.finish()

    cleanup_ddp()


if __name__ == '__main__':
    main()
