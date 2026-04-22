"""Build RF-DPO preference pairs from real user feedback signals.

Unlike SP-DPO (beam search self-play), RF-DPO uses actual user behavior
to construct preference pairs:
  - Chosen: items the user strongly engaged with (like/share/follow/comment/trade)
  - Rejected Easy: items the user gave negative feedback (report/dislike)
  - Rejected Hard: items the user weakly engaged with (click-only)

All rejected items are from the SAME user — real preference differences.

Usage:
    python run.py rf-dpo-prepare \\
        --sid_cache experiments/sid_cache/exp013-4096x3-12d-binary \\
        --output_dir experiments/rf_dpo_data/exp018 \\
        --difficulty all --n_rejected 20
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd

from rl.preference import save_preference_shard


# ============================================================
# Action bitmap signal tiers
# ============================================================

# Strong positive: deep engagement signals
#   like(2), share(4), follow(8), detail_follow(256), detail_like(512),
#   detail_share(1024), detail_comment(2048), trade_click(131072),
#   place_order(262144), live_duration(524288), comment(1048576)
STRONG_POSITIVE_MASK = (2 | 4 | 8 | 256 | 512 | 1024 | 2048 |
                        131072 | 262144 | 524288 | 1048576)

# Weak positive: surface engagement signals
#   click(1), coin_click(16), photo_click(64), profile_click(128),
#   detail_coin_click(8192), detail_profile_click(16384),
#   detail_photo_click(32768), video_detail_view(65536)
WEAK_POSITIVE_MASK = (1 | 16 | 64 | 128 | 8192 | 16384 | 32768 | 65536)

VIEW_EXIT_BIT = 4096  # bit 12: view_exit (not positive by itself)


def classify_action(action_bitmap):
    """Classify action_bitmap into signal tier.

    Args:
        action_bitmap: int32 action bitmap (can be scalar or numpy array)

    Returns:
        str: 'strong', 'weak', 'negative', or 'neutral'
        (or numpy array of strings if input is array)
    """
    if isinstance(action_bitmap, np.ndarray):
        result = np.full(len(action_bitmap), 'neutral', dtype='<U8')
        # Negative: sign bit set (bit 31, value < 0)
        result[action_bitmap < 0] = 'negative'
        # Weak positive: any weak bit set (but not strong, not negative)
        weak_mask = ((action_bitmap & WEAK_POSITIVE_MASK) > 0) & (action_bitmap >= 0)
        result[weak_mask] = 'weak'
        # Strong positive: any strong bit set (overrides weak)
        strong_mask = ((action_bitmap & STRONG_POSITIVE_MASK) > 0) & (action_bitmap >= 0)
        result[strong_mask] = 'strong'
        return result
    else:
        # Scalar
        if action_bitmap < 0:
            return 'negative'
        if action_bitmap & STRONG_POSITIVE_MASK:
            return 'strong'
        if action_bitmap & WEAK_POSITIVE_MASK:
            return 'weak'
        return 'neutral'


def _build_user_items_all(behavior_data, content_to_tokens, verbose_fn=print):
    """Group user interactions, keeping ALL signal tiers (including negative/neutral).

    Unlike ntp/train.py:_build_user_items which filters to positives only,
    this keeps negative feedback and view_exit items for use as RF-DPO rejected.

    Returns:
        uids_s, iids_s, ts_s, actions_s: sorted arrays
        starts, ends: per-user group boundaries
    """
    uids = behavior_data['uid']
    iids = behavior_data['iid']
    actions = behavior_data['action_bitmap']
    timestamps = behavior_data.get('first_ts')
    if timestamps is None:
        timestamps = np.arange(len(uids))

    verbose_fn(f"  Total interactions: {len(uids):,}")

    # Keep all items that have ANY action (including negative/view_exit)
    # Only filter out items with action_bitmap == 0 (no signal at all)
    action_mask = actions != 0
    orig_indices = np.where(action_mask)[0]
    uids_f = uids[orig_indices]
    iids_f = iids[orig_indices]
    ts_f = timestamps[orig_indices]
    actions_f = actions[orig_indices]

    # Filter: iid in SID dict
    valid_iids = set(content_to_tokens.keys())
    iid_mask = pd.Index(iids_f).isin(valid_iids)
    uids_f = uids_f[iid_mask]
    iids_f = iids_f[iid_mask]
    ts_f = ts_f[iid_mask]
    actions_f = actions_f[iid_mask]

    verbose_fn(f"  Valid interactions (all tiers): {len(uids_f):,}")

    # Sort by (uid, ts)
    sort_idx = np.lexsort((ts_f, uids_f))
    uids_s = uids_f[sort_idx]
    iids_s = iids_f[sort_idx]
    ts_s = ts_f[sort_idx]
    actions_s = actions_f[sort_idx]

    # Group boundaries
    boundaries = np.where(uids_s[1:] != uids_s[:-1])[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(uids_s)]])

    verbose_fn(f"  Users: {len(starts):,}")
    return uids_s, iids_s, ts_s, actions_s, starts, ends


def build_rf_preference_pairs(
    behavior_data,
    content_to_tokens,
    n_layers,
    n_eval_target=50000,
    difficulty='all',
    n_rejected=20,
    max_samples=None,
    max_items_per_user=170,
    verbose=True,
):
    """Build preference pairs from real user feedback.

    For each user, chronologically:
    1. Classify each item by action signal tier
    2. For each strong positive eval item (after split_ts):
       - Context = SID tokens of all preceding positive items
       - Chosen = this item's SID
       - Rejected Easy = same user's negative feedback items' SIDs
       - Rejected Hard = same user's weak positive items' SIDs

    Args:
        behavior_data: dict with uid, iid, action_bitmap, first_ts
        content_to_tokens: dict mapping content_id → list of SID tokens
        n_layers: number of SID layers
        n_eval_target: target number of eval items for time split
        difficulty: 'easy', 'hard', or 'all'
        n_rejected: max rejected per difficulty per sample
        max_samples: cap on pairs for debugging
        max_items_per_user: max items per user sequence (same as NTP)
        verbose: print progress

    Returns:
        list of dicts compatible with save_preference_shard()
    """
    verbose_fn = print if verbose else lambda x: None

    # ── Step 1: Group all user items ──
    uids_s, iids_s, ts_s, actions_s, starts, ends = \
        _build_user_items_all(behavior_data, content_to_tokens, verbose_fn)

    # ── Step 2: Compute time split ──
    sorted_ts = np.sort(ts_s)
    total_items = len(sorted_ts)
    split_idx = max(0, min(total_items - 1, total_items - n_eval_target))
    split_ts = float(sorted_ts[split_idx])
    actual_eval = int((sorted_ts > split_ts).sum())
    verbose_fn(f"  Time split: {actual_eval:,} eval items "
               f"(split_ts={split_ts:.0f})")

    # ── Step 3: Classify all actions ──
    tiers = classify_action(actions_s)

    # ── Step 4: Per-user preference pair construction ──
    tier_counts = {'strong': 0, 'weak': 0, 'negative': 0, 'neutral': 0}
    for t in ['strong', 'weak', 'negative', 'neutral']:
        tier_counts[t] = int((tiers == t).sum())
    verbose_fn(f"  Signal tiers: strong={tier_counts['strong']:,}, "
               f"weak={tier_counts['weak']:,}, "
               f"negative={tier_counts['negative']:,}, "
               f"neutral={tier_counts['neutral']:,}")

    pairs = []
    stats = {'easy': 0, 'hard': 0, 'skipped_no_ctx': 0,
             'skipped_no_rej': 0, 'users_with_pairs': 0}
    t0 = time.time()

    for u in range(len(starts)):
        s, e = starts[u], ends[u]
        n = e - s
        if n < 2:
            continue

        # Truncate long users (keep most recent)
        if n > max_items_per_user:
            offset = n - max_items_per_user
            s = s + offset
            n = max_items_per_user

        user_iids = iids_s[s:e]
        user_ts = ts_s[s:e]
        user_tiers = tiers[s:e]

        # Map iids to SID tokens (only items in SID dict)
        user_sids = []
        user_has_sid = []
        for iid in user_iids:
            sid = content_to_tokens.get(iid)
            if sid is not None:
                user_sids.append(sid)
                user_has_sid.append(True)
            else:
                user_sids.append(None)
                user_has_sid.append(False)

        # Collect same-user rejected pools (items before each eval item)
        # We collect ALL negative/weak items for this user as rejected candidates
        user_negative_sids = []
        user_weak_sids = []
        for i in range(n):
            if not user_has_sid[i]:
                continue
            if user_tiers[i] == 'negative':
                user_negative_sids.append(user_sids[i])
            elif user_tiers[i] == 'weak':
                user_weak_sids.append(user_sids[i])

        # Skip user if no rejected candidates at all
        if difficulty == 'easy' and not user_negative_sids:
            continue
        if difficulty == 'hard' and not user_weak_sids:
            continue
        if difficulty == 'all' and not user_negative_sids and not user_weak_sids:
            continue

        user_has_pair = False

        # For each eval item (after split_ts) that is strong positive
        for i in range(n):
            if user_ts[i] <= split_ts:
                continue
            if user_tiers[i] != 'strong':
                continue
            if not user_has_sid[i]:
                continue

            # Build context: all positive items (strong + weak) before this one
            ctx_tokens = []
            for j in range(i):
                if user_has_sid[j] and user_tiers[j] in ('strong', 'weak'):
                    ctx_tokens.extend(user_sids[j])
            if len(ctx_tokens) < n_layers:
                stats['skipped_no_ctx'] += 1
                continue

            chosen_sid = user_sids[i]

            # Sample rejected (same user)
            rej_easy = [sid for sid in user_negative_sids
                        if sid != chosen_sid][:n_rejected]
            rej_hard = [sid for sid in user_weak_sids
                        if sid != chosen_sid][:n_rejected]

            has_valid = False
            if difficulty == 'all':
                has_valid = len(rej_easy) > 0 or len(rej_hard) > 0
            elif difficulty == 'easy':
                has_valid = len(rej_easy) > 0
            elif difficulty == 'hard':
                has_valid = len(rej_hard) > 0

            if not has_valid:
                stats['skipped_no_rej'] += 1
                continue

            stats['easy'] += len(rej_easy)
            stats['hard'] += len(rej_hard)
            user_has_pair = True

            pairs.append({
                'context': ctx_tokens,
                'chosen': chosen_sid,
                'rejected_easy': rej_easy,
                'rejected_medium': [],  # unused in RF-DPO
                'rejected_hard': rej_hard,
            })

            if max_samples and len(pairs) >= max_samples:
                break

        if user_has_pair:
            stats['users_with_pairs'] += 1

        if max_samples and len(pairs) >= max_samples:
            break

        if verbose and (u + 1) % 100000 == 0:
            elapsed = time.time() - t0
            rate = (u + 1) / elapsed
            remaining = (len(starts) - u - 1) / rate
            mins, secs = divmod(int(remaining), 60)
            hrs, mins_r = divmod(mins, 60)
            eta = f"{hrs}h{mins_r:02d}m" if hrs else f"{mins}m{secs:02d}s"
            print(f"    [{u+1}/{len(starts)} users] {len(pairs)} pairs, "
                  f"ETA {eta}")

    elapsed = time.time() - t0
    n_pairs = len(pairs)
    verbose_fn(f"\n  RF-DPO pairs: {n_pairs:,} in {elapsed:.1f}s")
    if n_pairs > 0:
        verbose_fn(f"    Avg rejected/pair: "
                   f"easy={stats['easy']/n_pairs:.1f}, "
                   f"hard={stats['hard']/n_pairs:.1f}")
    verbose_fn(f"    Users with pairs: {stats['users_with_pairs']:,}")
    verbose_fn(f"    Skipped (no context): {stats['skipped_no_ctx']:,}")
    verbose_fn(f"    Skipped (no rejected): {stats['skipped_no_rej']:,}")

    return pairs


# ============================================================
# CLI entry point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Build RF-DPO preference pairs from real user feedback')
    parser.add_argument('--sid_cache', type=str, required=True,
                        help='Path to SID cache dir (semantic_ids.npy)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for preference pair shards')
    parser.add_argument('--difficulty', type=str, default='all',
                        choices=['easy', 'hard', 'all'],
                        help='Which difficulty levels to include')
    parser.add_argument('--n_rejected', type=int, default=20,
                        help='Max rejected per difficulty per sample')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Cap on total pairs (for debugging)')
    parser.add_argument('--date_start', type=str, default=None,
                        help='Behavior data start date (YYYY-MM-DD)')
    parser.add_argument('--date_end', type=str, default=None,
                        help='Behavior data end date (YYYY-MM-DD)')
    parser.add_argument('--n_eval_target', type=int, default=50000,
                        help='Target eval items for time split')
    return parser.parse_args()


def main():
    args = parse_args()

    # Skip if output already exists
    out_meta = os.path.join(args.output_dir, 'meta.json')
    if os.path.exists(out_meta):
        print(f"Output already exists at {args.output_dir}, skipping.")
        return

    t0 = time.time()

    print("=" * 60)
    print("RF-DPO Preference Pair Construction")
    print("=" * 60)
    print(f"  SID cache:    {args.sid_cache}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Difficulty:   {args.difficulty}")
    print(f"  N rejected:   {args.n_rejected}")

    # ── Load SID cache ──
    print(f"\nStep 1: Loading SID cache")
    sid_dict = np.load(
        os.path.join(args.sid_cache, 'semantic_ids.npy'),
        allow_pickle=True).item()

    content_to_tokens = {}
    for cid, sid_str in sid_dict.items():
        if isinstance(sid_str, str):
            content_to_tokens[cid] = [int(t) for t in sid_str.split('_')]
        else:
            content_to_tokens[cid] = [int(t) for t in sid_str]

    n_layers = len(next(iter(content_to_tokens.values())))
    print(f"  SID assignments: {len(content_to_tokens):,}, layers: {n_layers}")

    # ── Load behavior data ──
    print(f"\nStep 2: Loading behavior data")
    from eval.batch import load_all_behavior_data
    behavior_data = load_all_behavior_data(
        date_start=args.date_start, date_end=args.date_end)
    print(f"  Interactions: {len(behavior_data['uid']):,}")

    # ── Build preference pairs ──
    print(f"\nStep 3: Building RF-DPO preference pairs")
    pairs = build_rf_preference_pairs(
        behavior_data=behavior_data,
        content_to_tokens=content_to_tokens,
        n_layers=n_layers,
        n_eval_target=args.n_eval_target,
        difficulty=args.difficulty,
        n_rejected=args.n_rejected,
        max_samples=args.max_samples,
    )

    del behavior_data, sid_dict

    # ── Save ──
    print(f"\nStep 4: Saving")
    os.makedirs(args.output_dir, exist_ok=True)
    shard_path = os.path.join(args.output_dir, 'preference_shard_0.npz')
    save_preference_shard(pairs, shard_path, n_layers)
    file_size = os.path.getsize(shard_path) / 1e6
    print(f"  {len(pairs):,} pairs → {shard_path} ({file_size:.1f}MB)")

    # ── Save meta ──
    meta = {
        'n_shards': 1,
        'n_layers': n_layers,
        'n_pairs': len(pairs),
        'difficulty': args.difficulty,
        'n_rejected': args.n_rejected,
        'sid_cache': args.sid_cache,
        'date_start': args.date_start,
        'date_end': args.date_end,
        'source': 'rf-dpo',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone! ({elapsed:.1f}s)")


if __name__ == '__main__':
    main()
