"""Build preference pairs for SP-DPO via beam search.

For each eval item in the NTP dataset:
1. Run constrained beam search to generate candidates
2. Classify non-ground-truth candidates by prefix match difficulty:
   - Easy:   0 layers match (L0 already different)
   - Medium: 1 layer matches (L0 same, L1 different)
   - Hard:   2 layers match (L0+L1 same, L2 different)
3. Save as npz shards (one per DDP rank)

Usage:
    torchrun --nproc_per_node=8 run.py sp-dpo-prepare \\
        --sft_checkpoint experiments/ntp_checkpoints/exp015-scale-04-11M \\
        --preprocessed_dir experiments/ntp_data/exp013 \\
        --output_dir experiments/sp_dpo_data/exp017 \\
        --beam_size 50 --n_rejected 20 --difficulty all
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from ntp.model import SIDTrie, constrained_beam_search


def classify_rejected(ground_truth, beam_sids, n_layers):
    """Classify beam search results by prefix match difficulty.

    Args:
        ground_truth: list of ints, length n_layers (e.g. [10, 20, 30])
        beam_sids: tensor (K, n_layers) — all beam results
        n_layers: number of SID layers

    Returns:
        dict with keys 'easy', 'medium', 'hard', each a list of SID lists.
    """
    gt = list(ground_truth)
    result = {'easy': [], 'medium': [], 'hard': []}

    for k in range(beam_sids.size(0)):
        sid = beam_sids[k].tolist()
        if sid == gt:
            continue  # skip ground truth itself

        # Count matching prefix layers
        match_depth = 0
        for li in range(n_layers):
            if sid[li] == gt[li]:
                match_depth += 1
            else:
                break

        if match_depth == 0:
            result['easy'].append(sid)
        elif match_depth == 1:
            result['medium'].append(sid)
        elif match_depth >= 2 and match_depth < n_layers:
            result['hard'].append(sid)
        # match_depth == n_layers means identical SID (already skipped above)

    return result


def _prefix_locked_generate(
    model, ctx, gt, sid_trie, n_layers, device,
    beam_size=50, n_rejected=20,
    ctx_kv_caches=None, initial_logits=None,
):
    """Progressive prefix-locked beam search for one eval item.

    Runs up to 3 beam searches with increasing prefix lock:
    1. Full beam (no prefix) → Easy candidates (L0 ≠ GT)
    2. Lock L0=GT → all results have L0 match → Medium (L1 ≠ GT) + Hard (L1=GT, L2≠GT)
    3. Lock L0+L1=GT → all results have L0+L1 match → Hard (L2 ≠ GT)

    When ctx_kv_caches is provided, context encoding is shared across all 3 passes.

    Returns:
        (rej_easy, rej_medium, rej_hard, ctx_kv_caches, initial_logits)
    """
    gt_tensor = torch.tensor(gt, dtype=torch.long, device=device)
    seen = set()  # dedup across runs
    seen.add(tuple(gt))  # exclude ground truth

    rej_easy = []
    rej_medium = []
    rej_hard = []

    def _collect(beam_sids):
        for k in range(beam_sids.size(0)):
            sid = beam_sids[k].tolist()
            key = tuple(sid)
            if key in seen:
                continue
            seen.add(key)

            match_depth = 0
            for li in range(n_layers):
                if sid[li] == gt[li]:
                    match_depth += 1
                else:
                    break

            if match_depth == 0 and len(rej_easy) < n_rejected:
                rej_easy.append(sid)
            elif match_depth == 1 and len(rej_medium) < n_rejected:
                rej_medium.append(sid)
            elif 2 <= match_depth < n_layers and len(rej_hard) < n_rejected:
                rej_hard.append(sid)

    # Pass 1: full beam → mostly Easy candidates (also produces ctx_kv_caches)
    beams, _, ctx_kv_caches = constrained_beam_search(
        model, ctx, sid_trie, beam_size=beam_size,
        ctx_kv_caches=ctx_kv_caches, initial_logits=initial_logits)
    _collect(beams[0])

    # Pass 2: lock L0 → Medium + Hard candidates (reuse ctx_kv_caches)
    if len(rej_medium) < n_rejected or len(rej_hard) < n_rejected:
        prefix_l0 = gt_tensor[:1].unsqueeze(0)  # (1, 1)
        beams2, _, _ = constrained_beam_search(
            model, ctx, sid_trie, beam_size=beam_size, prefix=prefix_l0,
            ctx_kv_caches=ctx_kv_caches)
        _collect(beams2[0])

    # Pass 3: lock L0+L1 → Hard candidates only (reuse ctx_kv_caches)
    if len(rej_hard) < n_rejected and n_layers >= 3:
        prefix_l01 = gt_tensor[:2].unsqueeze(0)  # (1, 2)
        beams3, _, _ = constrained_beam_search(
            model, ctx, sid_trie, beam_size=beam_size, prefix=prefix_l01,
            ctx_kv_caches=ctx_kv_caches)
        _collect(beams3[0])

    return rej_easy, rej_medium, rej_hard, ctx_kv_caches, initial_logits


def _extract_eval_items(sequences, n_layers, max_samples=None):
    """Extract eval items from NTP data sequences."""
    eval_items = []
    for seq in sequences:
        split_pos = seq['split_pos']
        tokens = seq['tokens']
        eval_cids = seq['eval_cids']
        split_item_idx = split_pos // n_layers

        for ei in range(len(eval_cids)):
            item_idx = split_item_idx + ei
            if item_idx < 1:
                continue
            ctx_end = item_idx * n_layers
            context_tokens = tokens[:ctx_end]
            target_sid = tokens[ctx_end:ctx_end + n_layers]
            if len(target_sid) < n_layers:
                continue
            eval_items.append({
                'context': context_tokens,
                'target_sid': target_sid,
            })

    if max_samples and len(eval_items) > max_samples:
        random.seed(42)
        eval_items = random.sample(eval_items, max_samples)
    return eval_items


def _extract_eval_items_grouped(sequences, n_layers, max_samples=None):
    """Extract eval items grouped by sequence for cross-item KV cache sharing.

    Returns list of (seq_tokens, items) where items within each group
    are ordered by increasing context length.
    """
    groups = []
    total_items = 0
    for seq in sequences:
        split_pos = seq['split_pos']
        tokens = seq['tokens']
        eval_cids = seq['eval_cids']
        split_item_idx = split_pos // n_layers

        items = []
        for ei in range(len(eval_cids)):
            item_idx = split_item_idx + ei
            if item_idx < 1:
                continue
            ctx_end = item_idx * n_layers
            target_sid = tokens[ctx_end:ctx_end + n_layers]
            if len(target_sid) < n_layers:
                continue
            items.append({
                'ctx_end': ctx_end,
                'target_sid': target_sid,
            })
        if items:
            groups.append((tokens, items))
            total_items += len(items)

    if max_samples and total_items > max_samples:
        # Subsample items across groups, preserving within-group order
        random.seed(42)
        all_indices = []
        for gi, (_, items) in enumerate(groups):
            for ii in range(len(items)):
                all_indices.append((gi, ii))
        sampled = set(random.sample(all_indices, max_samples))

        new_groups = []
        for gi, (tokens, items) in enumerate(groups):
            new_items = [items[ii] for ii in range(len(items))
                         if (gi, ii) in sampled]
            if new_items:
                new_groups.append((tokens, new_items))
        groups = new_groups
        total_items = max_samples

    return groups, total_items


def build_preference_pairs(
    model, sequences, sid_trie, n_layers, device,
    beam_size=50, n_rejected=20, difficulty='all', max_samples=None,
    verbose=True, prefix_locked=False,
):
    """Generate preference pairs from eval items via beam search.

    Two modes:
    - prefix_locked=False (paper): one beam search from L0, classify by prefix match.
    - prefix_locked=True (ours): progressive prefix-locked beam search.
      For each eval item, runs up to 3 beam searches:
        1. Full beam (no prefix) → Easy candidates
        2. Lock L0=GT → Medium+Hard candidates (guaranteed L0 match)
        3. Lock L0+L1=GT → Hard candidates (guaranteed L0+L1 match)
      Dedup across runs. Guarantees sufficient Medium/Hard candidates.

    Uses KV cache for three levels of compute reuse:
    - Cross-step: context encoded once per beam search call
    - Cross-pass: 3 prefix-locked passes share context KV
    - Cross-item: consecutive eval items from the same sequence share
      incrementally extended context KV

    Args:
        model: NTPModel (eval mode, on device)
        sequences: list of dicts from load_shard_full()
        sid_trie: SIDTrie for constrained beam search
        n_layers: number of SID layers
        device: torch device
        beam_size: beam search width
        n_rejected: max rejected candidates per difficulty
        difficulty: 'easy', 'medium', 'hard', or 'all'
        max_samples: cap on number of eval items (for debugging)
        verbose: print progress
        prefix_locked: if True, use progressive prefix-locked beam search

    Returns:
        list of dicts, each with:
            'context': list[int] — context tokens
            'chosen': list[int] — ground truth SID (n_layers)
            'rejected_easy': list[list[int]] — easy rejected SIDs
            'rejected_medium': list[list[int]] — medium rejected SIDs
            'rejected_hard': list[list[int]] — hard rejected SIDs
    """
    use_kv_cache = hasattr(model, 'forward_cached')

    if use_kv_cache and not max_samples:
        # Grouped mode: cross-item KV cache sharing
        groups, total_items = _extract_eval_items_grouped(
            sequences, n_layers, max_samples)
    else:
        # Flat mode: fallback when max_samples shuffles ordering
        eval_items = _extract_eval_items(sequences, n_layers, max_samples)
        groups = None
        total_items = len(eval_items)

    mode_str = "prefix-locked" if prefix_locked else "paper (full beam)"
    cache_str = " + KV cache" if use_kv_cache else ""
    if verbose:
        print(f"  Building preference pairs: {total_items:,} eval items, "
              f"beam_size={beam_size}, n_rejected={n_rejected}, "
              f"mode={mode_str}{cache_str}")

    pairs = []
    stats = {'easy': 0, 'medium': 0, 'hard': 0, 'skipped': 0}
    pairs_with = {'easy': 0, 'medium': 0, 'hard': 0}
    t0 = time.time()
    idx = 0  # global item counter for progress

    def _process_item(context_tokens, gt, ctx_kv_caches=None, initial_logits=None):
        """Run beam search for one eval item, return (pair_or_None, kv, logits)."""
        ctx = torch.tensor(
            context_tokens, dtype=torch.long, device=device).unsqueeze(0)

        if prefix_locked:
            rej_easy, rej_medium, rej_hard, ctx_kv_caches, initial_logits = \
                _prefix_locked_generate(
                    model, ctx, gt, sid_trie, n_layers, device,
                    beam_size=beam_size, n_rejected=n_rejected,
                    ctx_kv_caches=ctx_kv_caches,
                    initial_logits=initial_logits,
                )
        else:
            beams, scores, ctx_kv_caches = constrained_beam_search(
                model, ctx, sid_trie, beam_size=beam_size,
                ctx_kv_caches=ctx_kv_caches, initial_logits=initial_logits)
            beam_sids = beams[0]
            classified = classify_rejected(gt, beam_sids, n_layers)
            rej_easy = classified['easy'][:n_rejected]
            rej_medium = classified['medium'][:n_rejected]
            rej_hard = classified['hard'][:n_rejected]
            if initial_logits is None and ctx_kv_caches is not None:
                initial_logits = None  # not needed for non-prefix-locked

        # Check validity
        has_valid = False
        if difficulty == 'all':
            has_valid = len(rej_easy) > 0 or len(rej_medium) > 0 or len(rej_hard) > 0
        elif difficulty == 'easy':
            has_valid = len(rej_easy) > 0
        elif difficulty == 'medium':
            has_valid = len(rej_medium) > 0
        elif difficulty == 'hard':
            has_valid = len(rej_hard) > 0

        if not has_valid:
            return None, rej_easy, rej_medium, rej_hard, ctx_kv_caches, initial_logits

        pair = {
            'context': context_tokens,
            'chosen': gt,
            'rejected_easy': rej_easy,
            'rejected_medium': rej_medium,
            'rejected_hard': rej_hard,
        }
        return pair, rej_easy, rej_medium, rej_hard, ctx_kv_caches, initial_logits

    def _record(pair, rej_easy, rej_medium, rej_hard):
        """Update stats and pairs list."""
        if pair is None:
            stats['skipped'] += 1
            return
        stats['easy'] += len(rej_easy)
        stats['medium'] += len(rej_medium)
        stats['hard'] += len(rej_hard)
        if rej_easy:
            pairs_with['easy'] += 1
        if rej_medium:
            pairs_with['medium'] += 1
        if rej_hard:
            pairs_with['hard'] += 1
        pairs.append(pair)

    def _print_progress(idx):
        if not verbose or (idx + 1) % 500 != 0:
            return
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        remaining = (total_items - idx - 1) / rate
        mins, secs = divmod(int(remaining), 60)
        hrs, mins = divmod(mins, 60)
        eta = f"{hrs}h{mins:02d}m" if hrs else f"{mins}m{secs:02d}s"
        n_pairs = len(pairs)
        print(f"    [{idx+1}/{total_items}] {n_pairs} pairs, "
              f"E={stats['easy']/max(n_pairs,1):.1f}/pair, "
              f"M={stats['medium']/max(n_pairs,1):.1f}/pair, "
              f"H={stats['hard']/max(n_pairs,1):.1f}/pair, "
              f"skip={stats['skipped']}, ETA {eta}")

    if groups is not None:
        # ── Grouped mode: cross-item KV cache sharing ──
        for seq_tokens, items in groups:
            shared_kv = None
            shared_logits = None
            prev_ctx_len = 0

            for item in items:
                ctx_end = item['ctx_end']
                context_tokens = seq_tokens[:ctx_end]
                gt = item['target_sid']

                if use_kv_cache and shared_kv is not None:
                    # Extend cache with new tokens
                    new_tokens = seq_tokens[prev_ctx_len:ctx_end]
                    if new_tokens:
                        new_tensor = torch.tensor(
                            new_tokens, dtype=torch.long,
                            device=device).unsqueeze(0)
                        shared_logits, shared_kv, _, _ = model.forward_cached(
                            generated_tokens=new_tensor, kv_caches=shared_kv)

                pair, re, rm, rh, shared_kv, shared_logits = _process_item(
                    context_tokens, gt,
                    ctx_kv_caches=shared_kv, initial_logits=shared_logits)
                _record(pair, re, rm, rh)
                prev_ctx_len = ctx_end

                _print_progress(idx)
                idx += 1
    else:
        # ── Flat mode ──
        for idx, item in enumerate(eval_items):
            pair, re, rm, rh, _, _ = _process_item(
                item['context'], item['target_sid'])
            _record(pair, re, rm, rh)
            _print_progress(idx)

    if verbose:
        elapsed = time.time() - t0
        n = len(pairs)
        print(f"  Done: {n:,} pairs from {total_items:,} eval items in {elapsed:.1f}s")
        print(f"    Mode: {mode_str}{cache_str}")
        print(f"    Per-difficulty pair counts:")
        print(f"      Easy:   {pairs_with['easy']:,} pairs, {stats['easy']:,} rejected total "
              f"(avg {stats['easy']/max(pairs_with['easy'],1):.1f}/pair)")
        print(f"      Medium: {pairs_with['medium']:,} pairs, {stats['medium']:,} rejected total "
              f"(avg {stats['medium']/max(pairs_with['medium'],1):.1f}/pair)")
        print(f"      Hard:   {pairs_with['hard']:,} pairs, {stats['hard']:,} rejected total "
              f"(avg {stats['hard']/max(pairs_with['hard'],1):.1f}/pair)")
        print(f"    Skipped (no valid rejected): {stats['skipped']}")

    return pairs


def save_preference_shard(pairs, path, n_layers):
    """Save preference pairs as compressed npz.

    Format: flat arrays + offset arrays for variable-length data,
    following the pattern from ntp/preprocess.py.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not pairs:
        np.savez_compressed(
            path,
            context_flat=np.array([], dtype=np.int32),
            context_offsets=np.array([0], dtype=np.int64),
            chosen_sids=np.array([], dtype=np.int32).reshape(0, n_layers),
            rejected_easy_flat=np.array([], dtype=np.int32),
            rejected_easy_offsets=np.array([0], dtype=np.int64),
            rejected_medium_flat=np.array([], dtype=np.int32),
            rejected_medium_offsets=np.array([0], dtype=np.int64),
            rejected_hard_flat=np.array([], dtype=np.int32),
            rejected_hard_offsets=np.array([0], dtype=np.int64),
            n_layers=np.array(n_layers, dtype=np.int32),
        )
        return

    context_flat = []
    context_offsets = [0]
    chosen_sids = []

    for key in ('rejected_easy', 'rejected_medium', 'rejected_hard'):
        locals()[f'{key}_flat'] = []
        locals()[f'{key}_offsets'] = [0]

    rej_easy_flat, rej_easy_offsets = [], [0]
    rej_medium_flat, rej_medium_offsets = [], [0]
    rej_hard_flat, rej_hard_offsets = [], [0]

    for pair in pairs:
        context_flat.extend(pair['context'])
        context_offsets.append(context_offsets[-1] + len(pair['context']))
        chosen_sids.append(pair['chosen'])

        for sid in pair['rejected_easy']:
            rej_easy_flat.extend(sid)
        rej_easy_offsets.append(rej_easy_offsets[-1] + len(pair['rejected_easy']))

        for sid in pair['rejected_medium']:
            rej_medium_flat.extend(sid)
        rej_medium_offsets.append(rej_medium_offsets[-1] + len(pair['rejected_medium']))

        for sid in pair['rejected_hard']:
            rej_hard_flat.extend(sid)
        rej_hard_offsets.append(rej_hard_offsets[-1] + len(pair['rejected_hard']))

    np.savez_compressed(
        path,
        context_flat=np.array(context_flat, dtype=np.int32),
        context_offsets=np.array(context_offsets, dtype=np.int64),
        chosen_sids=np.array(chosen_sids, dtype=np.int32),
        rejected_easy_flat=np.array(rej_easy_flat, dtype=np.int32),
        rejected_easy_offsets=np.array(rej_easy_offsets, dtype=np.int64),
        rejected_medium_flat=np.array(rej_medium_flat, dtype=np.int32),
        rejected_medium_offsets=np.array(rej_medium_offsets, dtype=np.int64),
        rejected_hard_flat=np.array(rej_hard_flat, dtype=np.int32),
        rejected_hard_offsets=np.array(rej_hard_offsets, dtype=np.int64),
        n_layers=np.array(n_layers, dtype=np.int32),
    )


def load_preference_shard(path):
    """Load preference pairs from npz shard.

    Returns:
        list of dicts with 'context', 'chosen', 'rejected_easy/medium/hard'.
    """
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


def _load_model(ckpt_path, device):
    """Load NTPModel from checkpoint."""
    from ntp.model import NTPModel
    from ntp.baseline import NTPProbe

    ckpt = torch.load(
        os.path.join(ckpt_path, 'probe.pt'),
        map_location=device, weights_only=False)
    cfg = ckpt['config']
    model_type = cfg.pop('model_type', 'probe')

    if model_type == 's-tier':
        model = NTPModel(**cfg)
    else:
        model = NTPProbe(**cfg)

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    return model, cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Build SP-DPO preference pairs')
    parser.add_argument('--sft_checkpoint', type=str, required=True,
                        help='Path to SFT model checkpoint directory')
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                        help='Path to preprocessed NTP data shards')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for preference pair shards')
    parser.add_argument('--beam_size', type=int, default=50)
    parser.add_argument('--n_rejected', type=int, default=20,
                        help='Max rejected candidates per difficulty level')
    parser.add_argument('--difficulty', type=str, default='all',
                        choices=['easy', 'medium', 'hard', 'all'],
                        help='Which difficulty levels to include')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Cap on eval items (for debugging)')
    parser.add_argument('--prefix_locked', action='store_true',
                        help='Use prefix-locked beam search for guaranteed M/H candidates')
    return parser.parse_args()


def main():
    args = parse_args()

    # Skip if output already exists
    out_meta = os.path.join(args.output_dir, 'meta.json')
    if os.path.exists(out_meta):
        print(f"Output already exists at {args.output_dir}, skipping.")
        return

    # DDP setup (optional — works single-GPU too)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        import torch.distributed as dist
        dist.init_process_group('nccl', timeout=__import__('datetime').timedelta(minutes=30))

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main = (local_rank == 0)

    if is_main:
        print(f"SP-DPO Preference Pair Construction")
        print(f"  SFT checkpoint: {args.sft_checkpoint}")
        print(f"  Preprocessed:   {args.preprocessed_dir}")
        print(f"  Output:         {args.output_dir}")
        print(f"  Beam size:      {args.beam_size}")
        print(f"  N rejected:     {args.n_rejected}")
        print(f"  Difficulty:     {args.difficulty}")
        print(f"  Prefix locked:  {args.prefix_locked}")
        print(f"  World size:     {world_size}")

    # Load meta
    meta_path = os.path.join(args.preprocessed_dir, 'meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta['n_layers']
    sid_cache_dir = meta['sid_cache']

    # Load model
    if is_main:
        print(f"\n  Loading model from {args.sft_checkpoint}...")
    model, cfg = _load_model(args.sft_checkpoint, device)

    # Build SID trie
    from ntp.eval import _build_sid_to_items
    sid_to_items = _build_sid_to_items(sid_cache_dir)
    sid_trie = SIDTrie(sid_to_items, n_layers)

    # Load this rank's shard
    from ntp.preprocess import load_shard_full
    shard_path = os.path.join(args.preprocessed_dir, f'train_shard_{local_rank}.npz')
    if not os.path.exists(shard_path):
        # Fallback: single shard
        shard_path = os.path.join(args.preprocessed_dir, 'train_shard_0.npz')
    sequences = load_shard_full(shard_path)
    # Keep only sequences with eval items
    sequences = [s for s in sequences
                 if s['split_pos'] < len(s['tokens']) and len(s['eval_cids']) > 0]

    if is_main:
        print(f"  Rank {local_rank}: {len(sequences)} sequences with eval items")

    # Build preference pairs
    gen_t0 = time.time()
    with torch.no_grad():
        pairs = build_preference_pairs(
            model, sequences, sid_trie, n_layers, device,
            beam_size=args.beam_size,
            n_rejected=args.n_rejected,
            difficulty=args.difficulty,
            max_samples=args.max_samples,
            verbose=is_main,
            prefix_locked=args.prefix_locked,
        )
    gen_elapsed = time.time() - gen_t0

    # Save shard
    os.makedirs(args.output_dir, exist_ok=True)
    shard_out = os.path.join(args.output_dir, f'preference_shard_{local_rank}.npz')
    save_preference_shard(pairs, shard_out, n_layers)

    if is_main:
        print(f"\n  Saved {len(pairs)} pairs to {shard_out}")

    # Save meta
    if is_main:
        meta_out = {
            'n_shards': world_size,
            'n_layers': n_layers,
            'beam_size': args.beam_size,
            'n_rejected': args.n_rejected,
            'difficulty': args.difficulty,
            'prefix_locked': args.prefix_locked,
            'sft_checkpoint': args.sft_checkpoint,
            'preprocessed_dir': args.preprocessed_dir,
            'n_pairs_rank0': len(pairs),
            'wall_time_s': round(gen_elapsed, 1),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
            json.dump(meta_out, f, indent=2)
        print(f"  Meta saved to {args.output_dir}/meta.json")

    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()
