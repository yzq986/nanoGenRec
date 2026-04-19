"""Build preference pairs for SP-DPO.

For each eval item in the NTP dataset, generates rejected candidates
at three difficulty levels:
   - Easy:   0 layers match (L0 already different)
   - Medium: 1 layer matches (L0 same, L1 different)
   - Hard:   2 layers match (L0+L1 same, L2 different)

Hard and Medium use specialized generation (fix GT prefix, sample
diverging layer) instead of beam search — mathematically equivalent
to beam search filtering but orders of magnitude faster and batched.
Easy uses beam search for diverse non-matching candidates.

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
import torch.nn.functional as F

from gr_demo.ntp.model import SIDTrie, constrained_beam_search


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


def _generate_negatives_batched(
    model, ctx_padded, ctx_lengths, gt_sids, n_layers, n_rejected, device,
):
    """Generate Hard and Medium negatives in a single batched forward pass.

    Hard:   fix GT L0+L1, take top-K L2 (exclude GT_L2)
    Medium: fix GT L0, take top-K L1 (exclude GT_L1), then top-1 L2 per L1

    Args:
        model: NTPModel on device
        ctx_padded: (B, T_ctx) right-padded context tokens
        ctx_lengths: (B,) actual context lengths
        gt_sids: (B, n_layers) ground truth SID tokens
        n_layers: number of SID layers (must be 3)
        n_rejected: max rejected per difficulty per sample
        device: torch device

    Returns:
        hard_rejected: list of B lists, each containing up to n_rejected SIDs
        medium_rejected: list of B lists, each containing up to n_rejected SIDs
    """
    B = ctx_padded.size(0)

    # Build input: [ctx..., GT_L0, GT_L1] — enough to predict all 3 layers
    sid_prefix = gt_sids[:, :-1]  # (B, n_layers-1) = (B, 2)
    full_input = torch.cat([ctx_padded, sid_prefix], dim=1)  # (B, T_ctx + 2)
    T = full_input.size(1)

    # Single forward pass — gets hidden states for all prediction positions
    positions = torch.arange(T, device=device).unsqueeze(0)
    x = model._embed_tokens(full_input) + model.pos_emb(positions)
    hidden = model._transformer_forward(x)  # (B, T, D)

    batch_idx = torch.arange(B, device=device)

    # ── Hard: top-K L2 given GT L0+L1 ──
    pos_l2 = ctx_lengths - 1 + 2  # position predicting L2
    h_l2 = hidden[batch_idx, pos_l2]  # (B, D)
    logits_l2 = model.output_projs[2](h_l2)  # (B, C2)
    # Mask out GT L2 so it doesn't appear in rejected
    logits_l2[batch_idx, gt_sids[:, 2]] = float('-inf')
    topk_l2 = logits_l2.topk(min(n_rejected, logits_l2.size(1)), dim=-1).indices  # (B, K)

    hard_rejected = []
    for b in range(B):
        sids = []
        for k in range(topk_l2.size(1)):
            sids.append([gt_sids[b, 0].item(), gt_sids[b, 1].item(), topk_l2[b, k].item()])
        hard_rejected.append(sids)

    # ── Medium: top-K L1 given GT L0, then top-1 L2 per L1 candidate ──
    pos_l1 = ctx_lengths - 1 + 1  # position predicting L1
    h_l1 = hidden[batch_idx, pos_l1]  # (B, D)
    logits_l1 = model.output_projs[1](h_l1)  # (B, C1)
    # Mask out GT L1
    logits_l1[batch_idx, gt_sids[:, 1]] = float('-inf')
    topk_l1 = logits_l1.topk(min(n_rejected, logits_l1.size(1)), dim=-1).indices  # (B, K)
    K_med = topk_l1.size(1)

    # For each L1 candidate, we need P(L2 | ctx, GT_L0, L1_cand).
    # Build batched input: (B*K, T_ctx + 2) with [ctx, GT_L0, L1_cand]
    ctx_exp = ctx_padded.unsqueeze(1).expand(-1, K_med, -1).reshape(B * K_med, -1)
    len_exp = ctx_lengths.unsqueeze(1).expand(-1, K_med).reshape(B * K_med)
    gt_l0_exp = gt_sids[:, 0:1].unsqueeze(1).expand(-1, K_med, -1).reshape(B * K_med, 1)
    l1_cands = topk_l1.reshape(B * K_med, 1)  # (B*K, 1)

    med_input = torch.cat([ctx_exp, gt_l0_exp, l1_cands], dim=1)  # (B*K, T_ctx + 2)
    T_med = med_input.size(1)

    # Forward pass for medium (may need chunking for large B*K)
    chunk_size = max(1, 40_000_000 // (T_med * T_med))  # conservative memory cap
    total = B * K_med
    med_l2_tokens = torch.zeros(total, dtype=torch.long, device=device)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_input = med_input[start:end]
        chunk_len = len_exp[start:end]
        T_c = chunk_input.size(1)
        pos_c = torch.arange(T_c, device=device).unsqueeze(0)
        x_c = model._embed_tokens(chunk_input) + model.pos_emb(pos_c)
        h_c = model._transformer_forward(x_c)
        bidx_c = torch.arange(end - start, device=device)
        pos_l2_c = chunk_len - 1 + 2
        h_l2_c = h_c[bidx_c, pos_l2_c]
        logits_l2_c = model.output_projs[2](h_l2_c)
        med_l2_tokens[start:end] = logits_l2_c.argmax(dim=-1)

    med_l2_tokens = med_l2_tokens.reshape(B, K_med)

    medium_rejected = []
    for b in range(B):
        sids = []
        for k in range(K_med):
            sids.append([gt_sids[b, 0].item(), topk_l1[b, k].item(), med_l2_tokens[b, k].item()])
        medium_rejected.append(sids)

    return hard_rejected, medium_rejected


def build_preference_pairs(
    model, sequences, sid_trie, n_layers, device,
    beam_size=50, n_rejected=20, difficulty='all', max_samples=None,
    verbose=True, gen_batch_size=64,
):
    """Generate preference pairs using specialized per-difficulty generation.

    Hard/Medium: batched forward pass (fix GT prefix, sample diverging layer).
    Easy: beam search (diverse non-matching candidates).

    Args:
        model: NTPModel (eval mode, on device)
        sequences: list of dicts from load_shard_full()
        sid_trie: SIDTrie for constrained beam search
        n_layers: number of SID layers
        device: torch device
        beam_size: beam search width (used for Easy only)
        n_rejected: max rejected candidates per difficulty
        difficulty: 'easy', 'medium', 'hard', or 'all'
        max_samples: cap on number of eval items (for debugging)
        verbose: print progress
        gen_batch_size: batch size for Hard/Medium batched generation

    Returns:
        list of dicts, each with:
            'context': list[int] — context tokens
            'chosen': list[int] — ground truth SID (n_layers)
            'rejected_easy': list[list[int]] — easy rejected SIDs
            'rejected_medium': list[list[int]] — medium rejected SIDs
            'rejected_hard': list[list[int]] — hard rejected SIDs
    """
    # Extract eval items (same pattern as ntp/eval.py:252-273)
    eval_items = []
    for seq in sequences:
        split_pos = seq['split_pos']
        tokens = seq['tokens']
        eval_cids = seq['eval_cids']
        n_items_in_seq = len(tokens) // n_layers
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

    need_easy = difficulty in ('easy', 'all')
    need_medium = difficulty in ('medium', 'all')
    need_hard = difficulty in ('hard', 'all')

    if verbose:
        mode_str = []
        if need_hard or need_medium:
            mode_str.append(f"batched (Hard/Medium, bs={gen_batch_size})")
        if need_easy:
            mode_str.append(f"beam search (Easy, beam={beam_size})")
        print(f"  Building preference pairs: {len(eval_items):,} eval items, "
              f"n_rejected={n_rejected}, mode={' + '.join(mode_str)}")

    pairs = []
    stats = {'easy': 0, 'medium': 0, 'hard': 0, 'skipped': 0}
    t0 = time.time()

    # ── Phase 1: Batched Hard/Medium generation ──
    hard_all = [[] for _ in range(len(eval_items))]
    medium_all = [[] for _ in range(len(eval_items))]

    if need_hard or need_medium:
        for batch_start in range(0, len(eval_items), gen_batch_size):
            batch_end = min(batch_start + gen_batch_size, len(eval_items))
            batch_items = eval_items[batch_start:batch_end]
            B = len(batch_items)

            # Pad contexts
            ctx_lens = [len(item['context']) for item in batch_items]
            max_ctx = max(ctx_lens)
            ctx_padded = torch.zeros(B, max_ctx, dtype=torch.long, device=device)
            ctx_lengths = torch.tensor(ctx_lens, dtype=torch.long, device=device)
            for i, item in enumerate(batch_items):
                ctx_padded[i, :ctx_lens[i]] = torch.tensor(item['context'], dtype=torch.long)

            gt_sids = torch.tensor(
                [item['target_sid'] for item in batch_items],
                dtype=torch.long, device=device)

            hard_batch, medium_batch = _generate_negatives_batched(
                model, ctx_padded, ctx_lengths, gt_sids, n_layers, n_rejected, device)

            for i in range(B):
                idx = batch_start + i
                if need_hard:
                    hard_all[idx] = hard_batch[i]
                if need_medium:
                    medium_all[idx] = medium_batch[i]

            if verbose and (batch_end % (gen_batch_size * 10) == 0 or batch_end == len(eval_items)):
                elapsed = time.time() - t0
                rate = batch_end / elapsed
                remaining = (len(eval_items) - batch_end) / rate if rate > 0 else 0
                mins, secs = divmod(int(remaining), 60)
                eta = f"{mins}m{secs:02d}s"
                print(f"    [Hard/Medium] {batch_end}/{len(eval_items)} "
                      f"({rate:.0f} items/s, ETA {eta})")

    # ── Phase 2: Beam search for Easy (sequential, slower) ──
    easy_all = [[] for _ in range(len(eval_items))]

    if need_easy:
        t1 = time.time()
        for idx, item in enumerate(eval_items):
            ctx = torch.tensor(item['context'], dtype=torch.long, device=device).unsqueeze(0)
            gt = item['target_sid']

            beams, scores = constrained_beam_search(model, ctx, sid_trie, beam_size=beam_size)
            beam_sids = beams[0]  # (K, n_layers)

            classified = classify_rejected(gt, beam_sids, n_layers)
            easy_all[idx] = classified['easy'][:n_rejected]

            if verbose and (idx + 1) % 500 == 0:
                elapsed = time.time() - t1
                rate = (idx + 1) / elapsed
                remaining = (len(eval_items) - idx - 1) / rate
                mins, secs = divmod(int(remaining), 60)
                hrs, mins = divmod(mins, 60)
                eta = f"{hrs}h{mins:02d}m" if hrs else f"{mins}m{secs:02d}s"
                print(f"    [Easy beam] {idx+1}/{len(eval_items)} "
                      f"({rate:.1f} items/s, ETA {eta})")

    # ── Assemble pairs ──
    for idx, item in enumerate(eval_items):
        rej_easy = easy_all[idx]
        rej_medium = medium_all[idx]
        rej_hard = hard_all[idx]

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
            stats['skipped'] += 1
            continue

        stats['easy'] += len(rej_easy)
        stats['medium'] += len(rej_medium)
        stats['hard'] += len(rej_hard)

        pairs.append({
            'context': item['context'],
            'chosen': item['target_sid'],
            'rejected_easy': rej_easy,
            'rejected_medium': rej_medium,
            'rejected_hard': rej_hard,
        })

    if verbose:
        elapsed = time.time() - t0
        n = len(pairs)
        print(f"  Done: {n:,} pairs in {elapsed:.1f}s")
        if n > 0:
            print(f"    Avg per pair: easy={stats['easy']/n:.1f}, "
                  f"medium={stats['medium']/n:.1f}, hard={stats['hard']/n:.1f}")
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
    from gr_demo.ntp.model import NTPModel
    from gr_demo.ntp.baseline import NTPProbe

    ckpt = torch.load(
        os.path.join(ckpt_path, 'probe.pt'),
        map_location=device, weights_only=False)
    cfg = ckpt['config']
    model_type = cfg.pop('model_type', 'probe')

    if model_type == 's-tier':
        model = NTPModel(**cfg)
    else:
        model = NTPProbe(**cfg)

    model.load_state_dict(ckpt['model_state_dict'])
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
    return parser.parse_args()


def main():
    args = parse_args()

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
    from gr_demo.ntp.eval import _build_sid_to_items
    sid_to_items = _build_sid_to_items(sid_cache_dir)
    sid_trie = SIDTrie(sid_to_items, n_layers)

    # Load this rank's shard
    from gr_demo.ntp.preprocess import load_shard_full
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
    with torch.no_grad():
        pairs = build_preference_pairs(
            model, sequences, sid_trie, n_layers, device,
            beam_size=args.beam_size,
            n_rejected=args.n_rejected,
            difficulty=args.difficulty,
            max_samples=args.max_samples,
            verbose=is_main,
        )

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
            'sft_checkpoint': args.sft_checkpoint,
            'preprocessed_dir': args.preprocessed_dir,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
            json.dump(meta_out, f, indent=2)
        print(f"  Meta saved to {args.output_dir}/meta.json")

    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()
