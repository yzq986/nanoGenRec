"""
Semantic ID Next Token Prediction — Eval Only.

Loads a pre-trained NTP checkpoint and runs eval using unified sequences.

Teacher-forced metrics (PPL, depth hit@10) computed in one batched forward pass
on eval positions (pos >= split_pos). Beam search recall only on small subsample.

Usage:
    1. Train:  torchrun --nproc_per_node=8 run.py train-ntp --preprocessed_dir ...
    2. Eval:   python run.py hyperparam --ntp_checkpoint ... --run_ntp
"""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gr_demo.metrics.base import BaseMetric, MetricResult
from gr_demo.ntp.baseline import NTPProbe
from gr_demo.ntp.model import NTPModel, SIDTrie, constrained_beam_search


def _build_sid_to_items(sid_cache_dir):
    """Rebuild sid_to_items from SID cache (semantic_ids.npy)."""
    from collections import defaultdict
    sid_dict = np.load(
        os.path.join(sid_cache_dir, 'semantic_ids.npy'), allow_pickle=True
    ).item()
    sid_to_items = defaultdict(set)
    for cid, sid_str in sid_dict.items():
        if isinstance(sid_str, str):
            sid_to_items[sid_str].add(cid)
        else:
            sid_to_items['_'.join(str(t) for t in sid_str)].add(cid)
    return dict(sid_to_items)


def _load_eval_sequences(preprocessed_dir, n_shards):
    """Load all shards and return unified sequences that have eval positions."""
    from gr_demo.ntp.preprocess import load_shard_full
    all_seqs = []
    for i in range(n_shards):
        shard_path = os.path.join(preprocessed_dir, f'train_shard_{i}.npz')
        seqs = load_shard_full(shard_path)
        # Keep only sequences that have eval items (split_pos < len(tokens))
        for s in seqs:
            if s['split_pos'] < len(s['tokens']) and len(s['eval_cids']) > 0:
                all_seqs.append(s)
    return all_seqs


def _batched_teacher_forced_eval(probe, sequences, n_layers, device, batch_size=2048,
                                 verbose=True):
    """Batched teacher-forced eval on eval positions of unified sequences.

    For each sequence, eval positions are where the predicted token index >= split_pos.
    In LM-style (input=tokens[:-1], target=tokens[1:]):
      position i predicts token[i+1].
      token[i+1] is eval when i+1 >= split_pos, i.e. i >= split_pos - 1.

    Returns dict with:
        'avg_loss': float
        'ppl': float
        'depth_hit_10': list of per-layer hit@10 rates
        'n_eval_positions': int (total eval positions across all sequences)
    """
    n_seqs = len(sequences)
    total_loss = 0.0
    total_eval_positions = 0
    total_eval_items = 0
    # Per-layer loss tracking
    per_layer_loss = [0.0] * n_layers
    per_layer_count = [0] * n_layers
    # Per-layer independent hit@10
    indep_hit_counts = [0] * n_layers
    # Prefix-based hit@10: layer li = "layers 0..li ALL hit"
    prefix_hit_counts = [0] * n_layers

    # Sort by length for efficient batching
    sorted_indices = sorted(range(n_seqs), key=lambda i: len(sequences[i]['tokens']))

    t0 = time.time()
    processed = 0

    for batch_start in range(0, n_seqs, batch_size):
        batch_end = min(batch_start + batch_size, n_seqs)
        batch_indices = sorted_indices[batch_start:batch_end]
        batch_seqs = [sequences[i] for i in batch_indices]
        B = len(batch_seqs)

        tokens_list = [s['tokens'] for s in batch_seqs]
        split_positions = torch.tensor([s['split_pos'] for s in batch_seqs],
                                       dtype=torch.long, device=device)
        lengths = torch.tensor([len(t) for t in tokens_list],
                               dtype=torch.long, device=device)

        max_len = lengths.max().item()
        padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, toks in enumerate(tokens_list):
            padded[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)

        # LM-style shift
        input_tokens = padded[:, :-1]
        target_tokens = padded[:, 1:]
        T = input_tokens.size(1)

        arange = torch.arange(T, device=device).unsqueeze(0)
        valid_mask = arange < (lengths.unsqueeze(1) - 1)

        # Eval mask: position i is eval when i+1 >= split_pos, i.e. i >= split_pos - 1
        eval_mask = valid_mask & (arange >= (split_positions.unsqueeze(1) - 1))

        if not eval_mask.any():
            continue

        # Forward pass
        L = n_layers
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = probe._embed_tokens(input_tokens) + probe.pos_emb(positions)

        if hasattr(probe, 'encoder'):
            hidden = probe.encoder(x, is_causal=True)
        else:
            hidden = probe._transformer_forward(x)

        # Per-layer loss and per-position hit@10
        hidden_flat = hidden.reshape(-1, hidden.size(-1))
        target_flat = target_tokens.reshape(-1)
        mask_flat = eval_mask.reshape(-1)

        # Position → layer mapping: target at position i has layer (i+1) % L
        pos_layer = ((torch.arange(T, device=device) + 1) % L)
        pos_layer_2d = pos_layer.unsqueeze(0).expand(B, -1)  # (B, T)
        pos_layer_flat = pos_layer_2d.reshape(-1)

        # Per-position hit@10 stored in (B, T) tensor
        hit_2d = torch.zeros(B, T, dtype=torch.bool, device=device)

        batch_loss = 0.0
        batch_n = 0
        for li in range(L):
            layer_mask = mask_flat & (pos_layer_flat == li)
            if not layer_mask.any():
                continue
            logits = probe.output_projs[li](hidden_flat[layer_mask])
            targets_l = target_flat[layer_mask]

            layer_loss = F.cross_entropy(logits, targets_l, reduction='sum').item()
            layer_n = layer_mask.sum().item()
            batch_loss += layer_loss
            batch_n += layer_n
            per_layer_loss[li] += layer_loss
            per_layer_count[li] += layer_n

            topk_vals = min(10, logits.size(-1))
            hit = (logits.topk(topk_vals, dim=-1).indices == targets_l.unsqueeze(1)).any(dim=-1)

            # Independent per-layer count
            indep_hit_counts[li] += hit.sum().item()

            # Store hits back into (B, T) for prefix tracking
            layer_mask_2d = eval_mask & (pos_layer_2d == li)
            hit_2d[layer_mask_2d] = hit

        # Prefix-based hit@10 across layers (item-level)
        # Layer-0 eval positions mark the start of each eval item
        layer0_eval = eval_mask & (pos_layer_2d == 0)  # (B, T)
        l0_b, l0_t = torch.where(layer0_eval)
        n_items_batch = len(l0_b)

        for li in range(L):
            check_t = l0_t + li
            valid = check_t < T
            hits_at_li = torch.zeros(n_items_batch, dtype=torch.bool, device=device)
            hits_at_li[valid] = hit_2d[l0_b[valid], check_t[valid]]

            if li == 0:
                prefix_hit = hits_at_li
            else:
                prefix_hit = prefix_hit & hits_at_li

            prefix_hit_counts[li] += prefix_hit.sum().item()

        total_eval_items += n_items_batch
        total_loss += batch_loss
        total_eval_positions += batch_n
        processed += B

        if verbose and processed % (batch_size * 4) < batch_size:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (n_seqs - processed) / rate if rate > 0 else 0
            eta_m, eta_s = divmod(int(remaining), 60)
            print(f"    teacher-forced: {processed:,}/{n_seqs:,} seqs "
                  f"({rate:.0f} seqs/s, ETA {eta_m}m{eta_s:02d}s)")

    if total_eval_positions == 0:
        return {'avg_loss': 0, 'ppl': 1.0, 'depth_hit_10': [0] * n_layers,
                'depth_hit_10_indep': [0] * n_layers,
                'n_eval_positions': 0, 'n_eval_items': 0}

    n_per_layer = total_eval_positions // n_layers
    avg_loss = total_loss / total_eval_positions
    ppl = np.exp(avg_loss)
    indep_h10 = [c / max(n_per_layer, 1) for c in indep_hit_counts]
    prefix_h10 = [c / max(total_eval_items, 1) for c in prefix_hit_counts]
    layer_ppl = [np.exp(per_layer_loss[li] / max(per_layer_count[li], 1))
                 for li in range(n_layers)]

    if verbose:
        print(f"  Per-layer PPL: {['L' + str(i) + '=' + f'{p:.2f}' for i, p in enumerate(layer_ppl)]}")
        print(f"    L0 = cross-item prediction (hard)")
        print(f"    L1..L{n_layers-1} = intra-item autocompletion (teacher-forced, easy)")

    return {
        'avg_loss': avg_loss,
        'ppl': ppl,
        'layer_ppl': layer_ppl,
        'depth_hit_10': prefix_h10,
        'depth_hit_10_indep': indep_h10,
        'n_eval_positions': total_eval_positions,
        'n_eval_items': total_eval_items,
    }


def _beam_search_recall(probe, sequences, sid_trie, sid_to_items, n_layers,
                        device, recall_beam_size=500, n_recall_samples=5000,
                        verbose=True):
    """Beam search recall on a small subsample of eval items.

    For each sampled eval item, uses all preceding tokens as context,
    runs constrained beam search, and checks recall@K.

    Returns dict with recall@K values.
    """
    # Collect all eval items across sequences
    eval_items = []
    for seq in sequences:
        split_pos = seq['split_pos']
        tokens = seq['tokens']
        eval_cids = seq['eval_cids']
        n_items_in_seq = len(tokens) // n_layers
        split_item_idx = split_pos // n_layers

        for ei, cid in enumerate(eval_cids):
            item_idx = split_item_idx + ei
            if item_idx < 1:  # need at least 1 preceding item for context
                continue
            # Context: all tokens before this item
            ctx_end = item_idx * n_layers
            context_tokens = tokens[:ctx_end]
            # Target: this item's SID tokens
            target_tokens = tokens[ctx_end:ctx_end + n_layers]
            eval_items.append({
                'context': context_tokens,
                'target': target_tokens,
                'cid': cid,
            })

    if not eval_items:
        return {}

    # Subsample
    if len(eval_items) > n_recall_samples:
        random.seed(42)
        eval_items = random.sample(eval_items, n_recall_samples)

    if verbose:
        print(f"  Beam search recall on {len(eval_items):,} items "
              f"(beam_size={recall_beam_size})")

    recall_ks = [10, 50, 100, 500]
    item_recall = {k: 0 for k in recall_ks}
    depth_correct = [0] * n_layers
    target_sid_found = 0  # how often the correct SID appears anywhere in beams
    t0 = time.time()

    for idx, item in enumerate(eval_items):
        ctx = torch.tensor(item['context'], dtype=torch.long, device=device).unsqueeze(0)
        target = item['target']
        target_cid = item['cid']
        target_sid_str = '_'.join(str(t) for t in target)

        beams, scores = constrained_beam_search(
            probe, ctx, sid_trie, beam_size=recall_beam_size)

        # Check if target SID appears in any beam
        found_rank = -1
        for ki in range(beams.size(1)):
            sid_str = '_'.join(str(t.item()) for t in beams[0, ki])
            if sid_str == target_sid_str:
                found_rank = ki
                break
        if found_rank >= 0:
            target_sid_found += 1

        # Depth accuracy (top-1 beam)
        pred_top1 = beams[0, 0]
        prefix_ok = True
        for li in range(n_layers):
            prefix_ok = prefix_ok and (pred_top1[li].item() == target[li])
            if prefix_ok:
                depth_correct[li] += 1
            else:
                break

        # Item recall
        candidates = []
        seen = set()
        for ki in range(beams.size(1)):
            sid_str = '_'.join(str(t.item()) for t in beams[0, ki])
            for cand in sid_to_items.get(sid_str, set()):
                if cand not in seen:
                    candidates.append(cand)
                    seen.add(cand)
        for k in recall_ks:
            if target_cid in set(candidates[:k]):
                item_recall[k] += 1

        # Debug output for first 10 samples
        if verbose and idx < 10:
            n_items_mapped = len(candidates)
            top5_strs = []
            for ki in range(min(5, beams.size(1))):
                s = '_'.join(str(t.item()) for t in beams[0, ki])
                sc = scores[0, ki].item() if scores is not None else 0
                top5_strs.append(f"{s}({sc:.2f})")
            found_str = f"FOUND@{found_rank}" if found_rank >= 0 else "NOT_FOUND"
            print(f"    [sample {idx}] ctx_len={len(item['context'])} "
                  f"target_sid={target_sid_str} target_cid={target_cid} "
                  f"items_mapped={n_items_mapped} {found_str}")
            print(f"      top5: {' | '.join(top5_strs)}")

        # Progress
        if verbose and (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            remaining = (len(eval_items) - idx - 1) / rate
            eta_m, eta_s = divmod(int(remaining), 60)
            r10 = item_recall[10] / (idx + 1)
            r500 = item_recall[500] / (idx + 1)
            sid_found_rate = target_sid_found / (idx + 1)
            d_acc = [depth_correct[li] / (idx + 1) for li in range(n_layers)]
            d_acc_str = '/'.join(f'{a:.3f}' for a in d_acc)
            print(f"    beam {idx+1:,}/{len(eval_items):,} "
                  f"({rate:.1f}/s, ETA {eta_m}m{eta_s:02d}s) "
                  f"R@10={r10:.4f} R@500={r500:.4f} "
                  f"SID_found={sid_found_rate:.4f} depth_acc={d_acc_str}")

    n = len(eval_items)
    target_sid_found_rate = target_sid_found / max(n, 1)
    results = {}
    for k in recall_ks:
        results[f'item_recall@{k}'] = item_recall[k] / n
    results['depth_acc_beam'] = [c / n for c in depth_correct]
    results['target_sid_found_rate'] = target_sid_found_rate
    results['n_recall_samples'] = n

    if verbose:
        print(f"  Beam search summary: target_sid_found={target_sid_found_rate:.4f} "
              f"({target_sid_found}/{n})")

    return results


# ============================================================
# Metric class
# ============================================================

class SemanticIDPredictionMetric(BaseMetric):
    """SID Next Token Prediction — lightweight probe.

    2-layer Transformer, ~5M params. Fast to train and evaluate.
    """

    name = 'semantic_id_prediction'
    requires_model = False
    requires_semantic_ids = True

    thresholds = {'excellent': 50, 'good': 100, 'acceptable': 150}

    def assess_quality(self, value: float) -> str:
        if value <= self.thresholds['excellent']:
            return 'excellent'
        elif value <= self.thresholds['good']:
            return 'good'
        elif value <= self.thresholds['acceptable']:
            return 'acceptable'
        return 'poor'

    def compute(
        self,
        embeddings: torch.Tensor,
        model: Optional[Any] = None,
        semantic_ids: Optional[List[str]] = None,
        layer_assignments: Optional[List[torch.Tensor]] = None,
        behavior_data: Optional[Dict] = None,
        content_id_to_idx: Optional[Dict[str, int]] = None,
        content_ids: Optional[np.ndarray] = None,
        ntp_checkpoint: Optional[str] = None,
        n_items: int = 10,
        batch_size: int = 4096,
        beam_size: int = 5,
        recall_beam_size: int = 50,
        eval_sample_size: int = 50000,
        device: str = 'cuda',
        verbose: bool = True,
        force_autoregressive: bool = False,
        **kwargs
    ) -> MetricResult:
        """Eval-only: load pre-trained probe from checkpoint, run eval.

        Uses unified sequences from preprocessed shards. Teacher-forced metrics
        (PPL, hit@10) computed in one batched forward pass. Beam search recall
        on a small subsample (5K).
        """
        if ntp_checkpoint is None:
            return MetricResult(
                name=self.name, value=0.0,
                details={'error': 'ntp_checkpoint not provided. Run `train-ntp` first.'},
                status='unknown',
            )

        device = device if torch.cuda.is_available() else 'cpu'

        # ── Load checkpoint ──
        if verbose:
            print(f"  Loading checkpoint from {ntp_checkpoint}")

        ckpt = torch.load(
            os.path.join(ntp_checkpoint, 'probe.pt'),
            map_location='cpu', weights_only=False,
        )
        probe_config = ckpt['config']
        model_type = probe_config.pop('model_type', 'probe')

        if model_type == 's-tier':
            probe = NTPModel(**probe_config).to(device)
        else:
            probe = NTPProbe(**probe_config).to(device)
        state_dict = ckpt['model_state_dict']
        # Migrate old checkpoints: decoder → encoder (fix non-causal cross-attn bug)
        if model_type != 's-tier' and any(k.startswith('decoder.') for k in state_dict):
            new_sd = {}
            for k, v in state_dict.items():
                new_k = k.replace('decoder.', 'encoder.', 1) if k.startswith('decoder.') else k
                # Drop cross-attention weights (DecoderLayer has multihead_attn, EncoderLayer doesn't)
                if '.multihead_attn.' in new_k:
                    continue
                new_sd[new_k] = v
            state_dict = new_sd
            if verbose:
                print(f"  Migrated old decoder checkpoint → encoder (dropped cross-attn weights)")
        probe.load_state_dict(state_dict)
        probe.eval()

        n_params = sum(p.numel() for p in probe.parameters())
        n_layers = probe_config['n_sid_layers']
        n_clusters_per_layer = probe_config['n_clusters_per_layer']

        if verbose:
            print(f"  {model_type}: {n_params / 1e6:.1f}M params")

        # ── Load train meta ──
        meta_path = os.path.join(ntp_checkpoint, 'train_meta.json')
        with open(meta_path) as f:
            train_meta = json.load(f)
        sid_cache_dir = train_meta['sid_cache']
        preprocessed_dir = train_meta.get('preprocessed_dir', '')
        n_train = train_meta.get('n_train', 0)

        # ── Load eval sequences from preprocessed shards ──
        if preprocessed_dir and os.path.isdir(preprocessed_dir):
            prep_meta_path = os.path.join(preprocessed_dir, 'meta.json')
            with open(prep_meta_path) as f:
                prep_meta = json.load(f)
            n_shards = prep_meta['n_shards']

            if verbose:
                print(f"  Loading eval data from {preprocessed_dir} ({n_shards} shards)")
            eval_sequences = _load_eval_sequences(preprocessed_dir, n_shards)
        else:
            # Fallback: no preprocessed dir — need to rebuild
            if verbose:
                print(f"  No preprocessed_dir — rebuilding sequences from SID cache")
            sid_dict = np.load(
                os.path.join(sid_cache_dir, 'semantic_ids.npy'), allow_pickle=True
            ).item()
            from gr_demo.eval.batch import load_all_behavior_data
            behavior_data_loaded = load_all_behavior_data()
            from gr_demo.ntp.train import build_unified_sequences
            sequences, _nl, _ncpl, _split_ts = build_unified_sequences(
                sid_dict, behavior_data_loaded, n_items=n_items)
            eval_sequences = [s for s in sequences
                              if s['split_pos'] < len(s['tokens']) and len(s['eval_cids']) > 0]
            del sid_dict, behavior_data_loaded, sequences

        if verbose:
            print(f"  Eval sequences: {len(eval_sequences):,}")

        # ── Subsample eval sequences if needed ──
        if eval_sample_size > 0 and len(eval_sequences) > eval_sample_size:
            random.seed(42)
            eval_sequences = random.sample(eval_sequences, eval_sample_size)
            if verbose:
                print(f"  Subsampled to {len(eval_sequences):,} sequences")

        # ── Teacher-forced eval (batched forward) ──
        if verbose:
            print(f"\n  Teacher-forced eval (batched)...")
        with torch.no_grad():
            tf_results = _batched_teacher_forced_eval(
                probe, eval_sequences, n_layers, device,
                batch_size=batch_size, verbose=verbose)

        ppl = tf_results['ppl']
        depth_h10 = tf_results['depth_hit_10']
        depth_h10_indep = tf_results.get('depth_hit_10_indep', depth_h10)
        layer_ppl = tf_results.get('layer_ppl', [])

        if verbose:
            print(f"  Perplexity: {ppl:.2f}")
            if layer_ppl:
                print(f"  Per-layer PPL: {[f'L{i}={p:.2f}' for i, p in enumerate(layer_ppl)]}")
            print(f"  Depth hit@10 (prefix): {[f'{h:.3f}' for h in depth_h10]}")
            print(f"  Depth hit@10 (indep):  {[f'{h:.3f}' for h in depth_h10_indep]}")
            print(f"  Eval items: {tf_results.get('n_eval_items', 0):,}, "
                  f"positions: {tf_results['n_eval_positions']:,}")

        # ── Beam search recall (small subsample) ──
        if verbose:
            print(f"\n  Building sid_to_items from {sid_cache_dir}")
        sid_to_items = _build_sid_to_items(sid_cache_dir)
        sid_trie = SIDTrie(sid_to_items, n_layers)

        n_recall_samples = min(200, eval_sample_size) if eval_sample_size > 0 else 200

        with torch.no_grad():
            beam_results = _beam_search_recall(
                probe, eval_sequences, sid_trie, sid_to_items, n_layers,
                device, recall_beam_size=recall_beam_size,
                n_recall_samples=n_recall_samples, verbose=verbose)

        # ── Assemble results ──
        details = {
            'depth_acc_beam': beam_results.get('depth_acc_beam', [0] * n_layers),
            'depth_hit@10': depth_h10,
            'n_eval_positions': tf_results['n_eval_positions'],
            'n_eval_sequences': len(eval_sequences),
            'n_recall_samples': beam_results.get('n_recall_samples', 0),
            'n_train': n_train,
            'n_params': n_params,
            'mode': 'autoregressive',
            'ntp_checkpoint': ntp_checkpoint,
        }
        for k, v in beam_results.items():
            if k.startswith('item_recall@'):
                details[k] = v

        if verbose:
            for k in ['item_recall@10', 'item_recall@50', 'item_recall@100', 'item_recall@500']:
                if k in details:
                    print(f"  {k}: {details[k]:.4f}")

        return MetricResult(
            name=self.name,
            value=round(ppl, 4),
            layer_values=beam_results.get('depth_acc_beam', [0] * n_layers),
            details=details,
            status=self.assess_quality(ppl),
        )
