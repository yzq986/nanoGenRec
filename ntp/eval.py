"""
Semantic ID Next Token Prediction — Eval Only.

Loads a pre-trained NTPProbe checkpoint and runs beam search + recall.

Usage:
    1. Train:  torchrun --nproc_per_node=8 run.py train-ntp --sid_cache ...
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
from torch.utils.data import DataLoader

from gr_demo.metrics.base import BaseMetric, MetricResult
from gr_demo.ntp.baseline import NTPProbe, SIDSequenceDataset
from gr_demo.ntp.model import NTPModel, SIDTrie, constrained_beam_search
from gr_demo.ntp.train import EvalSequenceDataset, eval_collate_fn


def _batched_varlen_teacher_forced(probe, input_batch, target_batch, lengths, n_layers, device):
    """Batched teacher-forced eval for variable-length inputs.

    Instead of per-sample forward, constructs a padded teacher batch
    (ctx + tgt[:-1] per sample, right-padded) and does ONE forward pass.
    Then gathers per-sample logits at the correct positions.

    Returns:
        batch_loss: float, sum of per-sample average losses
        depth_hit_10: list of per-layer hit@10 counts for this batch
    """
    B = input_batch.size(0)

    # Build teacher sequences: ctx[:L] + tgt[:-1] for each sample, right-padded
    teacher_lens = lengths + (n_layers - 1)  # (B,)
    max_tlen = teacher_lens.max().item()

    teacher_batch = torch.zeros(B, max_tlen, dtype=torch.long, device=device)
    for bi in range(B):
        L = lengths[bi].item()
        teacher_batch[bi, :L] = input_batch[bi, :L]
        teacher_batch[bi, L:L + n_layers - 1] = target_batch[bi, :-1]

    # Single batched forward through probe internals
    positions = torch.arange(max_tlen, device=device).unsqueeze(0)
    x = probe._embed_tokens(teacher_batch) + probe.pos_emb(positions)

    if hasattr(probe, 'decoder'):
        # NTPProbe: nn.TransformerDecoder (needs explicit causal mask)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(max_tlen, device=device)
        hidden = probe.decoder(x, x, tgt_mask=causal_mask)
    else:
        # NTPModel: TransformerLayers generate their own causal masks
        hidden = probe._transformer_forward(x)

    # Extract per-sample, per-layer logits
    # For sample bi with context length L:
    #   position L-1 predicts target[0] at layer L % n_layers
    #   position L   predicts target[1] at layer (L+1) % n_layers
    #   position L+1 predicts target[2] at layer (L+2) % n_layers
    # Generically: position (L - 1 + li) predicts target[li] at layer (L + li) % n_layers

    batch_loss = 0.0
    depth_hit_10 = [0] * n_layers

    # Gather all positions we need: for each sample, n_layers positions
    # pos[bi, li] = lengths[bi] - 1 + li
    li_range = torch.arange(n_layers, device=device)  # (n_layers,)
    gather_pos = (lengths.unsqueeze(1) - 1 + li_range.unsqueeze(0))  # (B, n_layers)

    for li in range(n_layers):
        pos = gather_pos[:, li]  # (B,)
        target_layer_ids = (lengths + li) % n_layers  # (B,) — which output_proj to use

        # Group by target_layer for efficient projection
        h_at_pos = hidden[torch.arange(B, device=device), pos]  # (B, D)

        layer_losses = torch.zeros(B, device=device)
        layer_logits_topk = torch.zeros(B, 10, dtype=torch.long, device=device)

        for tl in range(n_layers):
            mask = (target_layer_ids == tl)
            if not mask.any():
                continue
            logits = probe.output_projs[tl](h_at_pos[mask])  # (N, C_tl)
            targets_l = target_batch[mask, li]
            layer_losses[mask] = F.cross_entropy(logits, targets_l, reduction='none')
            topk_vals = min(10, logits.size(-1))
            layer_logits_topk[mask, :topk_vals] = logits.topk(topk_vals, dim=-1).indices

        batch_loss += layer_losses.sum().item() / n_layers

        # Depth hit@10 (prefix-based: must hit at all preceding layers too)
        hit = (layer_logits_topk == target_batch[:, li:li + 1]).any(dim=-1)  # (B,)
        if li == 0:
            prefix_hit = hit
        else:
            prefix_hit = prefix_hit & hit
        depth_hit_10[li] = prefix_hit.sum().item()

    return batch_loss, depth_hit_10


def _load_eval_v2(prep_dir, verbose=True):
    """Load eval data from v2 numpy format (fast, avoids pickle on large dicts)."""
    from gr_demo.ntp.preprocess import load_eval_data
    return load_eval_data(prep_dir)


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
        """Eval-only: load pre-trained probe from checkpoint, run beam search + recall.

        Requires ntp_checkpoint pointing to a directory with probe.pt + eval_data.pt
        (produced by `python run.py train-ntp`).
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
        probe.load_state_dict(ckpt['model_state_dict'])
        probe.eval()

        n_params = sum(p.numel() for p in probe.parameters())
        n_layers = probe_config['n_sid_layers']
        n_clusters_per_layer = probe_config['n_clusters_per_layer']
        use_parallel = probe_config['parallel']

        if verbose:
            mode_str = "parallel (MTP)" if use_parallel else "autoregressive"
            print(f"  {model_type}: {n_params / 1e6:.1f}M params, {mode_str}")

        # ── Load eval data ──
        eval_ckpt = torch.load(
            os.path.join(ntp_checkpoint, 'eval_data.pt'),
            map_location='cpu', weights_only=False,
        )
        if eval_ckpt.get('format') == 'v2':
            # v2: load from numpy files (fast)
            prep_dir = eval_ckpt['preprocessed_dir']
            eval_data, eval_cids, sid_to_items = _load_eval_v2(prep_dir, verbose)
        else:
            # v1: inline pickle (legacy checkpoints)
            eval_data = eval_ckpt['eval_data']
            eval_cids = eval_ckpt['eval_cids']
            sid_to_items = eval_ckpt['sid_to_items']

        # Build SID trie for constrained beam search
        sid_trie = SIDTrie(sid_to_items, n_layers)
        if verbose:
            n_sids = sum(len(v) for v in sid_trie.children[0].values())
            print(f"  SID trie: {n_sids:,} root tokens, "
                  f"{len(sid_to_items):,} complete SIDs")

        # Subsample eval if needed
        if eval_sample_size > 0 and len(eval_data) > eval_sample_size:
            random.seed(42)
            indices = random.sample(range(len(eval_data)), eval_sample_size)
            eval_data = [eval_data[i] for i in indices]
            eval_cids = [eval_cids[i] for i in indices]

        if verbose:
            print(f"  Eval samples: {len(eval_data):,}")

        # ── Load train meta for reporting ──
        meta_path = os.path.join(ntp_checkpoint, 'train_meta.json')
        n_train = 0
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                train_meta = json.load(f)
            n_train = train_meta.get('n_train', 0)

        # ── Eval ──
        # Both probe and s-tier now use packed training with variable-length eval data.
        # Detect varlen by checking if input lengths vary (new format) vs fixed (legacy ckpt).
        if len(eval_data) > 0:
            input_lens = set(len(d[0]) for d in eval_data[:100])
            varlen_eval = len(input_lens) > 1
        else:
            varlen_eval = False

        if varlen_eval:
            eval_loader = DataLoader(
                EvalSequenceDataset(eval_data), batch_size=batch_size,
                shuffle=False, num_workers=2, pin_memory=True,
                collate_fn=eval_collate_fn,
            )
            if verbose:
                input_lens = [len(d[0]) for d in eval_data]
                print(f"  Variable-length eval: min={min(input_lens)}, "
                      f"max={max(input_lens)}, avg={np.mean(input_lens):.0f} tokens")
        else:
            eval_loader = DataLoader(
                SIDSequenceDataset(eval_data), batch_size=batch_size,
                shuffle=False, num_workers=2, pin_memory=True,
            )

        eval_losses = []
        depth_hit_10 = [0] * n_layers
        total_eval = 0
        depth_correct = [0] * n_layers
        recall_ks = [10, 50, 100, 500]
        item_recall = {k: 0 for k in recall_ks}
        eval_offset = 0
        eval_t0 = time.time()

        with torch.no_grad():
            for batch in eval_loader:
                if varlen_eval:
                    input_batch, target_batch, lengths = batch
                    lengths = lengths.to(device, non_blocking=True)
                else:
                    input_batch, target_batch = batch
                    lengths = None

                input_batch = input_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                B = input_batch.size(0)

                # ── Teacher-forced metrics (loss + hit@10) ──
                if use_parallel:
                    logits_list = probe(input_batch)  # [(B, C_l), ...]

                    loss = sum(
                        F.cross_entropy(logits_list[l], target_batch[:, l])
                        for l in range(n_layers)
                    ) / n_layers
                    eval_losses.append(loss.item() * B)

                    prefix_hit_10 = torch.ones(B, dtype=torch.bool, device=device)
                    for i in range(n_layers):
                        top10 = logits_list[i].topk(10, dim=-1).indices
                        hit = (top10 == target_batch[:, i:i+1]).any(dim=-1)
                        prefix_hit_10 = prefix_hit_10 & hit
                        depth_hit_10[i] += prefix_hit_10.sum().item()

                elif varlen_eval:
                    # Batched teacher-forced for variable-length inputs
                    batch_loss, batch_dh10 = _batched_varlen_teacher_forced(
                        probe, input_batch, target_batch, lengths,
                        n_layers, device)
                    eval_losses.append(batch_loss)
                    for li in range(n_layers):
                        depth_hit_10[li] += batch_dh10[li]

                else:
                    teacher_input = torch.cat([input_batch, target_batch[:, :-1]], dim=1)
                    logits_list = probe(teacher_input, return_last_n=n_layers)

                    loss = sum(
                        F.cross_entropy(logits_list[l], target_batch[:, l])
                        for l in range(n_layers)
                    ) / n_layers
                    eval_losses.append(loss.item() * B)

                    prefix_hit_10 = torch.ones(B, dtype=torch.bool, device=device)
                    for i in range(n_layers):
                        top10 = logits_list[i].topk(10, dim=-1).indices
                        hit = (top10 == target_batch[:, i:i+1]).any(dim=-1)
                        prefix_hit_10 = prefix_hit_10 & hit
                        depth_hit_10[i] += prefix_hit_10.sum().item()

                # ── Beam search + recall (per-sample, with progress) ──
                if not use_parallel:
                    actual_beam = max(beam_size, recall_beam_size)

                    for bi in range(B):
                        if varlen_eval:
                            L_ctx = lengths[bi].item()
                            sample = input_batch[bi, :L_ctx].unsqueeze(0)
                        else:
                            sample = input_batch[bi:bi + 1]

                        sample_beams, _ = constrained_beam_search(
                            probe, sample, sid_trie, beam_size=actual_beam)

                        # Depth accuracy (top-1 beam)
                        pred_top1 = sample_beams[0, 0]
                        prefix_ok = True
                        for li in range(n_layers):
                            prefix_ok = prefix_ok and (pred_top1[li] == target_batch[bi, li]).item()
                            if prefix_ok:
                                depth_correct[li] += 1
                            else:
                                break

                        # Item recall
                        target_cid = eval_cids[eval_offset + bi]
                        candidates = []
                        seen = set()
                        for ki in range(sample_beams.size(1)):
                            sid_str = '_'.join(str(t.item()) for t in sample_beams[0, ki])
                            for item in sid_to_items.get(sid_str, set()):
                                if item not in seen:
                                    candidates.append(item)
                                    seen.add(item)
                        for k in recall_ks:
                            if target_cid in set(candidates[:k]):
                                item_recall[k] += 1

                        # Progress every 500 samples
                        global_idx = eval_offset + bi + 1
                        if verbose and global_idx % 500 == 0:
                            elapsed = time.time() - eval_t0
                            rate = global_idx / elapsed
                            remaining = (len(eval_data) - global_idx) / rate
                            eta_m, eta_s = divmod(int(remaining), 60)
                            r10 = item_recall[10] / global_idx if global_idx > 0 else 0
                            r500 = item_recall[500] / global_idx if global_idx > 0 else 0
                            print(f"    eval {global_idx:,}/{len(eval_data):,} "
                                  f"({rate:.1f} samples/s, ETA {eta_m}m{eta_s:02d}s) "
                                  f"R@10={r10:.4f} R@500={r500:.4f}")

                eval_offset += B
                total_eval += B

        if total_eval == 0:
            return MetricResult(name=self.name, value=0.0,
                                details={'error': 'No eval samples'}, status='unknown')

        avg_loss = sum(eval_losses) / total_eval
        ppl = np.exp(avg_loss)
        depth_acc = [c / total_eval for c in depth_correct]
        depth_h10 = [h / total_eval for h in depth_hit_10]

        details = {
            'depth_acc_beam': depth_acc,
            'depth_hit@10': depth_h10,
            'n_eval': total_eval,
            'n_train': n_train,
            'n_params': n_params,
            'mode': 'parallel' if use_parallel else 'autoregressive',
            'ntp_checkpoint': ntp_checkpoint,
        }
        for k in recall_ks:
            details[f'item_recall@{k}'] = item_recall[k] / total_eval

        if verbose:
            print(f"  Perplexity: {ppl:.2f}")
            print(f"  Depth acc (beam): {[f'{a:.3f}' for a in depth_acc]}")
            print(f"  Depth hit@10: {[f'{h:.3f}' for h in depth_h10]}")
            for k in recall_ks:
                print(f"  Item recall@{k}: {details[f'item_recall@{k}']:.4f}")

        return MetricResult(
            name=self.name,
            value=round(ppl, 4),
            layer_values=depth_acc,
            details=details,
            status=self.assess_quality(ppl),
        )
