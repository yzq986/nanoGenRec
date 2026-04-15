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
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gr_demo.metrics.base import BaseMetric, MetricResult
from gr_demo.ntp.model import NTPProbe, SIDSequenceDataset


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
        probe = NTPProbe(**probe_config).to(device)
        probe.load_state_dict(ckpt['model_state_dict'])
        probe.eval()

        n_params = sum(p.numel() for p in probe.parameters())
        n_layers = probe_config['n_sid_layers']
        n_clusters_per_layer = probe_config['n_clusters_per_layer']
        use_parallel = probe_config['parallel']

        if verbose:
            mode_str = "parallel (MTP)" if use_parallel else "autoregressive"
            print(f"  NTPProbe: {n_params / 1e6:.1f}M params, {mode_str}")

        # ── Load eval data ──
        eval_ckpt = torch.load(
            os.path.join(ntp_checkpoint, 'eval_data.pt'),
            map_location='cpu', weights_only=False,
        )
        eval_data = eval_ckpt['eval_data']
        eval_cids = eval_ckpt['eval_cids']
        sid_to_items = eval_ckpt['sid_to_items']

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

        with torch.no_grad():
            for input_batch, target_batch in eval_loader:
                input_batch = input_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                B = input_batch.size(0)

                if use_parallel:
                    logits_list = probe(input_batch)  # [(B, C_l), ...]
                else:
                    teacher_input = torch.cat([input_batch, target_batch[:, :-1]], dim=1)
                    logits_list = probe(teacher_input, return_last_n=n_layers)

                # Per-layer CE loss
                loss = sum(
                    F.cross_entropy(logits_list[l], target_batch[:, l])
                    for l in range(n_layers)
                ) / n_layers
                eval_losses.append(loss.item() * B)

                # Per-layer hit@10
                prefix_hit_10 = torch.ones(B, dtype=torch.bool, device=device)
                for i in range(n_layers):
                    top10 = logits_list[i].topk(10, dim=-1).indices
                    hit = (top10 == target_batch[:, i:i+1]).any(dim=-1)
                    prefix_hit_10 = prefix_hit_10 & hit
                    depth_hit_10[i] += prefix_hit_10.sum().item()

                if not use_parallel:

                    # Beam search for recall
                    actual_beam = max(beam_size, recall_beam_size)
                    chunk_size = max(1, 2048 // actual_beam)
                    beam_parts = []
                    for ci in range(0, B, chunk_size):
                        chunk = input_batch[ci:ci + chunk_size]
                        beams = probe.beam_search(chunk, beam_size=actual_beam)
                        beam_parts.append(beams)
                    all_beams = torch.cat(beam_parts, dim=0)

                    pred_top1 = all_beams[:, 0, :]
                    prefix_match = torch.ones(B, dtype=torch.bool, device=device)
                    for i in range(n_layers):
                        prefix_match = prefix_match & (pred_top1[:, i] == target_batch[:, i])
                        depth_correct[i] += prefix_match.sum().item()

                    beams_cpu = all_beams.cpu()
                    for j in range(B):
                        target_cid = eval_cids[eval_offset + j]
                        candidates = []
                        seen = set()
                        for bi in range(beams_cpu.size(1)):
                            sid_str = '_'.join(str(t.item()) for t in beams_cpu[j, bi])
                            for item in sid_to_items.get(sid_str, set()):
                                if item not in seen:
                                    candidates.append(item)
                                    seen.add(item)
                        for k in recall_ks:
                            if target_cid in set(candidates[:k]):
                                item_recall[k] += 1

                eval_offset += B
                total_eval += B

                if verbose and eval_offset % (batch_size * 10) == 0:
                    print(f"    eval {eval_offset:,}/{len(eval_data):,}")

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
