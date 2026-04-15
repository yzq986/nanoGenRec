"""
Semantic ID Next Token Prediction — Lightweight Probe (RPG-style)

2-layer Transformer decoder, ~5M params. Designed as a fast probe to
evaluate SID quality, NOT as a production model. Following RPG (Meta,
KDD'25): the simplest model that saturates on good SIDs.

Design principle: Embedding 质量决定 SID 上限，NTP 模型只是 probe。
复杂模型见 sid_prediction_old.py (archived).
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .base import BaseMetric, MetricResult


# ============================================================
# Dataset
# ============================================================

class SIDSequenceDataset(Dataset):
    """(input_tokens, target_tokens) pairs from user behavior sequences."""

    def __init__(self, samples: List[Tuple[list, list]]):
        self.inputs = [torch.tensor(s[0], dtype=torch.long) for s in samples]
        self.targets = [torch.tensor(s[1], dtype=torch.long) for s in samples]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ============================================================
# Model: 2-layer Transformer decoder (RPG-style probe)
# ============================================================

class NTPProbe(nn.Module):
    """Lightweight next-token prediction probe.

    2-layer causal Transformer decoder. No MoE, no KV cache.
    ~5M params with default config. Trains in minutes.

    For autoregressive SIDs (RKMeans/FSQ, n_layers <= 4):
        Teacher-forced training, beam search eval.
    For parallel SIDs (OPQ, n_layers >= 5):
        Independent MLP heads per token position (RPG-style MTP).
    """

    def __init__(
        self,
        n_clusters: int,
        n_sid_layers: int,
        n_items: int = 10,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        parallel: bool = False,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_sid_layers = n_sid_layers
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.parallel = parallel
        self.seq_len = n_items * n_sid_layers

        # Token + position embeddings
        self.token_emb = nn.Embedding(n_clusters, embed_dim)
        max_len = self.seq_len + n_sid_layers
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        # Transformer layers
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_transformer_layers)

        if parallel:
            # RPG-style: independent MLP head per SID token position
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, ffn_dim), nn.GELU(),
                    nn.Linear(ffn_dim, n_clusters),
                )
                for _ in range(n_sid_layers)
            ])
        else:
            # Shared output projection (autoregressive)
            self.output_proj = nn.Linear(embed_dim, n_clusters)

    def forward(self, input_tokens: torch.Tensor,
                generated_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_tokens: (B, seq_len) history SID tokens
            generated_tokens: (B, k) already-generated target tokens (AR mode only)
        Returns:
            logits: (B, n_clusters) for AR mode, or (B, n_sid_layers, n_clusters) for parallel
        """
        B = input_tokens.size(0)
        device = input_tokens.device

        if self.parallel:
            # Encode history, then independent prediction per position
            positions = torch.arange(self.seq_len, device=device).unsqueeze(0)
            x = self.token_emb(input_tokens) + self.pos_emb(positions)

            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                self.seq_len, device=device
            )
            memory = self.decoder(x, x, tgt_mask=causal_mask)

            # Pool last position as sequence representation
            s = memory[:, -1, :]  # (B, D)

            # Independent head per SID token
            all_logits = torch.stack([head(s) for head in self.heads], dim=1)  # (B, L, C)
            return all_logits
        else:
            # Autoregressive: concatenate history + generated tokens
            if generated_tokens is not None and generated_tokens.size(1) > 0:
                tokens = torch.cat([input_tokens, generated_tokens], dim=1)
            else:
                tokens = input_tokens

            T = tokens.size(1)
            positions = torch.arange(T, device=device).unsqueeze(0)
            x = self.token_emb(tokens) + self.pos_emb(positions)

            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
            out = self.decoder(x, x, tgt_mask=causal_mask)

            logits = self.output_proj(out[:, -1, :])  # (B, n_clusters)
            return logits

    @torch.no_grad()
    def beam_search(self, input_tokens: torch.Tensor, beam_size: int = 5) -> torch.Tensor:
        """Simple beam search for autoregressive mode.

        Returns:
            beams: (B, beam_size, n_sid_layers) top beam results
        """
        B = input_tokens.size(0)
        device = input_tokens.device
        L = self.n_sid_layers

        # Start: (B, beam_size, 0) empty generated sequences
        # Score: (B, beam_size)
        beams = torch.zeros(B, 1, 0, dtype=torch.long, device=device)
        scores = torch.zeros(B, 1, device=device)

        for step in range(L):
            n_beams = beams.size(1)

            # Flatten (B, n_beams) -> (B*n_beams,)
            input_exp = input_tokens.unsqueeze(1).expand(-1, n_beams, -1).reshape(B * n_beams, -1)
            gen_exp = beams.reshape(B * n_beams, -1) if step > 0 else None

            logits = self.forward(input_exp, gen_exp)  # (B*n_beams, C)
            log_probs = F.log_softmax(logits, dim=-1)  # (B*n_beams, C)
            C = log_probs.size(-1)

            log_probs = log_probs.view(B, n_beams, C)
            candidate_scores = scores.unsqueeze(-1) + log_probs  # (B, n_beams, C)

            # Select top-k from (n_beams * C) candidates
            flat_scores = candidate_scores.view(B, -1)  # (B, n_beams*C)
            topk_scores, topk_idx = flat_scores.topk(beam_size, dim=-1)  # (B, beam_size)

            beam_idx = topk_idx // C
            token_idx = topk_idx % C

            # Gather and extend beams
            prev_beams = torch.gather(
                beams, 1, beam_idx.unsqueeze(-1).expand(-1, -1, step)
            ) if step > 0 else torch.zeros(B, beam_size, 0, dtype=torch.long, device=device)

            new_token = token_idx.unsqueeze(-1)  # (B, beam_size, 1)
            beams = torch.cat([prev_beams, new_token], dim=-1)  # (B, beam_size, step+1)
            scores = topk_scores

        return beams  # (B, beam_size, L)


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
        self.validate_inputs(embeddings, model, semantic_ids)

        if behavior_data is None or content_id_to_idx is None:
            return MetricResult(
                name=self.name, value=0.0,
                details={'error': 'behavior_data or content_id_to_idx not provided'},
                status='unknown',
            )

        device = device if torch.cuda.is_available() else 'cpu'

        # ── Build content_id -> tokens ──
        content_to_tokens = {}
        idx_to_cid = {v: k for k, v in content_id_to_idx.items()}
        for idx, sid in enumerate(semantic_ids):
            if idx in idx_to_cid:
                content_to_tokens[idx_to_cid[idx]] = [int(t) for t in sid.split('_')]

        n_layers = len(next(iter(content_to_tokens.values())))
        n_clusters = max(max(t) for t in content_to_tokens.values()) + 1

        # ── Build user sequences ──
        uids = behavior_data['uid']
        iids = behavior_data['iid']
        actions = behavior_data['action_bitmap']
        timestamps = behavior_data.get('first_ts')

        sid_to_items = defaultdict(set)
        for cid, tokens in content_to_tokens.items():
            sid_to_items['_'.join(str(t) for t in tokens)].add(cid)

        user_items = defaultdict(list)
        for i in range(len(uids)):
            uid, iid, action = uids[i], iids[i], actions[i]
            if action > 0 and iid in content_to_tokens:
                ts = timestamps[i] if timestamps is not None else i
                user_items[uid].append((ts, content_to_tokens[iid], iid))

        # Generate samples, time-sorted, 80/20 split
        all_samples = []
        for uid, items in user_items.items():
            if len(items) < n_items + 1:
                continue
            items.sort(key=lambda x: x[0])
            for i in range(len(items) - n_items):
                input_tokens = []
                for j in range(n_items):
                    input_tokens.extend(items[i + j][1])
                target_tokens = items[i + n_items][1]
                target_cid = items[i + n_items][2]
                target_ts = items[i + n_items][0]
                all_samples.append((input_tokens, target_tokens, target_cid, target_ts))

        if not all_samples:
            return MetricResult(name=self.name, value=0.0,
                                details={'error': 'No valid sequences'}, status='unknown')

        all_samples.sort(key=lambda x: x[3])
        split_idx = int(len(all_samples) * 0.8)
        train_data = [(s[0], s[1]) for s in all_samples[:split_idx]]
        eval_data = [(s[0], s[1]) for s in all_samples[split_idx:]]
        eval_cids = [s[2] for s in all_samples[split_idx:]]

        import random
        if eval_sample_size > 0 and len(eval_data) > eval_sample_size:
            random.seed(42)
            indices = random.sample(range(len(eval_data)), eval_sample_size)
            eval_data = [eval_data[i] for i in indices]
            eval_cids = [eval_cids[i] for i in indices]

        if verbose:
            print(f"  Samples: {len(all_samples):,} (train={len(train_data):,}, eval={len(eval_data):,})")
            print(f"  SID: {n_layers} layers x {n_clusters} clusters")

        use_parallel = n_layers >= 5 and not force_autoregressive

        # ── Build model ──
        probe = NTPProbe(
            n_clusters=n_clusters, n_sid_layers=n_layers, n_items=n_items,
            embed_dim=256, n_heads=4, n_transformer_layers=2, ffn_dim=512,
            parallel=use_parallel,
        ).to(device)

        n_params = sum(p.numel() for p in probe.parameters())
        if verbose:
            mode_str = "parallel (MTP)" if use_parallel else "autoregressive"
            print(f"  NTPProbe: {n_params / 1e6:.1f}M params, {mode_str}")

        # ── Train ──
        # Cap training samples to avoid extremely long training
        import random as _rand
        max_train_samples = 2_000_000
        if len(train_data) > max_train_samples:
            if verbose:
                print(f"  Capping training data: {len(train_data):,} → {max_train_samples:,}")
            _rand.seed(42)
            train_data = _rand.sample(train_data, max_train_samples)

        train_loader = DataLoader(
            SIDSequenceDataset(train_data), batch_size=batch_size,
            shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
        )

        optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * 3)
        n_batches = len(train_loader)

        probe.train()
        for epoch in range(3):
            total_loss = 0
            for step, (input_batch, target_batch) in enumerate(train_loader):
                input_batch = input_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)

                if use_parallel:
                    logits = probe(input_batch)  # (B, L, C)
                    loss = F.cross_entropy(
                        logits.reshape(-1, n_clusters), target_batch.reshape(-1)
                    )
                else:
                    loss = 0.0
                    for i in range(n_layers):
                        gen = target_batch[:, :i] if i > 0 else None
                        logits = probe(input_batch, gen)
                        loss = loss + F.cross_entropy(logits, target_batch[:, i])
                    loss = loss / n_layers

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

                if verbose and (step + 1) % 100 == 0:
                    print(f"    Epoch {epoch+1}/3 step {step+1}/{n_batches}: loss={total_loss/(step+1):.4f}")

            avg = total_loss / n_batches
            if verbose:
                print(f"  Epoch {epoch+1}/3: loss={avg:.4f}")

        # ── Eval ──
        probe.eval()
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
                    logits = probe(input_batch)  # (B, L, C)
                    loss = F.cross_entropy(
                        logits.reshape(-1, n_clusters), target_batch.reshape(-1)
                    )
                    eval_losses.append(loss.item() * B)

                    # Per-layer hit@10
                    for i in range(n_layers):
                        top10 = logits[:, i, :].topk(10, dim=-1).indices
                        hit = (top10 == target_batch[:, i:i+1]).any(dim=-1)
                        depth_hit_10[i] += hit.sum().item()

                    # Item recall: take argmax per position as predicted SID
                    pred_sids = logits.argmax(dim=-1)  # (B, L)
                    prefix_match = torch.ones(B, dtype=torch.bool, device=device)
                    for i in range(n_layers):
                        prefix_match = prefix_match & (pred_sids[:, i] == target_batch[:, i])
                        depth_correct[i] += prefix_match.sum().item()

                    for j in range(B):
                        sid_str = '_'.join(str(t.item()) for t in pred_sids[j])
                        target_cid = eval_cids[eval_offset + j]
                        candidates = list(sid_to_items.get(sid_str, set()))
                        for k in recall_ks:
                            if target_cid in set(candidates[:k]):
                                item_recall[k] += 1
                else:
                    # AR eval: teacher-forced loss + hit@10
                    loss = 0.0
                    prefix_hit_10 = torch.ones(B, dtype=torch.bool, device=device)
                    for i in range(n_layers):
                        gen = target_batch[:, :i] if i > 0 else None
                        logits = probe(input_batch, gen)
                        loss = loss + F.cross_entropy(logits, target_batch[:, i])
                        top10 = logits.topk(10, dim=-1).indices
                        hit = (top10 == target_batch[:, i:i+1]).any(dim=-1)
                        prefix_hit_10 = prefix_hit_10 & hit
                        depth_hit_10[i] += prefix_hit_10.sum().item()
                    eval_losses.append(loss.item() * B)

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

        if total_eval == 0:
            return MetricResult(name=self.name, value=0.0,
                                details={'error': 'No eval samples'}, status='unknown')

        avg_loss = sum(eval_losses) / total_eval
        ppl = np.exp(avg_loss / n_layers) if not use_parallel else np.exp(avg_loss)
        depth_acc = [c / total_eval for c in depth_correct]
        depth_h10 = [h / total_eval for h in depth_hit_10]

        details = {
            'depth_acc_beam': depth_acc,
            'depth_hit@10': depth_h10,
            'n_eval': total_eval,
            'n_train': len(train_data),
            'n_params': n_params,
            'mode': 'parallel' if use_parallel else 'autoregressive',
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
