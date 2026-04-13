"""
Semantic ID Next Token Prediction Metric (参考 OneRec)

输入: 用户历史 k 个 item 的所有 tokens (3*k 个 token)
输出: 自回归预测下一个 item 的 3 个 token

生成过程:
  Step 1: [历史 3k tokens] → 预测 next_L1
  Step 2: [历史 3k tokens, next_L1] → 预测 next_L2
  Step 3: [历史 3k tokens, next_L1, next_L2] → 预测 next_L3

推理时使用 Beam Search。

Reference: OneRec (arxiv 2506.13695)
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .base import BaseMetric, MetricResult


# ============================================================
# Dataset
# ============================================================

class ListDataset(Dataset):
    """预构建的样本列表 Dataset"""

    def __init__(self, samples: List[Tuple[List[int], List[int]]], n_layers: int):
        self.samples = samples
        self.n_layers = n_layers

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_tokens, target_tokens = self.samples[idx]
        return (
            torch.tensor(input_tokens, dtype=torch.long),   # (n_items * n_layers,)
            torch.tensor(target_tokens, dtype=torch.long),  # (n_layers,)
        )


# ============================================================
# Mixture of Experts (参考 Mixtral / OneRec-V2)
# ============================================================

class ExpertFFN(nn.Module):
    """Single expert: SwiGLU FFN (same as Mixtral/Llama)"""

    def __init__(self, embed_dim: int, expert_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, expert_dim, bias=False)  # gate
        self.w2 = nn.Linear(expert_dim, embed_dim, bias=False)  # down
        self.w3 = nn.Linear(embed_dim, expert_dim, bias=False)  # up
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class SparseMoEBlock(nn.Module):
    """Sparse Mixture of Experts block with top-k routing.

    Based on Mixtral architecture:
    - Linear router → softmax → top-k selection
    - Dispatch tokens to selected experts
    - Weighted combination of expert outputs
    - Load balancing auxiliary loss

    Reference: Mixtral (arxiv 2401.04088), OneRec-V2 (arxiv 2508.20900)
    """

    def __init__(
        self,
        embed_dim: int,
        expert_dim: int,
        n_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = nn.Linear(embed_dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(embed_dim, expert_dim, dropout) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            output: (batch, seq_len, embed_dim)
            aux_loss: scalar load balancing loss
        """
        orig_shape = x.shape  # (B, S, D)
        x_flat = x.view(-1, orig_shape[-1])  # (B*S, D)
        n_tokens = x_flat.size(0)

        # Router: (B*S, n_experts)
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)  # (B*S, top_k)
        # Renormalize selected weights
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Dispatch and combine
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (B*S,)
            weights = top_k_weights[:, k]  # (B*S,)

            for expert_idx in range(self.n_experts):
                mask = (expert_indices == expert_idx)
                if not mask.any():
                    continue
                expert_input = x_flat[mask]
                expert_output = self.experts[expert_idx](expert_input)
                output[mask] += weights[mask].unsqueeze(-1) * expert_output

        # Load balancing auxiliary loss (Switch Transformer style)
        # f_i = fraction of tokens routed to expert i
        # P_i = mean router probability for expert i
        # loss = n_experts * sum(f_i * P_i)
        expert_mask = F.one_hot(top_k_indices, self.n_experts).float()  # (B*S, top_k, n_experts)
        tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)  # (n_experts,) fraction
        router_prob_per_expert = router_probs.mean(dim=0)  # (n_experts,)
        aux_loss = self.n_experts * (tokens_per_expert * router_prob_per_expert).sum()

        return output.view(orig_shape), aux_loss


# ============================================================
# 自回归 Transformer 模型
# ============================================================

class CausalTransformerLayer(nn.Module):
    """单层 causal self-attention + FFN/MoE，支持 KV cache"""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        use_moe: bool = False,
        n_experts: int = 8,
        top_k: int = 2,
        expert_dim: int = 1024,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)

        if use_moe:
            self.ffn = SparseMoEBlock(embed_dim, expert_dim, n_experts, top_k, dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, embed_dim) — 全序列或仅新 token
            kv_cache: (batch, cached_len, embed_dim) — cached pre-norm KV

        Returns:
            output: (batch, seq_len, embed_dim)
            new_kv_cache: (batch, total_len, embed_dim)
            aux_loss: scalar MoE load balancing loss (0.0 if not using MoE)
        """
        residual = x
        x_norm = self.norm1(x)

        if kv_cache is not None:
            cached_kv = kv_cache  # (batch, cached_len, embed_dim)
            # key/value = cached + current; query = current only
            kv = torch.cat([cached_kv, x_norm], dim=1)
        else:
            kv = x_norm

        total_len = kv.size(1)
        query_len = x_norm.size(1)

        # Causal mask: query positions can only attend to positions <= themselves
        # Shape: (query_len, total_len)
        attn_mask = torch.ones(query_len, total_len, device=x.device, dtype=torch.bool)
        for i in range(query_len):
            # query position i (global pos = total_len - query_len + i) attends to 0..global_pos
            global_pos = total_len - query_len + i
            attn_mask[i, :global_pos + 1] = False
        # True = masked out in PyTorch convention

        attn_out, _ = self.attn(x_norm, kv, kv, attn_mask=attn_mask)
        x = residual + attn_out

        # FFN or MoE
        x_norm2 = self.norm2(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(x_norm2)
            x = x + ffn_out
        else:
            x = x + self.ffn(x_norm2)
            aux_loss = torch.tensor(0.0, device=x.device)

        # Update cache: kv contains all pre-norm tokens seen so far
        return x, kv, aux_loss


class AutoregressiveNTPModel(nn.Module):
    """自回归 Next Token Prediction 模型，支持 MoE + KV cache 加速 beam search。

    输入历史 token 序列，自回归预测下一个 item 的 tokens。

    Model sizes (参考 OneRec-V2):
        S: embed_dim=256, n_transformer_layers=6, n_heads=8, n_experts=8, top_k=2  (~27M, ~7M active)
        M: embed_dim=512, n_transformer_layers=8, n_heads=8, n_experts=16, top_k=2 (~200M, ~30M active)
        L: embed_dim=768, n_transformer_layers=12, n_heads=12, n_experts=32, top_k=2 (~800M, ~60M active)
    """

    def __init__(
        self,
        n_clusters: int = 256,
        n_layers: int = 3,
        n_items: int = 10,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_transformer_layers: int = 6,
        dropout: float = 0.1,
        # MoE config
        use_moe: bool = True,
        n_experts: int = 8,
        top_k: int = 2,
        expert_dim: int = 1024,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_layers = n_layers
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_transformer_layers = n_transformer_layers
        self.use_moe = use_moe
        self.aux_loss_coef = aux_loss_coef
        self.seq_len = n_items * n_layers  # 历史 token 数量

        # Token embedding
        self.token_embedding = nn.Embedding(n_clusters, embed_dim)

        # Position embedding (最大长度 = 历史 + 生成的 tokens)
        self.max_len = self.seq_len + n_layers
        self.pos_embedding = nn.Embedding(self.max_len, embed_dim)

        # Transformer layers (手动管理, 支持 KV cache)
        self.layers = nn.ModuleList([
            CausalTransformerLayer(
                embed_dim, n_heads, dropout,
                use_moe=use_moe, n_experts=n_experts,
                top_k=top_k, expert_dim=expert_dim,
            )
            for _ in range(n_transformer_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(embed_dim, n_clusters)

    def forward(
        self,
        input_tokens: torch.Tensor,
        generated_tokens: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Args:
            input_tokens: (batch, seq_len) 历史 tokens
            generated_tokens: (batch, k) 已生成的 tokens (k < n_layers)
            kv_caches: per-layer KV cache (None = full recompute)

        Returns:
            logits: (batch, n_clusters) 下一个 token 的 logits
            new_kv_caches: updated per-layer KV caches
            total_aux_loss: accumulated MoE load balancing loss (0 if no MoE)
        """
        device = input_tokens.device

        if kv_caches is not None:
            # Incremental: only process the last generated token
            assert generated_tokens is not None and generated_tokens.size(1) > 0
            new_token = generated_tokens[:, -1:]  # (batch, 1)
            curr_pos = kv_caches[0].size(1)  # how many tokens already cached
            token_emb = self.token_embedding(new_token)
            pos_emb = self.pos_embedding(torch.tensor([curr_pos], device=device).unsqueeze(0))
            x = token_emb + pos_emb  # (batch, 1, embed_dim)
        else:
            # Full compute (training or first call)
            if generated_tokens is not None and generated_tokens.size(1) > 0:
                all_tokens = torch.cat([input_tokens, generated_tokens], dim=1)
            else:
                all_tokens = input_tokens

            curr_len = all_tokens.size(1)
            token_emb = self.token_embedding(all_tokens)
            positions = torch.arange(curr_len, device=device).unsqueeze(0)
            pos_emb = self.pos_embedding(positions)
            x = token_emb + pos_emb

        # Transformer layers with KV cache
        new_kv_caches = []
        total_aux_loss = torch.tensor(0.0, device=device)
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache, aux_loss = layer(x, kv_cache=cache_i)
            new_kv_caches.append(new_cache)
            total_aux_loss = total_aux_loss + aux_loss

        # 取最后位置预测下一个 token
        last_hidden = x[:, -1, :]
        logits = self.output_proj(last_hidden)

        return logits, new_kv_caches, total_aux_loss

    def generate_with_beam_search(
        self,
        input_tokens: torch.Tensor,
        beam_size: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched Beam Search — 将 (batch, beam) 展平为大 batch，每 step 只需 1 次 forward

        Args:
            input_tokens: (batch, seq_len) 历史 tokens
            beam_size: beam 大小

        Returns:
            all_beams: (batch, beam_size, n_layers)
            all_scores: (batch, beam_size)
        """
        B = input_tokens.size(0)
        beam_size = min(beam_size, self.n_clusters)

        # Step 1: forward on original batch → (B, n_clusters)
        logits_0, _, _ = self.forward(input_tokens)
        log_probs_0 = F.log_softmax(logits_0, dim=-1)
        top_scores, top_tokens = log_probs_0.topk(beam_size, dim=-1)  # (B, beam)

        beams = top_tokens.unsqueeze(-1)  # (B, beam, 1)
        scores = top_scores  # (B, beam)

        # Step 2+: expand beams, one batched forward per step
        for step in range(1, self.n_layers):
            # (B, seq) → (B*beam, seq)
            input_exp = input_tokens.unsqueeze(1).expand(-1, beam_size, -1).reshape(B * beam_size, -1)
            # (B, beam, step) → (B*beam, step)
            gen_exp = beams.reshape(B * beam_size, -1)

            # Single batched forward
            logits, _, _ = self.forward(input_exp, generated_tokens=gen_exp)
            log_probs = F.log_softmax(logits, dim=-1)  # (B*beam, n_clusters)

            # Top-K per beam → (B*beam, beam)
            top_k_scores, top_k_tokens = log_probs.topk(beam_size, dim=-1)

            # Reshape → (B, beam_old, beam_new)
            top_k_scores = top_k_scores.view(B, beam_size, beam_size)
            top_k_tokens = top_k_tokens.view(B, beam_size, beam_size)

            # Candidate scores: (B, beam, 1) + (B, beam, beam) → (B, beam*beam)
            cand_scores = (scores.unsqueeze(-1) + top_k_scores).view(B, -1)

            # Candidate tokens: cat prev beams + new token → (B, beam*beam, step+1)
            prev = beams.unsqueeze(2).expand(-1, -1, beam_size, -1)
            cands = torch.cat([prev, top_k_tokens.unsqueeze(-1)], dim=-1).view(B, -1, step + 1)

            # Select top beam_size per batch
            scores, top_idx = cand_scores.topk(beam_size, dim=-1)
            beams = torch.gather(cands, 1, top_idx.unsqueeze(-1).expand(-1, -1, step + 1))

        return beams, scores


class BeamSearchModule(nn.Module):
    """Wrapper to make beam search compatible with DataParallel.

    DataParallel only distributes forward() and splits the first tensor dim.
    beam_size is stored as attribute (not a forward arg) to avoid DP trying to split it.
    """

    def __init__(self, model: AutoregressiveNTPModel, beam_size: int = 50):
        super().__init__()
        self.model = model
        self.beam_size = beam_size

    def forward(self, input_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.generate_with_beam_search(input_tokens, beam_size=self.beam_size)


# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(model, dataloader, optimizer, device):
    """训练 (Teacher Forcing)"""
    model.train()
    total_loss = 0
    total_tokens = 0
    aux_loss_coef = getattr(model, 'aux_loss_coef', 0.01)

    for input_tokens, target_tokens in dataloader:
        input_tokens = input_tokens.to(device)
        target_tokens = target_tokens.to(device)
        batch_size = input_tokens.size(0)

        optimizer.zero_grad()
        ce_loss = 0
        total_aux = torch.tensor(0.0, device=device)

        # 自回归: 每个 target token
        for i in range(model.n_layers):
            if i == 0:
                gen_tokens = None
            else:
                gen_tokens = target_tokens[:, :i]  # Teacher forcing

            logits, _, aux_loss = model(input_tokens, generated_tokens=gen_tokens)
            ce_loss = ce_loss + F.cross_entropy(logits, target_tokens[:, i])
            total_aux = total_aux + aux_loss.mean()

        loss = ce_loss + aux_loss_coef * total_aux
        loss.backward()
        optimizer.step()

        total_loss += ce_loss.item() * batch_size
        total_tokens += batch_size * model.n_layers

    return {'loss': total_loss / (total_tokens / model.n_layers)}


def evaluate_model(model, dataloader, device, beam_size: int = 5):
    """评估"""
    model.eval()
    total_loss = 0
    total_samples = 0

    depth_correct = [0] * model.n_layers   # prefix-depth accuracy (beam)
    depth_hit_5 = [0] * model.n_layers     # prefix-depth hit@5 (teacher forcing)
    depth_hit_10 = [0] * model.n_layers    # prefix-depth hit@10 (teacher forcing)

    with torch.no_grad():
        for input_tokens, target_tokens in dataloader:
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)
            batch_size = input_tokens.size(0)

            # 1. Loss (Teacher Forcing) + prefix-depth Hit@K
            loss = 0
            prefix_hit_5 = torch.ones(batch_size, dtype=torch.bool, device=device)
            prefix_hit_10 = torch.ones(batch_size, dtype=torch.bool, device=device)
            for i in range(model.n_layers):
                gen_tokens = target_tokens[:, :i] if i > 0 else None
                logits, _, _ = model(input_tokens, generated_tokens=gen_tokens)
                loss = loss + F.cross_entropy(logits, target_tokens[:, i])

                # Hit@K at this layer (teacher forcing)
                top5 = logits.topk(5, dim=-1).indices
                top10 = logits.topk(10, dim=-1).indices
                hit5_i = (top5 == target_tokens[:, i:i+1]).any(dim=-1)
                hit10_i = (top10 == target_tokens[:, i:i+1]).any(dim=-1)

                # Prefix-depth: all layers up to depth must hit
                prefix_hit_5 = prefix_hit_5 & hit5_i
                prefix_hit_10 = prefix_hit_10 & hit10_i
                depth_hit_5[i] += prefix_hit_5.sum().item()
                depth_hit_10[i] += prefix_hit_10.sum().item()

            total_loss += loss.item() * batch_size

            # 2. Beam Search — prefix-depth accuracy
            all_beams, _ = model.generate_with_beam_search(input_tokens, beam_size=beam_size)
            pred_tokens = all_beams[:, 0, :]  # top-1

            prefix_match = torch.ones(batch_size, dtype=torch.bool, device=device)
            for i in range(model.n_layers):
                prefix_match = prefix_match & (pred_tokens[:, i] == target_tokens[:, i])
                depth_correct[i] += prefix_match.sum().item()

            total_samples += batch_size

    avg_loss = total_loss / total_samples

    return {
        'loss': avg_loss,
        'perplexity': np.exp(avg_loss / model.n_layers),
        'depth_acc_beam': [c / total_samples for c in depth_correct],
        'depth_hit@5': [h / total_samples for h in depth_hit_5],
        'depth_hit@10': [h / total_samples for h in depth_hit_10],
    }


# ============================================================
# Metric Class
# ============================================================

class SemanticIDPredictionMetric(BaseMetric):
    """Semantic ID Next Token Prediction (参考 OneRec)

    输入前 k 个 item 的 tokens，自回归预测下一个 item 的 tokens。
    使用 Beam Search 推理。
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

    @staticmethod
    def _run_eval(
        ntp_model: nn.Module,
        eval_loader: DataLoader,
        n_layers: int,
        beam_size: int,
        device: str,
        sid_to_items: Optional[Dict[str, set]] = None,
        eval_target_cids: Optional[List[str]] = None,
        recall_beam_size: int = 50,
        run_beam: bool = True,
        verbose: bool = True,
        beam_search_module: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Run evaluation on a model.

        Args:
            run_beam: Whether to run beam search (set False for baseline to save time)
            verbose: Print progress
            beam_search_module: DataParallel-wrapped BeamSearchModule for multi-GPU beam search
        """
        ntp_model.eval()
        eval_losses = []
        depth_correct = [0] * n_layers
        depth_hit_5 = [0] * n_layers
        depth_hit_10 = [0] * n_layers
        total_eval = 0

        # Item recall tracking
        compute_recall = run_beam and sid_to_items is not None and eval_target_cids is not None
        recall_ks = [10, 50, 100, 500]
        item_recall = {k: 0 for k in recall_ks}
        actual_beam = max(beam_size, recall_beam_size) if compute_recall else beam_size
        eval_offset = 0
        n_batches = len(eval_loader)

        use_amp = (device != 'cpu' and torch.cuda.is_available())

        with torch.no_grad():
            for batch_idx, (input_batch, target_batch) in enumerate(eval_loader):
                input_batch = input_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                curr_batch_size = input_batch.size(0)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    # Teacher forcing: loss + Hit@K
                    loss = 0
                    prefix_hit_5 = torch.ones(curr_batch_size, dtype=torch.bool, device=device)
                    prefix_hit_10 = torch.ones(curr_batch_size, dtype=torch.bool, device=device)
                    for i in range(n_layers):
                        gen_tokens = target_batch[:, :i] if i > 0 else None
                        logits, _, _ = ntp_model(input_batch, generated_tokens=gen_tokens)
                        loss = loss + F.cross_entropy(logits, target_batch[:, i])

                        top5 = logits.topk(5, dim=-1).indices
                        top10 = logits.topk(10, dim=-1).indices
                        hit5_i = (top5 == target_batch[:, i:i+1]).any(dim=-1)
                        hit10_i = (top10 == target_batch[:, i:i+1]).any(dim=-1)

                        prefix_hit_5 = prefix_hit_5 & hit5_i
                        prefix_hit_10 = prefix_hit_10 & hit10_i
                        depth_hit_5[i] += prefix_hit_5.sum().item()
                        depth_hit_10[i] += prefix_hit_10.sum().item()

                eval_losses.append(loss.item() * curr_batch_size)

                if run_beam:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                        if beam_search_module is not None:
                            # DataParallel splits batch across GPUs automatically
                            # Update beam_size on the underlying module
                            bsm = beam_search_module.module if hasattr(beam_search_module, 'module') else beam_search_module
                            bsm.beam_size = actual_beam
                            all_beams, _ = beam_search_module(input_batch)
                        else:
                            # Single-GPU fallback with chunking
                            beam_chunk = max(1, 2048 // actual_beam)
                            beam_parts = []
                            for ci in range(0, curr_batch_size, beam_chunk):
                                chunk = input_batch[ci:ci + beam_chunk]
                                beams_chunk, _ = ntp_model.generate_with_beam_search(
                                    chunk, beam_size=actual_beam
                                )
                                beam_parts.append(beams_chunk)
                            all_beams = torch.cat(beam_parts, dim=0)
                    pred_tokens = all_beams[:, 0, :]

                    # SID prefix-depth accuracy (top-1 beam)
                    prefix_match = torch.ones(curr_batch_size, dtype=torch.bool, device=device)
                    for i in range(n_layers):
                        prefix_match = prefix_match & (pred_tokens[:, i] == target_batch[:, i])
                        depth_correct[i] += prefix_match.sum().item()

                    # Item-level recall
                    if compute_recall:
                        # Move to CPU once for recall computation
                        beams_cpu = all_beams.cpu()
                        for sample_idx in range(curr_batch_size):
                            target_cid = eval_target_cids[eval_offset + sample_idx]
                            candidate_items = []
                            seen = set()
                            for beam_idx in range(beams_cpu.size(1)):
                                sid_str = '_'.join(
                                    str(t.item()) for t in beams_cpu[sample_idx, beam_idx]
                                )
                                for item in sid_to_items.get(sid_str, set()):
                                    if item not in seen:
                                        candidate_items.append(item)
                                        seen.add(item)
                            for k in recall_ks:
                                if target_cid in set(candidate_items[:k]):
                                    item_recall[k] += 1

                eval_offset += curr_batch_size
                total_eval += curr_batch_size

                # Progress
                if verbose and (batch_idx + 1) % 20 == 0:
                    pct = (batch_idx + 1) / n_batches * 100
                    print(f"    Eval: {pct:.0f}% ({total_eval:,}/{n_batches * curr_batch_size:,})")

        if total_eval == 0:
            return {'loss': 0, 'perplexity': 0, 'depth_acc_beam': [0]*n_layers,
                    'depth_hit@5': [0]*n_layers, 'depth_hit@10': [0]*n_layers, 'n_eval': 0}

        avg_loss = sum(eval_losses) / total_eval
        result = {
            'loss': avg_loss,
            'perplexity': np.exp(avg_loss / n_layers),
            'depth_acc_beam': [c / total_eval for c in depth_correct],
            'depth_hit@5': [h / total_eval for h in depth_hit_5],
            'depth_hit@10': [h / total_eval for h in depth_hit_10],
            'n_eval': total_eval,
        }

        if compute_recall:
            for k in recall_ks:
                result[f'item_recall@{k}'] = item_recall[k] / total_eval

        return result

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

        # Build content_id -> tokens
        content_to_tokens = {}
        idx_to_cid = {v: k for k, v in content_id_to_idx.items()}
        for idx, sid in enumerate(semantic_ids):
            if idx in idx_to_cid:
                tokens = [int(t) for t in sid.split('_')]
                content_to_tokens[idx_to_cid[idx]] = tokens

        n_layers = len(list(content_to_tokens.values())[0])
        n_clusters = max(max(t) for t in content_to_tokens.values()) + 1

        # Build user sequences (按时间排序)
        uids = behavior_data['uid']
        iids = behavior_data['iid']
        actions = behavior_data['action_bitmap']
        timestamps = behavior_data.get('first_ts')  # 可选，兼容旧数据

        # Build SID string -> content_ids reverse mapping (for item recall)
        sid_to_items = defaultdict(set)
        for cid, tokens in content_to_tokens.items():
            sid_str = '_'.join(str(t) for t in tokens)
            sid_to_items[sid_str].add(cid)

        # 收集每个用户的 (timestamp, tokens, content_id)
        user_items = defaultdict(list)
        for i in range(len(uids)):
            uid, iid, action = uids[i], iids[i], actions[i]
            if action > 0 and iid in content_to_tokens:
                ts = timestamps[i] if timestamps is not None else i
                user_items[uid].append((ts, content_to_tokens[iid], iid))

        # 按时间排序，生成样本 (uid, input_tokens, target_tokens, target_ts, target_cid)
        all_samples_with_ts = []
        valid_user_count = 0

        for uid, items in user_items.items():
            if len(items) < n_items + 1:
                continue
            valid_user_count += 1
            items.sort(key=lambda x: x[0])  # 按时间排序

            for i in range(len(items) - n_items):
                input_tokens = []
                for j in range(n_items):
                    input_tokens.extend(items[i + j][1])
                target_tokens = items[i + n_items][1]
                target_ts = items[i + n_items][0]
                target_cid = items[i + n_items][2]
                all_samples_with_ts.append((uid, input_tokens, target_tokens, target_ts, target_cid))

        if not all_samples_with_ts:
            return MetricResult(name=self.name, value=0.0, details={'error': 'No valid sequences'}, status='unknown')

        # 按时间全局排序
        all_samples_with_ts.sort(key=lambda x: x[3])

        # 按时间 split: 前 80% train, 后 20% eval
        n_total = len(all_samples_with_ts)
        split_idx = int(n_total * 0.8)

        train_samples_with_uid = all_samples_with_ts[:split_idx]
        eval_all = all_samples_with_ts[split_idx:]

        # Sample eval set if too large (beam search is the bottleneck)
        import random
        if eval_sample_size > 0 and len(eval_all) > eval_sample_size:
            random.seed(42)
            eval_all = random.sample(eval_all, eval_sample_size)

        eval_samples = [(s[1], s[2]) for s in eval_all]
        eval_target_cids = [s[4] for s in eval_all]

        if verbose:
            n_eval_full = n_total - split_idx
            sampled_str = f" (sampled from {n_eval_full})" if len(eval_samples) < n_eval_full else ""
            print(f"  Users: {valid_user_count}, Total samples: {n_total}")
            print(f"  Train: {len(train_samples_with_uid)}, Eval: {len(eval_samples)}{sampled_str} (by time split)")
            print(f"  n_layers: {n_layers}, n_clusters: {n_clusters}")

        # Model (S-tier MoE: ~27M params, ~7M active)
        n_gpus = torch.cuda.device_count() if device == 'cuda' else 0
        ntp_model = AutoregressiveNTPModel(
            n_clusters=n_clusters,
            n_layers=n_layers,
            n_items=n_items,
            embed_dim=256,
            n_heads=8,
            n_transformer_layers=6,
            use_moe=True,
            n_experts=8,
            top_k=2,
            expert_dim=1024,
            aux_loss_coef=0.01,
        ).to(device)
        ntp_model_aux_coef = ntp_model.aux_loss_coef

        # Multi-GPU: DataParallel
        raw_model = ntp_model  # keep reference for eval (DP wrapper doesn't expose custom methods)
        beam_module = None
        if n_gpus > 1:
            ntp_model = nn.DataParallel(ntp_model)
            beam_module = nn.DataParallel(BeamSearchModule(raw_model))
            if verbose:
                print(f"  Using {n_gpus} GPUs (DataParallel, beam search included)")

        optimizer = torch.optim.AdamW(ntp_model.parameters(), lr=1e-3, weight_decay=1e-5)
        scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

        total_params = sum(p.numel() for p in ntp_model.parameters())
        if verbose:
            arch_parts = [f"d={raw_model.embed_dim}", f"L={raw_model.n_transformer_layers}"]
            if raw_model.use_moe:
                arch_parts.append("MoE")
            print(f"  Model: {total_params:,} params ({', '.join(arch_parts)})")
            print(f"  Batch size: {batch_size}, AMP: BF16, Device: {device}")

        # DataLoader kwargs
        dl_kwargs = {'pin_memory': True, 'num_workers': 4} if device == 'cuda' else {}

        # ============================================================
        # Phase 0: Baseline eval (untrained model)
        # ============================================================
        eval_dataset = ListDataset(eval_samples, n_layers)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, **dl_kwargs)

        if verbose:
            print(f"  [Baseline] evaluating (untrained, recall_beam={recall_beam_size})...")
        eval_t0 = time.time()
        baseline_metrics = self._run_eval(
            raw_model, eval_loader, n_layers, beam_size, device,
            sid_to_items=sid_to_items, eval_target_cids=eval_target_cids,
            recall_beam_size=recall_beam_size,
            run_beam=True, verbose=verbose,
            beam_search_module=beam_module,
        )
        baseline_elapsed = time.time() - eval_t0

        if verbose:
            print(f"  [Baseline] done in {baseline_elapsed:.1f}s")
            print(f"  [Baseline] Perplexity: {baseline_metrics['perplexity']:.2f} (random: {n_clusters})")
            print(f"    Depth Acc (beam): {[f'{a:.4f}' for a in baseline_metrics['depth_acc_beam']]}")
            print(f"    Depth Hit@10: {[f'{h:.4f}' for h in baseline_metrics['depth_hit@10']]}")
            for rk_key in sorted(k for k in baseline_metrics if k.startswith('item_recall@')):
                print(f"    {rk_key}: {baseline_metrics[rk_key]:.4f}")

        # ============================================================
        # Phase 1: Train on train_samples (uid-level shuffle)
        # ============================================================
        # 按 uid 分组，shuffle 用户顺序，用户内保持时间顺序
        random.seed(42)

        uid_to_samples = defaultdict(list)
        for uid, inp, tgt, ts, _cid in train_samples_with_uid:
            uid_to_samples[uid].append((inp, tgt))

        # Shuffle 用户顺序
        uid_list = list(uid_to_samples.keys())
        random.shuffle(uid_list)

        # 按用户顺序展开 (用户内保持时间顺序)
        train_samples = []
        for uid in uid_list:
            train_samples.extend(uid_to_samples[uid])

        train_dataset = ListDataset(train_samples, n_layers)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, **dl_kwargs)

        if verbose:
            print(f"  Training ({len(train_samples):,} samples, {len(train_loader)} batches)...")

        ntp_model.train()
        n_train_batches = len(train_loader)
        train_t0 = time.time()
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # AMP BF16 forward
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                ce_loss = 0
                total_aux = torch.tensor(0.0, device=device)
                for i in range(n_layers):
                    gen_tokens = target_batch[:, :i] if i > 0 else None
                    logits, _, aux_loss = ntp_model(input_batch, generated_tokens=gen_tokens)
                    ce_loss = ce_loss + F.cross_entropy(logits, target_batch[:, i])
                    # DataParallel gathers per-GPU aux_loss scalars into a vector
                    total_aux = total_aux + aux_loss.mean()
                loss = ce_loss + ntp_model_aux_coef * total_aux

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Progress with timing
            if verbose and (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - train_t0
                progress = (batch_idx + 1) / n_train_batches
                eta = elapsed / progress * (1 - progress)
                aux_str = f" | Aux: {total_aux.item():.4f}" if ntp_model_aux_coef > 0 else ""
                print(f"    Train: {progress*100:.1f}% | CE: {ce_loss.item():.4f}{aux_str} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

        train_elapsed = time.time() - train_t0
        if verbose:
            print(f"  Training done in {train_elapsed:.1f}s ({len(train_samples)/train_elapsed:.0f} samples/s)")

        # ============================================================
        # Phase 2: Eval on eval_samples (after training)
        # ============================================================
        if verbose:
            print(f"  Evaluating on {len(eval_samples)} samples (beam={beam_size}, recall_beam={recall_beam_size})...")

        eval_t0 = time.time()
        trained_metrics = self._run_eval(
            raw_model, eval_loader, n_layers, beam_size, device,
            sid_to_items=sid_to_items, eval_target_cids=eval_target_cids,
            recall_beam_size=recall_beam_size,
            run_beam=True, verbose=verbose,
            beam_search_module=beam_module,
        )
        trained_eval_elapsed = time.time() - eval_t0

        if verbose:
            print(f"  Eval done in {trained_eval_elapsed:.1f}s")

        ppl = trained_metrics['perplexity']
        depth_acc = trained_metrics['depth_acc_beam']
        hit5 = trained_metrics['depth_hit@5']
        hit10 = trained_metrics['depth_hit@10']

        if verbose:
            print(f"  [Trained] Results:")
            print(f"    Perplexity: {ppl:.2f} (baseline: {baseline_metrics['perplexity']:.2f}, random: {n_clusters})")
            print(f"    Depth Acc (beam): {[f'{a:.4f}' for a in depth_acc]}")
            print(f"    Depth Hit@10: {[f'{h:.4f}' for h in hit10]}")
            for rk_key in sorted(k for k in trained_metrics if k.startswith('item_recall@')):
                rk = trained_metrics[rk_key]
                bk = baseline_metrics.get(rk_key)
                bk_str = f", baseline: {bk:.4f}" if bk is not None else ""
                print(f"    {rk_key}: {rk:.4f}{bk_str}")

        # Collision stats for context
        sid_counts = [len(items) for items in sid_to_items.values()]
        avg_collision = np.mean(sid_counts) if sid_counts else 0

        details = {
            'perplexity': ppl,
            'random_perplexity': n_clusters,
            'ntp_loss': trained_metrics['loss'],
            'depth_acc_beam': depth_acc,
            'depth_hit@5': hit5,
            'depth_hit@10': hit10,
            'baseline_perplexity': baseline_metrics['perplexity'],
            'baseline_depth_acc_beam': baseline_metrics['depth_acc_beam'],
            'baseline_depth_hit@5': baseline_metrics['depth_hit@5'],
            'baseline_depth_hit@10': baseline_metrics['depth_hit@10'],
            'beam_size': beam_size,
            'recall_beam_size': recall_beam_size,
            'n_items': n_items,
            'n_layers': n_layers,
            'n_clusters': n_clusters,
            'n_train': len(train_samples),
            'n_eval': len(eval_samples),
            'n_unique_sids': len(sid_to_items),
            'avg_items_per_sid': round(avg_collision, 2),
            # Model architecture
            'model_params': total_params,
            'model_embed_dim': 256,
            'model_n_transformer_layers': 6,
            'model_n_heads': 8,
            'model_use_moe': True,
            'model_n_experts': 8,
            'model_top_k': 2,
            'model_expert_dim': 1024,
            # Timing
            'baseline_eval_time_s': round(baseline_elapsed, 1),
            'train_time_s': round(train_elapsed, 1),
            'train_samples_per_s': round(len(train_samples) / train_elapsed),
            'trained_eval_time_s': round(trained_eval_elapsed, 1),
            'total_time_s': round(baseline_elapsed + train_elapsed + trained_eval_elapsed, 1),
        }

        # Item recall results (dynamic keys from _run_eval)
        for key in trained_metrics:
            if key.startswith('item_recall@'):
                details[key] = trained_metrics[key]
        for key in baseline_metrics:
            if key.startswith('item_recall@'):
                details[f'baseline_{key}'] = baseline_metrics[key]

        return MetricResult(
            name=self.name,
            value=ppl,
            layer_values=depth_acc,
            details=details,
            status=self.assess_quality(ppl),
        )
