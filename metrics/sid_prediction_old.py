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
    """单层 self-attention + FFN/MoE，支持 causal/bidirectional + KV cache"""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        use_moe: bool = False,
        n_experts: int = 8,
        top_k: int = 2,
        expert_dim: int = 1024,
        causal: bool = True,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.causal = causal
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

        if self.causal:
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
        else:
            attn_mask = None  # Bidirectional attention

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


class ParallelNTPModel(nn.Module):
    """并行 Next Token Prediction 模型（用于 OPQ parallel SIDs）。

    与 AutoregressiveNTPModel 不同：
    - 双向 attention（causal=False）
    - 8 个独立 MLP head 并行预测 m 组 logits
    - 无 KV cache / beam search — 一次 forward 出所有 logits
    """

    def __init__(
        self,
        n_clusters: int = 256,
        n_layers: int = 8,
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

        # Position embedding (历史 tokens only, no generation)
        self.pos_embedding = nn.Embedding(self.seq_len, embed_dim)

        # Transformer layers (bidirectional, causal=False)
        self.layers = nn.ModuleList([
            CausalTransformerLayer(
                embed_dim, n_heads, dropout,
                use_moe=use_moe, n_experts=n_experts,
                top_k=top_k, expert_dim=expert_dim,
                causal=False,
            )
            for _ in range(n_transformer_layers)
        ])

        # m independent MLP prediction heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.GELU(),
                nn.Linear(512, n_clusters),
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        input_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_tokens: (batch, seq_len) — 历史 n_items * n_layers tokens

        Returns:
            logits: (batch, n_layers, n_clusters) — m 组独立 logits
            aux_loss: scalar MoE load balancing loss
        """
        device = input_tokens.device
        curr_len = input_tokens.size(1)

        token_emb = self.token_embedding(input_tokens)
        positions = torch.arange(curr_len, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb  # (batch, seq_len, embed_dim)

        # Bidirectional transformer layers
        total_aux_loss = torch.tensor(0.0, device=device)
        for layer in self.layers:
            x, _, aux_loss = layer(x, kv_cache=None)
            total_aux_loss = total_aux_loss + aux_loss

        # Take last position hidden state → (batch, embed_dim)
        last_hidden = x[:, -1, :]

        # m independent heads → (batch, n_layers, n_clusters)
        logits = torch.stack([head(last_hidden) for head in self.heads], dim=1)

        return logits, total_aux_loss


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
# Graph-Constrained Decoding (for parallel OPQ predictions)
# ============================================================

def _score_sids(
    log_probs: torch.Tensor,
    valid_sids: torch.Tensor,
    sid_indices: torch.Tensor,
) -> torch.Tensor:
    """Score a subset of SIDs for each batch element.

    Args:
        log_probs: (batch, m, M) per-digit log probabilities
        valid_sids: (N, m) all SID codes (should be on same device as log_probs)
        sid_indices: (batch, C) indices into valid_sids to score

    Returns:
        scores: (batch, C) = sum of log_probs[b, j, sid[j]] over j
    """
    batch_size, C = sid_indices.shape
    m = log_probs.shape[1]
    device = log_probs.device

    # Gather SID codes: (batch, C, m)
    idx = sid_indices.to(valid_sids.device)
    codes = valid_sids[idx].to(device)  # (batch, C, m)

    # Score each digit: log_probs[b, j, codes[b,c,j]]
    scores = torch.zeros(batch_size, C, device=device)
    for j in range(m):
        # log_probs[:, j, :] is (batch, M)
        # codes[:, :, j] is (batch, C)
        digit_scores = torch.gather(
            log_probs[:, j, :],  # (batch, M)
            1,
            codes[:, :, j],  # (batch, C)
        )  # (batch, C)
        scores += digit_scores

    return scores


def graph_constrained_decode(
    log_probs: torch.Tensor,
    valid_sids: torch.Tensor,
    graph_neighbors: torch.Tensor,
    b: int = 10,
    q: int = 3,
    k: int = 100,
    top_n: int = 500,
    prefilter_per_digit: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Graph-constrained decoding for parallel OPQ predictions.

    Memory-efficient: never materializes (batch, N) score matrix.
    Instead uses per-digit prefiltering for seeds then graph expansion.

    Args:
        log_probs: (batch, m, M) — per-digit log probabilities
        valid_sids: (N, m) int tensor — all valid SID codes
        graph_neighbors: (N, k) int tensor — neighbor adjacency list
        b: beam width (seeds per iteration)
        q: number of graph expansion iterations
        k: neighbors per seed (from graph)
        top_n: final number of candidates to return
        prefilter_per_digit: top-K per digit for initial seed selection

    Returns:
        top_sids: (batch, top_n, m) — top candidate SID codes
        top_scores: (batch, top_n) — scores
    """
    batch_size, m, M = log_probs.shape
    N = valid_sids.shape[0]
    device = log_probs.device

    valid_sids_dev = valid_sids.to(device)

    # 1. Initial seed selection via per-digit prefiltering (GPU-friendly)
    #    For digit 0, take top-K values → find all SIDs matching → score them
    _, top_digit0 = log_probs[:, 0, :].topk(prefilter_per_digit, dim=-1)  # (batch, K)

    # Build candidate set on GPU: for each batch, SIDs whose digit 0 ∈ top-K
    digit0_all = valid_sids_dev[:, 0]  # (N,)
    seed_candidates_list = []
    for bi in range(batch_size):
        top_vals = top_digit0[bi]  # (K,) on device
        # Boolean mask: which SIDs have digit 0 matching any of top_vals
        mask = (digit0_all.unsqueeze(-1) == top_vals.unsqueeze(0)).any(dim=-1)  # (N,)
        cand_idx = mask.nonzero(as_tuple=True)[0]
        # Limit candidates
        if cand_idx.size(0) > 10000:
            perm = torch.randperm(cand_idx.size(0), device=device)[:10000]
            cand_idx = cand_idx[perm]
        seed_candidates_list.append(cand_idx)

    # Pad to same length for batched scoring
    max_cand = max(c.size(0) for c in seed_candidates_list)
    if max_cand == 0:
        max_cand = 1  # edge case
    seed_candidates = torch.zeros(batch_size, max_cand, dtype=torch.long, device=device)
    cand_mask = torch.zeros(batch_size, max_cand, dtype=torch.bool, device=device)
    for bi in range(batch_size):
        n_c = seed_candidates_list[bi].size(0)
        seed_candidates[bi, :n_c] = seed_candidates_list[bi]
        cand_mask[bi, :n_c] = True

    # Score seed candidates
    seed_scores = _score_sids(log_probs, valid_sids_dev, seed_candidates)  # (batch, max_cand)
    seed_scores[~cand_mask] = -1e9  # mask invalid

    # Select top-b seeds
    seed_b = min(b, max_cand)
    _, top_idx = seed_scores.topk(seed_b, dim=-1)  # (batch, b)
    current_indices = torch.gather(seed_candidates, 1, top_idx)  # (batch, b)

    # 2. Iterative graph expansion
    for _ in range(q):
        # Get neighbors: (batch, b) → (batch, b, k)
        neighbors = graph_neighbors[current_indices.cpu()].to(device)
        neighbor_flat = neighbors.reshape(batch_size, -1)  # (batch, b*k)

        # Combine seeds + neighbors
        combined = torch.cat([current_indices, neighbor_flat], dim=-1)  # (batch, b + b*k)

        # Score all candidates
        combined_scores = _score_sids(log_probs, valid_sids_dev, combined)

        # Select top-b
        top_b = min(b, combined.size(1))
        _, top_idx = combined_scores.topk(top_b, dim=-1)
        current_indices = torch.gather(combined, 1, top_idx)

    # 3. Final expansion to get top_n candidates
    neighbors = graph_neighbors[current_indices.cpu()].to(device)
    neighbor_flat = neighbors.reshape(batch_size, -1)
    final_combined = torch.cat([current_indices, neighbor_flat], dim=-1)
    final_scores = _score_sids(log_probs, valid_sids_dev, final_combined)

    actual_top_n = min(top_n, final_combined.size(1))
    top_scores, top_idx = final_scores.topk(actual_top_n, dim=-1)
    top_sid_indices = torch.gather(final_combined, 1, top_idx)

    # Retrieve actual SID codes
    top_sids = valid_sids_dev[top_sid_indices.cpu()].to(device)

    return top_sids, top_scores


# ============================================================
# Metric Class
# ============================================================

class SemanticIDPredictionMetric(BaseMetric):
    """Semantic ID Next Token Prediction (参考 OneRec)

    输入前 k 个 item 的 tokens，自回归预测下一个 item 的 tokens。
    支持:
    - 自回归模型 + Beam Search (RKMeans/FSQ, n_layers <= 4)
    - 并行模型 + Graph-Constrained Decoding (OPQ, n_layers >= 5)
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

        # Decide model type: parallel for OPQ (n_layers >= 5), AR otherwise
        # force_autoregressive=True forces AR even for OPQ (EXP-005 baseline)
        use_parallel = (n_layers >= 5) and not force_autoregressive

        n_gpus = torch.cuda.device_count() if device == 'cuda' else 0

        if use_parallel:
            return self._compute_parallel(
                n_clusters=n_clusters, n_layers=n_layers, n_items=n_items,
                sid_to_items=sid_to_items,
                train_samples_with_uid=train_samples_with_uid,
                eval_samples=eval_samples, eval_target_cids=eval_target_cids,
                batch_size=batch_size, recall_beam_size=recall_beam_size,
                device=device, n_gpus=n_gpus, verbose=verbose,
            )
        else:
            return self._compute_autoregressive(
                n_clusters=n_clusters, n_layers=n_layers, n_items=n_items,
                sid_to_items=sid_to_items,
                train_samples_with_uid=train_samples_with_uid,
                eval_samples=eval_samples, eval_target_cids=eval_target_cids,
                batch_size=batch_size, beam_size=beam_size,
                recall_beam_size=recall_beam_size,
                device=device, n_gpus=n_gpus, verbose=verbose,
            )

    def _compute_autoregressive(
        self,
        n_clusters, n_layers, n_items,
        sid_to_items, train_samples_with_uid, eval_samples, eval_target_cids,
        batch_size, beam_size, recall_beam_size, device, n_gpus, verbose,
    ) -> MetricResult:
        """Original autoregressive NTP pipeline (for RKMeans/FSQ, n_layers <= 4)."""
        import random

        # Model (S-tier MoE: ~27M params, ~7M active)
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

        # Eval dataset (reused after training)
        eval_dataset = ListDataset(eval_samples, n_layers)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, **dl_kwargs)

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
            print(f"    Perplexity: {ppl:.2f} (random: {n_clusters})")
            print(f"    Depth Acc (beam): {[f'{a:.4f}' for a in depth_acc]}")
            print(f"    Depth Hit@10: {[f'{h:.4f}' for h in hit10]}")
            for rk_key in sorted(k for k in trained_metrics if k.startswith('item_recall@')):
                print(f"    {rk_key}: {trained_metrics[rk_key]:.4f}")

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
            'train_time_s': round(train_elapsed, 1),
            'train_samples_per_s': round(len(train_samples) / train_elapsed),
            'trained_eval_time_s': round(trained_eval_elapsed, 1),
            'total_time_s': round(train_elapsed + trained_eval_elapsed, 1),
        }

        # Item recall results (dynamic keys from _run_eval)
        for key in trained_metrics:
            if key.startswith('item_recall@'):
                details[key] = trained_metrics[key]
        return MetricResult(
            name=self.name,
            value=ppl,
            layer_values=depth_acc,
            details=details,
            status=self.assess_quality(ppl),
        )

    def _compute_parallel(
        self,
        n_clusters, n_layers, n_items,
        sid_to_items, train_samples_with_uid, eval_samples, eval_target_cids,
        batch_size, recall_beam_size, device, n_gpus, verbose,
    ) -> MetricResult:
        """Parallel NTP pipeline for OPQ (n_layers >= 5).

        Uses ParallelNTPModel (bidirectional) + Graph-Constrained Decoding.
        """
        import random
        from gr_demo.model.opq import build_sid_graph

        if verbose:
            print(f"  [Parallel mode] OPQ m={n_layers}, M={n_clusters}")

        # Model
        ntp_model = ParallelNTPModel(
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

        raw_model = ntp_model
        if n_gpus > 1:
            ntp_model = nn.DataParallel(ntp_model)
            if verbose:
                print(f"  Using {n_gpus} GPUs (DataParallel)")

        optimizer = torch.optim.AdamW(ntp_model.parameters(), lr=1e-3, weight_decay=1e-5)
        scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

        total_params = sum(p.numel() for p in ntp_model.parameters())
        if verbose:
            arch_parts = [f"d={raw_model.embed_dim}", f"L={raw_model.n_transformer_layers}"]
            if raw_model.use_moe:
                arch_parts.append("MoE")
            arch_parts.append("parallel")
            print(f"  Model: {total_params:,} params ({', '.join(arch_parts)})")
            print(f"  Batch size: {batch_size}, AMP: BF16, Device: {device}")

        dl_kwargs = {'pin_memory': True, 'num_workers': 4} if device == 'cuda' else {}

        eval_dataset = ListDataset(eval_samples, n_layers)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, **dl_kwargs)

        # ============================================================
        # Phase 1: Parallel Training (MTP loss)
        # ============================================================
        random.seed(42)

        uid_to_samples = defaultdict(list)
        for uid, inp, tgt, ts, _cid in train_samples_with_uid:
            uid_to_samples[uid].append((inp, tgt))

        uid_list = list(uid_to_samples.keys())
        random.shuffle(uid_list)

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

            # Single forward → (batch, m, M) logits
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                logits, aux_loss = ntp_model(input_batch)  # (batch, m, M)
                # DataParallel: aux_loss may be a vector
                aux_loss = aux_loss.mean()

                # MTP loss = sum of per-digit CE
                ce_loss = sum(
                    F.cross_entropy(logits[:, j], target_batch[:, j])
                    for j in range(n_layers)
                )
                loss = ce_loss + ntp_model_aux_coef * aux_loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if verbose and (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - train_t0
                progress = (batch_idx + 1) / n_train_batches
                eta = elapsed / progress * (1 - progress)
                print(f"    Train: {progress*100:.1f}% | MTP-CE: {ce_loss.item():.4f} | Aux: {aux_loss.item():.4f} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

        train_elapsed = time.time() - train_t0
        if verbose:
            print(f"  Training done in {train_elapsed:.1f}s ({len(train_samples)/train_elapsed:.0f} samples/s)")

        # ============================================================
        # Phase 2: Build SID graph
        # ============================================================
        if verbose:
            print(f"  Building SID graph for {len(sid_to_items):,} unique SIDs...")

        graph_t0 = time.time()
        # Build valid_sids array: (N, m)
        sid_strings = list(sid_to_items.keys())
        valid_sids_np = np.array(
            [[int(t) for t in s.split('_')] for s in sid_strings],
            dtype=np.int64,
        )
        valid_sids_tensor = torch.tensor(valid_sids_np, dtype=torch.long)

        # Build sid_str -> index mapping for recall lookup
        sid_str_to_idx = {s: i for i, s in enumerate(sid_strings)}

        # Build graph using OPQ decode (need quantizer from kwargs or reconstruct)
        # Since we don't have the quantizer here, build graph from embedding similarity
        # Use the codes directly: decode via codebook lookup
        # We build a simple L2 graph on the SID code space instead
        graph_neighbors = build_sid_graph(
            quantizer=None,  # will use code-based fallback below
            valid_sids=valid_sids_np,
            top_k=100,
            verbose=verbose,
        ) if False else self._build_code_graph(valid_sids_np, top_k=100, verbose=verbose)

        graph_neighbors_tensor = torch.tensor(graph_neighbors, dtype=torch.long)
        graph_elapsed = time.time() - graph_t0

        if verbose:
            print(f"  SID graph built in {graph_elapsed:.1f}s")

        # ============================================================
        # Phase 3: Parallel Eval with Graph-Constrained Decoding
        # ============================================================
        if verbose:
            print(f"  Evaluating on {len(eval_samples)} samples (graph-constrained decoding)...")

        eval_t0 = time.time()
        ntp_model.eval()  # DP wrapper or raw model — both work

        eval_losses = []
        # Per-digit accuracy (independent, not prefix)
        digit_hit_1 = [0] * n_layers
        digit_hit_5 = [0] * n_layers
        digit_hit_10 = [0] * n_layers
        total_eval = 0

        # Item recall
        recall_ks = [10, 50, 100, 500]
        item_recall = {k: 0 for k in recall_ks}
        eval_offset = 0
        n_batches = len(eval_loader)

        use_amp = (device != 'cpu' and torch.cuda.is_available())

        # Pre-move GCD data to GPU 0 once
        valid_sids_gpu = valid_sids_tensor.to(device)
        graph_neighbors_gpu = graph_neighbors_tensor  # keep on CPU, GCD indexes into it

        with torch.no_grad():
            for batch_idx, (input_batch, target_batch) in enumerate(eval_loader):
                input_batch = input_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                curr_batch_size = input_batch.size(0)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    logits, _ = ntp_model(input_batch)  # (batch, m, M) — uses all GPUs via DP

                # MTP loss
                loss = sum(
                    F.cross_entropy(logits[:, j], target_batch[:, j])
                    for j in range(n_layers)
                )
                eval_losses.append(loss.item() * curr_batch_size)

                # Per-digit Hit@K (independent, NOT prefix)
                for j in range(n_layers):
                    digit_logits = logits[:, j]  # (batch, M)
                    top1 = digit_logits.topk(1, dim=-1).indices
                    top5 = digit_logits.topk(5, dim=-1).indices
                    top10 = digit_logits.topk(10, dim=-1).indices
                    target_j = target_batch[:, j:j+1]
                    digit_hit_1[j] += (top1 == target_j).any(dim=-1).sum().item()
                    digit_hit_5[j] += (top5 == target_j).any(dim=-1).sum().item()
                    digit_hit_10[j] += (top10 == target_j).any(dim=-1).sum().item()

                # Graph-Constrained Decoding for item recall
                log_probs = F.log_softmax(logits.float(), dim=-1)  # (batch, m, M)
                top_sids, top_scores = graph_constrained_decode(
                    log_probs, valid_sids_gpu, graph_neighbors_gpu,
                    b=10, q=3, k=100, top_n=500,
                )
                # top_sids: (batch, top_n, m)

                # Item recall
                top_sids_cpu = top_sids.cpu()
                for sample_idx in range(curr_batch_size):
                    target_cid = eval_target_cids[eval_offset + sample_idx]
                    candidate_items = []
                    seen = set()
                    for rank in range(top_sids_cpu.size(1)):
                        sid_str = '_'.join(
                            str(t.item()) for t in top_sids_cpu[sample_idx, rank]
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

                if verbose and (batch_idx + 1) % 20 == 0:
                    pct = (batch_idx + 1) / n_batches * 100
                    print(f"    Eval: {pct:.0f}% ({total_eval:,})")

        eval_elapsed = time.time() - eval_t0

        if total_eval == 0:
            return MetricResult(name=self.name, value=0.0, details={'error': 'No eval samples'}, status='unknown')

        avg_loss = sum(eval_losses) / total_eval
        ppl = np.exp(avg_loss / n_layers)

        # Per-digit accuracy lists
        hit1_rates = [digit_hit_1[j] / total_eval for j in range(n_layers)]
        hit5_rates = [digit_hit_5[j] / total_eval for j in range(n_layers)]
        hit10_rates = [digit_hit_10[j] / total_eval for j in range(n_layers)]

        if verbose:
            print(f"  Eval done in {eval_elapsed:.1f}s")
            print(f"  [Parallel] Results:")
            print(f"    Perplexity: {ppl:.2f} (random: {n_clusters})")
            print(f"    Digit Hit@1:  {[f'{h:.4f}' for h in hit1_rates]}")
            print(f"    Digit Hit@5:  {[f'{h:.4f}' for h in hit5_rates]}")
            print(f"    Digit Hit@10: {[f'{h:.4f}' for h in hit10_rates]}")
            for k in recall_ks:
                print(f"    item_recall@{k}: {item_recall[k] / total_eval:.4f}")

        sid_counts = [len(items) for items in sid_to_items.values()]
        avg_collision = np.mean(sid_counts) if sid_counts else 0

        details = {
            'perplexity': ppl,
            'random_perplexity': n_clusters,
            'ntp_loss': avg_loss,
            'depth_acc_beam': hit1_rates,  # for compat: use digit hit@1 as "depth_acc"
            'depth_hit@5': hit5_rates,
            'depth_hit@10': hit10_rates,
            'beam_size': 0,  # no beam search
            'recall_beam_size': recall_beam_size,
            'n_items': n_items,
            'n_layers': n_layers,
            'n_clusters': n_clusters,
            'n_train': len(train_samples),
            'n_eval': len(eval_samples),
            'n_unique_sids': len(sid_to_items),
            'avg_items_per_sid': round(avg_collision, 2),
            # Model architecture
            'model_type': 'parallel',
            'model_params': total_params,
            'model_embed_dim': 256,
            'model_n_transformer_layers': 6,
            'model_n_heads': 8,
            'model_use_moe': True,
            'model_n_experts': 8,
            'model_top_k': 2,
            'model_expert_dim': 1024,
            # Graph decoding
            'graph_n_sids': len(sid_strings),
            'graph_top_k': 100,
            'gcd_b': 10,
            'gcd_q': 3,
            # Timing
            'train_time_s': round(train_elapsed, 1),
            'train_samples_per_s': round(len(train_samples) / train_elapsed),
            'graph_build_time_s': round(graph_elapsed, 1),
            'trained_eval_time_s': round(eval_elapsed, 1),
            'total_time_s': round(train_elapsed + graph_elapsed + eval_elapsed, 1),
        }

        for k in recall_ks:
            details[f'item_recall@{k}'] = item_recall[k] / total_eval

        return MetricResult(
            name=self.name,
            value=ppl,
            layer_values=hit10_rates,
            details=details,
            status=self.assess_quality(ppl),
        )

    @staticmethod
    def _build_code_graph(
        valid_sids: np.ndarray,
        top_k: int = 100,
        verbose: bool = True,
    ) -> np.ndarray:
        """Build SID graph based on code-space Hamming-like distance.

        For each SID, find top_k nearest neighbors by number of matching digits.
        GPU-accelerated: distributes chunks across all available GPUs.

        Args:
            valid_sids: (N, m) int array
            top_k: neighbors per SID
            verbose: print progress

        Returns:
            neighbors: (N, top_k) int32 array
        """
        N, m = valid_sids.shape
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if verbose:
            print(f"  Building code-space SID graph: {N:,} SIDs, m={m}, top_k={top_k}, GPUs={n_gpus}")

        neighbors = np.zeros((N, top_k), dtype=np.int32)

        if n_gpus == 0:
            # CPU fallback
            sids_tensor = torch.tensor(valid_sids, dtype=torch.long)
            chunk_size = min(1024, N)
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                chunk = sids_tensor[start:end]
                matches = (chunk.unsqueeze(1) == sids_tensor.unsqueeze(0)).sum(dim=-1).float()
                for i in range(end - start):
                    matches[i, start + i] = -1
                _, topk_idx = matches.topk(top_k, dim=-1)
                neighbors[start:end] = topk_idx.numpy()
            return neighbors

        # Multi-GPU: split query chunks across GPUs
        # Keep full SID table on each GPU, process query chunks in parallel
        import concurrent.futures

        # Determine chunk size per GPU based on VRAM
        # (chunk, N, m) int64 tensor: chunk * N * m * 8 bytes
        # For N=5M, m=8: each row of the expanded tensor = 5M * 8 * 8 = 320MB
        # Safe chunk size: ~256 rows per GPU per iteration for 80GB A100
        chunk_per_gpu = min(256, max(1, N // (n_gpus * 4)))
        total_chunks = (N + chunk_per_gpu - 1) // chunk_per_gpu

        if verbose:
            print(f"    chunk_per_gpu={chunk_per_gpu}, total_chunks={total_chunks}")

        def process_chunk_on_gpu(args):
            gpu_id, start, end = args
            device = torch.device(f'cuda:{gpu_id}')
            sids_gpu = torch.tensor(valid_sids, dtype=torch.long, device=device)
            chunk = sids_gpu[start:end]  # (C, m)
            # Hamming similarity: count matching digits
            matches = (chunk.unsqueeze(1) == sids_gpu.unsqueeze(0)).sum(dim=-1).float()  # (C, N)
            # Mask self
            self_indices = torch.arange(start, end, device=device)
            matches[torch.arange(end - start, device=device), self_indices] = -1
            _, topk_idx = matches.topk(min(top_k, N - 1), dim=-1)
            return start, end, topk_idx.cpu().numpy()

        # Build work items: assign chunks round-robin to GPUs
        work_items = []
        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_per_gpu
            end = min(start + chunk_per_gpu, N)
            gpu_id = chunk_idx % n_gpus
            work_items.append((gpu_id, start, end))

        # Process with thread pool (GPU ops release GIL)
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = [executor.submit(process_chunk_on_gpu, item) for item in work_items]
            for future in concurrent.futures.as_completed(futures):
                start, end, topk = future.result()
                neighbors[start:end, :topk.shape[1]] = topk
                completed += end - start
                if verbose and (completed % (chunk_per_gpu * n_gpus * 5) == 0 or completed >= N):
                    print(f"    Graph: {completed:,}/{N:,}")

        if verbose:
            sample_n = min(1000, N)
            sample_idx = np.random.choice(N, sample_n, replace=False)
            avg_match = np.mean([
                np.sum(valid_sids[i] == valid_sids[neighbors[i, 0]])
                for i in sample_idx
            ])
            print(f"  Graph stats: avg top-1 digit match = {avg_match:.1f}/{m}")

        return neighbors
