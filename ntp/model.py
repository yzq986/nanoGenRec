"""
S-tier NTP Model — 6-layer MoE Transformer decoder with Loss-Free load balancing.

~39.5M total params, ~11M active (top-2 of 8 SwiGLU experts).
Designed for DDP training (ntp/train.py) and eval (ntp/eval.py).

Reference:
  - OneRec (arxiv 2506.13695): SwiGLU MoE architecture
  - DeepSeek / IDEA-onemall-4: Loss-Free dynamic bias MoE balancing
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SwiGLU Expert FFN (Mixtral / Llama style)
# ============================================================

class ExpertFFN(nn.Module):
    """Single expert: SwiGLU FFN — w2(silu(w1(x)) * w3(x))"""

    def __init__(self, embed_dim: int, expert_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, expert_dim, bias=False)  # gate
        self.w2 = nn.Linear(expert_dim, embed_dim, bias=False)  # down
        self.w3 = nn.Linear(embed_dim, expert_dim, bias=False)  # up
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ============================================================
# Sparse MoE with Loss-Free dynamic bias balancing
# ============================================================

class SparseMoEBlock(nn.Module):
    """Sparse Mixture of Experts with Loss-Free load balancing.

    Instead of Switch Transformer auxiliary loss (which interferes with
    the main task gradient), uses a non-gradient dynamic bias on router
    logits to steer token allocation toward uniform expert utilization.

    Reference: DeepSeek-V2 (arxiv 2405.04434), IDEA-onemall-4
    """

    def __init__(
        self,
        embed_dim: int,
        expert_dim: int,
        n_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        bias_lr: float = 0.01,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.bias_lr = bias_lr

        self.router = nn.Linear(embed_dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(embed_dim, expert_dim, dropout) for _ in range(n_experts)
        ])
        # Loss-Free: non-gradient bias, persisted via register_buffer
        self.register_buffer('expert_bias', torch.zeros(n_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])  # (B*S, D)

        # Router with bias injection
        router_logits = self.router(x_flat) + self.expert_bias
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection + renormalize
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Dispatch and combine
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            weights = top_k_weights[:, k]
            for ei in range(self.n_experts):
                mask = (expert_indices == ei)
                if not mask.any():
                    continue
                expert_out = self.experts[ei](x_flat[mask])
                output[mask] += weights[mask].unsqueeze(-1) * expert_out

        # Loss-Free bias update (training only, no gradient)
        if self.training:
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_indices, self.n_experts).float()
                freq = expert_mask.sum(dim=1).mean(dim=0)  # (n_experts,)
                self.expert_bias.add_(-self.bias_lr * (freq - 1.0 / self.n_experts))

        return output.view(orig_shape)


# ============================================================
# Transformer Layer (pre-norm, causal, supports MoE or dense FFN)
# ============================================================

class TransformerLayer(nn.Module):
    """Pre-norm Transformer layer: LayerNorm → Attention → LayerNorm → FFN/MoE."""

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
        self.causal = causal
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True,
        )
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        if self.causal:
            T = x_norm.size(1)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                T, device=x.device,
            )
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-norm FFN/MoE
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# NTPModel — S-tier decoder (6L MoE, ~39.5M params)
# ============================================================

class NTPModel(nn.Module):
    """S-tier Next Token Prediction model.

    6-layer causal Transformer with SwiGLU MoE (8 experts, top-2).
    Loss-Free dynamic bias for load balancing (no auxiliary loss).
    Per-layer token embeddings and output projections for different codebook sizes.

    Interface matches NTPProbe: same forward() / beam_search() signatures.
    """

    def __init__(
        self,
        n_clusters_per_layer: list,
        n_sid_layers: int,
        n_items: int = 10,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_transformer_layers: int = 6,
        dropout: float = 0.1,
        use_moe: bool = True,
        n_experts: int = 8,
        top_k: int = 2,
        expert_dim: int = 1024,
        parallel: bool = False,
    ):
        super().__init__()
        assert len(n_clusters_per_layer) == n_sid_layers
        self.n_clusters_per_layer = n_clusters_per_layer
        self.n_sid_layers = n_sid_layers
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.parallel = parallel
        self.seq_len = n_items * n_sid_layers

        # Per-layer token embeddings (different codebook per SID layer)
        self.token_embs = nn.ModuleList([
            nn.Embedding(nc, embed_dim) for nc in n_clusters_per_layer
        ])
        max_len = self.seq_len + n_sid_layers
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim, n_heads, dropout,
                use_moe=use_moe, n_experts=n_experts,
                top_k=top_k, expert_dim=expert_dim,
                causal=not parallel,
            )
            for _ in range(n_transformer_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # Per-layer output projections (different codebook sizes)
        self.output_projs = nn.ModuleList([
            nn.Linear(embed_dim, nc) for nc in n_clusters_per_layer
        ])

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Per-layer token embedding lookup. Position i uses token_embs[i % n_sid_layers]."""
        B, T = tokens.size()
        device = tokens.device
        L = self.n_sid_layers
        layer_ids = torch.arange(T, device=device) % L

        x = torch.zeros(B, T, self.embed_dim, device=device)
        for l in range(L):
            mask = (layer_ids == l)
            if mask.any():
                x[:, mask] = self.token_embs[l](tokens[:, mask])
        return x

    def _transformer_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through all transformer layers + final norm."""
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

    def forward(
        self,
        input_tokens: torch.Tensor,
        generated_tokens: Optional[torch.Tensor] = None,
        return_last_n: int = 1,
    ):
        """Same interface as NTPProbe.forward().

        Args:
            input_tokens: (B, seq_len) history SID tokens
            generated_tokens: (B, k) already-generated target tokens (AR mode)
            return_last_n: trailing positions to return logits for
        Returns:
            if parallel: list of n_sid_layers tensors [(B, C_l), ...]
            if return_last_n=1: (B, C_layer) single logit tensor
            if return_last_n>1: list of n tensors [(B, C_l0), (B, C_l1), ...]
        """
        device = input_tokens.device

        if self.parallel:
            positions = torch.arange(self.seq_len, device=device).unsqueeze(0)
            x = self._embed_tokens(input_tokens) + self.pos_emb(positions)
            out = self._transformer_forward(x)
            s = out[:, -1, :]
            return [self.output_projs[l](s) for l in range(self.n_sid_layers)]

        # Autoregressive
        if generated_tokens is not None and generated_tokens.size(1) > 0:
            tokens = torch.cat([input_tokens, generated_tokens], dim=1)
        else:
            tokens = input_tokens

        T = tokens.size(1)
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self._embed_tokens(tokens) + self.pos_emb(positions)
        out = self._transformer_forward(x)

        if return_last_n == 1:
            target_layer = T % self.n_sid_layers
            return self.output_projs[target_layer](out[:, -1, :])
        else:
            logits_list = []
            for i in range(return_last_n):
                pos = T - return_last_n + i
                target_layer = (pos + 1) % self.n_sid_layers
                logits_list.append(self.output_projs[target_layer](out[:, pos, :]))
            return logits_list

    @torch.no_grad()
    def beam_search(self, input_tokens: torch.Tensor, beam_size: int = 5) -> torch.Tensor:
        """Beam search for autoregressive mode. Same interface as NTPProbe.

        Returns:
            beams: (B, beam_size, n_sid_layers)
        """
        B = input_tokens.size(0)
        device = input_tokens.device
        L = self.n_sid_layers

        beams = torch.zeros(B, 1, 0, dtype=torch.long, device=device)
        scores = torch.zeros(B, 1, device=device)

        for step in range(L):
            n_beams = beams.size(1)
            input_exp = input_tokens.unsqueeze(1).expand(-1, n_beams, -1).reshape(B * n_beams, -1)
            gen_exp = beams.reshape(B * n_beams, -1) if step > 0 else None

            logits = self.forward(input_exp, gen_exp)
            log_probs = F.log_softmax(logits, dim=-1)
            C = log_probs.size(-1)

            log_probs = log_probs.view(B, n_beams, C)
            candidate_scores = scores.unsqueeze(-1) + log_probs
            flat_scores = candidate_scores.view(B, -1)
            topk_scores, topk_idx = flat_scores.topk(beam_size, dim=-1)

            beam_idx = topk_idx // C
            token_idx = topk_idx % C

            prev_beams = torch.gather(
                beams, 1, beam_idx.unsqueeze(-1).expand(-1, -1, step)
            ) if step > 0 else torch.zeros(B, beam_size, 0, dtype=torch.long, device=device)

            beams = torch.cat([prev_beams, token_idx.unsqueeze(-1)], dim=-1)
            scores = topk_scores

        return beams
