"""
QFormer: Learnable cross-attention compressor for frozen encoder hidden states.

Architecture (BLIP-2 / OneRec style):
    Frozen Encoder → S tokens × D_enc (last_hidden_state)
        ↓
    QFormer (N layers of cross-attention + self-attention + FFN)
        ↓
    M tokens × D_out (compressed item representation)
        ↓
    mean-pool → single vector → L2 normalize → contrastive loss / OPQ

Reference:
    - BLIP-2 (arxiv 2301.12597): Q-Former with learnable queries
    - OneRec (arxiv 2506.13695v4): miniCPM-V + 4-layer QFormer → SID
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QFormerLayer(nn.Module):
    """Single QFormer layer: self-attention → cross-attention → FFN."""

    def __init__(self, d_model: int, n_heads: int, d_enc: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads

        # Self-attention (among query tokens)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (queries attend to encoder hidden states)
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_k = nn.Linear(d_enc, d_model)
        self.cross_attn_v = nn.Linear(d_enc, d_model)
        self.cross_attn_out = nn.Linear(d_model, d_model)
        self.cross_attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,          # (B, M, d_model)
        encoder_hidden: torch.Tensor,    # (B, S, d_enc)
        encoder_mask: torch.Tensor | None = None,  # (B, S) bool, True=valid
    ) -> torch.Tensor:
        # 1) Self-attention among queries
        residual = queries
        queries = self.norm1(queries)
        queries = residual + self.self_attn(queries, queries, queries)[0]

        # 2) Cross-attention: queries attend to encoder hidden states
        residual = queries
        q = self.norm2(queries)
        q = self.cross_attn_q(q)                          # (B, M, d_model)
        k = self.cross_attn_k(encoder_hidden)              # (B, S, d_model)
        v = self.cross_attn_v(encoder_hidden)              # (B, S, d_model)

        B, M, _ = q.shape
        S = k.shape[1]
        # Reshape for multi-head attention
        q = q.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, M, hd)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, hd)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, hd)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if encoder_mask is not None:
            # encoder_mask: (B, S) → (B, 1, 1, S)
            attn_mask = ~encoder_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.cross_attn_drop(attn_weights)

        attn_out = torch.matmul(attn_weights, v)  # (B, H, M, hd)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, M, self.d_model)
        attn_out = self.cross_attn_out(attn_out)
        queries = residual + attn_out

        # 3) FFN
        residual = queries
        queries = residual + self.ffn(self.norm3(queries))

        return queries


class QFormer(nn.Module):
    """QFormer: learnable query tokens + N cross-attention layers.

    Args:
        num_queries: M, number of learnable query tokens (OneRec uses 4)
        num_layers: N, number of QFormer layers (OneRec uses 4)
        d_model: query/output dimension
        d_enc: encoder hidden state dimension (Qwen3-0.6B = 1024)
        n_heads: attention heads
        dropout: dropout rate
    """

    def __init__(
        self,
        num_queries: int = 4,
        num_layers: int = 2,
        d_model: int = 1024,
        d_enc: int = 1024,
        n_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)

        # QFormer layers
        self.layers = nn.ModuleList([
            QFormerLayer(d_model, n_heads, d_enc, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        encoder_hidden: torch.Tensor,               # (B, S, d_enc)
        encoder_mask: torch.Tensor | None = None,   # (B, S) bool
    ) -> torch.Tensor:
        """
        Returns:
            pooled: (B, d_model) mean-pooled over M query tokens, L2-normalized
        """
        B = encoder_hidden.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)  # (B, M, d_model)

        for layer in self.layers:
            queries = layer(queries, encoder_hidden, encoder_mask)

        queries = self.final_norm(queries)  # (B, M, d_model)

        # Mean pool over query tokens → single vector
        pooled = queries.mean(dim=1)  # (B, d_model)
        return pooled

    def param_count(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total"
