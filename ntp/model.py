"""
Semantic ID Next Token Prediction — Model & Dataset.

NTPProbe model definition + SIDSequenceDataset.
Training is in ntp/train.py (DDP). Eval is in ntp/eval.py.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


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
        n_clusters_per_layer: list,
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
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_transformer_layers)

        # Per-layer output projections (different codebook sizes)
        self.output_projs = nn.ModuleList([
            nn.Linear(embed_dim, nc) for nc in n_clusters_per_layer
        ])

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Per-layer token embedding lookup. Position i uses token_embs[i % n_sid_layers]."""
        B, T = tokens.size()
        device = tokens.device
        L = self.n_sid_layers

        # layer_ids: [0,1,2, 0,1,2, ..., 0,1,2] for each position
        layer_ids = torch.arange(T, device=device) % L

        # Gather per-layer embeddings
        x = torch.zeros(B, T, self.embed_dim, device=device)
        for l in range(L):
            mask = (layer_ids == l)  # (T,)
            if mask.any():
                x[:, mask] = self.token_embs[l](tokens[:, mask])
        return x

    def forward(self, input_tokens: torch.Tensor,
                generated_tokens: Optional[torch.Tensor] = None,
                return_last_n: int = 1):
        """
        Args:
            input_tokens: (B, seq_len) history SID tokens
            generated_tokens: (B, k) already-generated target tokens (AR mode only)
            return_last_n: number of trailing positions to return logits for.
        Returns:
            if return_last_n=1: (B, C_layer) single logit tensor
            if return_last_n>1: list of n tensors [(B, C_l0), (B, C_l1), ...] per-layer logits
        """
        B = input_tokens.size(0)
        device = input_tokens.device

        if self.parallel:
            # Encode history, then independent prediction per position
            positions = torch.arange(self.seq_len, device=device).unsqueeze(0)
            x = self._embed_tokens(input_tokens) + self.pos_emb(positions)

            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                self.seq_len, device=device
            )
            memory = self.decoder(x, x, tgt_mask=causal_mask)

            # Pool last position as sequence representation
            s = memory[:, -1, :]  # (B, D)

            # Per-layer output projection (different codebook sizes)
            return [self.output_projs[l](s) for l in range(self.n_sid_layers)]
        else:
            # Autoregressive: concatenate history + generated tokens
            if generated_tokens is not None and generated_tokens.size(1) > 0:
                tokens = torch.cat([input_tokens, generated_tokens], dim=1)
            else:
                tokens = input_tokens

            T = tokens.size(1)
            positions = torch.arange(T, device=device).unsqueeze(0)
            x = self._embed_tokens(tokens) + self.pos_emb(positions)

            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
            out = self.decoder(x, x, tgt_mask=causal_mask)

            if return_last_n == 1:
                # Position T-1 predicts next token at layer (T % n_sid_layers)
                target_layer = T % self.n_sid_layers
                return self.output_projs[target_layer](out[:, -1, :])  # (B, C_l)
            else:
                # return_last_n positions, each with its own codebook size
                logits_list = []
                for i in range(return_last_n):
                    pos = T - return_last_n + i
                    target_layer = (pos + 1) % self.n_sid_layers
                    logits_list.append(self.output_projs[target_layer](out[:, pos, :]))
                return logits_list  # [(B, C_l0), (B, C_l1), ...]

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
