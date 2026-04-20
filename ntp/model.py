"""
S-tier NTP Model — 6-layer MoE Transformer decoder with Loss-Free load balancing.

~39.5M total params, ~11M active (top-2 of 8 SwiGLU experts).
Designed for DDP training (ntp/train.py) and eval (ntp/eval.py).

Reference:
  - OneRec (arxiv 2506.13695): SwiGLU MoE architecture
  - DeepSeek / IDEA-onemall-4: Loss-Free dynamic bias MoE balancing
"""

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SID Trie for constrained beam search
# ============================================================

class SIDTrie:
    """Prefix trie over existing SID token sequences.

    Built from sid_to_items dict. At each layer, provides the set of
    valid next tokens given a prefix, ensuring beam search only
    produces SIDs that exist in the corpus.
    """

    def __init__(self, sid_to_items: Dict[str, set], n_layers: int):
        # children[layer] = dict mapping prefix_tuple → set of valid next tokens
        self.n_layers = n_layers
        self.children: List[Dict[tuple, Set[int]]] = [dict() for _ in range(n_layers)]

        for sid_str in sid_to_items:
            tokens = tuple(int(t) for t in sid_str.split('_'))
            if len(tokens) != n_layers:
                continue
            for layer in range(n_layers):
                prefix = tokens[:layer]
                if prefix not in self.children[layer]:
                    self.children[layer][prefix] = set()
                self.children[layer][prefix].add(tokens[layer])

    def valid_tokens(self, layer: int, prefix: tuple) -> Set[int]:
        """Return set of valid tokens at `layer` given `prefix` (tuple of ints)."""
        return self.children[layer].get(prefix, set())

    def root_tokens(self) -> Set[int]:
        """Valid tokens at layer 0 (no prefix)."""
        return self.children[0].get((), set())


@torch.no_grad()
def constrained_beam_search(
    model,
    input_tokens: torch.Tensor,
    trie: SIDTrie,
    beam_size: int = 500,
    prefix: torch.Tensor = None,
    ctx_kv_caches=None,
    initial_logits: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Trie-constrained beam search with KV cache — every beam is a real SID.

    At each decoding step, masks out tokens not in the trie given the
    current beam prefix. This guarantees every output beam corresponds
    to an actual item in the corpus.

    Optimized for B=1 (eval processes one sample at a time).

    Args:
        model: NTPModel with forward_cached() for KV-cached inference.
        input_tokens: (B, T) context tokens
        trie: SIDTrie built from sid_to_items
        beam_size: number of beams to keep
        prefix: (B, P) optional fixed prefix tokens. Beam search starts from
                layer P instead of 0. Use to lock L0 (P=1) or L0+L1 (P=2)
                for targeted Medium/Hard candidate generation.
        ctx_kv_caches: pre-computed KV caches for the context. If None,
                the context is encoded from input_tokens.
        initial_logits: (B, C) logits from the last context position.
                Required when ctx_kv_caches is provided and prefix is None.

    Returns:
        beams: (B, actual_beams, n_layers) — token indices
        scores: (B, actual_beams) — log-probabilities
        ctx_kv_caches: context KV caches (for reuse across passes/items)
    """
    B = input_tokens.size(0)
    device = input_tokens.device
    L = trie.n_layers

    # ── Phase 0: context encoding (or reuse) ──
    use_kv_cache = hasattr(model, 'forward_cached')
    if not use_kv_cache:
        # Fallback for NTPProbe or models without forward_cached
        return _constrained_beam_search_legacy(
            model, input_tokens, trie, beam_size, prefix)

    if ctx_kv_caches is None:
        initial_logits, ctx_kv_caches = model.forward_cached(input_tokens)

    # ── Phase 1: beam init ──
    if prefix is not None:
        P = prefix.size(1)
        beams = prefix.unsqueeze(1)  # (B, 1, P)
        start_step = P
        step_kv = [c.clone() for c in ctx_kv_caches]
        current_logits, step_kv = model.forward_cached(
            generated_tokens=prefix, kv_caches=step_kv)
    else:
        beams = torch.zeros(B, 1, 0, dtype=torch.long, device=device)
        start_step = 0
        step_kv = [c.clone() for c in ctx_kv_caches]
        current_logits = initial_logits
    scores = torch.zeros(B, 1, device=device)

    # ── Phase 2: decode ──
    for step in range(start_step, L):
        n_beams = beams.size(1)

        if step > start_step:
            last_tokens = beams[:, :, -1].reshape(B * n_beams, 1)
            current_logits, step_kv = model.forward_cached(
                generated_tokens=last_tokens, kv_caches=step_kv)

        log_probs = F.log_softmax(
            current_logits.view(B, n_beams, -1), dim=-1)
        C = log_probs.size(-1)

        # Build trie mask: group beams by prefix to minimize dict lookups
        mask = torch.zeros(B * n_beams, C, dtype=torch.bool, device=device)
        beams_cpu = beams.cpu()

        prefix_to_valid: Dict[tuple, List[int]] = {}
        for bi in range(B):
            for ki in range(n_beams):
                pfx = tuple(beams_cpu[bi, ki].tolist())
                if pfx not in prefix_to_valid:
                    valid = trie.valid_tokens(step, pfx)
                    prefix_to_valid[pfx] = list(valid) if valid else []

        for bi in range(B):
            for ki in range(n_beams):
                pfx = tuple(beams_cpu[bi, ki].tolist())
                valid_list = prefix_to_valid[pfx]
                if valid_list:
                    idx = bi * n_beams + ki
                    mask[idx, valid_list] = True

        # Mask invalid tokens
        log_probs = log_probs.masked_fill(
            ~mask.view(B, n_beams, C), float('-inf'))

        candidate_scores = scores.unsqueeze(-1) + log_probs
        flat_scores = candidate_scores.view(B, -1)

        n_valid = int(mask.sum().item())
        if n_valid == 0:
            break
        k = min(beam_size, n_valid, flat_scores.size(1))
        topk_scores, topk_idx = flat_scores.topk(k, dim=-1)

        beam_idx = topk_idx // C
        token_idx = topk_idx % C

        prev_beams = torch.gather(
            beams, 1, beam_idx.unsqueeze(-1).expand(-1, -1, step)
        ) if step > 0 else torch.zeros(B, k, 0, dtype=torch.long, device=device)

        beams = torch.cat([prev_beams, token_idx.unsqueeze(-1)], dim=-1)
        scores = topk_scores

        # ── KV cache beam gather ──
        for li in range(len(step_kv)):
            c = step_kv[li]  # (B*n_beams_old, T_cached, D)
            T_c, D = c.size(1), c.size(2)
            c = c.view(B, n_beams, T_c, D)
            c = torch.gather(
                c, 1,
                beam_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T_c, D))
            step_kv[li] = c.reshape(B * k, T_c, D)

        # Trim dead beams
        valid_beam_mask = scores > float('-inf')
        n_alive = int(valid_beam_mask.sum(dim=1).max().item())
        if n_alive < beams.size(1):
            n_keep = max(n_alive, 1)
            beams = beams[:, :n_keep]
            scores = scores[:, :n_keep]
            for li in range(len(step_kv)):
                step_kv[li] = step_kv[li][:B * n_keep]

    return beams, scores, ctx_kv_caches


def _constrained_beam_search_legacy(
    model, input_tokens, trie, beam_size=500, prefix=None,
):
    """Legacy beam search without KV cache (for NTPProbe or verification)."""
    B = input_tokens.size(0)
    device = input_tokens.device
    L = trie.n_layers

    if prefix is not None:
        P = prefix.size(1)
        beams = prefix.unsqueeze(1)
        scores = torch.zeros(B, 1, device=device)
        start_step = P
    else:
        beams = torch.zeros(B, 1, 0, dtype=torch.long, device=device)
        scores = torch.zeros(B, 1, device=device)
        start_step = 0

    for step in range(start_step, L):
        n_beams = beams.size(1)
        input_exp = input_tokens.unsqueeze(1).expand(
            -1, n_beams, -1).reshape(B * n_beams, -1)
        gen_exp = beams.reshape(B * n_beams, -1) if step > 0 else None

        logits = model.forward(input_exp, gen_exp)
        log_probs = F.log_softmax(logits, dim=-1)
        C = log_probs.size(-1)

        mask = torch.zeros(B * n_beams, C, dtype=torch.bool, device=device)
        beams_cpu = beams.cpu()

        prefix_to_valid: Dict[tuple, List[int]] = {}
        for bi in range(B):
            for ki in range(n_beams):
                pfx = tuple(beams_cpu[bi, ki].tolist())
                if pfx not in prefix_to_valid:
                    valid = trie.valid_tokens(step, pfx)
                    prefix_to_valid[pfx] = list(valid) if valid else []

        for bi in range(B):
            for ki in range(n_beams):
                pfx = tuple(beams_cpu[bi, ki].tolist())
                valid_list = prefix_to_valid[pfx]
                if valid_list:
                    idx = bi * n_beams + ki
                    mask[idx, valid_list] = True

        log_probs = log_probs.masked_fill(~mask, float('-inf'))
        log_probs = log_probs.view(B, n_beams, C)
        candidate_scores = scores.unsqueeze(-1) + log_probs
        flat_scores = candidate_scores.view(B, -1)

        n_valid = int(mask.sum().item())
        if n_valid == 0:
            break
        k = min(beam_size, n_valid, flat_scores.size(1))
        topk_scores, topk_idx = flat_scores.topk(k, dim=-1)

        beam_idx = topk_idx // C
        token_idx = topk_idx % C

        prev_beams = torch.gather(
            beams, 1, beam_idx.unsqueeze(-1).expand(-1, -1, step)
        ) if step > 0 else torch.zeros(B, k, 0, dtype=torch.long, device=device)

        beams = torch.cat([prev_beams, token_idx.unsqueeze(-1)], dim=-1)
        scores = topk_scores

        valid_beam_mask = scores > float('-inf')
        n_alive = int(valid_beam_mask.sum(dim=1).max().item())
        if n_alive < beams.size(1):
            beams = beams[:, :max(n_alive, 1)]
            scores = scores[:, :max(n_alive, 1)]

    return beams, scores, None


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

    Note on gradient checkpointing compatibility:
        expert_bias is updated in-place during forward (Loss-Free balancing).
        This makes forward non-idempotent: recomputing forward() produces
        different router decisions → different intermediate tensor shapes → crash.
        Set freeze_bias=True during gradient-checkpointed regions (e.g., DPO
        logprob computation) to prevent bias updates. NTP forward (not
        checkpointed) still updates bias normally.
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
        self.freeze_bias = False  # Set True during gradient-checkpointed forward

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
        # Skipped when freeze_bias=True (gradient checkpointing compatibility):
        # in-place bias update makes forward() non-idempotent, causing shape
        # mismatch when checkpoint recomputes forward during backward.
        if self.training and not self.freeze_bias:
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_indices, self.n_experts).float()
                freq = expert_mask.sum(dim=1).mean(dim=0)  # (n_experts,)
                self.expert_bias.add_(-self.bias_lr * (freq - 1.0 / self.n_experts))

        return output.view(orig_shape)


# ============================================================
# ENTP-Loss helper (shared by NTPModel and NTPProbe)
# ============================================================


def _gather_all(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather tensors across DDP ranks, preserving gradients for local shard."""
    import torch.distributed as dist
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    gathered[dist.get_rank()] = tensor
    return torch.cat(gathered, dim=0)


def _compute_entp_loss(
    hidden_flat: torch.Tensor,
    l0_proj: nn.Linear,
    pos_layer_flat: torch.Tensor,
    mask_flat: torch.Tensor,
    neg_tokens_flat: torch.Tensor,
    neg_mask_flat: torch.Tensor,
) -> torch.Tensor:
    """Compute ENTP penalty: −mean(log(1 − p_L0)) for unclicked exposures.

    Only operates at L0 prediction positions (pos_layer == 0) with valid negatives.

    Args:
        hidden_flat: (B*S, D) — transformer hidden states, flattened.
        l0_proj: output_projs[0] — L0 projection head.
        pos_layer_flat: (B*S,) — which SID layer each position predicts.
        mask_flat: (B*S,) — valid train positions.
        neg_tokens_flat: (B*S, K) — L0 tokens of negatives per position.
        neg_mask_flat: (B*S, K) — True for valid negatives.
    Returns:
        entp_loss: scalar.
    """
    # L0 positions with valid negatives
    l0_mask = mask_flat & (pos_layer_flat == 0)
    has_neg = neg_mask_flat.any(dim=-1)  # (B*S,)
    active = l0_mask & has_neg
    if not active.any():
        return hidden_flat.new_tensor(0.0)

    logits = l0_proj(hidden_flat[active])         # (N, C_l0)
    probs = F.softmax(logits, dim=-1)             # (N, C_l0)

    neg_tok = neg_tokens_flat[active]             # (N, K)
    neg_val = neg_mask_flat[active]               # (N, K)

    # Gather prob of each negative token; clamp -1 pad to 0 for safe indexing
    neg_probs = probs.gather(1, neg_tok.clamp(min=0))  # (N, K)

    penalty = -torch.log(1.0 - neg_probs.clamp(max=1.0 - 1e-6))  # (N, K)
    return (penalty * neg_val).sum() / neg_val.sum().clamp(min=1)


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

    def forward(self, x: torch.Tensor, kv_cache=None, use_cache: bool = False):
        # Pre-norm self-attention
        x_norm = self.norm1(x)

        if kv_cache is not None:
            kv = torch.cat([kv_cache, x_norm], dim=1)
        else:
            kv = x_norm

        if self.causal:
            Q_len = x_norm.size(1)
            KV_len = kv.size(1)
            if Q_len == KV_len:
                # Full sequence (no cache) — standard square mask
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    KV_len, device=x.device,
                )
                attn_out, _ = self.attn(x_norm, kv, kv, attn_mask=attn_mask)
            elif Q_len == 1:
                # Single new token attends to all prior — no mask needed
                attn_out, _ = self.attn(x_norm, kv, kv)
            else:
                # Rectangular causal mask: (Q_len, KV_len)
                attn_mask = torch.full(
                    (Q_len, KV_len), float('-inf'), device=x.device)
                for i in range(Q_len):
                    attn_mask[i, :KV_len - Q_len + i + 1] = 0.0
                attn_out, _ = self.attn(x_norm, kv, kv, attn_mask=attn_mask)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-norm FFN/MoE
        x = x + self.ffn(self.norm2(x))
        if use_cache:
            return x, kv
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
        max_seq_len: int = 0,
        contrastive_dim: int = 0,
        contrastive_item_dim: int = 1024,
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
        # max_seq_len=0 → legacy mode (short sequences only)
        default_len = self.seq_len + n_sid_layers
        self.max_seq_len = max(max_seq_len, default_len)
        self.pos_emb = nn.Embedding(self.max_seq_len, embed_dim)

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

        # Contrastive head (IDEA-onemall-0): project s₃ hidden → align with item embedding
        self.contrastive_dim = contrastive_dim
        if contrastive_dim > 0:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(embed_dim, contrastive_dim),
                nn.ReLU(),
                nn.Linear(contrastive_dim, contrastive_dim),
            )
            self.contrastive_item_proj = nn.Sequential(
                nn.Linear(contrastive_item_dim, contrastive_dim),
                nn.ReLU(),
                nn.Linear(contrastive_dim, contrastive_dim),
            )

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

    # ── KV-cached inference ──

    def _embed_tokens_at_offset(self, tokens: torch.Tensor, offset: int) -> torch.Tensor:
        """Embed tokens where position 0 corresponds to global position ``offset``."""
        B, T = tokens.size()
        device = tokens.device
        L = self.n_sid_layers
        layer_ids = (torch.arange(T, device=device) + offset) % L
        x = torch.zeros(B, T, self.embed_dim, device=device)
        for l in range(L):
            mask = (layer_ids == l)
            if mask.any():
                x[:, mask] = self.token_embs[l](tokens[:, mask])
        return x

    def _transformer_forward_cached(self, x: torch.Tensor, kv_caches=None):
        """Transformer forward with per-layer KV cache.

        Returns:
            out: (B, T_new, D) — output hidden states for new positions only.
            new_kv_caches: list of 6 tensors, each (B, T_cached + T_new, D).
        """
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, kv_cache=cache_i, use_cache=True)
            new_caches.append(new_cache)
        return self.final_norm(x), new_caches

    @torch.no_grad()
    def forward_cached(self, input_tokens=None, generated_tokens=None, kv_caches=None):
        """Inference-only forward with KV cache.

        Calling patterns:
            Cold start: ``forward_cached(input_tokens)`` — encodes full context.
            Incremental: ``forward_cached(generated_tokens=new, kv_caches=kv)``
                — encodes only *new* tokens using cached context.

        Returns:
            logits: (B, C) logits for the next token.
            kv_caches: list of per-layer caches for reuse.
        """
        if kv_caches is None:
            # Cold start — encode full sequence
            tokens = input_tokens
            if generated_tokens is not None and generated_tokens.size(1) > 0:
                tokens = torch.cat([input_tokens, generated_tokens], dim=1)
            T = tokens.size(1)
            device = tokens.device
            x = self._embed_tokens(tokens) + self.pos_emb(
                torch.arange(T, device=device).unsqueeze(0))
            out, kv_caches = self._transformer_forward_cached(x)
        else:
            # Incremental — only new tokens
            offset = kv_caches[0].size(1)
            new_tokens = generated_tokens
            T_new = new_tokens.size(1)
            device = new_tokens.device
            x = self._embed_tokens_at_offset(new_tokens, offset)
            x = x + self.pos_emb(
                torch.arange(offset, offset + T_new, device=device).unsqueeze(0))
            out, kv_caches = self._transformer_forward_cached(x, kv_caches)

        T_total = kv_caches[0].size(1)
        target_layer = T_total % self.n_sid_layers
        logits = self.output_projs[target_layer](out[:, -1, :])
        return logits, kv_caches

    def forward(
        self,
        input_tokens: torch.Tensor,
        generated_tokens: Optional[torch.Tensor] = None,
        return_last_n: int = 1,
        packed_targets: Optional[torch.Tensor] = None,
        packed_mask: Optional[torch.Tensor] = None,
        neg_l0_tokens: Optional[torch.Tensor] = None,
        neg_l0_mask: Optional[torch.Tensor] = None,
        entp_weight: float = 0.0,
        item_embeddings: Optional[torch.Tensor] = None,
        contrastive_weight: float = 0.0,
        contrastive_temp: float = 0.07,
    ):
        """Forward pass — supports both legacy (sliding window) and packed modes.

        Legacy mode (packed_targets=None):
            Same interface as NTPProbe.forward().
        Packed mode (packed_targets provided):
            LM-style causal training on full user sequences.
            Returns scalar loss averaged over valid positions.

        Args:
            input_tokens: (B, T) tokens. Legacy: history SID tokens. Packed: tokens[:, :-1].
            generated_tokens: (B, k) AR mode only (legacy).
            return_last_n: trailing positions for logits (legacy).
            packed_targets: (B, T) shifted targets for packed mode.
            packed_mask: (B, T) bool mask of valid target positions.
            neg_l0_tokens: (B, N_items, K) L0 tokens of unclicked exposures for ENTP loss.
            neg_l0_mask: (B, N_items, K) bool mask (True = valid negative).
            entp_weight: weight α for ENTP loss term.
            item_embeddings: (B, N_items, E) target item embeddings (for contrastive).
            contrastive_weight: weight α for contrastive loss.
            contrastive_temp: InfoNCE temperature τ.
        """
        if packed_targets is not None:
            return self._forward_packed(
                input_tokens, packed_targets, packed_mask,
                neg_l0_tokens=neg_l0_tokens, neg_l0_mask=neg_l0_mask,
                entp_weight=entp_weight,
                item_embeddings=item_embeddings,
                contrastive_weight=contrastive_weight,
                contrastive_temp=contrastive_temp,
            )

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

    def _forward_packed(
        self,
        input_tokens: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        neg_l0_tokens: Optional[torch.Tensor] = None,
        neg_l0_mask: Optional[torch.Tensor] = None,
        entp_weight: float = 0.0,
        item_embeddings: Optional[torch.Tensor] = None,
        contrastive_weight: float = 0.0,
        contrastive_temp: float = 0.07,
    ) -> torch.Tensor:
        """LM-style forward on packed user sequences. Returns scalar loss.

        Args:
            input_tokens: (B, S) — packed tokens (right-padded with 0)
            targets: (B, S) — shifted targets (right-padded, ignored via mask)
            target_mask: (B, S) — True for valid target positions
            neg_l0_tokens: (B, S, K) — L0 tokens of unclicked exposures.
            neg_l0_mask: (B, S, K) — True for valid negative tokens.
            entp_weight: α for ENTP loss.
            item_embeddings: (B, N_items, E) — target item embeddings.
            contrastive_weight: α for contrastive loss.
            contrastive_temp: InfoNCE temperature τ.
        Returns:
            loss: scalar
        """
        B, S = input_tokens.size()
        device = input_tokens.device
        L = self.n_sid_layers

        positions = torch.arange(S, device=device).unsqueeze(0)
        x = self._embed_tokens(input_tokens) + self.pos_emb(positions)
        hidden = self._transformer_forward(x)  # (B, S, D)

        # Flatten for efficient per-layer gather
        hidden_flat = hidden.reshape(-1, self.embed_dim)  # (B*S, D)
        target_flat = targets.reshape(-1)                 # (B*S,)
        mask_flat = target_mask.reshape(-1)                # (B*S,)

        # Position i in hidden predicts target at layer (i+1) % L
        pos_layer = ((torch.arange(S, device=device) + 1) % L)
        pos_layer_flat = pos_layer.unsqueeze(0).expand(B, -1).reshape(-1)

        total_loss = 0.0
        n_active_layers = 0
        for l in range(L):
            layer_mask = mask_flat & (pos_layer_flat == l)
            if not layer_mask.any():
                continue
            logits = self.output_projs[l](hidden_flat[layer_mask])  # (N_l, C_l)
            total_loss += F.cross_entropy(logits, target_flat[layer_mask])
            n_active_layers += 1

        ntp_loss = total_loss / max(n_active_layers, 1)

        # ── ENTP-Loss (optional) ──
        if entp_weight > 0 and neg_l0_tokens is not None:
            entp_loss = _compute_entp_loss(
                hidden_flat, self.output_projs[0],
                pos_layer_flat, mask_flat,
                neg_l0_tokens.reshape(-1, neg_l0_tokens.size(-1)),
                neg_l0_mask.reshape(-1, neg_l0_mask.size(-1)),
            )
            ntp_loss = ntp_loss + entp_weight * entp_loss

        # ── In-Batch Contrastive Loss (IDEA-onemall-0) ──
        if contrastive_weight > 0 and item_embeddings is not None and self.contrastive_dim > 0:
            cl_loss = self._compute_contrastive_loss(
                hidden, target_mask, item_embeddings, contrastive_temp)
            ntp_loss = ntp_loss + contrastive_weight * cl_loss

        return ntp_loss

    def _compute_contrastive_loss(self, hidden, target_mask, item_embeddings, temperature,
                                   sim_matrix_budget_mb=1024):
        """InfoNCE between s₃ hidden states and item embeddings.

        s₃ positions: where input layer = L-1 (position i % L == L-1).
        The hidden state here has encoded the full item SID (s₀..s₃).

        Adaptively caps the number of sampled pairs so the (N_total, N_total)
        similarity matrix fits within `sim_matrix_budget_mb` (default 1 GiB).
        N_total = local_pairs * world_size, matrix memory = N_total² * 4 bytes.
        """
        B, S, D = hidden.shape
        L = self.n_sid_layers
        device = hidden.device
        N_items = item_embeddings.size(1)

        s3_indices = torch.arange(L - 1, S, L, device=device)
        n_s3 = min(len(s3_indices), N_items)
        if n_s3 == 0:
            return torch.tensor(0.0, device=device)

        s3_pos = s3_indices[:n_s3]
        h_s3 = hidden[:, s3_pos, :]          # (B, n_s3, D)
        item_emb = item_embeddings[:, :n_s3, :]  # (B, n_s3, E)

        h_flat = h_s3.reshape(-1, D)          # (B*n_s3, D)
        e_flat = item_emb.reshape(-1, item_emb.size(-1))  # (B*n_s3, E)
        M_local = h_flat.size(0)

        # Adaptive cap: N_total² * 4 ≤ budget → N_total ≤ sqrt(budget/4)
        # N_total = max_local * world_size → max_local = N_total_max / world_size
        ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        n_total_max = int((sim_matrix_budget_mb * 1024 * 1024 / 4) ** 0.5)
        max_local = max(n_total_max // ws, 1)

        if M_local > max_local:
            idx = torch.randperm(M_local, device=device)[:max_local]
            h_flat = h_flat[idx]
            e_flat = e_flat[idx]

        h_proj = self.contrastive_proj(h_flat)
        h_proj = F.normalize(h_proj, dim=-1)
        e_proj = self.contrastive_item_proj(e_flat)
        e_proj = F.normalize(e_proj, dim=-1)

        if torch.distributed.is_initialized():
            h_all = _gather_all(h_proj)
            e_all = _gather_all(e_proj)
        else:
            h_all = h_proj
            e_all = e_proj

        # (N, N) where N ≤ max_pairs * world_size (e.g. 2048*8 = 16K → 1 GiB fp32)
        logits = torch.mm(h_all, e_all.t()) / temperature
        labels = torch.arange(logits.size(0), device=device)
        return F.cross_entropy(logits, labels)

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
