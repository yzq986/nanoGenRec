"""
S-tier NTP Model — 6-layer MoE Transformer decoder with Loss-Free load balancing.

~39.5M total params, ~11M active (top-2 of 8 SwiGLU experts).
Designed for DDP training (ntp/train.py) and eval (ntp/eval.py).

Reference:
  - OneRec (arxiv 2506.13695): SwiGLU MoE architecture
  - DeepSeek / IDEA-onemall-4: Loss-Free dynamic bias MoE balancing
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import math

from ntp.features import REGISTRY as _FEATURE_REGISTRY

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RoPE — Rotary Position Embedding (pluggable multi-dim)
# arxiv 2510.20455 (Roblox), split-by-dim variant
# ============================================================

@dataclass
class RopeDimSpec:
    """Specification for one dimension of a multi-dimensional RoPE."""
    name: str           # "order", "time", "layer" — or any future dim
    split_ratio: float  # fraction of head_dim half-planes to allocate
    source: str         # "position" | "timestamp" | "layer_id"
    base: float = 10000.0    # RoPE frequency base
    max_val: int = 0         # precompute table up to this value
                              # (0 = use max_seq_len for position, n_sid_layers for layer_id)


def build_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    dims: List[RopeDimSpec],
    n_sid_layers: int = 3,
) -> List[dict]:
    """Pre-compute RoPE frequency tensors for a list of RopeDimSpec dimensions.

    Returns a list of dicts, one per dim:
        {"spec": RopeDimSpec, "n_planes": int, "freq": Tensor}
    where:
        source="position" → freq_table shape (max_seq_len, n_planes)
        source="timestamp" → inv_freq shape (n_planes,)  (applied at runtime)
        source="layer_id"  → freq_table shape (n_sid_layers, n_planes)

    Allocation rule:
        - First dim always gets at least 1 plane.
        - Each dim gets floor(half * split_ratio) planes.
        - Last dim gets the remainder (all unallocated planes).
    """
    half = head_dim // 2
    results = []
    allocated = 0

    for i, dim in enumerate(dims):
        is_last = (i == len(dims) - 1)
        n_planes = max(1 if i == 0 else 0, int(half * dim.split_ratio))
        if is_last:
            n_planes = max(n_planes, half - allocated)
        n_planes = min(n_planes, half - allocated)
        allocated += n_planes

        max_v = dim.max_val if dim.max_val > 0 else (
            n_sid_layers if dim.source == 'layer_id' else max_seq_len)

        if n_planes > 0:
            inv_freq = 1.0 / (dim.base ** (
                torch.arange(0, n_planes * 2, 2).float() / (n_planes * 2)))
            if dim.source == 'timestamp':
                freq = inv_freq  # (n_planes,) — applied at runtime with actual values
            else:
                t = torch.arange(max_v).float()
                freq = torch.outer(t, inv_freq)  # (max_v, n_planes)
        else:
            if dim.source == 'timestamp':
                freq = torch.zeros(0)
            else:
                freq = torch.zeros(max_v, 0)

        results.append({"spec": dim, "n_planes": n_planes, "freq": freq})

    return results


def _rotate_segment(
    x_seg: torch.Tensor,   # (B, H, T, n_planes*2)
    angles: torch.Tensor,  # (B, 1, T, n_planes) or (B, H, T, n_planes)
) -> torch.Tensor:
    """Apply RoPE rotation to a segment of x using precomputed angles."""
    def rotate_half(v):
        v1, v2 = v[..., ::2], v[..., 1::2]
        return torch.stack([-v2, v1], dim=-1).flatten(-2)

    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return x_seg * cos + rotate_half(x_seg) * sin


def apply_rope(
    q: torch.Tensor,           # (B, H, T_q, head_dim)
    k: torch.Tensor,           # (B, H, T_k, head_dim)
    dim_inputs_q: List,        # one input tensor per dim for Q
    dim_inputs_k: List,        # one input tensor per dim for K
    rope_info: List[dict],     # from build_rope_freqs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply multi-dim RoPE to Q and K independently.

    dim_inputs: list of tensors, one per entry in rope_info, same order.
        source="position" → integer tensor (B, T) — item-order indices
        source="timestamp" → float tensor (B, T) — hours
        source="layer_id"  → integer tensor (B, T) — layer index 0..L-1
    """
    def _rot_one(x, dim_inputs):
        B, H, T, D = x.shape
        device = x.device
        parts = []
        cursor = 0
        for inp, info in zip(dim_inputs, rope_info):
            n_p = info["n_planes"]
            freq = info["freq"].to(device)
            spec = info["spec"]
            if n_p == 0:
                continue
            x_seg = x[..., cursor:cursor + n_p * 2]
            if spec.source == 'timestamp':
                # inv_freq shape (n_planes,); inp is float (B, T)
                ts = inp.unsqueeze(1).unsqueeze(-1).float()  # (B,1,T,1)
                angles = ts * freq.reshape(1, 1, 1, n_p)    # (B,1,T,n_p)
            else:
                # freq_table shape (max_v, n_planes); inp is integer (B, T)
                flat = inp.expand(B, T).reshape(-1).long().clamp(0, freq.size(0) - 1)
                angles = freq[flat].reshape(B, 1, T, n_p)
            parts.append(_rotate_segment(x_seg, angles))
            cursor += n_p * 2
        if cursor < D:
            parts.append(x[..., cursor:])
        return torch.cat(parts, dim=-1)

    B, H, T_q, _ = q.shape
    T_k = k.shape[2]

    def _fill_zeros(inp, T, device):
        """Replace None with zero tensor of appropriate shape/dtype."""
        if inp is not None:
            return inp
        return torch.zeros(1, T, device=device)

    q_inputs = [_fill_zeros(inp, T_q, q.device) for inp in dim_inputs_q]
    k_inputs = [_fill_zeros(inp, T_k, k.device) for inp in dim_inputs_k]

    def _expand(t, B, T):
        return t.expand(B, T) if t.size(0) == 1 else t

    q_inputs = [_expand(inp, B, T_q) for inp in q_inputs]
    k_inputs = [_expand(inp, B, T_k) for inp in k_inputs]

    return _rot_one(q, q_inputs), _rot_one(k, k_inputs)


# ── Backward-compatible legacy API ───────────────────────────────────────────

def build_torope_freqs(
    head_dim: int,
    max_seq_len: int,
    time_split_ratio: float = 0.5,
    layer_split_ratio: float = 0.0,
    index_base: float = 10000.0,
    time_base: float = 10000.0,
    n_sid_layers: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Legacy wrapper — use build_rope_freqs with RopeDimSpec for new code."""
    half = head_dim // 2
    n_layer_planes = int(half * layer_split_ratio) if layer_split_ratio > 0 else 0
    n_time_planes  = max(1, int(half * time_split_ratio))
    n_idx_planes   = half - n_time_planes - n_layer_planes
    if n_idx_planes < 1:
        n_idx_planes = 1
        n_time_planes = half - n_idx_planes - n_layer_planes

    if n_idx_planes > 0:
        inv_freq_idx = 1.0 / (index_base ** (
            torch.arange(0, n_idx_planes * 2, 2).float() / (n_idx_planes * 2)))
        t = torch.arange(max_seq_len).float()
        freq_idx = torch.outer(t, inv_freq_idx)
    else:
        freq_idx = torch.zeros(max_seq_len, 0)

    if n_time_planes > 0:
        inv_freq_time = 1.0 / (time_base ** (
            torch.arange(0, n_time_planes * 2, 2).float() / (n_time_planes * 2)))
    else:
        inv_freq_time = torch.zeros(0)

    if n_layer_planes > 0:
        inv_freq_layer = 1.0 / (index_base ** (
            torch.arange(0, n_layer_planes * 2, 2).float() / (n_layer_planes * 2)))
        t_layer = torch.arange(n_sid_layers).float()
        freq_layer = torch.outer(t_layer, inv_freq_layer)
    else:
        freq_layer = torch.zeros(n_sid_layers, 0)

    return freq_idx, inv_freq_time, freq_layer, n_idx_planes, n_time_planes, n_layer_planes


def _rotate_with_positions(
    x: torch.Tensor,
    positions: torch.Tensor,
    timestamps: torch.Tensor,
    freq_idx: torch.Tensor,
    inv_freq_time: torch.Tensor,
    freq_layer: torch.Tensor,
    n_idx_planes: int,
    n_time_planes: int,
    n_layer_planes: int,
    layers: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Legacy helper — apply 3-dim RoPE: [index | time | layer]."""
    B, H, T, D = x.shape
    device = x.device

    def rotate_half(v):
        v1, v2 = v[..., ::2], v[..., 1::2]
        return torch.stack([-v2, v1], dim=-1).flatten(-2)

    parts = []
    cursor = 0

    if n_idx_planes > 0:
        freq_idx = freq_idx.to(device)
        pos_flat = positions.expand(B, T).reshape(-1).long().clamp(0, freq_idx.size(0) - 1)
        angles = freq_idx[pos_flat].reshape(B, 1, T, n_idx_planes)
        cos_i = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin_i = torch.sin(angles).repeat_interleave(2, dim=-1)
        xi = x[..., cursor:cursor + n_idx_planes * 2]
        parts.append(xi * cos_i + rotate_half(xi) * sin_i)
        cursor += n_idx_planes * 2

    if n_time_planes > 0:
        inv_f = inv_freq_time.to(device)
        ts = timestamps.unsqueeze(1).unsqueeze(-1).float()
        angles_t = ts * inv_f.reshape(1, 1, 1, n_time_planes)
        cos_t = torch.cos(angles_t).repeat_interleave(2, dim=-1)
        sin_t = torch.sin(angles_t).repeat_interleave(2, dim=-1)
        xt = x[..., cursor:cursor + n_time_planes * 2]
        parts.append(xt * cos_t + rotate_half(xt) * sin_t)
        cursor += n_time_planes * 2

    if n_layer_planes > 0 and layers is not None:
        freq_layer = freq_layer.to(device)
        lay_flat = layers.expand(B, T).reshape(-1).long().clamp(0, freq_layer.size(0) - 1)
        angles_l = freq_layer[lay_flat].reshape(B, 1, T, n_layer_planes)
        cos_l = torch.cos(angles_l).repeat_interleave(2, dim=-1)
        sin_l = torch.sin(angles_l).repeat_interleave(2, dim=-1)
        xl = x[..., cursor:cursor + n_layer_planes * 2]
        parts.append(xl * cos_l + rotate_half(xl) * sin_l)
        cursor += n_layer_planes * 2

    if cursor < D:
        parts.append(x[..., cursor:])

    return torch.cat(parts, dim=-1)


def apply_torope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions_q: torch.Tensor,
    positions_k: torch.Tensor,
    freq_idx: torch.Tensor,
    inv_freq_time: torch.Tensor,
    freq_layer: torch.Tensor,
    n_idx_planes: int,
    n_time_planes: int,
    n_layer_planes: int,
    timestamps_q: Optional[torch.Tensor] = None,
    timestamps_k: Optional[torch.Tensor] = None,
    layers_q: Optional[torch.Tensor] = None,
    layers_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legacy wrapper — use apply_rope for new code."""
    B, H, T_q, D = q.shape
    T_k = k.shape[2]
    device = q.device

    if timestamps_q is None:
        timestamps_q = torch.zeros(positions_q.shape[0], T_q, device=device)
    if timestamps_k is None:
        timestamps_k = torch.zeros(positions_k.shape[0], T_k, device=device)

    def _expand(t, B, T):
        return t.expand(B, T) if t is not None and t.size(0) == 1 else t

    q_rot = _rotate_with_positions(
        q,
        _expand(positions_q, B, T_q),
        _expand(timestamps_q, B, T_q),
        freq_idx, inv_freq_time, freq_layer,
        n_idx_planes, n_time_planes, n_layer_planes,
        layers=_expand(layers_q, B, T_q),
    )
    k_rot = _rotate_with_positions(
        k,
        _expand(positions_k, B, T_k),
        _expand(timestamps_k, B, T_k),
        freq_idx, inv_freq_time, freq_layer,
        n_idx_planes, n_time_planes, n_layer_planes,
        layers=_expand(layers_k, B, T_k),
    )
    return q_rot, k_rot


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


def _build_trie_mask(
    beams: torch.Tensor,    # (B, n_beams, step) CPU tensor of token indices so far
    trie: 'SIDTrie',
    step: int,
    B: int,
    n_beams: int,
    n_vocab: int,
    device,
) -> torch.Tensor:
    """Build boolean trie validity mask of shape (B*n_beams, n_vocab).

    Groups beams by prefix to avoid redundant trie lookups.
    """
    mask = torch.zeros(B * n_beams, n_vocab, dtype=torch.bool, device=device)
    prefix_to_valid: Dict[tuple, List[int]] = {}
    for bi in range(B):
        for ki in range(n_beams):
            pfx = tuple(beams[bi, ki].tolist())
            if pfx not in prefix_to_valid:
                valid = trie.valid_tokens(step, pfx)
                prefix_to_valid[pfx] = list(valid) if valid else []
    for bi in range(B):
        for ki in range(n_beams):
            valid_list = prefix_to_valid[tuple(beams[bi, ki].tolist())]
            if valid_list:
                mask[bi * n_beams + ki, valid_list] = True
    return mask


@torch.no_grad()
def constrained_beam_search(
    model,
    input_tokens: torch.Tensor,
    trie: SIDTrie,
    beam_size: int = 500,
    prefix: torch.Tensor = None,
    ctx_kv_caches=None,
    initial_logits: torch.Tensor = None,
    ctx_side_features: Optional[Dict] = None,
    gen_side_features: Optional[Dict] = None,
    sampling_temperature: float = 0.0,
    ctx_timestamps: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Trie-constrained beam search / sampling with KV cache.

    Every returned candidate is a real SID (guaranteed by SIDTrie masking).

    Args:
        model: NTPModel with forward_cached() for KV-cached inference.
        input_tokens: (B, T) context tokens
        trie: SIDTrie built from sid_to_items
        beam_size: number of candidates to return
        prefix: (B, P) optional fixed prefix tokens.
        ctx_kv_caches: pre-computed KV caches for the context.
        initial_logits: (B, C) logits from the last context position.
        ctx_side_features: dict of context side features, e.g.
            {"time_gaps": (B,T) long, "action_levels": (B,T) long}.
        gen_side_features: dict of scalar side features for generated tokens, e.g.
            {"time_gaps": int, "action_levels": int}.
        sampling_temperature: if > 0, use constrained sampling instead of
            beam search. Each of the beam_size candidates is drawn
            independently from the trie-masked policy distribution at
            temperature T. T=1.0 = policy distribution, T<1 = sharper,
            T>1 = more uniform. sampling_temperature=0 (default) keeps
            the original beam search behaviour.

    Returns:
        beams: (B, actual_beams, n_layers) — token indices
        scores: (B, actual_beams) — cumulative log-probabilities
        ctx_kv_caches: context KV caches (for reuse across passes/items)
    """
    B = input_tokens.size(0)
    device = input_tokens.device
    L = trie.n_layers
    gen_sf = gen_side_features or {}

    # ── Phase 0: context encoding (or reuse) ──
    use_kv_cache = hasattr(model, 'forward_cached')
    if not use_kv_cache:
        # Fallback for NTPProbe or models without forward_cached
        return _constrained_beam_search_legacy(
            model, input_tokens, trie, beam_size, prefix)

    # ctx_kv_pos/ctx_kv_ts track TO-RoPE position/timestamp caches across incremental steps
    ctx_kv_pos = None
    ctx_kv_ts  = None

    if ctx_kv_caches is None:
        initial_logits, ctx_kv_caches, ctx_kv_pos, ctx_kv_ts = model.forward_cached(
            input_tokens, ctx_side_features=ctx_side_features,
            ctx_timestamps=ctx_timestamps)

    # Carry-forward timestamp for generated tokens: use last ctx timestamp
    # (target item timestamp unknown at inference time → assume same as last ctx item)
    gen_timestamp = None
    if model.use_rope and ctx_timestamps is not None:
        gen_timestamp = ctx_timestamps[:, -1:]  # (B, 1) — last ctx token's timestamp

    def _step_sf(n_tok, seq_len=1):
        """Build step side-feature dict via registry; shape (n_tok, seq_len)."""
        sf = {}
        for key, val in gen_sf.items():
            fdef = _FEATURE_REGISTRY.get(key)
            if fdef is None or fdef.inject != 'embed_add':
                continue
            if val is None or not hasattr(model, f'{key}_emb'):
                continue
            dtype = torch.long if fdef.dtype == 'long' else torch.float32
            sf[key] = torch.full((n_tok, seq_len), val, dtype=dtype, device=device)
        return sf or None

    def _step_ts(n_tok):
        """Build step_timestamp for TO-RoPE; carry-forward last ctx timestamp."""
        if gen_timestamp is None:
            return None
        return gen_timestamp.expand(n_tok, -1)  # (n_tok, 1)

    # ── Phase 1: beam init ──
    if prefix is not None:
        P = prefix.size(1)
        beams = prefix.unsqueeze(1)  # (B, 1, P)
        start_step = P
        step_kv = [c.clone() for c in ctx_kv_caches]
        step_kv_pos = ctx_kv_pos
        step_kv_ts  = ctx_kv_ts
        pfx_sf = _step_sf(B, P)
        current_logits, step_kv, step_kv_pos, step_kv_ts = model.forward_cached(
            generated_tokens=prefix, kv_caches=step_kv,
            step_side_features=pfx_sf or None,
            step_timestamp=_step_ts(B),
            kv_positions_cache=step_kv_pos, kv_timestamps_cache=step_kv_ts)
    else:
        beams = torch.zeros(B, 1, 0, dtype=torch.long, device=device)
        start_step = 0
        step_kv = [c.clone() for c in ctx_kv_caches]
        step_kv_pos = ctx_kv_pos
        step_kv_ts  = ctx_kv_ts
        current_logits = initial_logits
    scores = torch.zeros(B, 1, device=device)

    # ── Phase 2: decode ──
    for step in range(start_step, L):
        n_beams = beams.size(1)

        if step > start_step:
            last_tokens = beams[:, :, -1].reshape(B * n_beams, 1)
            sf_step = _step_sf(B * n_beams)
            # All beams share the same positions/timestamps — take first row and expand
            kv_pos_step = step_kv_pos[:1].expand(B * n_beams, -1) if step_kv_pos is not None else None
            kv_ts_step  = step_kv_ts[:1].expand(B * n_beams, -1)  if step_kv_ts  is not None else None
            current_logits, step_kv, step_kv_pos, step_kv_ts = model.forward_cached(
                generated_tokens=last_tokens, kv_caches=step_kv,
                step_side_features=sf_step,
                step_timestamp=_step_ts(B * n_beams),
                kv_positions_cache=kv_pos_step, kv_timestamps_cache=kv_ts_step)

        log_probs = F.log_softmax(
            current_logits.view(B, n_beams, -1), dim=-1)
        C = log_probs.size(-1)

        mask = _build_trie_mask(beams.cpu(), trie, step, B, n_beams, C, device)
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


@torch.no_grad()
def constrained_sampling(
    model,
    input_tokens: torch.Tensor,
    trie: SIDTrie,
    n_samples: int = 64,
    ctx_kv_caches=None,
    initial_logits: torch.Tensor = None,
    ctx_side_features: Optional[Dict] = None,
    gen_side_features: Optional[Dict] = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Trie-constrained ancestral sampling — every candidate is a real SID.

    Generates n_samples independent paths by sampling token-by-token from
    the trie-masked policy distribution. Unlike beam search, each path is
    sampled independently, so importance ratio ρ = π_θ/π_ref ≈ 1 when used
    with an on-policy ref model. This eliminates GRPO clip waste.

    Assumes B=1 (one context at a time, same as constrained_beam_search).

    Args:
        model:             NTPModel with forward_cached().
        input_tokens:      (1, T) context tokens.
        trie:              SIDTrie for validity masking.
        n_samples:         number of independent candidates to draw.
        ctx_kv_caches:     pre-computed context KV caches (or None to encode).
        initial_logits:    (1, C) logits at last context position.
        ctx_side_features: dict of context side features (e.g. time_gaps, action_levels).
        gen_side_features: dict of scalar side features for generated tokens.
        temperature:       softmax temperature. 1.0 = policy distribution,
                           <1 = sharper (less explore), >1 = more uniform.

    Returns:
        beams:  (1, n_unique, n_layers) — deduplicated sampled SIDs
        scores: (1, n_unique) — cumulative log-probs under policy
        ctx_kv_caches: context KV caches for reuse
    """
    device = input_tokens.device
    L = trie.n_layers
    gen_sf = gen_side_features or {}

    # ── Encode context (or reuse cached) ──
    if ctx_kv_caches is None:
        initial_logits, ctx_kv_caches, _, _ = model.forward_cached(
            input_tokens, ctx_side_features=ctx_side_features)

    def _step_sf(n):
        sf = {}
        for key, val in gen_sf.items():
            fdef = _FEATURE_REGISTRY.get(key)
            if fdef is None or fdef.inject != 'embed_add':
                continue
            if val is None or not hasattr(model, f'{key}_emb'):
                continue
            dtype = torch.long if fdef.dtype == 'long' else torch.float32
            sf[key] = torch.full((n, 1), val, dtype=dtype, device=device)
        return sf or None

    # ── Sample n_samples paths independently ──
    # paths[i] = list of token indices, one per layer
    # We batch all n_samples through the model together for efficiency.
    # At each layer, every sample has its own KV cache state (copied from ctx).

    # Expand context KV cache to n_samples copies
    # step_kv: list of (n_samples, T_ctx, D)
    step_kv = [c.expand(n_samples, -1, -1).clone() for c in ctx_kv_caches]

    # current_logits: (n_samples, C) — broadcast initial logits
    cur_logits = initial_logits.expand(n_samples, -1)  # (n_samples, C)

    sampled_tokens = []   # list of (n_samples,) tensors, one per layer
    log_prob_sum = torch.zeros(n_samples, device=device)

    for step in range(L):
        if step > 0:
            last_tok = sampled_tokens[-1].unsqueeze(1)  # (n_samples, 1)
            cur_logits, step_kv, _, _ = model.forward_cached(
                generated_tokens=last_tok, kv_caches=step_kv,
                step_side_features=_step_sf(n_samples))
            # cur_logits: (n_samples, C)

        C = cur_logits.size(-1)

        # Build per-sample trie mask
        # At step 0 all samples share the same prefix (), so one lookup suffices.
        # At step > 0 samples may diverge, so we look up each individually.
        if step == 0:
            valid_list = list(trie.valid_tokens(step, ()))
            if not valid_list:
                # No valid tokens — return empty
                empty = torch.zeros(1, 0, L, dtype=torch.long, device=device)
                empty_s = torch.zeros(1, 0, device=device)
                return empty, empty_s, ctx_kv_caches
            mask = torch.zeros(n_samples, C, dtype=torch.bool, device=device)
            mask[:, valid_list] = True
        else:
            prev = torch.stack(sampled_tokens, dim=1).cpu()  # (n_samples, step)
            mask = torch.zeros(n_samples, C, dtype=torch.bool, device=device)
            for i in range(n_samples):
                pfx = tuple(prev[i].tolist())
                vl = list(trie.valid_tokens(step, pfx))
                if vl:
                    mask[i, vl] = True

        # Masked logits → temperature softmax → sample
        masked_logits = cur_logits.masked_fill(~mask, float('-inf'))
        probs = F.softmax(masked_logits / temperature, dim=-1)

        # Handle any all-masked row (dead path) — sample 0 as placeholder
        dead = ~mask.any(dim=1)
        if dead.any():
            probs[dead] = 0.0
            probs[dead, 0] = 1.0  # placeholder, will be filtered later

        tok = torch.multinomial(probs, num_samples=1).squeeze(1)  # (n_samples,)
        tok_lp = torch.log(probs.gather(1, tok.unsqueeze(1)).squeeze(1) + 1e-40)
        log_prob_sum = log_prob_sum + tok_lp
        log_prob_sum[dead] = float('-inf')   # mark dead paths
        sampled_tokens.append(tok)

    # ── Assemble & deduplicate ──
    paths = torch.stack(sampled_tokens, dim=1)   # (n_samples, L)
    alive = log_prob_sum > float('-inf')
    paths = paths[alive]
    lps   = log_prob_sum[alive]

    if paths.size(0) == 0:
        empty = torch.zeros(1, 0, L, dtype=torch.long, device=device)
        empty_s = torch.zeros(1, 0, device=device)
        return empty, empty_s, ctx_kv_caches

    # Deduplicate identical paths (can occur when T is low)
    unique_paths, inv = torch.unique(paths, dim=0, return_inverse=True)
    # For duplicates keep the log-prob of the first occurrence
    n_unique = unique_paths.size(0)
    unique_lps = torch.full((n_unique,), float('-inf'), device=device)
    for i in range(paths.size(0)):
        uid = inv[i].item()
        if lps[i] > unique_lps[uid]:
            unique_lps[uid] = lps[i]

    beams  = unique_paths.unsqueeze(0)   # (1, n_unique, L)
    scores = unique_lps.unsqueeze(0)     # (1, n_unique)

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

        mask = _build_trie_mask(beams.cpu(), trie, step, B, n_beams, C, device)
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
    """Pre-norm Transformer layer: LayerNorm → Attention → LayerNorm → FFN/MoE.

    When rope_params is provided the layer uses manual SDPA (F.scaled_dot_product_attention)
    and applies RoPE to Q/K before the dot product.  Otherwise falls back to
    nn.MultiheadAttention (no RoPE).
    """

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
        rope_params: Optional[dict] = None,
        use_gate_attn: bool = False,
        # backward compat alias
        torope_params: Optional[dict] = None,
    ):
        super().__init__()
        self.causal = causal
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout
        self.rope_params = rope_params if rope_params is not None else torope_params  # None → standard APE path
        # backward compat: expose as torope_params too
        self.torope_params = self.rope_params

        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True,
        )
        # GateAttention: scalar sigmoid gate on attention output per position
        # gate(x) = sigmoid(W_g x) ∈ (0,1)^D, applied as attn_out * gate(x_norm)
        self.attn_gate = nn.Linear(embed_dim, embed_dim, bias=False) if use_gate_attn else None

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

    def _attn_with_rope(
        self,
        x_norm: torch.Tensor,       # (B, T_q, D)
        kv_src: torch.Tensor,        # (B, T_kv, D)
        positions_q: torch.Tensor,   # (B, T_q) or (1, T_q)
        positions_kv: torch.Tensor,  # (B, T_kv)
        timestamps_q: torch.Tensor,  # (B, T_q)
        timestamps_kv: torch.Tensor, # (B, T_kv)
        layers_q: Optional[torch.Tensor] = None,   # (B, T_q) int 0..L-1
        layers_kv: Optional[torch.Tensor] = None,  # (B, T_kv) int 0..L-1
    ) -> torch.Tensor:
        """Manual SDPA with 3-dim RoPE applied to Q and K."""
        B, T_q, D = x_norm.shape
        T_kv = kv_src.size(1)
        H, Dh = self.n_heads, self.head_dim
        tp = self.rope_params

        # Project Q, K, V
        W, b = self.attn.in_proj_weight, self.attn.in_proj_bias
        q = F.linear(x_norm, W[:D], b[:D] if b is not None else None)
        k = F.linear(kv_src, W[D:2*D], b[D:2*D] if b is not None else None)
        v = F.linear(kv_src, W[2*D:], b[2*D:] if b is not None else None)

        # Reshape to (B, H, T, Dh)
        q = q.view(B, T_q, H, Dh).transpose(1, 2)
        k = k.view(B, T_kv, H, Dh).transpose(1, 2)
        v = v.view(B, T_kv, H, Dh).transpose(1, 2)

        q, k = apply_torope(
            q, k,
            positions_q, positions_kv,
            tp['freq_idx'], tp['inv_freq_time'], tp['freq_layer'],
            tp['n_idx_planes'], tp['n_time_planes'], tp['n_layer_planes'],
            timestamps_q=timestamps_q,
            timestamps_k=timestamps_kv,
            layers_q=layers_q,
            layers_k=layers_kv,
        )

        # Scaled dot-product attention
        is_causal_call = self.causal and (T_q == T_kv)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal_call,
            attn_mask=None,
        )
        # Non-square causal mask (incremental decode with Q_len > 1 unlikely but handle it)
        if self.causal and not is_causal_call and T_q > 1:
            mask = torch.full((T_q, T_kv), float('-inf'), device=x_norm.device)
            for i in range(T_q):
                mask[i, :T_kv - T_q + i + 1] = 0.0
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask.unsqueeze(0).unsqueeze(0),
                dropout_p=self.dropout if self.training else 0.0,
            )

        attn_out = attn_out.transpose(1, 2).reshape(B, T_q, D)
        return F.linear(attn_out, self.attn.out_proj.weight, self.attn.out_proj.bias)

    # backward compat alias
    _attn_with_torope = _attn_with_rope

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
        use_cache: bool = False,
        positions: Optional[torch.Tensor] = None,        # (B,T) item-order indices
        timestamps: Optional[torch.Tensor] = None,       # (B,T) float hours
        layers: Optional[torch.Tensor] = None,           # (B,T) int 0..L-1
        kv_positions: Optional[torch.Tensor] = None,     # (B,T_kv)
        kv_timestamps: Optional[torch.Tensor] = None,    # (B,T_kv)
        kv_layers: Optional[torch.Tensor] = None,        # (B,T_kv)
    ):
        x_norm = self.norm1(x)

        if kv_cache is not None:
            kv = torch.cat([kv_cache, x_norm], dim=1)
        else:
            kv = x_norm

        use_rope = self.rope_params is not None and positions is not None and timestamps is not None

        if use_rope:
            if kv_positions is not None:
                pos_kv = kv_positions
                ts_kv  = kv_timestamps
                lay_kv = kv_layers
            else:
                pos_kv = positions
                ts_kv  = timestamps
                lay_kv = layers
            attn_out = self._attn_with_rope(
                x_norm, kv, positions, pos_kv, timestamps, ts_kv,
                layers_q=layers, layers_kv=lay_kv,
            )
        elif self.causal:
            Q_len = x_norm.size(1)
            KV_len = kv.size(1)
            if Q_len == KV_len:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    KV_len, device=x.device,
                )
                attn_out, _ = self.attn(x_norm, kv, kv, attn_mask=attn_mask)
            elif Q_len == 1:
                attn_out, _ = self.attn(x_norm, kv, kv)
            else:
                attn_mask = torch.full(
                    (Q_len, KV_len), float('-inf'), device=x.device)
                for i in range(Q_len):
                    attn_mask[i, :KV_len - Q_len + i + 1] = 0.0
                attn_out, _ = self.attn(x_norm, kv, kv, attn_mask=attn_mask)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)

        if self.attn_gate is not None:
            attn_out = attn_out * torch.sigmoid(self.attn_gate(x_norm))
        x = x + attn_out
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
        active_features: Optional[List[str]] = None,
        use_segment_emb: bool = False,
        # New RoPE API — takes a list of RopeDimSpec
        rope_dims: Optional[List[RopeDimSpec]] = None,
        # Legacy TO-RoPE API — kept for backward compat / old checkpoints
        use_torope: bool = False,
        torope_time_split: float = 0.5,
        torope_layer_split: float = 0.0,
        use_gate_attn: bool = False,
    ):
        super().__init__()
        assert len(n_clusters_per_layer) == n_sid_layers
        self.n_clusters_per_layer = n_clusters_per_layer
        self.n_sid_layers = n_sid_layers
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.parallel = parallel
        self.seq_len = n_items * n_sid_layers
        self.use_segment_emb = use_segment_emb

        # Resolve rope_dims: new API takes precedence; legacy use_torope auto-converts
        if rope_dims is not None:
            self.rope_dims = rope_dims
        elif use_torope:
            # Convert legacy torope params to RopeDimSpec list
            self.rope_dims = [
                RopeDimSpec(name='order', split_ratio=1.0 - torope_time_split - torope_layer_split,
                            source='position'),
                RopeDimSpec(name='time', split_ratio=torope_time_split, source='timestamp'),
            ]
            if torope_layer_split > 0:
                self.rope_dims.append(
                    RopeDimSpec(name='layer', split_ratio=torope_layer_split, source='layer_id'))
        else:
            self.rope_dims = None

        # Per-layer token embeddings (different codebook per SID layer)
        self.token_embs = nn.ModuleList([
            nn.Embedding(nc, embed_dim) for nc in n_clusters_per_layer
        ])
        # max_seq_len=0 → legacy mode (short sequences only)
        default_len = self.seq_len + n_sid_layers
        self.max_seq_len = max(max_seq_len, default_len)

        # Position embeddings: standard or segment (item_pos + layer_pos)
        # When use_rope=True these are replaced by RoPE (no learnable pos params).
        if self.use_rope:
            head_dim = embed_dim // n_heads
            # Build legacy-style buffers from rope_dims for backward compat
            # (also used by TransformerLayer which expects the old dict format)
            freq_idx, inv_freq_time, freq_layer, n_idx_planes, n_time_planes, n_layer_planes = \
                build_torope_freqs(
                    head_dim, self.max_seq_len,
                    time_split_ratio=torope_time_split if use_torope else
                        next((d.split_ratio for d in self.rope_dims if d.source == 'timestamp'), 0.5),
                    layer_split_ratio=torope_layer_split if use_torope else
                        next((d.split_ratio for d in self.rope_dims if d.source == 'layer_id'), 0.0),
                    n_sid_layers=n_sid_layers,
                )
            # Register under both old and new buffer names for full compat
            self.register_buffer('torope_freq_idx', freq_idx)
            self.register_buffer('torope_inv_freq_time', inv_freq_time)
            self.register_buffer('torope_freq_layer', freq_layer)
            # New indexed names
            self.register_buffer('rope_dim_0_freq', freq_idx)
            self.register_buffer('rope_dim_1_freq', inv_freq_time)
            self.register_buffer('rope_dim_2_freq', freq_layer)
            self.torope_n_idx_planes   = n_idx_planes
            self.torope_n_time_planes  = n_time_planes
            self.torope_n_layer_planes = n_layer_planes
            # Also store split ratios (used in probe_config serialization)
            _ts = torope_time_split if use_torope else \
                next((d.split_ratio for d in self.rope_dims if d.source == 'timestamp'), 0.5)
            _ls = torope_layer_split if use_torope else \
                next((d.split_ratio for d in self.rope_dims if d.source == 'layer_id'), 0.0)
            self.torope_time_split  = _ts
            self.torope_layer_split = _ls
        else:
            if use_segment_emb:
                max_n_items = self.max_seq_len // n_sid_layers + 1
                self.item_pos_emb = nn.Embedding(max_n_items, embed_dim)
                self.layer_pos_emb = nn.Embedding(n_sid_layers, embed_dim)
            else:
                self.pos_emb = nn.Embedding(self.max_seq_len, embed_dim)

        # Side feature embeddings — auto-created from registry for each active feature.
        self.active_features: List[str] = list(active_features) if active_features else []
        for key in self.active_features:
            fdef = _FEATURE_REGISTRY.get(key)
            if fdef is not None and fdef.inject == 'embed_add' and fdef.emb_size > 0:
                setattr(self, f'{key}_emb', nn.Embedding(fdef.emb_size, embed_dim))

        rope_layer_params = None
        if self.use_rope:
            rope_layer_params = {
                'freq_idx':       self.torope_freq_idx,
                'inv_freq_time':  self.torope_inv_freq_time,
                'freq_layer':     self.torope_freq_layer,
                'n_idx_planes':   n_idx_planes,
                'n_time_planes':  n_time_planes,
                'n_layer_planes': n_layer_planes,
            }

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim, n_heads, dropout,
                use_moe=use_moe, n_experts=n_experts,
                top_k=top_k, expert_dim=expert_dim,
                causal=not parallel,
                rope_params=rope_layer_params,
                use_gate_attn=use_gate_attn,
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

    @property
    def use_rope(self) -> bool:
        """True if this model uses RoPE (new or legacy API)."""
        return self.rope_dims is not None

    @property
    def use_torope(self) -> bool:
        """Backward compat alias for use_rope."""
        return self.use_rope

    def _build_rope_inputs(self, T: int, device, offset: int = 0):
        """Return (pos, timestamps, layers) for transformer forward — zero-overhead if not rope.

        All forward paths call this instead of computing rope params inline,
        so adding/changing RoPE variants only requires editing this one method.
        """
        if not self.use_rope:
            return None, None, None
        L = self.n_sid_layers
        pos_raw = torch.arange(offset, offset + T, device=device).unsqueeze(0)
        return pos_raw // L, torch.zeros(1, T, device=device), pos_raw % L

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

    def _get_pos_emb(self, positions: torch.Tensor) -> torch.Tensor:
        """Get positional embedding for given positions (standard or segment).
        Returns zeros when use_rope=True (position info lives in RoPE instead)."""
        if self.use_rope:
            return torch.zeros(*positions.shape, self.embed_dim,
                               device=positions.device, dtype=torch.get_default_dtype())
        if self.use_segment_emb:
            L = self.n_sid_layers
            item_pos = positions // L
            layer_pos = positions % L
            return self.item_pos_emb(item_pos) + self.layer_pos_emb(layer_pos)
        else:
            return self.pos_emb(positions)

    def _apply_embed_add_sf(self, x: torch.Tensor, sf: dict) -> torch.Tensor:
        """Add registered embed_add features to an already-embedded tensor."""
        for key, val in sf.items():
            fdef = _FEATURE_REGISTRY.get(key)
            if fdef is None or fdef.inject != 'embed_add':
                continue
            emb = getattr(self, f'{key}_emb', None)
            if emb is not None and val is not None:
                x = x + emb(val)
        return x

    def embed_with_features(
        self,
        tokens: torch.Tensor,        # (B, T)
        positions: torch.Tensor,     # (B, T) or (1, T)
        side_features: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Single source of truth for input embedding + side features injection.

        All training (forward) and inference (forward_cached, compute_sid_logprobs)
        paths must call this instead of combining _embed_tokens + _get_pos_emb +
        feature embeddings manually.  Adding a new side feature means editing
        exactly this one function.

        side_features: dict[str, Tensor] — any registered 'embed_add' features.
            'rope' features (timestamps) are handled in _forward_packed, not here.
        """
        x = self._embed_tokens(tokens) + self._get_pos_emb(positions)
        return self._apply_embed_add_sf(x, side_features or {})

    def _transformer_forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        layers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, positions=positions, timestamps=timestamps, layers=layers)
        return self.final_norm(x)

    def forward_tf(
        self,
        tokens: torch.Tensor,
        side_features: Optional[dict] = None,
    ) -> torch.Tensor:
        """Teacher-forced forward: embed tokens + run transformer with correct RoPE params.

        Single call site for all teacher-forced eval. Handles torope position/timestamp/layer
        internally so callers never need to know about n_sid_layers or torope flags.
        """
        B, T = tokens.size()
        device = tokens.device
        pos_raw = torch.arange(T, device=device).unsqueeze(0)
        x = self.embed_with_features(tokens, pos_raw, side_features)
        tf_pos, tf_ts, tf_lay = self._build_rope_inputs(T, device)
        return self._transformer_forward(x, positions=tf_pos, timestamps=tf_ts, layers=tf_lay)

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

    def _transformer_forward_cached(
        self,
        x: torch.Tensor,
        kv_caches=None,
        positions: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        layers: Optional[torch.Tensor] = None,
        kv_positions: Optional[torch.Tensor] = None,
        kv_timestamps: Optional[torch.Tensor] = None,
        kv_layers: Optional[torch.Tensor] = None,
    ):
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(
                x, kv_cache=cache_i, use_cache=True,
                positions=positions, timestamps=timestamps, layers=layers,
                kv_positions=kv_positions, kv_timestamps=kv_timestamps, kv_layers=kv_layers,
            )
            new_caches.append(new_cache)
        return self.final_norm(x), new_caches

    @torch.no_grad()
    def forward_cached(self, input_tokens=None, generated_tokens=None, kv_caches=None,
                        ctx_side_features=None, step_side_features=None,
                        ctx_timestamps=None, step_timestamp=None,
                        kv_positions_cache=None, kv_timestamps_cache=None):
        """Inference-only forward with KV cache.

        Calling patterns:
            Cold start: ``forward_cached(input_tokens)`` — encodes full context.
            Incremental: ``forward_cached(generated_tokens=new, kv_caches=kv)``
                — encodes only *new* tokens using cached context.

        Args:
            ctx_side_features: dict of side features for context tokens, e.g.
                {"time_gaps": (B,T_ctx) long, "action_levels": (B,T_ctx) long}.
            step_side_features: dict of side features for new (generated) tokens.
            ctx_timestamps: (B, T_ctx) float timestamps in hours (RoPE path).
            step_timestamp: (B, T_new) float timestamps for new tokens (RoPE path).
            kv_positions_cache: (B, T_kv) positions already in KV cache (RoPE incremental).
            kv_timestamps_cache: (B, T_kv) timestamps already in KV cache (RoPE incremental).

        Returns:
            logits: (B, C) logits for the next token.
            kv_caches: list of per-layer caches for reuse.
            (kv_positions, kv_timestamps) — updated position/timestamp caches (RoPE only).
        """
        ctx_sf = ctx_side_features or {}
        step_sf = step_side_features or {}

        if kv_caches is None:
            # Cold start — encode full sequence
            tokens = input_tokens
            if generated_tokens is not None and generated_tokens.size(1) > 0:
                tokens = torch.cat([input_tokens, generated_tokens], dim=1)
            T = tokens.size(1)
            device = tokens.device
            pos_raw = torch.arange(T, device=device).unsqueeze(0)
            positions = pos_raw  # used for APE token embedding lookup
            rope_pos, _, rope_lay = self._build_rope_inputs(T, device)

            # Pad each ctx side feature to full sequence length
            padded_sf = {}
            for key, feat in ctx_sf.items():
                if feat is not None:
                    T_ctx = feat.size(1)
                    if T_ctx < T:
                        pad = torch.zeros(feat.size(0), T - T_ctx,
                                          dtype=feat.dtype, device=device)
                        padded_sf[key] = torch.cat([feat, pad], dim=1)
                    else:
                        padded_sf[key] = feat
            x = self.embed_with_features(tokens, positions, padded_sf)

            # RoPE: build full timestamp tensor for context
            if self.use_rope:
                if ctx_timestamps is not None:
                    T_ctx = ctx_timestamps.size(1)
                    ts_pad = torch.zeros(ctx_timestamps.size(0), T - T_ctx,
                                         dtype=torch.float, device=device)
                    timestamps = torch.cat([ctx_timestamps.float(), ts_pad], dim=1)
                else:
                    timestamps = torch.zeros(tokens.size(0), T,
                                             dtype=torch.float, device=device)
            else:
                timestamps = None

            out, kv_caches = self._transformer_forward_cached(
                x, positions=rope_pos, timestamps=timestamps, layers=rope_lay)

            new_kv_pos = rope_pos if self.use_rope else None
            new_kv_ts  = timestamps if self.use_rope else None
            new_kv_lay = rope_lay  if self.use_rope else None
        else:
            # Incremental — only new tokens
            offset = kv_caches[0].size(1)
            new_tokens = generated_tokens
            T_new = new_tokens.size(1)
            device = new_tokens.device
            x = self._embed_tokens_at_offset(new_tokens, offset)
            pos_raw_inc = torch.arange(offset, offset + T_new, device=device).unsqueeze(0)
            positions = pos_raw_inc  # for APE (not used with rope)
            x = x + self._get_pos_emb(positions)
            x = self._apply_embed_add_sf(x, step_sf)

            step_pos, _, step_lay = self._build_rope_inputs(T_new, device, offset=offset)
            if self.use_rope:
                ts_new = step_timestamp.float() if step_timestamp is not None \
                    else torch.zeros(new_tokens.size(0), T_new, device=device)
                if kv_positions_cache is not None:
                    B_kv = kv_positions_cache.size(0)
                    full_pos = torch.cat([kv_positions_cache,
                                          step_pos.expand(B_kv, -1)], dim=1)
                    kv_ts_base = kv_timestamps_cache.float() if kv_timestamps_cache is not None \
                        else torch.zeros(B_kv, offset, device=device)
                    full_ts = torch.cat([kv_ts_base, ts_new.expand(B_kv, -1)], dim=1)
                    kv_pos_base, _, kv_lay_base = self._build_rope_inputs(offset, device)
                    full_lay = torch.cat([kv_lay_base.expand(B_kv, -1),
                                          step_lay.expand(B_kv, -1)], dim=1)
                else:
                    full_pos = step_pos
                    full_ts  = ts_new
                    full_lay = step_lay
                out, kv_caches = self._transformer_forward_cached(
                    x, kv_caches,
                    positions=step_pos, timestamps=ts_new, layers=step_lay,
                    kv_positions=full_pos, kv_timestamps=full_ts, kv_layers=full_lay,
                )
                new_kv_pos = full_pos
                new_kv_ts  = full_ts
                new_kv_lay = full_lay
            else:
                out, kv_caches = self._transformer_forward_cached(x, kv_caches)
                new_kv_pos = new_kv_ts = new_kv_lay = None

        T_total = kv_caches[0].size(1)
        target_layer = T_total % self.n_sid_layers
        logits = self.output_projs[target_layer](out[:, -1, :])
        return logits, kv_caches, new_kv_pos, new_kv_ts

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
        side_features: Optional[Dict] = None,
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
            side_features: dict of optional side feature tensors, e.g.
                {"time_gaps": (B,T) long, "action_levels": (B,T) long,
                 "timestamps": (B,T) float hours}.
        """
        if packed_targets is not None:
            return self._forward_packed(
                input_tokens, packed_targets, packed_mask,
                neg_l0_tokens=neg_l0_tokens, neg_l0_mask=neg_l0_mask,
                entp_weight=entp_weight,
                item_embeddings=item_embeddings,
                contrastive_weight=contrastive_weight,
                contrastive_temp=contrastive_temp,
                side_features=side_features,
            )

        device = input_tokens.device

        if self.parallel:
            positions = torch.arange(self.seq_len, device=device).unsqueeze(0)
            x = self._embed_tokens(input_tokens) + self._get_pos_emb(positions)
            tp_pos, tp_ts, tp_lay = self._build_rope_inputs(self.seq_len, device)
            out = self._transformer_forward(x, positions=tp_pos, timestamps=tp_ts, layers=tp_lay)
            s = out[:, -1, :]
            return [self.output_projs[l](s) for l in range(self.n_sid_layers)]

        # Autoregressive
        if generated_tokens is not None and generated_tokens.size(1) > 0:
            tokens = torch.cat([input_tokens, generated_tokens], dim=1)
        else:
            tokens = input_tokens

        T = tokens.size(1)
        pos_raw = torch.arange(T, device=device).unsqueeze(0)
        x = self._embed_tokens(tokens) + self._get_pos_emb(pos_raw)
        rope_pos, rope_ts, rope_lay = self._build_rope_inputs(T, device)
        out = self._transformer_forward(x, positions=rope_pos, timestamps=rope_ts, layers=rope_lay)

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
        side_features: Optional[Dict] = None,
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
            side_features: dict of optional tensors, e.g.
                {"time_gaps": (B,S) long, "action_levels": (B,S) long,
                 "timestamps": (B,S) float hours for TO-RoPE}.
        Returns:
            loss: scalar
        """
        sf = side_features or {}
        B, S = input_tokens.size()
        device = input_tokens.device
        L = self.n_sid_layers

        pos_raw = torch.arange(S, device=device).unsqueeze(0)
        positions = pos_raw
        x = self.embed_with_features(input_tokens, positions, sf)
        tp_pos, _, tp_lay = self._build_rope_inputs(S, device)
        if self.use_rope:
            raw_ts = sf.get('timestamps')
            tp_ts = raw_ts if raw_ts is not None else torch.zeros(1, S, device=device)
        else:
            tp_ts = None
        hidden = self._transformer_forward(x, positions=tp_pos, timestamps=tp_ts, layers=tp_lay)

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
                                   max_pairs=2048):
        """InfoNCE between s₃ hidden states and item embeddings (local in-batch).

        s₃ positions: where input layer = L-1 (position i % L == L-1).
        The hidden state here has encoded the full item SID (s₀..s₃).

        Uses only local (per-GPU) negatives to avoid OOM from cross-GPU gather.
        Each GPU samples up to max_pairs from its B×n_s3 pool.
        Memory: (max_pairs, max_pairs) * 4 bytes * ~3 (fwd+bwd+softmax)
        = 2048² * 12 ≈ 48 MB peak.
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

        if M_local > max_pairs:
            idx = torch.randperm(M_local, device=device)[:max_pairs]
            h_flat = h_flat[idx]
            e_flat = e_flat[idx]

        h_proj = self.contrastive_proj(h_flat)
        h_proj = F.normalize(h_proj, dim=-1)
        e_proj = self.contrastive_item_proj(e_flat)
        e_proj = F.normalize(e_proj, dim=-1)

        logits = torch.mm(h_proj, e_proj.t()) / temperature  # (N, N), N ≤ 4096
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
