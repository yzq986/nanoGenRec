"""
S-tier NTP Model — 6-layer MoE Transformer decoder with Loss-Free load balancing.

~39.5M total params, ~11M active (top-2 of 8 SwiGLU experts).
Designed for DDP training (ntp/train.py) and eval (ntp/eval.py).

Reference:
  - OneRec (arxiv 2506.13695): SwiGLU MoE architecture
  - DeepSeek / IDEA-onemall-4: Loss-Free dynamic bias MoE balancing
"""

from typing import Dict, List, Optional, Set, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TO-RoPE — Time-and-Order Rotary Position Embedding
# arxiv 2510.20455 (Roblox), split-by-dim variant
# ============================================================

def build_torope_freqs(
    head_dim: int,
    max_seq_len: int,
    time_split_ratio: float = 0.5,
    index_base: float = 10000.0,
    time_base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute TO-RoPE frequency tensors.

    head_dim must be even. The first (1-time_split_ratio) fraction of rotary
    planes encodes sequence index; the remaining fraction encodes wall-clock time.

    Returns:
        freq_idx:  (max_seq_len, head_dim//2) — angle per (position, plane) for index dim
        freq_time: (max_seq_len, head_dim//2) — angle per (position, plane) for time dim
        Both are pre-multiplied so that apply_rotary_emb only needs to index by position.
        The two tensors are *logically* used via split_pos below.
    """
    half = head_dim // 2
    n_time_planes = max(1, int(half * time_split_ratio))
    n_idx_planes = half - n_time_planes

    # Index planes: standard RoPE frequencies
    if n_idx_planes > 0:
        inv_freq_idx = 1.0 / (index_base ** (
            torch.arange(0, n_idx_planes * 2, 2).float() / (n_idx_planes * 2)))
        t = torch.arange(max_seq_len).float()
        freq_idx = torch.outer(t, inv_freq_idx)  # (max_seq_len, n_idx_planes)
    else:
        freq_idx = torch.zeros(max_seq_len, 0)

    # Time planes: RoPE with time_base (same formula, applied to τ at runtime)
    if n_time_planes > 0:
        inv_freq_time = 1.0 / (time_base ** (
            torch.arange(0, n_time_planes * 2, 2).float() / (n_time_planes * 2)))
    else:
        inv_freq_time = torch.zeros(0)

    return freq_idx, inv_freq_time, n_idx_planes, n_time_planes


def _rotate_with_positions(
    x: torch.Tensor,            # (B, H, T, head_dim)
    positions: torch.Tensor,    # (B, T) integer indices
    timestamps: torch.Tensor,   # (B, T) float hours
    freq_idx: torch.Tensor,     # (max_len, n_idx_planes)
    inv_freq_time: torch.Tensor,  # (n_time_planes,)
    n_idx_planes: int,
    n_time_planes: int,
) -> torch.Tensor:
    """Apply TO-RoPE rotation to a single tensor x given its positions and timestamps."""
    B, H, T, D = x.shape
    device = x.device

    def rotate_half(v):
        v1, v2 = v[..., ::2], v[..., 1::2]
        return torch.stack([-v2, v1], dim=-1).flatten(-2)

    parts = []
    if n_idx_planes > 0:
        freq_idx = freq_idx.to(device)
        pos_exp = positions.expand(B, T)
        pos_flat = pos_exp.reshape(-1).long().clamp(0, freq_idx.size(0) - 1)
        angles = freq_idx[pos_flat].reshape(B, 1, T, n_idx_planes)
        cos_i = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin_i = torch.sin(angles).repeat_interleave(2, dim=-1)
        xi = x[..., :n_idx_planes * 2]
        parts.append(xi * cos_i + rotate_half(xi) * sin_i)

    if n_time_planes > 0:
        inv_f = inv_freq_time.to(device)
        ts = timestamps.unsqueeze(1).unsqueeze(-1).float()  # (B,1,T,1)
        angles_t = ts * inv_f.reshape(1, 1, 1, n_time_planes)
        cos_t = torch.cos(angles_t).repeat_interleave(2, dim=-1)
        sin_t = torch.sin(angles_t).repeat_interleave(2, dim=-1)
        start = n_idx_planes * 2
        xt = x[..., start:start + n_time_planes * 2]
        parts.append(xt * cos_t + rotate_half(xt) * sin_t)

    total_rotated = (n_idx_planes + n_time_planes) * 2
    if total_rotated < D:
        parts.append(x[..., total_rotated:])

    return torch.cat(parts, dim=-1)


def apply_torope(
    q: torch.Tensor,            # (B, H, T_q, head_dim)
    k: torch.Tensor,            # (B, H, T_k, head_dim)
    positions_q: torch.Tensor,  # (B, T_q) or (1, T_q) — integer sequence indices for Q
    positions_k: torch.Tensor,  # (B, T_k) — integer sequence indices for K
    freq_idx: torch.Tensor,     # (max_seq_len, n_idx_planes) — precomputed
    inv_freq_time: torch.Tensor,  # (n_time_planes,)
    n_idx_planes: int,
    n_time_planes: int,
    timestamps_q: Optional[torch.Tensor] = None,  # (B, T_q) float hours — defaults to zeros
    timestamps_k: Optional[torch.Tensor] = None,  # (B, T_k) float hours — defaults to zeros
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply TO-RoPE (split-by-dim) to Q and K independently.

    Q and K may have different sequence lengths (cross-attention / KV-cache incremental).
    Index planes use precomputed freq_idx; time planes compute angles from timestamps
    * inv_freq_time at runtime.
    """
    B, H, T_q, D = q.shape
    T_k = k.shape[2]
    device = q.device

    if timestamps_q is None:
        timestamps_q = torch.zeros(positions_q.shape[0], T_q, device=device)
    if timestamps_k is None:
        timestamps_k = torch.zeros(positions_k.shape[0], T_k, device=device)

    q_rot = _rotate_with_positions(
        q, positions_q.expand(B, -1) if positions_q.size(0) == 1 else positions_q,
        timestamps_q.expand(B, -1) if timestamps_q.size(0) == 1 else timestamps_q,
        freq_idx, inv_freq_time, n_idx_planes, n_time_planes,
    )
    k_rot = _rotate_with_positions(
        k, positions_k.expand(B, -1) if positions_k.size(0) == 1 else positions_k,
        timestamps_k.expand(B, -1) if timestamps_k.size(0) == 1 else timestamps_k,
        freq_idx, inv_freq_time, n_idx_planes, n_time_planes,
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


@torch.no_grad()
def constrained_beam_search(
    model,
    input_tokens: torch.Tensor,
    trie: SIDTrie,
    beam_size: int = 500,
    prefix: torch.Tensor = None,
    ctx_kv_caches=None,
    initial_logits: torch.Tensor = None,
    ctx_time_gaps: torch.Tensor = None,
    ctx_action_levels: torch.Tensor = None,
    gen_time_gap: int = None,
    gen_action_level: int = None,
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
        ctx_time_gaps: (B, T) time gap buckets for context tokens.
        ctx_action_levels: (B, T) action levels for context tokens.
        gen_time_gap: scalar int, time_gap bucket for generated tokens.
        gen_action_level: scalar int, action_level for generated tokens.
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
            input_tokens, ctx_time_gaps=ctx_time_gaps,
            ctx_action_levels=ctx_action_levels,
            ctx_timestamps=ctx_timestamps)

    # ── Build step feature tensors for generated tokens ──
    _has_tg = gen_time_gap is not None and hasattr(model, 'time_gap_emb')
    _has_al = gen_action_level is not None and hasattr(model, 'action_emb')

    def _step_features(n_tok):
        """Return step_time_gap, step_action_level tensors for n_tok tokens."""
        stg = torch.full((n_tok, 1), gen_time_gap, dtype=torch.long,
                         device=device) if _has_tg else None
        sal = torch.full((n_tok, 1), gen_action_level, dtype=torch.long,
                         device=device) if _has_al else None
        return stg, sal

    # ── Phase 1: beam init ──
    if prefix is not None:
        P = prefix.size(1)
        beams = prefix.unsqueeze(1)  # (B, 1, P)
        start_step = P
        step_kv = [c.clone() for c in ctx_kv_caches]
        step_kv_pos = ctx_kv_pos
        step_kv_ts  = ctx_kv_ts
        stg_pfx = torch.full((B, P), gen_time_gap, dtype=torch.long,
                             device=device) if _has_tg else None
        sal_pfx = torch.full((B, P), gen_action_level, dtype=torch.long,
                             device=device) if _has_al else None
        current_logits, step_kv, step_kv_pos, step_kv_ts = model.forward_cached(
            generated_tokens=prefix, kv_caches=step_kv,
            step_time_gap=stg_pfx, step_action_level=sal_pfx,
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
            stg_step, sal_step = _step_features(B * n_beams)
            # All beams share the same positions — take first row and expand
            kv_pos_step = step_kv_pos[:1].expand(B * n_beams, -1) if step_kv_pos is not None else None
            kv_ts_step  = step_kv_ts[:1].expand(B * n_beams, -1)  if step_kv_ts  is not None else None
            current_logits, step_kv, step_kv_pos, step_kv_ts = model.forward_cached(
                generated_tokens=last_tokens, kv_caches=step_kv,
                step_time_gap=stg_step, step_action_level=sal_step,
                kv_positions_cache=kv_pos_step, kv_timestamps_cache=kv_ts_step)

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


@torch.no_grad()
def constrained_sampling(
    model,
    input_tokens: torch.Tensor,
    trie: SIDTrie,
    n_samples: int = 64,
    ctx_kv_caches=None,
    initial_logits: torch.Tensor = None,
    ctx_time_gaps: torch.Tensor = None,
    ctx_action_levels: torch.Tensor = None,
    gen_time_gap: int = None,
    gen_action_level: int = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Trie-constrained ancestral sampling — every candidate is a real SID.

    Generates n_samples independent paths by sampling token-by-token from
    the trie-masked policy distribution. Unlike beam search, each path is
    sampled independently, so importance ratio ρ = π_θ/π_ref ≈ 1 when used
    with an on-policy ref model. This eliminates GRPO clip waste.

    Assumes B=1 (one context at a time, same as constrained_beam_search).

    Args:
        model:          NTPModel with forward_cached().
        input_tokens:   (1, T) context tokens.
        trie:           SIDTrie for validity masking.
        n_samples:      number of independent candidates to draw.
        ctx_kv_caches:  pre-computed context KV caches (or None to encode).
        initial_logits: (1, C) logits at last context position.
        ctx_time_gaps:  (1, T) time gap buckets for context tokens.
        ctx_action_levels: (1, T) action levels for context tokens.
        gen_time_gap:   scalar int, time_gap for generated tokens.
        gen_action_level: scalar int, action_level for generated tokens.
        temperature:    softmax temperature. 1.0 = policy distribution,
                        <1 = sharper (less explore), >1 = more uniform.

    Returns:
        beams:  (1, n_unique, n_layers) — deduplicated sampled SIDs
        scores: (1, n_unique) — cumulative log-probs under policy
        ctx_kv_caches: context KV caches for reuse
    """
    device = input_tokens.device
    L = trie.n_layers

    # ── Encode context (or reuse cached) ──
    if ctx_kv_caches is None:
        initial_logits, ctx_kv_caches, _, _ = model.forward_cached(
            input_tokens, ctx_time_gaps=ctx_time_gaps,
            ctx_action_levels=ctx_action_levels)

    _has_tg = gen_time_gap is not None and hasattr(model, 'time_gap_emb')
    _has_al = gen_action_level is not None and hasattr(model, 'action_emb')

    def _feat(n):
        stg = torch.full((n, 1), gen_time_gap, dtype=torch.long,
                         device=device) if _has_tg else None
        sal = torch.full((n, 1), gen_action_level, dtype=torch.long,
                         device=device) if _has_al else None
        return stg, sal

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
            stg, sal = _feat(n_samples)
            cur_logits, step_kv, _, _ = model.forward_cached(
                generated_tokens=last_tok, kv_caches=step_kv,
                step_time_gap=stg, step_action_level=sal)
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
    """Pre-norm Transformer layer: LayerNorm → Attention → LayerNorm → FFN/MoE.

    When torope_params is provided the layer uses manual SDPA (F.scaled_dot_product_attention)
    and applies TO-RoPE to Q/K before the dot product.  Otherwise falls back to
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
        torope_params: Optional[dict] = None,
    ):
        super().__init__()
        self.causal = causal
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout
        self.torope_params = torope_params  # None → standard APE path

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

    def _attn_with_torope(
        self,
        x_norm: torch.Tensor,       # (B, T_q, D)
        kv_src: torch.Tensor,        # (B, T_kv, D)
        positions_q: torch.Tensor,   # (B, T_q) or (1, T_q)
        positions_kv: torch.Tensor,  # (B, T_kv)
        timestamps_q: torch.Tensor,  # (B, T_q)
        timestamps_kv: torch.Tensor, # (B, T_kv)
    ) -> torch.Tensor:
        """Manual SDPA with TO-RoPE applied to Q and K."""
        B, T_q, D = x_norm.shape
        T_kv = kv_src.size(1)
        H, Dh = self.n_heads, self.head_dim
        tp = self.torope_params

        # Project Q, K, V
        W, b = self.attn.in_proj_weight, self.attn.in_proj_bias
        q = F.linear(x_norm, W[:D], b[:D] if b is not None else None)
        k = F.linear(kv_src, W[D:2*D], b[D:2*D] if b is not None else None)
        v = F.linear(kv_src, W[2*D:], b[2*D:] if b is not None else None)

        # Reshape to (B, H, T, Dh)
        q = q.view(B, T_q, H, Dh).transpose(1, 2)
        k = k.view(B, T_kv, H, Dh).transpose(1, 2)
        v = v.view(B, T_kv, H, Dh).transpose(1, 2)

        # Apply TO-RoPE independently to Q (T_q tokens) and K (T_kv tokens)
        q, k = apply_torope(
            q, k,
            positions_q, positions_kv,
            tp['freq_idx'], tp['inv_freq_time'],
            tp['n_idx_planes'], tp['n_time_planes'],
            timestamps_q=timestamps_q,
            timestamps_k=timestamps_kv,
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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
        use_cache: bool = False,
        positions: Optional[torch.Tensor] = None,        # (B,T) or (1,T) — for TO-RoPE
        timestamps: Optional[torch.Tensor] = None,       # (B,T) — for TO-RoPE
        kv_positions: Optional[torch.Tensor] = None,     # (B,T_kv) — KV cache positions
        kv_timestamps: Optional[torch.Tensor] = None,    # (B,T_kv) — KV cache timestamps
    ):
        x_norm = self.norm1(x)

        if kv_cache is not None:
            kv = torch.cat([kv_cache, x_norm], dim=1)
        else:
            kv = x_norm

        use_torope = self.torope_params is not None and positions is not None and timestamps is not None

        if use_torope:
            # Positions and timestamps for the full KV sequence
            if kv_positions is not None:
                pos_kv = kv_positions
                ts_kv = kv_timestamps
            else:
                pos_kv = positions
                ts_kv = timestamps
            attn_out = self._attn_with_torope(x_norm, kv, positions, pos_kv, timestamps, ts_kv)
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
        n_time_buckets: int = 0,
        n_action_levels: int = 0,
        use_segment_emb: bool = False,
        use_torope: bool = False,
        torope_time_split: float = 0.5,
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
        self.use_torope = use_torope

        # Per-layer token embeddings (different codebook per SID layer)
        self.token_embs = nn.ModuleList([
            nn.Embedding(nc, embed_dim) for nc in n_clusters_per_layer
        ])
        # max_seq_len=0 → legacy mode (short sequences only)
        default_len = self.seq_len + n_sid_layers
        self.max_seq_len = max(max_seq_len, default_len)

        # Position embeddings: standard or segment (item_pos + layer_pos)
        # When use_torope=True these are replaced by RoPE (no learnable pos params).
        if use_torope:
            head_dim = embed_dim // n_heads
            freq_idx, inv_freq_time, n_idx_planes, n_time_planes = build_torope_freqs(
                head_dim, self.max_seq_len, time_split_ratio=torope_time_split,
            )
            # Register as buffers so they move with .to(device) but are not parameters
            self.register_buffer('torope_freq_idx', freq_idx)
            self.register_buffer('torope_inv_freq_time', inv_freq_time)
            self.torope_n_idx_planes = n_idx_planes
            self.torope_n_time_planes = n_time_planes
            self.torope_time_split = torope_time_split
        else:
            if use_segment_emb:
                max_n_items = self.max_seq_len // n_sid_layers + 1
                self.item_pos_emb = nn.Embedding(max_n_items, embed_dim)
                self.layer_pos_emb = nn.Embedding(n_sid_layers, embed_dim)
            else:
                self.pos_emb = nn.Embedding(self.max_seq_len, embed_dim)

        # Side information embeddings
        # With TO-RoPE: time is encoded via RoPE, so time_gap_emb is skipped.
        # action_emb is still additive (not positional), always included if requested.
        if n_time_buckets > 0 and not use_torope:
            self.time_gap_emb = nn.Embedding(n_time_buckets, embed_dim)
        if n_action_levels > 0:
            self.action_emb = nn.Embedding(n_action_levels, embed_dim)

        # Build TO-RoPE params dict for TransformerLayer (or None)
        torope_layer_params = None
        if use_torope:
            torope_layer_params = {
                'freq_idx': self.torope_freq_idx,
                'inv_freq_time': self.torope_inv_freq_time,
                'n_idx_planes': n_idx_planes,
                'n_time_planes': n_time_planes,
            }

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim, n_heads, dropout,
                use_moe=use_moe, n_experts=n_experts,
                top_k=top_k, expert_dim=expert_dim,
                causal=not parallel,
                torope_params=torope_layer_params,
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

    def _get_pos_emb(self, positions: torch.Tensor) -> torch.Tensor:
        """Get positional embedding for given positions (standard or segment).
        Returns zeros when use_torope=True (position info lives in RoPE instead)."""
        if self.use_torope:
            return torch.zeros(*positions.shape, self.embed_dim,
                               device=positions.device, dtype=torch.get_default_dtype())
        if self.use_segment_emb:
            L = self.n_sid_layers
            item_pos = positions // L
            layer_pos = positions % L
            return self.item_pos_emb(item_pos) + self.layer_pos_emb(layer_pos)
        else:
            return self.pos_emb(positions)

    def embed_with_features(
        self,
        tokens: torch.Tensor,        # (B, T)
        positions: torch.Tensor,     # (B, T) or (1, T)
        time_gaps: torch.Tensor = None,    # (B, T) optional
        action_levels: torch.Tensor = None,  # (B, T) optional
    ) -> torch.Tensor:
        """Single source of truth for input embedding + side features injection.

        All training (forward) and inference (forward_cached, compute_sid_logprobs)
        paths must call this instead of combining _embed_tokens + _get_pos_emb +
        feature embeddings manually.  Adding a new side feature means editing
        exactly this one function.
        """
        x = self._embed_tokens(tokens) + self._get_pos_emb(positions)
        if time_gaps is not None and hasattr(self, 'time_gap_emb'):
            x = x + self.time_gap_emb(time_gaps)
        if action_levels is not None and hasattr(self, 'action_emb'):
            x = x + self.action_emb(action_levels)
        return x

    def _transformer_forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run through all transformer layers + final norm.

        positions and timestamps are forwarded to each layer for TO-RoPE.
        Both are ignored when use_torope=False.
        """
        for layer in self.layers:
            x = layer(x, positions=positions, timestamps=timestamps)
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

    def _transformer_forward_cached(
        self,
        x: torch.Tensor,
        kv_caches=None,
        positions: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        kv_positions: Optional[torch.Tensor] = None,
        kv_timestamps: Optional[torch.Tensor] = None,
    ):
        """Transformer forward with per-layer KV cache.

        Returns:
            out: (B, T_new, D) — output hidden states for new positions only.
            new_kv_caches: list of 6 tensors, each (B, T_cached + T_new, D).
        """
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(
                x, kv_cache=cache_i, use_cache=True,
                positions=positions, timestamps=timestamps,
                kv_positions=kv_positions, kv_timestamps=kv_timestamps,
            )
            new_caches.append(new_cache)
        return self.final_norm(x), new_caches

    @torch.no_grad()
    def forward_cached(self, input_tokens=None, generated_tokens=None, kv_caches=None,
                        ctx_time_gaps=None, ctx_action_levels=None,
                        step_time_gap=None, step_action_level=None,
                        ctx_timestamps=None, step_timestamp=None,
                        kv_positions_cache=None, kv_timestamps_cache=None):
        """Inference-only forward with KV cache.

        Calling patterns:
            Cold start: ``forward_cached(input_tokens)`` — encodes full context.
            Incremental: ``forward_cached(generated_tokens=new, kv_caches=kv)``
                — encodes only *new* tokens using cached context.

        Args:
            ctx_time_gaps: (B, T_ctx) time gap buckets (non-TO-RoPE path).
            ctx_action_levels: (B, T_ctx) action levels.
            step_time_gap: (B, T_new) time gap for incremental tokens.
            step_action_level: (B, T_new) action level for incremental tokens.
            ctx_timestamps: (B, T_ctx) float timestamps in hours (TO-RoPE path).
            step_timestamp: (B, T_new) float timestamps for new tokens (TO-RoPE path).
            kv_positions_cache: (B, T_kv) positions already in KV cache (TO-RoPE incremental).
            kv_timestamps_cache: (B, T_kv) timestamps already in KV cache (TO-RoPE incremental).

        Returns:
            logits: (B, C) logits for the next token.
            kv_caches: list of per-layer caches for reuse.
            (kv_positions, kv_timestamps) — updated position/timestamp caches (TO-RoPE only).
        """
        if kv_caches is None:
            # Cold start — encode full sequence
            tokens = input_tokens
            if generated_tokens is not None and generated_tokens.size(1) > 0:
                tokens = torch.cat([input_tokens, generated_tokens], dim=1)
            T = tokens.size(1)
            device = tokens.device
            positions = torch.arange(T, device=device).unsqueeze(0)

            tg = al = None
            if ctx_time_gaps is not None:
                T_ctx = ctx_time_gaps.size(1)
                pad = torch.zeros(ctx_time_gaps.size(0), T - T_ctx,
                                  dtype=torch.long, device=device)
                tg = torch.cat([ctx_time_gaps, pad], dim=1)
            if ctx_action_levels is not None:
                T_ctx = ctx_action_levels.size(1)
                pad = torch.zeros(ctx_action_levels.size(0), T - T_ctx,
                                  dtype=torch.long, device=device)
                al = torch.cat([ctx_action_levels, pad], dim=1)
            x = self.embed_with_features(tokens, positions, tg, al)

            # TO-RoPE: build full timestamp tensor for context
            if self.use_torope:
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
                x, positions=positions, timestamps=timestamps)

            # Return updated KV position/timestamp caches for incremental decode
            new_kv_pos = positions if self.use_torope else None
            new_kv_ts = timestamps if self.use_torope else None
        else:
            # Incremental — only new tokens
            offset = kv_caches[0].size(1)
            new_tokens = generated_tokens
            T_new = new_tokens.size(1)
            device = new_tokens.device
            x = self._embed_tokens_at_offset(new_tokens, offset)
            positions = torch.arange(offset, offset + T_new, device=device).unsqueeze(0)
            x = x + self._get_pos_emb(positions)
            if step_time_gap is not None and hasattr(self, 'time_gap_emb'):
                x = x + self.time_gap_emb(step_time_gap)
            if step_action_level is not None and hasattr(self, 'action_emb'):
                x = x + self.action_emb(step_action_level)

            # TO-RoPE: pass full KV positions/timestamps so RoPE angles are correct
            if self.use_torope:
                ts_new = step_timestamp.float() if step_timestamp is not None \
                    else torch.zeros(new_tokens.size(0), T_new, device=device)
                if kv_positions_cache is not None:
                    B_kv = kv_positions_cache.size(0)
                    full_pos = torch.cat([kv_positions_cache,
                                          positions.expand(B_kv, -1)], dim=1)
                    kv_ts_base = kv_timestamps_cache.float() if kv_timestamps_cache is not None \
                        else torch.zeros(B_kv, offset, device=device)
                    full_ts = torch.cat([kv_ts_base,
                                         ts_new.expand(B_kv, -1)], dim=1)
                else:
                    full_pos = positions
                    full_ts = ts_new
                out, kv_caches = self._transformer_forward_cached(
                    x, kv_caches,
                    positions=positions, timestamps=ts_new,
                    kv_positions=full_pos, kv_timestamps=full_ts,
                )
                new_kv_pos = full_pos
                new_kv_ts = full_ts
            else:
                out, kv_caches = self._transformer_forward_cached(x, kv_caches)
                new_kv_pos = new_kv_ts = None

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
        time_gaps: Optional[torch.Tensor] = None,
        action_levels: Optional[torch.Tensor] = None,
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
            time_gaps: (B, T) time gap bucket indices (optional side feature).
            action_levels: (B, T) action level indices (optional side feature).
        """
        if packed_targets is not None:
            return self._forward_packed(
                input_tokens, packed_targets, packed_mask,
                neg_l0_tokens=neg_l0_tokens, neg_l0_mask=neg_l0_mask,
                entp_weight=entp_weight,
                item_embeddings=item_embeddings,
                contrastive_weight=contrastive_weight,
                contrastive_temp=contrastive_temp,
                time_gaps=time_gaps,
                action_levels=action_levels,
            )

        device = input_tokens.device

        if self.parallel:
            positions = torch.arange(self.seq_len, device=device).unsqueeze(0)
            x = self._embed_tokens(input_tokens) + self._get_pos_emb(positions)
            tp_pos = positions if self.use_torope else None
            tp_ts = torch.zeros(1, self.seq_len, device=device) if self.use_torope else None
            out = self._transformer_forward(x, positions=tp_pos, timestamps=tp_ts)
            s = out[:, -1, :]
            return [self.output_projs[l](s) for l in range(self.n_sid_layers)]

        # Autoregressive
        if generated_tokens is not None and generated_tokens.size(1) > 0:
            tokens = torch.cat([input_tokens, generated_tokens], dim=1)
        else:
            tokens = input_tokens

        T = tokens.size(1)
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self._embed_tokens(tokens) + self._get_pos_emb(positions)
        # For TO-RoPE: pass positions (timestamps default to zeros inside each layer)
        torope_pos = positions if self.use_torope else None
        torope_ts = torch.zeros(1, T, device=device) if self.use_torope else None
        out = self._transformer_forward(x, positions=torope_pos, timestamps=torope_ts)

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
        time_gaps: Optional[torch.Tensor] = None,
        action_levels: Optional[torch.Tensor] = None,
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
            time_gaps: (B, S) — time gap bucket indices.
            action_levels: (B, S) — action level indices.
        Returns:
            loss: scalar
        """
        B, S = input_tokens.size()
        device = input_tokens.device
        L = self.n_sid_layers

        positions = torch.arange(S, device=device).unsqueeze(0)
        x = self.embed_with_features(input_tokens, positions, time_gaps, action_levels)
        tp_pos = positions if self.use_torope else None
        tp_ts = torch.zeros(1, S, device=device) if self.use_torope else None
        hidden = self._transformer_forward(x, positions=tp_pos, timestamps=tp_ts)  # (B, S, D)

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
