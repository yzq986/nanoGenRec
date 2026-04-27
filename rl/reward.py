"""Pluggable reward functions for GRPO/ECPO training.

Usage example::

    reward = CompositeReward([
        ('behavior', 1.0, BehaviorReward(sid_to_score)),
        ('format',   0.5, FormatReward(sid_trie, n_layers, sample_k=5)),
        ('business', 0.3, BusinessReward(timeliness_fn, name='timeliness')),
    ])

    # In training step:
    rewards = reward(all_sids, ctx_exp, len_exp)   # (N,)
    wb.update(reward.metrics())                    # keys: reward/behavior_mean, etc.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor


# ── Protocols ──────────────────────────────────────────────────────────────

class RewardFn:
    """Base protocol for reward functions.

    Subclasses must implement __call__. Optionally implement metrics()
    to expose per-step diagnostics for logging (makes the class a
    DiagnosticReward automatically).
    """

    def __call__(
        self,
        sids: Tensor,            # (N, n_layers) integer SID tokens
        context_tokens: Tensor,  # (N, T_ctx) right-padded context
        context_lengths: Tensor, # (N,) actual context lengths
    ) -> Tensor:                 # (N,) float rewards
        raise NotImplementedError


class DiagnosticReward(RewardFn):
    """RewardFn that also emits per-step diagnostics for logging."""

    def metrics(self) -> Dict[str, float]:
        """Return diagnostic dict after the last __call__.

        Keys are short names (e.g. 'mean', 'legal_rate'). CompositeReward
        will namespace them as 'reward/{component_name}_{key}'.
        """
        raise NotImplementedError


# ── Concrete reward classes ─────────────────────────────────────────────────

class BehaviorReward(DiagnosticReward):
    """Reward from pre-computed behavior signal scores.

    Looks up reward with cascading prefix fallback for better coverage:
      full tuple → (l0, l1) prefix → (l0,) prefix → default

    Args:
        sid_to_score: mapping (l0, l1, ...) tuple → float reward.
            Also accepts prefix tuples of any length for fallback scoring.
        default_reward: reward for SIDs not in the mapping at any prefix level.
        prefix_scale: scale factor for prefix-level scores (< 1 to down-weight
            uncertain matches). Default 0.5.
    """

    def __init__(
        self,
        sid_to_score: Dict[Tuple[int, ...], float],
        default_reward: float = 0.0,
        prefix_scale: float = 0.5,
    ):
        self._sid_to_score = sid_to_score
        self._default = default_reward
        self._prefix_scale = prefix_scale
        self._last_mean: float = 0.0

    def _score(self, sid_tuple: Tuple[int, ...]) -> float:
        # Full match first
        s = self._sid_to_score.get(sid_tuple)
        if s is not None:
            return s
        # Cascade through shorter prefixes, scaled down
        for length in range(len(sid_tuple) - 1, 0, -1):
            prefix = sid_tuple[:length]
            s = self._sid_to_score.get(prefix)
            if s is not None:
                return s * (self._prefix_scale ** (len(sid_tuple) - length))
        return self._default

    def __call__(
        self,
        sids: Tensor,
        context_tokens: Tensor,
        context_lengths: Tensor,
    ) -> Tensor:
        sids_cpu = sids.cpu()
        scores = [self._score(tuple(sids_cpu[k].tolist()))
                  for k in range(sids_cpu.size(0))]
        out = torch.tensor(scores, dtype=torch.float32, device=sids.device)
        self._last_mean = float(out.mean().item())
        return out

    def metrics(self) -> Dict[str, float]:
        return {'mean': self._last_mean}


class FormatReward(DiagnosticReward):
    """Binary legality reward: 1.0 if SID exists in trie, 0.0 otherwise.

    Used as ECPO Format Reward. With sample_k=5, only checks a random
    sub-sample of K candidates per call (cost-saving, per ECPO paper).
    With sample_k=None, checks all N candidates.

    Legality is checked by walking each SID layer-by-layer through
    SIDTrie.valid_tokens(layer, prefix).

    Args:
        sid_trie: SIDTrie from ntp/model.py.
        n_layers: number of SID layers.
        sample_k: if set, randomly draw sample_k indices from N to check.
    """

    def __init__(self, sid_trie, n_layers: int, sample_k: Optional[int] = None):
        self._trie = sid_trie
        self._n_layers = n_layers
        self._sample_k = sample_k
        self._last_legal_rate: float = 0.0

    def _is_legal(self, sid_tuple: Tuple[int, ...]) -> bool:
        for layer in range(self._n_layers):
            prefix = sid_tuple[:layer]
            if sid_tuple[layer] not in self._trie.valid_tokens(layer, prefix):
                return False
        return True

    def __call__(
        self,
        sids: Tensor,
        context_tokens: Tensor,
        context_lengths: Tensor,
    ) -> Tensor:
        N = sids.size(0)
        sids_cpu = sids.cpu()
        rewards = torch.zeros(N, dtype=torch.float32)

        indices = list(range(N))
        if self._sample_k is not None and self._sample_k < N:
            indices = random.sample(indices, self._sample_k)

        n_legal = 0
        for k in indices:
            if self._is_legal(tuple(sids_cpu[k].tolist())):
                rewards[k] = 1.0
                n_legal += 1

        self._last_legal_rate = n_legal / len(indices) if indices else 0.0
        return rewards.to(sids.device)

    def metrics(self) -> Dict[str, float]:
        return {'legal_rate': self._last_legal_rate}


class ExternalReward(DiagnosticReward):
    """Adapter that wraps an arbitrary callable into the RewardFn interface.

    The wrapped function receives plain Python lists for portability:
        fn(sids: List[List[int]], contexts: List[List[int]]) -> List[float]

    Args:
        fn: callable with the signature above.
        name: used only in diagnostics (no functional effect).
    """

    def __init__(self, fn, name: str = 'external'):
        self._fn = fn
        self._name = name
        self._last_mean: float = 0.0

    def __call__(
        self,
        sids: Tensor,
        context_tokens: Tensor,
        context_lengths: Tensor,
    ) -> Tensor:
        sids_list = sids.cpu().tolist()
        ctxs_list = [
            context_tokens[i, :int(context_lengths[i].item())].cpu().tolist()
            for i in range(sids.size(0))
        ]
        scores = self._fn(sids_list, ctxs_list)
        out = torch.tensor(scores, dtype=torch.float32, device=sids.device)
        self._last_mean = float(out.mean().item())
        return out

    def metrics(self) -> Dict[str, float]:
        return {'mean': self._last_mean}


class BusinessReward(DiagnosticReward):
    """Configurable reward for business policy signals.

    Same interface as ExternalReward. Provided as a separate class so
    callers can distinguish business-policy rewards from external model
    rewards at the type level and assign them distinct names/weights in
    CompositeReward.

    Example uses: timeliness (newer items score higher), author weighting
    (preferred publisher bonus), category diversity penalty.

    Args:
        fn: callable(sids_list, contexts_list) -> List[float].
        name: diagnostic label.
    """

    def __init__(self, fn, name: str = 'business'):
        self._fn = fn
        self._name = name
        self._last_mean: float = 0.0

    def __call__(
        self,
        sids: Tensor,
        context_tokens: Tensor,
        context_lengths: Tensor,
    ) -> Tensor:
        sids_list = sids.cpu().tolist()
        ctxs_list = [
            context_tokens[i, :int(context_lengths[i].item())].cpu().tolist()
            for i in range(sids.size(0))
        ]
        scores = self._fn(sids_list, ctxs_list)
        out = torch.tensor(scores, dtype=torch.float32, device=sids.device)
        self._last_mean = float(out.mean().item())
        return out

    def metrics(self) -> Dict[str, float]:
        return {'mean': self._last_mean}


# ── Action bitmap quality weights (v0420 production spec) ─────────────────────
#
# Based on the production scoring sheet (0420版本):
#   score ∝ log10(1 + Σ w_i × action_i) × quality × freshness
#
# We map action_bitmap bits to additive weights, then take log10(1 + sum).
# Bits from export_behavior.py; weights from production spec columns.
#
# Skipped (need real-time model inference):
#   点击率预估 (w=15), 关注率预估 (w=1000) — require CTR/follow-rate model
#   效率类 (comments/likes/trades per hour) — require real-time aggregation
#
_ACTION_ADDITIVE_WEIGHTS: List[Tuple[int, float]] = [
    # (bitmask,  additive_weight)  — all matching bits are summed
    (262144,  4000.0),  # place_order        → 交易 (highest)
    (131072,  1.0),     # trade_click        → 交易相关点击
    (524288,  5.0),     # live_duration      → 近24h收藏 proxy (live engagement)
    (8,       4000.0),  # follow (feed)      → 关注
    (256,     4000.0),  # detail_follow      → 关注
    (2,       1.0),     # like (feed)        → 点赞
    (512,     1.0),     # detail_like        → 点赞
    (1048576, 2000.0),  # comment            → 评论
    (2048,    2000.0),  # detail_comment     → 评论
    (4,       3.0),     # share (feed)       → 分享
    (1024,    3.0),     # detail_share       → 分享
    (16,      0.1),     # coin_click (feed)  → 消费类点击
    (8192,    0.1),     # detail_coin_click  → 消费类点击
    (65536,   0.1),     # video_detail_view  → 消费类
    (32,      0.1),     # comment_click      → 消费类
    (1,       0.1),     # click (feed)       → 消费类点击
]
_NEGATIVE_BIT = -2147483648  # negative_feedback / dislike / report
_VIEW_EXIT_BIT = 4096        # excluded (not a positive signal)

# Production time-decay: TimeDecay(t) = exp(-ln(1.1) × t / 150)
# where t is age in hours. τ_hours = 150 / ln(1.1) ≈ 1582h ≈ 65.9 days
_TIME_DECAY_TAU_HOURS = 1.0 * 24.0        # τ = 1 day, aligns with 3d distribution cutoff
_TIME_DECAY_RATE = 1.0 / _TIME_DECAY_TAU_HOURS  # per-hour decay rate


def _bitmap_to_quality(action_bitmap: int) -> float:
    """Map action_bitmap → log10(1 + weighted_sum) score.

    Aligns with production v0420 spec: score is log-scaled additive
    combination of all matched action weights. Returns negative score
    for negative_feedback, 0.0 if only view_exit or no positive action.
    """
    if action_bitmap & _NEGATIVE_BIT:
        return -1.0
    bm = action_bitmap & ~_VIEW_EXIT_BIT
    if bm <= 0:
        return 0.0
    total = sum(w for mask, w in _ACTION_ADDITIVE_WEIGHTS if bm & mask)
    return math.log10(1.0 + total)


class WeightedBehaviorReward(DiagnosticReward):
    """Continuous item-level reward: quality × freshness.

    Reward = log10(1 + Σ action_weights) × exp(-age_hours / 24)

    Quality weights align with production v0420 spec (additive, log-scaled).
    Freshness: τ = 1 day (24h) — aligns with 3d distribution cutoff (3d→0.05).

    HEPO (Hierarchical Evidence Policy Optimization) prefix scoring:
    When a full SID is not found, falls back to shorter prefixes with
    per-depth scale factors. `hepo_scales` controls the scale at each
    fallback depth:  hepo_scales[0] → L0 only (shallowest),
                     hepo_scales[-1] → full match minus one layer.
    Default [0.1, 0.5] for 3-layer SIDs means:
        L0 match → quality × freshness × 0.1  (weak signal: cluster-level)
        L0L1 match → quality × freshness × 0.5  (medium signal: sub-cluster)
        full match → quality × freshness × 1.0

    Args:
        sid_to_info: dict mapping sid_tuple → (action_bitmap: int, last_ts: float)
            where last_ts is unix timestamp of the most recent interaction.
        eval_ts: reference unix timestamp for freshness calculation (training cutoff).
            Defaults to current time if not provided.
        default_reward: reward for SIDs with no behavior data. Default 0.0.
        prefix_scale: fallback scale factor when hepo_scales is None. Default 0.5.
        hepo_scales: list of scale factors for prefix depths 1..n-1 (index 0 = shallowest).
            Length must equal n_layers - 1. If None, falls back to prefix_scale^depth.
    """

    def __init__(
        self,
        sid_to_info: Dict[Tuple[int, ...], Tuple[int, float]],
        eval_ts: Optional[float] = None,
        default_reward: float = 0.0,
        prefix_scale: float = 0.5,
        hepo_scales: Optional[List[float]] = None,
    ):
        self._sid_to_info = sid_to_info
        self._eval_ts = eval_ts if eval_ts is not None else __import__('time').time()
        self._default = default_reward
        self._prefix_scale = prefix_scale
        self._hepo_scales = hepo_scales  # hepo_scales[i] = scale for prefix of length i+1
        self._last_mean: float = 0.0
        self._last_coverage: float = 0.0

    def _scale_for_depth(self, full_len: int, match_len: int) -> float:
        """Return scale factor for a match at prefix length match_len < full_len."""
        missing = full_len - match_len
        if self._hepo_scales is not None:
            # hepo_scales[0] = scale for match_len=1 (shallowest), etc.
            idx = match_len - 1  # 0-indexed: match_len=1 → idx=0
            if 0 <= idx < len(self._hepo_scales):
                return self._hepo_scales[idx]
        return self._prefix_scale ** missing

    def _score(self, sid_tuple: Tuple[int, ...]) -> float:
        info = self._sid_to_info.get(sid_tuple)
        scale = 1.0
        if info is None:
            for length in range(len(sid_tuple) - 1, 0, -1):
                info = self._sid_to_info.get(sid_tuple[:length])
                if info is not None:
                    scale = self._scale_for_depth(len(sid_tuple), length)
                    break
        if info is None:
            return self._default

        action_bitmap, last_ts = info
        quality = _bitmap_to_quality(action_bitmap)
        if quality <= 0.0:
            return quality * scale  # preserve negative reward sign
        # Freshness decay: exp(-age_hours / 24), τ = 1 day
        age_hours = max(0.0, self._eval_ts - last_ts) / 3600.0
        freshness = math.exp(-_TIME_DECAY_RATE * age_hours)
        return quality * freshness * scale

    def __call__(
        self,
        sids: Tensor,
        context_tokens: Tensor,
        context_lengths: Tensor,
    ) -> Tensor:
        sids_cpu = sids.cpu()
        N = sids_cpu.size(0)
        scores = [self._score(tuple(sids_cpu[k].tolist())) for k in range(N)]
        out = torch.tensor(scores, dtype=torch.float32, device=sids.device)
        self._last_mean = float(out.mean().item())
        self._last_coverage = sum(1 for s in scores if s != 0.0) / N if N > 0 else 0.0
        return out

    def metrics(self) -> Dict[str, float]:
        return {'mean': self._last_mean, 'coverage': self._last_coverage}


# ── Novelty reward ───────────────────────────────────────────────────────────

class NoveltyReward(DiagnosticReward):
    """Penalize candidates that already appear in the user's context.

    Returns 0.0 for novel items (not seen in context), -penalty for items
    whose SID appears in the context token sequence.

    Context tokens are raw integers (SID layer tokens). A candidate SID
    (l0, l1, ..., ln) is considered "seen" if all its layer tokens appear
    consecutively in the context (i.e., the item was already recommended).

    Args:
        penalty: score assigned to non-novel (already-seen) items. Default -1.0.
        n_layers: number of SID layers. Inferred from sids.size(1) if not set.
    """

    def __init__(self, penalty: float = -1.0):
        self._penalty = penalty
        self._last_novel_rate: float = 1.0

    def __call__(
        self,
        sids: Tensor,
        context_tokens: Tensor,
        context_lengths: Tensor,
    ) -> Tensor:
        N = sids.size(0)
        n_layers = sids.size(1)
        sids_cpu = sids.cpu().tolist()
        ctx_cpu = context_tokens.cpu().tolist()
        lens_cpu = context_lengths.cpu().tolist()

        rewards = []
        n_novel = 0
        for i in range(N):
            sid = sids_cpu[i]
            length = int(lens_cpu[i])
            ctx = ctx_cpu[i][:length]
            # Check if sid tokens appear consecutively anywhere in context
            seen = False
            for j in range(len(ctx) - n_layers + 1):
                if ctx[j:j + n_layers] == sid:
                    seen = True
                    break
            if seen:
                rewards.append(self._penalty)
            else:
                rewards.append(0.0)
                n_novel += 1

        self._last_novel_rate = n_novel / N if N > 0 else 1.0
        return torch.tensor(rewards, dtype=torch.float32, device=sids.device)

    def metrics(self) -> Dict[str, float]:
        return {'novel_rate': self._last_novel_rate}


# ── Composite ────────────────────────────────────────────────────────────────

class CompositeReward(DiagnosticReward):
    """Weighted sum of multiple RewardFns.

    Args:
        components: list of (name, weight, reward_fn) triples.

    Returns from __call__: (N,) weighted-sum reward tensor.

    Metrics keys (for W&B logging):
        'reward/{name}_mean'    — mean reward from component {name}
        'reward/{name}_{key}'   — per-component DiagnosticReward.metrics() entries
        'reward/total_mean'     — mean of the final weighted sum

    Example::

        reward = CompositeReward([
            ('behavior', 1.0, BehaviorReward(sid_to_score)),
            ('format',   0.5, FormatReward(sid_trie, n_layers)),
            ('fresh',    0.3, BusinessReward(timeliness_fn, name='fresh')),
        ])
    """

    def __init__(self, components: List[Tuple[str, float, RewardFn]]):
        self._components = components
        self._last_total: Optional[Tensor] = None
        self._last_component_rewards: Dict[str, Tensor] = {}

    def __call__(
        self,
        sids: Tensor,
        context_tokens: Tensor,
        context_lengths: Tensor,
    ) -> Tensor:
        total = torch.zeros(sids.size(0), dtype=torch.float32, device=sids.device)
        self._last_component_rewards = {}
        for name, weight, fn in self._components:
            r = fn(sids, context_tokens, context_lengths).to(total.device)
            self._last_component_rewards[name] = r.detach()
            if weight != 0.0:
                total = total + weight * r
        self._last_total = total.detach()
        return total

    def metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, _, fn in self._components:
            if name in self._last_component_rewards:
                out[f'reward/{name}_mean'] = float(
                    self._last_component_rewards[name].mean().item())
            if isinstance(fn, DiagnosticReward):
                for k, v in fn.metrics().items():
                    out[f'reward/{name}_{k}'] = v
        if self._last_total is not None:
            out['reward/total_mean'] = float(self._last_total.mean().item())
        return out
