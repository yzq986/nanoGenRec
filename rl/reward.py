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
