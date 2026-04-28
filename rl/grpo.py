"""GRPO and ECPO policy optimization losses.

GRPO: Group Relative Policy Optimization (OneMall, arxiv 2601.21770)
  - Group-normalized advantage: A_i = (r_i - mean(r)) / std(r)
  - Clipped surrogate (PPO-style, eps=0.2)
  - L = E[min(rho * A, clip(rho, 1-eps, 1+eps) * A)]

ECPO: Early Clipped GRPO (OneRec, arxiv 2506.13695v4)
  - Same as GRPO plus early clip for negative-advantage candidates:
    For A_i < 0: replace pi_old with pi'_old = max(sg(pi_θ)/(1+eps+δ), pi_ref)
    This prevents rho from exceeding 1+eps+δ on bad samples, stopping
    gradient explosion when pi_θ → 0 for negatives.
  - delta=0.1 per ECPO paper.

group_offsets format: same as sample_offsets in dpo.py — (B+1,) int tensor,
group i spans [off[i] : off[i+1]]. All candidates are peers (no "chosen").
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def _sid_semantic_gate(
    sids: Tensor,           # (G, n_layers) int — SID tokens for the group
    best_idx: int,          # index of the highest-reward candidate in the group
) -> Tensor:
    """Compute per-candidate semantic gate based on SID prefix overlap with best candidate.

    Gate = shared_prefix_length / n_layers ∈ [0, 1].
    A candidate that shares all layers with the best item → gate=1 (max penalty).
    A candidate that shares nothing → gate=0 (standard penalty).

    Used by A2PO to amplify negative-advantage penalties for semantically
    similar but low-reward candidates (hard negatives).
    """
    best = sids[best_idx]           # (n_layers,)
    n_layers = sids.size(1)
    gates = torch.zeros(sids.size(0), device=sids.device)
    for j in range(sids.size(0)):
        if j == best_idx:
            continue
        shared = 0
        for l in range(n_layers):
            if sids[j, l] == best[l]:
                shared += 1
            else:
                break  # prefix match only
        gates[j] = shared / n_layers
    return gates  # (G,) — no grad needed, pure indexing


def _grpo_core(
    policy_lp: Tensor,        # (N,) log pi_θ — requires grad
    ref_lp: Tensor,           # (N,) log pi_ref — computed under no_grad
    rewards: Tensor,          # (N,) float reward — detached
    group_offsets: Tensor,    # (B+1,) group boundaries
    eps: float = 0.2,
    delta: float = 0.0,       # 0.0 = GRPO, >0 = ECPO early clip
    rank_norm: bool = False,  # True → rank-based advantage ∈ [-1,1] (robust to log-scale rewards)
    a2po: bool = False,       # True → A2PO: amplify neg-adv penalty for semantically similar candidates
    a2po_alpha: float = 1.0,  # A2PO gate scale: neg_adv *= (1 + alpha * semantic_gate)
    sids: Optional[Tensor] = None,  # (N, n_layers) required when a2po=True
    return_diagnostics: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict]]:
    # fp32 throughout for numerical stability
    policy_lp = policy_lp.float()
    ref_lp = ref_lp.float()
    rewards = rewards.float()

    offsets = group_offsets.tolist()
    B = len(offsets) - 1

    group_losses = []
    diag_adv_mean = []
    diag_adv_std = []
    diag_ratio_mean = []
    diag_clip_frac = []
    diag_kl_mean = []
    diag_reward_mean = []
    diag_reward_std = []
    diag_early_clip_frac = []  # ECPO only

    log_early_bound = None
    if delta > 0.0:
        log_early_bound = torch.log(
            torch.tensor(1.0 + eps + delta, device=policy_lp.device,
                         dtype=torch.float32))

    for i in range(B):
        start, end = int(offsets[i]), int(offsets[i + 1])
        G = end - start
        if G < 2:
            continue  # need at least 2 candidates to normalize

        g_policy = policy_lp[start:end]   # (G,) — has grad
        g_ref    = ref_lp[start:end]       # (G,) — no grad
        g_reward = rewards[start:end]      # (G,) — no grad

        # Group-normalized advantage
        if rank_norm:
            # Rank-based: adv ∈ [-1, 1], robust to log-scale reward distributions.
            # rank 0 = worst → -1, rank G-1 = best → +1.
            # Ties are broken by mean rank (equivalent to averaging tied positions).
            ranks = g_reward.argsort().argsort().float()   # (G,) — no grad
            if G > 1:
                adv = (2.0 * ranks / (G - 1)) - 1.0
            else:
                adv = torch.zeros_like(ranks)
            # rank_norm always has variance unless G=1 — no std check needed
        else:
            r_mean = g_reward.mean()
            r_std  = g_reward.std()
            if r_std.item() < 1e-6:
                continue   # all rewards identical → no learning signal, skip
            adv = (g_reward - r_mean) / r_std   # (G,) — no grad
            adv = adv.clamp(-5.0, 5.0)          # guard against extreme outliers

        # A2PO: amplify negative-advantage penalty for semantically similar candidates.
        # Candidates close to the best item in SID prefix space but with low reward
        # are the most informative hard negatives — penalize them more aggressively.
        if a2po and sids is not None:
            g_sids = sids[start:end]                     # (G, n_layers)
            best_idx = int(g_reward.argmax().item())
            gate = _sid_semantic_gate(g_sids, best_idx)  # (G,) ∈ [0,1]
            neg_mask = adv < 0
            if neg_mask.any():
                adv = adv.clone()
                adv[neg_mask] = adv[neg_mask] * (1.0 + a2po_alpha * gate[neg_mask])

        # Policy ratio rho = pi_θ / pi_ref
        log_rho = g_policy - g_ref.detach()   # grad through g_policy only
        log_rho = log_rho.clamp(-10.0, 10.0)  # prevent rho from exploding
        rho = log_rho.exp()

        # ECPO early clip: for negative-advantage candidates, tighten the
        # effective denominator so rho can't exceed 1+eps+delta
        if delta > 0.0:
            neg_mask = adv < 0
            if neg_mask.any():
                # pi'_old in log-space: max(sg(pi_θ) - log(1+eps+δ), log_pi_ref)
                log_pi_prime = torch.maximum(
                    g_policy.detach() - log_early_bound,
                    g_ref.detach(),
                )
                log_rho_neg = g_policy - log_pi_prime   # still has grad
                rho_eff = rho.clone()
                rho_eff[neg_mask] = log_rho_neg[neg_mask].exp()

                if return_diagnostics:
                    was_clipped = (log_pi_prime > g_ref.detach())[neg_mask]
                    diag_early_clip_frac.append(
                        float(was_clipped.float().mean().item()))
            else:
                rho_eff = rho
                if return_diagnostics:
                    diag_early_clip_frac.append(0.0)
        else:
            rho_eff = rho

        # Clipped surrogate loss
        rho_clipped = rho_eff.clamp(1.0 - eps, 1.0 + eps)
        group_loss = -torch.min(rho_eff * adv, rho_clipped * adv).mean()
        group_losses.append(group_loss)

        if return_diagnostics:
            diag_adv_mean.append(adv.detach().mean().item())
            diag_adv_std.append(adv.detach().std().item())
            diag_ratio_mean.append(rho_eff.detach().mean().item())
            clip_frac = (
                (rho_eff.detach() - rho_clipped.detach()).abs() > 1e-6
            ).float().mean().item()
            diag_clip_frac.append(clip_frac)
            # KL(π_θ || π_ref) ≈ E[log π_θ - log π_ref] per group
            diag_kl_mean.append((g_policy.detach() - g_ref.detach()).mean().item())
            diag_reward_mean.append(g_reward.detach().mean().item())
            diag_reward_std.append(g_reward.detach().std().item())

    if not group_losses:
        zero = torch.tensor(0.0, device=policy_lp.device, requires_grad=True)
        if return_diagnostics:
            return zero, {}
        return zero

    loss = torch.stack(group_losses).mean()

    if not return_diagnostics:
        return loss

    def _avg(lst: list) -> float:
        return float(sum(lst) / len(lst)) if lst else 0.0

    diag: Dict[str, float] = {
        'advantage_mean':    _avg(diag_adv_mean),
        'advantage_std':     _avg(diag_adv_std),
        'policy_ratio_mean': _avg(diag_ratio_mean),
        'clip_fraction':     _avg(diag_clip_frac),
        'kl_mean':           _avg(diag_kl_mean),
        'reward_mean':       _avg(diag_reward_mean),
        'reward_std':        _avg(diag_reward_std),
    }
    if delta > 0.0:
        diag['early_clip_fraction'] = _avg(diag_early_clip_frac)

    return loss, diag


def grpo_loss(
    policy_lp: Tensor,
    ref_lp: Tensor,
    rewards: Tensor,
    group_offsets: Tensor,
    eps: float = 0.2,
    rank_norm: bool = False,
    a2po: bool = False,
    a2po_alpha: float = 1.0,
    sids: Optional[Tensor] = None,
    return_diagnostics: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict]]:
    """GRPO loss (no early clip).

    Args:
        policy_lp:     (N,) log π_θ for all candidates, requires grad.
        ref_lp:        (N,) log π_ref, computed under no_grad.
        rewards:       (N,) detached rewards from reward_fn.
        group_offsets: (B+1,) group boundaries.
        eps:           PPO clip range (default 0.2).
        rank_norm:     if True, use rank-based advantage ∈ [-1,1] instead of
                       z-score normalization. Robust to log-scale distributions.
        a2po:          if True, amplify negative-advantage penalty for semantically
                       similar (hard negative) candidates via SID prefix overlap.
        a2po_alpha:    gate scale for A2PO (default 1.0).
        sids:          (N, n_layers) int tensor, required when a2po=True.
        return_diagnostics: if True, return (loss, diag_dict).

    Diagnostics keys: advantage_mean, advantage_std, policy_ratio_mean,
                      clip_fraction, reward_mean, reward_std.
    """
    return _grpo_core(
        policy_lp, ref_lp, rewards, group_offsets,
        eps=eps, delta=0.0, rank_norm=rank_norm,
        a2po=a2po, a2po_alpha=a2po_alpha, sids=sids,
        return_diagnostics=return_diagnostics,
    )


def ecpo_loss(
    policy_lp: Tensor,
    ref_lp: Tensor,
    rewards: Tensor,
    group_offsets: Tensor,
    eps: float = 0.2,
    delta: float = 0.1,
    rank_norm: bool = False,
    a2po: bool = False,
    a2po_alpha: float = 1.0,
    sids: Optional[Tensor] = None,
    return_diagnostics: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict]]:
    """ECPO loss — GRPO with early clip for negative-advantage candidates.

    Args:
        policy_lp:     (N,) log π_θ for all candidates, requires grad.
        ref_lp:        (N,) log π_ref, computed under no_grad.
        rewards:       (N,) detached rewards from reward_fn.
        group_offsets: (B+1,) group boundaries.
        eps:           PPO clip range (default 0.2).
        delta:         early clip margin (default 0.1 per ECPO paper).
        rank_norm:     if True, use rank-based advantage ∈ [-1,1].
        a2po:          if True, amplify negative-advantage penalty for semantically
                       similar (hard negative) candidates via SID prefix overlap.
        a2po_alpha:    gate scale for A2PO (default 1.0).
        sids:          (N, n_layers) int tensor, required when a2po=True.
        return_diagnostics: if True, return (loss, diag_dict).

    Additional diagnostics key: early_clip_fraction.
    """
    return _grpo_core(
        policy_lp, ref_lp, rewards, group_offsets,
        eps=eps, delta=delta, rank_norm=rank_norm,
        a2po=a2po, a2po_alpha=a2po_alpha, sids=sids,
        return_diagnostics=return_diagnostics,
    )
