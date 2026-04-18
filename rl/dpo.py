"""Softmax-DPO loss and sequence log-probability computation.

Implements the DPO loss from Align³GR (AAAI 2026 Oral, arxiv 2511.11255),
which supports 1 chosen vs N rejected per sample (Softmax-DPO).

Key design: sequence log-prob for a 3-token SID is computed in a single
forward pass by exploiting the causal Transformer architecture. Given
input [ctx..., sid_L0, sid_L1], causal masking ensures:
  - Position T-1 predicts P(L0 | ctx)
  - Position T   predicts P(L1 | ctx, L0)
  - Position T+1 predicts P(L2 | ctx, L0, L1)
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_sid_logprobs(
    model,
    context_tokens: Tensor,   # (B, T_ctx) right-padded
    context_lengths: Tensor,  # (B,) actual context lengths
    sid_tokens: Tensor,       # (B, n_layers)
    n_layers: int,
) -> Tensor:
    """Compute log P(sid | context) for each sample in a single forward pass.

    Builds input = [ctx..., sid_L0, ..., sid_{L-2}] (drops last SID token),
    runs one Transformer forward, and gathers log-probs at the correct
    positions using per-sample indexing.

    Returns:
        (B,) total log-probability per sample.
    """
    B = context_tokens.size(0)
    device = context_tokens.device

    # Build input: [ctx..., sid_L0, ..., sid_{L-2}]
    # We drop the last SID token because we only need to predict it, not
    # condition on it. The context already determines position T-1's
    # prediction of L0.
    sid_input = sid_tokens[:, :-1]  # (B, n_layers - 1)
    full_input = torch.cat([context_tokens, sid_input], dim=1)  # (B, T_ctx + L - 1)
    T = full_input.size(1)

    # Forward pass
    positions = torch.arange(T, device=device).unsqueeze(0)
    x = model._embed_tokens(full_input) + model.pos_emb(positions)
    hidden = model._transformer_forward(x)  # (B, T, D)

    # Extract log-probs at the 3 prediction positions.
    # For sample b, the positions are:
    #   L0: context_lengths[b] - 1  (last context token predicts L0)
    #   L1: context_lengths[b]      (sid_L0 predicts L1)
    #   L2: context_lengths[b] + 1  (sid_L1 predicts L2)
    batch_idx = torch.arange(B, device=device)
    total_logprob = torch.zeros(B, device=device, dtype=hidden.dtype)

    for li in range(n_layers):
        pos = context_lengths - 1 + li  # (B,) per-sample position
        h = hidden[batch_idx, pos]  # (B, D)
        logits = model.output_projs[li](h)  # (B, C_li)
        lp = F.log_softmax(logits, dim=-1)
        total_logprob = total_logprob + lp.gather(1, sid_tokens[:, li:li + 1]).squeeze(1)

    return total_logprob


def compute_sid_logprobs_batch(
    model,
    context_tokens: Tensor,   # (B, T_ctx) right-padded
    context_lengths: Tensor,  # (B,) actual context lengths
    all_sids: Tensor,         # (B, K, n_layers)
    n_layers: int,
    max_chunk: int = 16,
) -> Tensor:
    """Compute log-probs for K SID candidates per context.

    Expands each context K times and calls compute_sid_logprobs in
    micro-batches of max_chunk to avoid OOM on large B*K.

    Returns:
        (B, K) log-probabilities.
    """
    B, K, L = all_sids.shape
    # Expand context: (B, T) -> (B, K, T) -> (B*K, T)
    ctx_exp = context_tokens.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
    len_exp = context_lengths.unsqueeze(1).expand(-1, K).reshape(B * K)
    sids_flat = all_sids.reshape(B * K, L)

    total = B * K
    if total <= max_chunk:
        lp = compute_sid_logprobs(model, ctx_exp, len_exp, sids_flat, n_layers)
    else:
        chunks = []
        for start in range(0, total, max_chunk):
            end = min(start + max_chunk, total)
            chunk_lp = compute_sid_logprobs(
                model,
                ctx_exp[start:end],
                len_exp[start:end],
                sids_flat[start:end],
                n_layers,
            )
            chunks.append(chunk_lp)
        lp = torch.cat(chunks, dim=0)

    return lp.reshape(B, K)


def softmax_dpo_loss(
    policy_chosen_lp: Tensor,     # (B,) log pi_theta(y_w | x)
    policy_rejected_lp: Tensor,   # (B, N_rej) log pi_theta(y_l | x)
    ref_chosen_lp: Tensor,        # (B,) log pi_ref(y_w | x)
    ref_rejected_lp: Tensor,      # (B, N_rej) log pi_ref(y_l | x)
    rejected_mask: Tensor,        # (B, N_rej) True = valid rejected
    beta: float = 0.1,
) -> Tensor:
    """Softmax-DPO loss (Chen et al. 2024b, used in Align³GR).

    Supports 1 chosen vs N rejected per sample.

    Formula:
        r_l = beta * (log pi_theta(y_l)/pi_ref(y_l) - log pi_theta(y_w)/pi_ref(y_w))
        L = -mean_b[ log sigmoid(-logsumexp_l(r_l)) ]

    All computation in fp32 for numerical stability.

    Returns:
        Scalar loss.
    """
    # Ensure fp32 for stable sigmoid/logsumexp
    chosen_adv = (policy_chosen_lp - ref_chosen_lp).float()      # (B,)
    rejected_adv = (policy_rejected_lp - ref_rejected_lp).float()  # (B, N_rej)

    # r_l = beta * (rejected_advantage - chosen_advantage)
    r = beta * (rejected_adv - chosen_adv.unsqueeze(1))  # (B, N_rej)

    # Mask invalid rejected candidates to -inf for logsumexp
    r = r.masked_fill(~rejected_mask, float('-inf'))

    # Check for samples with no valid rejected candidates
    has_valid = rejected_mask.any(dim=1)  # (B,)
    if not has_valid.any():
        return torch.tensor(0.0, device=policy_chosen_lp.device, requires_grad=True)

    # logsumexp over rejected, then -logsigmoid(-lse)
    lse = torch.logsumexp(r[has_valid], dim=1)  # (n_valid,)
    loss = -F.logsigmoid(-lse).mean()

    return loss
