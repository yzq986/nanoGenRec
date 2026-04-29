"""Softmax-DPO loss and sequence log-probability computation.

Implements the DPO loss from Align³GR (AAAI 2026 Oral, arxiv 2511.11255),
which supports 1 chosen vs N rejected per sample (Softmax-DPO).

Key design: sequence log-prob for a 3-token SID is computed in a single
forward pass by exploiting the causal Transformer architecture. Given
input [ctx..., sid_L0, sid_L1], causal masking ensures:
  - Position T-1 predicts P(L0 | ctx)
  - Position T   predicts P(L1 | ctx, L0)
  - Position T+1 predicts P(L2 | ctx, L0, L1)

Memory design (gradient checkpointing):
  compute_sid_logprobs_batch processes B×K candidates in chunks of max_chunk.
  Without checkpointing, ALL chunks' computation graphs are retained in GPU
  memory simultaneously until backward() — causing OOM:

    dpo_batch=16, K=21 → 336 samples, max_chunk=64 → 6 chunks
    Each chunk: ~10 GB activations (full transformer forward, seq_len≈510)
    Total: 6 × 10 GB = 60 GB → OOM on A100 40GB

  With gradient checkpointing (torch.utils.checkpoint):
    Forward: each chunk runs without saving intermediate activations
    Backward: chunks are recomputed one at a time to get gradients
    Peak memory: 1 chunk ≈ 10 GB (constant, independent of total chunks)
    Cost: DPO forward computed 2× (forward + recompute), ~25% total step slowdown

  This decouples GPU memory from dpo_batch_size entirely. Only max_chunk
  (the micro-batch size per forward call) determines peak memory.
  dpo_batch_size can scale to 32, 64, or higher — only time cost increases
  linearly, not memory.

  Packed candidates (no padding waste):
    Each DPO pair has 1 chosen + variable N_i rejected (not padded to max).
    All valid candidates are packed flat with offset indices, so only real
    candidates are forwarded — no wasted compute on zero-padded SIDs.
    E.g., Hard difficulty averages 5.9 rejected/pair; without packing,
    batches pad to max_rej≈20 → 3× wasted forward passes.

  NTP batch_size vs dpo_batch_size — two independent concepts:
    NTP batch_size (auto-capped ≈149): sequences for NTP loss via _forward_packed
    dpo_batch_size (default 16): preference pairs for DPO loss
    Each DPO pair expands to 1+N_i candidates, each needing full forward.
    NTP backward finishes and releases memory before DPO forward starts,
    so the two batch sizes are independently constrained.
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from ntp.model import SparseMoEBlock


@contextmanager
def _freeze_moe_bias(model):
    """Temporarily freeze expert_bias updates in all SparseMoEBlock modules.

    Required for gradient checkpointing compatibility: MoE's Loss-Free
    bias update modifies expert_bias in-place during forward(), making
    it non-idempotent. Checkpoint recomputes forward() during backward,
    and the changed bias produces different router decisions → different
    intermediate tensor shapes → RuntimeError.

    This context manager sets freeze_bias=True on all SparseMoEBlock
    modules, preventing bias updates during forward. NTP forward (not
    checkpointed) still updates bias normally outside this context.
    """
    moe_blocks = [m for m in model.modules() if isinstance(m, SparseMoEBlock)]
    for m in moe_blocks:
        m.freeze_bias = True
    try:
        yield
    finally:
        for m in moe_blocks:
            m.freeze_bias = False


def compute_sid_logprobs(
    model,
    context_tokens: Tensor,        # (B, T_ctx) right-padded
    context_lengths: Tensor,       # (B,) actual context lengths
    sid_tokens: Tensor,            # (B, n_layers)
    n_layers: int,
    ctx_side_features: dict = None,   # {"time_gaps": (B,T_ctx), "action_levels": ...}
    gen_side_features: dict = None,   # {"time_gaps": int, "action_levels": int, ...}
) -> Tensor:
    """Compute log P(sid | context) for each sample in a single forward pass.

    Builds input = [ctx..., sid_L0, ..., sid_{L-2}] (drops last SID token),
    runs one Transformer forward, and gathers log-probs at the correct
    positions using per-sample indexing.

    Uses model.embed_with_features as the single source of truth for
    embedding + side-feature injection, so any new side feature added to
    embed_with_features is automatically active here.

    Returns:
        (B,) total log-probability per sample.
    """
    B = context_tokens.size(0)
    device = context_tokens.device
    ctx_sf = ctx_side_features or {}
    gen_sf = gen_side_features or {}

    sid_input = sid_tokens[:, :-1]  # (B, n_layers - 1)
    full_input = torch.cat([context_tokens, sid_input], dim=1)  # (B, T_ctx + L - 1)
    T = full_input.size(1)
    T_ctx = context_tokens.size(1)
    T_gen = T - T_ctx

    # Build per-token side features: context features + gen scalar repeated for SID positions
    full_sf = {}
    for key, ctx_feat in ctx_sf.items():
        if ctx_feat is not None:
            gen_val = gen_sf.get(key, 0)
            gen_feat = torch.full((B, T_gen), gen_val, dtype=ctx_feat.dtype, device=device)
            full_sf[key] = torch.cat([ctx_feat, gen_feat], dim=1)

    positions = torch.arange(T, device=device).unsqueeze(0)
    x = model.embed_with_features(full_input, positions, full_sf or None)
    rope_pos, _, rope_lay = model._build_rope_inputs(T, device)
    if model.use_rope:
        raw_ts = full_sf.get('timestamps')
        rope_ts = model._carry_forward_timestamps(raw_ts, T, device).expand(B, -1)
    else:
        rope_ts = None
    hidden = model._transformer_forward(x, positions=rope_pos, timestamps=rope_ts, layers=rope_lay)

    batch_idx = torch.arange(B, device=device)
    total_logprob = torch.zeros(B, device=device, dtype=hidden.dtype)
    for li in range(n_layers):
        pos = context_lengths - 1 + li  # (B,)
        h = hidden[batch_idx, pos]      # (B, D)
        logits = model.output_projs[li](h)
        lp = F.log_softmax(logits, dim=-1)
        total_logprob = total_logprob + lp.gather(1, sid_tokens[:, li:li + 1]).squeeze(1)

    return total_logprob


def _compute_chunk_logprobs(
    model,
    ctx_chunk: Tensor,
    len_chunk: Tensor,
    sids_chunk: Tensor,
    n_layers: int,
    ctx_sf_chunk: dict = None,
    gen_sf: dict = None,
) -> Tensor:
    """Thin wrapper for checkpointing — must be a plain function, not lambda."""
    return compute_sid_logprobs(
        model, ctx_chunk, len_chunk, sids_chunk, n_layers,
        ctx_side_features=ctx_sf_chunk, gen_side_features=gen_sf,
    )


def compute_sid_logprobs_batch(
    model,
    context_tokens: Tensor,        # (N, T_ctx) right-padded, pre-expanded
    context_lengths: Tensor,       # (N,) actual context lengths
    sid_tokens: Tensor,            # (N, n_layers) flat candidates
    n_layers: int,
    max_chunk: int = 64,
    ctx_side_features: dict = None,   # (N, T_ctx) per key, pre-expanded
    gen_side_features: dict = None,   # scalar per key
) -> Tensor:
    """Compute log-probs for N flat SID candidates (packed, no padding).

    Caller is responsible for expanding contexts to match candidates.
    Processes in micro-batches of max_chunk with gradient checkpointing
    when gradients are enabled (training).

    With checkpointing:
        Forward: intermediate activations discarded after each chunk.
        Backward: each chunk recomputed one at a time.
        Peak memory = 1 chunk ≈ 10GB, constant for any N.

    Checkpointing is only applied when torch.is_grad_enabled() is True
    (training). Reference model forward (under torch.no_grad) skips it.

    NOTE: when checkpointing is active, the caller MUST wrap both this
    call AND the subsequent backward() inside _freeze_moe_bias(model)
    to ensure the recompute during backward sees the same MoE router
    state. See trainer.py's DPO section and SparseMoEBlock docstring.

    Returns:
        (N,) log-probabilities.
    """
    N = sid_tokens.size(0)
    use_ckpt = torch.is_grad_enabled() and (N > max_chunk)
    ctx_sf = ctx_side_features or {}

    if N <= max_chunk:
        return compute_sid_logprobs(
            model, context_tokens, context_lengths, sid_tokens, n_layers,
            ctx_side_features=ctx_sf or None, gen_side_features=gen_side_features,
        )

    chunks = []
    for start in range(0, N, max_chunk):
        end = min(start + max_chunk, N)
        ctx_sf_chunk = {k: v[start:end] for k, v in ctx_sf.items()} if ctx_sf else None
        if use_ckpt:
            chunk_lp = torch_checkpoint(
                _compute_chunk_logprobs,
                model,
                context_tokens[start:end],
                context_lengths[start:end],
                sid_tokens[start:end],
                n_layers,
                ctx_sf_chunk,
                gen_side_features,
                use_reentrant=False,
            )
        else:
            chunk_lp = compute_sid_logprobs(
                model,
                context_tokens[start:end],
                context_lengths[start:end],
                sid_tokens[start:end],
                n_layers,
                ctx_side_features=ctx_sf_chunk,
                gen_side_features=gen_side_features,
            )
        chunks.append(chunk_lp)
    return torch.cat(chunks, dim=0)


def softmax_dpo_loss(
    policy_lp: Tensor,       # (N,) flat log pi_theta for all candidates
    ref_lp: Tensor,          # (N,) flat log pi_ref for all candidates
    sample_offsets: Tensor,  # (B+1,) boundaries; [off[i]] = chosen, [off[i]+1:off[i+1]] = rejected
    beta: float = 0.1,
    return_diagnostics: bool = False,
):
    """Softmax-DPO loss with packed candidates (no padding).

    Supports 1 chosen vs variable N rejected per sample.
    Each sample's candidates are contiguous in the flat arrays:
      sample i: chosen at index offsets[i], rejected at offsets[i]+1 : offsets[i+1]

    Formula:
        r_l = beta * (log pi_theta(y_l)/pi_ref(y_l) - log pi_theta(y_w)/pi_ref(y_w))
        L = -mean_b[ log sigmoid(-logsumexp_l(r_l)) ]

    All computation in fp32 for numerical stability.

    Returns:
        Scalar loss (default), or (loss, diagnostics_dict) when return_diagnostics=True.
        Diagnostics are detached scalars for logging — no gradient impact.
    """
    adv = (policy_lp - ref_lp).float()  # (N,)
    offsets = sample_offsets.tolist()
    B = len(offsets) - 1

    losses = []
    chosen_rewards = []
    rejected_rewards = []
    wins = 0
    n_pairs = 0

    for i in range(B):
        start, end = int(offsets[i]), int(offsets[i + 1])
        if end - start <= 1:
            continue  # no rejected for this sample
        chosen_adv = adv[start]
        rejected_adv = adv[start + 1:end]
        r = beta * (rejected_adv - chosen_adv)
        lse = torch.logsumexp(r, dim=0)
        losses.append(-F.logsigmoid(-lse))

        if return_diagnostics:
            chosen_rewards.append(chosen_adv.detach())
            rejected_rewards.append(rejected_adv.detach().mean())
            wins += int((chosen_adv > rejected_adv.max()).item())
            n_pairs += 1

    if not losses:
        zero = torch.tensor(0.0, device=policy_lp.device, requires_grad=True)
        if return_diagnostics:
            return zero, {}
        return zero

    loss = torch.stack(losses).mean()

    if not return_diagnostics:
        return loss

    cr = torch.stack(chosen_rewards)
    rr = torch.stack(rejected_rewards)
    diagnostics = {
        'chosen_reward': cr.mean().item(),
        'rejected_reward': rr.mean().item(),
        'reward_margin': (cr - rr).mean().item(),
        'preference_acc': wins / n_pairs if n_pairs > 0 else 0.0,
        'kl': adv.detach().mean().item(),
    }
    return loss, diagnostics
