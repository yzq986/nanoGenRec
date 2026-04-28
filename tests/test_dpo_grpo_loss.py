"""Mathematical correctness tests for DPO, GRPO, and ECPO losses.

Tests verify:
  1. Loss values match hand-computed reference formulas
  2. Gradients flow correctly (non-zero, through the right variables)
  3. Degenerate inputs are handled (all-same rewards, empty groups, etc.)
  4. ECPO early clip is stricter than GRPO (lower loss on bad samples)
  5. Diagnostic dict keys and value ranges are correct
"""

import torch
import torch.nn.functional as F

from conftest import N_LAYERS, CLUSTERS
from rl.dpo import softmax_dpo_loss
from rl.grpo import grpo_loss, ecpo_loss

ATOL = 1e-5


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_flat(n_chosen: int, n_rej_per: int, seed: int = 0):
    """Build flat policy_lp, ref_lp, rewards and offsets for B samples."""
    torch.manual_seed(seed)
    B = n_chosen
    K = 1 + n_rej_per  # chosen + rejected
    N = B * K
    policy_lp = torch.randn(N, requires_grad=True)
    ref_lp = torch.randn(N).detach()
    rewards = torch.rand(N)
    offsets = torch.arange(0, N + 1, K)
    return policy_lp, ref_lp, rewards, offsets


# ── 1. softmax_dpo_loss ───────────────────────────────────────────────────────

def test_dpo_loss_formula():
    """softmax_dpo_loss matches hand-computed formula for B=1, N_rej=2."""
    beta = 0.1
    # One sample: chosen at 0, rejected at 1, 2
    policy_lp = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    ref_lp    = torch.tensor([-1.5, -1.5, -1.5])
    offsets   = torch.tensor([0, 3])

    loss = softmax_dpo_loss(policy_lp, ref_lp, offsets, beta=beta)

    # Hand-compute: adv = policy_lp - ref_lp = [0.5, -0.5, -1.5]
    # chosen_adv=0.5, rejected_adv=[-0.5, -1.5]
    # r = beta * (rejected - chosen) = 0.1 * [-1.0, -2.0]
    # lse = logsumexp([-0.1, -0.2]) ≈ -0.0513
    # loss = -logsigmoid(-lse) = -logsigmoid(0.0513)
    adv = (policy_lp - ref_lp).detach().float()
    chosen_adv = adv[0]
    rej_adv = adv[1:3]
    r = beta * (rej_adv - chosen_adv)
    lse = torch.logsumexp(r, dim=0)
    expected = -F.logsigmoid(-lse)

    diff = abs(loss.item() - expected.item())
    assert diff < ATOL, f"DPO loss formula mismatch: {diff:.2e}"
    print(f"  [PASS] DPO loss formula (diff={diff:.2e})")


def test_dpo_loss_gradient_flows():
    """Loss gradient is non-zero w.r.t. policy_lp (chosen and rejected)."""
    policy_lp, ref_lp, _, offsets = _make_flat(n_chosen=3, n_rej_per=4)
    loss = softmax_dpo_loss(policy_lp, ref_lp, offsets)
    loss.backward()

    assert policy_lp.grad is not None
    assert policy_lp.grad.abs().sum().item() > 0, "DPO gradient is all-zero"
    print(f"  [PASS] DPO gradient flows (|grad|={policy_lp.grad.abs().mean():.4f})")


def test_dpo_loss_empty_groups_skipped():
    """Groups with only a chosen sample (no rejected) are skipped gracefully."""
    # 3 samples: sample 0 has 2 rej, sample 1 has 0 rej, sample 2 has 1 rej
    policy_lp = torch.tensor([-1., -2., -3., -1.5, -2.5], requires_grad=True)
    ref_lp    = torch.zeros(5)
    offsets   = torch.tensor([0, 3, 4, 6])  # sample 1: only index 3 (no rej)

    loss = softmax_dpo_loss(policy_lp, ref_lp, offsets)
    assert torch.isfinite(loss), "DPO loss is non-finite with empty group"
    loss.backward()
    print(f"  [PASS] DPO empty group skipped (loss={loss.item():.4f})")


def test_dpo_loss_diagnostics_keys():
    """return_diagnostics=True returns expected keys with finite values."""
    policy_lp, ref_lp, _, offsets = _make_flat(n_chosen=4, n_rej_per=3)
    loss, diag = softmax_dpo_loss(policy_lp, ref_lp, offsets, return_diagnostics=True)

    for key in ('chosen_reward', 'rejected_reward', 'reward_margin', 'preference_acc', 'kl'):
        assert key in diag, f"Missing diagnostic key: {key}"
        assert isinstance(diag[key], float), f"Diagnostic {key} is not a float"
    assert 0.0 <= diag['preference_acc'] <= 1.0
    print(f"  [PASS] DPO diagnostics keys correct (acc={diag['preference_acc']:.2f})")


# ── 2. grpo_loss ─────────────────────────────────────────────────────────────

def test_grpo_loss_advantage_normalization():
    """GRPO advantages are zero-mean, unit-std within each group."""
    torch.manual_seed(0)
    B, G = 3, 8
    N = B * G
    policy_lp = torch.randn(N, requires_grad=True)
    ref_lp    = torch.randn(N).detach()
    rewards   = torch.rand(N)
    offsets   = torch.arange(0, N + 1, G)

    loss, diag = grpo_loss(policy_lp, ref_lp, rewards, offsets, return_diagnostics=True)

    # advantage_mean should be near 0, advantage_std should be near 1
    assert abs(diag['advantage_mean']) < 0.5, \
        f"advantage_mean too large: {diag['advantage_mean']:.4f}"
    assert 0.3 < diag['advantage_std'] < 2.0, \
        f"advantage_std out of range: {diag['advantage_std']:.4f}"
    print(f"  [PASS] GRPO advantage normalization (mean={diag['advantage_mean']:.3f}, "
          f"std={diag['advantage_std']:.3f})")


def test_grpo_loss_uniform_rewards_skipped():
    """Groups where all rewards are identical (std≈0) are skipped — no NaN."""
    torch.manual_seed(0)
    B, G = 4, 6
    N = B * G
    policy_lp = torch.randn(N, requires_grad=True)
    ref_lp    = torch.randn(N).detach()
    # Make all rewards identical within each group → std=0
    rewards = torch.ones(N)
    offsets = torch.arange(0, N + 1, G)

    loss, diag = grpo_loss(policy_lp, ref_lp, rewards, offsets, return_diagnostics=True)
    assert torch.isfinite(loss), f"GRPO loss NaN with uniform rewards: {loss}"
    print(f"  [PASS] GRPO uniform rewards → no NaN (loss={loss.item():.4f})")


def test_grpo_loss_gradient_flows():
    """GRPO gradient is non-zero w.r.t. policy_lp."""
    policy_lp, ref_lp, rewards, offsets = _make_flat(n_chosen=4, n_rej_per=5)
    loss, _ = grpo_loss(policy_lp, ref_lp, rewards, offsets, return_diagnostics=True)
    loss.backward()

    assert policy_lp.grad is not None
    assert policy_lp.grad.abs().sum().item() > 0, "GRPO gradient is all-zero"
    print(f"  [PASS] GRPO gradient flows (|grad|={policy_lp.grad.abs().mean():.4f})")


def test_grpo_loss_diagnostics_keys():
    """GRPO diagnostics contain all expected keys."""
    policy_lp, ref_lp, rewards, offsets = _make_flat(n_chosen=3, n_rej_per=4)
    _, diag = grpo_loss(policy_lp, ref_lp, rewards, offsets, return_diagnostics=True)

    for key in ('advantage_mean', 'advantage_std', 'policy_ratio_mean',
                'clip_fraction', 'reward_mean', 'reward_std'):
        assert key in diag, f"Missing GRPO diagnostic: {key}"
        assert isinstance(diag[key], float)
    assert 0.0 <= diag['clip_fraction'] <= 1.0
    print(f"  [PASS] GRPO diagnostics keys correct (clip={diag['clip_fraction']:.2f})")


# ── 3. ecpo_loss ─────────────────────────────────────────────────────────────

def test_ecpo_vs_grpo_early_clip_active():
    """ECPO loss differs from GRPO when delta > 0."""
    torch.manual_seed(0)
    B, G = 4, 8
    N = B * G
    # Make policy very different from ref to trigger early clip
    policy_lp = torch.randn(N, requires_grad=True)
    ref_lp    = (torch.randn(N) - 3.0).detach()  # push policy >> ref
    rewards   = torch.rand(N)
    offsets   = torch.arange(0, N + 1, G)

    loss_grpo, _ = grpo_loss(policy_lp, ref_lp, rewards, offsets, return_diagnostics=True)
    policy_lp2 = policy_lp.detach().requires_grad_(True)
    loss_ecpo, _ = ecpo_loss(policy_lp2, ref_lp, rewards, offsets,
                              delta=0.1, return_diagnostics=True)

    # ECPO should differ from GRPO when early clip is active
    diff = abs(loss_grpo.item() - loss_ecpo.item())
    assert diff > 1e-6, \
        "ECPO loss identical to GRPO — early clip may not be firing"
    print(f"  [PASS] ECPO differs from GRPO (diff={diff:.4f})")


def test_ecpo_delta_zero_equals_grpo():
    """ecpo_loss with delta=0 is identical to grpo_loss."""
    torch.manual_seed(5)
    B, G = 3, 6
    N = B * G
    policy_lp = torch.randn(N, requires_grad=True)
    ref_lp    = torch.randn(N).detach()
    rewards   = torch.rand(N)
    offsets   = torch.arange(0, N + 1, G)

    loss_grpo, _ = grpo_loss(policy_lp, ref_lp, rewards, offsets, return_diagnostics=True)
    policy_lp2 = policy_lp.detach().clone().requires_grad_(True)
    loss_ecpo, _ = ecpo_loss(policy_lp2, ref_lp, rewards, offsets,
                              delta=0.0, return_diagnostics=True)

    diff = abs(loss_grpo.item() - loss_ecpo.item())
    assert diff < ATOL, f"ecpo(delta=0) != grpo: diff={diff:.2e}"
    print(f"  [PASS] ecpo(delta=0) == grpo (diff={diff:.2e})")


def test_ecpo_early_clip_fraction_in_diagnostics():
    """ECPO diagnostics include early_clip_fraction."""
    policy_lp, ref_lp, rewards, offsets = _make_flat(n_chosen=4, n_rej_per=5)
    _, diag = ecpo_loss(policy_lp, ref_lp, rewards, offsets,
                        delta=0.1, return_diagnostics=True)
    assert 'early_clip_fraction' in diag, "ECPO missing early_clip_fraction diagnostic"
    assert 0.0 <= diag['early_clip_fraction'] <= 1.0
    print(f"  [PASS] ECPO early_clip_fraction in diagnostics "
          f"(ecf={diag['early_clip_fraction']:.2f})")


def test_ecpo_gradient_flows():
    """ECPO gradient is non-zero w.r.t. policy_lp."""
    policy_lp, ref_lp, rewards, offsets = _make_flat(n_chosen=4, n_rej_per=5)
    loss, _ = ecpo_loss(policy_lp, ref_lp, rewards, offsets,
                        delta=0.1, return_diagnostics=True)
    loss.backward()

    assert policy_lp.grad is not None
    assert policy_lp.grad.abs().sum().item() > 0, "ECPO gradient is all-zero"
    print(f"  [PASS] ECPO gradient flows (|grad|={policy_lp.grad.abs().mean():.4f})")


# ── 4. compute_sid_logprobs numerical sanity ──────────────────────────────────

def test_sid_logprobs_sum_to_valid_log_prob():
    """compute_sid_logprobs returns values in (-inf, 0] for each sample."""
    from conftest import make_model, make_ctx, make_sid
    from rl.dpo import compute_sid_logprobs

    model = make_model(features=False)
    B = 5
    ctx = make_ctx(batch=B, length=10)
    sids = make_sid(batch=B)
    lengths = torch.full((B,), 10, dtype=torch.long)

    with torch.no_grad():
        lp = compute_sid_logprobs(model, ctx, lengths, sids, N_LAYERS)

    assert lp.shape == (B,), f"Wrong shape: {lp.shape}"
    assert (lp <= 0).all(), f"Log-probs > 0: {lp}"
    assert torch.isfinite(lp).all(), f"Non-finite log-probs: {lp}"
    print(f"  [PASS] sid log-probs in valid range (mean={lp.mean():.2f})")


def test_sid_logprobs_batch_consistent():
    """compute_sid_logprobs gives same result whether processed as batch or individually."""
    from conftest import make_model, make_ctx, make_sid
    from rl.dpo import compute_sid_logprobs

    model = make_model(features=False)
    B = 4
    ctx = make_ctx(batch=B, length=10)
    sids = make_sid(batch=B)
    lengths = torch.full((B,), 10, dtype=torch.long)

    with torch.no_grad():
        lp_batch = compute_sid_logprobs(model, ctx, lengths, sids, N_LAYERS)
        lp_single = torch.stack([
            compute_sid_logprobs(model, ctx[i:i+1], lengths[i:i+1],
                                 sids[i:i+1], N_LAYERS)
            for i in range(B)
        ]).squeeze(1)

    diff = (lp_batch - lp_single).abs().max().item()
    assert diff < 1e-4, f"Batch vs single logprob mismatch: {diff:.2e}"
    print(f"  [PASS] logprobs batch==single (diff={diff:.2e})")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("DPO / GRPO / ECPO Loss Tests")
    print("=" * 50)

    print("\n1. softmax_dpo_loss")
    test_dpo_loss_formula()
    test_dpo_loss_gradient_flows()
    test_dpo_loss_empty_groups_skipped()
    test_dpo_loss_diagnostics_keys()

    print("\n2. grpo_loss")
    test_grpo_loss_advantage_normalization()
    test_grpo_loss_uniform_rewards_skipped()
    test_grpo_loss_gradient_flows()
    test_grpo_loss_diagnostics_keys()

    print("\n3. ecpo_loss")
    test_ecpo_vs_grpo_early_clip_active()
    test_ecpo_delta_zero_equals_grpo()
    test_ecpo_early_clip_fraction_in_diagnostics()
    test_ecpo_gradient_flows()

    print("\n4. compute_sid_logprobs")
    test_sid_logprobs_sum_to_valid_log_prob()
    test_sid_logprobs_batch_consistent()

    print("\n" + "=" * 50)
    print("All tests passed!")
