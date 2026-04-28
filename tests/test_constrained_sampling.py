"""Tests for constrained_sampling() in ntp/model.py.

Covers:
  - Output shape and SIDTrie validity (every candidate is a real SID)
  - Deduplication
  - Temperature effect (T→0 concentrates, T→∞ spreads)
  - Comparison with constrained_beam_search (same interface)
  - Dead-path handling (trie with very few valid paths)
  - log-prob scores are finite and consistent
"""

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntp.model import SIDTrie, constrained_sampling, constrained_beam_search, NTPModel


# ── Minimal model + trie fixtures ───────────────────────────────────────────

def _make_small_model():
    """Tiny NTPModel for fast CPU testing."""
    model = NTPModel(
        n_clusters_per_layer=[16, 16, 16],
        n_sid_layers=3,
        n_items=10,
        embed_dim=32,
        n_heads=2,
        n_transformer_layers=2,
        n_experts=2,
        expert_dim=64,
        top_k=1,
        dropout=0.0,
    )
    model.eval()
    return model


def _make_trie(sids, n_layers=3):
    """Build SIDTrie from list of 'a_b_c' strings."""
    return SIDTrie({s: set() for s in sids}, n_layers=n_layers)


def _dummy_ctx(seq_len=8, vocab=16):
    """(1, seq_len) context tensor."""
    return torch.randint(0, vocab, (1, seq_len))


# ── Tests ────────────────────────────────────────────────────────────────────

def test_sampling_output_shape():
    """constrained_sampling returns (1, ≤n_samples, n_layers)."""
    torch.manual_seed(0)
    model = _make_small_model()
    sids = [f"{i}_{j}_{k}" for i in range(4) for j in range(4) for k in range(4)]
    trie = _make_trie(sids)
    ctx = _dummy_ctx()

    beams, scores, _ = constrained_sampling(model, ctx, trie, n_samples=20, temperature=1.0)

    assert beams.dim() == 3, f"Expected 3D tensor, got {beams.dim()}D"
    assert beams.size(0) == 1
    assert beams.size(2) == 3
    assert beams.size(1) <= 20
    assert scores.shape == (1, beams.size(1))
    print(f"  [PASS] output shape (1, {beams.size(1)}, 3), scores {scores.shape}")


def test_sampling_all_valid_sids():
    """Every returned candidate exists in the trie."""
    torch.manual_seed(1)
    model = _make_small_model()
    sids = ["0_0_0", "0_0_1", "0_1_0", "1_0_0", "1_1_1", "2_3_0", "3_2_1"]
    trie = _make_trie(sids)
    sid_set = set(tuple(int(x) for x in s.split("_")) for s in sids)
    ctx = _dummy_ctx()

    beams, scores, _ = constrained_sampling(model, ctx, trie, n_samples=50, temperature=1.0)

    cands = beams[0]  # (n, 3)
    for i in range(cands.size(0)):
        tup = tuple(cands[i].tolist())
        assert tup in sid_set, f"Candidate {tup} not in trie"
    print(f"  [PASS] all {cands.size(0)} candidates are valid SIDs")


def test_sampling_scores_finite():
    """Returned log-prob scores are finite."""
    torch.manual_seed(2)
    model = _make_small_model()
    sids = [f"{i}_{j}_{k}" for i in range(3) for j in range(3) for k in range(3)]
    trie = _make_trie(sids)
    ctx = _dummy_ctx()

    beams, scores, _ = constrained_sampling(model, ctx, trie, n_samples=30, temperature=1.0)

    assert torch.isfinite(scores).all(), f"Non-finite scores: {scores}"
    assert (scores <= 0).all(), "Log-probs should be ≤ 0"
    print(f"  [PASS] all scores finite and ≤ 0")


def test_sampling_deduplication():
    """With very low temperature (near-greedy), many samples collapse → dedup reduces count."""
    torch.manual_seed(3)
    model = _make_small_model()
    # Small trie → high collision rate at low T
    sids = ["0_0_0", "0_0_1", "0_1_0"]
    trie = _make_trie(sids)
    ctx = _dummy_ctx()

    beams_low_T, _, _ = constrained_sampling(
        model, ctx, trie, n_samples=100, temperature=0.01)
    beams_high_T, _, _ = constrained_sampling(
        model, ctx, trie, n_samples=100, temperature=5.0)

    # Low T: most samples identical → few unique after dedup
    # High T: uniform → more unique (up to len(sids)=3)
    assert beams_low_T.size(1) <= beams_high_T.size(1) or beams_high_T.size(1) == len(sids), \
        f"Low T={beams_low_T.size(1)} should have ≤ unique candidates than high T={beams_high_T.size(1)}"
    print(f"  [PASS] dedup: low-T={beams_low_T.size(1)} unique, high-T={beams_high_T.size(1)} unique")


def test_sampling_temperature_coverage():
    """Higher temperature explores more of the trie (more unique L0 tokens)."""
    torch.manual_seed(4)
    model = _make_small_model()
    # Many L0 options but one dominates at low T
    sids = [f"{i}_{j}_{k}" for i in range(8) for j in range(2) for k in range(2)]
    trie = _make_trie(sids)
    ctx = _dummy_ctx()

    beams_low,  _, _ = constrained_sampling(model, ctx, trie, n_samples=200, temperature=0.1)
    beams_high, _, _ = constrained_sampling(model, ctx, trie, n_samples=200, temperature=2.0)

    l0_low  = beams_low[0, :, 0].unique().numel()
    l0_high = beams_high[0, :, 0].unique().numel()
    assert l0_high >= l0_low, \
        f"High-T should cover ≥ L0 tokens: high={l0_high}, low={l0_low}"
    print(f"  [PASS] L0 coverage: low-T={l0_low}, high-T={l0_high}")


def test_sampling_same_interface_as_beam():
    """constrained_sampling and constrained_beam_search return compatible shapes."""
    torch.manual_seed(5)
    model = _make_small_model()
    sids = [f"{i}_{j}_{k}" for i in range(4) for j in range(4) for k in range(4)]
    trie = _make_trie(sids)
    ctx = _dummy_ctx()

    beams_s, scores_s, kv_s = constrained_sampling(
        model, ctx, trie, n_samples=16, temperature=1.0)
    beams_b, scores_b, kv_b = constrained_beam_search(
        model, ctx, trie, beam_size=16)

    # Both return (1, k, 3) beams and (1, k) scores
    assert beams_s.dim() == beams_b.dim() == 3
    assert beams_s.size(0) == beams_b.size(0) == 1
    assert beams_s.size(2) == beams_b.size(2) == 3
    assert scores_s.dim() == scores_b.dim() == 2
    print(f"  [PASS] compatible shapes: sampling={tuple(beams_s.shape)}, beam={tuple(beams_b.shape)}")


def test_sampling_ctx_kv_cache_reuse():
    """ctx_kv_caches returned by first call can be reused in second call."""
    torch.manual_seed(6)
    model = _make_small_model()
    sids = [f"{i}_{j}_{k}" for i in range(4) for j in range(4) for k in range(4)]
    trie = _make_trie(sids)
    ctx = _dummy_ctx()

    # First call — computes kv cache
    beams1, scores1, kv = constrained_sampling(
        model, ctx, trie, n_samples=10, temperature=1.0)

    # Second call — reuses kv cache (must provide initial_logits too)
    initial_logits, kv2 = model.forward_cached(ctx)
    beams2, scores2, _ = constrained_sampling(
        model, ctx, trie, n_samples=10, temperature=1.0,
        ctx_kv_caches=kv2, initial_logits=initial_logits)

    assert beams2.size(2) == 3
    assert torch.isfinite(scores2).all()
    print(f"  [PASS] ctx_kv_cache reuse works, got {beams2.size(1)} candidates")


def test_sampling_single_valid_sid():
    """Trie with only one valid SID always returns that SID."""
    torch.manual_seed(7)
    model = _make_small_model()
    sids = ["2_3_1"]
    trie = _make_trie(sids)
    ctx = _dummy_ctx()

    beams, scores, _ = constrained_sampling(
        model, ctx, trie, n_samples=20, temperature=1.0)

    assert beams.size(1) == 1, f"Should get exactly 1 unique SID, got {beams.size(1)}"
    tup = tuple(beams[0, 0].tolist())
    assert tup == (2, 3, 1), f"Wrong SID: {tup}"
    print(f"  [PASS] single-SID trie always returns that SID")


def test_sampling_n_samples_cap():
    """Can't return more unique candidates than trie has SIDs."""
    torch.manual_seed(8)
    model = _make_small_model()
    sids = ["0_0_0", "1_1_1", "2_2_2"]  # only 3 SIDs
    trie = _make_trie(sids)
    ctx = _dummy_ctx()

    beams, _, _ = constrained_sampling(
        model, ctx, trie, n_samples=1000, temperature=1.0)

    assert beams.size(1) <= 3, f"Cannot have > 3 unique SIDs, got {beams.size(1)}"
    print(f"  [PASS] unique count ({beams.size(1)}) capped by trie size (3)")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Constrained Sampling Tests")
    print("=" * 50)
    test_sampling_output_shape()
    test_sampling_all_valid_sids()
    test_sampling_scores_finite()
    test_sampling_deduplication()
    test_sampling_temperature_coverage()
    test_sampling_same_interface_as_beam()
    test_sampling_ctx_kv_cache_reuse()
    test_sampling_single_valid_sid()
    test_sampling_n_samples_cap()
    print("=" * 50)
    print("All tests passed!")
