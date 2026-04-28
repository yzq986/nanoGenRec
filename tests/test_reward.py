"""Tests for rl/reward.py — pluggable reward function system.

Covers:
  - BehaviorReward: lookup, prefix cascade fallback, default
  - FormatReward: legality check via SIDTrie, sample_k sub-sampling
  - ExternalReward / BusinessReward: callable adapter, metrics
  - WeightedBehaviorReward: quality × freshness, HEPO prefix scaling
  - _bitmap_to_quality: production scoring formula
  - CompositeReward: weighted sum, metrics namespacing
"""

import math
import time

import torch

from conftest import make_trie, N_LAYERS, CLUSTERS
from rl.reward import (
    BehaviorReward,
    BusinessReward,
    CompositeReward,
    ExternalReward,
    FormatReward,
    WeightedBehaviorReward,
    _bitmap_to_quality,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dummy_ctx(N: int = 4, T: int = 6):
    ctx = torch.zeros(N, T, dtype=torch.long)
    lengths = torch.full((N,), T, dtype=torch.long)
    return ctx, lengths


def _sids(tuples):
    """Convert list of tuples to (N, n_layers) tensor."""
    return torch.tensor(list(tuples), dtype=torch.long)


# ── BehaviorReward ────────────────────────────────────────────────────────────

def test_behavior_reward_full_match():
    """Full SID tuple match returns exact score."""
    scores = {(1, 2, 3): 0.8, (0, 0, 0): 0.1}
    r = BehaviorReward(scores)
    ctx, lengths = _dummy_ctx(2)
    sids = _sids([(1, 2, 3), (0, 0, 0)])
    out = r(sids, ctx, lengths)
    assert abs(out[0].item() - 0.8) < 1e-5
    assert abs(out[1].item() - 0.1) < 1e-5
    print(f"  [PASS] BehaviorReward full match")


def test_behavior_reward_prefix_fallback():
    """Missing full match falls back to shorter prefix with scale."""
    prefix_scale = 0.5
    scores = {(1, 2): 0.6}  # only L0L1 prefix
    r = BehaviorReward(scores, prefix_scale=prefix_scale)
    ctx, lengths = _dummy_ctx(1)
    sids = _sids([(1, 2, 9)])  # full SID not in scores
    out = r(sids, ctx, lengths)
    # Should fall back 1 level: scale = 0.5^1 = 0.5
    expected = 0.6 * (prefix_scale ** 1)
    assert abs(out[0].item() - expected) < 1e-5
    print(f"  [PASS] BehaviorReward prefix fallback (got {out[0].item():.4f}, expected {expected:.4f})")


def test_behavior_reward_default():
    """No match at any depth returns default_reward."""
    r = BehaviorReward({}, default_reward=0.0)
    ctx, lengths = _dummy_ctx(3)
    sids = _sids([(5, 6, 7), (1, 2, 3), (0, 0, 0)])
    out = r(sids, ctx, lengths)
    assert (out == 0.0).all()
    print(f"  [PASS] BehaviorReward default reward")


def test_behavior_reward_metrics():
    """metrics() returns 'mean' key after __call__."""
    scores = {(1, 2, 3): 1.0, (0, 0, 0): 0.0}
    r = BehaviorReward(scores)
    ctx, lengths = _dummy_ctx(2)
    sids = _sids([(1, 2, 3), (0, 0, 0)])
    r(sids, ctx, lengths)
    m = r.metrics()
    assert 'mean' in m
    assert abs(m['mean'] - 0.5) < 1e-5
    print(f"  [PASS] BehaviorReward metrics mean=0.5")


# ── FormatReward ──────────────────────────────────────────────────────────────

def test_format_reward_legal_sids():
    """Valid SIDs from trie get reward 1.0."""
    trie = make_trie(n_items=80)
    r = FormatReward(trie, N_LAYERS)

    # Grab some known-valid SIDs from the trie (use children[0] to list L0 tokens)
    valid_sids = []
    for l0 in list(trie.valid_tokens(0, ()))[:4]:
        l1_tokens = list(trie.valid_tokens(1, (l0,)))
        if not l1_tokens:
            continue
        l1 = l1_tokens[0]
        l2_tokens = list(trie.valid_tokens(2, (l0, l1)))
        if not l2_tokens:
            continue
        valid_sids.append([l0, l1, l2_tokens[0]])
    if not valid_sids:
        print(f"  [SKIP] FormatReward legal SIDs — trie too sparse")
        return

    ctx, lengths = _dummy_ctx(len(valid_sids))
    sids = torch.tensor(valid_sids, dtype=torch.long)
    out = r(sids, ctx, lengths)
    assert (out == 1.0).all(), f"Legal SIDs should get 1.0, got {out}"
    print(f"  [PASS] FormatReward legal SIDs → 1.0")


def test_format_reward_illegal_sids():
    """SIDs not in trie get reward 0.0."""
    trie = make_trie(n_items=20)
    r = FormatReward(trie, N_LAYERS)
    # Use token values well outside trie range
    ctx, lengths = _dummy_ctx(2)
    sids = _sids([(99, 99, 99), (88, 88, 88)])
    out = r(sids, ctx, lengths)
    assert (out == 0.0).all(), f"Illegal SIDs should get 0.0, got {out}"
    print(f"  [PASS] FormatReward illegal SIDs → 0.0")


def test_format_reward_sample_k():
    """sample_k sub-sampling only marks sampled indices."""
    trie = make_trie(n_items=40)
    r = FormatReward(trie, N_LAYERS, sample_k=2)
    ctx, lengths = _dummy_ctx(10)
    # All illegal SIDs — no matter which 2 are sampled, result should be 0
    sids = _sids([(99, 99, 99)] * 10)
    out = r(sids, ctx, lengths)
    # Non-sampled indices stay 0 (default); sampled illegal also 0
    assert (out == 0.0).all()
    print(f"  [PASS] FormatReward sample_k sub-sampling")


def test_format_reward_metrics():
    """metrics() returns legal_rate key."""
    trie = make_trie(n_items=40)
    r = FormatReward(trie, N_LAYERS)
    ctx, lengths = _dummy_ctx(1)
    sids = _sids([(99, 99, 99)])
    r(sids, ctx, lengths)
    m = r.metrics()
    assert 'legal_rate' in m
    assert m['legal_rate'] == 0.0
    print(f"  [PASS] FormatReward metrics legal_rate")


# ── ExternalReward / BusinessReward ───────────────────────────────────────────

def test_external_reward_callable():
    """ExternalReward passes correct list types to wrapped fn."""
    received = {}

    def my_fn(sids_list, ctxs_list):
        received['sids'] = sids_list
        received['ctxs'] = ctxs_list
        return [float(i) for i in range(len(sids_list))]

    r = ExternalReward(my_fn, name='test')
    ctx, lengths = _dummy_ctx(3, T=5)
    sids = torch.randint(0, 10, (3, N_LAYERS))
    out = r(sids, ctx, lengths)

    assert isinstance(received['sids'], list)
    assert isinstance(received['sids'][0], list)
    assert out.shape == (3,)
    assert list(out.cpu().numpy()) == [0.0, 1.0, 2.0]
    print(f"  [PASS] ExternalReward callable adapter")


def test_business_reward_same_as_external():
    """BusinessReward behaves identically to ExternalReward."""
    fn = lambda sids, ctxs: [1.0] * len(sids)
    r = BusinessReward(fn, name='timeliness')
    ctx, lengths = _dummy_ctx(4)
    sids = torch.zeros(4, N_LAYERS, dtype=torch.long)
    out = r(sids, ctx, lengths)
    assert (out == 1.0).all()
    assert 'mean' in r.metrics()
    print(f"  [PASS] BusinessReward")


# ── _bitmap_to_quality ────────────────────────────────────────────────────────

def test_bitmap_quality_negative_feedback():
    """Negative feedback bit returns -1.0."""
    neg_bitmap = -2147483648
    assert _bitmap_to_quality(neg_bitmap) == -1.0
    print(f"  [PASS] _bitmap_to_quality negative feedback → -1.0")


def test_bitmap_quality_zero():
    """Zero bitmap (no action) returns 0.0."""
    assert _bitmap_to_quality(0) == 0.0
    print(f"  [PASS] _bitmap_to_quality zero bitmap → 0.0")


def test_bitmap_quality_place_order():
    """place_order bit (262144) returns log10(1 + 4000) ≈ 3.602."""
    q = _bitmap_to_quality(262144)
    expected = math.log10(1 + 4000.0)
    assert abs(q - expected) < 1e-5, f"place_order quality: {q:.4f} != {expected:.4f}"
    print(f"  [PASS] _bitmap_to_quality place_order ≈ {expected:.4f}")


def test_bitmap_quality_additive():
    """Multiple bits accumulate additively before log."""
    # like (2, w=1.0) + share (4, w=3.0) = 4.0 → log10(5) ≈ 0.699
    bm = 2 | 4
    q = _bitmap_to_quality(bm)
    expected = math.log10(1 + 1.0 + 3.0)
    assert abs(q - expected) < 1e-5
    print(f"  [PASS] _bitmap_to_quality additive bits ≈ {expected:.4f}")


# ── WeightedBehaviorReward ────────────────────────────────────────────────────

def test_weighted_behavior_reward_full_match():
    """Full SID match: reward = quality × freshness."""
    now = time.time()
    age_hours = 10.0
    ts = now - age_hours * 3600
    bitmap = 2  # like (w=1.0)
    quality = math.log10(1 + 1.0)
    freshness = math.exp(-age_hours / 24.0)
    expected = quality * freshness

    sid_to_info = {(1, 2, 3): (bitmap, ts)}
    r = WeightedBehaviorReward(sid_to_info, eval_ts=now)
    ctx, lengths = _dummy_ctx(1)
    sids = _sids([(1, 2, 3)])
    out = r(sids, ctx, lengths)
    assert abs(out[0].item() - expected) < 1e-4, \
        f"WeightedBehaviorReward: {out[0].item():.6f} != {expected:.6f}"
    print(f"  [PASS] WeightedBehaviorReward full match (reward={out[0].item():.4f})")


def test_weighted_behavior_reward_hepo_prefix():
    """HEPO prefix fallback uses hepo_scales, not generic prefix_scale."""
    now = time.time()
    bitmap = 2  # like
    ts = now  # fresh
    quality = math.log10(1 + 1.0)

    # Only L0L1 prefix in dict (match_len=2 for 3-layer SID)
    sid_to_info = {(1, 2): (bitmap, ts)}
    hepo_scales = [0.1, 0.5]  # idx 0 = L0, idx 1 = L0L1
    r = WeightedBehaviorReward(sid_to_info, eval_ts=now, hepo_scales=hepo_scales)
    ctx, lengths = _dummy_ctx(1)
    sids = _sids([(1, 2, 9)])  # full SID not found, falls back to L0L1
    out = r(sids, ctx, lengths)

    # match_len=2 → idx = 2-1 = 1 → hepo_scales[1] = 0.5
    expected = quality * 1.0 * hepo_scales[1]  # freshness ≈ 1 at t=0
    assert abs(out[0].item() - expected) < 1e-3, \
        f"HEPO prefix: {out[0].item():.4f} != {expected:.4f}"
    print(f"  [PASS] WeightedBehaviorReward HEPO prefix scale")


def test_weighted_behavior_reward_coverage():
    """metrics() reports coverage (fraction of non-default rewards)."""
    now = time.time()
    sid_to_info = {(1, 2, 3): (2, now)}
    r = WeightedBehaviorReward(sid_to_info, eval_ts=now)
    ctx, lengths = _dummy_ctx(4)
    # 1 hit, 3 miss
    sids = _sids([(1, 2, 3), (0, 0, 0), (0, 0, 0), (0, 0, 0)])
    r(sids, ctx, lengths)
    m = r.metrics()
    assert 'coverage' in m, f"Missing coverage in metrics: {m}"
    assert abs(m['coverage'] - 0.25) < 0.01
    print(f"  [PASS] WeightedBehaviorReward coverage metric")


# ── CompositeReward ────────────────────────────────────────────────────────────

def test_composite_reward_weighted_sum():
    """CompositeReward output = weighted sum of components."""
    r1 = ExternalReward(lambda s, c: [1.0] * len(s), name='a')
    r2 = ExternalReward(lambda s, c: [2.0] * len(s), name='b')
    comp = CompositeReward([('a', 0.5, r1), ('b', 0.3, r2)])
    ctx, lengths = _dummy_ctx(3)
    sids = torch.zeros(3, N_LAYERS, dtype=torch.long)
    out = comp(sids, ctx, lengths)
    expected = 0.5 * 1.0 + 0.3 * 2.0  # = 1.1
    assert abs(out.mean().item() - expected) < 1e-5
    print(f"  [PASS] CompositeReward weighted sum (={expected:.2f})")


def test_composite_reward_metrics_namespacing():
    """CompositeReward namespaces sub-metrics as 'reward/{name}_{key}'."""
    r1 = BehaviorReward({(0, 0, 0): 1.0})
    r2 = FormatReward(make_trie(), N_LAYERS)
    comp = CompositeReward([('behavior', 1.0, r1), ('format', 0.5, r2)])
    ctx, lengths = _dummy_ctx(2)
    sids = _sids([(0, 0, 0), (99, 99, 99)])
    comp(sids, ctx, lengths)
    m = comp.metrics()
    assert 'reward/behavior_mean' in m, f"Missing behavior_mean in {list(m.keys())}"
    assert 'reward/format_legal_rate' in m, f"Missing format_legal_rate in {list(m.keys())}"
    assert 'reward/total_mean' in m
    print(f"  [PASS] CompositeReward metrics namespacing: {list(m.keys())}")


def test_composite_reward_single_component():
    """CompositeReward with one component behaves like that component × weight."""
    fn = lambda s, c: [0.5] * len(s)
    comp = CompositeReward([('solo', 2.0, ExternalReward(fn))])
    ctx, lengths = _dummy_ctx(3)
    sids = torch.zeros(3, N_LAYERS, dtype=torch.long)
    out = comp(sids, ctx, lengths)
    assert abs(out.mean().item() - 1.0) < 1e-5  # 2.0 × 0.5 = 1.0
    print(f"  [PASS] CompositeReward single component")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Reward Function Tests")
    print("=" * 50)

    print("\n1. BehaviorReward")
    test_behavior_reward_full_match()
    test_behavior_reward_prefix_fallback()
    test_behavior_reward_default()
    test_behavior_reward_metrics()

    print("\n2. FormatReward")
    test_format_reward_legal_sids()
    test_format_reward_illegal_sids()
    test_format_reward_sample_k()
    test_format_reward_metrics()

    print("\n3. ExternalReward / BusinessReward")
    test_external_reward_callable()
    test_business_reward_same_as_external()

    print("\n4. _bitmap_to_quality")
    test_bitmap_quality_negative_feedback()
    test_bitmap_quality_zero()
    test_bitmap_quality_place_order()
    test_bitmap_quality_additive()

    print("\n5. WeightedBehaviorReward")
    test_weighted_behavior_reward_full_match()
    test_weighted_behavior_reward_hepo_prefix()
    test_weighted_behavior_reward_coverage()

    print("\n6. CompositeReward")
    test_composite_reward_weighted_sum()
    test_composite_reward_metrics_namespacing()
    test_composite_reward_single_component()

    print("\n" + "=" * 50)
    print("All tests passed!")
