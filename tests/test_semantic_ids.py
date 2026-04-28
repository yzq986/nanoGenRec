"""Tests for model/semantic_ids.py and related pure-logic utilities.

Also tests:
  - rl/feedback.py: classify_action (bitmap classification)
  - rl/preference.py: classify_rejected (prefix match difficulty)
  - ntp/model.py: SIDTrie construction and traversal
"""

import torch
import numpy as np

from model.semantic_ids import generate_semantic_ids
from rl.feedback import classify_action, STRONG_POSITIVE_MASK, WEAK_POSITIVE_MASK
from rl.preference import classify_rejected
from ntp.model import SIDTrie


# ── generate_semantic_ids ────────────────────────────────────────────────────

class _FakeKMeans:
    """Minimal kmeans-layer stub: nearest centroid by L2."""
    def __init__(self, centroids: torch.Tensor):
        self.centroids = centroids  # (K, D)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D) → (N,) assignments
        dists = torch.cdist(x.float(), self.centroids.float())
        return dists.argmin(dim=1)


class _FakeRKMeansModel:
    """Minimal multi-layer residual quantization stub."""
    def __init__(self, layer_centroids_list, device='cpu'):
        self.kmeans_layers = [_FakeKMeans(c) for c in layer_centroids_list]
        self.n_layers = len(layer_centroids_list)
        self.primary_device = device


def test_generate_semantic_ids_format():
    """Output SIDs have format 'a_b_c' with correct number of layers."""
    torch.manual_seed(0)
    n_layers, K, D = 3, 8, 16
    layer_centroids = [torch.randn(K, D) for _ in range(n_layers)]
    model = _FakeRKMeansModel(layer_centroids)

    embeddings = torch.randn(20, D)
    sids = generate_semantic_ids(model, embeddings, normalize_residuals=False)

    assert len(sids) == 20
    for sid in sids:
        parts = sid.split('_')
        assert len(parts) == n_layers, f"Expected {n_layers} layers, got {len(parts)} in '{sid}'"
        for p in parts:
            assert p.isdigit(), f"Non-digit part '{p}' in '{sid}'"
    print(f"  [PASS] generate_semantic_ids format (n={len(sids)}, layers={n_layers})")


def test_generate_semantic_ids_values_in_range():
    """Cluster assignments are in [0, K)."""
    torch.manual_seed(1)
    n_layers, K, D = 2, 4, 8
    layer_centroids = [torch.randn(K, D) for _ in range(n_layers)]
    model = _FakeRKMeansModel(layer_centroids)

    embeddings = torch.randn(50, D)
    sids = generate_semantic_ids(model, embeddings, normalize_residuals=False)

    for sid in sids:
        parts = [int(x) for x in sid.split('_')]
        for idx, p in enumerate(parts):
            assert 0 <= p < K, f"Layer {idx} assignment {p} out of [0, {K})"
    print(f"  [PASS] generate_semantic_ids values in [0, {K})")


def test_generate_semantic_ids_single_layer():
    """Single-layer model produces SIDs like '3' (no underscore)."""
    torch.manual_seed(2)
    K, D = 6, 8
    model = _FakeRKMeansModel([torch.randn(K, D)])

    embeddings = torch.randn(10, D)
    sids = generate_semantic_ids(model, embeddings, normalize_residuals=False)

    for sid in sids:
        assert '_' not in sid, f"Single-layer SID should have no underscore: '{sid}'"
        assert 0 <= int(sid) < K
    print(f"  [PASS] generate_semantic_ids single layer (no underscore)")


def test_generate_semantic_ids_deterministic():
    """Same embeddings + model → same SIDs (no randomness)."""
    torch.manual_seed(3)
    n_layers, K, D = 2, 4, 16
    layer_centroids = [torch.randn(K, D) for _ in range(n_layers)]
    model = _FakeRKMeansModel(layer_centroids)

    embeddings = torch.randn(30, D)
    sids_a = generate_semantic_ids(model, embeddings, normalize_residuals=False)
    sids_b = generate_semantic_ids(model, embeddings, normalize_residuals=False)

    assert sids_a == sids_b, "generate_semantic_ids should be deterministic"
    print(f"  [PASS] generate_semantic_ids deterministic")


def test_generate_semantic_ids_with_normalization():
    """normalize_residuals=True runs without error and produces valid SIDs."""
    torch.manual_seed(4)
    n_layers, K, D = 2, 4, 16
    layer_centroids = [torch.randn(K, D) for _ in range(n_layers)]
    model = _FakeRKMeansModel(layer_centroids)

    embeddings = torch.randn(20, D)
    sids = generate_semantic_ids(model, embeddings, normalize_residuals=True)

    assert len(sids) == 20
    for sid in sids:
        parts = sid.split('_')
        assert len(parts) == n_layers
    print(f"  [PASS] generate_semantic_ids with L2 normalization")


# ── classify_action ──────────────────────────────────────────────────────────

def test_classify_action_negative():
    """Negative action_bitmap (sign bit set) → 'negative'."""
    assert classify_action(-1) == 'negative'
    assert classify_action(-2) == 'negative'
    print(f"  [PASS] classify_action negative (sign bit)")


def test_classify_action_strong():
    """Any strong-signal bit set → 'strong'."""
    # like bit (2)
    assert classify_action(2) == 'strong'
    # place_order bit (262144)
    assert classify_action(262144) == 'strong'
    # both weak and strong → strong wins
    assert classify_action(1 | 2) == 'strong'
    print(f"  [PASS] classify_action strong positive")


def test_classify_action_weak():
    """Weak bits only (no strong, no negative) → 'weak'."""
    # click (1)
    assert classify_action(1) == 'weak'
    # coin_click (16)
    assert classify_action(16) == 'weak'
    print(f"  [PASS] classify_action weak positive")


def test_classify_action_neutral():
    """Zero or view_exit only → 'neutral'."""
    VIEW_EXIT_BIT = 4096
    assert classify_action(0) == 'neutral'
    assert classify_action(VIEW_EXIT_BIT) == 'neutral'
    print(f"  [PASS] classify_action neutral (zero / view_exit only)")


def test_classify_action_numpy_array():
    """Vectorized numpy array input returns correct labels."""
    arr = np.array([0, 1, 2, -1, 262144, 4096], dtype=np.int32)
    result = classify_action(arr)
    assert result[0] == 'neutral'
    assert result[1] == 'weak'
    assert result[2] == 'strong'
    assert result[3] == 'negative'
    assert result[4] == 'strong'
    assert result[5] == 'neutral'
    print(f"  [PASS] classify_action numpy array vectorized")


# ── classify_rejected ────────────────────────────────────────────────────────

def test_classify_rejected_easy():
    """L0 mismatch → easy."""
    gt = [0, 0, 0]
    beam = torch.tensor([[1, 0, 0], [2, 5, 3]])
    result = classify_rejected(gt, beam, n_layers=3)
    assert len(result['easy']) == 2
    assert len(result['medium']) == 0
    assert len(result['hard']) == 0
    print(f"  [PASS] classify_rejected easy (L0 mismatch)")


def test_classify_rejected_medium():
    """L0 match, L1 mismatch → medium."""
    gt = [0, 0, 0]
    beam = torch.tensor([[0, 1, 0], [0, 2, 5]])
    result = classify_rejected(gt, beam, n_layers=3)
    assert len(result['easy']) == 0
    assert len(result['medium']) == 2
    assert len(result['hard']) == 0
    print(f"  [PASS] classify_rejected medium (L0=GT, L1 mismatch)")


def test_classify_rejected_hard():
    """L0+L1 match, L2 mismatch → hard."""
    gt = [0, 0, 0]
    beam = torch.tensor([[0, 0, 1], [0, 0, 5]])
    result = classify_rejected(gt, beam, n_layers=3)
    assert len(result['easy']) == 0
    assert len(result['medium']) == 0
    assert len(result['hard']) == 2
    print(f"  [PASS] classify_rejected hard (L0+L1=GT, L2 mismatch)")


def test_classify_rejected_skips_ground_truth():
    """Ground truth itself is excluded from all buckets."""
    gt = [1, 2, 3]
    beam = torch.tensor([[1, 2, 3], [0, 0, 0], [1, 2, 3]])  # 2 GT copies, 1 easy
    result = classify_rejected(gt, beam, n_layers=3)
    assert len(result['easy']) == 1
    assert len(result['medium']) == 0
    assert len(result['hard']) == 0
    print(f"  [PASS] classify_rejected skips ground truth")


def test_classify_rejected_mixed():
    """Mixed beam → correct bucket assignment."""
    gt = [3, 7, 1]
    beam = torch.tensor([
        [0, 0, 0],  # easy (L0≠gt)
        [3, 0, 0],  # medium (L0=gt, L1≠gt)
        [3, 7, 0],  # hard (L0+L1=gt, L2≠gt)
        [3, 7, 1],  # gt itself → skip
    ])
    result = classify_rejected(gt, beam, n_layers=3)
    assert len(result['easy']) == 1
    assert len(result['medium']) == 1
    assert len(result['hard']) == 1
    print(f"  [PASS] classify_rejected mixed buckets")


# ── SIDTrie ──────────────────────────────────────────────────────────────────

def _build_trie(sids, n_layers=3):
    """Build a SIDTrie from a list of SID strings like '0_1_2'."""
    sid_to_items = {sid: set() for sid in sids}
    return SIDTrie(sid_to_items=sid_to_items, n_layers=n_layers)


def test_sidtrie_valid_tokens_l0():
    """valid_tokens at layer 0 returns all L0 tokens."""
    sids = ['0_0_0', '1_0_0', '2_1_0']
    trie = _build_trie(sids)
    l0_tokens = set(trie.valid_tokens(0, ()))
    assert l0_tokens == {0, 1, 2}
    print(f"  [PASS] SIDTrie valid_tokens L0 = {l0_tokens}")


def test_sidtrie_valid_tokens_l1():
    """valid_tokens at layer 1 returns only tokens consistent with L0 prefix."""
    sids = ['0_1_0', '0_2_0', '1_3_0']
    trie = _build_trie(sids)
    l1_from_0 = set(trie.valid_tokens(1, (0,)))
    assert l1_from_0 == {1, 2}
    l1_from_1 = set(trie.valid_tokens(1, (1,)))
    assert l1_from_1 == {3}
    print(f"  [PASS] SIDTrie valid_tokens L1 prefix filtering")


def test_sidtrie_valid_tokens_l2():
    """valid_tokens at layer 2 returns only the leaf token for a given prefix."""
    sids = ['5_6_7', '5_6_8', '5_9_0']
    trie = _build_trie(sids)
    l2_from_5_6 = set(trie.valid_tokens(2, (5, 6)))
    assert l2_from_5_6 == {7, 8}
    l2_from_5_9 = set(trie.valid_tokens(2, (5, 9)))
    assert l2_from_5_9 == {0}
    print(f"  [PASS] SIDTrie valid_tokens L2 leaf lookup")


def test_sidtrie_empty_prefix_unknown():
    """Query with an unknown prefix returns empty set."""
    sids = ['0_0_0']
    trie = _build_trie(sids)
    unknown = set(trie.valid_tokens(1, (99,)))
    assert unknown == set()
    print(f"  [PASS] SIDTrie unknown prefix → empty set")


def test_sidtrie_duplicate_sids_deduplicated():
    """SIDs with same path share the same node; no extra tokens created."""
    # Same SID via two 'items' → still only one valid path
    sid_to_items = {'1_2_3': {'item_a', 'item_b'}, '1_2_4': {'item_c'}}
    trie = SIDTrie(sid_to_items=sid_to_items, n_layers=3)
    l2_tokens = set(trie.valid_tokens(2, (1, 2)))
    assert l2_tokens == {3, 4}
    print(f"  [PASS] SIDTrie duplicate SID paths deduplicated")


def test_sidtrie_root_tokens():
    """root_tokens() returns the same as valid_tokens(0, ())."""
    sids = ['10_0_0', '20_0_0', '30_0_0']
    trie = _build_trie(sids)
    assert set(trie.root_tokens()) == set(trie.valid_tokens(0, ()))
    assert set(trie.root_tokens()) == {10, 20, 30}
    print(f"  [PASS] SIDTrie root_tokens() == valid_tokens(0, ())")


def test_sidtrie_ignores_wrong_length():
    """SIDs with wrong number of layers are silently skipped."""
    sid_to_items = {'0_0': set(), '0_0_0': set()}  # only 3-layer is valid for n_layers=3
    trie = SIDTrie(sid_to_items=sid_to_items, n_layers=3)
    l0 = set(trie.valid_tokens(0, ()))
    assert 0 in l0  # from '0_0_0'
    print(f"  [PASS] SIDTrie ignores SIDs with wrong layer count")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Semantic IDs / Feedback / Preference / SIDTrie Tests")
    print("=" * 60)

    print("\n1. generate_semantic_ids")
    test_generate_semantic_ids_format()
    test_generate_semantic_ids_values_in_range()
    test_generate_semantic_ids_single_layer()
    test_generate_semantic_ids_deterministic()
    test_generate_semantic_ids_with_normalization()

    print("\n2. classify_action")
    test_classify_action_negative()
    test_classify_action_strong()
    test_classify_action_weak()
    test_classify_action_neutral()
    test_classify_action_numpy_array()

    print("\n3. classify_rejected")
    test_classify_rejected_easy()
    test_classify_rejected_medium()
    test_classify_rejected_hard()
    test_classify_rejected_skips_ground_truth()
    test_classify_rejected_mixed()

    print("\n4. SIDTrie")
    test_sidtrie_valid_tokens_l0()
    test_sidtrie_valid_tokens_l1()
    test_sidtrie_valid_tokens_l2()
    test_sidtrie_empty_prefix_unknown()
    test_sidtrie_duplicate_sids_deduplicated()
    test_sidtrie_root_tokens()
    test_sidtrie_ignores_wrong_length()

    print("\n" + "=" * 60)
    print("All tests passed!")
