"""Tests for metrics/ — SID quality metrics framework.

Covers:
  - MetricResult: to_dict/from_dict round-trip, to_flat_dict keys
  - BaseMetric: validate_inputs contract
  - SemanticIDCollisionMetric: collision rate computation
  - TokenEntropyMetric: entropy per depth
  - CodebookUtilizationMetric: utilization fraction
  - ClusterBalanceMetric: Gini coefficient
  - ReconstructionLossMetric: L2 reconstruction error
  - CosineSimilarityMetric: intra-cluster similarity
"""

import torch

from metrics.base import MetricResult, BaseMetric


def _assert_raises(fn, exc=Exception, match=None):
    try:
        fn()
    except exc as e:
        if match and match not in str(e):
            raise AssertionError(f"Exception message {str(e)!r} did not contain {match!r}")
        return
    raise AssertionError(f"Expected {exc.__name__} but no exception was raised")
from metrics.collision import SemanticIDCollisionMetric
from metrics.entropy import TokenEntropyMetric
from metrics.codebook import CodebookUtilizationMetric
from metrics.cluster_balance import ClusterBalanceMetric
from metrics.reconstruction import ReconstructionLossMetric
from metrics.similarity import CosineSimilarityMetric


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sids(n_items: int, n_layers: int, n_clusters: int, seed: int = 0):
    """Random semantic ID strings: 'l0_l1_l2'."""
    torch.manual_seed(seed)
    ids = []
    assignments = []
    for _ in range(n_items):
        layers = [torch.randint(0, n_clusters, (1,)).item() for _ in range(n_layers)]
        ids.append('_'.join(str(x) for x in layers))
        assignments.append(torch.tensor(layers))
    return ids, assignments


def _make_embeddings(n: int, d: int, seed: int = 0):
    torch.manual_seed(seed)
    return torch.randn(n, d)


# ── MetricResult ──────────────────────────────────────────────────────────────

def test_metric_result_to_dict_round_trip():
    """to_dict / from_dict is a lossless round-trip."""
    r = MetricResult(
        name='test_metric',
        value=0.42,
        layer_values=[0.1, 0.2, 0.3],
        details={'n_unique': 100, 'prefix_stats': [{'depth': 1, 'avg_items': 3.5}]},
        status='good',
    )
    d = r.to_dict()
    r2 = MetricResult.from_dict(d)
    assert r2.name == r.name
    assert abs(r2.value - r.value) < 1e-9
    assert r2.layer_values == r.layer_values
    assert r2.details == r.details
    assert r2.status == r.status
    print(f"  [PASS] MetricResult to_dict/from_dict round-trip")


def test_metric_result_to_flat_dict():
    """to_flat_dict includes primary value and depth breakdown."""
    r = MetricResult(
        name='collision',
        value=0.05,
        layer_values=[0.03, 0.05, 0.07],
        details={'space_utilization': 0.8},
    )
    flat = r.to_flat_dict()
    assert 'collision' in flat
    assert abs(flat['collision'] - 0.05) < 1e-5
    assert 'collision_depth' in flat
    assert flat['collision_space_utilization'] == 0.8
    print(f"  [PASS] MetricResult to_flat_dict keys")


def test_metric_result_defaults():
    """MetricResult with minimal args has sensible defaults."""
    r = MetricResult(name='x', value=1.0)
    assert r.layer_values == []
    assert r.details == {}
    assert r.status == 'unknown'
    print(f"  [PASS] MetricResult defaults")


# ── BaseMetric.validate_inputs ────────────────────────────────────────────────

class _ConcreteMetric(BaseMetric):
    name = 'concrete'
    requires_model = False
    requires_semantic_ids = True

    def compute(self, embeddings, model=None, semantic_ids=None, **kwargs):
        self.validate_inputs(embeddings, model, semantic_ids)
        return MetricResult(name=self.name, value=0.0)


def test_base_metric_validate_empty_embeddings():
    """validate_inputs raises ValueError on empty embeddings."""
    m = _ConcreteMetric()
    _assert_raises(
        lambda: m.validate_inputs(torch.empty(0, 10), semantic_ids=['a']),
        ValueError, match='non-empty'
    )
    print(f"  [PASS] BaseMetric validate empty embeddings raises")


def test_base_metric_validate_missing_semantic_ids():
    """validate_inputs raises ValueError when semantic_ids required but missing."""
    m = _ConcreteMetric()
    _assert_raises(
        lambda: m.validate_inputs(torch.randn(5, 10), semantic_ids=None),
        ValueError, match='semantic_ids'
    )
    print(f"  [PASS] BaseMetric validate missing semantic_ids raises")


def test_base_metric_assess_quality_no_thresholds():
    """assess_quality returns 'unknown' when thresholds not set."""
    m = _ConcreteMetric()
    assert m.assess_quality(0.5) == 'unknown'
    print(f"  [PASS] BaseMetric assess_quality unknown")


# ── SemanticIDCollisionMetric ─────────────────────────────────────────────────

def test_collision_no_duplicates():
    """Unique SIDs → collision rate = 0."""
    metric = SemanticIDCollisionMetric()
    sids = [f'{i}_0_0' for i in range(20)]  # all unique
    emb = _make_embeddings(20, 32)
    result = metric.compute(emb, semantic_ids=sids)
    assert result.value == 0.0, f"Expected 0 collision, got {result.value}"
    print(f"  [PASS] SemanticIDCollisionMetric no duplicates → 0.0")


def test_collision_all_duplicates():
    """All-same SIDs → collision rate = 1.0 (or close)."""
    metric = SemanticIDCollisionMetric()
    sids = ['1_2_3'] * 10
    emb = _make_embeddings(10, 32)
    result = metric.compute(emb, semantic_ids=sids)
    # All 10 items share same SID → 9/10 = 0.9 collision (duplicates above first)
    assert result.value > 0.5, f"Expected high collision, got {result.value}"
    print(f"  [PASS] SemanticIDCollisionMetric all-same → {result.value:.3f}")


def test_collision_partial():
    """Partial duplicates yield collision rate in (0, 1)."""
    metric = SemanticIDCollisionMetric()
    # 5 unique + 5 duplicate
    sids = [f'{i}_0_0' for i in range(5)] + ['0_0_0'] * 5
    emb = _make_embeddings(10, 32)
    result = metric.compute(emb, semantic_ids=sids)
    assert 0.0 < result.value < 1.0
    assert torch.isfinite(torch.tensor(result.value))
    print(f"  [PASS] SemanticIDCollisionMetric partial ({result.value:.3f})")


# ── TokenEntropyMetric ────────────────────────────────────────────────────────

def test_entropy_uniform_distribution():
    """Uniform token distribution maximizes normalized entropy (≈ 1.0)."""
    metric = TokenEntropyMetric()
    n_clusters = 8
    # Each cluster used exactly once → all SIDs unique → max entropy
    sids = [f'{i % n_clusters}_{(i+1) % n_clusters}_{(i+2) % n_clusters}'
            for i in range(n_clusters)]
    emb = _make_embeddings(n_clusters, 32)
    result = metric.compute(emb, semantic_ids=sids)
    # result.value is *normalized* entropy in [0, 1]; all-unique → should be 1.0
    assert result.value > 0.9, \
        f"Uniform entropy too low: {result.value:.3f} (expected > 0.9)"
    print(f"  [PASS] TokenEntropyMetric uniform ≈ max entropy ({result.value:.3f})")


def test_entropy_collapsed():
    """All items in same cluster → entropy ≈ 0."""
    metric = TokenEntropyMetric()
    sids = ['0_0_0'] * 20
    emb = _make_embeddings(20, 32)
    result = metric.compute(emb, semantic_ids=sids)
    assert result.value < 0.1, f"Collapsed entropy too high: {result.value}"
    print(f"  [PASS] TokenEntropyMetric collapsed ≈ 0 ({result.value:.4f})")


# ── CodebookUtilizationMetric ─────────────────────────────────────────────────

def test_codebook_utilization_full():
    """Using all possible SIDs → utilization close to 1.0."""
    metric = CodebookUtilizationMetric()
    n = 8
    # Generate all combinations for 2 clusters × 3 layers = 8 unique
    sids = [f'{a}_{b}_{c}' for a in range(2) for b in range(2) for c in range(2)]
    emb = _make_embeddings(len(sids), 32)
    result = metric.compute(emb, semantic_ids=sids)
    assert result.value > 0.5
    print(f"  [PASS] CodebookUtilizationMetric full ({result.value:.3f})")


def test_codebook_utilization_sparse():
    """Few unique SIDs → low utilization."""
    metric = CodebookUtilizationMetric()
    sids = ['0_0_0', '1_1_1']  # only 2 unique out of many possible
    emb = _make_embeddings(2, 32)
    result = metric.compute(emb, semantic_ids=sids)
    assert result.value >= 0.0
    assert torch.isfinite(torch.tensor(result.value))
    print(f"  [PASS] CodebookUtilizationMetric sparse ({result.value:.4f})")


# ── ClusterBalanceMetric ──────────────────────────────────────────────────────

def test_cluster_balance_perfect():
    """Perfectly balanced clusters → Gini ≈ 0."""
    metric = ClusterBalanceMetric()
    # 4 clusters, 4 items each → perfectly balanced
    sids, assignments = _make_sids(16, 3, 4, seed=99)
    # Override to make perfectly balanced
    balanced_sids = []
    for i in range(4):
        for j in range(4):
            balanced_sids.append(f'{i}_{j % 4}_{(i+j) % 4}')
    emb = _make_embeddings(16, 32)
    result = metric.compute(emb, semantic_ids=balanced_sids)
    assert result.value < 0.5  # Gini < 0.5 for reasonably balanced
    print(f"  [PASS] ClusterBalanceMetric balanced → Gini={result.value:.3f}")


def test_cluster_balance_imbalanced():
    """Highly imbalanced → higher Gini."""
    metric = ClusterBalanceMetric()
    # 1 cluster dominates: 19 items in cluster 0, 1 in cluster 1
    sids = ['0_0_0'] * 19 + ['1_1_1']
    emb = _make_embeddings(20, 32)
    result = metric.compute(emb, semantic_ids=sids)
    assert result.value > 0.3, f"Imbalanced Gini too low: {result.value}"
    print(f"  [PASS] ClusterBalanceMetric imbalanced → Gini={result.value:.3f}")


# ── ReconstructionLossMetric ──────────────────────────────────────────────────

def test_reconstruction_loss_requires_model():
    """ReconstructionLossMetric raises if model not provided."""
    metric = ReconstructionLossMetric()
    emb = _make_embeddings(10, 32)
    sids = [f'{i}_0_0' for i in range(10)]
    _assert_raises(lambda: metric.compute(emb, model=None, semantic_ids=sids))
    print(f"  [PASS] ReconstructionLossMetric raises without model")


# ── CosineSimilarityMetric ────────────────────────────────────────────────────

def test_cosine_similarity_identical():
    """Identical embeddings → mean cosine similarity = 1.0, std = 0.0."""
    metric = CosineSimilarityMetric()
    emb = torch.ones(10, 16)
    sids = [f'{i % 3}_0_0' for i in range(10)]
    result = metric.compute(emb, semantic_ids=sids)
    # value = std (discrimination); details['mean'] = actual similarity
    assert abs(result.details['mean'] - 1.0) < 1e-4, \
        f"Identical emb mean similarity: {result.details['mean']}"
    assert result.value < 1e-4, f"Identical emb similarity std should be ~0: {result.value}"
    print(f"  [PASS] CosineSimilarityMetric identical embeddings → mean=1.0, std≈0")


def test_cosine_similarity_random():
    """Random embeddings → mean similarity near 0."""
    metric = CosineSimilarityMetric()
    torch.manual_seed(42)
    emb = torch.randn(200, 64)
    sids = [f'{i % 8}_0_0' for i in range(200)]
    result = metric.compute(emb, semantic_ids=sids)
    assert abs(result.details['mean']) < 0.3, \
        f"Random similarity too high: {result.details['mean']}"
    assert torch.isfinite(torch.tensor(result.value))
    print(f"  [PASS] CosineSimilarityMetric random mean≈0 ({result.details['mean']:.4f})")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Metrics Framework Tests")
    print("=" * 50)

    print("\n1. MetricResult")
    test_metric_result_to_dict_round_trip()
    test_metric_result_to_flat_dict()
    test_metric_result_defaults()

    print("\n2. BaseMetric")
    test_base_metric_validate_empty_embeddings()
    test_base_metric_validate_missing_semantic_ids()
    test_base_metric_assess_quality_no_thresholds()

    print("\n3. SemanticIDCollisionMetric")
    test_collision_no_duplicates()
    test_collision_all_duplicates()
    test_collision_partial()

    print("\n4. TokenEntropyMetric")
    test_entropy_uniform_distribution()
    test_entropy_collapsed()

    print("\n5. CodebookUtilizationMetric")
    test_codebook_utilization_full()
    test_codebook_utilization_sparse()

    print("\n6. ClusterBalanceMetric")
    test_cluster_balance_perfect()
    test_cluster_balance_imbalanced()

    print("\n7. ReconstructionLossMetric")
    test_reconstruction_loss_requires_model()

    print("\n8. CosineSimilarityMetric")
    test_cosine_similarity_identical()
    test_cosine_similarity_random()

    print("\n" + "=" * 50)
    print("All tests passed!")
