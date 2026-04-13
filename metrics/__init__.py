"""
Metrics Evaluation Framework for Embedding & Semantic ID Quality

This module provides a modular, engineering-grade metrics framework for:
1. Evaluating embedding model quality
2. Comparing different RKMeans configurations
3. Supporting offline metrics comparison

Reference: OneRec (arxiv 2506.13695)
"""

from .base import BaseMetric, MetricResult, print_metric_result
from .reconstruction import ReconstructionLossMetric
from .codebook import CodebookUtilizationMetric
from .entropy import TokenEntropyMetric
from .similarity import CosineSimilarityMetric
from .effective_dim import EffectiveDimensionMetric
from .collision import SemanticIDCollisionMetric
from .cluster_balance import ClusterBalanceMetric
from .behavior import (
    UserSemanticConsistencyMetric,
    SemanticNeighborHitRateMetric,
    EmbeddingBehaviorCorrelationMetric,
    PositiveNegativeSeparationMetric,
)
from .sid_prediction import SemanticIDPredictionMetric
from .report import ReportGenerator

# Intrinsic metrics (no behavior data needed)
INTRINSIC_METRICS = {
    'reconstruction_loss': ReconstructionLossMetric,
    'codebook_utilization': CodebookUtilizationMetric,
    'entropy': TokenEntropyMetric,
    'cosine_similarity': CosineSimilarityMetric,
    'effective_dimension': EffectiveDimensionMetric,
    'semantic_id_collision': SemanticIDCollisionMetric,
    'cluster_balance': ClusterBalanceMetric,
}

# Behavior-based metrics (require user behavior data)
BEHAVIOR_METRICS = {
    'user_semantic_consistency': UserSemanticConsistencyMetric,
    'semantic_neighbor_hit_rate': SemanticNeighborHitRateMetric,
    'embedding_behavior_correlation': EmbeddingBehaviorCorrelationMetric,
    'positive_negative_separation': PositiveNegativeSeparationMetric,
    'semantic_id_prediction': SemanticIDPredictionMetric,
}

# All available metrics
AVAILABLE_METRICS = {**INTRINSIC_METRICS, **BEHAVIOR_METRICS}

__all__ = [
    'BaseMetric',
    'MetricResult',
    # Intrinsic metrics
    'ReconstructionLossMetric',
    'CodebookUtilizationMetric',
    'TokenEntropyMetric',
    'CosineSimilarityMetric',
    'EffectiveDimensionMetric',
    'SemanticIDCollisionMetric',
    'ClusterBalanceMetric',
    # Behavior-based metrics
    'UserSemanticConsistencyMetric',
    'SemanticNeighborHitRateMetric',
    'EmbeddingBehaviorCorrelationMetric',
    'PositiveNegativeSeparationMetric',
    'SemanticIDPredictionMetric',
    # Utilities
    'ReportGenerator',
    'print_metric_result',
    'AVAILABLE_METRICS',
    'INTRINSIC_METRICS',
    'BEHAVIOR_METRICS',
]
