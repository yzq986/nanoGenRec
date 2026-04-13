"""
Effective Dimension Metric

Measures how many dimensions of the embedding space are actually utilized
through PCA explained variance analysis.
"""

from typing import Any, Dict, List, Optional
import torch

from .base import BaseMetric, MetricResult


class EffectiveDimensionMetric(BaseMetric):
    """Effective Dimension via PCA

    Estimates how many dimensions contain meaningful information by
    computing the number of principal components needed to explain
    a given percentage of variance.

    Higher effective dimension (closer to actual dimension) indicates
    better information utilization.
    """

    name = 'effective_dimension'
    requires_model = False
    requires_semantic_ids = False

    # Higher utilization ratio is better
    thresholds = {
        'excellent': 0.7,  # 70%+ of dimensions used
        'good': 0.5,
        'acceptable': 0.3,
    }

    def assess_quality(self, value: float) -> str:
        """Higher dimension utilization is better"""
        if value >= self.thresholds['excellent']:
            return 'excellent'
        elif value >= self.thresholds['good']:
            return 'good'
        elif value >= self.thresholds['acceptable']:
            return 'acceptable'
        else:
            return 'poor'

    def compute(
        self,
        embeddings: torch.Tensor,
        model: Optional[Any] = None,
        semantic_ids: Optional[List[str]] = None,
        layer_assignments: Optional[List[torch.Tensor]] = None,
        sample_size: int = 10000,
        variance_thresholds: List[float] = [0.90, 0.95, 0.99],
        **kwargs
    ) -> MetricResult:
        """Compute effective dimension

        Args:
            embeddings: (N, D) tensor of embeddings
            sample_size: Number of samples for PCA
            variance_thresholds: List of variance percentages to report

        Returns:
            MetricResult with effective dimensions for different thresholds
        """
        self.validate_inputs(embeddings, model, semantic_ids)

        n = len(embeddings)
        total_dim = embeddings.shape[1]

        if n > sample_size:
            idx = torch.randperm(n)[:sample_size]
            sample = embeddings[idx]
        else:
            sample = embeddings

        # Center the data
        centered = sample - sample.mean(dim=0)

        # Compute SVD
        try:
            _, S, _ = torch.svd(centered)
            variance_explained = (S ** 2).cumsum(0) / (S ** 2).sum()

            # Compute effective dimensions for each threshold
            effective_dims = {}
            for threshold in variance_thresholds:
                key = f'dim_{int(threshold * 100)}'
                eff_dim = (variance_explained < threshold).sum().item() + 1
                effective_dims[key] = eff_dim

            # Primary metric: 95% variance dimension ratio
            eff_dim_95 = effective_dims.get('dim_95', effective_dims.get('dim_90', total_dim))
            utilization_ratio = eff_dim_95 / total_dim

            # Intrinsic dimensionality estimate (participation ratio)
            eigenvalues = (S ** 2) / (S ** 2).sum()
            participation_ratio = 1.0 / (eigenvalues ** 2).sum().item()

            # Spectral decay (how fast eigenvalues decay)
            top_10_ratio = (S[:10] ** 2).sum().item() / (S ** 2).sum().item() if len(S) >= 10 else 1.0

        except Exception as e:
            print(f"SVD failed: {e}, using fallback")
            effective_dims = {f'dim_{int(t*100)}': -1 for t in variance_thresholds}
            utilization_ratio = -1
            participation_ratio = -1
            top_10_ratio = -1

        status = self.assess_quality(utilization_ratio) if utilization_ratio > 0 else 'unknown'

        return MetricResult(
            name=self.name,
            value=utilization_ratio,
            layer_values=[],  # Not layer-specific
            details={
                'total_dimension': total_dim,
                'effective_dimensions': effective_dims,
                'utilization_ratio': utilization_ratio,
                'participation_ratio': participation_ratio,
                'top_10_variance_ratio': top_10_ratio,
                'sample_size': len(sample),
            },
            status=status,
        )
