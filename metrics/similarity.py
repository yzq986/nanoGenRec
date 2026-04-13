"""
Cosine Similarity Distribution Metric

Measures the pairwise cosine similarity distribution of embeddings.
Good embeddings should have moderate mean similarity with high variance.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F

from .base import BaseMetric, MetricResult


class CosineSimilarityMetric(BaseMetric):
    """Cosine Similarity Distribution

    Computes statistics of pairwise cosine similarities:
    - mean: Should be moderate (0.1-0.4) - too high = poor discrimination
    - std: Should be high (0.2+) - indicates good spread
    - min/max: Range of similarities

    Uses sampling for efficiency with large datasets.
    """

    name = 'cosine_similarity'
    requires_model = False
    requires_semantic_ids = False

    # Ideal: mean around 0.2-0.3, std > 0.2
    thresholds = {
        'excellent_mean_low': 0.1,
        'excellent_mean_high': 0.3,
        'good_mean_high': 0.4,
        'excellent_std': 0.25,
        'good_std': 0.2,
    }

    def assess_quality(self, value: float) -> str:
        """Custom assessment based on mean and std"""
        # Value here is the std (we want high std)
        if value >= self.thresholds['excellent_std']:
            return 'excellent'
        elif value >= self.thresholds['good_std']:
            return 'good'
        else:
            return 'acceptable'

    def compute(
        self,
        embeddings: torch.Tensor,
        model: Optional[Any] = None,
        semantic_ids: Optional[List[str]] = None,
        layer_assignments: Optional[List[torch.Tensor]] = None,
        sample_size: int = 5000,
        **kwargs
    ) -> MetricResult:
        """Compute cosine similarity distribution

        Args:
            embeddings: (N, D) tensor of embeddings
            sample_size: Number of samples for pairwise computation

        Returns:
            MetricResult with similarity statistics
        """
        self.validate_inputs(embeddings, model, semantic_ids)

        n = len(embeddings)
        if n > sample_size:
            idx = torch.randperm(n)[:sample_size]
            sample = embeddings[idx]
        else:
            sample = embeddings

        # Normalize for cosine similarity
        sample_norm = F.normalize(sample, dim=1)

        # Compute pairwise similarities
        sim_matrix = sample_norm @ sample_norm.t()

        # Extract upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
        sim_values = sim_matrix[mask]

        # Compute statistics
        sim_mean = sim_values.mean().item()
        sim_std = sim_values.std().item()
        sim_min = sim_values.min().item()
        sim_max = sim_values.max().item()
        sim_median = sim_values.median().item()

        # Percentiles
        sorted_sims = torch.sort(sim_values).values
        n_pairs = len(sim_values)
        p5 = sorted_sims[int(n_pairs * 0.05)].item()
        p25 = sorted_sims[int(n_pairs * 0.25)].item()
        p75 = sorted_sims[int(n_pairs * 0.75)].item()
        p95 = sorted_sims[int(n_pairs * 0.95)].item()

        # Assess quality based on std (higher is better)
        status = self.assess_quality(sim_std)

        # Additional quality check on mean
        mean_ok = self.thresholds['excellent_mean_low'] <= sim_mean <= self.thresholds['good_mean_high']
        if not mean_ok:
            if status == 'excellent':
                status = 'good'
            elif status == 'good':
                status = 'acceptable'

        return MetricResult(
            name=self.name,
            value=sim_std,  # Primary value is std (discrimination ability)
            layer_values=[],  # Not layer-specific
            details={
                'mean': sim_mean,
                'std': sim_std,
                'min': sim_min,
                'max': sim_max,
                'median': sim_median,
                'p5': p5,
                'p25': p25,
                'p75': p75,
                'p95': p95,
                'sample_size': len(sample),
                'n_pairs': n_pairs,
            },
            status=status,
        )
