"""
SID Distribution Entropy Metric

Measures how uniformly semantic IDs are distributed, reported per prefix depth.

  depth=1: entropy over "a_*_*" prefix distribution
  depth=2: entropy over "a_b_*" prefix distribution
  depth=3: entropy over "a_b_c" full SID distribution

Reference: OneRec (arxiv 2506.13695)
"""

from typing import Any, Dict, List, Optional
from collections import Counter
import math
import torch

from .base import BaseMetric, MetricResult


class TokenEntropyMetric(BaseMetric):
    """SID Distribution Entropy: H = -sum(p * log2(p))

    Measures the uniformity of SID distribution. Higher entropy means more
    uniform distribution = better utilization of the SID space.

    Reports normalized entropy at each prefix depth:
      depth 1: H(L1 prefix) / log2(N)
      depth 2: H(L1_L2 prefix) / log2(N^2)
      depth 3: H(full SID) / log2(N_total)

    Primary value = full-SID normalized entropy.
    """

    name = 'entropy'
    requires_model = False
    requires_semantic_ids = True

    # Higher normalized entropy is better
    thresholds = {
        'excellent': 0.95,
        'good': 0.90,
        'acceptable': 0.80,
    }

    def assess_quality(self, value: float) -> str:
        """Higher entropy is better"""
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
        chunk_size: int = 50000,
        **kwargs
    ) -> MetricResult:
        n_total = len(semantic_ids)

        # Full SID entropy (primary)
        sid_counts = Counter(semantic_ids)
        n_unique = len(sid_counts)

        max_entropy = math.log2(n_total) if n_total > 1 else 1.0

        entropy = 0.0
        for count in sid_counts.values():
            p = count / n_total
            if p > 0:
                entropy -= p * math.log2(p)

        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        # Per-depth prefix entropy
        depth_normalized = []
        depth_stats = []
        n_layers = 0
        n_clusters_per_layer = 0

        if semantic_ids and '_' in semantic_ids[0]:
            parts = [sid.split('_') for sid in semantic_ids]
            n_layers = len(parts[0])

            # Determine N
            for layer_idx in range(n_layers):
                layer_ids = [int(p[layer_idx]) for p in parts]
                max_id = max(layer_ids) + 1
                n_clusters_per_layer = max(n_clusters_per_layer, max_id)

            for depth in range(1, n_layers + 1):
                prefixes = ['_'.join(p[:depth]) for p in parts]
                prefix_counts = Counter(prefixes)
                n_unique_prefix = len(prefix_counts)

                # Max entropy for this depth = log2(N^depth)
                depth_max_entropy = depth * math.log2(n_clusters_per_layer) if n_clusters_per_layer > 1 else 1.0

                depth_h = 0.0
                for count in prefix_counts.values():
                    p = count / n_total
                    if p > 0:
                        depth_h -= p * math.log2(p)

                depth_norm = depth_h / depth_max_entropy if depth_max_entropy > 0 else 0.0
                depth_normalized.append(depth_norm)
                depth_stats.append({
                    'depth': depth,
                    'entropy': round(depth_h, 4),
                    'max_entropy': round(depth_max_entropy, 4),
                    'normalized': round(depth_norm, 4),
                    'n_unique_prefix': n_unique_prefix,
                })

        status = self.assess_quality(normalized)

        return MetricResult(
            name=self.name,
            value=normalized,
            layer_values=depth_normalized,
            details={
                'entropy': entropy,
                'max_entropy': max_entropy,
                'normalized_entropy': normalized,
                'n_total': n_total,
                'n_unique_sids': n_unique,
                'n_layers': n_layers,
                'n_clusters_per_layer': n_clusters_per_layer,
                'depth_stats': depth_stats,
            },
            status=status,
        )
