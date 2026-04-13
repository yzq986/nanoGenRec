"""
Semantic ID Collision Metric

Measures the rate of duplicate semantic IDs (different items with same ID).
Reports per-prefix-depth: collision rate, bucket size distribution (min/max/p50/p90/p99).
"""

from typing import Any, Dict, List, Optional
from collections import Counter
import torch

from .base import BaseMetric, MetricResult


class SemanticIDCollisionMetric(BaseMetric):
    """Semantic ID Collision Rate per prefix depth

    layer_values[i] = collision rate at depth i+1 (1 - n_unique_prefix / n_total).
    details['prefix_stats'][i] = bucket size distribution at depth i+1.

    For recall with "a_b_*" (depth=2), check prefix_stats[1]:
      avg/p50/p90/p99 → candidate pool size per prefix
    """

    name = 'semantic_id_collision'
    requires_model = False
    requires_semantic_ids = True

    # Lower is better
    thresholds = {
        'excellent': 0.01,
        'good': 0.05,
        'acceptable': 0.15,
    }

    def compute(
        self,
        embeddings: torch.Tensor,
        model: Optional[Any] = None,
        semantic_ids: Optional[List[str]] = None,
        layer_assignments: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> MetricResult:
        self.validate_inputs(embeddings, model, semantic_ids)

        n_total = len(semantic_ids)
        n_unique = len(set(semantic_ids))
        collision_rate = 1.0 - (n_unique / n_total)

        # Full collision stats
        id_counts = Counter(semantic_ids)
        count_values = list(id_counts.values())
        n_collided_ids = sum(1 for c in count_values if c > 1)
        n_items_in_collision = sum(c for c in count_values if c > 1)
        top_collisions = id_counts.most_common(10)

        # Per-prefix-depth analysis with bucket size distribution
        depth_collision_rates = []
        prefix_stats = []

        if semantic_ids and '_' in semantic_ids[0]:
            parts = [sid.split('_') for sid in semantic_ids]
            n_layers = len(parts[0])

            for depth in range(1, n_layers + 1):
                prefixes = ['_'.join(p[:depth]) for p in parts]
                prefix_counts = Counter(prefixes)
                n_unique_prefix = len(prefix_counts)
                depth_col = 1.0 - (n_unique_prefix / n_total)
                depth_collision_rates.append(round(depth_col, 4))

                # Bucket size distribution
                counts = sorted(prefix_counts.values())
                counts_t = torch.tensor(counts, dtype=torch.float32)
                n_buckets = len(counts)

                stat = {
                    'depth': depth,
                    'n_unique_prefix': n_unique_prefix,
                    'collision_rate': round(depth_col, 4),
                    'avg_items': round(n_total / n_unique_prefix, 2),
                    'min': counts[0],
                    'max': counts[-1],
                    'p50': int(counts_t.quantile(0.5).item()),
                    'p90': int(counts_t.quantile(0.9).item()),
                    'p99': int(counts_t.quantile(0.99).item()),
                }

                # Bucket size bins for quick overview
                bins = [1, 2, 5, 10, 50, 100, 500]
                for b in bins:
                    stat[f'le_{b}'] = sum(1 for c in counts if c <= b)
                stat['gt_500'] = sum(1 for c in counts if c > 500)

                prefix_stats.append(stat)

        status = self.assess_quality(collision_rate)

        return MetricResult(
            name=self.name,
            value=collision_rate,
            layer_values=depth_collision_rates,
            details={
                'n_total': n_total,
                'n_unique': n_unique,
                'collision_rate': collision_rate,
                'n_collided_ids': n_collided_ids,
                'n_items_in_collision': n_items_in_collision,
                'top_collisions': [(sid, count) for sid, count in top_collisions],
                'prefix_stats': prefix_stats,
            },
            status=status,
        )
