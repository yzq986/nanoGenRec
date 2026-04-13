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

        # Full collision stats
        id_counts = Counter(semantic_ids)
        count_values = list(id_counts.values())
        n_collided_ids = sum(1 for c in count_values if c > 1)
        n_items_in_collision = sum(c for c in count_values if c > 1)
        n_exclusive_items = n_total - n_items_in_collision
        top_collisions = id_counts.most_common(10)

        # OneMall-aligned metrics (arxiv 2601.21770 Table 5)
        # conflict_rate: SID→Item direction, fraction of SIDs shared by multiple items
        conflict_rate = n_collided_ids / n_unique if n_unique > 0 else 0.0
        # exclusivity: Item→SID direction, fraction of items with a unique SID
        exclusivity = n_exclusive_items / n_total if n_total > 0 else 0.0

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

                # Per-depth conflict_rate & exclusivity
                depth_prefix_counts = sorted(prefix_counts.values())
                n_collided_prefixes = sum(1 for c in depth_prefix_counts if c > 1)
                n_exclusive_at_depth = sum(1 for c in depth_prefix_counts if c == 1)
                depth_conflict = n_collided_prefixes / n_unique_prefix if n_unique_prefix > 0 else 0.0
                depth_exclusivity = n_exclusive_at_depth / n_total if n_total > 0 else 0.0

                # Bucket size distribution
                counts = depth_prefix_counts
                counts_t = torch.tensor(counts, dtype=torch.float32)
                n_buckets = len(counts)

                stat = {
                    'depth': depth,
                    'n_unique_prefix': n_unique_prefix,
                    'collision_rate': round(depth_col, 4),
                    'conflict_rate': round(depth_conflict, 4),
                    'exclusivity': round(depth_exclusivity, 4),
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

        status = self.assess_quality(conflict_rate)

        return MetricResult(
            name=self.name,
            value=conflict_rate,
            layer_values=depth_collision_rates,
            details={
                'n_total': n_total,
                'n_unique': n_unique,
                'conflict_rate': conflict_rate,
                'exclusivity': exclusivity,
                'n_collided_ids': n_collided_ids,
                'n_exclusive_items': n_exclusive_items,
                'n_items_in_collision': n_items_in_collision,
                'top_collisions': [(sid, count) for sid, count in top_collisions],
                'prefix_stats': prefix_stats,
            },
            status=status,
        )
