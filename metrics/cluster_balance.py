"""
SID Distribution Balance Metric

Measures how evenly items are distributed across the used SID combinations,
reported per prefix depth.

  depth=1: Gini over "a_*_*" prefix counts
  depth=2: Gini over "a_b_*" prefix counts
  depth=3: Gini over "a_b_c" full SID counts
"""

from typing import Any, Dict, List, Optional
from collections import Counter
import torch

from .base import BaseMetric, MetricResult


class ClusterBalanceMetric(BaseMetric):
    """SID Distribution Balance: Gini coefficient per prefix depth

    Measures how evenly items are distributed across SID prefixes.
    - Gini = 0: perfect equality (every prefix used equally)
    - Gini = 1: perfect inequality (all items map to one prefix)

    Reports Gini at each prefix depth:
      depth 1: Gini over L1 prefix counts
      depth 2: Gini over L1_L2 prefix counts
      depth 3: Gini over full SID counts

    Primary value = full-SID Gini.
    """

    name = 'cluster_balance'
    requires_model = False
    requires_semantic_ids = True

    # Lower Gini is better
    thresholds = {
        'excellent': 0.15,
        'good': 0.25,
        'acceptable': 0.40,
    }

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

        # Full SID balance (primary)
        sid_counts = Counter(semantic_ids)
        count_values = torch.tensor(list(sid_counts.values()), dtype=torch.float32)
        n_unique = len(sid_counts)

        gini = self._compute_gini(count_values)
        cv = (count_values.std() / count_values.mean()).item() if len(count_values) > 1 else 0.0
        min_max_ratio = (count_values.min() / count_values.max()).item() if count_values.max() > 0 else 0.0

        # Per-depth prefix balance
        depth_gini = []
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
                d_counts = torch.tensor(list(prefix_counts.values()), dtype=torch.float32)
                n_unique_prefix = len(prefix_counts)

                d_gini = self._compute_gini(d_counts)
                d_cv = (d_counts.std() / d_counts.mean()).item() if len(d_counts) > 1 else 0.0

                depth_gini.append(d_gini)
                depth_stats.append({
                    'depth': depth,
                    'gini': round(d_gini, 4),
                    'cv': round(d_cv, 4),
                    'n_unique_prefix': n_unique_prefix,
                    'min_count': d_counts.min().item(),
                    'max_count': d_counts.max().item(),
                    'mean_count': round(d_counts.mean().item(), 2),
                })

        status = self.assess_quality(gini)

        return MetricResult(
            name=self.name,
            value=gini,
            layer_values=depth_gini,
            details={
                'overall_gini': gini,
                'overall_cv': cv,
                'min_max_ratio': min_max_ratio,
                'n_total': n_total,
                'n_unique_sids': n_unique,
                'min_count': count_values.min().item(),
                'max_count': count_values.max().item(),
                'mean_count': count_values.mean().item(),
                'n_layers': n_layers,
                'n_clusters_per_layer': n_clusters_per_layer,
                'depth_stats': depth_stats,
            },
            status=status,
        )

    def _compute_gini(self, values: torch.Tensor) -> float:
        """Compute Gini coefficient

        Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
        where x_i are sorted values
        """
        sorted_values = torch.sort(values).values
        n = len(sorted_values)
        total = sorted_values.sum()

        if total == 0:
            return 0.0

        gini = (2 * (torch.arange(1, n + 1, dtype=torch.float32) * sorted_values).sum() / (n * total)) - (n + 1) / n
        return gini.item()
