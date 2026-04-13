"""
Codebook Utilization Metric

Measures the percentage of the N^d theoretical SID space actually used,
reported per prefix depth.

  depth=1: unique "a_*_*" / N^1
  depth=2: unique "a_b_*" / N^2
  depth=3: unique "a_b_c" / N^3  (full SID)

Reference: OneRec (arxiv 2506.13695)
"""

from typing import Any, Dict, List, Optional
import torch

from .base import BaseMetric, MetricResult


class CodebookUtilizationMetric(BaseMetric):
    """Codebook Utilization per prefix depth

    Reports utilization at each depth:
      depth 1: n_unique_L1_prefixes / N
      depth 2: n_unique_L1L2_prefixes / N^2
      depth 3: n_unique_full_sids / N^3

    layer_values[i] = utilization at depth i+1.
    Primary value = full-depth utilization (n_unique / N^L).
    """

    name = 'codebook_utilization'
    requires_model = False
    requires_semantic_ids = True

    thresholds = {}

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
        n_unique = len(set(semantic_ids))

        # Parse SIDs
        depth_stats = []
        layer_utilizations = []
        n_layers = 0
        n_clusters_per_layer = 0

        if semantic_ids and '_' in semantic_ids[0]:
            parts = [sid.split('_') for sid in semantic_ids]
            n_layers = len(parts[0])

            # Determine per-layer codebook size (max id + 1 per layer)
            n_per_layer = []
            for layer_idx in range(n_layers):
                layer_ids = [int(p[layer_idx]) for p in parts]
                max_id = max(layer_ids) + 1
                n_per_layer.append(max_id)
                n_clusters_per_layer = max(n_clusters_per_layer, max_id)

            # Per-depth utilization (theoretical = product of per-layer sizes)
            for depth in range(1, n_layers + 1):
                prefixes = set('_'.join(p[:depth]) for p in parts)
                n_unique_at_depth = len(prefixes)
                theoretical = 1
                for i in range(depth):
                    theoretical *= n_per_layer[i]
                util = n_unique_at_depth / theoretical if theoretical > 0 else 0
                layer_utilizations.append(util)
                depth_stats.append({
                    'depth': depth,
                    'n_unique': n_unique_at_depth,
                    'theoretical_space': theoretical,
                    'utilization': util,
                })

        # Primary: full-depth utilization
        if n_clusters_per_layer > 0 and n_layers > 0:
            theoretical_space = 1
            for n in n_per_layer:
                theoretical_space *= n
            space_utilization = n_unique / theoretical_space
        else:
            theoretical_space = 0
            space_utilization = 0

        return MetricResult(
            name=self.name,
            value=space_utilization,
            layer_values=layer_utilizations,
            details={
                'n_total': n_total,
                'n_unique_sids': n_unique,
                'n_clusters_per_layer': n_clusters_per_layer,
                'n_layers': n_layers,
                'theoretical_space': theoretical_space,
                'space_utilization': space_utilization,
                'item_unique_ratio': n_unique / n_total,
                'depth_stats': depth_stats,
            },
            status='unknown',
        )
