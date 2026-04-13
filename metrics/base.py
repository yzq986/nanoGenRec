"""
Base classes for metrics framework

Provides:
- MetricResult: Standardized data class for metric outputs
- BaseMetric: Abstract base class for all metrics
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch


@dataclass
class MetricResult:
    """Standardized result container for metrics

    Attributes:
        name: Metric identifier (e.g., 'reconstruction_loss')
        value: Primary aggregate value
        layer_values: Per-layer values (if applicable)
        details: Additional statistics and metadata
        status: Quality assessment ('excellent', 'good', 'acceptable', 'poor')
    """
    name: str
    value: float
    layer_values: List[float] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = 'unknown'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'value': self.value,
            'layer_values': self.layer_values,
            'details': self.details,
            'status': self.status,
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dict for grid search / tabular storage.

        Returns keys like:
            {name}: primary value
            {name}_depth: layer_values list
            {name}_prefix_stats: details['prefix_stats']
            {name}_depth_stats: details['depth_stats']
            {name}_space_utilization: details['space_utilization']
            {name}_prefix_avg_items: derived from prefix_stats avg_items
        """
        flat: Dict[str, Any] = {self.name: round(self.value, 4)}

        if self.layer_values:
            flat[f'{self.name}_depth'] = [round(v, 4) for v in self.layer_values]

        for detail_key in ('prefix_stats', 'depth_stats'):
            if detail_key in self.details:
                flat[f'{self.name}_{detail_key}'] = self.details[detail_key]

        if 'space_utilization' in self.details:
            flat[f'{self.name}_space_utilization'] = self.details['space_utilization']

        # Derive prefix_avg_items from prefix_stats if present
        if 'prefix_stats' in self.details:
            avg_items = [s.get('avg_items', 0) for s in self.details['prefix_stats']]
            if avg_items:
                flat[f'{self.name}_prefix_avg_items'] = avg_items

        return flat

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricResult':
        """Create from dictionary"""
        return cls(
            name=data['name'],
            value=data['value'],
            layer_values=data.get('layer_values', []),
            details=data.get('details', {}),
            status=data.get('status', 'unknown'),
        )


class BaseMetric(ABC):
    """Abstract base class for all metrics

    Subclasses must implement:
    - name: Unique identifier for the metric
    - compute(): Main computation method

    Optionally override:
    - requires_model: Whether RKMeans model is needed
    - requires_semantic_ids: Whether semantic IDs are needed
    - assess_quality(): Status assessment logic
    """

    # Metric identifier
    name: str = 'base_metric'

    # Requirements
    requires_model: bool = False
    requires_semantic_ids: bool = False

    # Thresholds for quality assessment (override in subclasses)
    thresholds: Dict[str, float] = {}

    @abstractmethod
    def compute(
        self,
        embeddings: torch.Tensor,
        model: Optional[Any] = None,
        semantic_ids: Optional[List[str]] = None,
        layer_assignments: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> MetricResult:
        """Compute the metric

        Args:
            embeddings: (N, D) tensor of embeddings
            model: Optional RKMeans model (if requires_model=True)
            semantic_ids: Optional list of semantic ID strings
            layer_assignments: Optional per-layer cluster assignments
            **kwargs: Additional metric-specific parameters

        Returns:
            MetricResult with computed values
        """
        pass

    def assess_quality(self, value: float) -> str:
        """Assess quality based on thresholds

        Override this method for custom assessment logic.
        Default uses thresholds dict with keys: excellent, good, acceptable
        """
        if not self.thresholds:
            return 'unknown'

        # Default: lower is better
        if value <= self.thresholds.get('excellent', float('-inf')):
            return 'excellent'
        elif value <= self.thresholds.get('good', float('-inf')):
            return 'good'
        elif value <= self.thresholds.get('acceptable', float('-inf')):
            return 'acceptable'
        else:
            return 'poor'

    def validate_inputs(
        self,
        embeddings: torch.Tensor,
        model: Optional[Any] = None,
        semantic_ids: Optional[List[str]] = None,
    ) -> None:
        """Validate required inputs are provided

        Raises:
            ValueError: If required inputs are missing
        """
        if self.requires_model and model is None:
            raise ValueError(f"Metric '{self.name}' requires model but none provided")
        if self.requires_semantic_ids and semantic_ids is None:
            raise ValueError(f"Metric '{self.name}' requires semantic_ids but none provided")
        if embeddings is None or len(embeddings) == 0:
            raise ValueError(f"Metric '{self.name}' requires non-empty embeddings")


def print_metric_result(result: MetricResult) -> None:
    """Pretty-print a MetricResult.

    Shows: value + status, per-depth layer_values, structured depth/prefix stats,
    and key scalar details.
    """
    print(f"  {result.name}: {result.value:.4f} ({result.status})")

    # Per-depth values
    if result.layer_values:
        for i, lv in enumerate(result.layer_values):
            print(f"    depth {i+1}: {lv:.4f}")

    # Structured depth details
    for key in ('depth_stats', 'prefix_stats', 'depth_acc_beam', 'depth_hit@5', 'depth_hit@10'):
        if key in result.details:
            val = result.details[key]
            if isinstance(val, list) and val and isinstance(val[0], dict):
                for d in val:
                    depth = d.get('depth', '?')
                    parts = [f"{k}={v}" for k, v in d.items() if k != 'depth']
                    print(f"    depth {depth}: {', '.join(parts)}")
            elif isinstance(val, list):
                print(f"    {key}: {[f'{v:.4f}' for v in val]}")

    # Key scalar details
    for key in (
        'n_unique_sids', 'n_valid_users', 'n_contents_evaluated',
        'spearman_correlation', 'p_value', 'n_pairs',
        'mean_separation', 'n_users',
        'user_mean_similarity', 'random_mean_similarity', 'lift_over_random',
        'mean_hit_rate', 'prefix_layers',
        'perplexity', 'random_perplexity', 'baseline_perplexity',
        'avg_items_per_sid',
    ):
        if key in result.details:
            val = result.details[key]
            if isinstance(val, float):
                print(f"    {key}: {val:.4f}")
            else:
                print(f"    {key}: {val}")

    # Item recall (with baseline comparison)
    for key in ('item_recall@10', 'item_recall@50', 'item_recall@100', 'item_recall@500'):
        if key in result.details:
            val = result.details[key]
            baseline_key = f'baseline_{key}'
            baseline = result.details.get(baseline_key)
            bstr = f" (baseline: {baseline:.4f})" if baseline is not None else ""
            print(f"    {key}: {val:.4f}{bstr}")

    # Baseline list details (depth_acc, hit@K)
    for key in ('baseline_depth_acc_beam', 'baseline_depth_hit@5', 'baseline_depth_hit@10'):
        if key in result.details:
            val = result.details[key]
            if isinstance(val, list):
                print(f"    {key}: {[f'{v:.4f}' for v in val]}")
