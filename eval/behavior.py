"""BehaviorMetricsEvaluator + behavior 数据加载。"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from gr_demo.metrics import (
    INTRINSIC_METRICS,
    BEHAVIOR_METRICS,
    AVAILABLE_METRICS,
    MetricResult,
    ReportGenerator,
    print_metric_result,
)
from gr_demo.data.loaders import (
    load_results_from_s3, load_model_from_s3,
    load_local_results, load_local_model,
)
from gr_demo.eval.wrapper import RKMeansModelWrapper
from gr_demo.eval.evaluator import MetricsEvaluator


# ============================================================
# Behavior Data Loading
# ============================================================

def load_behavior_from_s3(
    s3_path: str,
    content_ids_filter: Optional[set] = None,
) -> Dict[str, np.ndarray]:
    """Load user behavior data from S3"""
    import pandas as pd
    import s3fs

    print(f"Loading behavior data from {s3_path}...")

    fs = s3fs.S3FileSystem()
    s3_path_clean = s3_path.replace('s3://', '')

    # Handle both single file and directory
    if s3_path_clean.endswith('.parquet'):
        files = [s3_path_clean]
    else:
        files = fs.glob(f"{s3_path_clean}/*.parquet")

    print(f"Found {len(files)} parquet files")

    dfs = []
    for i, f in enumerate(files):
        with fs.open(f, 'rb') as file:
            df = pd.read_parquet(file)
            dfs.append(df)
        if (i + 1) % 5 == 0:
            print(f"  Loaded {i + 1}/{len(files)} files...")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df):,} rows")

    # Filter by content_ids if provided
    if content_ids_filter:
        original_len = len(combined_df)
        combined_df = combined_df[combined_df['iid'].isin(content_ids_filter)]
        print(f"Filtered to {len(combined_df):,} rows ({len(combined_df)/original_len:.1%} matched)")

    # Extract arrays
    behavior_data = {
        'uid': combined_df['uid'].values,
        'iid': combined_df['iid'].values,
        'action_bitmap': combined_df['action_bitmap'].values.astype(np.int32),
    }

    # Stats
    n_users = len(np.unique(behavior_data['uid']))
    n_items = len(np.unique(behavior_data['iid']))
    n_positive = np.sum(behavior_data['action_bitmap'] > 0)
    n_negative = np.sum(behavior_data['action_bitmap'] < 0)

    print(f"Stats:")
    print(f"  Unique users: {n_users:,}")
    print(f"  Unique items: {n_items:,}")
    print(f"  Positive interactions: {n_positive:,}")
    print(f"  Negative interactions: {n_negative:,}")

    return behavior_data


def load_behavior_local(
    path: str,
    content_ids_filter: Optional[set] = None,
) -> Dict[str, np.ndarray]:
    """Load behavior data from local parquet"""
    import pandas as pd

    print(f"Loading behavior data from {path}...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows")

    if content_ids_filter:
        df = df[df['iid'].isin(content_ids_filter)]
        print(f"Filtered to {len(df):,} rows")

    return {
        'uid': df['uid'].values,
        'iid': df['iid'].values,
        'action_bitmap': df['action_bitmap'].values.astype(np.int32),
    }


# ============================================================
# Extended Evaluator with Behavior
# ============================================================

class BehaviorMetricsEvaluator(MetricsEvaluator):
    """Extended evaluator with behavior data support"""

    def __init__(
        self,
        embeddings: torch.Tensor,
        content_ids: np.ndarray,
        semantic_ids: List[str],
        model: Optional[Any] = None,
        behavior_data: Optional[Dict] = None,
        device: str = 'cuda',
    ):
        super().__init__(
            embeddings=embeddings,
            model=model,
            semantic_ids=semantic_ids,
            device=device,
        )
        self.content_ids = content_ids
        self.behavior_data = behavior_data

        # Build content_id -> index mapping
        self.content_id_to_idx = {
            cid: idx for idx, cid in enumerate(content_ids)
        }

    def register_intrinsic_metrics(self) -> 'BehaviorMetricsEvaluator':
        """Register only intrinsic metrics"""
        return self.register_metrics(list(INTRINSIC_METRICS.keys()))

    def register_behavior_metrics(self) -> 'BehaviorMetricsEvaluator':
        """Register only behavior-based metrics"""
        return self.register_metrics(list(BEHAVIOR_METRICS.keys()))

    def evaluate(self, metric_kwargs: Optional[Dict[str, Dict]] = None) -> Dict[str, MetricResult]:
        """Run all registered metrics with behavior data."""
        metric_kwargs = metric_kwargs or {}

        # Pre-compute assignments if needed
        needs_assignments = any(
            m.requires_model for m in self.metrics.values()
        )
        if needs_assignments and self.model is not None:
            self._precompute_assignments()

        self.results = {}

        for name, metric in self.metrics.items():
            print(f"\nComputing {name}...")

            # Check requirements
            if metric.requires_model and self.model is None:
                print(f"  Skipping: requires model but none provided")
                continue
            if metric.requires_semantic_ids and self.semantic_ids is None:
                print(f"  Skipping: requires semantic_ids but none provided")
                continue

            # Check behavior data for behavior metrics
            is_behavior_metric = name in BEHAVIOR_METRICS
            if is_behavior_metric and self.behavior_data is None:
                print(f"  Skipping: requires behavior_data but none provided")
                continue

            try:
                # Build kwargs
                kwargs = {
                    'embeddings': self.embeddings,
                    'model': self.model,
                    'semantic_ids': self.semantic_ids,
                    'layer_assignments': self.layer_assignments,
                }

                # Add behavior data for behavior metrics
                if is_behavior_metric:
                    kwargs['behavior_data'] = self.behavior_data
                    kwargs['content_id_to_idx'] = self.content_id_to_idx
                    kwargs['content_ids'] = self.content_ids

                if self.model:
                    kwargs['normalize_residuals'] = self.model.normalize_residuals

                # Merge per-metric extra kwargs
                if name in metric_kwargs:
                    kwargs.update(metric_kwargs[name])

                result = metric.compute(**kwargs)
                self.results[name] = result
                print_metric_result(result)

            except Exception as e:
                print(f"  Error computing {name}: {e}")
                import traceback
                traceback.print_exc()

        return self.results


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Embedding Evaluation with User Behavior Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input paths
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to results parquet (S3 or local)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to RKMeans model .pt file')
    parser.add_argument('--behavior_path', type=str, default=None,
                        help='Path to user behavior parquet (S3 or local)')

    # Metric selection
    parser.add_argument('--intrinsic_only', action='store_true',
                        help='Only compute intrinsic metrics')
    parser.add_argument('--behavior_only', action='store_true',
                        help='Only compute behavior-based metrics')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                        help='Specific metrics to compute')

    # Output
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Output directory')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name for reports')

    # Processing
    parser.add_argument('--sample_size', type=int, default=0,
                        help='Sample size (0 = all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Embedding Evaluation with User Behavior")
    print("=" * 60)

    # Determine model name
    if args.model_name:
        model_name = args.model_name
    elif args.results_path:
        # Extract from path
        parts = args.results_path.replace('s3://', '').split('/')
        for p in parts:
            if 'qwen' in p.lower():
                model_name = p
                break
        else:
            model_name = 'unknown'
    else:
        model_name = 'unknown'

    print(f"Model: {model_name}")
    print(f"Results: {args.results_path}")
    print(f"Behavior: {args.behavior_path or 'N/A'}")
    print("=" * 60)

    # Load results
    if args.results_path.startswith('s3://'):
        content_ids, embeddings, semantic_ids = load_results_from_s3(
            args.results_path, args.sample_size
        )
    else:
        content_ids, embeddings, semantic_ids = load_local_results(
            args.results_path, args.sample_size
        )

    if embeddings is None:
        print("Error: No embeddings found")
        return 1

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Load model if provided
    model = None
    if args.model_path and not args.behavior_only:
        if args.model_path.startswith('s3://'):
            model_data = load_model_from_s3(args.model_path)
        else:
            model_data = load_local_model(args.model_path)
        model = RKMeansModelWrapper(model_data, device=args.device)

    # Load behavior data if provided
    behavior_data = None
    if args.behavior_path:
        content_ids_set = set(content_ids)
        if args.behavior_path.startswith('s3://'):
            behavior_data = load_behavior_from_s3(args.behavior_path, content_ids_set)
        else:
            behavior_data = load_behavior_local(args.behavior_path, content_ids_set)

    # Create evaluator
    evaluator = BehaviorMetricsEvaluator(
        embeddings=embeddings_tensor,
        content_ids=content_ids,
        semantic_ids=semantic_ids,
        model=model,
        behavior_data=behavior_data,
        device=args.device,
    )

    # Register metrics
    if args.metrics:
        evaluator.register_metrics(args.metrics)
    elif args.intrinsic_only:
        evaluator.register_intrinsic_metrics()
    elif args.behavior_only:
        evaluator.register_behavior_metrics()
    else:
        # Register all applicable metrics
        evaluator.register_intrinsic_metrics()
        if behavior_data is not None:
            evaluator.register_behavior_metrics()

    # Run evaluation
    metadata = {
        'n_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'n_unique_semantic_ids': len(set(semantic_ids)),
        'has_behavior_data': behavior_data is not None,
        'results_path': args.results_path,
        'behavior_path': args.behavior_path or 'N/A',
    }

    if behavior_data is not None:
        metadata['n_behavior_interactions'] = len(behavior_data['uid'])
        metadata['n_behavior_users'] = len(np.unique(behavior_data['uid']))

    report_paths = evaluator.evaluate_and_report(
        model_name=model_name,
        output_dir=args.output_dir,
        metadata=metadata,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\n[Intrinsic Metrics]")
    for name in INTRINSIC_METRICS:
        if name in evaluator.results:
            result = evaluator.results[name]
            icon = {'excellent': '✓', 'good': '○', 'acceptable': '△', 'poor': '✗'}.get(result.status, '?')
            print(f"  [{icon}] {name}: {result.value:.4f} ({result.status})")

    print("\n[Behavior Metrics]")
    for name in BEHAVIOR_METRICS:
        if name in evaluator.results:
            result = evaluator.results[name]
            icon = {'excellent': '✓', 'good': '○', 'acceptable': '△', 'poor': '✗'}.get(result.status, '?')
            print(f"  [{icon}] {name}: {result.value:.4f} ({result.status})")

    print("\nReports:")
    for fmt, path in report_paths.items():
        print(f"  {fmt}: {path}")

    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
