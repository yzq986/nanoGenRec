"""EvalConfig + MetricsEvaluator + CLI。"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from metrics import (
    AVAILABLE_METRICS,
    BaseMetric,
    MetricResult,
    ReportGenerator,
    print_metric_result,
)
from data.loaders import (
    load_results_from_s3, load_model_from_s3,
    load_local_results, load_local_model,
)
from eval.wrapper import RKMeansModelWrapper, load_model_wrapper


# ============================================================
# Configuration
# ============================================================

class EvalConfig:
    """Evaluation configuration"""
    # Metrics that don't require RKMeans model
    EMBEDDING_ONLY_METRICS = ['cosine_similarity', 'effective_dimension']

    # Metrics that require semantic IDs
    SEMANTIC_ID_METRICS = ['semantic_id_collision']

    # Metrics that require RKMeans model
    MODEL_METRICS = ['reconstruction_loss', 'codebook_utilization', 'entropy', 'cluster_balance']

    # Default output directory
    DEFAULT_OUTPUT_DIR = 'eval_results'


# ============================================================
# Metrics Evaluator
# ============================================================

class MetricsEvaluator:
    """Main evaluator class that orchestrates metric computation"""

    def __init__(
        self,
        embeddings: torch.Tensor,
        model: Optional[Any] = None,
        semantic_ids: Optional[List[str]] = None,
        device: str = 'cuda',
    ):
        """Initialize evaluator

        Args:
            embeddings: (N, D) tensor of embeddings
            model: Optional RKMeans model wrapper
            semantic_ids: Optional list of semantic ID strings
            device: Computation device
        """
        self.embeddings = embeddings
        self.model = model
        self.semantic_ids = semantic_ids
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.metrics: Dict[str, BaseMetric] = {}
        self.results: Dict[str, MetricResult] = {}

        # Pre-compute layer assignments if model is provided (saves recomputation)
        self.layer_assignments: Optional[List[torch.Tensor]] = None

    def register_metric(self, metric: BaseMetric) -> 'MetricsEvaluator':
        """Register a metric for evaluation"""
        self.metrics[metric.name] = metric
        return self

    def register_metrics(self, metric_names: List[str]) -> 'MetricsEvaluator':
        """Register multiple metrics by name"""
        for name in metric_names:
            if name not in AVAILABLE_METRICS:
                print(f"Warning: Unknown metric '{name}', skipping")
                continue
            metric_class = AVAILABLE_METRICS[name]
            self.register_metric(metric_class())
        return self

    def register_all_metrics(self) -> 'MetricsEvaluator':
        """Register all available metrics"""
        return self.register_metrics(list(AVAILABLE_METRICS.keys()))

    def _precompute_assignments(self):
        """Pre-compute layer assignments to avoid redundant computation"""
        if self.model is None or self.layer_assignments is not None:
            return

        n_samples = self.embeddings.shape[0]
        n_layers = len(self.model.kmeans_layers)
        chunk_size = 50000

        print(f"Pre-computing layer assignments ({n_samples:,} samples, {n_layers} layers)...")

        self.layer_assignments = []
        current_residuals = self.embeddings.clone()

        # Normalize input once (layer 0 only), matching generate_semantic_ids
        if self.model.normalize_residuals:
            print(f"  Normalizing input embeddings (layer 0 only)...")
            normalized = []
            for i in range(0, n_samples, chunk_size):
                chunk = current_residuals[i:i+chunk_size].to(self.device)
                chunk = F.normalize(chunk, p=2, dim=1).cpu()
                normalized.append(chunk)
            current_residuals = torch.cat(normalized, dim=0)

        for layer_idx, kmeans in enumerate(self.model.kmeans_layers):
            print(f"  Layer {layer_idx + 1}/{n_layers}...", end=" ", flush=True)

            # No per-layer normalization — residuals keep raw scale

            # Get assignments (via predict — works for both KMeans and FSQ layers)
            all_assignments = []
            for i in range(0, n_samples, chunk_size):
                chunk = current_residuals[i:i+chunk_size].to(self.device)
                batch_assignments = kmeans.predict(chunk).cpu()
                all_assignments.append(batch_assignments)
            assignments = torch.cat(all_assignments, dim=0)
            self.layer_assignments.append(assignments)

            # Compute residuals for next layer: raw subtraction (no scaling)
            layer_centroids = kmeans.centroids[assignments].cpu()
            current_residuals = current_residuals - layer_centroids

            print("done", flush=True)

        print(f"  All {n_layers} layers computed")

    def evaluate(self) -> Dict[str, MetricResult]:
        """Run all registered metrics"""
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

            try:
                result = metric.compute(
                    embeddings=self.embeddings,
                    model=self.model,
                    semantic_ids=self.semantic_ids,
                    layer_assignments=self.layer_assignments,
                    normalize_residuals=self.model.normalize_residuals if self.model else True,
                )
                self.results[name] = result
                print_metric_result(result)

            except Exception as e:
                print(f"  Error computing {name}: {e}")
                import traceback
                traceback.print_exc()

        return self.results

    def evaluate_and_report(
        self,
        model_name: str,
        output_dir: str = 'eval_results',
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Run evaluation and generate reports"""
        # Run evaluation
        results = self.evaluate()

        # Generate reports
        generator = ReportGenerator(
            model_name=model_name,
            output_dir=output_dir,
            metadata=metadata or {},
        )
        generator.add_results(results)

        return generator.generate_all()


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Embedding & Semantic ID Metrics Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full evaluation with model
    python -m gr_demo eval \\
        --results_path s3://bucket/results.parquet \\
        --model_path s3://bucket/model.pt

    # Embedding-only metrics
    python -m gr_demo eval --results_path local/results.parquet --embedding_only

    # Specific metrics
    python -m gr_demo eval --results_path ... --metrics reconstruction_loss entropy

Available metrics:
    - reconstruction_loss: Quantization precision (requires model)
    - codebook_utilization: Codebook usage rate (requires model)
    - entropy: Token distribution uniformity (requires model)
    - cosine_similarity: Embedding discrimination (embedding only)
    - effective_dimension: Dimension utilization (embedding only)
    - semantic_id_collision: Unique ID rate (requires semantic IDs)
    - cluster_balance: Cluster size distribution (requires model)
        """
    )

    # Input paths
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to results parquet (S3 or local)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to RKMeans model .pt file (S3 or local)')

    # Metric selection
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                        help='Specific metrics to compute')
    parser.add_argument('--all_metrics', action='store_true',
                        help='Compute all available metrics')
    parser.add_argument('--embedding_only', action='store_true',
                        help='Only compute embedding-based metrics (no model required)')

    # Output
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Output directory for reports')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name for reports (auto-detected if not specified)')

    # Processing
    parser.add_argument('--sample_size', type=int, default=0,
                        help='Sample size for evaluation (0 = use all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Embedding & Semantic ID Metrics Evaluation")
    print("=" * 60)

    # Determine model name
    if args.model_name:
        model_name = args.model_name
    elif args.model_path:
        # Extract from path
        model_name = os.path.basename(os.path.dirname(args.model_path.rstrip('/')))
    else:
        model_name = 'unknown'

    print(f"Model: {model_name}")
    print(f"Results: {args.results_path}")
    print(f"Model path: {args.model_path or 'N/A'}")
    print("=" * 60)

    # Load data
    if args.results_path.startswith('s3://'):
        content_ids, embeddings, semantic_ids = load_results_from_s3(
            args.results_path, args.sample_size
        )
    else:
        content_ids, embeddings, semantic_ids = load_local_results(
            args.results_path, args.sample_size
        )

    if embeddings is None:
        print("Error: No embeddings found in results file")
        return 1

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Load model if provided
    model = None
    if args.model_path and not args.embedding_only:
        if args.model_path.startswith('s3://'):
            model_data = load_model_from_s3(args.model_path)
        else:
            model_data = load_local_model(args.model_path)
        model = load_model_wrapper(model_data, device=args.device)

    # Determine which metrics to run
    if args.metrics:
        metric_names = args.metrics
    elif args.embedding_only:
        metric_names = EvalConfig.EMBEDDING_ONLY_METRICS
    elif args.all_metrics:
        metric_names = list(AVAILABLE_METRICS.keys())
    else:
        # Default: run all applicable metrics
        if model is not None:
            metric_names = list(AVAILABLE_METRICS.keys())
        else:
            metric_names = EvalConfig.EMBEDDING_ONLY_METRICS + EvalConfig.SEMANTIC_ID_METRICS

    print(f"\nMetrics to compute: {metric_names}")

    # Create evaluator
    evaluator = MetricsEvaluator(
        embeddings=embeddings_tensor,
        model=model,
        semantic_ids=semantic_ids,
        device=args.device,
    )
    evaluator.register_metrics(metric_names)

    # Run evaluation and generate reports
    metadata = {
        'n_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'n_unique_semantic_ids': len(set(semantic_ids)),
        'results_path': args.results_path,
        'model_path': args.model_path or 'N/A',
    }

    report_paths = evaluator.evaluate_and_report(
        model_name=model_name,
        output_dir=args.output_dir,
        metadata=metadata,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, result in evaluator.results.items():
        status_icon = {'excellent': '✓', 'good': '○', 'acceptable': '△', 'poor': '✗'}.get(result.status, '?')
        print(f"  [{status_icon}] {name}: {result.value:.4f} ({result.status})")

    print("\nReports generated:")
    for fmt, path in report_paths.items():
        print(f"  {fmt}: {path}")

    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
