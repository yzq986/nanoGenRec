#!/usr/bin/env python3
"""EXP-011: Codebook Size Ablation — cached KMeans, sweep FSQ + OPQ.

Optimized flow:
  1. Load embeddings + behavior data ONCE
  2. Group FSQ configs by KMeans cluster size → train KMeans once per group
  3. For each FSQ config in group: train only FSQ layer on cached residuals
  4. Run OPQ configs separately (no KMeans sharing)

~2-3x faster than running each config independently.
"""

import json
import os
import sys
import time
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from gr_demo.model.rkmeans import FaissKMeansLayer
from gr_demo.model.fsq import FSQ_LEVEL_CONFIGS, LearnedFSQLayer
from gr_demo.model.rkmeans_fsq import ResKmeansFSQ, generate_semantic_ids_fsq
from gr_demo.model.opq import OPQQuantizer
from gr_demo.eval.wrapper import ResKmeansFSQModelWrapper, OPQModelWrapper
from gr_demo.eval.behavior import BehaviorMetricsEvaluator
from gr_demo.metrics import INTRINSIC_METRICS


# ============================================================
# Experiment configs
# ============================================================

FSQ_CONFIGS = [
    # (name, kmeans_clusters, fsq_levels_key, description)
    ('exp011-1024x3-5d',         1024, '5d_1024',  '1024x3 multi-level [4,4,4,4,4]'),
    ('exp011-1024x3-10d-binary', 1024, '10d_1024', '1024x3 binary [2]x10'),
    ('exp011-4096x3-6d',         4096, '6d_4096',  '4096x3 OneMall [4,4,4,4,4,4]'),
    ('exp011-4096x3-12d-binary', 4096, '12d_4096', '4096x3 binary [2]x12'),
]

OPQ_CONFIGS = [
    # (name, n_subvectors, n_clusters_per_sub, description)
    ('exp011-opq-3x1024', 3, 1024, 'OPQ 3x1024 (30 bit)'),
    ('exp011-opq-3x4096', 3, 4096, 'OPQ 3x4096 (36 bit)'),
]

FSQ_PROJECTION = 'mlp'
FSQ_MLP_HIDDEN = 64
FSQ_EPOCHS = 50
NITER = 25
NREDO = 3
NORMALIZE_RESIDUALS = True


# ============================================================
# Data loading (once)
# ============================================================

def load_data():
    """Load embeddings (filtered by exposure) + behavior data."""
    from gr_demo.config import MODEL_CONFIGS, EFS_EMBEDDING_CACHE
    from gr_demo.data.loaders import load_exposed_iids

    model_key = 'qwen3-0.6b'
    embedding_cache_dir = f'{EFS_EMBEDDING_CACHE}/{model_key}'
    incremental_path = f'{embedding_cache_dir}/incremental_cache.npy'
    embedding_path = f'{embedding_cache_dir}/embeddings.npy'
    content_ids_path = f'{embedding_cache_dir}/content_ids.npy'

    print("Loading embeddings...")
    t0 = time.time()
    if os.path.exists(incremental_path):
        cache_dict = np.load(incremental_path, allow_pickle=True).item()
        content_ids = np.array(list(cache_dict.keys()))
        embeddings = np.array(list(cache_dict.values()), dtype=np.float32)
    elif os.path.exists(embedding_path):
        embeddings = np.load(embedding_path, allow_pickle=True)
        content_ids = np.load(content_ids_path, allow_pickle=True)
    else:
        raise FileNotFoundError(f"No embedding cache at {embedding_cache_dir}")
    print(f"  {len(content_ids):,} embeddings, dim={embeddings.shape[1]} ({time.time()-t0:.1f}s)")

    # Filter by exposure
    print("Filtering by exposed IIDs...")
    exposed_iids = load_exposed_iids('auto')
    cid_str = np.array([str(c) for c in content_ids])
    mask = np.isin(cid_str, list(exposed_iids))
    embeddings = embeddings[mask]
    content_ids = content_ids[mask]
    print(f"  {len(embeddings):,} exposed items")

    tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Behavior data
    print("Loading behavior data...")
    from gr_demo.eval.batch import load_all_behavior_data
    behavior_data = load_all_behavior_data()
    print(f"  {len(behavior_data['uid']):,} interactions")

    return tensor, content_ids, behavior_data


# ============================================================
# KMeans caching
# ============================================================

def train_kmeans_layers(embeddings: torch.Tensor, n_clusters: int, num_gpus: int):
    """Train 2-layer KMeans, return (kmeans_layers, residuals_after_L2)."""
    n_samples = embeddings.shape[0]
    n_features = embeddings.shape[1]
    device = "cuda:0" if num_gpus > 0 else "cpu"

    # Normalize input
    print("  Normalizing embeddings...")
    normalized = []
    for i in range(0, n_samples, 100000):
        chunk = embeddings[i:i+100000].to(device)
        chunk = F.normalize(chunk, p=2, dim=1).cpu()
        normalized.append(chunk)
    current_residuals = torch.cat(normalized, dim=0)

    kmeans_layers = []
    for layer_idx in range(2):
        print(f"\n  Training KMeans Layer {layer_idx+1}/2 (clusters={n_clusters})...")
        km = FaissKMeansLayer(n_clusters, n_features, gpu=(num_gpus > 0))
        km.train(current_residuals, niter=NITER, nredo=NREDO)
        kmeans_layers.append(km)

        # Compute residuals
        new_residuals = []
        for i in range(0, n_samples, 50000):
            chunk = current_residuals[i:i+50000]
            assignments = km.predict(chunk)
            assigned_centroids = km.centroids[assignments].cpu()
            new_residuals.append(chunk - assigned_centroids)
        current_residuals = torch.cat(new_residuals, dim=0)

        norm = torch.norm(current_residuals, dim=1).mean().item()
        print(f"  Residual norm: {norm:.6f}")

    return kmeans_layers, current_residuals


def run_fsq_on_residuals(
    kmeans_layers, residuals, embeddings, content_ids,
    n_clusters, fsq_levels_key, behavior_data, device='cuda',
):
    """Train FSQ on pre-computed residuals, evaluate, return metrics dict."""
    fsq_levels = FSQ_LEVEL_CONFIGS[fsq_levels_key]
    n_features = embeddings.shape[1]

    print(f"  Training FSQ layer: {fsq_levels_key} {fsq_levels}...")
    t0 = time.time()

    fsq_layer = LearnedFSQLayer(
        fsq_levels, n_features,
        hidden_dim=FSQ_MLP_HIDDEN,
        epochs=FSQ_EPOCHS,
        device=device,
    )
    fsq_layer.train(residuals)
    fsq_time = time.time() - t0

    # Build full model for eval (assemble from cached parts)
    model = ResKmeansFSQ.__new__(ResKmeansFSQ)
    model.n_kmeans_clusters = n_clusters
    model.fsq_levels = fsq_levels
    model.n_features = n_features
    model.normalize_residuals = NORMALIZE_RESIDUALS
    model.n_layers = 3
    model.primary_device = device
    model.kmeans_layers = kmeans_layers
    model.fsq_layer = fsq_layer

    # Generate SIDs
    semantic_ids = generate_semantic_ids_fsq(model, embeddings, NORMALIZE_RESIDUALS)

    # Build wrapper for eval
    model_data = {
        'model_type': 'rkmeans_fsq',
        'centroids_list': [km.get_centroids().cpu() for km in kmeans_layers],
        'fsq_state': fsq_layer.save_state(),
        'normalize_residuals': NORMALIZE_RESIDUALS,
        'n_layers': 3,
        'n_kmeans_clusters': n_clusters,
        'n_features': n_features,
        'fsq_levels': fsq_levels,
    }
    wrapper = ResKmeansFSQModelWrapper(model_data, device=device)

    # Evaluate
    evaluator = BehaviorMetricsEvaluator(
        embeddings=embeddings, content_ids=content_ids,
        semantic_ids=semantic_ids, model=wrapper,
        behavior_data=behavior_data, device=device,
    )
    evaluator.register_metrics(list(INTRINSIC_METRICS.keys()))
    evaluator.register_metrics(['embedding_hit_rate', 'semantic_neighbor_hit_rate'])
    metric_results = evaluator.evaluate()

    results = {}
    for mr in metric_results.values():
        results.update(mr.to_flat_dict())

    from gr_demo.model.fsq import _codebook_size
    results['quantizer_type'] = 'rkmeans_fsq'
    results['fsq_levels_key'] = fsq_levels_key
    results['fsq_codebook_size'] = _codebook_size(fsq_levels)
    results['fsq_projection'] = FSQ_PROJECTION
    results['fsq_mlp_hidden'] = FSQ_MLP_HIDDEN

    return results, fsq_time


def run_opq_config(embeddings, content_ids, behavior_data, n_sub, n_cpersub, device='cuda'):
    """Run single OPQ experiment."""
    from gr_demo.eval.hyperparam import run_single_experiment_opq
    n_features = embeddings.shape[1]

    metrics, train_time = run_single_experiment_opq(
        train_embeddings=embeddings, eval_embeddings=embeddings,
        content_ids=content_ids, n_features=n_features,
        n_subvectors=n_sub, n_clusters_per_sub=n_cpersub,
        behavior_data=behavior_data, device=device,
        only_sid=False, run_ntp=False,
    )
    return metrics, train_time


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("EXP-011: Codebook Size Ablation (cached KMeans)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # ── Step 1: Load data once ──
    t_start = time.time()
    embeddings, content_ids, behavior_data = load_data()
    load_time = time.time() - t_start
    print(f"\nData loaded in {load_time:.1f}s\n")

    all_results = []
    exp_dir = os.path.join(REPO_ROOT, "experiments", "hyperparam", f"{date.today().isoformat()}_exp011-codebook-ablation")
    os.makedirs(exp_dir, exist_ok=True)
    json_path = os.path.join(exp_dir, "results.json")

    def save_results():
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Step 2: FSQ configs, grouped by KMeans cluster size ──
    from collections import defaultdict
    groups = defaultdict(list)
    for name, clusters, fsq_key, desc in FSQ_CONFIGS:
        groups[clusters].append((name, fsq_key, desc))

    for clusters, configs in sorted(groups.items()):
        print(f"\n{'#' * 60}")
        print(f"# KMeans cluster={clusters} — {len(configs)} FSQ variants")
        print(f"{'#' * 60}")

        t0 = time.time()
        kmeans_layers, residuals = train_kmeans_layers(embeddings, clusters, num_gpus)
        kmeans_time = time.time() - t0
        print(f"\n  KMeans trained in {kmeans_time:.1f}s (shared across {len(configs)} configs)")

        for name, fsq_key, desc in configs:
            print(f"\n  >>> {name}: {desc}")
            metrics, fsq_time = run_fsq_on_residuals(
                kmeans_layers, residuals, embeddings, content_ids,
                clusters, fsq_key, behavior_data, device,
            )

            col = metrics.get('semantic_id_collision', 'N/A')
            snhr = metrics.get('semantic_neighbor_hit_rate', 'N/A')
            print(f"      collision={col}  semantic_neighbor_HR={snhr}  fsq_time={fsq_time:.0f}s")

            all_results.append({
                'name': name,
                'description': desc,
                'quantizer_type': 'rkmeans_fsq',
                'num_clusters': clusters,
                'fsq_levels_key': fsq_key,
                'fsq_levels': FSQ_LEVEL_CONFIGS[fsq_key],
                'metrics': metrics,
                'kmeans_time': round(kmeans_time, 1),
                'fsq_time': round(fsq_time, 1),
            })
            save_results()

    # ── Step 3: OPQ configs ──
    for name, n_sub, n_cpersub, desc in OPQ_CONFIGS:
        print(f"\n{'#' * 60}")
        print(f"# {name}: {desc}")
        print(f"{'#' * 60}")

        metrics, train_time = run_opq_config(
            embeddings, content_ids, behavior_data,
            n_sub, n_cpersub, device,
        )

        col = metrics.get('semantic_id_collision', 'N/A')
        snhr = metrics.get('semantic_neighbor_hit_rate', 'N/A')
        print(f"    collision={col}  semantic_neighbor_HR={snhr}  time={train_time:.0f}s")

        all_results.append({
            'name': name,
            'description': desc,
            'quantizer_type': 'opq',
            'n_subvectors': n_sub,
            'n_clusters_per_sub': n_cpersub,
            'metrics': metrics,
            'train_time': round(train_time, 1),
        })
        save_results()

    # ── Summary ──
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"EXP-011 Complete — {len(all_results)} configs in {total_time:.0f}s")
    print(f"{'=' * 60}")
    print(f"\nResults: {json_path}")
    print(f"\n{'Reference (EXP-008):'}")
    print(f"  A: MLP-FSQ 1024x1024x4096 [4,4,4,4,4,4] → semantic_neighbor_HR=0.078, collision=10.7%")
    print(f"  B: OPQ 4x256 (32 bit)                     → semantic_neighbor_HR=0.050, collision=3.5%")
    print(f"  C: OPQ 8x256 (64 bit)                     → semantic_neighbor_HR=0.033, collision=0.06%")

    print(f"\n{'New results:'}")
    for r in all_results:
        m = r['metrics']
        col = m.get('semantic_id_collision', '?')
        snhr = m.get('semantic_neighbor_hit_rate', '?')
        if isinstance(col, float):
            col = f"{col:.4f}"
        if isinstance(snhr, float):
            snhr = f"{snhr:.4f}"
        print(f"  {r['name']:35s} collision={col:>8s}  semantic_neighbor_HR={snhr:>8s}")


if __name__ == '__main__':
    main()
