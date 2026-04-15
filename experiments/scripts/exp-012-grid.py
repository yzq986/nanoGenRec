#!/usr/bin/env python3
"""EXP-012: Tokenizer Grid Search — KMeans cluster × FSQ type × OPQ control.

Grid: 4 cluster sizes (1024/2048/4096/8192) × 2 FSQ types (binary/multi) + 4 OPQ
Total: 12 configs, only 4 KMeans trainings (cached per cluster size).

Multi-GPU: different cluster-size groups run on different GPUs in parallel.
  --gpus 0,1,2,3   → 4 groups in parallel
  --gpus 0          → serial on GPU 0

Only 4 key metrics: collision, codebook_util, cluster_balance, semantic_neighbor_HR.
Merges existing EXP-011 results to avoid re-running completed configs.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


# ============================================================
# Grid configs
# ============================================================

FSQ_CONFIGS = [
    # (name, kmeans_clusters, fsq_levels_key, description)
    # 1024×3
    ('1024-multi',  1024, '5d_1024',  '1024×3 multi [4]×5'),
    ('1024-binary', 1024, '10d_1024', '1024×3 binary [2]×10'),
    # 2048×3
    ('2048-multi',  2048, '6d_2048',  '2048×3 mixed [4,4,4,4,4,2]'),
    ('2048-binary', 2048, '11d_2048', '2048×3 binary [2]×11'),
    # 4096×3
    ('4096-multi',  4096, '6d_4096',  '4096×3 multi [4]×6'),
    ('4096-binary', 4096, '12d_4096', '4096×3 binary [2]×12'),
    # 8192×3
    ('8192-multi',  8192, '7d_8192',  '8192×3 mixed [4,4,4,4,4,4,2]'),
    ('8192-binary', 8192, '13d_8192', '8192×3 binary [2]×13'),
]

OPQ_CONFIGS = [
    # (name, n_subvectors, n_clusters_per_sub, description)
    ('opq-3x1024', 3, 1024, 'OPQ 3×1024 (30 bit)'),
    ('opq-3x2048', 3, 2048, 'OPQ 3×2048 (33 bit)'),
    ('opq-3x4096', 3, 4096, 'OPQ 3×4096 (36 bit)'),
    ('opq-3x8192', 3, 8192, 'OPQ 3×8192 (39 bit)'),
]

# Key metrics only (skip reconstruction_loss, entropy, cosine_sim, effective_dim, embedding_HR)
KEY_METRICS_INTRINSIC = ['semantic_id_collision', 'codebook_utilization', 'cluster_balance']
KEY_METRICS_BEHAVIOR = ['semantic_neighbor_hit_rate']

FSQ_PROJECTION = 'mlp'
FSQ_MLP_HIDDEN = 64
FSQ_EPOCHS = 50
NITER = 25
NREDO = 3
NORMALIZE_RESIDUALS = True


def compute_total_bits(n_clusters, fsq_levels):
    km_bits = math.log2(n_clusters)
    fsq_bits = sum(math.log2(l) for l in fsq_levels)
    return int(2 * km_bits + fsq_bits)


# ============================================================
# Data loading (once, in main process)
# ============================================================

def load_data():
    """Load embeddings (filtered by exposure) + behavior data."""
    from gr_demo.config import EFS_EMBEDDING_CACHE
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
# KMeans training
# ============================================================

def train_kmeans_layers(embeddings: torch.Tensor, n_clusters: int, device: str):
    """Train 2-layer KMeans, return (kmeans_layers, residuals_after_L2)."""
    from gr_demo.model.rkmeans import FaissKMeansLayer

    n_samples = embeddings.shape[0]
    n_features = embeddings.shape[1]
    use_gpu = device.startswith('cuda')

    # Normalize input
    print(f"  [{device}] Normalizing embeddings...")
    normalized = []
    for i in range(0, n_samples, 100000):
        chunk = embeddings[i:i+100000].to(device)
        chunk = F.normalize(chunk, p=2, dim=1).cpu()
        normalized.append(chunk)
    current_residuals = torch.cat(normalized, dim=0)

    kmeans_layers = []
    for layer_idx in range(2):
        print(f"  [{device}] Training KMeans Layer {layer_idx+1}/2 (clusters={n_clusters})...")
        km = FaissKMeansLayer(n_clusters, n_features, gpu=use_gpu)
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
        print(f"  [{device}] Residual norm: {norm:.6f}")

    return kmeans_layers, current_residuals


# ============================================================
# FSQ evaluation (on cached residuals)
# ============================================================

def run_fsq_on_residuals(
    kmeans_layers, residuals, embeddings, content_ids,
    n_clusters, fsq_levels_key, behavior_data, device='cuda',
):
    """Train FSQ on pre-computed residuals, evaluate with key metrics only."""
    from gr_demo.model.fsq import FSQ_LEVEL_CONFIGS, LearnedFSQLayer, _codebook_size
    from gr_demo.model.rkmeans_fsq import ResKmeansFSQ, generate_semantic_ids_fsq
    from gr_demo.eval.wrapper import ResKmeansFSQModelWrapper
    from gr_demo.eval.behavior import BehaviorMetricsEvaluator

    fsq_levels = FSQ_LEVEL_CONFIGS[fsq_levels_key]
    n_features = embeddings.shape[1]

    print(f"    Training FSQ: {fsq_levels_key} {fsq_levels}...")
    t0 = time.time()

    fsq_layer = LearnedFSQLayer(
        fsq_levels, n_features,
        hidden_dim=FSQ_MLP_HIDDEN,
        epochs=FSQ_EPOCHS,
        device=device,
    )
    fsq_layer.train(residuals)
    fsq_time = time.time() - t0

    # Build full model for SID generation
    model = ResKmeansFSQ.__new__(ResKmeansFSQ)
    model.n_kmeans_clusters = n_clusters
    model.fsq_levels = fsq_levels
    model.n_features = n_features
    model.normalize_residuals = NORMALIZE_RESIDUALS
    model.n_layers = 3
    model.primary_device = device
    model.kmeans_layers = kmeans_layers
    model.fsq_layer = fsq_layer

    semantic_ids = generate_semantic_ids_fsq(model, embeddings, NORMALIZE_RESIDUALS)

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

    # Evaluate — key metrics only
    evaluator = BehaviorMetricsEvaluator(
        embeddings=embeddings, content_ids=content_ids,
        semantic_ids=semantic_ids, model=wrapper,
        behavior_data=behavior_data, device=device,
    )
    evaluator.register_metrics(KEY_METRICS_INTRINSIC)
    evaluator.register_metrics(KEY_METRICS_BEHAVIOR)
    metric_results = evaluator.evaluate()

    results = {}
    for mr in metric_results.values():
        results.update(mr.to_flat_dict())

    # Extract neighbor_coverage from snHR details
    if 'semantic_neighbor_hit_rate' in metric_results:
        snhr_details = metric_results['semantic_neighbor_hit_rate'].details
        results['neighbor_coverage'] = snhr_details.get('neighbor_coverage')
        results['n_contents_evaluated'] = snhr_details.get('n_contents_evaluated')

    results['quantizer_type'] = 'rkmeans_fsq'
    results['fsq_levels_key'] = fsq_levels_key
    results['fsq_codebook_size'] = _codebook_size(fsq_levels)
    results['total_bits'] = compute_total_bits(n_clusters, fsq_levels)

    return results, fsq_time


# ============================================================
# OPQ evaluation (inline, key metrics only)
# ============================================================

def run_opq_config(embeddings, content_ids, behavior_data, n_sub, n_cpersub, device='cuda'):
    """Run single OPQ experiment with key metrics only."""
    from gr_demo.model.opq import OPQQuantizer
    from gr_demo.eval.wrapper import OPQModelWrapper
    from gr_demo.eval.behavior import BehaviorMetricsEvaluator

    n_features = embeddings.shape[1]
    t0 = time.time()

    model = OPQQuantizer(
        n_features=n_features,
        n_subvectors=n_sub,
        n_clusters_per_sub=n_cpersub,
        normalize_input=NORMALIZE_RESIDUALS,
    )
    model.train(embeddings)
    train_time = time.time() - t0

    codes = model.encode(embeddings)
    semantic_ids = model.generate_semantic_ids(embeddings)

    model_data = {
        'model_type': 'opq',
        'n_features': n_features,
        'n_subvectors': n_sub,
        'n_clusters_per_sub': n_cpersub,
        'normalize_input': NORMALIZE_RESIDUALS,
        'rotation': model._rotation,
        'codebooks': model._codebooks,
    }
    model_wrapper = OPQModelWrapper(model_data, codes=codes, device=device)

    layer_assignments = [torch.tensor(codes[:, j], dtype=torch.long) for j in range(n_sub)]

    evaluator = BehaviorMetricsEvaluator(
        embeddings=embeddings, content_ids=content_ids,
        semantic_ids=semantic_ids, model=model_wrapper,
        behavior_data=behavior_data, device=device,
    )
    evaluator.layer_assignments = layer_assignments
    evaluator.register_metrics(KEY_METRICS_INTRINSIC)
    evaluator.register_metrics(KEY_METRICS_BEHAVIOR)
    metric_results = evaluator.evaluate()

    results = {}
    for mr in metric_results.values():
        results.update(mr.to_flat_dict())

    # Extract neighbor_coverage from snHR details
    if 'semantic_neighbor_hit_rate' in metric_results:
        snhr_details = metric_results['semantic_neighbor_hit_rate'].details
        results['neighbor_coverage'] = snhr_details.get('neighbor_coverage')
        results['n_contents_evaluated'] = snhr_details.get('n_contents_evaluated')

    results['quantizer_type'] = 'opq'
    results['n_subvectors'] = n_sub
    results['n_clusters_per_sub'] = n_cpersub
    results['total_bits'] = int(n_sub * math.log2(n_cpersub))

    return results, train_time


# ============================================================
# Merge existing EXP-011 results
# ============================================================

# Map EXP-011 result dir names → grid search config names
EXP011_NAME_MAP = {
    'exp011-1024x3-5d':         '1024-multi',
    'exp011-1024x3-10d-binary': '1024-binary',
    'exp011-4096x3-6d':         '4096-multi',
    'exp011-4096x3-12d-binary': '4096-binary',
    'exp011-opq-3x1024':        'opq-3x1024',
    'exp011-opq-3x4096':        'opq-3x4096',
}


def load_existing_results():
    """Scan EXP-011 result directories for completed configs."""
    existing = {}
    hyperparam_dir = os.path.join(REPO_ROOT, "experiments", "hyperparam")
    if not os.path.isdir(hyperparam_dir):
        return existing

    for dirname in os.listdir(hyperparam_dir):
        if 'exp011' not in dirname:
            continue
        results_path = os.path.join(hyperparam_dir, dirname, "results.json")
        if not os.path.exists(results_path):
            continue

        # Extract config name from dirname (e.g. "2026-04-15_exp011-1024x3-5d")
        parts = dirname.split('_', 1)
        if len(parts) < 2:
            continue
        exp011_name = parts[1]

        grid_name = EXP011_NAME_MAP.get(exp011_name)
        if not grid_name:
            continue

        try:
            with open(results_path) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                metrics = data[0].get('metrics', {})
                # Extract the 4 key metrics
                existing[grid_name] = {
                    'semantic_id_collision': metrics.get('semantic_id_collision'),
                    'codebook_utilization': metrics.get('codebook_utilization'),
                    'cluster_balance': metrics.get('cluster_balance'),
                    'semantic_neighbor_hit_rate': metrics.get('semantic_neighbor_hit_rate'),
                    'quantizer_type': metrics.get('quantizer_type', data[0].get('quantizer_type')),
                    'source': 'exp-011',
                }
                # Carry over FSQ/OPQ specific fields
                if 'fsq_levels_key' in metrics:
                    existing[grid_name]['fsq_levels_key'] = metrics['fsq_levels_key']
                    existing[grid_name]['fsq_codebook_size'] = metrics.get('fsq_codebook_size')
                if 'n_subvectors' in metrics:
                    existing[grid_name]['n_subvectors'] = metrics['n_subvectors']
                    existing[grid_name]['n_clusters_per_sub'] = metrics['n_clusters_per_sub']

                print(f"  Merged EXP-011 result: {exp011_name} → {grid_name}")
        except (json.JSONDecodeError, KeyError):
            continue

    return existing


# ============================================================
# GPU worker: run one KMeans group
# ============================================================

def run_fsq_group(cluster_size, fsq_configs, embeddings, content_ids, behavior_data, device):
    """Run a single KMeans group: train KMeans once, sweep FSQ variants."""
    from gr_demo.model.fsq import FSQ_LEVEL_CONFIGS

    print(f"\n{'#' * 60}")
    print(f"# [{device}] KMeans cluster={cluster_size} — {len(fsq_configs)} FSQ variants")
    print(f"{'#' * 60}")

    t0 = time.time()
    kmeans_layers, residuals = train_kmeans_layers(embeddings, cluster_size, device)
    kmeans_time = time.time() - t0
    print(f"  [{device}] KMeans trained in {kmeans_time:.1f}s")

    group_results = []
    for name, fsq_key, desc in fsq_configs:
        print(f"\n  [{device}] >>> {name}: {desc}")
        metrics, fsq_time = run_fsq_on_residuals(
            kmeans_layers, residuals, embeddings, content_ids,
            cluster_size, fsq_key, behavior_data, device,
        )

        col = metrics.get('semantic_id_collision', 'N/A')
        snhr = metrics.get('semantic_neighbor_hit_rate', 'N/A')
        print(f"    [{device}] collision={col}  snHR={snhr}  fsq={fsq_time:.0f}s")

        group_results.append({
            'name': name,
            'description': desc,
            'quantizer_type': 'rkmeans_fsq',
            'num_clusters': cluster_size,
            'fsq_levels_key': fsq_key,
            'fsq_levels': FSQ_LEVEL_CONFIGS[fsq_key],
            'total_bits': metrics.get('total_bits'),
            'metrics': metrics,
            'kmeans_time': round(kmeans_time, 1),
            'fsq_time': round(fsq_time, 1),
        })

    return group_results


def run_opq_group(opq_configs, embeddings, content_ids, behavior_data, device):
    """Run OPQ configs on a given device."""
    results = []
    for name, n_sub, n_cpersub, desc in opq_configs:
        print(f"\n  [{device}] >>> {name}: {desc}")
        metrics, train_time = run_opq_config(
            embeddings, content_ids, behavior_data,
            n_sub, n_cpersub, device,
        )

        col = metrics.get('semantic_id_collision', 'N/A')
        snhr = metrics.get('semantic_neighbor_hit_rate', 'N/A')
        print(f"    [{device}] collision={col}  snHR={snhr}  time={train_time:.0f}s")

        results.append({
            'name': name,
            'description': desc,
            'quantizer_type': 'opq',
            'n_subvectors': n_sub,
            'n_clusters_per_sub': n_cpersub,
            'total_bits': metrics.get('total_bits'),
            'metrics': metrics,
            'train_time': round(train_time, 1),
        })

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='EXP-012: Tokenizer Grid Search')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='Comma-separated GPU IDs (e.g. "0,1,2,3,4,5,6,7"). Default: "0,1,2,3"')
    parser.add_argument('--skip_opq', action='store_true',
                        help='Skip OPQ control experiments')
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    n_gpus = len(gpu_ids)

    print("=" * 60)
    print("EXP-012: Tokenizer Grid Search")
    print(f"  GPUs: {gpu_ids}")
    print(f"  FSQ configs: {len(FSQ_CONFIGS)}")
    print(f"  OPQ configs: {len(OPQ_CONFIGS)} {'(skipped)' if args.skip_opq else ''}")
    print(f"  Metrics: {KEY_METRICS_INTRINSIC + KEY_METRICS_BEHAVIOR}")
    print("=" * 60)

    # ── Step 0: Output directory ──
    exp_dir = os.path.join(REPO_ROOT, "experiments", "hyperparam",
                           f"{date.today().isoformat()}_exp012-grid-search")
    os.makedirs(exp_dir, exist_ok=True)
    json_path = os.path.join(exp_dir, "results.json")

    # ── Step 1: Merge existing EXP-011 results ──
    print("\nChecking for existing EXP-011 results...")
    existing = load_existing_results()

    # ── Step 2: Load data once ──
    t_start = time.time()
    embeddings, content_ids, behavior_data = load_data()
    print(f"\nData loaded in {time.time()-t_start:.1f}s\n")

    all_results = {}  # name → result dict

    # Add existing results
    for name, metrics in existing.items():
        all_results[name] = {
            'name': name,
            'metrics': metrics,
            'source': 'exp-011 (merged)',
        }

    def save_results():
        out = sorted(all_results.values(), key=lambda r: r['name'])
        with open(json_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)

    # ── Step 3: FSQ grid — group by cluster size ──
    groups = defaultdict(list)
    for name, clusters, fsq_key, desc in FSQ_CONFIGS:
        if name in existing:
            print(f"  Skipping {name} (already have EXP-011 result)")
            continue
        groups[clusters].append((name, fsq_key, desc))

    if groups:
        cluster_sizes = sorted(groups.keys())
        # Assign GPUs round-robin to groups
        gpu_assignment = {cs: f'cuda:{gpu_ids[i % n_gpus]}' for i, cs in enumerate(cluster_sizes)}

        print(f"\nFSQ groups to run: {len(cluster_sizes)}")
        for cs in cluster_sizes:
            print(f"  cluster={cs} → {gpu_assignment[cs]} ({len(groups[cs])} configs)")

        if n_gpus > 1 and len(cluster_sizes) > 1:
            # Multi-GPU parallel execution
            print(f"\nRunning {len(cluster_sizes)} KMeans groups in parallel on {n_gpus} GPUs...")

            # Group cluster sizes by GPU to avoid CUDA conflicts
            gpu_to_clusters = defaultdict(list)
            for cs in cluster_sizes:
                gpu_to_clusters[gpu_assignment[cs]].append(cs)

            with ProcessPoolExecutor(max_workers=n_gpus) as executor:
                futures = {}
                for device, cs_list in gpu_to_clusters.items():
                    for cs in cs_list:
                        future = executor.submit(
                            run_fsq_group, cs, groups[cs],
                            embeddings, content_ids, behavior_data, device,
                        )
                        futures[future] = cs

                for future in as_completed(futures):
                    cs = futures[future]
                    try:
                        group_results = future.result()
                        for r in group_results:
                            all_results[r['name']] = r
                        save_results()
                        print(f"\n  Completed KMeans group {cs} ({len(group_results)} configs)")
                    except Exception as e:
                        print(f"\n  ERROR in KMeans group {cs}: {e}")
                        import traceback
                        traceback.print_exc()
        else:
            # Single GPU serial execution
            device = f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
            for cs in cluster_sizes:
                group_results = run_fsq_group(
                    cs, groups[cs], embeddings, content_ids, behavior_data, device,
                )
                for r in group_results:
                    all_results[r['name']] = r
                save_results()

    # ── Step 4: OPQ configs ──
    if not args.skip_opq:
        pending_opq = [(n, ns, nc, d) for n, ns, nc, d in OPQ_CONFIGS if n not in existing]
        if pending_opq:
            print(f"\n{'#' * 60}")
            print(f"# OPQ controls: {len(pending_opq)} configs")
            print(f"{'#' * 60}")

            if n_gpus > 1 and len(pending_opq) > 1:
                # Distribute OPQ configs across GPUs
                with ProcessPoolExecutor(max_workers=min(n_gpus, len(pending_opq))) as executor:
                    futures = {}
                    for i, (name, n_sub, n_cpersub, desc) in enumerate(pending_opq):
                        device = f'cuda:{gpu_ids[i % n_gpus]}'
                        future = executor.submit(
                            run_opq_group, [(name, n_sub, n_cpersub, desc)],
                            embeddings, content_ids, behavior_data, device,
                        )
                        futures[future] = name

                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            opq_results = future.result()
                            for r in opq_results:
                                all_results[r['name']] = r
                            save_results()
                        except Exception as e:
                            print(f"\n  ERROR in OPQ {name}: {e}")
                            import traceback
                            traceback.print_exc()
            else:
                device = f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
                opq_results = run_opq_group(pending_opq, embeddings, content_ids, behavior_data, device)
                for r in opq_results:
                    all_results[r['name']] = r
                save_results()

    # ── Summary ──
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"EXP-012 Complete — {len(all_results)} configs in {total_time:.0f}s")
    print(f"Results: {json_path}")
    print(f"{'=' * 60}")

    # Print comparison table sorted by semantic_neighbor_HR
    rows = []
    for r in all_results.values():
        m = r.get('metrics', r)  # merged results have metrics at top level
        col = m.get('semantic_id_collision')
        snhr = m.get('semantic_neighbor_hit_rate')
        gini = m.get('cluster_balance')
        ncov = m.get('neighbor_coverage')
        bits = r.get('total_bits') or m.get('total_bits', '?')
        source = r.get('source', 'new')
        rows.append((r['name'], bits, col, snhr, ncov, gini, source))

    rows.sort(key=lambda x: -(x[3] or 0))

    print(f"\n{'Name':25s} {'Bits':>5s} {'Collision':>10s} {'snHR':>8s} {'Coverage':>8s} {'Gini':>8s} {'Source':>12s}")
    print("-" * 85)
    for name, bits, col, snhr, ncov, gini, source in rows:
        def fmt(v, f='.4f'):
            return f'{v:{f}}' if isinstance(v, (int, float)) else str(v or '?')
        print(f"{name:25s} {str(bits):>5s} {fmt(col):>10s} {fmt(snhr):>8s} {fmt(ncov):>8s} {fmt(gini):>8s} {source:>12s}")


if __name__ == '__main__':
    main()
