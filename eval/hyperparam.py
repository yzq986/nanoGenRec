"""超参数网格搜索。"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import date, datetime
from typing import Dict, List, Tuple

import numpy as np
import torch

from gr_demo.config import MODEL_CONFIGS, EFS_EMBEDDING_CACHE
from gr_demo.model.rkmeans import ResidualQuantizationMultiGPU
from gr_demo.model.semantic_ids import generate_semantic_ids
from gr_demo.model.fsq import FSQ_LEVEL_CONFIGS
from gr_demo.model.rkmeans_fsq import ResKmeansFSQ, generate_semantic_ids_fsq
from gr_demo.model.opq import OPQQuantizer
from gr_demo.data.loaders import load_exposed_iids
from gr_demo.eval.wrapper import RKMeansModelWrapper, ResKmeansFSQModelWrapper, OPQModelWrapper
from gr_demo.eval.evaluator import MetricsEvaluator
from gr_demo.eval.behavior import BehaviorMetricsEvaluator

from gr_demo.metrics import INTRINSIC_METRICS, BEHAVIOR_METRICS, MetricResult


# ============================================================
# 搜索空间配置
# ============================================================

GRID = {
    'num_clusters': [64, 128, 256, 512, 1024, 2048, 4096],
    'niter':        [25],
    'nredo':        [3],
}

# 固定参数
NUM_LAYERS = 3
NORMALIZE_RESIDUALS = True


# ============================================================
# 单次实验
# ============================================================

def run_single_experiment(
    train_embeddings: torch.Tensor,
    eval_embeddings: torch.Tensor,
    content_ids: np.ndarray,
    n_features: int,
    num_clusters: int,
    niter: int,
    nredo: int,
    num_gpus: int,
    behavior_data: Dict = None,
    device: str = 'cuda',
    recall_beam_size: int = 50,
    eval_sample_size: int = 50000,
    only_sid: bool = False,
    run_ntp: bool = False,
) -> Tuple[Dict[str, float], float]:
    """运行单次 RKMeans 训练 + eval，返回 (metrics_dict, train_seconds)"""
    from gr_demo.eval.behavior import BehaviorMetricsEvaluator

    t0 = time.time()

    # 训练
    model = ResidualQuantizationMultiGPU(
        n_layers=NUM_LAYERS,
        n_clusters=num_clusters,
        n_features=n_features,
        normalize_residuals=NORMALIZE_RESIDUALS,
        num_gpus=num_gpus,
    )
    model.train(train_embeddings, niter=niter, nredo=nredo)

    train_time = time.time() - t0

    # 生成 SID (在 eval 集上)
    semantic_ids = generate_semantic_ids(model, eval_embeddings, NORMALIZE_RESIDUALS)

    # 构造 model wrapper for metrics that need model
    model_data = {
        'centroids_list': model.get_centroids_list(),
        'normalize_residuals': NORMALIZE_RESIDUALS,
        'n_layers': model.n_layers,
        'n_clusters': model.n_clusters,
        'n_features': model.n_features,
    }
    model_wrapper = RKMeansModelWrapper(model_data, device=device)

    # 选择 evaluator: 有 behavior_data 时用 BehaviorMetricsEvaluator
    if behavior_data is not None:
        evaluator = BehaviorMetricsEvaluator(
            embeddings=eval_embeddings,
            content_ids=content_ids,
            semantic_ids=semantic_ids,
            model=model_wrapper,
            behavior_data=behavior_data,
            device=device,
        )
        if not only_sid:
            evaluator.register_metrics(list(INTRINSIC_METRICS.keys()))
        evaluator.register_metrics(['embedding_hit_rate'])
        evaluator.register_metrics(['semantic_neighbor_hit_rate'])
        sid_kwargs = {}
        if run_ntp:
            evaluator.register_metrics(['semantic_id_prediction'])
            sid_kwargs['semantic_id_prediction'] = {
                'device': device,
                'recall_beam_size': recall_beam_size,
                'eval_sample_size': eval_sample_size,
            }
        metric_results = evaluator.evaluate(metric_kwargs=sid_kwargs)
    else:
        evaluator = MetricsEvaluator(
            embeddings=eval_embeddings,
            model=model_wrapper,
            semantic_ids=semantic_ids,
            device=device,
        )
        evaluator.register_metrics(list(INTRINSIC_METRICS.keys()))
        metric_results = evaluator.evaluate()

    # Flatten all MetricResult into a single dict for grid-search storage
    results = {}
    for mr in metric_results.values():
        results.update(mr.to_flat_dict())

    # Backward-compatible keys used by report generation
    if 'codebook_utilization' in metric_results:
        cb = metric_results['codebook_utilization']
        results['space_utilization'] = cb.details.get('space_utilization', 0)
        results['depth_codebook_util'] = cb.layer_values
        depth_stats = cb.details.get('depth_stats', [])
        results['depth_unique_prefixes'] = [d.get('n_unique', 0) for d in depth_stats]

    if 'semantic_id_collision' in metric_results:
        col = metric_results['semantic_id_collision']
        prefix_stats = col.details.get('prefix_stats', [])
        results['prefix_avg_items'] = [s.get('avg_items', 0) for s in prefix_stats]

    # SID prediction flat keys
    if 'semantic_id_prediction' in metric_results:
        sid = metric_results['semantic_id_prediction']
        results['ntp_perplexity'] = round(sid.value, 4)
        results['ntp_depth_acc'] = sid.layer_values
        results['ntp_depth_hit@10'] = sid.details.get('depth_hit@10')
        # Item recall (the key non-monotonic metric)
        for k in (10, 50, 100, 500):
            results[f'item_recall@{k}'] = sid.details.get(f'item_recall@{k}')

    return results, train_time


def run_single_experiment_fsq(
    train_embeddings: torch.Tensor,
    eval_embeddings: torch.Tensor,
    content_ids: np.ndarray,
    n_features: int,
    num_clusters: int,
    fsq_levels: list,
    fsq_levels_key: str,
    niter: int,
    nredo: int,
    num_gpus: int,
    behavior_data: Dict = None,
    device: str = 'cuda',
    recall_beam_size: int = 50,
    eval_sample_size: int = 50000,
    only_sid: bool = False,
    run_ntp: bool = False,
    fsq_projection: str = 'pca',
    fsq_mlp_hidden: int = 128,
    fsq_epochs: int = 50,
) -> Tuple[Dict[str, float], float]:
    """Run single ResKmeansFSQ train + eval, return (metrics_dict, train_seconds)"""
    from gr_demo.eval.behavior import BehaviorMetricsEvaluator

    t0 = time.time()

    model = ResKmeansFSQ(
        n_kmeans_clusters=num_clusters,
        fsq_levels=fsq_levels,
        n_features=n_features,
        normalize_residuals=NORMALIZE_RESIDUALS,
        num_gpus=num_gpus,
        fsq_projection=fsq_projection,
        fsq_mlp_hidden=fsq_mlp_hidden,
        fsq_epochs=fsq_epochs,
    )
    model.train(train_embeddings, niter=niter, nredo=nredo)

    train_time = time.time() - t0

    semantic_ids = generate_semantic_ids_fsq(model, eval_embeddings, NORMALIZE_RESIDUALS)

    # Build model wrapper
    model_data = {
        'model_type': 'rkmeans_fsq',
        'centroids_list': [km.get_centroids().cpu() for km in model.kmeans_layers],
        'fsq_state': model.fsq_layer.save_state(),
        'normalize_residuals': NORMALIZE_RESIDUALS,
        'n_layers': model.n_layers,
        'n_kmeans_clusters': num_clusters,
        'n_features': n_features,
        'fsq_levels': fsq_levels,
    }
    model_wrapper = ResKmeansFSQModelWrapper(model_data, device=device)

    # Evaluate
    if behavior_data is not None:
        evaluator = BehaviorMetricsEvaluator(
            embeddings=eval_embeddings,
            content_ids=content_ids,
            semantic_ids=semantic_ids,
            model=model_wrapper,
            behavior_data=behavior_data,
            device=device,
        )
        if not only_sid:
            evaluator.register_metrics(list(INTRINSIC_METRICS.keys()))
        evaluator.register_metrics(['embedding_hit_rate'])
        evaluator.register_metrics(['semantic_neighbor_hit_rate'])
        sid_kwargs = {}
        if run_ntp:
            evaluator.register_metrics(['semantic_id_prediction'])
            sid_kwargs['semantic_id_prediction'] = {
                'device': device,
                'recall_beam_size': recall_beam_size,
                'eval_sample_size': eval_sample_size,
            }
        metric_results = evaluator.evaluate(metric_kwargs=sid_kwargs)
    else:
        evaluator = MetricsEvaluator(
            embeddings=eval_embeddings,
            model=model_wrapper,
            semantic_ids=semantic_ids,
            device=device,
        )
        evaluator.register_metrics(list(INTRINSIC_METRICS.keys()))
        metric_results = evaluator.evaluate()

    # Flatten results (same as run_single_experiment)
    results = {}
    for mr in metric_results.values():
        results.update(mr.to_flat_dict())

    if 'codebook_utilization' in metric_results:
        cb = metric_results['codebook_utilization']
        results['space_utilization'] = cb.details.get('space_utilization', 0)
        results['depth_codebook_util'] = cb.layer_values
        depth_stats = cb.details.get('depth_stats', [])
        results['depth_unique_prefixes'] = [d.get('n_unique', 0) for d in depth_stats]

    if 'semantic_id_collision' in metric_results:
        col = metric_results['semantic_id_collision']
        prefix_stats = col.details.get('prefix_stats', [])
        results['prefix_avg_items'] = [s.get('avg_items', 0) for s in prefix_stats]

    if 'semantic_id_prediction' in metric_results:
        sid = metric_results['semantic_id_prediction']
        results['ntp_perplexity'] = round(sid.value, 4)
        results['ntp_depth_acc'] = sid.layer_values
        results['ntp_depth_hit@10'] = sid.details.get('depth_hit@10')
        for k in (10, 50, 100, 500):
            results[f'item_recall@{k}'] = sid.details.get(f'item_recall@{k}')

    # Add FSQ-specific fields
    results['quantizer_type'] = 'rkmeans_fsq'
    results['fsq_levels_key'] = fsq_levels_key
    from gr_demo.model.fsq import _codebook_size
    results['fsq_codebook_size'] = _codebook_size(fsq_levels)
    results['fsq_projection'] = fsq_projection
    results['fsq_mlp_hidden'] = fsq_mlp_hidden
    results['fsq_epochs'] = fsq_epochs

    return results, train_time


def run_single_experiment_opq(
    train_embeddings: torch.Tensor,
    eval_embeddings: torch.Tensor,
    content_ids: np.ndarray,
    n_features: int,
    n_subvectors: int,
    n_clusters_per_sub: int,
    behavior_data: Dict = None,
    device: str = 'cuda',
    recall_beam_size: int = 50,
    eval_sample_size: int = 50000,
    only_sid: bool = False,
    run_ntp: bool = False,
    force_autoregressive: bool = False,
) -> Tuple[Dict[str, float], float]:
    """Run single OPQ train + eval, return (metrics_dict, train_seconds)"""

    t0 = time.time()

    model = OPQQuantizer(
        n_features=n_features,
        n_subvectors=n_subvectors,
        n_clusters_per_sub=n_clusters_per_sub,
        normalize_input=NORMALIZE_RESIDUALS,
    )
    model.train(train_embeddings)

    train_time = time.time() - t0

    # Encode + generate SIDs
    codes = model.encode(eval_embeddings)  # (N, m)
    semantic_ids = model.generate_semantic_ids(eval_embeddings)

    # Build model wrapper (with pre-computed codes)
    model_data = {
        'model_type': 'opq',
        'n_features': n_features,
        'n_subvectors': n_subvectors,
        'n_clusters_per_sub': n_clusters_per_sub,
        'normalize_input': NORMALIZE_RESIDUALS,
        'rotation': model._rotation,
        'codebooks': model._codebooks,
    }
    model_wrapper = OPQModelWrapper(model_data, codes=codes, device=device)

    # Pre-compute layer_assignments from OPQ codes to skip evaluator's
    # _precompute_assignments() which does residual subtraction (wrong for OPQ)
    layer_assignments = [torch.tensor(codes[:, j], dtype=torch.long) for j in range(n_subvectors)]

    # Evaluate
    if behavior_data is not None:
        evaluator = BehaviorMetricsEvaluator(
            embeddings=eval_embeddings,
            content_ids=content_ids,
            semantic_ids=semantic_ids,
            model=model_wrapper,
            behavior_data=behavior_data,
            device=device,
        )
        evaluator.layer_assignments = layer_assignments
        if not only_sid:
            evaluator.register_metrics(list(INTRINSIC_METRICS.keys()))
        evaluator.register_metrics(['embedding_hit_rate'])
        evaluator.register_metrics(['semantic_neighbor_hit_rate'])
        sid_kwargs = {}
        if run_ntp:
            evaluator.register_metrics(['semantic_id_prediction'])
            sid_kwargs['semantic_id_prediction'] = {
                'device': device,
                'recall_beam_size': recall_beam_size,
                'eval_sample_size': eval_sample_size,
                'force_autoregressive': force_autoregressive,
            }
        metric_results = evaluator.evaluate(metric_kwargs=sid_kwargs)
    else:
        evaluator = MetricsEvaluator(
            embeddings=eval_embeddings,
            model=model_wrapper,
            semantic_ids=semantic_ids,
            device=device,
        )
        evaluator.layer_assignments = layer_assignments  # OPQ: skip residual subtraction
        evaluator.register_metrics(list(INTRINSIC_METRICS.keys()))
        metric_results = evaluator.evaluate()

    # Flatten results
    results = {}
    for mr in metric_results.values():
        results.update(mr.to_flat_dict())

    if 'codebook_utilization' in metric_results:
        cb = metric_results['codebook_utilization']
        results['space_utilization'] = cb.details.get('space_utilization', 0)
        results['depth_codebook_util'] = cb.layer_values
        depth_stats = cb.details.get('depth_stats', [])
        results['depth_unique_prefixes'] = [d.get('n_unique', 0) for d in depth_stats]

    if 'semantic_id_collision' in metric_results:
        col = metric_results['semantic_id_collision']
        prefix_stats = col.details.get('prefix_stats', [])
        results['prefix_avg_items'] = [s.get('avg_items', 0) for s in prefix_stats]

    if 'semantic_id_prediction' in metric_results:
        sid = metric_results['semantic_id_prediction']
        results['ntp_perplexity'] = round(sid.value, 4)
        results['ntp_depth_acc'] = sid.layer_values
        results['ntp_depth_hit@10'] = sid.details.get('depth_hit@10')
        for k in (10, 50, 100, 500):
            results[f'item_recall@{k}'] = sid.details.get(f'item_recall@{k}')

    # OPQ-specific fields
    results['quantizer_type'] = 'opq'
    results['n_subvectors'] = n_subvectors
    results['n_clusters_per_sub'] = n_clusters_per_sub

    return results, train_time


# ============================================================
# 加载数据 (复用 main 的逻辑)
# ============================================================

def load_data(args) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """加载 embedding 并按曝光过滤，返回 (train_tensor, eval_tensor, content_ids)"""

    model_key = args.model
    _, embedding_dim, _, _ = MODEL_CONFIGS[model_key]

    embedding_cache_dir = f'{EFS_EMBEDDING_CACHE}/{model_key}'
    embedding_cache_path = f'{embedding_cache_dir}/embeddings.npy'
    content_ids_cache_path = f'{embedding_cache_dir}/content_ids.npy'

    if args.skip_embedding:
        print("Loading embeddings from cache...")

        # 优先读 incremental_cache.npy (encode_multiprocess.py 产出的 dict 格式)
        incremental_cache_path = f'{embedding_cache_dir}/incremental_cache.npy'
        if os.path.exists(incremental_cache_path):
            cache_dict = np.load(incremental_cache_path, allow_pickle=True).item()
            content_ids = np.array(list(cache_dict.keys()))
            embeddings = np.array(list(cache_dict.values()), dtype=np.float32)
            print(f"Loaded {len(content_ids):,} embeddings from incremental_cache, dim={embeddings.shape[1]}")
        elif os.path.exists(embedding_cache_path) and os.path.exists(content_ids_cache_path):
            embeddings = np.load(embedding_cache_path, allow_pickle=True)
            content_ids = np.load(content_ids_cache_path, allow_pickle=True)
            print(f"Loaded {len(content_ids):,} embeddings from cache, dim={embeddings.shape[1]}")
        else:
            raise FileNotFoundError(f"Cache not found at {embedding_cache_dir}. Run encode_multiprocess.py first.")
    else:
        raise ValueError("Grid search requires --skip_embedding (embeddings must be pre-cached)")

    # 曝光过滤
    if args.behavior_path:
        print("Filtering by exposed IIDs...")
        exposed_iids = load_exposed_iids(args.behavior_path)
        cid_str = np.array([str(cid) for cid in content_ids])
        mask = np.isin(cid_str, list(exposed_iids))
        train_embeddings = embeddings[mask]
        content_ids = content_ids[mask]
        print(f"Exposed items: {len(train_embeddings):,} / {len(cid_str):,}")
    else:
        train_embeddings = embeddings

    train_tensor = torch.tensor(train_embeddings, dtype=torch.float32)
    # eval 在训练集上跑 (和 --eval_intrinsic 一致)
    eval_tensor = train_tensor

    return train_tensor, eval_tensor, content_ids


# ============================================================
# Markdown 报告生成
# ============================================================

METRIC_ORDER = [
    'semantic_id_collision',
    'entropy',
    'cluster_balance',
    'reconstruction_loss',
    'cosine_similarity',
    'effective_dimension',
]

METRIC_SHORT = {
    'semantic_id_collision': 'collision',
    'entropy': 'entropy',
    'cluster_balance': 'balance(Gini)',
    'reconstruction_loss': 'recon_loss',
    'cosine_similarity': 'cos_sim',
    'effective_dimension': 'eff_dim',
}


def generate_report(all_results: List[dict], output_path: str):
    """生成 Markdown 报告"""

    # Split OPQ results from RKMeans/FSQ results
    opq_results = [r for r in all_results if r.get('quantizer_type') == 'opq']
    rk_results = [r for r in all_results if r.get('quantizer_type') != 'opq']

    lines = []

    # OPQ report
    if opq_results:
        _generate_opq_report(lines, opq_results)

    # RKMeans/FSQ report
    if rk_results:
        if opq_results:
            lines.append("")
            lines.append("---")
            lines.append("")
        _generate_rkmeans_report(lines, rk_results)

    # 写入文件
    report = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")

    return report


def _generate_opq_report(lines: List[str], all_results: List[dict]):
    """Generate report section for OPQ results."""
    import math

    has_ntp = any('ntp_perplexity' in r['metrics'] for r in all_results)

    lines.append("# OPQ 超参数搜索结果")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**模型**: qwen3-0.6b (1024d)")
    lines.append(f"**量化器**: OPQ (Optimized Product Quantization)")
    lines.append(f"**实验数量**: {len(all_results)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 完整结果")
    lines.append("")

    if has_ntp:
        header = "| # | m (tokens) | M (vocab) | bits | collision | recon_loss | ntp_ppl | **recall@50** | recall@100 | recall@500 | time(s) |"
        sep    = "|---|-----------|-----------|------|-----------|------------|---------|---------------|------------|------------|---------|"
    else:
        header = "| # | m (tokens) | M (vocab) | bits | collision | entropy | Gini | recon_loss | time(s) |"
        sep    = "|---|-----------|-----------|------|-----------|---------|------|------------|---------|"
    lines.append(header)
    lines.append(sep)

    if has_ntp:
        sorted_results = sorted(all_results, key=lambda r: r['metrics'].get('item_recall@50') or 0, reverse=True)
    else:
        sorted_results = sorted(all_results, key=lambda r: r.get('n_subvectors', 0))

    for i, r in enumerate(sorted_results, 1):
        met = r['metrics']
        n_sub = r.get('n_subvectors', '?')
        n_cpersub = r.get('n_clusters_per_sub', 256)
        bits = int(n_sub * math.log2(n_cpersub)) if isinstance(n_sub, int) else '?'
        col = f"{met.get('semantic_id_collision', 0):.4f}"
        rec = f"{met.get('reconstruction_loss', 0):.4f}"

        if has_ntp:
            ntp_ppl = met.get('ntp_perplexity')
            ntp_str = f"{ntp_ppl:.2f}" if ntp_ppl is not None else "N/A"
            r50 = met.get('item_recall@50')
            r100 = met.get('item_recall@100')
            r500 = met.get('item_recall@500')
            r50_str = f"{r50:.4f}" if r50 is not None else "N/A"
            r100_str = f"{r100:.4f}" if r100 is not None else "N/A"
            r500_str = f"{r500:.4f}" if r500 is not None else "N/A"
            row = f"| {i} | {n_sub} | {n_cpersub} | {bits} | {col} | {rec} | {ntp_str} | **{r50_str}** | {r100_str} | {r500_str} | {r['train_time']:.0f} |"
        else:
            ent = f"{met.get('entropy', 0):.4f}"
            gini = f"{met.get('cluster_balance', 0):.4f}"
            row = f"| {i} | {n_sub} | {n_cpersub} | {bits} | {col} | {ent} | {gini} | {rec} | {r['train_time']:.0f} |"
        lines.append(row)

    lines.append("")

    # NTP detail section
    if has_ntp:
        lines.append("## Per-Digit Accuracy")
        lines.append("")
        for i, r in enumerate(sorted_results, 1):
            met = r['metrics']
            n_sub = r.get('n_subvectors', '?')
            h10 = met.get('ntp_depth_hit@10', [])
            if h10:
                h10_str = ', '.join(f'd{j+1}={v:.4f}' for j, v in enumerate(h10[:8]))
                avg_h10 = sum(h10) / len(h10) if h10 else 0
                lines.append(f"**m={n_sub}**: avg={avg_h10:.4f} ({h10_str})")
            else:
                lines.append(f"**m={n_sub}**: N/A")
        lines.append("")


def _generate_rkmeans_report(lines: List[str], all_results: List[dict]):
    """Generate report section for RKMeans/FSQ results (original logic)."""
    has_fsq = any(r.get('quantizer_type') == 'rkmeans_fsq' for r in all_results)
    title = "RKMeans + FSQ 超参数网格搜索结果" if has_fsq else "RKMeans 超参数网格搜索结果"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**模型**: qwen3-0.6b (1024d)")
    lines.append(f"**固定参数**: {NUM_LAYERS} layers, normalize_residuals=True")
    if has_fsq:
        fsq_keys = sorted(set(r.get('fsq_levels_key', '') for r in all_results if r.get('quantizer_type') == 'rkmeans_fsq'))
        lines.append(f"**量化器**: rkmeans_fsq (2 KMeans + 1 FSQ)")
        lines.append(f"**FSQ configs**: {', '.join(fsq_keys)}")
    lines.append(f"**实验数量**: {len(all_results)}")
    lines.append("")

    # ---- 1. 完整结果表 ----
    lines.append("---")
    lines.append("")
    lines.append("## 1. 完整结果")
    lines.append("")

    # 检测是否有 NTP 结果
    has_ntp = any('ntp_perplexity' in r['metrics'] for r in all_results)

    if has_ntp:
        if has_fsq:
            header = "| # | clusters | L3 | niter | nredo | collision | recon_loss | ntp_ppl | **recall@50** | recall@100 | recall@500 | d1_h@10 | d2_h@10 | d3_h@10 | time(s) |"
            sep    = "|---|----------|----|-------|-------|-----------|------------|---------|---------------|------------|------------|---------|---------|---------|---------|"
        else:
            header = "| # | clusters | niter | nredo | collision | recon_loss | ntp_ppl | **recall@50** | recall@100 | recall@500 | d1_h@10 | d2_h@10 | d3_h@10 | time(s) |"
            sep    = "|---|----------|-------|-------|-----------|------------|---------|---------------|------------|------------|---------|---------|---------|---------|"
    else:
        if has_fsq:
            header = "| # | clusters | L3 | niter | nredo | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |"
            sep    = "|---|----------|----|-------|-------|-----------|----------|------------|--------|----------|----------|---------|"
        else:
            header = "| # | clusters | niter | nredo | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |"
            sep    = "|---|----------|-------|-------|-----------|----------|------------|--------|----------|----------|---------|"
    lines.append(header)
    lines.append(sep)

    # 按 item_recall@50 降序 (有 NTP 时)，否则按 collision 升序
    if has_ntp:
        sorted_results = sorted(all_results, key=lambda r: r['metrics'].get('item_recall@50') or 0, reverse=True)
    else:
        sorted_results = sorted(all_results, key=lambda r: r['metrics'].get('semantic_id_collision', 999))

    for i, r in enumerate(sorted_results, 1):
        m = r['metrics']
        col = f"{m.get('semantic_id_collision', 0):.4f}"
        rec = f"{m.get('reconstruction_loss', 0):.4f}"
        l3_label = r.get('fsq_levels_key', 'KM') if has_fsq else None

        if has_ntp:
            ntp_ppl = m.get('ntp_perplexity')
            ntp_str = f"{ntp_ppl:.2f}" if ntp_ppl is not None else "N/A"

            r50 = m.get('item_recall@50')
            r100 = m.get('item_recall@100')
            r500 = m.get('item_recall@500')
            r50_str = f"{r50:.4f}" if r50 is not None else "N/A"
            r100_str = f"{r100:.4f}" if r100 is not None else "N/A"
            r500_str = f"{r500:.4f}" if r500 is not None else "N/A"

            h10 = m.get('ntp_depth_hit@10', [])
            h10_str = [f"{v:.4f}" if v is not None else "N/A" for v in (h10[:3] if h10 else [])]
            while len(h10_str) < 3:
                h10_str.append("N/A")

            if has_fsq:
                row = f"| {i} | {r['num_clusters']} | {l3_label} | {r['niter']} | {r['nredo']} | {col} | {rec} | {ntp_str} | **{r50_str}** | {r100_str} | {r500_str} | {h10_str[0]} | {h10_str[1]} | {h10_str[2]} | {r['train_time']:.0f} |"
            else:
                row = f"| {i} | {r['num_clusters']} | {r['niter']} | {r['nredo']} | {col} | {rec} | {ntp_str} | **{r50_str}** | {r100_str} | {r500_str} | {h10_str[0]} | {h10_str[1]} | {h10_str[2]} | {r['train_time']:.0f} |"
        else:
            sp_util = m.get('space_utilization')
            sp_str = f"{sp_util:.2e}" if sp_util is not None else "N/A"
            pf = m.get('prefix_avg_items', [])
            if pf and len(pf) >= 3:
                pf_str = [f"{v:.1f}" for v in pf[:3]]
            else:
                pf_str = ["N/A", "N/A", "N/A"]
            if has_fsq:
                row = f"| {i} | {r['num_clusters']} | {l3_label} | {r['niter']} | {r['nredo']} | {col} | {sp_str} | {rec} | {pf_str[0]} | {pf_str[1]} | {pf_str[2]} | {r['train_time']:.0f} |"
            else:
                row = f"| {i} | {r['num_clusters']} | {r['niter']} | {r['nredo']} | {col} | {sp_str} | {rec} | {pf_str[0]} | {pf_str[1]} | {pf_str[2]} | {r['train_time']:.0f} |"

        lines.append(row)

    lines.append("")

    # ---- 2. 各维度单独对比 ----
    unique_clusters = sorted(set(r['num_clusters'] for r in all_results))
    unique_niters = sorted(set(r['niter'] for r in all_results))
    unique_nredos = sorted(set(r['nredo'] for r in all_results))

    section_idx = 2

    # num_clusters 对比
    if len(unique_clusters) > 1:
        niter_mode = Counter(r['niter'] for r in all_results).most_common(1)[0][0]
        nredo_mode = Counter(r['nredo'] for r in all_results).most_common(1)[0][0]
        lines.append("---")
        lines.append("")
        lines.append(f"## {section_idx}. num_clusters 对比 (niter={niter_mode}, nredo={nredo_mode})")
        lines.append("")
        _add_subset_table(lines, all_results, fix={'niter': niter_mode, 'nredo': nredo_mode}, vary='num_clusters')
        section_idx += 1

    # niter 对比
    if len(unique_niters) > 1:
        cls_mode = Counter(r['num_clusters'] for r in all_results).most_common(1)[0][0]
        nredo_mode = Counter(r['nredo'] for r in all_results).most_common(1)[0][0]
        lines.append(f"## {section_idx}. niter 对比 (num_clusters={cls_mode}, nredo={nredo_mode})")
        lines.append("")
        _add_subset_table(lines, all_results, fix={'num_clusters': cls_mode, 'nredo': nredo_mode}, vary='niter')
        section_idx += 1

    # nredo 对比
    if len(unique_nredos) > 1:
        cls_mode = Counter(r['num_clusters'] for r in all_results).most_common(1)[0][0]
        niter_mode = Counter(r['niter'] for r in all_results).most_common(1)[0][0]
        lines.append(f"## {section_idx}. nredo 对比 (num_clusters={cls_mode}, niter={niter_mode})")
        lines.append("")
        _add_subset_table(lines, all_results, fix={'num_clusters': cls_mode, 'niter': niter_mode}, vary='nredo')
        section_idx += 1

    # ---- 3. 最优配置 ----
    lines.append("---")
    lines.append("")
    sort_key = "item_recall@50" if has_ntp else "collision"
    lines.append(f"## {section_idx}. 最优配置 (按 {sort_key} 排序 Top 5)")
    lines.append("")

    for i, r in enumerate(sorted_results[:5], 1):
        m = r['metrics']
        parts = [f"clusters={r['num_clusters']}, niter={r['niter']}, nredo={r['nredo']}"]
        if r.get('quantizer_type') == 'rkmeans_fsq':
            parts.append(f"L3=FSQ({r.get('fsq_levels_key', '?')})")
        if has_ntp:
            r50 = m.get('item_recall@50')
            r50_str = f"{r50:.4f}" if r50 is not None else "N/A"
            parts.append(f"**recall@50={r50_str}**")
            parts.append(f"ntp_ppl={m.get('ntp_perplexity', 'N/A')}")
        parts.append(f"collision={m.get('semantic_id_collision', 'N/A')}")
        parts.append(f"time={r['train_time']:.0f}s")
        lines.append(f"**#{i}**: " + ", ".join(parts))
        lines.append("")


def _add_subset_table(lines, all_results, fix: dict, vary: str):
    """添加固定部分变量、只变一个维度的子表"""

    subset = [r for r in all_results
              if all(r[k] == v for k, v in fix.items())]
    subset = sorted(subset, key=lambda r: r[vary])

    if not subset:
        lines.append(f"_(no matching experiments)_")
        lines.append("")
        return

    has_ntp = any('ntp_perplexity' in r['metrics'] for r in subset)

    if has_ntp:
        header = f"| {vary} | collision | recon_loss | ntp_ppl | **recall@50** | recall@100 | recall@500 | time(s) |"
        sep    = f"|---|-----------|------------|---------|---------------|------------|------------|---------|"
    else:
        header = f"| {vary} | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |"
        sep    = f"|---|-----------|----------|------------|--------|----------|----------|---------|"
    lines.append(header)
    lines.append(sep)

    for r in subset:
        m = r['metrics']

        if has_ntp:
            ntp_ppl = m.get('ntp_perplexity')
            ntp_str = f"{ntp_ppl:.2f}" if ntp_ppl is not None else "N/A"
            r50 = m.get('item_recall@50')
            r100 = m.get('item_recall@100')
            r500 = m.get('item_recall@500')
            r50_str = f"{r50:.4f}" if r50 is not None else "N/A"
            r100_str = f"{r100:.4f}" if r100 is not None else "N/A"
            r500_str = f"{r500:.4f}" if r500 is not None else "N/A"
            vals = [
                f"{m.get('semantic_id_collision', 0):.4f}",
                f"{m.get('reconstruction_loss', 0):.4f}",
                ntp_str, f"**{r50_str}**", r100_str, r500_str,
                f"{r['train_time']:.0f}",
            ]
        else:
            sp_util = m.get('space_utilization')
            sp_str = f"{sp_util:.2e}" if sp_util is not None else "N/A"
            pf = m.get('prefix_avg_items', [])
            if pf and len(pf) >= 3:
                pf_str = [f"{v:.1f}" for v in pf[:3]]
            else:
                pf_str = ["N/A", "N/A", "N/A"]
            vals = [
                f"{m.get('semantic_id_collision', 0):.4f}",
                sp_str,
                f"{m.get('reconstruction_loss', 0):.4f}",
                pf_str[0], pf_str[1], pf_str[2],
                f"{r['train_time']:.0f}",
            ]
        row = f"| {r[vary]} | " + " | ".join(vals) + " |"
        lines.append(row)

    lines.append("")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='RKMeans Hyperparameter Grid Search')
    parser.add_argument('--model', type=str, default='qwen3-0.6b',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--skip_embedding', action='store_true',
                        help='Must use cached embeddings (required unless --sid_cache)')
    parser.add_argument('--behavior_path', type=str, default=None,
                        help='S3 path to behavior data for exposed IID filtering')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (default: auto-generated from cluster list)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output markdown file path (default: experiments/hyperparam/{date}_{name}/report.md)')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing JSON results (skip already-run configs)')

    # 可选: 覆盖搜索空间
    parser.add_argument('--clusters', type=int, nargs='+', default=None,
                        help='Override num_clusters grid (e.g. --clusters 256 512 1024)')
    parser.add_argument('--niters', type=int, nargs='+', default=None,
                        help='Override niter grid (e.g. --niters 25 50 100)')
    parser.add_argument('--nredos', type=int, nargs='+', default=None,
                        help='Override nredo grid (e.g. --nredos 1 3 5)')

    # Quantizer type
    parser.add_argument('--quantizer', type=str, default='rkmeans',
                        choices=['rkmeans', 'rkmeans_fsq', 'opq'],
                        help='Quantizer type: rkmeans | rkmeans_fsq | opq')
    parser.add_argument('--fsq_levels', type=str, nargs='+', default=None,
                        help=f'FSQ level config keys to sweep (available: {list(FSQ_LEVEL_CONFIGS.keys())})')
    parser.add_argument('--fsq_projection', type=str, default='pca',
                        choices=['pca', 'mlp'],
                        help='FSQ projection type: pca (linear) or mlp (learned)')
    parser.add_argument('--fsq_mlp_hidden', type=int, default=128,
                        help='Hidden dim for MLP projection (default: 128)')
    parser.add_argument('--fsq_epochs', type=int, default=50,
                        help='Training epochs for MLP projection (default: 50)')

    # OPQ 相关
    parser.add_argument('--n_subvectors', type=int, nargs='+', default=None,
                        help='OPQ: number of subvectors (tokens per item). e.g. --n_subvectors 8 16 32')
    parser.add_argument('--n_clusters_per_sub', type=int, default=256,
                        help='OPQ: clusters per subvector codebook (must be power of 2, default: 256)')

    # SID prediction (NTP) 相关
    parser.add_argument('--skip_ntp', action='store_true',
                        help='(deprecated, NTP is now off by default)')
    parser.add_argument('--run_ntp', action='store_true',
                        help='Run SID prediction NTP (off by default, slow)')
    parser.add_argument('--force_ar', action='store_true',
                        help='Force autoregressive model even for OPQ (EXP-005 baseline)')
    parser.add_argument('--only-sid', action='store_true',
                        help='Only run SID prediction (skip intrinsic metrics)')
    parser.add_argument('--recall_beam_size', type=int, default=50,
                        help='Beam size for item recall in NTP eval (default: 50)')
    parser.add_argument('--eval_sample_size', type=int, default=50000,
                        help='Max eval samples for NTP (0=all, default: 50000)')
    parser.add_argument('--sid_cache', type=str, default=None,
                        help='Path to preprocess-sid cache dir (skip tokenizer training)')
    parser.add_argument('--ntp_checkpoint', type=str, default=None,
                        help='Path to train-ntp checkpoint dir (probe.pt + eval_data.pt)')
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def run_from_sid_cache(args):
    """Load cached SIDs from preprocess-sid, run eval only (skip tokenizer training)."""
    from gr_demo.eval.behavior import BehaviorMetricsEvaluator
    from gr_demo.eval.wrapper import load_model_wrapper

    cache_dir = args.sid_cache
    run_ntp = args.run_ntp and not args.skip_ntp

    # ── NTP-only fast path: checkpoint has everything, skip heavy loading ──
    if run_ntp and args.ntp_checkpoint:
        print(f"NTP eval from checkpoint: {args.ntp_checkpoint}")
        from gr_demo.metrics.sid_prediction import SemanticIDPredictionMetric
        metric = SemanticIDPredictionMetric()
        metric_result = metric.compute(
            embeddings=None,
            ntp_checkpoint=args.ntp_checkpoint,
            device=args.device,
            recall_beam_size=args.recall_beam_size,
            eval_sample_size=args.eval_sample_size,
        )
        metric_results = {metric_result.name: metric_result}
    else:
        # Full path: load SID cache + embeddings + behavior for tokenizer metrics
        print(f"Loading SID cache from {cache_dir}")

        config_path = os.path.join(cache_dir, 'config.json')
        with open(config_path) as f:
            cache_config = json.load(f)
        print(f"  Tokenizer: clusters={cache_config['num_clusters']}, "
              f"fsq={cache_config['fsq_levels_key']}, "
              f"items={cache_config['n_items']:,}, "
              f"collision={cache_config['collision_rate']:.4f}")

        sid_dict = np.load(os.path.join(cache_dir, 'semantic_ids.npy'), allow_pickle=True).item()
        content_ids = np.array(list(sid_dict.keys()))
        semantic_ids = [sid_dict[cid] for cid in content_ids]
        print(f"  Loaded {len(semantic_ids):,} SID assignments")

        model_data = torch.load(os.path.join(cache_dir, 'quantizer.pt'), map_location='cpu', weights_only=False)
        model_wrapper = load_model_wrapper(model_data, device=args.device)

        # Load embeddings
        model_key = args.model
        _, embedding_dim, _, _ = MODEL_CONFIGS[model_key]
        embedding_cache_dir = f'{EFS_EMBEDDING_CACHE}/{model_key}'
        incremental_cache_path = f'{embedding_cache_dir}/incremental_cache.npy'
        embedding_cache_path = f'{embedding_cache_dir}/embeddings.npy'
        content_ids_cache_path = f'{embedding_cache_dir}/content_ids.npy'

        if os.path.exists(incremental_cache_path):
            emb_dict = np.load(incremental_cache_path, allow_pickle=True).item()
            embeddings = np.array([emb_dict[cid] for cid in content_ids if cid in emb_dict], dtype=np.float32)
            valid_mask = np.array([cid in emb_dict for cid in content_ids])
            content_ids = content_ids[valid_mask]
            semantic_ids = [sid for sid, v in zip(semantic_ids, valid_mask) if v]
        elif os.path.exists(embedding_cache_path):
            all_embeddings = np.load(embedding_cache_path, allow_pickle=True)
            all_cids = np.load(content_ids_cache_path, allow_pickle=True)
            cid_to_idx = {str(c): i for i, c in enumerate(all_cids)}
            indices = [cid_to_idx[cid] for cid in content_ids if cid in cid_to_idx]
            embeddings = all_embeddings[indices]
        else:
            raise FileNotFoundError(f"Embedding cache not found at {embedding_cache_dir}")

        eval_tensor = torch.tensor(embeddings, dtype=torch.float32)
        print(f"  Loaded {len(eval_tensor):,} embeddings")

        # Load behavior data
        from gr_demo.eval.batch import load_all_behavior_data
        behavior_data = load_all_behavior_data()
        print(f"  Behavior data: {len(behavior_data['uid']):,} interactions")

        evaluator = BehaviorMetricsEvaluator(
            embeddings=eval_tensor,
            content_ids=content_ids,
            semantic_ids=semantic_ids,
            model=model_wrapper,
            behavior_data=behavior_data,
            device=args.device,
        )

        if not args.only_sid:
            evaluator.register_metrics(list(INTRINSIC_METRICS.keys()))
        evaluator.register_metrics(['embedding_hit_rate'])
        evaluator.register_metrics(['semantic_neighbor_hit_rate'])

        metric_results = evaluator.evaluate(metric_kwargs={})

    # Output results
    exp_name = args.name or 'sid-cache'
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(repo_root, "experiments", "hyperparam", f"{date.today().isoformat()}_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)

    results = {}
    for mr in metric_results.values():
        results.update(mr.to_flat_dict())

    result_entry = {
        'quantizer_type': 'rkmeans_fsq',
        'sid_cache': cache_dir,
        'metrics': results,
        **{k: v for k, v in cache_config.items() if k in ('num_clusters', 'fsq_levels_key', 'collision_rate', 'n_items')},
    }

    json_path = os.path.join(exp_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump([result_entry], f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    for key, val in sorted(results.items()):
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")


def main():
    args = parse_args()

    # --sid_cache 快速路径: 跳过 tokenizer 训练，直接用缓存的 SID
    if args.sid_cache:
        return run_from_sid_cache(args)

    if not args.skip_embedding:
        raise ValueError("Grid search requires --skip_embedding (or use --sid_cache)")

    # 搜索空间
    clusters_list = args.clusters or GRID['num_clusters']
    niter_list = args.niters or GRID['niter']
    nredo_list = args.nredos or GRID['nredo']
    quantizer_type = args.quantizer

    # FSQ levels sweep
    fsq_levels_keys = []
    if quantizer_type == 'rkmeans_fsq':
        fsq_levels_keys = args.fsq_levels or list(FSQ_LEVEL_CONFIGS.keys())
        for key in fsq_levels_keys:
            if key not in FSQ_LEVEL_CONFIGS:
                raise ValueError(f"Unknown FSQ level config: {key}. Available: {list(FSQ_LEVEL_CONFIGS.keys())}")

    # OPQ subvectors sweep
    opq_subvectors_list = []
    if quantizer_type == 'opq':
        opq_subvectors_list = args.n_subvectors or [8, 16, 32]

    if quantizer_type == 'rkmeans_fsq':
        total = len(clusters_list) * len(niter_list) * len(nredo_list) * len(fsq_levels_keys)
        print(f"Grid search (rkmeans_fsq): {len(clusters_list)} clusters x {len(niter_list)} niters x {len(nredo_list)} nredos x {len(fsq_levels_keys)} fsq_levels = {total} experiments")
        print(f"  clusters:   {clusters_list}")
        print(f"  niter:      {niter_list}")
        print(f"  nredo:      {nredo_list}")
        print(f"  fsq_levels: {fsq_levels_keys}")
    elif quantizer_type == 'opq':
        total = len(opq_subvectors_list)
        print(f"Grid search (opq): {len(opq_subvectors_list)} subvector configs = {total} experiments")
        print(f"  n_subvectors:      {opq_subvectors_list}")
        print(f"  n_clusters_per_sub: {args.n_clusters_per_sub}")
    else:
        total = len(clusters_list) * len(niter_list) * len(nredo_list)
        print(f"Grid search: {len(clusters_list)} clusters x {len(niter_list)} niters x {len(nredo_list)} nredos = {total} experiments")
        print(f"  clusters: {clusters_list}")
        print(f"  niter:    {niter_list}")
        print(f"  nredo:    {nredo_list}")

    # 构造输出目录: experiments/hyperparam/{date}_{name}/
    exp_name = args.name or f"clusters-{'_'.join(str(c) for c in clusters_list)}"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(repo_root, "experiments", "hyperparam", f"{date.today().isoformat()}_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)
    json_path = os.path.join(exp_dir, "results.json")
    output_path = args.output or os.path.join(exp_dir, "report.md")
    print(f"Output dir: {exp_dir}")

    # 加载已有结果 (--append 模式)
    existing_results = []
    existing_keys = set()
    if args.append and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_results = json.load(f)
        for r in existing_results:
            if r.get('quantizer_type') == 'rkmeans_fsq':
                key = (r['num_clusters'], r['niter'], r['nredo'], 'fsq', r.get('fsq_levels_key', ''))
            elif r.get('quantizer_type') == 'opq':
                key = ('opq', r.get('n_subvectors'), r.get('n_clusters_per_sub'))
            else:
                key = (r['num_clusters'], r['niter'], r['nredo'])
            existing_keys.add(key)
        print(f"Loaded {len(existing_results)} existing results from {json_path}")

    # 加载数据
    train_tensor, eval_tensor, content_ids = load_data(args)
    n_features = train_tensor.shape[1]
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # 加载 behavior 数据 (用于 embedding_hit_rate 和可选的 SID prediction)
    behavior_data = None
    run_ntp = args.run_ntp and not args.skip_ntp
    need_behavior = args.behavior_path or args.only_sid or run_ntp
    if need_behavior:
        from gr_demo.eval.batch import load_all_behavior_data
        try:
            behavior_data = load_all_behavior_data()
            print(f"Behavior data loaded: {len(behavior_data['uid']):,} interactions")
        except Exception as e:
            print(f"Warning: Could not load behavior data: {e}")
            print("Behavior metrics will be skipped")

    # 运行网格搜索
    all_results = list(existing_results)
    exp_idx = 0

    # Build experiment configs
    experiment_configs = []
    if quantizer_type == 'opq':
        for n_sub in opq_subvectors_list:
            experiment_configs.append({
                'quantizer_type': 'opq',
                'n_subvectors': n_sub,
                'n_clusters_per_sub': args.n_clusters_per_sub,
            })
    elif quantizer_type == 'rkmeans_fsq':
        for num_clusters in clusters_list:
            for niter in niter_list:
                for nredo in nredo_list:
                    for fsq_key in fsq_levels_keys:
                        experiment_configs.append({
                            'num_clusters': num_clusters,
                            'niter': niter,
                            'nredo': nredo,
                            'quantizer_type': 'rkmeans_fsq',
                            'fsq_levels_key': fsq_key,
                        })
    else:
        for num_clusters in clusters_list:
            for niter in niter_list:
                for nredo in nredo_list:
                    experiment_configs.append({
                        'num_clusters': num_clusters,
                        'niter': niter,
                        'nredo': nredo,
                        'quantizer_type': 'rkmeans',
                    })

    for cfg in experiment_configs:
        exp_idx += 1
        is_opq = cfg['quantizer_type'] == 'opq'
        is_fsq = cfg['quantizer_type'] == 'rkmeans_fsq'

        # Build unique key and label for dedup
        if is_opq:
            n_sub = cfg['n_subvectors']
            n_cpersub = cfg['n_clusters_per_sub']
            config_key = ('opq', n_sub, n_cpersub)
            label = f"opq: m={n_sub}, M={n_cpersub}"
        elif is_fsq:
            num_clusters = cfg['num_clusters']
            niter, nredo = cfg['niter'], cfg['nredo']
            fsq_key = cfg.get('fsq_levels_key', '')
            config_key = (num_clusters, niter, nredo, 'fsq', fsq_key)
            label = f"clusters={num_clusters}, niter={niter}, nredo={nredo}, fsq={fsq_key}"
        else:
            num_clusters = cfg['num_clusters']
            niter, nredo = cfg['niter'], cfg['nredo']
            config_key = (num_clusters, niter, nredo)
            label = f"clusters={num_clusters}, niter={niter}, nredo={nredo}"

        # Skip already-run configs in append mode
        if config_key in existing_keys:
            print(f"\n[{exp_idx}/{total}] {label} — SKIPPED (already exists)")
            continue

        print(f"\n{'#'*60}")
        print(f"# Experiment {exp_idx}/{total}: {label}")
        print(f"{'#'*60}")

        if is_opq:
            metrics, train_time = run_single_experiment_opq(
                train_embeddings=train_tensor,
                eval_embeddings=eval_tensor,
                content_ids=content_ids,
                n_features=n_features,
                n_subvectors=cfg['n_subvectors'],
                n_clusters_per_sub=cfg['n_clusters_per_sub'],
                behavior_data=behavior_data,
                device=args.device,
                recall_beam_size=args.recall_beam_size,
                eval_sample_size=args.eval_sample_size,
                only_sid=args.only_sid,
                run_ntp=run_ntp,
                force_autoregressive=args.force_ar,
            )
        elif is_fsq:
            fsq_levels = FSQ_LEVEL_CONFIGS[fsq_key]
            metrics, train_time = run_single_experiment_fsq(
                train_embeddings=train_tensor,
                eval_embeddings=eval_tensor,
                content_ids=content_ids,
                n_features=n_features,
                num_clusters=num_clusters,
                fsq_levels=fsq_levels,
                fsq_levels_key=fsq_key,
                niter=niter,
                nredo=nredo,
                num_gpus=num_gpus,
                behavior_data=behavior_data,
                device=args.device,
                recall_beam_size=args.recall_beam_size,
                eval_sample_size=args.eval_sample_size,
                only_sid=args.only_sid,
                run_ntp=run_ntp,
                fsq_projection=args.fsq_projection,
                fsq_mlp_hidden=args.fsq_mlp_hidden,
                fsq_epochs=args.fsq_epochs,
            )
        else:
            metrics, train_time = run_single_experiment(
                train_embeddings=train_tensor,
                eval_embeddings=eval_tensor,
                content_ids=content_ids,
                n_features=n_features,
                num_clusters=num_clusters,
                niter=niter,
                nredo=nredo,
                num_gpus=num_gpus,
                behavior_data=behavior_data,
                device=args.device,
                recall_beam_size=args.recall_beam_size,
                eval_sample_size=args.eval_sample_size,
                only_sid=args.only_sid,
                run_ntp=run_ntp,
            )

        result = {
            'quantizer_type': cfg['quantizer_type'],
            'metrics': metrics,
            'train_time': train_time,
        }
        if is_opq:
            result['n_subvectors'] = cfg['n_subvectors']
            result['n_clusters_per_sub'] = cfg['n_clusters_per_sub']
        else:
            result['num_clusters'] = num_clusters
            result['niter'] = niter
            result['nredo'] = nredo
        if is_fsq:
            result['fsq_levels_key'] = fsq_key
            result['fsq_levels'] = FSQ_LEVEL_CONFIGS[fsq_key]
            result['fsq_codebook_size'] = metrics.get('fsq_codebook_size', 0)

        all_results.append(result)

        # 实时打印关键指标
        col_val = metrics.get('semantic_id_collision', 'N/A')
        ent = metrics.get('entropy', 'N/A')
        bal = metrics.get('cluster_balance', 'N/A')
        rec = metrics.get('reconstruction_loss', 'N/A')
        print(f"\n>>> [{exp_idx}/{total}] {label}  time={train_time:.0f}s")
        print(f"    collision={col_val}  entropy={ent}  balance={bal}  recon={rec}")
        if is_opq:
            print(f"    n_subvectors={cfg['n_subvectors']}, n_clusters_per_sub={cfg['n_clusters_per_sub']}")
        if is_fsq:
            print(f"    fsq_codebook_size={metrics.get('fsq_codebook_size', 'N/A')}")
        if 'ntp_perplexity' in metrics:
            ppl = metrics['ntp_perplexity']
            print(f"    ★ ntp_ppl={ppl}")
            recall_parts = []
            for k in (10, 50, 100, 500):
                rk = metrics.get(f'item_recall@{k}')
                if rk is not None:
                    recall_parts.append(f"@{k}={rk:.4f}")
            if recall_parts:
                print(f"    ★ item_recall: {', '.join(recall_parts)}")
        # Print per-depth stats (limit to first 4 for OPQ which may have many layers)
        for key in ('semantic_id_collision_depth', 'entropy_depth', 'cluster_balance_depth', 'codebook_utilization_depth'):
            if key in metrics:
                short = key.replace('_depth', '')
                vals = metrics[key]
                shown = vals[:4]
                suffix = f" ... ({len(vals)} total)" if len(vals) > 4 else ""
                print(f"    {short}: {['d%d=%.4f' % (i+1, v) for i, v in enumerate(shown)]}{suffix}")

        # 每次实验后保存中间结果 (JSON)
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # 生成 Markdown 报告
    report = generate_report(all_results, output_path)
    print(report)

    # 总耗时
    total_time = sum(r['train_time'] for r in all_results)
    print(f"\nTotal training time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    main()
