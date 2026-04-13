"""超参数网格搜索。"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch

from gr_demo.config import MODEL_CONFIGS, EFS_EMBEDDING_CACHE
from gr_demo.model.rkmeans import ResidualQuantizationMultiGPU
from gr_demo.model.semantic_ids import generate_semantic_ids
from gr_demo.data.loaders import load_exposed_iids
from gr_demo.eval.wrapper import RKMeansModelWrapper
from gr_demo.eval.evaluator import MetricsEvaluator

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
        evaluator.register_metrics(list(INTRINSIC_METRICS.keys()))
        evaluator.register_metrics(['semantic_id_prediction'])
        sid_kwargs = {
            'semantic_id_prediction': {
                'device': device,
                'recall_beam_size': recall_beam_size,
                'eval_sample_size': eval_sample_size,
            }
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
        results['ntp_baseline_ppl'] = sid.details.get('baseline_perplexity')
        results['ntp_depth_acc'] = sid.layer_values
        results['ntp_depth_hit@10'] = sid.details.get('depth_hit@10')
        results['ntp_baseline_hit@10'] = sid.details.get('baseline_depth_hit@10')
        # Item recall (the key non-monotonic metric)
        for k in (10, 50, 100, 500):
            results[f'item_recall@{k}'] = sid.details.get(f'item_recall@{k}')
            results[f'baseline_item_recall@{k}'] = sid.details.get(f'baseline_item_recall@{k}')

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

    lines = []
    lines.append("# RKMeans 超参数网格搜索结果")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**模型**: qwen3-0.6b (1024d)")
    lines.append(f"**固定参数**: {NUM_LAYERS} layers, normalize_residuals=True")
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
        header = "| # | clusters | niter | nredo | collision | recon_loss | ntp_ppl | **recall@50** | recall@100 | recall@500 | d1_h@10 | d2_h@10 | d3_h@10 | time(s) |"
        sep    = "|---|----------|-------|-------|-----------|------------|---------|---------------|------------|------------|---------|---------|---------|---------|"
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

            row = f"| {i} | {r['num_clusters']} | {r['niter']} | {r['nredo']} | {col} | {rec} | {ntp_str} | **{r50_str}** | {r100_str} | {r500_str} | {h10_str[0]} | {h10_str[1]} | {h10_str[2]} | {r['train_time']:.0f} |"
        else:
            sp_util = m.get('space_utilization')
            sp_str = f"{sp_util:.2e}" if sp_util is not None else "N/A"
            pf = m.get('prefix_avg_items', [])
            if pf and len(pf) >= 3:
                pf_str = [f"{v:.1f}" for v in pf[:3]]
            else:
                pf_str = ["N/A", "N/A", "N/A"]
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
        if has_ntp:
            r50 = m.get('item_recall@50')
            r50_str = f"{r50:.4f}" if r50 is not None else "N/A"
            parts.append(f"**recall@50={r50_str}**")
            parts.append(f"ntp_ppl={m.get('ntp_perplexity', 'N/A')}")
        parts.append(f"collision={m.get('semantic_id_collision', 'N/A')}")
        parts.append(f"time={r['train_time']:.0f}s")
        lines.append(f"**#{i}**: " + ", ".join(parts))
        lines.append("")

    # 写入文件
    report = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")

    return report


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
    parser.add_argument('--skip_embedding', action='store_true', required=True,
                        help='Must use cached embeddings')
    parser.add_argument('--behavior_path', type=str, default=None,
                        help='S3 path to behavior data for exposed IID filtering')
    parser.add_argument('--output', type=str, default='HYPERPARAM_SEARCH_RESULTS.md',
                        help='Output markdown file path')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing JSON results (skip already-run configs)')

    # 可选: 覆盖搜索空间
    parser.add_argument('--clusters', type=int, nargs='+', default=None,
                        help='Override num_clusters grid (e.g. --clusters 256 512 1024)')
    parser.add_argument('--niters', type=int, nargs='+', default=None,
                        help='Override niter grid (e.g. --niters 25 50 100)')
    parser.add_argument('--nredos', type=int, nargs='+', default=None,
                        help='Override nredo grid (e.g. --nredos 1 3 5)')

    # SID prediction (NTP) 相关
    parser.add_argument('--skip_ntp', action='store_true',
                        help='Skip SID prediction NTP (only run intrinsic metrics)')
    parser.add_argument('--recall_beam_size', type=int, default=50,
                        help='Beam size for item recall in NTP eval (default: 50)')
    parser.add_argument('--eval_sample_size', type=int, default=50000,
                        help='Max eval samples for NTP (0=all, default: 50000)')
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def main():
    args = parse_args()

    # 搜索空间
    clusters_list = args.clusters or GRID['num_clusters']
    niter_list = args.niters or GRID['niter']
    nredo_list = args.nredos or GRID['nredo']

    total = len(clusters_list) * len(niter_list) * len(nredo_list)
    print(f"Grid search: {len(clusters_list)} clusters x {len(niter_list)} niters x {len(nredo_list)} nredos = {total} experiments")
    print(f"  clusters: {clusters_list}")
    print(f"  niter:    {niter_list}")
    print(f"  nredo:    {nredo_list}")

    # 加载已有结果 (--append 模式)
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             args.output.replace('.md', '.json'))
    existing_results = []
    existing_keys = set()
    if args.append and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_results = json.load(f)
        for r in existing_results:
            key = (r['num_clusters'], r['niter'], r['nredo'])
            existing_keys.add(key)
        print(f"Loaded {len(existing_results)} existing results from {json_path}")

    # 加载数据
    train_tensor, eval_tensor, content_ids = load_data(args)
    n_features = train_tensor.shape[1]
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # 加载 behavior 数据 (用于 SID prediction)
    behavior_data = None
    if args.behavior_path and not args.skip_ntp:
        from gr_demo.eval.batch import load_all_behavior_data
        try:
            behavior_data = load_all_behavior_data()
            print(f"Behavior data loaded: {len(behavior_data['uid']):,} interactions")
        except Exception as e:
            print(f"Warning: Could not load behavior data: {e}")
            print("SID prediction will be skipped")

    # 运行网格搜索
    all_results = list(existing_results)
    exp_idx = 0

    for num_clusters in clusters_list:
        for niter in niter_list:
            for nredo in nredo_list:
                exp_idx += 1

                # Skip already-run configs in append mode
                if (num_clusters, niter, nredo) in existing_keys:
                    print(f"\n[{exp_idx}/{total}] clusters={num_clusters}, niter={niter}, nredo={nredo} — SKIPPED (already exists)")
                    continue

                print(f"\n{'#'*60}")
                print(f"# Experiment {exp_idx}/{total}: clusters={num_clusters}, niter={niter}, nredo={nredo}")
                print(f"{'#'*60}")

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
                )

                result = {
                    'num_clusters': num_clusters,
                    'niter': niter,
                    'nredo': nredo,
                    'metrics': metrics,
                    'train_time': train_time,
                }
                all_results.append(result)

                # 实时打印关键指标
                col_val = metrics.get('semantic_id_collision', 'N/A')
                ent = metrics.get('entropy', 'N/A')
                bal = metrics.get('cluster_balance', 'N/A')
                rec = metrics.get('reconstruction_loss', 'N/A')
                print(f"\n>>> [{exp_idx}/{total}] clusters={num_clusters} niter={niter} nredo={nredo}  time={train_time:.0f}s")
                print(f"    collision={col_val}  entropy={ent}  balance={bal}  recon={rec}")
                if 'ntp_perplexity' in metrics:
                    ppl = metrics['ntp_perplexity']
                    base = metrics.get('ntp_baseline_ppl', 'N/A')
                    print(f"    ★ ntp_ppl={ppl}  baseline_ppl={base}")
                    recall_parts = []
                    for k in (10, 50, 100, 500):
                        rk = metrics.get(f'item_recall@{k}')
                        if rk is not None:
                            recall_parts.append(f"@{k}={rk:.4f}")
                    if recall_parts:
                        print(f"    ★ item_recall: {', '.join(recall_parts)}")
                for key in ('semantic_id_collision_depth', 'entropy_depth', 'cluster_balance_depth', 'codebook_utilization_depth'):
                    if key in metrics:
                        short = key.replace('_depth', '')
                        vals = metrics[key]
                        print(f"    {short}: {['d%d=%.4f' % (i+1, v) for i, v in enumerate(vals)]}")

                # 每次实验后保存中间结果 (JSON)
                with open(json_path, 'w') as f:
                    json.dump(all_results, f, indent=2)

    # 生成 Markdown 报告
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    report = generate_report(all_results, output_path)
    print(report)

    # 总耗时
    total_time = sum(r['train_time'] for r in all_results)
    print(f"\nTotal training time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    main()
