"""多模型对比报告。"""

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List


def load_model_results(eval_dir: str) -> Dict[str, Dict]:
    """加载所有模型的评估结果"""
    results = {}

    for model_name in os.listdir(eval_dir):
        model_dir = os.path.join(eval_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        json_path = os.path.join(model_dir, 'report.json')
        if not os.path.exists(json_path):
            print(f"Warning: No report.json found for {model_name}")
            continue

        with open(json_path, 'r') as f:
            results[model_name] = json.load(f)

    return results


def generate_comparison_table(results: Dict[str, Dict]) -> str:
    """生成 Markdown 对比表格"""
    if not results:
        return "No results to compare"

    # 获取所有 metric 名称
    metric_names = set()
    for model_data in results.values():
        metric_names.update(model_data.get('metrics', {}).keys())
    metric_names = sorted(metric_names)

    # 模型排序
    models = sorted(results.keys())

    lines = []

    # Header
    header = ['Metric'] + models
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '|'.join(['---'] * len(header)) + '|')

    # Data rows
    for metric in metric_names:
        row = [metric.replace('_', ' ').title()]
        for model in models:
            model_metrics = results[model].get('metrics', {})
            if metric in model_metrics:
                value = model_metrics[metric]['value']
                status = model_metrics[metric].get('status', 'unknown')
                # Format based on metric type
                if 'utilization' in metric or 'entropy' in metric or 'dimension' in metric:
                    cell = f"{value:.1%}"
                elif 'collision' in metric or 'gini' in metric or 'balance' in metric:
                    cell = f"{value:.2%}"
                else:
                    cell = f"{value:.4f}"
                # Add status indicator
                status_icon = {'excellent': '🟢', 'good': '🟡', 'acceptable': '🟠', 'poor': '🔴'}.get(status, '')
                row.append(f"{cell} {status_icon}")
            else:
                row.append('N/A')
        lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(lines)


def generate_ranking(results: Dict[str, Dict]) -> str:
    """生成模型排名"""
    if not results:
        return "No results to rank"

    # 计算综合得分
    scores = {}
    for model, data in results.items():
        metrics = data.get('metrics', {})
        excellent = sum(1 for m in metrics.values() if m.get('status') == 'excellent')
        good = sum(1 for m in metrics.values() if m.get('status') == 'good')
        acceptable = sum(1 for m in metrics.values() if m.get('status') == 'acceptable')
        poor = sum(1 for m in metrics.values() if m.get('status') == 'poor')

        # 加权得分: excellent=3, good=2, acceptable=1, poor=0
        score = excellent * 3 + good * 2 + acceptable * 1
        scores[model] = {
            'score': score,
            'excellent': excellent,
            'good': good,
            'acceptable': acceptable,
            'poor': poor,
        }

    # 排序
    ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)

    lines = []
    lines.append('| Rank | Model | Score | Excellent | Good | Acceptable | Poor |')
    lines.append('|------|-------|-------|-----------|------|------------|------|')

    for rank, (model, data) in enumerate(ranked, 1):
        medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, '')
        lines.append(f"| {rank} {medal} | {model} | {data['score']} | {data['excellent']} | {data['good']} | {data['acceptable']} | {data['poor']} |")

    return '\n'.join(lines)


def generate_best_per_metric(results: Dict[str, Dict]) -> str:
    """每个 metric 的最佳模型"""
    if not results:
        return "No results"

    # 收集所有 metrics
    metric_names = set()
    for model_data in results.values():
        metric_names.update(model_data.get('metrics', {}).keys())

    lines = []
    lines.append('| Metric | Best Model | Value | Status |')
    lines.append('|--------|------------|-------|--------|')

    for metric in sorted(metric_names):
        best_model = None
        best_value = None
        best_status = None

        # 判断是否是 "越高越好" 的 metric
        higher_is_better = any(x in metric for x in ['utilization', 'entropy', 'dimension', 'similarity'])

        for model, data in results.items():
            metrics = data.get('metrics', {})
            if metric not in metrics:
                continue

            value = metrics[metric]['value']
            status = metrics[metric].get('status', 'unknown')

            if best_value is None:
                best_model = model
                best_value = value
                best_status = status
            elif higher_is_better and value > best_value:
                best_model = model
                best_value = value
                best_status = status
            elif not higher_is_better and value < best_value:
                best_model = model
                best_value = value
                best_status = status

        if best_model:
            status_icon = {'excellent': '🟢', 'good': '🟡', 'acceptable': '🟠', 'poor': '🔴'}.get(best_status, '')
            if 'utilization' in metric or 'entropy' in metric or 'dimension' in metric:
                value_str = f"{best_value:.1%}"
            elif 'collision' in metric or 'gini' in metric:
                value_str = f"{best_value:.2%}"
            else:
                value_str = f"{best_value:.4f}"
            lines.append(f"| {metric.replace('_', ' ').title()} | **{best_model}** | {value_str} | {status_icon} {best_status} |")

    return '\n'.join(lines)


def generate_comparison_report(results: Dict[str, Dict], output_dir: str):
    """生成完整的对比报告"""
    os.makedirs(output_dir, exist_ok=True)

    # Markdown report
    lines = []
    lines.append(f'# Embedding Model Comparison Report')
    lines.append('')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Models compared: {len(results)}')
    lines.append('')

    lines.append('## Model Ranking')
    lines.append('')
    lines.append(generate_ranking(results))
    lines.append('')

    lines.append('## Best Model per Metric')
    lines.append('')
    lines.append(generate_best_per_metric(results))
    lines.append('')

    lines.append('## Full Comparison Table')
    lines.append('')
    lines.append(generate_comparison_table(results))
    lines.append('')

    lines.append('## Interpretation Guide')
    lines.append('')
    lines.append('| Metric | Direction | Ideal | Description |')
    lines.append('|--------|-----------|-------|-------------|')
    lines.append('| Reconstruction Loss | Lower ↓ | < 0.05 | Quantization precision |')
    lines.append('| Codebook Utilization | Higher ↑ | 100% | Codebook capacity usage |')
    lines.append('| Entropy | Higher ↑ | > 95% | Token distribution uniformity |')
    lines.append('| Cosine Similarity Std | Higher ↑ | > 0.25 | Embedding discrimination |')
    lines.append('| Effective Dimension | Higher ↑ | > 70% | Dimension utilization |')
    lines.append('| Collision Rate | Lower ↓ | < 1% | Semantic ID uniqueness |')
    lines.append('| Cluster Gini | Lower ↓ | < 0.15 | Cluster balance |')
    lines.append('')

    # Model details
    lines.append('## Model Details')
    lines.append('')
    for model in sorted(results.keys()):
        meta = results[model].get('metadata', {})
        lines.append(f'### {model}')
        lines.append('')
        lines.append(f'- Samples: {meta.get("n_samples", "N/A"):,}')
        lines.append(f'- Embedding dim: {meta.get("embedding_dim", "N/A")}')
        lines.append(f'- Unique semantic IDs: {meta.get("n_unique_semantic_ids", "N/A"):,}')
        lines.append('')

    md_path = os.path.join(output_dir, 'COMPARISON.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Markdown report: {md_path}")

    # JSON report
    json_report = {
        'timestamp': datetime.now().isoformat(),
        'models': list(results.keys()),
        'results': results,
    }
    json_path = os.path.join(output_dir, 'comparison.json')
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"JSON report: {json_path}")

    # CSV for easy spreadsheet import
    csv_path = os.path.join(output_dir, 'comparison.csv')

    metric_names = set()
    for model_data in results.values():
        metric_names.update(model_data.get('metrics', {}).keys())
    metric_names = sorted(metric_names)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model'] + metric_names)
        for model in sorted(results.keys()):
            row = [model]
            for metric in metric_names:
                metrics = results[model].get('metrics', {})
                if metric in metrics:
                    row.append(metrics[metric]['value'])
                else:
                    row.append('')
            writer.writerow(row)
    print(f"CSV report: {csv_path}")

    return md_path


def parse_args():
    parser = argparse.ArgumentParser(description='Compare multiple embedding model evaluations')
    parser.add_argument('--eval_dir', type=str, default='eval_results',
                        help='Directory containing model evaluation results')
    parser.add_argument('--output', type=str, default='comparison_report',
                        help='Output directory for comparison reports')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Multi-Model Comparison")
    print("=" * 60)

    # Load all results
    results = load_model_results(args.eval_dir)
    print(f"Loaded {len(results)} model results: {list(results.keys())}")

    if not results:
        print("No results found!")
        return 1

    # Generate comparison report
    generate_comparison_report(results, args.output)

    print("=" * 60)
    print("Comparison complete!")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    exit(main())
