"""Pareto frontier: PPL vs Recall scatter plot.

Usage:
    python -m viz.pareto exp019-* exp017-* --baseline exp017-fixed-medium
    python -m viz.pareto exp019-* --save pareto.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from viz.loader import load_experiments, get_eval_metrics, get_train_config


DIFFICULTY_MARKERS = {
    'easy': 'o',
    'medium': 's',
    'hard': '^',
    'all': 'D',
}

RECALL_KEY_PRIORITY = ['R@10', 'R@500']


def plot_pareto(experiments, baseline_name=None, recall_key=None, save_path=None):
    """Scatter plot of PPL vs Recall with config annotations."""
    points = []
    for exp in experiments:
        ev = get_eval_metrics(exp['meta'])
        cfg = get_train_config(exp['meta'])
        if ev is None or ev.get('ppl') is None:
            continue

        rk = recall_key
        if rk is None:
            for k in RECALL_KEY_PRIORITY:
                if k in ev and ev[k] is not None:
                    rk = k
                    break
        if rk is None:
            continue

        recall = ev.get(rk)
        if recall is None:
            continue

        points.append({
            'name': exp['name'],
            'ppl': ev['ppl'],
            'recall': recall * 100 if recall <= 1.0 else recall,
            'difficulty': cfg['difficulty'],
            'dpo_weight': cfg['dpo_weight'],
            'pure_dpo': cfg['pure_dpo'],
            'is_baseline': exp['name'] == baseline_name,
        })

    if not points:
        print("No experiments with eval results found.")
        return

    if rk is None:
        rk = 'Recall'

    fig, ax = plt.subplots(figsize=(10, 7))

    for pt in points:
        marker = DIFFICULTY_MARKERS.get(pt['difficulty'], 'o')
        color = 'red' if pt['pure_dpo'] else 'steelblue'
        size = 200 if pt['is_baseline'] else 80
        edge = 'gold' if pt['is_baseline'] else 'none'
        zorder = 10 if pt['is_baseline'] else 5

        ax.scatter(pt['ppl'], pt['recall'],
                   marker='*' if pt['is_baseline'] else marker,
                   c=color, s=size, edgecolors=edge, linewidths=2,
                   zorder=zorder)

        offset = (5, 5)
        fontsize = 7
        short_name = pt['name'].replace('exp0', 'E')
        label = short_name
        if pt['dpo_weight'] > 0 and not pt['pure_dpo']:
            label += f' λ={pt["dpo_weight"]}'
        ax.annotate(label, (pt['ppl'], pt['recall']),
                    textcoords='offset points', xytext=offset,
                    fontsize=fontsize, alpha=0.7)

    ax.set_xlabel('Perplexity (lower is better)', fontsize=12)
    ax.set_ylabel(f'{rk} % (higher is better)', fontsize=12)
    ax.set_title('Post-Training Pareto: PPL vs Recall', fontsize=14)

    if any(pt['ppl'] > 200 for pt in points):
        ax.set_xscale('log')

    # Legend
    from matplotlib.lines import Line2D
    handles = []
    handles.append(Line2D([0], [0], marker='o', color='steelblue',
                          linestyle='', label='Joint NTP+DPO'))
    handles.append(Line2D([0], [0], marker='o', color='red',
                          linestyle='', label='Pure DPO'))
    for diff, mk in DIFFICULTY_MARKERS.items():
        handles.append(Line2D([0], [0], marker=mk, color='gray',
                              linestyle='', label=f'Difficulty: {diff}'))
    if baseline_name:
        handles.append(Line2D([0], [0], marker='*', color='gold',
                              markersize=12, linestyle='', label='Baseline'))
    ax.legend(handles=handles, fontsize=8, loc='lower left')

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Pareto scatter: PPL vs Recall')
    parser.add_argument('experiments', nargs='+',
                        help='Experiment name patterns')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Baseline experiment name (shown as star)')
    parser.add_argument('--recall', type=str, default=None,
                        choices=['R@10', 'R@500'],
                        help='Which recall metric to use (default: auto)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file')
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)
    if not experiments:
        print(f"No experiments found matching: {args.experiments}")
        sys.exit(1)

    print(f"Loaded {len(experiments)} experiments with eval data")
    plot_pareto(experiments, baseline_name=args.baseline,
                recall_key=args.recall, save_path=args.save)


if __name__ == '__main__':
    main()
