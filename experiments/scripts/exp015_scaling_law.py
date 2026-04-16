#!/usr/bin/env python3
"""EXP-015: Scaling Law Analysis — Fit L̂(N) = a + b / N^α from experiment results.

Loads all exp015-scale-* results from experiments/results/ntp/,
fits a power law to (active_params, eval_loss), and plots the curve.

Usage:
    python experiments/scripts/exp015_scaling_law.py
    python experiments/scripts/exp015_scaling_law.py --include-exp013  # add exp013 data points
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def load_results(results_dir, include_exp013=False):
    """Load scaling law data points from experiment result JSONs."""
    points = []

    # EXP-015 configs
    for path in sorted(glob.glob(os.path.join(results_dir, 'exp015-scale-*.json'))):
        with open(path) as f:
            data = json.load(f)
        n_active = data.get('n_active_params', data['n_params'])
        eval_data = data.get('eval', {})
        loss = eval_data.get('avg_loss')
        ppl = eval_data.get('ppl')
        if loss is None:
            print(f"  SKIP {os.path.basename(path)}: no eval loss")
            continue
        points.append({
            'name': data['name'],
            'n_params': data['n_params'],
            'n_active': n_active,
            'total_tokens': data.get('total_tokens', 0),
            'total_flops': data.get('total_flops', 0),
            'loss': loss,
            'ppl': ppl,
            'recall@10': eval_data.get('item_recall@10'),
            'recall@50': eval_data.get('item_recall@50'),
            'recall@100': eval_data.get('item_recall@100'),
            'recall@500': eval_data.get('item_recall@500'),
            'source': path,
        })

    # Optionally include EXP-013 data points
    if include_exp013:
        for name in ['exp013-probe', 'exp013-s-tier']:
            path = os.path.join(results_dir, f'{name}.json')
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)
            n_active = data.get('n_active_params', data['n_params'])
            eval_data = data.get('eval', {})
            loss = eval_data.get('avg_loss')
            if loss is None:
                continue
            points.append({
                'name': name,
                'n_params': data['n_params'],
                'n_active': n_active,
                'total_tokens': data.get('total_tokens', 0),
                'total_flops': data.get('total_flops', 0),
                'loss': loss,
                'ppl': eval_data.get('ppl'),
                'recall@10': eval_data.get('item_recall@10'),
                'recall@50': eval_data.get('item_recall@50'),
                'recall@100': eval_data.get('item_recall@100'),
                'recall@500': eval_data.get('item_recall@500'),
                'source': path,
            })

    points.sort(key=lambda p: p['n_active'])
    return points


def fit_scaling_law(N_active, losses):
    """Fit L̂(N) = a + b / N^α using scipy.optimize.curve_fit.

    Returns (a, b, alpha), or None if scipy is not available.
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    def power_law(N, a, b, alpha):
        return a + b / np.power(N, alpha)

    N = np.array(N_active, dtype=np.float64)
    L = np.array(losses, dtype=np.float64)

    # Initial guesses: a ~ min(L), b ~ range * N_mid^0.5, alpha ~ 0.5
    a0 = L.min() * 0.8
    alpha0 = 0.5
    b0 = (L.max() - a0) * np.median(N) ** alpha0

    try:
        popt, pcov = curve_fit(power_law, N, L, p0=[a0, b0, alpha0],
                               bounds=([0, 0, 0.01], [np.inf, np.inf, 2.0]),
                               maxfev=10000)
        return popt
    except Exception as e:
        print(f"  curve_fit failed: {e}")
        return None


def plot_scaling_law(points, params, output_path):
    """Plot scaling law: data points + fitted curve. Saves to PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    N = np.array([p['n_active'] for p in points])
    L = np.array([p['loss'] for p in points])
    names = [p['name'].replace('exp015-scale-', '').replace('exp013-', '*') for p in points]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Loss vs Active Params (log-log)
    ax1.scatter(N / 1e6, L, s=80, zorder=5, color='#2563eb')
    for i, name in enumerate(names):
        ax1.annotate(name, (N[i] / 1e6, L[i]), textcoords="offset points",
                     xytext=(5, 5), fontsize=7, color='gray')

    if params is not None:
        a, b, alpha = params
        N_fit = np.logspace(np.log10(N.min() * 0.5), np.log10(N.max() * 2), 200)
        L_fit = a + b / np.power(N_fit, alpha)
        ax1.plot(N_fit / 1e6, L_fit, 'r--', linewidth=1.5,
                 label=f'$L = {a:.3f} + {b:.1f} / N^{{{alpha:.3f}}}$')
        ax1.legend(fontsize=10)

    ax1.set_xscale('log')
    ax1.set_xlabel('Active Parameters (M)')
    ax1.set_ylabel('Eval Loss')
    ax1.set_title('NTP Scaling Law: Loss vs Model Size')
    ax1.grid(True, alpha=0.3)

    # Right: Recall@100 vs Active Params (log scale x)
    recalls = [p.get('recall@100') for p in points]
    if all(r is not None for r in recalls):
        R = np.array(recalls) * 100
        ax2.scatter(N / 1e6, R, s=80, zorder=5, color='#16a34a')
        for i, name in enumerate(names):
            ax2.annotate(name, (N[i] / 1e6, R[i]), textcoords="offset points",
                         xytext=(5, 5), fontsize=7, color='gray')
        ax2.set_xscale('log')
        ax2.set_xlabel('Active Parameters (M)')
        ax2.set_ylabel('Recall@100 (%)')
        ax2.set_title('Recall@100 vs Model Size')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Recall data not available', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12, color='gray')
        ax2.set_title('Recall@100 vs Model Size')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='EXP-015 Scaling Law Analysis')
    parser.add_argument('--include-exp013', action='store_true',
                        help='Include exp013 data points')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(repo_root, 'experiments', 'results', 'ntp')

    print("=" * 60)
    print("EXP-015: NTP Scaling Law Analysis")
    print("=" * 60)

    points = load_results(results_dir, include_exp013=args.include_exp013)
    if len(points) < 2:
        print(f"\nOnly {len(points)} data points found. Need at least 2.")
        print("Run exp-015.sh first to generate results.")
        sys.exit(1)

    # Print table
    print(f"\n{'Name':<28} {'Total':>8} {'Active':>8} {'Tokens':>10} {'Loss':>7} {'PPL':>7} {'R@10':>6} {'R@100':>6} {'R@500':>6}")
    print("-" * 100)
    for p in points:
        tokens_str = f"{p['total_tokens']/1e6:.0f}M" if p['total_tokens'] else 'N/A'
        r10 = f"{p['recall@10']*100:.1f}" if p.get('recall@10') else 'N/A'
        r100 = f"{p['recall@100']*100:.1f}" if p.get('recall@100') else 'N/A'
        r500 = f"{p['recall@500']*100:.1f}" if p.get('recall@500') else 'N/A'
        print(f"{p['name']:<28} {p['n_params']/1e6:>7.1f}M {p['n_active']/1e6:>7.1f}M "
              f"{tokens_str:>10} {p['loss']:>7.4f} {p['ppl']:>7.1f} "
              f"{r10:>6} {r100:>6} {r500:>6}")

    # Fit scaling law
    N_active = [p['n_active'] for p in points]
    losses = [p['loss'] for p in points]

    print(f"\nFitting L̂(N) = a + b / N^α ...")
    params = fit_scaling_law(N_active, losses)
    if params is not None:
        a, b, alpha = params
        print(f"  a     = {a:.4f}  (irreducible loss floor)")
        print(f"  b     = {b:.1f}")
        print(f"  alpha = {alpha:.4f}")
        print(f"\n  L̂(N) = {a:.3f} + {b:.1f} / N^{alpha:.3f}")

        # Predictions
        print(f"\n  Predictions:")
        for n_m in [1, 5, 10, 50, 100, 500, 1000]:
            n = n_m * 1e6
            pred = a + b / n ** alpha
            print(f"    N={n_m:>5}M active → L̂ = {pred:.4f}  (PPL ≈ {np.exp(pred):.1f})")
    else:
        print("  scipy not available. Install with: pip install scipy")
        print("  Skipping curve fit.")

    # Plot
    plot_dir = os.path.join(repo_root, 'experiments', 'results', 'ntp')
    plot_path = os.path.join(plot_dir, 'exp015-scaling-law.png')
    plot_scaling_law(points, params, plot_path)

    # Save summary JSON
    summary = {
        'experiment': 'exp-015',
        'n_configs': len(points),
        'points': [{
            'name': p['name'],
            'n_params': p['n_params'],
            'n_active_params': p['n_active'],
            'total_tokens': p['total_tokens'],
            'eval_loss': p['loss'],
            'eval_ppl': p['ppl'],
        } for p in points],
    }
    if params is not None:
        summary['scaling_law'] = {
            'formula': f'L(N) = {params[0]:.4f} + {params[1]:.1f} / N^{params[2]:.4f}',
            'a': float(params[0]),
            'b': float(params[1]),
            'alpha': float(params[2]),
        }
    summary_path = os.path.join(plot_dir, 'exp015-scaling-law-summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
