"""Degradation budget table: delta% vs baseline for each config.

Usage:
    python -m viz.degradation exp019-* --baseline exp017-fixed-medium
    python -m viz.degradation exp018-* exp019-* --baseline exp017-fixed-medium --save deg.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from viz.loader import load_experiments, get_eval_metrics, get_train_config


def compute_degradation(experiments, baseline_name):
    """Compute metric deltas vs baseline.

    Returns:
        baseline_metrics: dict
        rows: list of dicts with name, config, and delta_* fields
    """
    baseline = None
    others = []
    for exp in experiments:
        ev = get_eval_metrics(exp['meta'])
        cfg = get_train_config(exp['meta'])
        if ev is None:
            continue
        entry = {'name': exp['name'], 'metrics': ev, 'config': cfg}
        if exp['name'] == baseline_name:
            baseline = entry
        others.append(entry)

    if baseline is None:
        print(f"Baseline '{baseline_name}' not found in experiments.")
        return None, []

    bm = baseline['metrics']
    rows = []
    for entry in others:
        if entry['name'] == baseline_name:
            continue
        em = entry['metrics']
        row = {
            'name': entry['name'],
            'config': entry['config'],
        }
        for key in ['ppl', 'R@10', 'R@500']:
            bv = bm.get(key)
            ev = em.get(key)
            dkey = f'Δ{key.upper()}' if key == 'ppl' else f'Δ{key}'
            if bv is not None and ev is not None and bv != 0:
                row[dkey] = (ev - bv) / bv * 100
                row[key] = ev
            else:
                row[dkey] = None
                row[key] = ev
        rows.append(row)

    return bm, rows


def print_table(baseline_metrics, rows):
    """Print degradation table to stdout."""
    bm = baseline_metrics

    header = f"{'Experiment':<30} {'λ':>5} {'Diff':>6} {'PPL':>8} {'ΔPPL':>8} {'R@10':>7} {'ΔR@10':>8} {'R@500':>7} {'ΔR@500':>8}"
    print("\n" + "=" * len(header))
    print("Degradation vs Baseline")
    print(f"Baseline PPL={bm.get('ppl', '?'):.1f}, "
          f"R@10={_fmt_pct(bm.get('R@10'))}, "
          f"R@500={_fmt_pct(bm.get('R@500'))}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for row in sorted(rows, key=lambda r: r.get('ppl') or 9999):
        cfg = row['config']
        lam = f"{cfg['dpo_weight']:.2f}" if not cfg['pure_dpo'] else 'pure'
        diff = cfg['difficulty'][:4]
        ppl = f"{row.get('ppl', 0):.1f}" if row.get('ppl') else '—'
        dppl = _fmt_delta(row.get('ΔPPL'))
        r10 = _fmt_pct(row.get('R@10'))
        dr10 = _fmt_delta(row.get('ΔR@10'))
        r500 = _fmt_pct(row.get('R@500'))
        dr500 = _fmt_delta(row.get('ΔR@500'))
        print(f"{row['name']:<30} {lam:>5} {diff:>6} {ppl:>8} {dppl:>8} {r10:>7} {dr10:>8} {r500:>7} {dr500:>8}")

    print("=" * len(header))


def plot_degradation(baseline_metrics, rows, save_path=None, clamp_ppl=1000):
    """Bar chart of degradation deltas.

    Args:
        clamp_ppl: cap ΔPPL display at this % to prevent extreme outliers
                   from squashing the readable range. Clamped bars get a
                   '>>>' annotation showing the true value.
    """
    rows = [r for r in rows if r.get('ΔPPL') is not None]
    if not rows:
        return

    rows = sorted(rows, key=lambda r: r.get('ppl') or 9999)
    names = [r['name'].replace('exp0', 'E') for r in rows]
    metrics = ['ΔPPL', 'ΔR@10', 'ΔR@500']
    available = [m for m in metrics if any(r.get(m) is not None for r in rows)]

    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, max(4, 0.5 * len(rows))))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        raw_vals = [r.get(metric, 0) or 0 for r in rows]

        if metric == 'ΔPPL':
            vals = [min(v, clamp_ppl) for v in raw_vals]
            clamped = [v > clamp_ppl for v in raw_vals]
        else:
            vals = raw_vals
            clamped = [False] * len(vals)

        is_bad_up = metric == 'ΔPPL'
        colors = [('#d32f2f' if v > 0 else '#388e3c') if is_bad_up
                  else ('#388e3c' if v > 0 else '#d32f2f')
                  for v in vals]

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, vals, color=colors, alpha=0.8)

        for bar, raw, clip, v in zip(bars, raw_vals, clamped, vals):
            if clip:
                label = f'>>> +{raw:,.0f}%'
            else:
                sign = '+' if raw > 0 else ''
                label = f'{sign}{raw:.1f}%'
            x = bar.get_width()
            ha = 'left' if x >= 0 else 'right'
            offset = max(abs(max(vals) - min(vals)) * 0.02, 1)
            ax.text(x + (offset if x >= 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    label, va='center', ha=ha, fontsize=8, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(f'{metric} (%)')
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

    fig.suptitle('Degradation vs Baseline (%)', fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def _fmt_pct(v):
    if v is None:
        return '—'
    if v <= 1.0:
        return f"{v * 100:.1f}%"
    return f"{v:.1f}%"


def _fmt_delta(v):
    if v is None:
        return '—'
    sign = '+' if v > 0 else ''
    return f"{sign}{v:.1f}%"


def main():
    parser = argparse.ArgumentParser(description='Degradation budget table')
    parser.add_argument('experiments', nargs='+',
                        help='Experiment name patterns')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Baseline experiment name')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file (also prints table)')
    args = parser.parse_args()

    all_patterns = args.experiments + [args.baseline]
    experiments = load_experiments(all_patterns)
    if not experiments:
        print(f"No experiments found.")
        sys.exit(1)

    bm, rows = compute_degradation(experiments, args.baseline)
    if bm is None:
        sys.exit(1)

    print_table(bm, rows)

    if rows:
        plot_degradation(bm, rows, save_path=args.save)


if __name__ == '__main__':
    main()
