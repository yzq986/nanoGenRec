"""Pipeline waterfall chart: metric changes across training stages.

Shows how PPL and Recall evolve through NTP → SP-DPO → RF-DPO stages.

Usage:
    python -m viz.pipeline \
        --ntp exp016-B-14d-S \
        --spdpo exp017-fixed-medium \
        --rfdpo exp019-joint-easy exp019-joint-hard
    python -m viz.pipeline \
        --ntp exp016-B-14d-S \
        --spdpo exp017-fixed-medium \
        --rfdpo exp019-* \
        --save pipeline.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from viz.loader import load_experiments, get_eval_metrics, get_train_config


def build_pipeline_data(ntp_name, spdpo_name, rfdpo_patterns):
    """Build pipeline stage data.

    Returns:
        list of stage dicts: [{name, stage, ppl, R@10, R@500, ...}, ...]
    """
    all_patterns = [ntp_name, spdpo_name] + rfdpo_patterns
    experiments = load_experiments(all_patterns)

    name_to_exp = {e['name']: e for e in experiments}
    stages = []

    # Stage 1: NTP
    if ntp_name in name_to_exp:
        ev = get_eval_metrics(name_to_exp[ntp_name]['meta'])
        if ev:
            stages.append({
                'name': ntp_name,
                'stage': 'NTP',
                'stage_idx': 0,
                **ev,
            })

    # Stage 2: SP-DPO
    if spdpo_name in name_to_exp:
        ev = get_eval_metrics(name_to_exp[spdpo_name]['meta'])
        if ev:
            stages.append({
                'name': spdpo_name,
                'stage': 'SP-DPO',
                'stage_idx': 1,
                **ev,
            })

    # Stage 3: RF-DPO (multiple configs)
    for exp in experiments:
        if exp['name'] in (ntp_name, spdpo_name):
            continue
        ev = get_eval_metrics(exp['meta'])
        cfg = get_train_config(exp['meta'])
        if ev:
            diff = cfg['difficulty']
            lam = cfg['dpo_weight']
            mode = 'pure' if cfg['pure_dpo'] else f'λ={lam}'
            label = f"RF-DPO ({diff}, {mode})"
            stages.append({
                'name': exp['name'],
                'stage': label,
                'stage_idx': 2,
                **ev,
            })

    return stages


def plot_pipeline(stages, metrics=None, save_path=None):
    """Waterfall chart showing metric progression across stages."""
    if not stages:
        print("No stage data available.")
        return

    if metrics is None:
        metrics = ['ppl', 'R@10', 'R@500']
    metrics = [m for m in metrics if any(s.get(m) is not None for s in stages)]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    stage_colors = {0: '#2196F3', 1: '#4CAF50', 2: '#FF9800'}
    stage_names_map = {0: 'NTP', 1: 'SP-DPO', 2: 'RF-DPO'}

    for ax, metric in zip(axes, metrics):
        names = []
        values = []
        colors = []

        for s in sorted(stages, key=lambda x: (x['stage_idx'], x['name'])):
            v = s.get(metric)
            if v is None:
                continue
            if metric.startswith('R@') and v <= 1.0:
                v = v * 100

            short = s['name'].replace('exp0', 'E')
            if s['stage_idx'] == 2:
                short = s['stage'].replace('RF-DPO ', '')
            names.append(short)
            values.append(v)
            colors.append(stage_colors.get(s['stage_idx'], 'gray'))

        if not values:
            continue

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.6)

        for bar, val in zip(bars, values):
            fmt = f'{val:.1f}' if metric == 'ppl' else f'{val:.1f}%'
            ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    fmt, va='center', fontsize=9)

        # Reference line from SP-DPO stage
        spdpo_vals = [v for s, v in zip(stages, values)
                      if s.get('stage_idx') == 1 and s.get(metric) is not None]
        if spdpo_vals:
            ref = spdpo_vals[0]
            if metric.startswith('R@') and ref <= 1.0:
                ref = ref * 100
            ax.axvline(ref, color='green', linestyle='--', alpha=0.5, linewidth=1.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(_metric_label(metric))
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')
        ax.invert_yaxis()

        if metric == 'ppl' and max(values) / max(min(values), 1) > 20:
            ax.set_xscale('log')

    handles = [mpatches.Patch(color=c, label=n)
               for idx, (c, n) in enumerate(
                   [(stage_colors[k], stage_names_map[k]) for k in sorted(stage_colors)])]
    fig.legend(handles=handles, loc='upper right', fontsize=10)
    fig.suptitle('Pipeline Progression: NTP → SP-DPO → RF-DPO', fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def print_pipeline_table(stages):
    """Print pipeline progression as text table."""
    metrics = ['ppl', 'R@10', 'R@50', 'R@100', 'R@500']

    header = f"{'Stage':<40} {'PPL':>8} {'R@10':>7} {'R@50':>7} {'R@100':>7} {'R@500':>7}"
    print("\n" + "=" * len(header))
    print("Pipeline Progression")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    prev_ppl = None
    for s in sorted(stages, key=lambda x: (x['stage_idx'], x['name'])):
        name = s['stage'] if s['stage_idx'] < 2 else s['stage']
        short_name = f"{name} ({s['name']})"
        if len(short_name) > 39:
            short_name = short_name[:36] + '...'

        ppl = s.get('ppl')
        ppl_str = f"{ppl:.1f}" if ppl else '—'
        if prev_ppl and ppl:
            delta = (ppl - prev_ppl) / prev_ppl * 100
            sign = '+' if delta > 0 else ''
            ppl_str += f" ({sign}{delta:.0f}%)"

        cols = [ppl_str.rjust(8)]
        for m in ['R@10', 'R@50', 'R@100', 'R@500']:
            v = s.get(m)
            if v is not None:
                if v <= 1.0:
                    v = v * 100
                cols.append(f"{v:.1f}%".rjust(7))
            else:
                cols.append('—'.rjust(7))

        print(f"{short_name:<40} {'  '.join(cols)}")

        if s.get('stage_idx', 0) <= 1 and ppl:
            prev_ppl = ppl

    print("=" * len(header))


def _metric_label(metric):
    if metric == 'ppl':
        return 'Perplexity (lower is better)'
    return f'{metric} % (higher is better)'


def main():
    parser = argparse.ArgumentParser(description='Pipeline waterfall chart')
    parser.add_argument('--ntp', type=str, required=True,
                        help='NTP stage experiment name')
    parser.add_argument('--spdpo', type=str, required=True,
                        help='SP-DPO stage experiment name')
    parser.add_argument('--rfdpo', nargs='+', required=True,
                        help='RF-DPO experiment name patterns')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file')
    args = parser.parse_args()

    stages = build_pipeline_data(args.ntp, args.spdpo, args.rfdpo)
    if not stages:
        print("No pipeline data found.")
        sys.exit(1)

    print(f"Loaded {len(stages)} stages")
    print_pipeline_table(stages)

    if stages:
        plot_pipeline(stages, save_path=args.save)


if __name__ == '__main__':
    main()
