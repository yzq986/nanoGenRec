"""Training dynamics visualization: NTP loss, DPO loss, grad_norm over steps.

Usage:
    python -m viz.training_dynamics exp019-joint-easy exp019-joint-hard
    python -m viz.training_dynamics exp019-* --save dynamics.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from viz.loader import load_experiments


def plot_training_dynamics(experiments, save_path=None, smooth=0):
    """Plot NTP loss, DPO loss, grad_norm, and alignment metrics.

    Args:
        smooth: smoothing window. 0 = auto (2% of longest experiment).
    """
    n_exps = len(experiments)
    if n_exps == 0:
        print("No experiments found.")
        return

    if smooth == 0:
        max_steps = max(len(exp['log']) for exp in experiments)
        smooth = max(1, max_steps // 50)

    has_ntp = any(
        any(r.get('ntp_loss', 0) > 0 for r in exp['log'])
        for exp in experiments
    )
    has_alignment = any(
        any('chosen_reward' in r for r in exp['log'])
        for exp in experiments
    )
    n_rows = (1 if has_ntp else 0) + 2 + (2 if has_alignment else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_exps, 10)))

    for i, exp in enumerate(experiments):
        log = exp['log']
        if not log:
            continue

        steps = [r['step'] for r in log]
        color = colors[i % 10]
        label = exp['name'].replace('exp0', 'E')
        lw = 1.5

        row = 0

        if has_ntp:
            ntp = [r.get('ntp_loss', 0) for r in log]
            if any(v > 0 for v in ntp):
                axes[row].plot(steps, _smooth(ntp, smooth),
                               color=color, label=label, alpha=0.85, linewidth=lw)
            row += 1

        dpo = [r.get('dpo_loss', 0) for r in log]
        axes[row].plot(steps, _smooth(dpo, smooth),
                       color=color, label=label, alpha=0.85, linewidth=lw)

        gnorm = [r.get('grad_norm', 0) for r in log]
        axes[row + 1].plot(steps, _smooth(gnorm, smooth),
                           color=color, label=label, alpha=0.85, linewidth=lw)

        if has_alignment:
            cr = [r.get('chosen_reward', 0) for r in log]
            rr = [r.get('rejected_reward', 0) for r in log]
            if any(r != 0 for r in cr):
                axes[row + 2].plot(steps, _smooth(cr, smooth),
                                   color=color, label=f'{label} chosen',
                                   alpha=0.85, linewidth=lw, linestyle='-')
                axes[row + 2].plot(steps, _smooth(rr, smooth),
                                   color=color, label=f'{label} rejected',
                                   alpha=0.6, linewidth=lw, linestyle='--')

            pa = [r.get('preference_acc', 0) for r in log]
            if any(v > 0 for v in pa):
                axes[row + 3].plot(steps, _smooth(pa, smooth),
                                   color=color, label=label, alpha=0.85, linewidth=lw)

    row = 0
    if has_ntp:
        axes[row].set_ylabel('NTP Loss')
        axes[row].legend(fontsize=8, loc='upper right')
        axes[row].grid(True, alpha=0.3)
        row += 1

    axes[row].set_ylabel('DPO Loss')
    axes[row].legend(fontsize=8, loc='upper right')
    axes[row].grid(True, alpha=0.3)

    axes[row + 1].set_ylabel('Grad Norm')
    axes[row + 1].legend(fontsize=8, loc='upper right')
    axes[row + 1].grid(True, alpha=0.3)

    if has_alignment:
        axes[row + 2].set_ylabel('Implicit Reward')
        axes[row + 2].legend(fontsize=7, loc='upper right', ncol=2)
        axes[row + 2].grid(True, alpha=0.3)
        axes[row + 2].axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        axes[row + 3].set_ylabel('Preference Acc')
        axes[row + 3].set_ylim(-0.05, 1.05)
        axes[row + 3].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='random')
        axes[row + 3].legend(fontsize=8, loc='lower right')
        axes[row + 3].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Step')
    fig.suptitle(f'Training Dynamics (smooth={smooth})', fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def _smooth(values, window):
    """Simple moving average smoothing."""
    if window <= 1 or len(values) <= window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid').tolist()


def main():
    parser = argparse.ArgumentParser(description='Plot training dynamics')
    parser.add_argument('experiments', nargs='+',
                        help='Experiment name patterns (e.g. exp019-*)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file instead of showing')
    parser.add_argument('--smooth', type=int, default=5,
                        help='Smoothing window (default: 5)')
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)
    if not experiments:
        print(f"No experiments found matching: {args.experiments}")
        sys.exit(1)

    print(f"Loaded {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  {exp['name']}: {len(exp['log'])} steps")

    plot_training_dynamics(experiments, save_path=args.save, smooth=args.smooth)


if __name__ == '__main__':
    main()
