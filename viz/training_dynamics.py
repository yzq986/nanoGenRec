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


def plot_training_dynamics(experiments, save_path=None, smooth=5):
    """Plot NTP loss, DPO loss, and grad_norm for multiple experiments."""
    n_exps = len(experiments)
    if n_exps == 0:
        print("No experiments found.")
        return

    has_ntp = any(
        any(r.get('ntp_loss', 0) > 0 for r in exp['log'])
        for exp in experiments
    )
    n_rows = 3 if has_ntp else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_exps, 1)))

    for i, exp in enumerate(experiments):
        log = exp['log']
        if not log:
            continue

        steps = [r['step'] for r in log]
        color = colors[i]
        label = exp['name']

        row = 0

        if has_ntp:
            ntp = [r.get('ntp_loss', 0) for r in log]
            if any(v > 0 for v in ntp):
                axes[row].plot(steps, _smooth(ntp, smooth),
                               color=color, label=label, alpha=0.8)
            row += 1

        dpo = [r.get('dpo_loss', 0) for r in log]
        axes[row].plot(steps, _smooth(dpo, smooth),
                       color=color, label=label, alpha=0.8)

        gnorm = [r.get('grad_norm', 0) for r in log]
        axes[row + 1].plot(steps, _smooth(gnorm, smooth),
                           color=color, label=label, alpha=0.8)

    row = 0
    if has_ntp:
        axes[row].set_ylabel('NTP Loss')
        axes[row].legend(fontsize=8)
        axes[row].grid(True, alpha=0.3)
        row += 1

    axes[row].set_ylabel('DPO Loss')
    axes[row].legend(fontsize=8)
    axes[row].grid(True, alpha=0.3)

    axes[row + 1].set_ylabel('Grad Norm')
    axes[row + 1].set_xlabel('Step')
    axes[row + 1].legend(fontsize=8)
    axes[row + 1].grid(True, alpha=0.3)

    fig.suptitle('Training Dynamics', fontsize=14)
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
