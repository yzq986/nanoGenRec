#!/usr/bin/env python3
"""EXP-016 analysis: Data Scaling Law — Loss vs Data Window.

Plots loss, PPL, and recall vs training data days for S (17.5M) and M+ (101M).
Key finding: 14d (~130M tokens) is optimal; more data hurts due to distribution shift.
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(repo_root, 'experiments', 'results', 'ntp')

configs = [
    ('A-7d', 7, 65),
    ('B-14d', 14, 130),
    ('C-31d', 31, 262),
    ('D-62d', 62, 441),
    ('E-90d', 90, 553),
]

users_by_days = {7: 1.02e6, 14: 1.69e6, 31: 3.04e6, 62: 4.86e6, 90: 6.18e6}

s_data = []
m_data = []

for prefix, days, approx_tokens in configs:
    for model_tag, model_label, store in [('-S', 'S', s_data), ('-M', 'M+', m_data)]:
        path = os.path.join(results_dir, f'exp016-{prefix}{model_tag}.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        ev = d.get('eval', {})
        if not ev.get('ppl'):
            continue
        store.append({
            'days': days,
            'tokens_m': d.get('total_tokens', approx_tokens * 1e6) / 1e6,
            'users': users_by_days.get(days, 0),
            'ppl': ev['ppl'],
            'loss': ev['avg_loss'],
            'r10': ev.get('item_recall@10', 0),
            'r100': ev.get('item_recall@100', 0),
            'r500': ev.get('item_recall@500', 0),
        })

print(f"S model: {len(s_data)} data points")
print(f"M+ model: {len(m_data)} data points")

for label, data in [('S (17.5M)', s_data), ('M+ (101M)', m_data)]:
    print(f"\n{label}:")
    print(f"  {'Days':>5} {'Tokens':>8} {'Users':>8} {'PPL':>8} {'Loss':>8} {'R@100':>7} {'R@500':>7}")
    for d in sorted(data, key=lambda x: x['days']):
        print(f"  {d['days']:>5d} {d['tokens_m']:>7.0f}M {d['users']/1e6:>7.2f}M "
              f"{d['ppl']:>8.2f} {d['loss']:>8.4f} {d['r100']:>6.1%} {d['r500']:>6.1%}")

# ── Plot ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('EXP-016: Data Scaling Law — Loss vs Training Data Window', fontsize=14, fontweight='bold')

colors = {'S': '#2196F3', 'M+': '#E91E63'}
markers = {'S': 'o', 'M+': 's'}

def plot_series(ax, data, key, label, color, marker):
    days = [d['days'] for d in sorted(data, key=lambda x: x['days'])]
    vals = [d[key] for d in sorted(data, key=lambda x: x['days'])]
    ax.plot(days, vals, f'-{marker}', color=color, label=label, markersize=8, linewidth=2)
    best_idx = np.argmin(vals) if key == 'loss' else np.argmax(vals)
    ax.annotate(f'{vals[best_idx]:.3f}' if key == 'loss' else f'{vals[best_idx]:.2f}',
                xy=(days[best_idx], vals[best_idx]),
                textcoords='offset points', xytext=(0, 12), ha='center',
                fontsize=9, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# (1) Loss vs Days
ax = axes[0, 0]
plot_series(ax, s_data, 'loss', 'S (17.5M active)', colors['S'], markers['S'])
plot_series(ax, m_data, 'loss', 'M+ (101M active)', colors['M+'], markers['M+'])
ax.axvline(x=14, color='gray', linestyle='--', alpha=0.5, label='14d optimal')
ax.set_xlabel('Training Data (days)')
ax.set_ylabel('Eval Loss')
ax.set_title('Eval Loss vs Data Window')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (2) PPL vs Days
ax = axes[0, 1]
plot_series(ax, s_data, 'ppl', 'S (17.5M active)', colors['S'], markers['S'])
plot_series(ax, m_data, 'ppl', 'M+ (101M active)', colors['M+'], markers['M+'])
ax.axvline(x=14, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Training Data (days)')
ax.set_ylabel('PPL')
ax.set_title('Perplexity vs Data Window')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (3) Recall@500 vs Days
ax = axes[1, 0]
plot_series(ax, s_data, 'r500', 'S R@500', colors['S'], markers['S'])
plot_series(ax, m_data, 'r500', 'M+ R@500', colors['M+'], markers['M+'])
plot_series(ax, s_data, 'r100', 'S R@100', colors['S'], 'v')
plot_series(ax, m_data, 'r100', 'M+ R@100', colors['M+'], 'D')
ax.axvline(x=14, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Training Data (days)')
ax.set_ylabel('Recall')
ax.set_title('Item Recall vs Data Window')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# (4) Loss vs Tokens (log-x) with user count annotation
ax = axes[1, 1]
for label_short, data, color, marker in [('S', s_data, colors['S'], markers['S']),
                                          ('M+', m_data, colors['M+'], markers['M+'])]:
    data_sorted = sorted(data, key=lambda x: x['tokens_m'])
    tokens = [d['tokens_m'] for d in data_sorted]
    losses = [d['loss'] for d in data_sorted]
    ax.plot(tokens, losses, f'-{marker}', color=color, label=f'{label_short}', markersize=8, linewidth=2)
    for d in data_sorted:
        ax.annotate(f'{d["days"]}d\n{d["users"]/1e6:.1f}M users',
                    xy=(d['tokens_m'], d['loss']),
                    textcoords='offset points',
                    xytext=(10, -5 if label_short == 'S' else 5),
                    fontsize=7, color=color, alpha=0.8)
ax.set_xlabel('Training Tokens (M)')
ax.set_ylabel('Eval Loss')
ax.set_xscale('log')
ax.set_title('Loss vs Tokens (log scale) — NOT Chinchilla')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.02,
        'More tokens = more users (1M→6M), NOT longer sequences\n'
        'Distribution shift from stale users dominates after 14d',
        transform=ax.transAxes, ha='center', fontsize=8, style='italic',
        color='gray')

plt.tight_layout()
out_path = os.path.join(results_dir, 'exp016-data-scaling.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")

summary = {
    'experiment': 'EXP-016',
    'finding': 'Optimal training window is ~14 days for both S and M+ models',
    'optimal_s': {'days': 14, 'tokens_m': 130, 'ppl': s_data[1]['ppl'], 'loss': s_data[1]['loss']},
    'optimal_m': {'days': 14, 'tokens_m': 130, 'ppl': m_data[1]['ppl'], 'loss': m_data[1]['loss']},
    's_results': [{k: d[k] for k in ['days', 'tokens_m', 'ppl', 'loss', 'r100', 'r500']} for d in sorted(s_data, key=lambda x: x['days'])],
    'm_results': [{k: d[k] for k in ['days', 'tokens_m', 'ppl', 'loss', 'r100', 'r500']} for d in sorted(m_data, key=lambda x: x['days'])],
}
summary_path = os.path.join(results_dir, 'exp016-summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {summary_path}")
