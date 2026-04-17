#!/usr/bin/env python3
"""Recover exp015 results from train_meta.json → experiments/results/ntp/.

Use when checkpoints exist but results JSON was not saved (e.g. script
interrupted before the git commit step).

Usage:
    python experiments/scripts/exp015_recover_results.py
"""

import glob
import json
import os

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ckpt_base = os.path.join(repo_root, 'experiments', 'ntp_checkpoints')
results_dir = os.path.join(repo_root, 'experiments', 'results', 'ntp')
os.makedirs(results_dir, exist_ok=True)

pattern = os.path.join(ckpt_base, 'exp015-scale-*', 'train_meta.json')
recovered = 0

for meta_path in sorted(glob.glob(pattern)):
    name = os.path.basename(os.path.dirname(meta_path))
    results_path = os.path.join(results_dir, f'{name}.json')

    with open(meta_path) as f:
        meta = json.load(f)

    eval_data = meta.get('eval', {})
    train_data = meta.get('train', {})

    result = {
        'name': name,
        'model_type': 's-tier',
        'n_params': meta.get('n_params', 0),
        'n_active_params': train_data.get('n_active_params', meta.get('n_params', 0)),
        'total_tokens': train_data.get('total_tokens', 0),
        'total_flops': train_data.get('total_flops', 0),
        'sid_cache': meta.get('sid_cache', ''),
        'eval': eval_data,
    }

    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)

    ppl = eval_data.get('ppl', 'N/A')
    n_active = result['n_active_params']
    print(f"  {name}: n_active={n_active/1e6:.1f}M, PPL={ppl} → {results_path}")
    recovered += 1

print(f"\nRecovered {recovered} results to {results_dir}/")
if recovered > 0:
    print("\nNext steps:")
    print("  git add experiments/results/ntp/exp015-*.json")
    print("  git commit -m 'EXP-015 results: scaling law 7 configs'")
    print("  ./push.sh")
