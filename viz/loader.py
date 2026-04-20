"""Load experiment data from checkpoints."""

import glob
import json
import os

CKPT_ROOT = 'experiments/ntp_checkpoints'


def find_experiments(patterns):
    """Resolve experiment name patterns to checkpoint dirs.

    Args:
        patterns: list of experiment name patterns (supports shell glob)
            e.g. ['exp019-*', 'exp017-fixed-medium']

    Returns:
        list of (name, dir_path) tuples, sorted by name
    """
    results = {}
    for pat in patterns:
        full_pat = os.path.join(CKPT_ROOT, pat)
        for path in glob.glob(full_pat):
            if os.path.isdir(path):
                name = os.path.basename(path)
                results[name] = path
    return sorted(results.items())


def load_meta(exp_dir):
    """Load train_meta.json from experiment directory."""
    path = os.path.join(exp_dir, 'train_meta.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_train_log(exp_dir):
    """Load train_log.jsonl as list of dicts."""
    path = os.path.join(exp_dir, 'train_log.jsonl')
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_experiments(patterns):
    """Load multiple experiments with meta + log.

    Returns:
        list of dicts with keys: name, dir, meta, log
    """
    experiments = []
    for name, dir_path in find_experiments(patterns):
        meta = load_meta(dir_path)
        if meta is None:
            continue
        log = load_train_log(dir_path)
        experiments.append({
            'name': name,
            'dir': dir_path,
            'meta': meta,
            'log': log,
        })
    return experiments


def get_eval_metrics(meta):
    """Extract key eval metrics from train_meta.

    Returns:
        dict with ppl, recall@K, etc. or None if no eval.
    """
    ev = meta.get('eval')
    if not ev:
        return None

    result = {
        'ppl': ev.get('ppl'),
        'avg_loss': ev.get('avg_loss'),
    }

    for k in [10, 50, 100, 500]:
        key = f'item_recall@{k}'
        if key in ev and ev[key] is not None:
            result[f'R@{k}'] = ev[key]

    if 'R@500' not in result and 'target_sid_found_rate' in ev:
        result['R@500'] = ev['target_sid_found_rate']

    if 'R@10' not in result and 'depth_hit@10' in ev:
        hits = ev['depth_hit@10']
        if hits:
            result['R@10'] = hits[-1]

    return result


def get_alignment_metrics(meta):
    """Extract alignment metrics from train_summary.

    Returns:
        dict with reward/preference stats, or None if not available.
    """
    train = meta.get('train', {})
    if 'avg_reward_margin' not in train:
        return None
    return {
        'avg_chosen_reward': train.get('avg_chosen_reward'),
        'avg_rejected_reward': train.get('avg_rejected_reward'),
        'avg_reward_margin': train.get('avg_reward_margin'),
        'avg_preference_acc': train.get('avg_preference_acc'),
    }


def get_train_config(meta):
    """Extract training config from meta."""
    train = meta.get('train', {})
    return {
        'dpo_weight': train.get('dpo_weight', 0),
        'dpo_beta': train.get('dpo_beta', 0.1),
        'difficulty': train.get('difficulty', 'all'),
        'pure_dpo': train.get('pure_dpo', False),
        'lr': train.get('lr', 1e-4) if 'lr' in train else meta.get('lr', 1e-4),
        'n_steps': train.get('n_steps', 0),
        'wall_time_s': train.get('wall_time_s', 0),
    }
