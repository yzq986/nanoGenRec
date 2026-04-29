#!/usr/bin/env python3
"""Backfill registry.json from existing train_meta.json checkpoints."""
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).parent.parent
CKPT_DIR = REPO_ROOT / "experiments" / "ntp_checkpoints"
REGISTRY_PATH = REPO_ROOT / "experiments" / "registry.json"

# Known config mapping: checkpoint name → config params
# Manually curated from experiments/log.md + scripts
KNOWN_CONFIGS = {
    "exp020-hard-lam03": {
        "model": "probe", "sid_cache_name": "exp020-hard-lam03",
        "date_start": "2026-03-18", "date_end": "2026-03-31",
        "use_segment_emb": False, "use_torope": False, "use_gate_attn": False,
        "batch_size": 4096, "lr": None, "entp_weight": 0.0,
        "contrastive_weight": 0.0, "contrastive_temp": 0.07, "contrastive_dim": 128,
        "action_l2_only": False, "min_action_level": 1,
        "n_items": 10, "max_seq_len": 512, "n_eval_target": 50000,
        "torope_time_split": 0.5,
    },
    "exp043-s-0.6b": {
        "model": "s-tier", "sid_cache_name": "exp026-0.6b-14d",
        "date_start": "2026-03-18", "date_end": "2026-03-31",
        "use_segment_emb": True, "use_torope": False, "use_gate_attn": False,
        "batch_size": 4096, "lr": None, "entp_weight": 0.0,
        "contrastive_weight": 0.0, "contrastive_temp": 0.07, "contrastive_dim": 128,
        "action_l2_only": False, "min_action_level": 1,
        "n_items": 10, "max_seq_len": 512, "n_eval_target": 50000,
        "torope_time_split": 0.5,
    },
    "exp044b-torope-ts05": {
        "model": "s-tier", "sid_cache_name": "exp026-0.6b-14d",
        "date_start": "2026-03-18", "date_end": "2026-03-31",
        "use_segment_emb": True, "use_torope": True, "torope_time_split": 0.5,
        "use_gate_attn": False,
        "batch_size": 4096, "lr": None, "entp_weight": 0.0,
        "contrastive_weight": 0.0, "contrastive_temp": 0.07, "contrastive_dim": 128,
        "action_l2_only": False, "min_action_level": 1,
        "n_items": 10, "max_seq_len": 512, "n_eval_target": 50000,
    },
    "exp044b-torope-ts025": {
        "model": "s-tier", "sid_cache_name": "exp026-0.6b-14d",
        "date_start": "2026-03-18", "date_end": "2026-03-31",
        "use_segment_emb": True, "use_torope": True, "torope_time_split": 0.25,
        "use_gate_attn": False,
        "batch_size": 4096, "lr": None, "entp_weight": 0.0,
        "contrastive_weight": 0.0, "contrastive_temp": 0.07, "contrastive_dim": 128,
        "action_l2_only": False, "min_action_level": 1,
        "n_items": 10, "max_seq_len": 512, "n_eval_target": 50000,
    },
    "exp044b-torope-ts05-notg": {
        "model": "s-tier", "sid_cache_name": "exp026-0.6b-14d",
        "date_start": "2026-03-18", "date_end": "2026-03-31",
        "use_segment_emb": True, "use_torope": True, "torope_time_split": 0.5,
        "use_gate_attn": False,
        "batch_size": 4096, "lr": None, "entp_weight": 0.0,
        "contrastive_weight": 0.0, "contrastive_temp": 0.07, "contrastive_dim": 128,
        "action_l2_only": False, "min_action_level": 1,
        "n_items": 10, "max_seq_len": 512, "n_eval_target": 50000,
    },
}

HASH_EXCLUDE = {
    "behavior_path", "output_dir", "preprocessed_dir",
    "n_gpus", "log", "dry_run", "eval_only",
    "shift_features", "seed",
}

def config_hash(cfg: dict) -> str:
    hashable = {k: v for k, v in cfg.items() if k not in HASH_EXCLUDE}
    canonical = json.dumps(hashable, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=REPO_ROOT).decode().strip()[:12]
    except Exception:
        return "unknown"

def main():
    registry = {}
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text())

    commit = get_git_commit()
    n_added = 0

    for ckpt_name, cfg in KNOWN_CONFIGS.items():
        meta_path = CKPT_DIR / ckpt_name / "train_meta.json"
        results = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            results = meta.get("eval", {})

        h = config_hash(cfg)
        registry[h] = {
            "name": ckpt_name,
            "config": {k: v for k, v in cfg.items() if k not in HASH_EXCLUDE},
            "results": results,
            "git_commit": commit,
            "registered_at": datetime.now().isoformat(),
            "backfilled": True,
        }
        r500 = results.get("item_recall@500")
        print(f"  {ckpt_name:<40} hash={h}  R@500={r500:.1%}" if r500
              else f"  {ckpt_name:<40} hash={h}  (no results yet)")
        n_added += 1

    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, sort_keys=True))
    print(f"\nDone. {n_added} entries written to {REGISTRY_PATH}")

if __name__ == "__main__":
    main()
