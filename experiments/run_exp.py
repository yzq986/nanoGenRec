#!/usr/bin/env python3
"""Experiment runner with config registry and dedup.

Usage:
    python experiments/run_exp.py configs/exp-047.yaml [--check] [--no-smoke] [--force]

    --check   : show similar past experiments and exit (no training)
    --no-smoke: skip smoke test
    --force   : run even if hash already in registry
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = REPO_ROOT / "experiments" / "registry.json"
CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"
CKPT_DIR = REPO_ROOT / "experiments" / "ntp_checkpoints"
SID_CACHE_ROOT = REPO_ROOT / "experiments" / "sid_cache"
NTP_DATA_ROOT = REPO_ROOT / "experiments" / "ntp_data"
BEHAVIOR_CACHE = "/mnt/workspace/gr-demo-behavior-cache"

# Keys excluded from hash (paths / environment / runtime)
HASH_EXCLUDE = {
    "behavior_path", "output_dir", "preprocessed_dir",
    "n_gpus", "log", "dry_run", "eval_only",
    # fixed constants — not variables
    "shift_features", "seed",
}

# Keys that are multi-config metadata, not model params
META_KEYS = {"name", "description", "base"}


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def save_registry(registry: dict):
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, sort_keys=True))


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_config(yaml_path: Path) -> dict:
    """Merge base + delta, fill argparse defaults, return resolved config."""
    raw = load_yaml(yaml_path)

    # Load base
    base_name = raw.get("base", "_base.yaml")
    base = load_yaml(CONFIGS_DIR / base_name)

    # Merge: base ← delta (delta wins)
    resolved = {**base, **raw}

    # Remove meta keys
    for k in META_KEYS:
        resolved.pop(k, None)

    return resolved


def config_hash(resolved: dict) -> str:
    """Hash the resolved config, excluding path/env keys."""
    hashable = {k: v for k, v in resolved.items() if k not in HASH_EXCLUDE}
    canonical = json.dumps(hashable, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def find_similar(registry: dict, resolved: dict, top_n: int = 5) -> list[dict]:
    """Return past experiments sorted by number of differing keys."""
    results = []
    for h, entry in registry.items():
        past = entry.get("config", {})
        diffs = {}
        all_keys = set(resolved) | set(past)
        for k in all_keys:
            if k in HASH_EXCLUDE or k in META_KEYS:
                continue
            v_new = resolved.get(k)
            v_old = past.get(k)
            if v_new != v_old:
                diffs[k] = (v_old, v_new)
        results.append({
            "hash": h,
            "name": entry.get("name", "?"),
            "diffs": diffs,
            "n_diffs": len(diffs),
            "results": entry.get("results", {}),
            "commit": entry.get("git_commit", "?"),
        })
    results.sort(key=lambda x: x["n_diffs"])
    return results[:top_n]


def print_similar(similar: list[dict]):
    print("\nSimilar past experiments:")
    print(f"  {'name':<32}  {'diffs':<50}  {'R@500':>6}  commit")
    print(f"  {'-'*32}  {'-'*50}  {'-'*6}  {'-'*8}")
    for s in similar:
        r500 = s["results"].get("item_recall@500", None)
        r500_str = f"{r500:.1%}" if r500 else "  n/a"
        if s["n_diffs"] == 0:
            diff_str = "[IDENTICAL CONFIG]"
        else:
            parts = [f"{k}: {v[0]}→{v[1]}" for k, v in list(s["diffs"].items())[:3]]
            if len(s["diffs"]) > 3:
                parts.append(f"(+{len(s['diffs'])-3} more)")
            diff_str = ", ".join(parts)
        print(f"  {s['name']:<32}  {diff_str:<50}  {r500_str:>6}  {s['commit'][:8]}")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def register_experiment(name: str, resolved: dict, h: str, results: dict = None):
    registry = load_registry()
    registry[h] = {
        "name": name,
        "config": {k: v for k, v in resolved.items() if k not in HASH_EXCLUDE},
        "results": results or {},
        "git_commit": get_git_commit(),
        "registered_at": datetime.now().isoformat(),
    }
    save_registry(registry)
    print(f"  Registered: hash={h}, name={name}")


def update_results(h: str, results: dict):
    """Update results for an existing registry entry."""
    registry = load_registry()
    if h in registry:
        registry[h]["results"] = results
        registry[h]["results_updated_at"] = datetime.now().isoformat()
        save_registry(registry)


def build_torchrun_cmd(resolved: dict, name: str, n_gpus: int,
                       no_smoke: bool = False, dry_run: bool = False) -> list[str]:
    """Build torchrun command from resolved config."""
    sid_cache = str(SID_CACHE_ROOT / resolved["sid_cache_name"])
    ntp_data = str(NTP_DATA_ROOT / f"{resolved['sid_cache_name'].replace('sid_cache/', '')}")
    output_dir = str(CKPT_DIR / name)

    cmd = [
        "torchrun", f"--nproc_per_node={n_gpus}",
        "run.py", "train-ntp",
        "--preprocessed_dir", ntp_data,
        "--output_dir", output_dir,
        "--name", name,
        "--model", resolved["model"],
        "--batch_size", str(resolved["batch_size"]),
        "--entp_weight", str(resolved["entp_weight"]),
        "--contrastive_weight", str(resolved["contrastive_weight"]),
    ]

    if resolved.get("lr"):
        cmd += ["--lr", str(resolved["lr"])]
    if resolved.get("use_segment_emb"):
        cmd.append("--use_segment_emb")
    if resolved.get("use_torope"):
        cmd += ["--use_torope", "--torope_time_split", str(resolved["torope_time_split"])]
        if resolved.get("torope_layer_split", 0.0) > 0:
            cmd += ["--torope_layer_split", str(resolved["torope_layer_split"])]
    if resolved.get("use_gate_attn"):
        cmd.append("--use_gate_attn")
    if dry_run:
        cmd.append("--dry_run")

    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to experiment yaml")
    parser.add_argument("--check", action="store_true", help="show similar exps and exit")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--force", action="store_true", help="run even if hash exists")
    parser.add_argument("--n-gpus", type=int, default=None)
    args = parser.parse_args()

    yaml_path = Path(args.config)
    if not yaml_path.is_absolute():
        yaml_path = Path.cwd() / yaml_path

    raw = load_yaml(yaml_path)
    name = raw.get("name") or yaml_path.stem
    resolved = resolve_config(yaml_path)
    h = config_hash(resolved)
    registry = load_registry()

    print(f"\nExperiment: {name}")
    print(f"  Config hash: {h}")

    # Always show similar experiments
    similar = find_similar(registry, resolved)
    if similar:
        print_similar(similar)
        # Exact match warning
        exact = [s for s in similar if s["n_diffs"] == 0]
        if exact and not args.force:
            print(f"\n  !! IDENTICAL config already in registry: {exact[0]['name']} "
                  f"(commit {exact[0]['commit'][:8]})")
            r500 = exact[0]["results"].get("item_recall@500")
            if r500:
                print(f"     R@500={r500:.1%} — use --force to re-run anyway")
            else:
                print(f"     No results yet — use --force to re-run anyway")
            sys.exit(0)

    if args.check:
        sys.exit(0)

    # Detect n_gpus
    n_gpus = args.n_gpus
    if n_gpus is None:
        try:
            import torch
            n_gpus = max(1, torch.cuda.device_count())
        except Exception:
            n_gpus = 1
    print(f"  GPUs: {n_gpus}")

    # Register before running (so parallel runs don't double-start)
    register_experiment(name, resolved, h)

    # Build and run command
    cmd = build_torchrun_cmd(resolved, name, n_gpus, no_smoke=args.no_smoke)
    print(f"\n  CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\n  ERROR: training failed (exit {result.returncode})")
        sys.exit(result.returncode)

    # Read results from train_meta.json and update registry
    meta_path = CKPT_DIR / name / "train_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        results = meta.get("eval", {})
        update_results(h, results)
        r500 = results.get("item_recall@500")
        print(f"\n  Done. R@500={r500:.1%}" if r500 else "\n  Done.")

    print(f"  Registry updated: {h} → {name}")


if __name__ == "__main__":
    main()
