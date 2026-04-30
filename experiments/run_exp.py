#!/usr/bin/env python3
"""Experiment runner with config registry and dedup.

Usage:
    # Single config
    python experiments/run_exp.py configs/exp-047.yaml [--check] [--no-smoke] [--force]

    # Multi-variant config (variants: list in yaml)
    python experiments/run_exp.py configs/exp-047.yaml [--only exp047-a] [--check]

    --check      : show similar past experiments for each variant and exit
    --only NAME  : run a single variant by name (useful for resuming)
    --no-smoke   : skip smoke test
    --force      : run even if hash already in registry
    --commit     : git add experiments/ && git commit && ./push.sh after all variants done
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

# Keys excluded from hash (paths / environment / runtime)
HASH_EXCLUDE = {
    "behavior_path", "output_dir", "preprocessed_dir",
    "n_gpus", "log", "dry_run", "eval_only",
    "shift_features", "seed",
}

# Keys that are multi-config metadata, not model params
META_KEYS = {"name", "description", "base", "variants"}


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
    """Merge _base.yaml + experiment yaml (no variants). Returns flat resolved config."""
    raw = load_yaml(yaml_path)
    base_name = raw.get("base", "_base.yaml")
    base = load_yaml(CONFIGS_DIR / base_name)
    resolved = {**base, **raw}
    for k in META_KEYS:
        resolved.pop(k, None)
    return resolved


def resolve_variants(yaml_path: Path) -> list[tuple[str, dict]]:
    """
    Expand variants list. Each variant = base_config + shared_keys + variant_overrides.
    Returns list of (name, resolved_dict).
    If no variants key, returns single entry using yaml name.
    """
    raw = load_yaml(yaml_path)
    base_name = raw.get("base", "_base.yaml")
    base = load_yaml(CONFIGS_DIR / base_name)

    # Shared = raw minus meta and variants
    shared = {k: v for k, v in raw.items() if k not in META_KEYS and k != "variants"}

    variants_raw = raw.get("variants")
    if not variants_raw:
        # Single config
        name = raw.get("name") or yaml_path.stem
        resolved = {**base, **shared}
        return [(name, resolved)]

    results = []
    for v in variants_raw:
        v = dict(v)
        name = v.pop("name")
        resolved = {**base, **shared, **v}
        results.append((name, resolved))
    return results


def config_hash(resolved: dict) -> str:
    hashable = {k: v for k, v in resolved.items() if k not in HASH_EXCLUDE}
    canonical = json.dumps(hashable, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def find_similar(registry: dict, resolved: dict, top_n: int = 5) -> list[dict]:
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
    print("  Similar past experiments:")
    print(f"    {'name':<32}  {'diffs':<50}  {'R@500':>6}  commit")
    print(f"    {'-'*32}  {'-'*50}  {'-'*6}  {'-'*8}")
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
        print(f"    {s['name']:<32}  {diff_str:<50}  {r500_str:>6}  {s['commit'][:8]}")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def register_experiment(name: str, resolved: dict, h: str):
    registry = load_registry()
    registry[h] = {
        "name": name,
        "config": {k: v for k, v in resolved.items() if k not in HASH_EXCLUDE},
        "results": {},
        "git_commit": get_git_commit(),
        "registered_at": datetime.now().isoformat(),
    }
    save_registry(registry)


def update_results(h: str, results: dict):
    registry = load_registry()
    if h in registry:
        registry[h]["results"] = results
        registry[h]["results_updated_at"] = datetime.now().isoformat()
        save_registry(registry)


def build_torchrun_cmd(resolved: dict, name: str, n_gpus: int, dry_run: bool = False) -> list[str]:
    ntp_data_name = resolved.get("ntp_data_name") or resolved["sid_cache_name"]
    ntp_data = str(NTP_DATA_ROOT / ntp_data_name)
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
    if resolved.get("embed_dim") and resolved["embed_dim"] != 256:
        cmd += ["--embed_dim", str(resolved["embed_dim"])]
    if resolved.get("n_transformer_layers"):
        cmd += ["--n_transformer_layers", str(resolved["n_transformer_layers"])]
    if resolved.get("n_experts") and resolved["n_experts"] != 8:
        cmd += ["--n_experts", str(resolved["n_experts"])]
    if resolved.get("top_k") and resolved["top_k"] != 2:
        cmd += ["--top_k", str(resolved["top_k"])]
    if resolved.get("expert_dim"):
        cmd += ["--expert_dim", str(resolved["expert_dim"])]
    if resolved.get("use_segment_emb"):
        cmd.append("--use_segment_emb")
    # New RoPE API
    if resolved.get("use_rope"):
        cmd.append("--use_rope")
        if resolved.get("rope_dims"):
            cmd += ["--rope_dims", str(resolved["rope_dims"])]
    # Legacy TO-RoPE API (backward compat for old configs)
    elif resolved.get("use_torope"):
        cmd += ["--use_torope", "--torope_time_split", str(resolved.get("torope_time_split", 0.5))]
        if resolved.get("torope_layer_split", 0.0) > 0:
            cmd += ["--torope_layer_split", str(resolved["torope_layer_split"])]
    if resolved.get("use_gate_attn"):
        cmd.append("--use_gate_attn")
    if dry_run:
        cmd.append("--dry_run")

    return cmd


def build_eval_cmd(name: str, n_gpus: int, n_recall: int = 1000) -> list[str]:
    return [
        "torchrun", f"--nproc_per_node={n_gpus}",
        "run.py", "eval-ntp",
        "--checkpoint", str(CKPT_DIR / name),
        "--n_recall", str(n_recall),
    ]


# ── Tokenizer (preprocess-sid) support ──────────────────────────────────────

TOKENIZER_HASH_EXCLUDE = {"behavior_path", "output_dir", "date_start", "date_end", "seed"}

def build_preprocess_sid_cmd(resolved: dict, name: str) -> list[str]:
    output_dir = str(SID_CACHE_ROOT / name)
    behavior_cache = "/mnt/workspace/gr-demo-behavior-cache"
    cmd = [
        "python", "run.py", "preprocess-sid",
        "--model",      resolved["model"],
        "--output_dir", output_dir,
        "--behavior_path", resolved.get("behavior_path", behavior_cache),
        "--date_start", resolved["date_start"],
        "--date_end",   resolved["date_end"],
        "--num_clusters", str(resolved["num_clusters"]),
        "--fsq_levels",   resolved["fsq_levels"],
        "--fsq_mlp_hidden", str(resolved["fsq_mlp_hidden"]),
        "--fsq_projection", resolved.get("fsq_projection", "mlp"),
    ]
    if resolved.get("fsq_epochs"):
        cmd += ["--fsq_epochs", str(resolved["fsq_epochs"])]
    return cmd


def compute_gini(counts):
    import numpy as np
    x = np.sort(np.array(counts, dtype=np.float64))
    n = len(x)
    total = x.sum()
    if total == 0 or n == 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2 * (idx * x).sum() / (n * total)) - (n + 1) / n)


def eval_sid_metrics(sid_dir: Path) -> dict:
    """Compute CR and Gini_d1/d2/d3 from a SID cache directory."""
    import numpy as np
    from collections import Counter
    sid_file = sid_dir / "semantic_ids.npy"
    if not sid_file.exists():
        return {}
    data = np.load(sid_file, allow_pickle=True).item()
    sids = list(data.values())
    n_items = len(sids)
    parts = [s.split("_") for s in sids]
    n_unique = len(Counter(sids))
    collision = 1.0 - n_unique / n_items
    ginis = []
    for depth in range(1, len(parts[0]) + 1):
        prefixes = ["_".join(p[:depth]) for p in parts]
        counts = list(Counter(prefixes).values())
        ginis.append(compute_gini(counts))
    return {
        "n_items": n_items,
        "n_unique_sids": n_unique,
        "collision_rate": collision,
        "gini_d1": ginis[0] if len(ginis) > 0 else None,
        "gini_d2": ginis[1] if len(ginis) > 1 else None,
        "gini_d3": ginis[2] if len(ginis) > 2 else None,
    }


def run_one_tokenizer(name: str, resolved: dict, force: bool) -> dict:
    """Run preprocess-sid + compute Gini/CR. Returns results dict."""
    h = config_hash(resolved)
    registry = load_registry()

    print(f"\n{'='*60}")
    print(f"  Variant: {name}  (hash={h})")

    similar = find_similar(registry, resolved)
    if similar:
        print_similar(similar)

    exact = [s for s in similar if s["n_diffs"] == 0]
    if exact and not force:
        cr = exact[0]["results"].get("collision_rate")
        print(f"\n  !! IDENTICAL config already in registry: {exact[0]['name']}")
        if cr is not None:
            print(f"     CR={cr:.2%} — skipping. Use --force to re-run.")
        else:
            print(f"     No results yet — skipping. Use --force to re-run.")
        return exact[0]["results"]

    sid_dir = SID_CACHE_ROOT / name
    if (sid_dir / "config.json").exists() and not force:
        print(f"\n  SID cache found, skipping preprocess.")
    else:
        register_experiment(name, resolved, h)
        cmd = build_preprocess_sid_cmd(resolved, name)
        print(f"\n  CMD: {' '.join(cmd)}\n")
        r = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if r.returncode != 0:
            print(f"\n  ERROR: preprocess-sid failed (exit {r.returncode})"); sys.exit(r.returncode)

    results = eval_sid_metrics(sid_dir)
    if results:
        update_results(h, results)
        print(f"  CR={results['collision_rate']:.2%}  Gini_d2={results.get('gini_d2', '?'):.4f}  n_items={results['n_items']:,}")
    return results


def print_tokenizer_summary(variant_results: list[tuple[str, dict]]):
    print(f"\n{'='*75}")
    print(f"  Tokenizer Results Summary")
    print(f"  {'Name':<35} {'N_items':>9} {'CR':>8} {'Gini_d1':>8} {'Gini_d2':>8} {'Gini_d3':>8}")
    print(f"  {'-'*35} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, res in variant_results:
        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else "  n/a"
        n = res.get("n_items", "?")
        n_s = f"{n:,}" if isinstance(n, int) else str(n)
        cr = res.get("collision_rate")
        cr_s = f"{cr:.2%}" if isinstance(cr, float) else "  n/a"
        print(f"  {name:<35} {n_s:>9} {cr_s:>8} {fmt(res.get('gini_d1')):>8} "
              f"{fmt(res.get('gini_d2')):>8} {fmt(res.get('gini_d3')):>8}")


def run_one(name: str, resolved: dict, n_gpus: int, force: bool, no_smoke: bool) -> dict:
    """Train + eval one variant. Returns results dict."""
    h = config_hash(resolved)
    registry = load_registry()

    print(f"\n{'='*60}")
    print(f"  Variant: {name}  (hash={h})")

    similar = find_similar(registry, resolved)
    if similar:
        print_similar(similar)

    exact = [s for s in similar if s["n_diffs"] == 0]
    if exact and not force:
        r500 = exact[0]["results"].get("item_recall@500")
        print(f"\n  !! IDENTICAL config already in registry: {exact[0]['name']}")
        if r500:
            print(f"     R@500={r500:.1%} — skipping. Use --force to re-run.")
        else:
            print(f"     No results yet — skipping. Use --force to re-run.")
        return exact[0]["results"]

    # Smoke test (dry_run=True, 2 steps)
    if not no_smoke:
        print(f"\n  >>> Smoke test...")
        smoke_cmd = build_torchrun_cmd(resolved, f"{name}-smoke", n_gpus, dry_run=True)
        r = subprocess.run(smoke_cmd, cwd=str(REPO_ROOT))
        if r.returncode != 0:
            print(f"  SMOKE FAILED"); sys.exit(r.returncode)
        import shutil
        smoke_dir = CKPT_DIR / f"{name}-smoke"
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir)
        print(f"  Smoke PASSED")

    ckpt_dir = CKPT_DIR / name
    if (ckpt_dir / "train_meta.json").exists() and not force:
        print(f"\n  Checkpoint found, skipping training.")
    else:
        register_experiment(name, resolved, h)
        train_cmd = build_torchrun_cmd(resolved, name, n_gpus)
        print(f"\n  CMD: {' '.join(train_cmd)}\n")
        r = subprocess.run(train_cmd, cwd=str(REPO_ROOT))
        if r.returncode != 0:
            print(f"\n  ERROR: training failed (exit {r.returncode})"); sys.exit(r.returncode)

    # Full eval
    print(f"\n  >>> Full eval (n_recall=1000)...")
    eval_cmd = build_eval_cmd(name, n_gpus)
    subprocess.run(eval_cmd, cwd=str(REPO_ROOT))

    # Read results
    meta_path = ckpt_dir / "train_meta.json"
    results = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        results = meta.get("eval", meta)
        update_results(h, results)

    return results


def print_summary(variant_results: list[tuple[str, dict]]):
    print(f"\n{'='*70}")
    print(f"  Results summary")
    print(f"  {'Name':<40} {'R@10':>6} {'R@500':>7} {'PPL':>8}")
    print(f"  {'-'*40} {'-'*6} {'-'*7} {'-'*8}")
    for name, res in variant_results:
        r10  = res.get("item_recall@10",  "?")
        r500 = res.get("item_recall@500", "?")
        ppl  = res.get("ppl", "?")
        r10_s  = f"{r10:.3f}"  if isinstance(r10,  float) else r10
        r500_s = f"{r500:.3f}" if isinstance(r500, float) else r500
        ppl_s  = f"{ppl:.1f}"  if isinstance(ppl,  float) else ppl
        print(f"  {name:<40} {r10_s:>6} {r500_s:>7} {ppl_s:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to experiment yaml (relative to cwd or absolute)")
    parser.add_argument("--check",   action="store_true", help="show similar exps and exit")
    parser.add_argument("--only",    default=None,        help="run only this variant name")
    parser.add_argument("--no-smoke",action="store_true")
    parser.add_argument("--force",   action="store_true", help="re-run even if hash exists")
    parser.add_argument("--commit",  action="store_true", help="git commit + push when done")
    parser.add_argument("--n-gpus",  type=int, default=None)
    args = parser.parse_args()

    yaml_path = Path(args.config)
    if not yaml_path.is_absolute():
        yaml_path = Path.cwd() / yaml_path

    raw = load_yaml(yaml_path)
    exp_name = raw.get("name") or yaml_path.stem

    variants = resolve_variants(yaml_path)
    if args.only:
        variants = [(n, r) for n, r in variants if n == args.only]
        if not variants:
            print(f"ERROR: variant '{args.only}' not found"); sys.exit(1)

    # phase may come from base YAML — read from first resolved variant
    phase = raw.get("phase") or (variants[0][1].get("phase") if variants else None) or "ntp"
    print(f"\nExperiment: {exp_name}  ({len(variants)} variant(s))  phase={phase}")

    if args.check:
        registry = load_registry()
        for name, resolved in variants:
            h = config_hash(resolved)
            print(f"\n  [{name}] hash={h}")
            similar = find_similar(registry, resolved)
            if similar:
                print_similar(similar)
        sys.exit(0)

    variant_results = []

    if phase == "tokenizer":
        for name, resolved in variants:
            results = run_one_tokenizer(name, resolved, force=args.force)
            variant_results.append((name, results))
        print_tokenizer_summary(variant_results)
    else:
        n_gpus = args.n_gpus
        if n_gpus is None:
            try:
                import torch
                n_gpus = max(1, torch.cuda.device_count())
            except Exception:
                n_gpus = 1
        print(f"  GPUs: {n_gpus}")

        for name, resolved in variants:
            results = run_one(name, resolved, n_gpus, force=args.force, no_smoke=args.no_smoke)
            variant_results.append((name, results))
        print_summary(variant_results)

    if args.commit:
        subprocess.run(["git", "add", "experiments/"], cwd=str(REPO_ROOT))
        if phase == "tokenizer":
            crs = [f"CR={r.get('collision_rate', 0):.2%}" for _, r in variant_results if r]
            msg = f"{exp_name} complete: {', '.join(crs)}"
        else:
            r500s = [f"{r.get('item_recall@500', 0):.1%}" for _, r in variant_results if r]
            msg = f"{exp_name} complete: {', '.join(r500s)}"
        subprocess.run(["git", "commit", "-m", msg,
                        "--trailer", "Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"],
                       cwd=str(REPO_ROOT))
        subprocess.run(["./push.sh"], cwd=str(REPO_ROOT))

    print(f"\nDone.")


if __name__ == "__main__":
    main()
