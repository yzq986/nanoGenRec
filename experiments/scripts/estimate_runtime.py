#!/usr/bin/env python3
"""Estimate NTP training runtime from historical experiment data.

Scans experiments/ntp_checkpoints/*/train_meta.json, fits a throughput model,
and predicts wall time for a given (active_params, total_tokens, gpus) config.

Usage:
    python experiments/scripts/estimate_runtime.py \\
        --active_params 17388544 --total_tokens 132000000 --gpus 8

    python experiments/scripts/estimate_runtime.py \\
        --active_params 17388544 --date_range_days 14 --gpus 8

    python experiments/scripts/estimate_runtime.py --budget 30 --list-history
"""

import argparse
import glob
import json
import math
import os
import sys

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments", "ntp_checkpoints",
)

TOKENS_PER_DAY = {
    7: 65_000_000,
    14: 130_000_000,
    31: 220_000_000,
    62: 440_000_000,
    90: 553_000_000,
}

SAFETY_MULTIPLIER = 1.2


def load_history():
    """Load (name, active_params, total_tokens, throughput, wall_time, world_size) from checkpoints."""
    records = []
    pattern = os.path.join(CHECKPOINT_DIR, "*", "train_meta.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            meta = json.load(f)
        train = meta.get("train", {})
        params = train.get("n_active_params")
        tokens = train.get("total_tokens")
        throughput = train.get("throughput_tok_per_s")
        wall = train.get("wall_time_s")
        gpus = train.get("world_size")
        if all(v is not None for v in [params, tokens, throughput, wall, gpus]):
            if tokens < 500_000:
                continue
            name = os.path.basename(os.path.dirname(path))
            records.append({
                "name": name,
                "active_params": params,
                "total_tokens": tokens,
                "throughput": throughput,
                "wall_time_s": wall,
                "world_size": gpus,
            })
    return records


def fit_throughput_model(records, target_gpus=8):
    """Fit log(throughput) = a * log(params) + b using least squares."""
    filtered = [r for r in records if r["world_size"] == target_gpus]
    if len(filtered) < 3:
        filtered = records

    xs = [math.log(r["active_params"]) for r in filtered]
    ys = [math.log(r["throughput"]) for r in filtered]
    n = len(xs)

    if n < 2:
        return None, None

    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return None, None

    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n
    return a, b


def predict_throughput(a, b, active_params):
    return math.exp(a * math.log(active_params) + b)


def estimate_tokens_from_days(days):
    """Interpolate total tokens from date range in days."""
    if days in TOKENS_PER_DAY:
        return TOKENS_PER_DAY[days]

    sorted_days = sorted(TOKENS_PER_DAY.keys())
    if days <= sorted_days[0]:
        return int(TOKENS_PER_DAY[sorted_days[0]] * days / sorted_days[0])
    if days >= sorted_days[-1]:
        return int(TOKENS_PER_DAY[sorted_days[-1]] * days / sorted_days[-1])

    for i in range(len(sorted_days) - 1):
        d0, d1 = sorted_days[i], sorted_days[i + 1]
        if d0 <= days <= d1:
            t0, t1 = TOKENS_PER_DAY[d0], TOKENS_PER_DAY[d1]
            frac = (days - d0) / (d1 - d0)
            return int(t0 + frac * (t1 - t0))
    return int(days * TOKENS_PER_DAY[14] / 14)


def find_closest(records, active_params, total_tokens):
    """Find the historically closest experiment."""
    best = None
    best_dist = float("inf")
    for r in records:
        dp = abs(math.log(r["active_params"]) - math.log(active_params))
        dt = abs(math.log(r["total_tokens"]) - math.log(total_tokens))
        dist = dp + 0.3 * dt
        if dist < best_dist:
            best_dist = dist
            best = r
    return best


def main():
    parser = argparse.ArgumentParser(description="Estimate NTP training runtime")
    parser.add_argument("--active_params", type=int, help="Number of active parameters")
    parser.add_argument("--total_tokens", type=int, help="Total training tokens")
    parser.add_argument("--date_range_days", type=int, help="Date range in days (alternative to --total_tokens)")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs (default: 8)")
    parser.add_argument("--budget", type=float, default=30.0, help="Time budget in minutes (default: 30)")
    parser.add_argument("--list-history", action="store_true", help="List all historical experiments")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    records = load_history()

    if not records:
        print("ERROR: No historical data found in", CHECKPOINT_DIR)
        sys.exit(1)

    if args.list_history:
        print(f"{'Name':<35} {'Params':>12} {'Tokens':>12} {'Throughput':>12} {'Wall(s)':>10} {'GPUs':>5}")
        print("-" * 90)
        for r in records:
            print(f"{r['name']:<35} {r['active_params']:>12,} {r['total_tokens']:>12,} "
                  f"{r['throughput']:>12,.1f} {r['wall_time_s']:>10.1f} {r['world_size']:>5}")
        return

    if args.active_params is None:
        parser.error("--active_params is required")

    if args.total_tokens is None and args.date_range_days is None:
        parser.error("Either --total_tokens or --date_range_days is required")

    total_tokens = args.total_tokens
    if total_tokens is None:
        total_tokens = estimate_tokens_from_days(args.date_range_days)

    a, b = fit_throughput_model(records, target_gpus=args.gpus)
    if a is None:
        print("ERROR: Not enough data to fit throughput model")
        sys.exit(1)

    predicted_throughput = predict_throughput(a, b, args.active_params)
    estimated_wall = total_tokens / predicted_throughput
    conservative_wall = estimated_wall * SAFETY_MULTIPLIER
    budget_seconds = args.budget * 60
    within_budget = conservative_wall <= budget_seconds

    closest = find_closest(records, args.active_params, total_tokens)

    result = {
        "active_params": args.active_params,
        "total_tokens": total_tokens,
        "gpus": args.gpus,
        "predicted_throughput_tok_s": round(predicted_throughput, 1),
        "estimated_wall_s": round(estimated_wall, 1),
        "conservative_wall_s": round(conservative_wall, 1),
        "budget_s": budget_seconds,
        "within_budget": within_budget,
        "closest_match": closest["name"] if closest else None,
        "closest_wall_s": closest["wall_time_s"] if closest else None,
        "model_coefficients": {"a": round(a, 4), "b": round(b, 4)},
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 60)
    print("  Runtime Estimation")
    print("=" * 60)
    print(f"  Active params:   {args.active_params:>14,}")
    print(f"  Total tokens:    {total_tokens:>14,}")
    if args.date_range_days:
        print(f"    (from {args.date_range_days}d date range)")
    print(f"  GPUs:            {args.gpus:>14}")
    print()
    print(f"  Predicted throughput: {predicted_throughput:>10,.0f} tok/s")
    print(f"  Estimated wall time:  {estimated_wall:>10.1f} s  ({estimated_wall/60:.1f} min)")
    print(f"  Conservative (1.2x):  {conservative_wall:>10.1f} s  ({conservative_wall/60:.1f} min)")
    print(f"  Budget:               {budget_seconds:>10.0f} s  ({args.budget:.0f} min)")
    print()

    if within_budget:
        print("  Status: WITHIN BUDGET")
    else:
        overage = conservative_wall - budget_seconds
        print(f"  Status: OVER BUDGET by {overage:.0f}s ({overage/60:.1f} min)")

    if closest:
        print()
        print(f"  Closest historical match: {closest['name']}")
        print(f"    params={closest['active_params']:,}  tokens={closest['total_tokens']:,}")
        print(f"    actual wall time: {closest['wall_time_s']:.1f}s ({closest['wall_time_s']/60:.1f} min)")

    print("=" * 60)


if __name__ == "__main__":
    main()
