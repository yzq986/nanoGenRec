#!/usr/bin/env python3
"""Simple public baselines for the MovieLens reproducibility path."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from public_benchmarks.movielens_cpu import filter_sequences, load_movielens


def build_eval_examples(seqs: dict[int, list[int]], max_items: int) -> tuple[list[dict], list[list[int]]]:
    examples = []
    train_histories = []
    for uid, items in seqs.items():
        if max_items:
            items = items[-max_items:]
        if len(items) <= 2:
            continue
        context = items[:-1]
        examples.append({"uid": uid, "context": context, "target_mid": items[-1]})
        train_histories.append(context)
    return examples, train_histories


def recall_metrics(name: str, examples: list[dict], ks: list[int], rank_fn) -> dict[str, float]:
    hits = {k: 0 for k in ks}
    for ex in examples:
        candidates = rank_fn(ex)
        for k in ks:
            hits[k] += int(ex["target_mid"] in candidates[:k])
    n = max(len(examples), 1)
    return {f"{name}_recall@{k}": hits[k] / n for k in ks}


def build_itemknn(train_histories: list[list[int]], window: int) -> dict[int, Counter]:
    cooccurrence: dict[int, Counter] = defaultdict(Counter)
    for items in train_histories:
        for pos, mid in enumerate(items):
            lo = max(0, pos - window)
            hi = min(len(items), pos + window + 1)
            for j in range(lo, hi):
                if j == pos:
                    continue
                cooccurrence[mid][items[j]] += 1.0 / max(abs(j - pos), 1)
    return cooccurrence


def evaluate_baselines(args) -> dict[str, object]:
    data = load_movielens(args.dataset, Path(args.data_dir))
    seqs = filter_sequences(data, args.min_rating, args.min_user_items, args.max_users)
    eval_examples, train_histories = build_eval_examples(seqs, args.max_items_per_user)
    random.Random(args.seed).shuffle(eval_examples)
    sample = eval_examples[:args.eval_samples]
    ks = sorted({1, 5, 10, 50, 100, 500, args.beam_size})

    popularity = Counter(mid for items in train_histories for mid in items)
    popularity_rank = [
        mid for mid, _count in sorted(popularity.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    itemknn = build_itemknn(train_histories, args.itemknn_window)

    def itemknn_rank(ex):
        scores: Counter = Counter()
        for recency, mid in enumerate(reversed(ex["context"][-args.itemknn_recent:]), start=1):
            for cand, score in itemknn.get(mid, {}).most_common(args.itemknn_candidates_per_seed):
                scores[cand] += score / recency
        ranked = [mid for mid, _score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
        seen = set(ranked)
        ranked.extend(mid for mid in popularity_rank if mid not in seen)
        return ranked

    metrics: dict[str, object] = {
        "dataset": args.dataset,
        "n_users": len(seqs),
        "n_eval_examples": len(eval_examples),
        "n_eval_samples": len(sample),
        "min_rating": args.min_rating,
        "min_user_items": args.min_user_items,
        "max_items_per_user": args.max_items_per_user,
        "seed": args.seed,
        "protocol": "same filtered users and final-item target as public-movielens",
    }
    metrics.update(recall_metrics("popularity", sample, ks, lambda _ex: popularity_rank))
    metrics.update(recall_metrics("last_item_repeat", sample, ks, lambda ex: list(reversed(ex["context"]))))
    metrics.update(recall_metrics("itemknn", sample, ks, itemknn_rank))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="MovieLens public baseline evaluator")
    parser.add_argument("--dataset", choices=["ml-latest-small", "ml-1m", "ml-20m"], default="ml-1m")
    parser.add_argument("--data_dir", default="public_benchmarks/data")
    parser.add_argument("--output", default="")
    parser.add_argument("--min_rating", type=float, default=4.0)
    parser.add_argument("--min_user_items", type=int, default=10)
    parser.add_argument("--max_users", type=int, default=0)
    parser.add_argument("--max_items_per_user", type=int, default=100)
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument("--beam_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--itemknn_window", type=int, default=5)
    parser.add_argument("--itemknn_recent", type=int, default=5)
    parser.add_argument("--itemknn_candidates_per_seed", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_baselines(args)
    print(json.dumps(metrics, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
