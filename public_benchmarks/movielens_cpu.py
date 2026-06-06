#!/usr/bin/env python3
"""CPU-friendly public MovieLens benchmark path for nanoGenRec.

This script is intentionally small and self-contained. It verifies the public
reproducibility loop without requiring private behavior logs, Qwen embeddings,
Faiss, or GPU resources:

  download/load public MovieLens data
  -> build CPU semantic IDs from title/genre features
  -> train a tiny dense NTPModel
  -> run SID-constrained beam-search recall

The default dataset is ``synthetic`` for fast tests. Use ``ml-latest-small`` for
a real public smoke run, and ``ml-1m``/``ml-20m`` when compute is available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from ntp.model import NTPModel, SIDTrie, constrained_beam_search


DATASETS = {
    "ml-latest-small": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "inner": "ml-latest-small",
        "ratings": "ratings.csv",
        "movies": "movies.csv",
        "format": "csv",
    },
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "inner": "ml-1m",
        "ratings": "ratings.dat",
        "movies": "movies.dat",
        "format": "dat",
    },
    "ml-20m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "inner": "ml-20m",
        "ratings": "ratings.csv",
        "movies": "movies.csv",
        "format": "csv",
    },
}


@dataclass
class MovieLensData:
    movies: dict[int, tuple[str, str]]
    user_sequences: dict[int, list[tuple[int, float, int]]]


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def tokenize_text(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]


def download_dataset(name: str, data_dir: Path) -> Path:
    spec = DATASETS[name]
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / spec["inner"]
    if extracted.exists():
        return extracted

    archive = data_dir / f"{spec['inner']}.zip"
    if not archive.exists():
        print(f"Downloading {name} from {spec['url']}")
        urllib.request.urlretrieve(spec["url"], archive)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    return extracted


def load_movielens(name: str, data_dir: Path) -> MovieLensData:
    root = download_dataset(name, data_dir)
    spec = DATASETS[name]
    movies: dict[int, tuple[str, str]] = {}
    user_sequences: dict[int, list[tuple[int, float, int]]] = defaultdict(list)

    if spec["format"] == "csv":
        with (root / spec["movies"]).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                movies[int(row["movieId"])] = (row["title"], row["genres"])
        with (root / spec["ratings"]).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid = int(row["userId"])
                mid = int(row["movieId"])
                rating = float(row["rating"])
                ts = int(row["timestamp"])
                if mid in movies:
                    user_sequences[uid].append((mid, rating, ts))
    else:
        with (root / spec["movies"]).open(encoding="latin-1") as f:
            for line in f:
                mid, title, genres = line.rstrip("\n").split("::")
                movies[int(mid)] = (title, genres)
        with (root / spec["ratings"]).open(encoding="latin-1") as f:
            for line in f:
                uid, mid, rating, ts = line.rstrip("\n").split("::")
                mid_i = int(mid)
                if mid_i in movies:
                    user_sequences[int(uid)].append((mid_i, float(rating), int(ts)))

    for seq in user_sequences.values():
        seq.sort(key=lambda x: x[2])
    return MovieLensData(movies=movies, user_sequences=dict(user_sequences))


def synthetic_movielens(n_users: int = 240, n_items: int = 120, seed: int = 42) -> MovieLensData:
    rng = random.Random(seed)
    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller"]
    movies = {}
    for mid in range(1, n_items + 1):
        g = genres[mid % len(genres)]
        movies[mid] = (f"Synthetic {g} Movie {mid} ({1990 + mid % 30})", g)

    user_sequences: dict[int, list[tuple[int, float, int]]] = {}
    for uid in range(1, n_users + 1):
        pref = rng.randrange(len(genres))
        length = rng.randint(8, 28)
        seq = []
        for t in range(length):
            if rng.random() < 0.75:
                candidates = [m for m in movies if m % len(genres) == pref]
            else:
                candidates = list(movies)
            mid = rng.choice(candidates)
            rating = 4.0 if mid % len(genres) == pref else 3.0
            seq.append((mid, rating, 1_700_000_000 + uid * 1000 + t))
        user_sequences[uid] = seq
    return MovieLensData(movies=movies, user_sequences=user_sequences)


def filter_sequences(
    data: MovieLensData,
    min_rating: float,
    min_user_items: int,
    max_users: int,
) -> dict[int, list[int]]:
    seqs = {}
    for uid, events in sorted(data.user_sequences.items()):
        items = [mid for mid, rating, _ts in events if rating >= min_rating]
        deduped = []
        for mid in items:
            if not deduped or deduped[-1] != mid:
                deduped.append(mid)
        if len(deduped) >= min_user_items:
            seqs[uid] = deduped
        if max_users and len(seqs) >= max_users:
            break
    return seqs


def build_item_features(movies: dict[int, tuple[str, str]], dim: int) -> tuple[list[int], np.ndarray]:
    movie_ids = sorted(movies)
    feats = np.zeros((len(movie_ids), dim), dtype=np.float32)
    for i, mid in enumerate(movie_ids):
        title, genres = movies[mid]
        tokens = tokenize_text(title)
        tokens.extend(f"genre:{g.lower()}" for g in genres.split("|") if g and g != "(no genres listed)")
        year = re.search(r"\((\d{4})\)", title)
        if year:
            tokens.append(f"year:{year.group(1)[:3]}0s")
        for tok in tokens:
            h = stable_hash(tok)
            idx = h % dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            feats[i, idx] += sign
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return movie_ids, feats / np.maximum(norms, 1e-6)


def kmeans(x: np.ndarray, k: int, iters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    init = rng.choice(n, size=k, replace=False)
    centroids = x[init].copy()
    assignments = np.zeros(n, dtype=np.int64)

    for _ in range(iters):
        dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_assignments = dists.argmin(axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for ci in range(k):
            mask = assignments == ci
            if mask.any():
                centroids[ci] = x[mask].mean(axis=0)
            else:
                centroids[ci] = x[rng.integers(0, n)]
    return assignments, centroids


def build_semantic_ids(
    movies: dict[int, tuple[str, str]],
    n_clusters: tuple[int, int, int],
    feature_dim: int,
    kmeans_iters: int,
    seed: int,
) -> tuple[dict[int, tuple[int, int, int]], list[int]]:
    movie_ids, features = build_item_features(movies, feature_dim)
    residual = features.copy()
    layers = []
    actual_clusters = []
    for li, k in enumerate(n_clusters):
        assign, centroids = kmeans(residual, k, kmeans_iters, seed + li)
        layers.append(assign)
        actual_clusters.append(int(centroids.shape[0]))
        residual = residual - centroids[assign]
    sid = {
        mid: (int(layers[0][i]), int(layers[1][i]), int(layers[2][i]))
        for i, mid in enumerate(movie_ids)
    }
    return sid, actual_clusters


def split_train_eval(seqs: dict[int, list[int]], sid: dict[int, tuple[int, ...]], max_items: int):
    train_examples = []
    eval_examples = []
    for uid, items in seqs.items():
        items = [mid for mid in items if mid in sid]
        if len(items) < 3:
            continue
        if max_items:
            items = items[-max_items:]
        train_items = items[:-1]
        target = items[-1]
        tokens = [tok for mid in train_items for tok in sid[mid]]
        if len(tokens) >= 2:
            train_examples.append(tokens)
            eval_examples.append({"uid": uid, "context": tokens, "target_mid": target, "target_sid": sid[target]})
    return train_examples, eval_examples


def batch_iter(examples: list[list[int]], batch_size: int, seed: int) -> Iterable[list[list[int]]]:
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[i] for i in order[start:start + batch_size]]


def train_tiny_ntp(
    examples: list[list[int]],
    n_clusters_per_layer: list[int],
    args,
    device: torch.device,
) -> NTPModel:
    model = NTPModel(
        n_clusters_per_layer=n_clusters_per_layer,
        n_sid_layers=3,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_transformer_layers=args.layers,
        dropout=args.dropout,
        use_moe=False,
        max_seq_len=args.max_seq_len + 3,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    for epoch in range(args.epochs):
        losses = []
        for batch in batch_iter(examples, args.batch_size, args.seed + epoch):
            max_len = min(max(len(x) for x in batch), args.max_seq_len)
            if max_len < 2:
                continue
            padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            lengths = []
            for i, seq in enumerate(batch):
                seq = seq[-max_len:]
                lengths.append(len(seq))
                padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            inp = padded[:, :-1]
            tgt = padded[:, 1:]
            mask = torch.zeros_like(tgt, dtype=torch.bool)
            for i, length in enumerate(lengths):
                mask[i, :max(length - 1, 0)] = True
            loss = model(inp, packed_targets=tgt, packed_mask=mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = sum(losses) / max(len(losses), 1)
        print(f"epoch {epoch + 1}/{args.epochs}: loss={mean_loss:.4f} ppl={math.exp(min(mean_loss, 20)):.2f}")
    return model


def evaluate(
    model: NTPModel,
    eval_examples: list[dict],
    sid_to_items: dict[str, set],
    args,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    trie = SIDTrie(sid_to_items, n_layers=3)
    sample = eval_examples[:]
    random.Random(args.seed).shuffle(sample)
    sample = sample[:args.eval_samples]

    recall_ks = [1, 5, 10, 50]
    hits = {k: 0 for k in recall_ks}
    sid_found = 0
    with torch.no_grad():
        for i, ex in enumerate(sample):
            ctx_window = max(1, args.max_seq_len - 3)
            ctx = torch.tensor(ex["context"][-ctx_window:], dtype=torch.long, device=device).unsqueeze(0)
            target_sid = "_".join(str(t) for t in ex["target_sid"])
            beams, _scores, _ = constrained_beam_search(model, ctx, trie, beam_size=args.beam_size)
            candidates = []
            seen = set()
            found = False
            for bi in range(beams.size(1)):
                sid_str = "_".join(str(int(t)) for t in beams[0, bi])
                if sid_str == target_sid:
                    found = True
                for mid in sid_to_items.get(sid_str, set()):
                    if mid not in seen:
                        candidates.append(mid)
                        seen.add(mid)
            sid_found += int(found)
            for k in recall_ks:
                hits[k] += int(ex["target_mid"] in candidates[:k])
            if args.verbose_eval and i < 5:
                print(f"eval sample {i}: target={ex['target_mid']} candidates[:5]={candidates[:5]}")

    n = max(len(sample), 1)
    out = {f"item_recall@{k}": hits[k] / n for k in recall_ks}
    out["target_sid_found_rate"] = sid_found / n
    out["n_eval_samples"] = len(sample)
    return out


def write_artifacts(output_dir: Path, sid: dict[int, tuple[int, ...]], n_clusters: list[int], metrics: dict, args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_ids = {mid: "_".join(str(t) for t in toks) for mid, toks in sid.items()}
    np.save(output_dir / "semantic_ids.npy", semantic_ids)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "n_clusters_per_layer": n_clusters,
            "embed_dim": args.embed_dim,
            "layers": args.layers,
            "epochs": args.epochs,
            "device": args.device,
            "purpose": "CPU public reproducibility smoke path",
        }, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="CPU MovieLens nanoGenRec public smoke benchmark")
    parser.add_argument("--dataset", choices=["synthetic", *DATASETS.keys()], default="ml-latest-small")
    parser.add_argument("--data_dir", default="public_benchmarks/data")
    parser.add_argument("--output_dir", default="public_benchmarks/runs/movielens_cpu")
    parser.add_argument("--min_rating", type=float, default=4.0)
    parser.add_argument("--min_user_items", type=int, default=5)
    parser.add_argument("--max_users", type=int, default=2000)
    parser.add_argument("--max_items_per_user", type=int, default=50)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--clusters", default="64,64,64")
    parser.add_argument("--kmeans_iters", type=int, default=10)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose_eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset == "synthetic":
        data = synthetic_movielens(seed=args.seed)
    else:
        data = load_movielens(args.dataset, Path(args.data_dir))

    seqs = filter_sequences(data, args.min_rating, args.min_user_items, args.max_users)
    used_items = sorted({mid for items in seqs.values() for mid in items})
    movies = {mid: data.movies[mid] for mid in used_items if mid in data.movies}
    clusters = tuple(int(x) for x in args.clusters.split(","))
    if len(clusters) != 3:
        raise ValueError("--clusters must contain exactly three comma-separated integers")

    print(f"dataset={args.dataset} users={len(seqs):,} items={len(movies):,}")
    sid, actual_clusters = build_semantic_ids(movies, clusters, args.feature_dim, args.kmeans_iters, args.seed)
    train_examples, eval_examples = split_train_eval(seqs, sid, args.max_items_per_user)
    print(f"train_users={len(train_examples):,} eval_users={len(eval_examples):,} clusters={actual_clusters}")
    if not train_examples or not eval_examples:
        raise RuntimeError("No train/eval examples after filtering; relax min_user_items or max_users.")

    sid_to_items: dict[str, set] = defaultdict(set)
    for mid, toks in sid.items():
        sid_to_items["_".join(str(t) for t in toks)].add(mid)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = train_tiny_ntp(train_examples, actual_clusters, args, device)
    metrics = evaluate(model, eval_examples, dict(sid_to_items), args, device)
    metrics.update({
        "dataset": args.dataset,
        "n_users": len(seqs),
        "n_items": len(movies),
        "n_train_examples": len(train_examples),
        "n_eval_examples": len(eval_examples),
    })
    print(json.dumps(metrics, indent=2))
    write_artifacts(Path(args.output_dir), sid, actual_clusters, metrics, args)


if __name__ == "__main__":
    main()
