#!/usr/bin/env python3
"""Public MovieLens benchmark path for nanoGenRec.

This script is intentionally small and self-contained. It verifies the public
reproducibility loop without requiring private behavior logs, Faiss, or GPU
resources. For strict parity with the repository "How It Works" path, use
``--feature_source qwen`` to build MovieLens item embeddings with Qwen3.

  download/load public MovieLens data
  -> build item embeddings from Qwen3 or lightweight public features
  -> build CPU semantic IDs
  -> train a tiny dense NTPModel
  -> run a lightweight GRPO-style reward-alignment stage
  -> run SID-constrained beam-search recall

The default dataset is ``synthetic`` for fast tests. Use ``ml-latest-small`` for
a real public smoke run, and ``ml-1m``/``ml-20m`` when compute is available.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from ntp.model import NTPModel, SIDTrie, constrained_beam_search
from rl.dpo import compute_sid_logprobs_batch
from rl.grpo import grpo_loss


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


def build_text_item_features(movies: dict[int, tuple[str, str]], dim: int) -> tuple[list[int], np.ndarray]:
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


def build_collab_item_features(
    movie_ids: list[int],
    seqs: dict[int, list[int]],
    dim: int,
    window: int,
) -> np.ndarray:
    """Hashed item co-occurrence features from public behavior sequences."""
    item_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    feats = np.zeros((len(movie_ids), dim), dtype=np.float32)
    for items in seqs.values():
        items = [mid for mid in items if mid in item_to_idx]
        for pos, mid in enumerate(items):
            row = item_to_idx[mid]
            lo = max(0, pos - window)
            hi = min(len(items), pos + window + 1)
            for j in range(lo, hi):
                if j == pos:
                    continue
                ctx = items[j]
                h = stable_hash(str(ctx))
                feats[row, h % dim] += 1.0 / max(abs(j - pos), 1)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.maximum(norms, 1e-6)


def build_qwen_item_features(
    movies: dict[int, tuple[str, str]],
    model_name: str,
    batch_size: int,
    cache_dir: Path,
    max_length: int,
    device_arg: str,
) -> tuple[list[int], np.ndarray]:
    """Encode public MovieLens item text with Qwen3 embeddings."""
    from model.embedders import Qwen3TextEmbedder

    movie_ids = sorted(movies)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "movielens_qwen_embeddings.npy"
    cached = {}
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True).item()
        print(f"loaded qwen embedding cache: {len(cached):,} items", flush=True)

    texts = []
    missing_ids = []
    for mid in movie_ids:
        if mid in cached:
            continue
        title, genres = movies[mid]
        genre_text = ", ".join(g for g in genres.split("|") if g and g != "(no genres listed)")
        texts.append(f"Movie title: {title}\nGenres: {genre_text}")
        missing_ids.append(mid)

    if missing_ids:
        if device_arg == "auto":
            qwen_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            qwen_device = device_arg
        dtype = torch.float16 if qwen_device == "cuda" else torch.float32
        embedder = Qwen3TextEmbedder(
            model_name,
            device=qwen_device,
            max_length=max_length,
            torch_dtype=dtype,
        )
        print(f"qwen encoding items={len(missing_ids):,} device={qwen_device}", flush=True)
        for start in range(0, len(missing_ids), batch_size):
            batch_ids = missing_ids[start:start + batch_size]
            batch_texts = texts[start:start + batch_size]
            emb = embedder.encode(batch_texts).detach().cpu().float().numpy()
            for mid, vec in zip(batch_ids, emb):
                cached[mid] = vec.astype(np.float32)
            np.save(cache_path, cached)
            print(
                f"qwen encoded {min(start + batch_size, len(missing_ids)):,}/"
                f"{len(missing_ids):,}",
                flush=True,
            )

    features = np.array([cached[mid] for mid in movie_ids], dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return movie_ids, features / np.maximum(norms, 1e-6)


def build_item_features(
    movies: dict[int, tuple[str, str]],
    seqs: dict[int, list[int]],
    dim: int,
    feature_source: str,
    collab_window: int,
    args=None,
) -> tuple[list[int], np.ndarray]:
    if feature_source == "qwen":
        if args is None:
            raise ValueError("qwen feature source requires CLI args")
        cache_dir = Path(args.qwen_cache_dir) if args.qwen_cache_dir else Path(args.output_dir)
        return build_qwen_item_features(
            movies,
            args.qwen_model,
            args.qwen_batch_size,
            cache_dir,
            args.qwen_max_length,
            args.qwen_device,
        )

    movie_ids, text_features = build_text_item_features(movies, dim)
    if feature_source == "text":
        return movie_ids, text_features

    collab_features = build_collab_item_features(movie_ids, seqs, dim, collab_window)
    if feature_source == "collab":
        return movie_ids, collab_features
    if feature_source == "hybrid":
        features = np.concatenate([text_features, collab_features], axis=1)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return movie_ids, features / np.maximum(norms, 1e-6)
    raise ValueError(f"unknown feature_source: {feature_source}")


def kmeans(
    x: np.ndarray,
    k: int,
    iters: int,
    seed: int,
    sample_size: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    init = rng.choice(n, size=k, replace=False)
    centroids = x[init].copy()
    train_x = x
    if sample_size and sample_size < n:
        train_x = x[rng.choice(n, size=max(k, sample_size), replace=False)]
    assignments = np.zeros(train_x.shape[0], dtype=np.int64)

    for _ in range(iters):
        dists = ((train_x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_assignments = dists.argmin(axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for ci in range(k):
            mask = assignments == ci
            if mask.any():
                centroids[ci] = train_x[mask].mean(axis=0)
            else:
                centroids[ci] = train_x[rng.integers(0, train_x.shape[0])]
    full_dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return full_dists.argmin(axis=1), centroids


def build_semantic_ids(
    movies: dict[int, tuple[str, str]],
    seqs: dict[int, list[int]],
    n_clusters: tuple[int, int, int],
    feature_dim: int,
    feature_source: str,
    collab_window: int,
    kmeans_iters: int,
    kmeans_sample_size: int,
    seed: int,
    args=None,
) -> tuple[dict[int, tuple[int, int, int]], list[int]]:
    movie_ids, features = build_item_features(
        movies, seqs, feature_dim, feature_source, collab_window, args)
    residual = features.copy()
    layers = []
    actual_clusters = []
    for li, k in enumerate(n_clusters):
        print(f"kmeans layer {li}: k={k} dim={residual.shape[1]}", flush=True)
        assign, centroids = kmeans(residual, k, kmeans_iters, seed + li, kmeans_sample_size)
        layers.append(assign)
        actual_clusters.append(int(centroids.shape[0]))
        residual = residual - centroids[assign]
    sid = {
        mid: (int(layers[0][i]), int(layers[1][i]), int(layers[2][i]))
        for i, mid in enumerate(movie_ids)
    }
    return sid, actual_clusters


def split_train_eval(
    seqs: dict[int, list[int]],
    sid: dict[int, tuple[int, ...]],
    max_items: int,
    train_mode: str,
    min_context_items: int,
):
    train_examples = []
    alignment_examples = []
    eval_examples = []
    for uid, items in seqs.items():
        items = [mid for mid in items if mid in sid]
        if len(items) <= min_context_items:
            continue
        if max_items:
            items = items[-max_items:]

        eval_context = items[:-1]
        eval_target = items[-1]
        eval_tokens = [tok for mid in eval_context for tok in sid[mid]]
        if len(eval_tokens) >= 2:
            eval_examples.append({
                "uid": uid,
                "context": eval_tokens,
                "target_mid": eval_target,
                "target_sid": sid[eval_target],
            })

        if train_mode == "last":
            train_items = items[:-1]
            tokens = [tok for mid in train_items for tok in sid[mid]]
            if len(tokens) >= 2:
                train_examples.append(tokens)
            if len(train_items) > min_context_items:
                align_context = train_items[:-1]
                align_target = train_items[-1]
                align_tokens = [tok for mid in align_context for tok in sid[mid]]
                if len(align_tokens) >= 2:
                    alignment_examples.append({
                        "uid": uid,
                        "context": align_tokens,
                        "target_mid": align_target,
                        "target_sid": sid[align_target],
                    })
        elif train_mode == "sliding":
            # Leave the final item for evaluation. Every earlier prefix becomes
            # one LM sequence that ends at the next-item SID target.
            for end in range(min_context_items + 1, len(items)):
                prefix_plus_target = items[:end]
                if prefix_plus_target[-1] == eval_target:
                    continue
                tokens = [tok for mid in prefix_plus_target for tok in sid[mid]]
                if len(tokens) >= 2:
                    train_examples.append(tokens)
                align_context = items[:end - 1]
                align_target = items[end - 1]
                align_tokens = [tok for mid in align_context for tok in sid[mid]]
                if len(align_tokens) >= 2:
                    alignment_examples.append({
                        "uid": uid,
                        "context": align_tokens,
                        "target_mid": align_target,
                        "target_sid": sid[align_target],
                    })
        else:
            raise ValueError(f"unknown train_mode: {train_mode}")
    return train_examples, alignment_examples, eval_examples


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
        print(
            f"epoch {epoch + 1}/{args.epochs}: "
            f"loss={mean_loss:.4f} ppl={math.exp(min(mean_loss, 20)):.2f}",
            flush=True,
        )
    return model


def batch_alignment_examples(
    examples: list[dict],
    batch_size: int,
    seed: int,
) -> Iterable[list[dict]]:
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[i] for i in order[start:start + batch_size]]


def _make_alignment_batch(
    batch: list[dict],
    sid_values: list[tuple[int, int, int]],
    args,
    device: torch.device,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack target and negative SID candidates for one public GRPO update."""
    group_size = max(2, args.rl_group_size)
    ctx_window = max(1, args.max_seq_len - 3)
    contexts = [ex["context"][-ctx_window:] for ex in batch]
    max_ctx = max(len(x) for x in contexts)

    flat_contexts = []
    flat_lengths = []
    flat_sids = []
    rewards = []
    offsets = [0]

    for ex, ctx in zip(batch, contexts):
        target = tuple(int(t) for t in ex["target_sid"])
        candidates = [target]
        while len(candidates) < group_size:
            neg = rng.choice(sid_values)
            if neg != target:
                candidates.append(neg)

        for cand in candidates:
            padded = ctx + [0] * (max_ctx - len(ctx))
            flat_contexts.append(padded)
            flat_lengths.append(len(ctx))
            flat_sids.append(cand)
            rewards.append(1.0 if cand == target else 0.0)
        offsets.append(offsets[-1] + len(candidates))

    return (
        torch.tensor(flat_contexts, dtype=torch.long, device=device),
        torch.tensor(flat_lengths, dtype=torch.long, device=device),
        torch.tensor(flat_sids, dtype=torch.long, device=device),
        torch.tensor(rewards, dtype=torch.float32, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
    )


def align_with_public_grpo(
    model: NTPModel,
    alignment_examples: list[dict],
    sid: dict[int, tuple[int, ...]],
    args,
    device: torch.device,
) -> dict[str, float]:
    """Lightweight public reward-alignment stage.

    The production stack supports SP-DPO, RF-DPO, GRPO, and ECPO on generated
    candidates. The public MovieLens path uses the same SID log-probability and
    GRPO loss primitives, but keeps the reward source redistributable: the held
    out next SID in each training prefix receives reward 1, sampled valid SIDs
    receive reward 0.
    """
    if args.rl_steps <= 0:
        return {
            "rl_enabled": False,
            "rl_steps_completed": 0,
        }
    if not alignment_examples:
        return {
            "rl_enabled": True,
            "rl_steps_completed": 0,
            "rl_skipped_reason": "no_alignment_examples",
        }

    sid_values = sorted({tuple(int(t) for t in toks) for toks in sid.values()})
    if len(sid_values) < 2:
        return {
            "rl_enabled": True,
            "rl_steps_completed": 0,
            "rl_skipped_reason": "not_enough_sid_candidates",
        }

    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.rl_lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed + 10_000)
    model.train()

    losses = []
    reward_means = []
    clip_fracs = []
    steps = 0
    while steps < args.rl_steps:
        for batch in batch_alignment_examples(
            alignment_examples, args.rl_batch_size, args.seed + 20_000 + steps
        ):
            ctx, lengths, cand_sids, rewards, offsets = _make_alignment_batch(
                batch, sid_values, args, device, rng)
            policy_lp = compute_sid_logprobs_batch(
                model, ctx, lengths, cand_sids, n_layers=3, max_chunk=args.rl_max_chunk)
            with torch.no_grad():
                ref_lp = compute_sid_logprobs_batch(
                    ref_model, ctx, lengths, cand_sids, n_layers=3, max_chunk=args.rl_max_chunk)
            loss, diag = grpo_loss(
                policy_lp,
                ref_lp,
                rewards,
                offsets,
                eps=args.rl_eps,
                rank_norm=args.rl_rank_norm,
                return_diagnostics=True,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.rl_grad_clip)
            opt.step()

            steps += 1
            losses.append(float(loss.detach().cpu()))
            reward_means.append(float(rewards.mean().detach().cpu()))
            clip_fracs.append(float(diag.get("clip_fraction", 0.0)))
            if steps == 1 or steps % args.rl_log_every == 0 or steps == args.rl_steps:
                print(
                    f"rl step {steps}/{args.rl_steps}: "
                    f"loss={losses[-1]:.4f} reward_mean={reward_means[-1]:.3f} "
                    f"clip_fraction={clip_fracs[-1]:.3f}",
                    flush=True,
                )
            if steps >= args.rl_steps:
                break

    return {
        "rl_enabled": True,
        "rl_algo": "public_grpo_exact_sid_reward",
        "rl_steps_completed": steps,
        "rl_group_size": args.rl_group_size,
        "rl_batch_size": args.rl_batch_size,
        "rl_loss_last": losses[-1] if losses else 0.0,
        "rl_loss_mean": sum(losses) / max(len(losses), 1),
        "rl_reward_mean": sum(reward_means) / max(len(reward_means), 1),
        "rl_clip_fraction_mean": sum(clip_fracs) / max(len(clip_fracs), 1),
    }


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

    recall_ks = sorted({1, 5, 10, 50, 100, 500, args.beam_size})
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
                for mid in sid_to_items.get(sid_str, []):
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


def build_sid_to_items(
    sid: dict[int, tuple[int, ...]],
    seqs: dict[int, list[int]],
) -> dict[str, list[int]]:
    """Map each SID to items sorted by public-train popularity tie-breaker."""
    freq = Counter(mid for items in seqs.values() for mid in items)
    grouped: dict[str, list[int]] = defaultdict(list)
    for mid, toks in sid.items():
        grouped["_".join(str(t) for t in toks)].append(mid)
    return {
        sid_str: sorted(items, key=lambda mid: (-freq.get(mid, 0), mid))
        for sid_str, items in grouped.items()
    }


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
            "feature_source": args.feature_source,
            "train_mode": args.train_mode,
            "embed_dim": args.embed_dim,
            "layers": args.layers,
            "epochs": args.epochs,
            "rl_steps": args.rl_steps,
            "rl_group_size": args.rl_group_size,
            "rl_batch_size": args.rl_batch_size,
            "device": args.device,
            "purpose": "public reproducibility path with optional GRPO-style alignment",
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
    parser.add_argument("--feature_source", choices=["text", "collab", "hybrid", "qwen"], default="hybrid")
    parser.add_argument("--collab_window", type=int, default=5)
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--qwen_batch_size", type=int, default=16)
    parser.add_argument("--qwen_cache_dir", default="")
    parser.add_argument("--qwen_max_length", type=int, default=256)
    parser.add_argument("--qwen_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--clusters", default="64,64,64")
    parser.add_argument("--kmeans_iters", type=int, default=10)
    parser.add_argument("--kmeans_sample_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--rl_steps", type=int, default=0)
    parser.add_argument("--rl_lr", type=float, default=1e-5)
    parser.add_argument("--rl_batch_size", type=int, default=8)
    parser.add_argument("--rl_group_size", type=int, default=4)
    parser.add_argument("--rl_eps", type=float, default=0.2)
    parser.add_argument("--rl_grad_clip", type=float, default=1.0)
    parser.add_argument("--rl_max_chunk", type=int, default=64)
    parser.add_argument("--rl_log_every", type=int, default=10)
    parser.add_argument("--rl_rank_norm", action="store_true", default=True)
    parser.add_argument("--no_rl_rank_norm", action="store_false", dest="rl_rank_norm")
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--train_mode", choices=["last", "sliding"], default="sliding")
    parser.add_argument("--min_context_items", type=int, default=2)
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

    print(f"dataset={args.dataset} users={len(seqs):,} items={len(movies):,}", flush=True)
    sid, actual_clusters = build_semantic_ids(
        movies, seqs, clusters, args.feature_dim, args.feature_source,
        args.collab_window, args.kmeans_iters, args.kmeans_sample_size, args.seed, args)
    train_examples, alignment_examples, eval_examples = split_train_eval(
        seqs, sid, args.max_items_per_user, args.train_mode, args.min_context_items)
    print(f"train_examples={len(train_examples):,} eval_users={len(eval_examples):,} "
          f"alignment_examples={len(alignment_examples):,} "
          f"clusters={actual_clusters} feature_source={args.feature_source} "
          f"train_mode={args.train_mode}", flush=True)
    if not train_examples or not eval_examples:
        raise RuntimeError("No train/eval examples after filtering; relax min_user_items or max_users.")

    sid_to_items = build_sid_to_items(sid, seqs)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    model = train_tiny_ntp(train_examples, actual_clusters, args, device)
    rl_metrics = align_with_public_grpo(model, alignment_examples, sid, args, device)
    metrics = evaluate(model, eval_examples, sid_to_items, args, device)
    metrics.update({
        "dataset": args.dataset,
        "n_users": len(seqs),
        "n_items": len(movies),
        "n_train_examples": len(train_examples),
        "n_alignment_examples": len(alignment_examples),
        "n_eval_examples": len(eval_examples),
        "feature_source": args.feature_source,
        "train_mode": args.train_mode,
    })
    metrics.update(rl_metrics)
    print(json.dumps(metrics, indent=2))
    write_artifacts(Path(args.output_dir), sid, actual_clusters, metrics, args)


if __name__ == "__main__":
    main()
