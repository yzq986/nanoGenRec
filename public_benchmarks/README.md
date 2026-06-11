# Public Benchmarks

This directory contains redistributable benchmark paths that do not depend on
private behavior data, Qwen embeddings, Faiss, or GPU resources.

The first path is a CPU-friendly MovieLens smoke benchmark. Its purpose is to
verify that nanoGenRec can run the complete public loop:

```text
MovieLens ratings and metadata
  -> CPU title/genre Semantic IDs
  -> tiny NTP training
  -> SID-constrained beam-search evaluation
  -> metrics.json + semantic_ids.npy
```

It is a reproducibility path, not a public SOTA claim.

## Quick Smoke

Fast synthetic test:

```bash
python run.py public-movielens \
    --dataset synthetic \
    --output_dir /tmp/nanogenrec-public-smoke \
    --epochs 1 \
    --max_users 120 \
    --clusters 16,16,16 \
    --embed_dim 32 \
    --layers 1 \
    --eval_samples 20 \
    --beam_size 10
```

Real public MovieLens smoke:

```bash
python run.py public-movielens \
    --dataset ml-latest-small \
    --output_dir public_benchmarks/runs/ml-latest-small-smoke \
    --epochs 1 \
    --max_users 200 \
    --clusters 16,16,16 \
    --embed_dim 32 \
    --layers 1 \
    --eval_samples 20 \
    --beam_size 10
```

## Larger Runs

The same script supports:

```bash
python run.py public-movielens --dataset ml-1m
python run.py public-movielens --dataset ml-20m
```

For a stronger public benchmark, increase `--max_users`, `--clusters`,
`--embed_dim`, `--layers`, `--epochs`, `--beam_size`, and `--eval_samples`.
CPU is enough for smoke runs; free T4/L4/A100 time can be used for larger runs.

## Colab T4 Path

For a free GPU run, open
[nanogenrec_colab.ipynb](nanogenrec_colab.ipynb) in Google Colab and select
`Runtime` -> `Change runtime type` -> `T4 GPU`.

The recommended first T4 run is:

```bash
python run.py public-movielens \
    --dataset ml-latest-small \
    --output_dir public_benchmarks/runs/ml-latest-small-colab-t4 \
    --min_rating 5.0 \
    --min_user_items 5 \
    --max_users 0 \
    --max_items_per_user 100 \
    --feature_source hybrid \
    --collab_window 5 \
    --clusters 16,16,16 \
    --feature_dim 96 \
    --kmeans_iters 5 \
    --kmeans_sample_size 2048 \
    --train_mode sliding \
    --min_context_items 2 \
    --embed_dim 96 \
    --n_heads 4 \
    --layers 2 \
    --batch_size 128 \
    --epochs 8 \
    --eval_samples 1000 \
    --beam_size 1000 \
    --max_seq_len 128 \
    --device cuda
```

Estimated free T4 scale:

| Dataset | Rating filter | Approx. users | Approx. items | Recommended use |
|---------|---------------|---------------|---------------|-----------------|
| `ml-latest-small` | `--min_rating 5.0` | hundreds | a few thousand | First credible public GPU result. |
| `ml-latest-small` | `--min_rating 4.0` | hundreds | 6k+ | Denser sanity check after the first run. |
| `ml-1m` | `--min_rating 4.0` | thousands | 3k+ | Next public-scale run if the session is stable. |
| `ml-20m` | any | tens of thousands | 20k+ | Not recommended on free Colab without reducing users/eval. |

On a free T4, the practical bottleneck is usually session stability and beam
evaluation latency rather than model memory. Start with `ml-latest-small` and
`beam_size=1000`; then try `ml-1m` with `epochs=5`, `clusters=64,64,64`,
`embed_dim=128`, `layers=3`, and `eval_samples=1000`. Save outputs to Drive
when using Colab because free runtimes can disconnect.

## Current Public Results

A checked-in full CPU result is available at
[results/ml-latest-small-full-cpu.md](results/ml-latest-small-full-cpu.md).
The smaller smoke run is retained at
[results/ml-latest-small-smoke.md](results/ml-latest-small-smoke.md).
The first Colab T4 GPU result is available at
[results/ml-1m-colab-t4.md](results/ml-1m-colab-t4.md).

Summary:

| Dataset | Device | Users | Items | Model | Eval samples | item_recall@50 | item_recall@500 | item_recall@1000 | target SID found |
|---------|--------|-------|-------|-------|--------------|----------------|-----------------|------------------|------------------|
| `ml-latest-small` | CPU | 603 | 6,298 | dense 2-layer, dim=96 | 500 | 0.032 | - | - | 0.052 |
| `ml-1m` | Colab T4 | 5,950 | 3,532 | dense 3-layer, dim=128 | 1,000 | 0.290 | 0.725 | 0.852 | 0.899 |

These results validate the public path at smoke-test and free-GPU scale. They
should not be read as competitive public leaderboard claims because they use
lightweight hashed public features rather than the production Qwen/Faiss
tokenizer path.

## Outputs

The output directory contains:

| File | Meaning |
|------|---------|
| `semantic_ids.npy` | Movie ID to 3-token SID mapping. |
| `metrics.json` | Recall and run metadata. |
| `meta.json` | Dataset/model/config metadata. |

## Notes

- The tokenizer uses hashed title/genre features and numpy KMeans, so it is
  intentionally weaker than the production Qwen/Faiss SID path.
- The metric is full-loop recall from SID-constrained generation over real item
  IDs. Tiny smoke settings may produce low or zero recall; that is acceptable
  for CI-style path validation.
