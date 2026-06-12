# Public Benchmarks

This directory contains redistributable benchmark paths that do not depend on
private behavior data or Faiss.

The main public path is a Colab T4 MovieLens reproduction. Its purpose is to
verify that nanoGenRec can run the strict "How It Works" loop on public data:

```text
MovieLens ratings and metadata
  -> Qwen3 item embeddings
  -> CPU KMeans Semantic IDs
  -> tiny NTP training
  -> lightweight GRPO-style reward alignment
  -> SID-constrained beam-search evaluation
  -> metrics.json + semantic_ids.npy
```

Fast CPU/hash-feature settings are kept as developer smoke tests. They are
useful for CI and local debugging, but they are not the strict public framework
run. This directory is a reproducibility path, not a public SOTA claim.

## Quick Smoke

Fast synthetic developer test:

```bash
python run.py public-movielens \
    --dataset synthetic \
    --output_dir /tmp/nanogenrec-public-smoke \
    --epochs 1 \
    --max_users 120 \
    --clusters 16,16,16 \
    --embed_dim 32 \
    --layers 1 \
    --rl_steps 1 \
    --rl_batch_size 2 \
    --rl_group_size 4 \
    --eval_samples 20 \
    --beam_size 10
```

Real public MovieLens developer smoke:

```bash
python run.py public-movielens \
    --dataset ml-latest-small \
    --output_dir public_benchmarks/runs/ml-latest-small-smoke \
    --epochs 1 \
    --max_users 200 \
    --clusters 16,16,16 \
    --embed_dim 32 \
    --layers 1 \
    --rl_steps 1 \
    --rl_batch_size 2 \
    --rl_group_size 4 \
    --eval_samples 20 \
    --beam_size 10
```

## Larger Runs

The same script supports:

```bash
python run.py public-movielens --dataset ml-1m
python run.py public-movielens --dataset ml-20m
```

For a larger public reproducibility run, increase `--max_users`, `--clusters`,
`--embed_dim`, `--layers`, `--epochs`, `--beam_size`, and `--eval_samples`.
CPU is enough for smoke runs; free T4/L4/A100 time can be used for larger runs.

## Colab T4 Path

For a free GPU run, open
[nanogenrec_colab.ipynb](https://colab.research.google.com/github/yzq986/nanoGenRec/blob/master/public_benchmarks/nanogenrec_colab.ipynb)
in Google Colab and select
`Runtime` -> `Change runtime type` -> `T4 GPU`.

The recommended strict T4 run is:

```bash
python run.py public-movielens \
    --dataset ml-latest-small \
    --output_dir public_benchmarks/runs/ml-latest-small-qwen-rl-t4 \
    --min_rating 5.0 \
    --min_user_items 5 \
    --max_users 0 \
    --max_items_per_user 100 \
    --feature_source qwen \
    --qwen_device cuda \
    --qwen_batch_size 16 \
    --clusters 16,16,16 \
    --kmeans_iters 5 \
    --kmeans_sample_size 2048 \
    --train_mode sliding \
    --min_context_items 2 \
    --embed_dim 96 \
    --n_heads 4 \
    --layers 2 \
    --batch_size 128 \
    --epochs 8 \
    --rl_steps 50 \
    --rl_batch_size 8 \
    --rl_group_size 8 \
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

On a free T4, the practical bottleneck is usually Qwen embedding download,
beam-evaluation latency, and session stability. Start with `ml-latest-small`
and `beam_size=1000`; then try `ml-1m` with `epochs=5`,
`clusters=64,64,64`, `embed_dim=128`, `layers=3`, `rl_steps=100`, and
`eval_samples=1000`. Save outputs to Drive when using Colab because free
runtimes can disconnect.

## Current Public Results

A checked-in full CPU/hash-feature result is available at
[results/ml-latest-small-full-cpu.md](results/ml-latest-small-full-cpu.md).
The smaller smoke run is retained at
[results/ml-latest-small-smoke.md](results/ml-latest-small-smoke.md).
The strict Qwen+RL Colab T4 GPU result is available at
[results/ml-1m-qwen-rl-t4.md](results/ml-1m-qwen-rl-t4.md).
The earlier hybrid-feature Colab T4 GPU result is available at
[results/ml-1m-colab-t4.md](results/ml-1m-colab-t4.md).
Simple public baselines are recorded at
[results/ml-1m-baselines.md](results/ml-1m-baselines.md) for transparency and
debugging, but they are not part of the headline public proof.

Summary:

| Dataset | Device | Users | Items | Model | Eval samples | item_recall@50 | item_recall@500 | item_recall@1000 | target SID found |
|---------|--------|-------|-------|-------|--------------|----------------|-----------------|------------------|------------------|
| `ml-latest-small` | CPU | 603 | 6,298 | dense 2-layer, dim=96 | 500 | 0.032 | - | - | 0.052 |
| `ml-1m` | Colab T4 | 5,950 | 3,532 | Qwen SID + dense 3-layer + public GRPO | 1,000 | 0.279 | 0.722 | 0.860 | 0.886 |
| `ml-1m` | Colab T4 | 5,950 | 3,532 | hybrid SID + dense 3-layer | 1,000 | 0.290 | 0.725 | 0.852 | 0.899 |

These checked-in results validate the smoke-test and free-GPU scale paths. They
should not be read as competitive public leaderboard claims. The strict Qwen+RL
row is the release result that follows the repository "How It Works" path; the
hybrid row is retained as an earlier lightweight-feature reproducibility run.

## Outputs

The output directory contains:

| File | Meaning |
|------|---------|
| `semantic_ids.npy` | Movie ID to 3-token SID mapping. |
| `metrics.json` | Recall and run metadata. |
| `meta.json` | Dataset/model/config metadata. |

## Notes

- The strict public path uses Qwen3 text embeddings and numpy KMeans. The
  developer smoke path uses hashed title/genre features and is intentionally
  weaker.
- The metric is full-loop recall from SID-constrained generation over real item
  IDs. Tiny smoke settings may produce low or zero recall; that is acceptable
  for CI-style path validation.
