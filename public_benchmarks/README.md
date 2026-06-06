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

## Current Smoke Result

A checked-in CPU smoke result is available at
[results/ml-latest-small-smoke.md](results/ml-latest-small-smoke.md).

Summary:

| Dataset | Device | Users | Items | Model | item_recall@50 | target SID found |
|---------|--------|-------|-------|-------|----------------|------------------|
| `ml-latest-small` | CPU | 500 | 5,667 | dense 2-layer, dim=64 | 0.010 | 0.030 |

This result validates the public path. It should not be read as a competitive
benchmark because it uses weak title/genre hash features, a tiny model, and only
100 eval samples.

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
