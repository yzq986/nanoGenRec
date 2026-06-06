# MovieLens Latest Small CPU Smoke Result

This result verifies the public nanoGenRec loop on a real redistributable dataset.
It is not intended as a competitive public benchmark.

## Command

```bash
python3 run.py public-movielens \
    --dataset ml-latest-small \
    --output_dir public_benchmarks/runs/ml-latest-small-smoke \
    --epochs 2 \
    --max_users 500 \
    --clusters 32,32,32 \
    --embed_dim 64 \
    --n_heads 4 \
    --layers 2 \
    --batch_size 64 \
    --eval_samples 100 \
    --beam_size 50 \
    --max_seq_len 96 \
    --kmeans_iters 5
```

## Setup

| Field | Value |
|-------|-------|
| Dataset | `ml-latest-small` |
| Device | CPU |
| Users | 500 |
| Items | 5,667 |
| Train examples | 500 |
| Eval examples | 500 |
| Eval samples | 100 |
| SID clusters | 32,32,32 |
| Model | dense NTP, 2 layers, embed_dim=64 |
| Epochs | 2 |

## Result

| Metric | Value |
|--------|-------|
| Final train loss | 3.5338 |
| Final train PPL | 34.25 |
| item_recall@1 | 0.000 |
| item_recall@5 | 0.000 |
| item_recall@10 | 0.000 |
| item_recall@50 | 0.010 |
| target_sid_found_rate | 0.030 |

## Interpretation

The run confirms that the public path can download MovieLens, build CPU Semantic
IDs, train a tiny NTP model, and run SID-constrained full-recall evaluation
without private data or GPU resources. The recall numbers are intentionally
reported as smoke-test evidence only; the tokenizer uses weak title/genre hash
features and the model is deliberately tiny.
