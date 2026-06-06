# MovieLens Latest Small Full CPU Result

This result verifies the public nanoGenRec loop on the full `ml-latest-small`
user set that passes filtering. It is still not intended as a competitive public
benchmark, but it is larger than the minimal smoke run.

## Command

```bash
python3 run.py public-movielens \
    --dataset ml-latest-small \
    --output_dir public_benchmarks/runs/ml-latest-small-full-cpu \
    --epochs 5 \
    --max_users 0 \
    --clusters 64,64,64 \
    --embed_dim 96 \
    --n_heads 4 \
    --layers 2 \
    --batch_size 64 \
    --eval_samples 500 \
    --beam_size 100 \
    --max_seq_len 128 \
    --kmeans_iters 10
```

## Setup

| Field | Value |
|-------|-------|
| Dataset | `ml-latest-small` |
| Device | CPU |
| Users | 603 |
| Items | 6,298 |
| Train examples | 603 |
| Eval examples | 603 |
| Eval samples | 500 |
| SID clusters | 64,64,64 |
| Model | dense NTP, 2 layers, embed_dim=96 |
| Epochs | 5 |
| Beam size | 100 |

## Training

| Epoch | Loss | PPL |
|-------|------|-----|
| 1 | 4.2725 | 71.70 |
| 2 | 4.1927 | 66.20 |
| 3 | 4.1205 | 61.59 |
| 4 | 4.0580 | 57.86 |
| 5 | 4.0093 | 55.11 |

## Evaluation

| Metric | Value |
|--------|-------|
| item_recall@1 | 0.000 |
| item_recall@5 | 0.006 |
| item_recall@10 | 0.008 |
| item_recall@50 | 0.032 |
| target_sid_found_rate | 0.052 |

## Interpretation

The run confirms that a larger public CPU setting can complete the full
nanoGenRec path: MovieLens download, CPU Semantic ID construction, tiny NTP
training, and SID-constrained beam-search evaluation over real item IDs.

The absolute recall is low because this path deliberately avoids the production
Qwen/Faiss tokenizer and GPU-scale model. It is evidence of public
reproducibility, not evidence of public-dataset SOTA.
