# MovieLens 1M Colab T4 Result

This result records a public Google Colab T4 run of the nanoGenRec MovieLens
path. It uses only redistributable MovieLens 1M ratings and metadata, builds
hashed hybrid semantic-ID features, trains the tiny NTP model, and evaluates
SID-constrained item recall.

## Command

```bash
python run.py public-movielens \
    --dataset ml-1m \
    --output_dir public_benchmarks/runs/ml-1m-colab-t4 \
    --min_rating 4.0 \
    --min_user_items 10 \
    --max_users 0 \
    --max_items_per_user 100 \
    --feature_source hybrid \
    --collab_window 5 \
    --clusters 64,64,64 \
    --feature_dim 128 \
    --kmeans_iters 5 \
    --kmeans_sample_size 8192 \
    --train_mode sliding \
    --min_context_items 2 \
    --embed_dim 128 \
    --n_heads 4 \
    --layers 3 \
    --batch_size 256 \
    --epochs 5 \
    --eval_samples 1000 \
    --beam_size 1000 \
    --max_seq_len 128 \
    --device cuda
```

## Setup

| Field | Value |
|-------|-------|
| Dataset | `ml-1m` |
| Device | Colab T4 / CUDA |
| Users | 5,950 |
| Items | 3,532 |
| Train examples | 348,363 |
| Eval examples | 5,950 |
| Eval samples | 1,000 |
| SID clusters | 64,64,64 |
| Feature source | hybrid title/genre + collaborative hashes |
| Train mode | sliding next-item prefixes |
| Model | dense NTP, 3 layers, embed_dim=128 |
| Epochs | 5 |
| Beam size | 1,000 |

## Training

| Epoch | Loss | PPL |
|-------|------|-----|
| 1 | 2.5869 | 13.29 |
| 2 | 2.1271 | 8.39 |
| 3 | 2.0437 | 7.72 |
| 4 | 1.9972 | 7.37 |
| 5 | 1.9634 | 7.12 |

## Evaluation

Summary row for paper consistency checks: `ml-1m` uses 5,950 users, 3,532 items, 348,363 train examples, and obtains R@10=10.5%, R@100=40.4%, R@500=72.5%, R@1000=85.2%.

| Metric | Value |
|--------|-------|
| item_recall@1 | 0.019 |
| item_recall@5 | 0.062 |
| item_recall@10 | 0.105 |
| item_recall@50 | 0.290 |
| item_recall@100 | 0.404 |
| item_recall@500 | 0.725 |
| item_recall@1000 | 0.852 |
| target_sid_found_rate | 89.9% (0.899) |

Simple public baselines on the same split are recorded in
[ml-1m-baselines.md](ml-1m-baselines.md). The nanoGenRec public path beats
global popularity at all reported cutoffs, while ItemKNN co-occurrence remains
stronger on this dense MovieLens setting.

## Interpretation

This is the first public GPU-scale reproducibility result in the repository.
It demonstrates that the open MovieLens path can move beyond smoke testing:
on a free Colab-class GPU, the framework runs a complete public generative
recommendation loop and reaches `item_recall@500=0.725` over 1,000 held-out
examples.

This result should still be read as an end-to-end reproducibility result, not
as a public leaderboard claim. The semantic IDs are built from lightweight
hashed public features rather than the production Qwen/Faiss tokenizer path.
