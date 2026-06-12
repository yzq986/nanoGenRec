# MovieLens 1M Qwen+RL Colab T4 Result

This result records the strict public nanoGenRec path on Google Colab T4. It
uses only redistributable MovieLens 1M ratings and metadata, builds Qwen3 item
embeddings from public movie text, constructs CPU residual-KMeans Semantic IDs,
trains the tiny NTP model, runs a lightweight GRPO-style reward-alignment
stage, and evaluates SID-constrained item recall.

## Command

```bash
python run.py public-movielens \
    --dataset ml-1m \
    --output_dir public_benchmarks/runs/ml-1m-qwen-rl-t4 \
    --min_rating 4.0 \
    --min_user_items 10 \
    --max_users 0 \
    --max_items_per_user 100 \
    --feature_source qwen \
    --qwen_device cuda \
    --qwen_batch_size 16 \
    --clusters 64,64,64 \
    --kmeans_iters 5 \
    --kmeans_sample_size 8192 \
    --train_mode sliding \
    --min_context_items 2 \
    --embed_dim 128 \
    --n_heads 4 \
    --layers 3 \
    --batch_size 256 \
    --epochs 5 \
    --rl_steps 100 \
    --rl_batch_size 8 \
    --rl_group_size 8 \
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
| Alignment examples | 348,363 |
| Eval examples | 5,950 |
| Eval samples | 1,000 |
| SID clusters | 64,64,64 |
| Feature source | Qwen3 item embeddings, dim=1024 |
| Train mode | sliding next-item prefixes |
| Model | dense NTP, 3 layers, embed_dim=128 |
| Epochs | 5 |
| RL stage | public GRPO exact-SID reward, 100 steps, group=8, batch=8 |
| Beam size | 1,000 |

## Training

| Epoch | Loss | PPL |
|-------|------|-----|
| 1 | 2.7257 | 15.27 |
| 2 | 2.2124 | 9.14 |
| 3 | 2.1227 | 8.35 |
| 4 | 2.0725 | 7.95 |
| 5 | 2.0365 | 7.66 |

## Alignment

| Step | Loss | Reward mean | Clip fraction |
|------|------|-------------|---------------|
| 1 | 1.0846 | 0.125 | 0.875 |
| 10 | 0.8353 | 0.125 | 0.875 |
| 20 | 0.8230 | 0.125 | 0.875 |
| 30 | 1.2591 | 0.125 | 0.859 |
| 40 | 0.3487 | 0.125 | 0.906 |
| 50 | 0.1537 | 0.125 | 0.922 |
| 60 | 0.2518 | 0.125 | 0.953 |
| 70 | 0.2524 | 0.125 | 0.828 |
| 80 | 0.6556 | 0.125 | 0.922 |
| 90 | 0.1752 | 0.125 | 0.953 |
| 100 | 0.1987 | 0.125 | 0.859 |

Mean RL loss is 0.4199 and mean clip fraction is 0.8997. The high clip fraction
shows that this public alignment stage is a runnable framework check rather
than a tuned public recommender recipe.

## Evaluation

Summary row for paper consistency checks: `ml-1m-qwen-rl` uses 5,950 users, 3,532 items, 348,363 train examples, 348,363 alignment examples, and obtains R@10=10.0%, R@100=38.4%, R@500=72.2%, R@1000=86.0%.

| Metric | Value |
|--------|-------|
| item_recall@1 | 0.022 |
| item_recall@5 | 0.067 |
| item_recall@10 | 0.100 |
| item_recall@50 | 0.279 |
| item_recall@100 | 0.384 |
| item_recall@500 | 0.722 |
| item_recall@1000 | 0.860 |
| target_sid_found_rate | 88.6% (0.886) |

Simple public baselines on the same split are recorded in
[ml-1m-baselines.md](ml-1m-baselines.md). The strict Qwen+RL run beats global
popularity at all reported cutoffs and reaches the strongest checked-in
R@1000 among the public nanoGenRec runs. ItemKNN co-occurrence remains stronger
at R@10--R@500 on this dense MovieLens setting.

## Interpretation

This is the first checked-in public result that follows the repository
"How It Works" path: public ratings/metadata -> Qwen3 item embeddings -> CPU
Semantic IDs -> NTP training -> reward alignment -> SID-constrained full-recall
evaluation. It validates the executable framework loop on public data and a
free-GPU environment. It is not a tuned MovieLens leaderboard claim.
