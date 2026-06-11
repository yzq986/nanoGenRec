# MovieLens 1M Public Baselines

These baselines use the same filtered MovieLens 1M split as the Colab-T4
nanoGenRec public run: `min_rating=4.0`, `min_user_items=10`,
`max_items_per_user=100`, final item as target, `seed=42`, and 1,000 sampled
evaluation users.

The baselines are intentionally simple and reproducible:

- `popularity`: global item popularity over training histories.
- `last_item_repeat`: reverse the user's context items.
- `itemknn`: item co-occurrence over public training histories with a window of
  5 and the 5 most recent context items as seeds.

## Command

```bash
python3 public_benchmarks/baselines.py \
    --dataset ml-1m \
    --output public_benchmarks/results/ml-1m-baselines.json \
    --min_rating 4.0 \
    --min_user_items 10 \
    --max_users 0 \
    --max_items_per_user 100 \
    --eval_samples 1000 \
    --beam_size 1000 \
    --seed 42
```

## Results

| Method | R@1 | R@5 | R@10 | R@50 | R@100 | R@500 | R@1000 |
|--------|-----|-----|------|------|-------|-------|--------|
| Popularity | 0.5% | 1.4% | 2.2% | 10.6% | 20.6% | 56.1% | 74.7% |
| Last item repeat | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| ItemKNN co-occurrence | 2.5% | 8.4% | 13.9% | 33.7% | 46.6% | 78.0% | 88.5% |
| nanoGenRec public path | 1.9% | 6.2% | 10.5% | 29.0% | 40.4% | 72.5% | 85.2% |

## Interpretation

The public nanoGenRec path beats the global popularity baseline, but the simple
ItemKNN co-occurrence baseline is stronger on MovieLens 1M. This is expected for
a small, dense collaborative-filtering dataset and reinforces the claim boundary:
the public MovieLens path is an end-to-end reproducibility check for the GR
framework, not a public leaderboard claim.
