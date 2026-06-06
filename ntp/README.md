# ntp/

[English](README.md) | [Chinese](README.zh.md)

Next-token prediction recommender over Semantic ID sequences.

The NTP module trains Transformer + MoE models that read user behavior histories encoded as SID tokens and generate the next item SID under a constrained beam search.

## Files

| File | Purpose |
|------|---------|
| `model.py` | `NTPModel`, MoE Transformer, TO-RoPE, side features, KV-cache inference. |
| `train.py` | DDP training entry point, unified sequences, SFT and joint NTP+DPO losses. |
| `eval.py` | Evaluation-only utilities, constrained beam search, SIDTrie decoding. |
| `preprocess.py` | Converts behavior sequences into SID-token shards. |
| `features.py` | Side feature definitions such as time gaps, action levels, timestamps, and segments. |
| `baseline.py` | Non-neural baselines such as popularity and co-occurrence. |

## Model Tiers

| Tier | embed_dim | Layers | Experts | top_k | Active Params | Status |
|------|-----------|--------|---------|-------|---------------|--------|
| S-tier | 256 | 6 | 8 | 2 | ~17.5M | validated |
| M-tier | 512 | 8 | 8 | 2 | ~71.6M | validated, R@500=70.2% |
| L-tier | 512 | 12 | 16 | 2 | ~101.1M | validated, RL starting point |

## Current Full-Eval Baselines

| Config | R@500 | PPL | Source |
|--------|-------|-----|--------|
| M-tier bare, 0.6B SID | 70.2% | 18.54 | EXP-043 |
| M-tier, 4B SID | 70.4% | 16.55 | EXP-043 |
| L-tier with validated options | 64.1% | 20.7 | EXP-047 |

Use [experiments/logs/ntp/README.md](../experiments/logs/ntp/README.md) for the current phase summary and experiment lineage.

## Usage

```bash
# Preprocess behavior into SID-token shards
python run.py preprocess-ntp \
    --sid_cache experiments/sid_cache/exp049-0.6b-nc8192-h128 \
    --output_dir experiments/ntp_data/exp049-0.6b-nc8192-h128 \
    --date_start 2026-03-18 \
    --date_end 2026-03-31 \
    --n_workers 64

# Train with torchrun
PYTHONPATH=. torchrun --nproc_per_node=8 run.py train-ntp \
    --config experiments/configs/exp-047.yaml

# Preferred experiment runner
python experiments/run_exp.py experiments/configs/exp-047.yaml --no-smoke --commit

# Full evaluation for reported numbers
PYTHONPATH=. torchrun --nproc_per_node=8 run.py eval-ntp \
    --checkpoint experiments/ntp_checkpoints/<name> \
    --n_recall 1000
```

## Side Features

All side features flow through `side_features: dict[str, Tensor]`.

| Feature | Injection | Meaning |
|---------|-----------|---------|
| `time_gaps` | embedding add | Bucketed time gap between events. |
| `action_levels` | embedding add | Behavior intensity level. |
| `timestamps` | TO-RoPE | Continuous hour timestamp used in Q/K rotation. |
| `segment_emb` | embedding add | User behavior segment marker. |

The single embedding entry point is `NTPModel.embed_with_features`. Do not manually rebuild token embedding plus feature embedding in callers.

## Data Contract

`preprocess.py` writes shards consumed by `train.py`. For each new feature, verify the full path:

| Stage | Required Check |
|-------|----------------|
| Preprocess | `save_shard` and `load_shard` store and restore the feature. |
| Sequence build | `build_unified_sequences` fills nonzero values. |
| Training | `side_features_lists` passes the key into the model. |
| Evaluation | `eval.py` forwards the same key into beam search. |
| Generation | `constrained_beam_search` carries the feature during generated steps. |

Train/eval feature mismatch invalidates comparisons. This is the highest-risk class of NTP bug.

## Known Pitfalls

- Inline eval during training uses a restricted candidate set and is only a health check.
- Full comparison numbers must use `run.py eval-ntp --n_recall 1000`.
- Timestamps can silently become all zeros if any stage filters non-`embed_add` features.
- `semantic_ids.npy` is a mapping from item ID to SID string; SIDTrie construction must use values, not keys.
