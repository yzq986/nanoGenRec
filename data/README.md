# data/

[English](README.md) | [Chinese](README.zh.md)

Data loading, export, embedding synchronization, and distributed encoding utilities.

This module connects raw item/behavior data to the tokenizer and NTP pipelines. It supports both local and remote paths, but downstream experiments should consume stable cache directories under `experiments/`.

## Files

| File | Purpose |
|------|---------|
| `loaders.py` | Shared S3/local loading and export helpers. |
| `encode_distributed.py` | Multi-GPU text embedding with `torchrun`. |
| `export_content.py` | PySpark export for exposed item text and image URLs. |
| `export_behavior.py` | PySpark export for user behavior bitmaps. |
| `sync_embeddings.py` | Embedding cache synchronization helper. |
| `migrate_shards.py` | Utility for shard migration and compatibility updates. |

## Usage

```bash
# Distributed embedding from the repo root
PYTHONPATH=. torchrun --nproc_per_node=8 data/encode_distributed.py \
    --model qwen3-0.6b

# CLI wrapper for sync tasks
python run.py sync-embeddings --help

# Download model assets when needed
python run.py download-model --help
```

## Data Contracts

| Dataset | Required Fields | Used By |
|---------|-----------------|---------|
| Item content | item ID, text, optional image URL | Embedding and tokenizer training. |
| Behavior events | user ID, item ID, timestamp, action bitmap | NTP preprocessing and behavior metrics. |
| Embedding cache | item ID, dense vector, metadata | Tokenizer training and proxy metrics. |
| SID cache | item ID -> SID mapping | NTP preprocessing and constrained decoding. |

NTP data windows must be compatible with SID caches: every behavior item used for training/evaluation should be covered by the SID cache.

## Distributed Encoding

`encode_distributed.py` is designed for large item sets:

- rank 0 coordinates model download and cache merge;
- all ranks encode disjoint shards;
- already encoded content IDs are skipped;
- CUDA OOM retries reduce batch size;
- final outputs are merged into a stable cache.

## Dataset Notes

The behavior distribution is strongly long-tailed. In the observed 2026-01-25 to 2026-03-31 window:

| Window | Users | Positive Events | Mean/User | P50 | P95 | P99 |
|--------|-------|-----------------|-----------|-----|-----|-----|
| 7d | 1.54M | 23.9M | 15.6 | 3 | 68 | 220 |
| 14d | 2.51M | 53.1M | 21.2 | 3 | 92 | 331 |
| 31d | 4.55M | 129.7M | 28.5 | 3 | 118 | 499 |
| 62d | 7.29M | 261.8M | 35.9 | 3 | 138 | 669 |
| 66d | 7.85M | 299.0M | 38.1 | 3 | 146 | 715 |

For `max_seq_len=512` and 3 SID tokens per item, NTP keeps the most recent 170 items per user. This affects only a small fraction of users, but those users contribute a large fraction of raw interactions.
