# Generative Recommendation Architecture Design

[English](ARCHITECTURE.md) | [Chinese](ARCHITECTURE.zh.md)

## References

- OneRec: arXiv 2506.13695
- OneRec-V2, "Lazy Decoder-Only": arXiv 2508.20900
- OneRec author article: https://zhuanlan.zhihu.com/p/1918350919508140128
- Mixtral MoE: arXiv 2401.04088

## Core Insights From OneRec

### 1. Model Scale and Deployment

- The target model size is 10B, with an online 1B MoE deployment (24 experts, top-2, 13% active rate).
- MFU needs to stay around 20-30% or higher, so the architecture must stay simple and avoid unusual operators that harm hardware utilization.
- The core advantage of generative recommendation is a much larger solution space than discriminative retrieval/ranking, which can absorb more compute.

### 2. Tokenizer Design

- The codebook must be small enough for co-occurrence learning. OneRec maps 10B item IDs into a 8192x3 codebook.
- The tokenizer should be parallel rather than residual.
  - Residual coding constrains the retrieval space: L1 prediction determines the L2 search space, which determines the L3 search space.
  - Parallel coding predicts L1, L2, and L3 independently, allowing grid-style combinatorial search.

### 3. End-to-End System

- One model replaces the traditional retrieval, coarse ranking, fine ranking, and reranking pipeline.
- This is an architecture-level replacement rather than an optimization of a single stage.

### 4. Removing Sparse Item Embedding Tables

- Traditional recommendation systems keep one embedding row per item, resulting in multi-billion-parameter sparse tables that require sharding.
- Generative recommendation represents items as Semantic ID tokens, so only a compact codebook embedding is needed.
- The model parameters live in the Transformer and MoE blocks rather than in a massive sparse table.

## OneRec-V2 Lazy Decoder-Only Architecture

### Key Change: Remove the Encoder

- V1 uses an encoder-decoder architecture.
- V2 removes the encoder and uses a lightweight context processor to generate static KV pairs.
- The decoder handles only 3 target tokens, reducing compute by about 94%.

### MoE Configuration

- 53 routed experts plus 1 shared expert, top-3 routing.
- Load balancing follows the Switch Transformer auxiliary-loss style.

### Model Scale From the Paper

| Config | Params | Active |
|--------|--------|--------|
| Dense 0.1B | 0.1B | 0.1B |
| Dense 1B | 1B | 1B |
| Dense 8B | 8B | 8B |
| MoE 4B | 4B | 0.5B |

### Scaling Law

```text
L_hat(N) = 3.13 + 3660 / N^0.489
```

### Online Deployment

- 1B model, beam=512, latency=36 ms, MFU=62%.
- 3 Semantic ID tokens per item with parallel, non-residual tokenization.

## Our Model Tiers

Designed around the OneRec architecture and an 8xA100 environment.

|                     | S: eval/tuning | M: experiments | L: near online |
|---------------------|----------------|----------------|----------------|
| embed_dim           | 256            | 512            | 1024           |
| n_layers            | 6              | 12             | 24             |
| n_heads             | 8              | 8              | 16             |
| MoE experts         | 8              | 16             | 24             |
| MoE top-k           | 2              | 2              | 2              |
| expert FFN dim      | 1024 (4x)      | 2048 (4x)      | 4096 (4x)      |
| attention per layer | 0.26M          | 1.0M           | 4.2M           |
| MoE FFN per layer   | 4.2M           | 33.6M          | 201M           |
| total per layer     | 4.5M           | 34.6M          | 205M           |
| total params        | ~39.5M         | ~415M          | ~4.9B          |
| active params       | ~11M           | ~55M           | ~420M          |
| codebook            | 256x3          | 256x3          | 8192x3         |
| sequence length     | 30 tokens      | 30 tokens      | 30 tokens      |
| 8xA100 training     | single-GPU quick runs | DDP becomes useful | DDP + AMP required |
| 8xA100 inference    | easy           | easy           | needs optimization |

Measured S-tier size: 39.5M total parameters and about 11M active parameters. SwiGLU uses three matrices, so it is slightly larger than the initial estimate.

## Current Implementation Status

### Implemented

- `ExpertFFN`: SwiGLU FFN with gate/up/down projections.
- `SparseMoEBlock`: linear router, softmax, top-k dispatch, weighted combination.
- Switch Transformer style load-balancing auxiliary loss.
- `CausalTransformerLayer`: supports `use_moe=True/False`.
- `AutoregressiveNTPModel`: S-tier default config, 256d, 6 layers, 8 heads, 8 experts, top-2.
- Training: cross-entropy loss plus 0.01 x auxiliary loss, AMP BF16, DataParallel.
- Inference: batched beam search with flattened `B x beam` forwarding.
- KV-cache incremental decoding.

### Planned

- Parallel tokenizer: replace residual RKMeans with independently encoded levels.
- Context processor: OneRec-V2 style lazy decoder-only path.
- M-tier experiments with DDP.
- L-tier experiments with DDP plus model sharding.
- Larger codebook, from 256 to 8192, for L-tier.

## Key Files

| File | Description |
|------|-------------|
| `ntp/model.py` | NTP model, MoE layers, inference utilities. |
| `ntp/train.py` | NTP training pipeline. |
| `model/train.py` | End-to-end training CLI: encode, tokenize, SID export. |
| `tokenizer/rkmeans.py` | RKMeans tokenizer, currently residual coding. |
| `eval/batch.py` | Batch evaluation entry point. |
| `eval/hyperparam.py` | Hyperparameter search. |
| `eval/behavior.py` | Behavior metric framework. |

## Design Decisions

### 2026-04-13: MoE Architecture

Decision: use a simple Mixtral-style MoE implementation instead of MegaBlocks or DeepSpeed MoE.

Rationale:

1. OneRec emphasizes MFU; a simple architecture is more important than specialized operators at this stage.
2. The Mixtral pattern is mature and has a compact core implementation.
3. The project can migrate to MegaBlocks later for block-sparse optimization.

### 2026-04-13: SwiGLU vs GELU FFN

Decision: use SwiGLU for experts and GELU for non-MoE FFN blocks.

Rationale: SwiGLU has been validated in Llama/Mixtral-style models. It adds parameters but keeps per-expert compute manageable.

### 2026-04-13: No DDP for S-Tier

Decision: use DataParallel instead of DDP for S-tier.

Rationale: the 39.5M-parameter S-tier model trains quickly on one GPU; DDP process startup overhead dominates. M-tier is the point where DDP becomes worthwhile.

### 2026-04-13: Residual Tokenizer Limitation

Problem: the current RKMeans tokenizer is residual, which constrains beam-search space.

Plan: implement a parallel tokenizer where each level is encoded independently instead of conditioning later levels on earlier residuals.
