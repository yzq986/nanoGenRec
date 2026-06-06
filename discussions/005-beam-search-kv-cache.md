# 005: KV Cache for Beam Search — 3-Level Compute Reuse

[English](005-beam-search-kv-cache.md) | [Chinese](005-beam-search-kv-cache.zh.md)

**Date**: 2026-04-19

## Problem

`constrained_beam_search` is the bottleneck in SP-DPO preference pair generation and eval recall computation. The current implementation is extremely wasteful:

- **Per-step redundancy**: Each decoding step re-encodes the full context through 6 transformer layers, but only uses the last position's output.
- **Per-beam redundancy**: At step 1+, the context is replicated across all beams (B=50) — 50 identical forward passes.
- **Per-pass redundancy**: Prefix-locked mode runs 3 beam searches on the same context.
- **Per-item redundancy**: Consecutive eval items from the same user sequence have incrementally growing contexts that share a long common prefix.

Measured cost: **45 minutes on 8x A100** for ~60K eval items.

## Analysis

For one eval item with context length C, beam_size=50, n_layers=3:

| Step | Beams | Forward calls | Tokens processed |
|------|-------|---------------|-----------------|
| Pass 1, step 0 | 1 | 1 | C |
| Pass 1, step 1 | 50 | 1 (batched) | 50 x (C+1) |
| Pass 1, step 2 | 50 | 1 (batched) | 50 x (C+2) |
| Pass 2, step 1 | 1 | 1 | C+1 |
| Pass 2, step 2 | 50 | 1 (batched) | 50 x (C+2) |
| Pass 3, step 2 | 1 | 1 | C+2 |
| **Total** | | **6** | **~153C** |

With ~3.1 eval items per sequence, the grand total is ~153C x 3.1 = **~475C** tokens processed per sequence — but the unique context data is only ~C tokens.

## Solution: 3-Level KV Cache

### Level 1: Cross-step (within one beam search call)

Encode context once, cache the per-layer pre-norm hidden states. Subsequent decoding steps process only the 1 new generated token, attending to the cached KV.

**Cache content**: Pre-norm hidden states (`norm1(x)` output), following the pattern from `metrics/sid_prediction_old.py`'s `CausalTransformerLayer`. The `nn.MultiheadAttention` module natively supports different Q vs K/V lengths — `query=(B,1,D)` with `key/value=(B,T,D)`.

**Beam gather**: After topk beam selection, KV cache entries must be gathered/reindexed to match surviving beams:
```python
c = step_kv[li].view(B, n_beams, T_cached, D)
c = torch.gather(c, 1, beam_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T_c, D))
step_kv[li] = c.reshape(B * k, T_cached, D)
```

**Causal mask**: When Q_len=1 (single new token), no mask needed — the last position can attend to all prior positions.

### Level 2: Cross-pass (within one eval item)

3 prefix-locked passes share the same context. `constrained_beam_search` returns `ctx_kv_caches` (the context-only cache). Passes 2 and 3 receive it as input, skip context encoding.

Key: `ctx_kv_caches` is `.clone()`'d before each pass so beam search modifications don't corrupt the shared cache.

### Level 3: Cross-item (within one sequence)

Consecutive eval items from the same sequence have contexts:
```
item 0: tokens[:15]           (5 items x 3 layers)
item 1: tokens[:18]           (6 items x 3 layers)  
item 2: tokens[:21]           (7 items x 3 layers)
```

The shared prefix KV from item 0 is extended by 3 tokens for item 1, then 3 more for item 2. This requires `_extract_eval_items_grouped()` to preserve the sequence grouping and process items in context-length order.

## Implementation

| Component | Change |
|-----------|--------|
| `TransformerLayer.forward` | Add `kv_cache`, `use_cache` params. Returns `(x, kv)` when `use_cache=True` |
| `NTPModel.forward_cached` | New method: cold start (full encode) or incremental (new tokens only) |
| `NTPModel._embed_tokens_at_offset` | Handle per-layer embeddings with position offset |
| `constrained_beam_search` | Accept `ctx_kv_caches` + `initial_logits`, return 3-tuple |
| `_constrained_beam_search_legacy` | Preserved original for NTPProbe fallback + verification |
| `_prefix_locked_generate` | Thread `ctx_kv_caches` across 3 passes |
| `build_preference_pairs` | Grouped iteration with cross-item cache sharing |

## Memory Overhead

| Component | Size |
|-----------|------|
| Context KV (B=1, T=30, D=256, 6 layers) | 180 KB |
| Beam-expanded KV (50 beams, T=33, 6 layers) | ~10 MB |
| **Peak total** | **~10 MB** (negligible vs 40GB A100) |

## Expected Speedup

Token processing: 153C x N_items → ~C (amortized). Theoretical ~450x reduction.

Practical factors:
- Trie mask computation (CPU-bound Python dict lookups) becomes a larger fraction
- KV cache memory bandwidth overhead
- Beam gather overhead

Conservative estimate: **10-15x wall time improvement** (45min -> 3-5min).

## Verification

8 numerical equivalence tests in `tests/test_kv_cache.py`:
1. TransformerLayer: full vs incremental (multi-token and single-token)
2. NTPModel: `forward()` vs `forward_cached()` (cold start, incremental, token-by-token)
3. Beam search: cached vs legacy (no prefix, with prefix, cache reuse)

All pass with max diff < 1e-6 (float32 precision).

## Applicability

- **SP-DPO preference generation** (primary target)
- **Eval beam search recall** (`ntp/eval.py`) — automatic via `constrained_beam_search` upgrade
- **Online inference** — same KV cache applies to production beam search
