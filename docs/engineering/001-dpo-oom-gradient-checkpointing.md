# 001: DPO Gradient Checkpointing + MoE Compatibility + Memory Fragmentation

[English](001-dpo-oom-gradient-checkpointing.md) | [Chinese](001-dpo-oom-gradient-checkpointing.zh.md)

**Date**: 2026-04-19
**Context**: EXP-017 SP-DPO training on 8xA100 40GB, 45.8M params
**Files**: `rl/dpo.py`, `rl/trainer.py`, `ntp/model.py`

## Background

SP-DPO training has two losses per step:

- **NTP loss**: a large batch, around 149 sequences with `seq_len=510`, through `_forward_packed`.
- **DPO loss**: a small batch of 16 preference pairs, expanded into K=21 candidates, through `compute_sid_logprobs_batch`.

The order is NTP forward, NTP backward, DPO forward, DPO backward, optimizer step.

Important distinction: `batch_size` for NTP and `dpo_batch_size` are different parameters. NTP batch size controls the number of sequences and is auto-capped based on GPU memory. DPO batch size controls the number of preference pairs, and each pair expands into `K = 1 + N_rej` candidates.

## Problem Chain

### Problem 1: DPO Forward OOM at About 60 GB

Symptom: NTP with batch=149 completed forward and backward, but DPO forward OOMed.

Root cause: `compute_sid_logprobs_batch` expands `dpo_batch=16 x K=21` into 336 candidates and splits them into 6 chunks with `max_chunk=64`. Each chunk runs a full Transformer forward, and all 6 chunk computation graphs stay in memory until `backward()`:

```text
chunk 1 forward -> graph retained, about 10 GB activations
chunk 2 forward -> graph retained, about 10 GB
...
chunk 6 forward -> graph retained, about 10 GB
total: 6 x 10 GB = 60 GB -> OOM
```

Options considered:

| Option | Benefit | Drawback |
|--------|---------|----------|
| Reduce `dpo_batch`, 16 -> 4 | One-line change | Noisier gradients and does not solve the root cause |
| Gradient checkpointing | Decouples memory from batch size | Recomputes DPO forward during backward, about 25% extra step time |

Decision: use `torch.utils.checkpoint.checkpoint` around each chunk forward.

- Forward keeps only inputs and does not save intermediate activations.
- Backward recomputes one chunk at a time, computes gradients, then releases memory.
- Peak memory becomes one chunk, a constant independent of `dpo_batch_size`.

Additional benefit: `dpo_batch_size` can be increased freely; it affects time but not peak memory.

Commit: `b6811c1`.

### Problem 2: Shape Mismatch, 926 vs 939

Symptom: checkpoint recomputation failed during backward with saved tensor shape `(926, 256)` but recomputed shape `(939, 256)`.

Root cause: the model uses `SparseMoEBlock`. Its loss-free load-balancing mechanism updates `expert_bias` in place inside `forward()`:

```python
if self.training:
    self.expert_bias.add_(...)
```

This makes `forward()` non-idempotent. Router top-k decisions determine the shapes of `x_flat[mask]` tensors for each expert. When the bias changes, routing changes, and intermediate tensor shapes change.

Timeline:

1. Forward records checkpoint metadata with bias A and shape 926.
2. Forward exits after updating bias to B.
3. Backward recomputes forward with bias B and shape 939, then crashes.

Fix: add a `freeze_bias` flag to `SparseMoEBlock`:

```python
if self.training and not self.freeze_bias:
    self.expert_bias.add_(...)
```

Freeze the bias during DPO, while the NTP forward path continues to update it normally.

Commit: `d7689f6`.

### Problem 3: Freeze Scope Was Too Narrow

Symptom: shape mismatch still occurred after adding `freeze_bias`.

Root cause: the `_freeze_moe_bias` context manager only wrapped the forward loop inside `compute_sid_logprobs_batch`. Checkpoint recomputation happens inside `backward()`, after the context manager had already exited:

```python
with _freeze_moe_bias(model):
    for chunk in chunks:
        chunk_lp = checkpoint(...)   # forward is frozen

(dpo_weight * loss).backward()       # recompute is no longer frozen
```

Fix: move `_freeze_moe_bias` to `trainer.py` and wrap the whole DPO section, including forward, loss construction, and backward:

```python
with _freeze_moe_bias(raw_policy):
    policy_lp = compute_sid_logprobs_batch(...)
    dpo_loss = softmax_dpo_loss(...)
    (dpo_weight * dpo_loss).backward()
```

Commit: `815fc0d`.

### Problem 4: all_reduce OOM From NCCL CUDA Allocation

Symptom: DPO forward and backward completed, but gradient all-reduce OOMed while allocating only about 183 MB for a flat gradient tensor.

Root cause: CUDA caching allocator fragmentation.

Memory timeline:

1. NTP forward allocates about 30 GB of mixed-size activations.
2. NTP backward frees activations, but the CUDA caching allocator keeps fragmented blocks.
3. DPO checkpoint forward/backward allocates and releases one chunk repeatedly, increasing fragmentation.
4. all-reduce needs one contiguous 183 MB allocation. There is plenty of free cached memory, but not enough contiguous space.

Two fixes:

**A. DPO memory reserve**

When DPO is enabled, the NTP batch auto-cap subtracts an additional 3 GB. NTP batch decreased from about 149 to 136, leaving space for DPO checkpoint peak, NCCL buffers, and fragmentation margin.

**B. Pre-allocate the flat gradient buffer**

Allocate the all-reduce flat buffer once before the training loop, while memory is clean. Each step fills it via `copy_`, avoiding a new `torch.cat` allocation during the fragmented phase.

`torch.cuda.empty_cache()` was deliberately not used. Earlier validation showed that it forces CUDA device synchronization and causes periodic GPU utilization drops to 0%.

Commit: `4130978`.

## Validation Results

Source data: exp017-spdpo-easy before the fix, from `train_meta.json`, compared with exp017-fixed-hard after the fix from live logs. Same dataset, same hardware, same model.

| Metric | Easy before fix | Fixed-hard after fix | Change |
|--------|-----------------|----------------------|--------|
| NTP batch | 46 | 136 | 3x |
| DPO batch | 4 | 16 | 4x |
| tok/s | 9,038 | 17,123 | 1.9x |
| Total training time | about 4.0h, 14,377s | about 2.0h | -50% |
| Steps | 4,599 | 1,555 | -66%, due to larger batch |
| GPU util | unstable under memory pressure | 79-84%, stable | improved |
| GPU memory | at the limit with repeated OOMs | 91-95%, 37.4-38.9 GB | stable |
| OOM | repeated tuning required | none | fixed |

## Why Throughput Nearly Doubled

1. **NTP batch became 3x larger**: 46 -> 136. The auto-cap formula went through several corrections and now models per-sample memory accurately. Larger batches mean fewer steps and less optimizer/communication overhead.
2. **Gradient checkpointing removed allocator thrashing**: before the fix, six chunk graphs occupied about 60 GB at once. After the fix, peak DPO memory is one chunk. Although DPO forward is recomputed, reduced allocator pressure more than pays for it.
3. **DPO batch became 4x larger**: 4 -> 16. Checkpointing decouples DPO memory from DPO batch size, improving throughput and gradient stability.
4. **Pre-allocated all-reduce buffer**: avoids allocation failure exactly when memory is fragmented.

Lesson: memory optimization is not only about avoiding OOM. Lower peak memory can improve throughput by removing CUDA allocator stalls, even when it adds recomputation.

## Key Takeaways

1. **Gradient checkpointing and MoE can conflict**: any state mutation inside `forward()`, such as loss-free bias updates, can make recomputation inconsistent. Stateful updates must be frozen in checkpointed regions.
2. **The freeze scope must cover `backward()`**: checkpoint recomputation happens during backward, not inside the original forward loop.
3. **CUDA fragmentation is a hidden failure mode**: peak memory can look sufficient while small contiguous allocations still fail. Pre-allocation and reserve margins are more reliable than `empty_cache()`.
4. **Lower peak memory can improve throughput**: eliminating allocator pressure doubled effective throughput despite recomputation.
5. **NTP batch and DPO batch are independent**: the auto-cap formula covers NTP batch size. DPO memory needs a separate reserve and checkpointing strategy.
