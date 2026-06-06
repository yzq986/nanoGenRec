# 003: SP-DPO Training Engineering — OOM, DDP, and Joint Optimization

[English](003-sp-dpo-training-engineering.md) | [Chinese](003-sp-dpo-training-engineering.zh.md)

**Date**: 2026-04-18
**Context**: Engineering challenges and solutions encountered during the implementation of EXP-017 SP-DPO

---

## Background

The SP-DPO paper only has loss formulas and ablation tables, but no training engineering details. Getting it running - especially the Joint NTP+DPO training on the 8xA100 40GB - went through a lot of pitfalls. This discussion documents key engineering issues and solutions.

---

## 1. Joint NTP+DPO training process

Each training step does two things:

```
Step i:
  ┌─────────────────────────────────────────────┐
  │ 1. NTP forward + backward                    │
  │    input: (context, ground_truth_next_hop)    │
  │    loss:  cross-entropy on 3-layer SID        │
  │    → ntp_loss.backward()                      │
│ → gradients are accumulated into policy_model.grad │
│ → Release NTP calculation graph │
  ├─────────────────────────────────────────────┤
  │ 2. DPO forward + backward                    │
  │    a) ref_model no_grad forward → ref logprobs│
  │    b) policy_model with_grad forward          │
  │    c) softmax_dpo_loss(policy, ref, chosen,   │
  │       rejected, mask, beta)                   │
  │    → (λ * dpo_loss).backward()                │
│ → gradients are accumulated into policy_model.grad │
  ├─────────────────────────────────────────────┤
  │ 3. optimizer.step()                           │
  │    total gradient ≈ ∇ntp + λ·∇dpo             │
  └─────────────────────────────────────────────┘
```

Key: **Do backward twice separately**. If the calculation graphs of NTP and DPO are allowed to exist at the same time, the video memory will directly explode. After NTP backward is completed, all intermediate activations are released, and then DPO forward+backward is performed.

Final loss:
```
total_loss = ntp_loss + λ * dpo_loss (λ=0.1)
```

But it is not **to add the loss layer and then unify backward, but to go backward separately, and the gradients are naturally superimposed.

---

## 2. Video memory analysis: why DPO is so much more expensive than NTP

### NTP memory status

A batch of NTP is an ordinary (B, T) sequence for cross-entropy, which is no different from any LM.
B=46, T=510 → activations per sample ~230MB → total activations ~10GB, plus model parameters ~1GB → ~12GB.

### DPO’s B×K Explosion

DPO needs to count 1 chosen + N rejected as log-prob. For batch=4, n_rejected=20:

```
Each preference pair:
  chosen:   1 sequence
  rejected: up to 20 sequences
→ up to 21 sequences per pair

batch_size=4:
  4 × 21 = 84 sequences (B×K expansion)
```

**Question**: The policy forward of these 84 sequences must have gradients, and the calculation graph must be retained until backward().

```
84 sequences × ~230 MB/sequence ≈ 19 GB (activated only)
+ no_grad forward of ref model has been released
+ Model parameters ~1GB
→ ~20-25 GB total (comfortable range)
```

But if n_rejected=20, batch=8:
```
8 × 21 = 168 sequences × 230 MB ≈ 38 GB → OOM on 40GB A100
```

### Why micro-batching cannot save with-grad forward

Intuition: Wouldn’t it be enough to divide the 84 sequences into 4 chunks of 21 and forward them in batches?

**Effective for no_grad forward (ref model)**: After each chunk forward, the results are retrieved, the intermediate state is released, and the next chunk reuses the video memory.

**Not valid for with_grad forward (policy model)**: The calculation graph of each chunk must be retained until the last unified backward(). The calculation graphs of the 4 chunks all exist in the video memory at the same time, which is no different from forwarding 84 chunks at a time.

```
Chunk 1: forward → keep calculation graph (~5GB)
Chunk 2: forward → Keep the calculation graph (~5GB) ← The graph of chunk 1 is still there
Chunk 3: forward → Keep the calculation graph (~5GB) ← The graph of 1+2 is still there
Chunk 4: forward → Keep the calculation graph (~5GB) ← The graph of 1+2+3 is still there
← 20GB in total, the same as not dividing into chunks
```

This is why the final solution is to directly limit the total amount of B×K instead of micro-batching.

### Final graphics memory configuration

| 组件 | 显存 |
|------|------|
| ModelParameter (17.5M, bf16) | ~35 MB |
| Optimizer states (Adam, fp32) | ~140 MB |
| NTP 激活 (B=46, T=510) | ~10 GB |
| → NTP backward 后释放 | 0 |
| Ref no_grad forward (B×K=84, chunked) | ~1 GB peak |
| Policy with_grad forward (B×K=84) | ~19 GB |
| 杂项 (buffers, fragmentation) | ~2-3 GB |
| **Total peak** | **~22-25 GB** |

There's plenty of headroom on the 40GB A100.

---

## 3. DDP incompatibility: from pitfalls to manual All-Reduce

### question

PyTorch DDP assumes that there is only one forward-backward per step. SP-DPO has two:

```python
# DDP expectations:
model.forward() → loss.backward() → model.step()

# Actual SP-DPO:
model.forward() → ntp_loss.backward() # First time
model.forward() → dpo_loss.backward() # The second time (DDP has synchronized the gradient of the first time)
model.step()
```

The first backward triggers DDP’s automatic all-reduce, but we haven’t finished DPO’s backward yet.
DDP doesn’t know that there are gradients to be added later.

### Tried solutions

**Option 1: `no_sync()` context manager**

```python
with ddp_model.no_sync():
    ntp_loss.backward()
    dpo_loss.backward()
# Automatically synchronize when exiting no_sync
```

**Cause of failure**: DPO forward must use raw model (without going through DDP wrapper), because DPO has direct access to per-layer output projection and so on. The backward generated by `raw_model.forward()` does not go through the DDP hook, so the deferred sync of `no_sync()` will never be triggered.

**Option 2: MoE + small batch → undefined gradients**

NTP batch=32 + MoE (8 experts, top_k=2) → Some experts may not receive tokens at all on certain ranks → The gradient of the expert parameter = None → DDP all-reduce encounters None vs non-None → errors across ranks.

### Final solution: remove DDP and manually All-Reduce

```python
# No DDP wrapper required
raw_policy = policy_model # Use the original model directly

# ... NTP backward ...
# ... DPO backward ...

# After all gradients are accumulated, synchronize manually
if world_size > 1:
    for p in policy_model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

optimizer.step()
```

Advantages:
- Full control over sync timing
- Compatible with any number of backwards
- Not restricted by DDP hooks
- None gradient can be safely skipped

Disadvantages:
- One more explicit loop through all parameters (the model is only 17.5M, the overhead is negligible)
- No gradient bucketing optimization of DDP (but the communication volume of 17.5M parameters is minimal)

---

## 4. Gradient Checkpointing is not compatible with MoE

In order to reduce the activation memory of B×K, gradient checkpointing was tried (intermediate activations are not saved in the forward direction and recalculated in the backward direction).

### Reason for failure: MoE’s Loss-Free Balancing

The model uses Loss-Free MoE (DeepSeek-V3 style). Each expert has a **dynamic bias**, which is adjusted in real time according to the load during runtime:

```python
#Route recorded in forward direction: token → expert 3, expert 7
# When recalculating, the dynamic bias has changed → token → expert 3, expert 5
# → output shape mismatch → crash
```

Gradient checkpointing requires that the output of the re-forward is exactly the same as the original forward. But the dynamic bias of MoE may be different between the two forwards (because other batches have changed the statistics in the middle).

This is a **known compatibility issue** with MoE + Gradient Checkpointing and cannot be bypassed in the short term (unless the MoE routing logic is rewritten to save and restore bias state).

---

## 5. What do NTP and DPO optimize respectively?

### NTP Loss (Cross Entropy)

```
L_NTP = -Σ log P(sid_layer_i | context, sid_layer_1..i-1)
```

- **Goal**: Let the probability of the model predicting the correct token at each position be as high as possible
- **Optimized for**: absolute accuracy (average per-position accuracy)
- **Don't care**: How high the model scored other candidates

### DPO Loss (Softmax-DPO)

```
L_DPO = -log σ(-logsumexp_l[ β(log π/π_ref(rejected_l) - log π/π_ref(chosen)) ])
```

- **Goal**: Increase the probability gap between chosen and rejected
- **Optimization is**: ranking quality (ranking quality)
- **Don't care**: What is the absolute probability of chosen, as long as it is higher than rejected

### Effect of Joint Training

```
total gradient = ∇L_NTP + λ · ∇L_DPO
```

- NTP gradient keeps the model’s basic generation capabilities from degrading
- DPO gradient fine-tunes the probability distribution to make the ground truth rank higher among the candidates
- λ=0.1 means that the adjustment intensity of DPO is 1/10 of NTP - it is a fine adjustment, not a major change

This is why **Recall@K is the real evaluation metric** and not perplexity:
- PPL measures absolute probability (the goal of NTP)
- Recall@K measures whether the ground truth can enter top-K (the goal of DPO)
- SP-DPO may make PPL worse (absolute probability decreases), but Recall improves (ranking increases)

---

## 6. Training Loss behavior

Normal training log (EXP-017, Config 1: Easy, first 400 steps):

```
step  50: loss=3.07  ntp=2.79  dpo=2.81  lr=1.0e-04
step 100: loss=3.06  ntp=2.79  dpo=2.73  lr=1.0e-04
step 200: loss=3.06  ntp=2.79  dpo=2.67  lr=1.0e-04
step 400: loss=3.06  ntp=2.79  dpo=2.67  lr=1.0e-04
```

Interpretation:
1. **NTP loss stable (~2.79)** — OK. NTP is well trained and should not fluctuate significantly due to DPO. If NTP loss suddenly rises, it means that DPO is destroying basic capabilities.
2. **DPO loss decreased (2.95→2.67)** — Okay. The model learns to distinguish chosen vs rejected.
3. **Total loss decreases slowly** — contributed by DPO, because λ=0.1 so the change amplitude is small.
4. **Video Memory Peaks and Troughs** — Normal. NTP backward releases activation → valley; DPO forward reallocates → peak. Each step has this cycle.

---

## 7. Future optimization direction: Chunked Backward

Currently B×K=84 is a hard constraint. If you want a larger batch (the paper uses batch=1024), you need chunked backward:

```
1. First no_grad forward all B×K sequences and collect the value of hidden states at each prediction position
2. Divide chunk forward+backward:
   for chunk in chunks:
       h_chunk = hidden_chunk.detach().requires_grad_(True)
       logits = output_proj(h_chunk)
       lp = log_softmax(logits).gather(...)
       loss_chunk = dpo_loss(lp, ...)
       loss_chunk.backward()
# The calculation graph of chunk is released immediately
3. Total gradient = sum of gradients of all chunks
```

Core insight: Softmax-DPO loss only requires the hidden state of 3 prediction positions for each sequence (not the activation of the entire sequence). You can first use no_grad to get these hidden states, and then calculate the loss and gradient in batches.

But the backpropagation of hidden states → output → loss cannot penetrate back to the Transformer, so this only optimizes the gradient of the output projection layer. To fully optimize Transformer parameters, a more complex gradient relay mechanism is required.

Reserved as an optional optimization for EXP-019+.

---

## Summary

| 问题 | 解法 | 要点 |
|------|------|------|
| NTP+DPO 计算图共存 OOM | 分离 backward | NTP backward 先释放，再做 DPO |
| B×K 爆炸 | 限制 batch=4, n_rej=20 | with-grad forward 无法 micro-batch |
| DDP + 双 backward 不兼容 | 手动 all-reduce | 去掉 DDP wrapper，显式同步 |
| MoE + gradient checkpoint | 放弃 checkpointing | Loss-Free bias 非确定性路由 |
| 显存波动 | 正常 | NTP 释放→DPO 分配的周期 |
