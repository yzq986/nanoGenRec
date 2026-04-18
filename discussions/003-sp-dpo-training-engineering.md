# 003: SP-DPO Training Engineering — OOM, DDP, and Joint Optimization

**Date**: 2026-04-18
**Context**: EXP-017 SP-DPO 实现过程中遇到的工程挑战和解决方案

---

## 背景

SP-DPO 的论文只有 loss 公式和消融表格，没有训练工程细节。把它跑起来——尤其是在 8xA100 40GB 上跑 Joint NTP+DPO 训练——踩了大量坑。这篇 discussion 记录关键工程问题和解法。

---

## 1. Joint NTP+DPO 训练流程

每个 training step 做两件事：

```
Step i:
  ┌─────────────────────────────────────────────┐
  │ 1. NTP forward + backward                    │
  │    input: (context, ground_truth_next_hop)    │
  │    loss:  cross-entropy on 3-layer SID        │
  │    → ntp_loss.backward()                      │
  │    → gradients 累积到 policy_model.grad        │
  │    → 释放 NTP 计算图                           │
  ├─────────────────────────────────────────────┤
  │ 2. DPO forward + backward                    │
  │    a) ref_model no_grad forward → ref logprobs│
  │    b) policy_model with_grad forward          │
  │    c) softmax_dpo_loss(policy, ref, chosen,   │
  │       rejected, mask, beta)                   │
  │    → (λ * dpo_loss).backward()                │
  │    → gradients 累积到 policy_model.grad        │
  ├─────────────────────────────────────────────┤
  │ 3. optimizer.step()                           │
  │    total gradient ≈ ∇ntp + λ·∇dpo             │
  └─────────────────────────────────────────────┘
```

关键：**两次 backward 分开做**。如果让 NTP 和 DPO 的计算图同时存在，显存直接爆。NTP backward 完后释放所有中间激活，再做 DPO forward+backward。

最终的 loss：
```
total_loss = ntp_loss + λ * dpo_loss     (λ=0.1)
```

但**不是**在 loss 层面加完再统一 backward，而是分别 backward，梯度自然叠加。

---

## 2. 显存分析：为什么 DPO 比 NTP 贵那么多

### NTP 的显存情况

NTP 一个 batch 就是普通的 (B, T) 序列做 cross-entropy，跟任何 LM 没区别。
B=46, T=510 → 每个样本的激活 ~230MB → 总激活 ~10GB，加上模型参数 ~1GB → ~12GB。

### DPO 的 B×K 爆炸

DPO 需要对 1 chosen + N rejected 都算 log-prob。对于 batch=4, n_rejected=20：

```
每个 preference pair:
  chosen:   1 sequence
  rejected: up to 20 sequences
  → 最多 21 sequences per pair

batch_size=4:
  4 × 21 = 84 sequences (B×K expansion)
```

**问题**：这 84 个序列的 policy forward 必须带梯度，计算图要保留到 backward()。

```
84 sequences × ~230 MB/sequence ≈ 19 GB (仅激活)
+ ref model 的 no_grad forward 已经释放
+ 模型参数 ~1GB
→ 总共 ~20-25 GB（舒适范围）
```

但如果 n_rejected=20, batch=8：
```
8 × 21 = 168 sequences × 230 MB ≈ 38 GB → OOM on 40GB A100
```

### 为什么 micro-batching 救不了 with-grad forward

直觉：把 84 个序列分成 4 个 chunk of 21，分批 forward，不就行了？

**对 no_grad forward（ref model）有效**：每个 chunk forward 完取出结果，释放中间状态，下一个 chunk 复用显存。

**对 with_grad forward（policy model）无效**：每个 chunk 的计算图必须保留到最后统一 backward()。4 个 chunk 的计算图**全部同时存在于显存中**，跟一次性 forward 84 个没区别。

```
Chunk 1: forward → 保留计算图 (~5GB)
Chunk 2: forward → 保留计算图 (~5GB)   ← chunk 1 的图还在
Chunk 3: forward → 保留计算图 (~5GB)   ← 1+2 的图还在
Chunk 4: forward → 保留计算图 (~5GB)   ← 1+2+3 的图还在
                                       ← 总共 20GB，跟不分 chunk 一样
```

这就是为什么最终方案是**直接限制 B×K 总量**，而不是 micro-batching。

### 最终显存配置

| 组件 | 显存 |
|------|------|
| 模型参数 (17.5M, bf16) | ~35 MB |
| Optimizer states (Adam, fp32) | ~140 MB |
| NTP 激活 (B=46, T=510) | ~10 GB |
| → NTP backward 后释放 | 0 |
| Ref no_grad forward (B×K=84, chunked) | ~1 GB peak |
| Policy with_grad forward (B×K=84) | ~19 GB |
| 杂项 (buffers, fragmentation) | ~2-3 GB |
| **Total peak** | **~22-25 GB** |

在 40GB A100 上有充足余量。

---

## 3. DDP 不兼容：从踩坑到手动 All-Reduce

### 问题

PyTorch DDP 假设每个 step 只有一次 forward-backward。SP-DPO 有两次：

```python
# DDP 的期望：
model.forward() → loss.backward() → model.step()

# SP-DPO 的实际：
model.forward()  → ntp_loss.backward()     # 第一次
model.forward()  → dpo_loss.backward()     # 第二次（DDP 已经同步了第一次的梯度）
model.step()
```

第一次 backward 触发 DDP 自动 all-reduce，但我们还没做完 DPO 的 backward。
DDP 不知道后面还有梯度要加。

### 尝试过的方案

**方案 1：`no_sync()` context manager**

```python
with ddp_model.no_sync():
    ntp_loss.backward()
    dpo_loss.backward()
# 退出 no_sync 时自动同步
```

**失败原因**：DPO forward 必须用 raw model（不经过 DDP wrapper），因为 DPO 内部有 per-layer output projection 等直接访问。`raw_model.forward()` 产生的 backward 不经过 DDP 的 hook，所以 `no_sync()` 的 deferred sync 永远不会触发。

**方案 2：MoE + 小 batch → undefined gradients**

NTP batch=32 + MoE (8 experts, top_k=2) → 某些 expert 在某些 rank 上可能完全收不到 token → 该 expert 参数的 gradient = None → DDP all-reduce 在跨 rank 遇到 None vs non-None → 错误。

### 最终方案：去掉 DDP，手动 All-Reduce

```python
# 不用 DDP wrapper
raw_policy = policy_model  # 直接用原模型

# ... NTP backward ...
# ... DPO backward ...

# 所有梯度都累积好之后，手动同步
if world_size > 1:
    for p in policy_model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

optimizer.step()
```

优点：
- 完全控制同步时机
- 兼容任意多次 backward
- 不受 DDP hook 限制
- None gradient 可以安全跳过

缺点：
- 多了一次显式循环所有参数（模型只有 17.5M，开销可忽略）
- 没有 DDP 的 gradient bucketing 优化（但 17.5M 参数的通信量极小）

---

## 4. Gradient Checkpointing 与 MoE 不兼容

为了降低 B×K 的激活显存，尝试过 gradient checkpointing（前向时不保存中间激活，backward 时重新计算）。

### 失败原因：MoE 的 Loss-Free Balancing

模型用的是 Loss-Free MoE（DeepSeek-V3 风格），每个 expert 有一个**动态 bias**，运行时根据负载实时调整：

```python
# 前向时记录的路由：token → expert 3, expert 7
# 重新计算时，dynamic bias 已经变了 → token → expert 3, expert 5
# → 输出 shape 不匹配 → 崩溃
```

Gradient checkpointing 要求**重新前向的输出和原始前向完全一致**。但 MoE 的 dynamic bias 在两次前向之间可能不同（因为中间有其他 batch 经过改变了统计量）。

这是 MoE + Gradient Checkpointing 的一个**已知兼容性问题**，短期内无法绕过（除非重写 MoE 路由逻辑来保存和恢复 bias 状态）。

---

## 5. NTP 和 DPO 分别优化什么

### NTP Loss（交叉熵）

```
L_NTP = -Σ log P(sid_layer_i | context, sid_layer_1..i-1)
```

- **目标**：让模型在每个位置预测正确 token 的概率尽可能高
- **优化的是**：绝对准确性（average per-position accuracy）
- **不关心**：模型对其他候选打了多高的分

### DPO Loss（Softmax-DPO）

```
L_DPO = -log σ(-logsumexp_l[ β(log π/π_ref(rejected_l) - log π/π_ref(chosen)) ])
```

- **目标**：让 chosen 相对 rejected 的概率差距拉大
- **优化的是**：ranking quality（排序质量）
- **不关心**：chosen 的绝对概率是多少，只要它比 rejected 高就行

### Joint Training 的效果

```
total gradient = ∇L_NTP + λ · ∇L_DPO
```

- NTP gradient 保持模型的基础生成能力不退化
- DPO gradient 微调概率分布，让 ground truth 在候选中排名更靠前
- λ=0.1 表示 DPO 的调整力度是 NTP 的 1/10 —— 是微调，不是大改

这就是为什么 **Recall@K 是真正的评估指标**而不是 perplexity：
- PPL 衡量的是绝对概率（NTP 的目标）
- Recall@K 衡量的是 ground truth 能否进入 top-K（DPO 的目标）
- SP-DPO 可能让 PPL 变差（绝对概率下降），但 Recall 改善（排名提升）

---

## 6. 训练 Loss 行为

正常训练日志 (EXP-017, Config 1: Easy, 前 400 步)：

```
step  50: loss=3.07  ntp=2.79  dpo=2.81  lr=1.0e-04
step 100: loss=3.06  ntp=2.79  dpo=2.73  lr=1.0e-04
step 200: loss=3.06  ntp=2.79  dpo=2.67  lr=1.0e-04
step 400: loss=3.06  ntp=2.79  dpo=2.67  lr=1.0e-04
```

解读：
1. **NTP loss 稳定 (~2.79)** — 好的。NTP 已经训练到位，不应该因为 DPO 而大幅波动。如果 NTP loss 突然上升，说明 DPO 在破坏基础能力。
2. **DPO loss 下降 (2.95→2.67)** — 好的。模型在学会区分 chosen vs rejected。
3. **Total loss 缓慢下降** — 由 DPO 贡献，因为 λ=0.1 所以变化幅度小。
4. **显存波峰波谷** — 正常。NTP backward 释放激活 → 谷；DPO forward 重新分配 → 峰。每个 step 都有这个周期。

---

## 7. 未来优化方向：Chunked Backward

当前 B×K=84 是硬约束。如果要更大的 batch（论文用 batch=1024），需要 chunked backward：

```
1. 先 no_grad forward 全部 B×K 个序列，收集 hidden states 在每个 prediction position 的值
2. 分 chunk forward+backward:
   for chunk in chunks:
       h_chunk = hidden_chunk.detach().requires_grad_(True)
       logits = output_proj(h_chunk)
       lp = log_softmax(logits).gather(...)
       loss_chunk = dpo_loss(lp, ...)
       loss_chunk.backward()
       # chunk 的计算图立即释放
3. 总梯度 = 所有 chunk 梯度之和
```

核心 insight：Softmax-DPO loss 只需要每个序列在 3 个 prediction position 的 hidden state（不是整个序列的激活）。可以先用 no_grad 拿到这些 hidden states，再分批算 loss 和梯度。

但 **hidden states → output → loss 的反向传播不能穿透回 Transformer**，所以这只优化了 output projection 层的梯度。要完整优化 Transformer 参数，需要更复杂的 gradient relay 机制。

留作 EXP-019+ 的可选优化。

---

## 总结

| 问题 | 解法 | 要点 |
|------|------|------|
| NTP+DPO 计算图共存 OOM | 分离 backward | NTP backward 先释放，再做 DPO |
| B×K 爆炸 | 限制 batch=4, n_rej=20 | with-grad forward 无法 micro-batch |
| DDP + 双 backward 不兼容 | 手动 all-reduce | 去掉 DDP wrapper，显式同步 |
| MoE + gradient checkpoint | 放弃 checkpointing | Loss-Free bias 非确定性路由 |
| 显存波动 | 正常 | NTP 释放→DPO 分配的周期 |
