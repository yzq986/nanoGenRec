# 001: DPO Gradient Checkpointing + MoE 兼容 + 显存碎片化

[English](001-dpo-oom-gradient-checkpointing.md) | [中文](001-dpo-oom-gradient-checkpointing.zh.md)

**Date**: 2026-04-19
**Context**: EXP-017 SP-DPO 训练（8×A100 40GB, 45.8M params）
**Files**: `rl/dpo.py`, `rl/trainer.py`, `ntp/model.py`

---

## 背景

SP-DPO 训练每步包含两个 loss：
- **NTP loss**: 大 batch（~149 sequences, seq_len=510）走 `_forward_packed`
- **DPO loss**: 小 batch（16 preference pairs）展开为 K=21 candidates，走 `compute_sid_logprobs_batch`

两者顺序执行：NTP forward → NTP backward → DPO forward → DPO backward → optimizer step。

**关键概念区分**：NTP batch_size 和 dpo_batch_size 是完全不同的参数。
NTP batch 控制序列数量，auto-cap 公式根据 GPU 显存计算上限；
DPO batch 控制 preference pair 数量，每个 pair 展开为 K=1+N_rej 个 candidate。

---

## 问题链（4 个问题逐层暴露）

### Problem 1: DPO Forward OOM（~60 GB）

**现象**: NTP batch=149 通过 forward+backward，但 DPO forward 阶段 OOM。

**根因**: `compute_sid_logprobs_batch` 把 dpo_batch=16 × K=21 = 336 个 candidate 分成 6 个 chunk（max_chunk=64），每个 chunk 做完整 transformer forward。所有 6 个 chunk 的**计算图同时保留**在显存中，直到 `backward()` 才释放：

```
chunk 1 forward → 保留计算图（~10 GB activations）
chunk 2 forward → 保留计算图（~10 GB）
...
chunk 6 forward → 保留计算图（~10 GB）
total: 6 × 10 GB = 60 GB → OOM
```

**方案选择**:

| 方案 | 优点 | 缺点 |
|------|------|------|
| A. 减 dpo_batch (16→4) | 改动一行 | 梯度更 noisy，不解决根本问题 |
| B. Gradient checkpointing | 显存和 batch 解耦 | DPO forward 算两遍（~25% step 时间） |

选择 **B**：`torch.utils.checkpoint.checkpoint` 包裹每个 chunk 的 forward。
- Forward 阶段：中间激活值不保存，只记住输入
- Backward 阶段：逐个 chunk 重算 forward → 计算梯度 → 释放
- 峰值显存 = 1 个 chunk（常数），与 dpo_batch_size 无关

**额外收益**：dpo_batch_size 可以自由增大（32, 64, ...），只增加时间不增加显存。

**Commit**: `b6811c1`

---

### Problem 2: Shape Mismatch（926 vs 939）

**现象**: Gradient checkpointing backward 时报错：saved tensor shape (926, 256) ≠ recomputed shape (939, 256)。

**根因**: 模型使用 **Sparse MoE**（SparseMoEBlock），其 Loss-Free 负载均衡机制在每次 `forward()` 中**就地更新** `expert_bias`：

```python
# SparseMoEBlock.forward():
if self.training:
    self.expert_bias.add_(...)  # in-place update
```

这使 forward() **不幂等**。Router 的 top-k 选择决定了 `x_flat[mask]` 的形状（哪些 token 分给哪个 expert）。Bias 变了 → router 决策变了 → 中间 tensor 形状变了。

时间线：
1. Forward（checkpoint 记录 metadata）：bias = A → shape = 926
2. Forward 结束，bias 被更新为 B
3. Backward recompute：bias = B → shape = 939 → crash

**解决方案**: `SparseMoEBlock` 新增 `freeze_bias` 标志：

```python
if self.training and not self.freeze_bias:
    self.expert_bias.add_(...)
```

DPO 期间设 `freeze_bias=True`，NTP forward 正常更新。

**Commit**: `d7689f6`

---

### Problem 3: freeze 范围不够

**现象**: 修了 freeze_bias 后仍然 shape mismatch。

**根因**: `_freeze_moe_bias` context manager 只包裹了 `compute_sid_logprobs_batch` 内的 forward 循环。但 checkpoint 的 recompute 发生在 `backward()` 调用时，此时 **已退出 context manager**：

```python
with _freeze_moe_bias(model):       # ← freeze_bias = True
    for chunk in chunks:
        chunk_lp = checkpoint(...)   # forward: OK, bias frozen
                                     # ← freeze_bias = False (exited!)
(dpo_weight * loss).backward()       # recompute: bias NOT frozen → crash
```

**解决方案**: 把 `_freeze_moe_bias` 移到 `trainer.py`，包裹整个 DPO 段（forward + loss + backward）：

```python
with _freeze_moe_bias(raw_policy):   # ← freeze_bias = True
    policy_lp = compute_sid_logprobs_batch(...)  # forward
    dpo_loss = softmax_dpo_loss(...)
    (dpo_weight * dpo_loss).backward()           # backward recompute: still frozen
                                     # ← freeze_bias = False
```

**Commit**: `815fc0d`

---

### Problem 4: all_reduce OOM（NCCL CUDA OOM）

**现象**: DPO forward+backward 成功完成，但 gradient all_reduce 时 OOM。需要分配的只是 ~183 MB 的 flat gradient tensor。

**根因**: CUDA 缓存分配器碎片化。

Memory timeline：
1. NTP forward：分配 ~30 GB activations（多种大小的 tensor）
2. NTP backward：释放 activations，但 CUDA 缓存分配器**保留碎片化的 block**
3. DPO checkpoint forward/backward：在碎片间分配/释放 1 chunk，进一步碎片化
4. all_reduce：需要 183 MB 连续内存 → 缓存中有 ~30 GB freed blocks，但无法拼出 183 MB 连续区域 → OOM

**两个修复**:

**A. DPO memory reserve（auto-cap 扣 3 GB）**

NTP batch_size auto-cap 公式在 DPO 启用时额外扣除 3 GB：
- NTP batch 从 ~149 降到 ~136
- 为 DPO checkpoint 峰值、NCCL buffer（~256 MB）、碎片余量留空间

**B. 预分配 flat gradient buffer**

训练循环前（显存干净时）一次性分配 all_reduce 用的 flat buffer（183 MB）。
每步用 `copy_` 填充，不再 `torch.cat` 新分配。直接避免碎片化时刻的分配。

注意：**不使用 `torch.cuda.empty_cache()`**。之前验证过 empty_cache 会强制 CUDA device synchronize，导致 GPU 利用率出现波峰波谷（周期性降到 0%）。

**Commit**: `4130978`

---

## 验证结果

### Before vs After 对比

数据来源：exp017-spdpo-easy（修复前，train_meta.json）vs exp017-fixed-hard（修复后，实时日志）。
同一数据集（~130M tokens），同一硬件（8×A100 40GB），同一模型（45.8M params）。

|  | Easy（修复前） | Fixed-hard（修复后） | 变化 |
|---|---|---|---|
| NTP batch | 46 | 136 | 3× |
| DPO batch | 4 | 16 | 4× |
| tok/s | 9,038 | 17,123 | **1.9×** |
| 总训练时间 | ~4.0h (14,377s) | ~2.0h | **-50%** |
| Steps | 4,599 | 1,555 | -66% (更大 batch → 更少 steps) |
| GPU util | 不稳定（内存压力） | 79-84% (stable) | |
| GPU mem | 极限（反复 OOM） | 91-95% (37.4-38.9 GB) | |
| OOM | 多次，需反复调参 | 无 | |

### 速度翻倍的原因分析

修复前 NTP batch 只能到 46（auto-cap 公式不完善 + DPO 计算图占满显存），修复后提升到 136，这是最大的加速来源。具体因素：

1. **NTP batch 3× 更大**（46 → 136）：auto-cap 公式经过 5 轮修正（漏算 attention matrix → 漏算 dropout mask → 漏算 QKV/FFN → 安全系数不足 → DPO reserve），最终准确建模了 per-sample 显存开销。更大的 batch = 更少的 steps = 更少的 optimizer/通信开销。

2. **Gradient checkpointing 消除 allocator thrashing**：修复前 6 个 chunk 的计算图（~60 GB）同时在显存中，CUDA 分配器处于极端内存压力下。修复后峰值仅 ~10 GB（1 chunk），分配器不再反复碎片化/回收，GPU 计算不被内存分配 stall 打断。虽然 DPO forward 算了两遍（checkpoint 代价），省下来的内存管理开销远超重算成本。

3. **DPO batch 4× 更大**（4 → 16）：修复前为避免 OOM 被迫压到 4，每步只 4 个 preference pair。修复后 checkpointing 让 DPO 显存与 batch 解耦，可安全使用 16。更大的 DPO batch 不仅更快（amortize overhead），梯度信号也更稳定。

4. **预分配 all_reduce buffer**：避免碎片化时刻的内存分配失败，消除 NCCL 通信中断。

**Lesson**: 显存优化不只是避免 OOM — 降低 peak memory 可以显著提升吞吐量，即使增加了计算量。

---

## 关键 Takeaways

1. **Gradient checkpointing + MoE 有兼容性问题**：任何在 `forward()` 中修改 module state 的操作（如 Loss-Free bias update）都会导致 recompute 不一致。必须在 checkpoint 区域冻结所有 stateful update。

2. **freeze 的范围必须覆盖 backward()**：checkpoint 的 recompute 发生在 `backward()` 内部，不是 forward 循环内部。Context manager 必须包裹到 backward 之后。

3. **CUDA 碎片化是隐性杀手**：即使峰值显存理论上够用，交替分配/释放不同大小的 tensor 会导致碎片化，使得小内存分配也可能失败。预分配和留余量比 empty_cache 更好。

5. **降低 peak memory 可能提升吞吐量**：即使增加了计算量（checkpoint recompute），减轻内存压力消除了 CUDA allocator thrashing，实际吞吐量反而翻倍。显存优化不只是避免 OOM。

4. **NTP batch 和 DPO batch 是独立概念**：auto-cap 公式只管 NTP batch。DPO 的显存开销需要单独考虑，通过 reserve 机制扣除。
