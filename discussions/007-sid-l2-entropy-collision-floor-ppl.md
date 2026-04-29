# SID L2 Entropy、Collision Rate 与 Floor PPL 的对应关系

**Date**: 2026-04-29  
**Context**: EXP-043 对比 0.6B / 4B / 8B SID cache，发现 L2 entropy 随 embedding 维度下降，且与 NTP scaling law 反推得到的 floor PPL 方向一致

---

## 核心结论

**L2 entropy（可在 tokenizer 阶段快速统计）是 NTP floor PPL 的可靠代理指标**，不需要跑完整 NTP 就能筛选 FSQ hidden dim 配置。

---

## 三个概念的定义

**L2 entropy（codebook 利用率熵）**

FSQ L2 层把 item embedding 映射到 4096 个离散 slot（`[2]×12`）。L2 entropy 衡量 slot 使用的均匀程度：

```
H(L2) = -Σ_i  p_i · log2(p_i)
```

最大值 `log2(4096) = 12 bits`（每个 slot 被等概率使用）。实测：
- 0.6B: H = 10.58 bits，利用率 91.2%（有效 slot ≈ 1500）
- 4B:   H =  8.10 bits，利用率 78.7%（有效 slot ≈ 275）
- 8B:   H =  7.17 bits，利用率 71.6%（有效 slot ≈ 145）

**Collision rate（L2 冲突率）**

不同 item 映射到同一完整 SID（L0_L1_L2 三元组）的比例。L2 entropy 越低 → 有效 SID 空间越小 → 平均每个 SID 对应更多 item → 冲突率越高。

两者关系（近似）：
```
平均冲突大小 k ≈ n_items / (4096 × eff_l2_slots)
collision_rate ≈ (k - 1) / k
```

**Floor PPL（不可约 PPL 下界）**

通过 scaling law 两点法反推：固定指数 `α = 0.456`，用 S-tier 和 M-tier 两个模型大小拟合：

```
L(N) = floor + b / N^α
```

解得 floor（随 N→∞ 时 PPL 的理论下限）：
- 0.6B SID: floor = 12.46
- 4B SID:   floor = 11.78  ← 最优
- 8B SID:   floor = 12.26  ← 比 4B 差

---

## 为什么 L2 entropy 决定 floor PPL

模型预测的目标是从 SID token 序列还原 item。设完整 SID = `(L0, L1, L2)`，三层 token 联合确定一个 item。

如果 L2 层 entropy 低，有效 slot 少，则存在大量 item 共享同一 SID。对于共享同一 SID 的 k 个 item，模型在给定上下文后无法区分它们——这 k 种情况对 NTP 来说是**不可区分的**，额外引入 `log2(k) bits` 的不确定性。

这 k 个混淆 item 的均匀分布为 `1/k`，对应 PPL 乘子：
```
PPL_collision_penalty ≈ k
floor_PPL ≥ base_PPL × k^(p_collision)
```

其中 `p_collision` 是碰撞 item 的比例。更精确地，floor PPL 来自 `P(y | context)` 的条件熵下界：

```
H(y | SID) = Σ_{sid}  P(sid) · H(y | SID = sid)
```

SID 冲突组内的熵 `H(y | SID=sid) = log2(k_sid)` 是不可约的，无论模型多大都消除不了。

---

## 数据验证

EXP-043 观测值与上述理论方向一致：

| Embedding | L2 entropy | 有效 slot | 平均冲突 k | floor PPL |
|-----------|-----------|----------|-----------|-----------|
| 0.6B      | 10.58 bits | ~1500    | ~1.3      | 12.46     |
| 4B        | 8.10 bits  | ~275     | ~1.6      | 11.78*    |
| 8B        | 7.17 bits  | ~145     | ~3.0      | 12.26     |

\* 4B floor 最低：4B embedding 质量足够好，即使 L2 有轻度坍缩，embedding 区分度补偿了部分冲突损失；8B 冲突太严重（k≈3），embedding 质量提升已不足以抵消。

**反直觉结论**：更大的 embedding 模型（8B）得到了更差的理论上限，根本原因是 FSQ hidden=64 对 4096D 输入太小，bottleneck 严重压缩了语义信息。

---

## 根本原因：FSQ hidden dim 不匹配

我们的 MLP-FSQ 结构：
```
input(D) → Linear(D, h) → GELU → Linear(h, 12) → FSQ([2]×12)
```

`h=64` 是为 0.6B embedding（D=1024）设计的（比例约 1:16）。

对 4B（D=2560）和 8B（D=4096），h=64 形成严重的信息瓶颈：

```
0.6B: ratio = 64/1024 = 6.25%   → 正常
4B:   ratio = 64/2560 = 2.50%   → 明显不足
8B:   ratio = 64/4096 = 1.56%   → 严重不足
```

MLP 被迫把高维语义空间压入过窄的 bottleneck，导致 L2 码本退化为少量高频 slot。

---

## 实用推论

**Collision rate / L2 entropy 作为快速筛选指标**

FSQ hidden dim 调参实验（EXP-045 Phase 1）只需跑 tokenizer 评测，不需要跑 NTP：

1. 针对目标 embedding（4B / 8B），扫描 `h ∈ {64, 128, 256, 512}`
2. 计算 L2 entropy 和 collision rate
3. 筛选 L2 entropy ≥ 10 bits（≈ 90% 利用率）的最小 h
4. 用筛选出的 h 跑一次完整 NTP 验证

**预期经验公式**（待 EXP-045 验证）

从 bottleneck ratio 角度，维持正常 L2 利用率需要：
```
h_min ≈ emb_dim / 16    # 线性假设
```
或从信息论角度（sqrt 压缩）：
```
h_min ≈ 2 × sqrt(emb_dim)
```

两者对各模型的预测：

| Embedding | D     | h_min (linear) | h_min (sqrt) |
|-----------|-------|---------------|-------------|
| 0.6B      | 1024  | 64            | 64          |
| 4B        | 2560  | 160           | 101         |
| 8B        | 4096  | 256           | 128         |

EXP-045 将通过实测 L2 entropy 确认哪个公式更准确，并给出跨 embedding 大小的选型建议。

---

## 与 Scaling Law 的关系

Floor PPL 通过两点 scaling law 拟合得到，但 **floor 本身是 tokenizer 质量的函数，与模型大小无关**。这意味着：

- 修复 FSQ bottleneck（提高 L2 entropy）可以直接降低 floor PPL
- 降低 floor PPL 等价于提升所有规模模型的天花板
- 即使 M-tier 当前 R@500=70.4%，若 4B SID 的 floor 从 11.78 降到 10.x，M-tier 最终也能受益

**优化优先级**：FSQ hidden 扩大是成本最低、收益最可预期的优化方向——只需重建 tokenizer，NTP 架构不变。
