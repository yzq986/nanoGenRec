# 004: Prefix-Locked vs Paper Beam Search — SP-DPO Candidate Generation

**Date**: 2026-04-19
**Context**: EXP-017 SP-DPO 实现中发现论文 beam search 方法在 Medium/Hard 候选数量上存在固有瓶颈

---

## 问题

Align3GR 的 SP-DPO 用 beam search 生成 rejected candidates，然后按 prefix n-gram match 分难度：

| Difficulty | 定义 | 含义 |
|-----------|------|------|
| Easy | L0 ≠ GT | 粗粒度就不同 |
| Medium | L0 = GT, L1 ≠ GT | 粗粒度相同，中等难度 |
| Hard | L0+L1 = GT, L2 ≠ GT | 高度相似，仅细粒度不同 |

问题在于：**beam search 从 L0 开始自由采样，绝大多数 beam path 的 L0 不等于 GT**。

SID 体系是 4096×4096×4096（3 层，每层 4096 clusters）。SFT baseline 的 `depth_acc_beam` L0 只有 0.030（beam top-1 L0 命中率 3%）。即使 B=200：

- Easy candidates: ~190 条（L0 就不同，几乎所有 beam）
- Medium candidates: ~5-10 条（需要 L0 恰好命中 GT）
- Hard candidates: ~0-2 条（需要 L0+L1 都命中 GT）

这意味着 **Medium 和 Hard 阶段的训练数据严重不足**。论文没有讨论这个问题。

---

## 渐进式训练能否缓解？

论文的 self-play 渐进设计（Easy→Medium→Hard）本身有一定缓解作用：

1. Easy DPO 训练提升 L0 判别力（EXP-017 Easy: L0 acc 0.030→0.041，+37%）
2. 改进后的模型 beam search 会产出更多 L0 命中的 candidate → Medium 增多
3. Medium DPO 训练提升 L1 判别力 → Hard 增多

但这是一个**间接效应**，受限于 beam size 和模型改进幅度。如果每阶段的改进不够大（比如 Easy DPO 只把 L0 acc 从 3% 提到 4.1%），Medium/Hard 数据量仍然很少。

---

## Prefix-Locked 方案

直接锁定 GT 前缀，beam search 剩余层：

| 采样方式 | L0 | L1 | L2 | 保证产出 |
|---------|----|----|-----|---------|
| 论文 beam search | 采样 | 采样 | 采样 | 大部分 Easy |
| Lock L0=GT | **固定** | 采样 | 采样 | 全部 Medium+Hard |
| Lock L0+L1=GT | **固定** | **固定** | 采样 | 全部 Hard |

实现：`constrained_beam_search` 加 `prefix` 参数，跳过前 P 层的 beam search，直接从 layer P 开始展开。

对于每个 eval item，最多跑 3 次 beam search（渐进锁定）：
1. 完整 beam → Easy candidates
2. 锁 L0 → Medium + Hard candidates
3. 锁 L0+L1 → Hard candidates

跨 pass dedup，每个难度上限 `n_rejected=20`。

---

## 两种方法的本质区别

### 论文方法（beam search + classify）

```
P(rejected | context) ∝ model_score(context → rejected_sid)
```

rejected 是**模型认为最可能的错误答案**——beam search 自然排序，高分候选被选为 rejected。这些是模型"最容易犯的错误"。

### Prefix-locked 方法

```
P(rejected | context, prefix=GT[:p]) ∝ model_score(context → prefix + remaining)
```

rejected 是**在正确前缀约束下模型认为最可能的答案**。这些候选跟 GT 共享前缀，语义上更接近，但不一定是模型无约束下会犯的错误。

### 关键问题：哪种 rejected 更有效？

**论文方法的 Hard candidates**（如果有的话）：
- 模型在自由 beam search 中恰好走对了 L0+L1，但 L2 走错
- 这是模型**真实的混淆模式**——它确实会犯这个错
- 但数量极少

**Prefix-locked 的 Hard candidates**：
- 模型被迫从正确的 L0+L1 出发，选择最可能的 L2
- 数量充足（B=200 条 Hard candidates）
- 但这些错误不一定是模型在自由生成时会犯的
- 类似于"把学生按到正确答案前两步，然后看最后一步犯什么错"

**可能的结果**：
1. Prefix-locked 更好：充足的 Hard 数据 > 数据真实性，模型学到更精细的 L2 区分
2. 论文方法更好：虽然 Hard 数据少，但每条都是真实混淆，信号更强
3. 差不多：DPO loss 本身对数据量不敏感（chosen vs rejected 的对比信号比绝对数量重要）

---

## 实验设计（EXP-017）

| Config | 采样 | 目的 |
|--------|------|------|
| Config 2 | Easy model beam B=200（论文方法） | self-play baseline |
| Config 3 | Easy model prefix-locked B=200 | 渐进锁定采样 |

两组共享 Easy 阶段，只有 Medium/Hard 的 candidates 不同。对比 eval 指标（PPL, Recall, depth_acc_beam）。

**预期观察**：
- Config 3 的 preference 统计应显示 Medium/Hard pairs 数量远多于 Config 2
- 训练 loss 行为可能不同（Config 3 DPO loss 可能更难下降，因为 locked prefix candidates 更接近 GT）
- Recall 改善不确定——取决于"数据量 vs 数据真实性"的 tradeoff

---

## 扩展思考

### 如果 prefix-locked 更好

说明论文的 beam search + classify 方案在深层（L1, L2）的信号太稀疏，渐进锁定是更好的 curriculum 策略。可以考虑：
- 更激进的锁定：Easy 用完整 beam，Medium/Hard 全部用 prefix-locked
- 动态 beam size：Easy B=50（足够），Medium locked B=100，Hard locked B=200

### 如果论文方法更好

说明"模型真实犯的错误"比"人为构造的困难样本"更有效。这对 RL alignment 有启示——**对比学习的 hard negative mining 不一定越难越好**，关键是 negative 要在模型的实际错误分布上。

### 混合方案

两者不互斥。可以：
1. 完整 beam search 获取所有难度的"真实混淆"
2. Prefix-locked 补充不足的 Medium/Hard
3. 真实混淆样本权重更高（因为信号更强）
