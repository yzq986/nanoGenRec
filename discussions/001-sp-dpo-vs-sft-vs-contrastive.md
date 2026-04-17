# 001: SP-DPO vs SFT vs Contrastive Learning

**Date**: 2026-04-17
**Context**: NTP 模型训练稳定后，讨论 RL alignment 的入门方案

---

## 背景

SP-DPO (Self-Play DPO) 来自 Align³GR（快手，AAAI 2026 Oral，arxiv 2511.11255）。
在讨论过程中，两个核心问题浮现：SP-DPO 跟 SFT 有什么区别？跟对比学习又有什么区别？

---

## Q1: SP-DPO 和 SFT 的区别是什么？

### 表面上看确实很像

| | SFT (当前 NTP) | SP-DPO |
|--|----------------|--------|
| 正样本 | ground truth | ground truth |
| 负样本 | 无（隐式） | 模型自己生成的 |

SFT 也是拿 ground truth 训练，SP-DPO 的 chosen 也是 ground truth —— 区别在哪？

### 核心区别：loss 函数做的事不一样

**SFT（交叉熵）**：只说"把正确答案的概率推高"

```
L_SFT = -log P(ground_truth | context)
```

它**完全不关心**模型给错误候选打了多高的分。假设模型当前：
- P(正确 item) = 0.3
- P(很像但错误的 item) = 0.25

SFT 只管把 0.3 往上推。但因为这两个 item 的 SID 前缀相同（比如 L1、L2 都一样），
推高正确 item 的概率时，**错误 item 的概率很可能也跟着涨**，因为它们共享前两层的
token embedding 梯度。

**DPO**：说的是"拉大正确和错误之间的差距"

```
L_DPO = -log σ( β · [log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)] )
                      ~~~~~~~~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~~
                        推高 chosen 的概率            推低 rejected 的概率
```

它**同时做两件事**：推高 chosen，**并且显式压低 rejected**。

### 一个具体例子

用户历史 → ground truth 下一跳 SID = `[10, 20, 30]`

模型 beam search 生成了这些候选：

| 候选 | SID | 模型给的概率 |
|------|-----|------------|
| A | [10, 20, 30] | 0.15 (正确) |
| B | [10, 20, 77] | 0.12 (L1L2 对了，L3 错了) |
| C | [10, 55, 88] | 0.10 (只 L1 对了) |
| D | [99, 88, 77] | 0.03 (完全错) |

**SFT 看到的**：只有 A 是 target，算 `-log 0.15`，反向传播。B、C、D 的存在它完全不知道。

**SP-DPO 看到的**：
- chosen = A
- rejected = {B, C, D}
- loss 目标：把 A 和 B 之间的概率差距拉开，把 A 和 C 之间的差距拉开...

**关键**：B（`[10, 20, 77]`）跟正确答案只差一个 token，SFT 的梯度根本区分不了它俩
（前两步的 CE loss 对 B 也是正向的）。但 SP-DPO **显式地说"B 是错的，压低它"**。

### 一句话总结

> **SFT 只告诉模型"什么是对的"，SP-DPO 额外告诉模型"你当前犯的哪些错误是错的"。**

SFT 是绝对信号（maximize ground truth），DPO 是对比信号（widen the gap）。
SP-DPO 的"自博弈"本质就是：**用模型自己的错误作为针对性的负样本**，
而不是 SFT 那样对所有非 ground truth 一视同仁。

### 但也要保持清醒

SP-DPO 跟"真正的 RL"（有外部 reward model 的 GRPO/ECPO）比起来，确实更接近 SFT
的范畴。论文消融也说明了这一点 —— SP-DPO 单独的增益只有 **+4.7%~7.8%**，真正大的
提升来自后面的 RF-DPO（引入真实用户反馈）。SP-DPO 更像是一个
**SFT → 真正 RL 之间的过渡方案**。

---

## Q2: SP-DPO 和对比学习的区别是什么？

对比学习也是"拉近正样本、推远负样本"，和 DPO 做的事情看起来一模一样。

### 形式上的相似性

**对比学习（InfoNCE）**：
```
L = -log [ exp(sim(anchor, pos)) / Σ exp(sim(anchor, neg_i)) ]
```
推高 anchor 和 positive 的相似度，推低和 negative 的相似度。

**DPO**：
```
L = -log σ( β · [log π(chosen|x)/π_ref(chosen|x) - log π(rejected|x)/π_ref(rejected|x)] )
```
推高 chosen 的生成概率，推低 rejected 的生成概率。

结构高度相似 —— 都是对比式的 loss。但区别在三个地方：

### 区别 1：优化的对象不同

| | 对比学习 | DPO |
|--|---------|-----|
| 作用在 | **表示空间**（embedding） | **生成概率**（token-by-token） |
| 输出 | 一个向量，算 cosine similarity | 一个序列的联合概率 P(L1)·P(L2\|L1)·P(L3\|L1,L2) |
| 粒度 | item 级别："这个 item 整体像不像" | token 级别："在 L1=10, L2=20 的条件下，L3 应该选 30 而不是 77" |

这是最本质的区别。SID 是 3 步自回归生成的，DPO 调整的是**每一步的条件概率**。

举例：chosen=`[10,20,30]`，rejected=`[10,20,77]`

- **对比学习**：把这两个 SID 视为两个"整体"，在 embedding 空间拉远。但它不知道
  "前两步是对的，只有第三步错了"。
- **DPO**：P(L3=30 | L1=10, L2=20) 要升高，P(L3=77 | L1=10, L2=20) 要降低。
  精确到第三步条件概率的修正。

### 区别 2：有 reference model 约束

DPO 的 loss 里有 **π_ref**（冻结的 SFT 模型），对比学习没有：

```
DPO 优化的不是 "让 chosen 概率最大化"
而是   "让 chosen 相对于 π_ref 的提升 > rejected 相对于 π_ref 的提升"
```

这个约束防止模型跑太远 —— 本质上等价于 RL 中的 KL penalty。

对比学习没有这个锚点，表示空间可以自由漂移，容易发生 **representation collapse**
或者把 SFT 阶段学到的知识冲掉。

### 区别 3：理论来源不同

- **对比学习**：来自度量学习 / 自监督学习，目标是学好表示
- **DPO**：来自 RLHF，是 KL 约束下 reward 最大化的**闭式解**

DPO 等价于：
```
max_π  E[reward(chosen)] - β·KL(π || π_ref)
```
只不过不需要显式训练 reward model，直接用 preference pair 隐式优化。
对比学习没有这个 RL 对应关系。

### 本项目的直接证据

EXP-007 做过 **I2I 对比学习微调 tokenizer embedding**，结论是**无效**。
对比学习在 embedding 空间拉远不同 item，但这对下游自回归生成的帮助有限 ——
因为 NTP 模型的瓶颈不在"表示不够好"，而在"生成决策不够准"。

DPO 直接作用在生成决策层面，所以理论上更 match 这个问题。

### 一句话总结

> **对比学习在表示空间说"这俩不像"，DPO 在生成空间说"在这个上下文下，应该生成这个而不是那个"。**

---

## 方法谱系

```
SFT（只有正样本）
  → 对比学习（加负样本，在 embedding 空间优化）
    → DPO（加负样本，在生成概率空间优化，带 KL 约束）
      → GRPO/PPO（真正的 RL，带显式 reward model）
```

SP-DPO 处在中间位置 —— 比对比学习更适合生成式模型，但比 GRPO 更轻量。

---

## SP-DPO 补充：Align³GR 论文细节

### 基本流程

```
NTP 模型 (SFT 训练好的)
    │  beam search 生成候选 SID
    ▼
模型自己的生成结果 = rejected (负样本)
用户真实下一跳 item  = chosen   (正样本)
    │
    ▼
构造 preference pair → Softmax-DPO loss 优化
```

### Prefix N-gram 难度定义

利用 SID 3 层层级结构定义负样本难度：

| 难度 | Prefix 重叠 | 含义 |
|------|------------|------|
| Easy | 无共享前缀 | 完全不相关，容易区分 |
| Medium | L1 相同 | 粗粒度类目相同，中等难度 |
| Hard | L1+L2 相同 | 高度相似，仅细粒度不同 |

### 渐进式训练（Curriculum Learning）

```
Stage 1 (Easy)  → π_θ^1 成为下一阶段 π_ref
Stage 2 (Medium) → π_θ^2 成为下一阶段 π_ref
Stage 3 (Hard)  → π_θ^3 成为下一阶段 π_ref (SP-DPO 结束)
Stage 4 (RF-DPO Easy:  liked vs disliked)
Stage 5 (RF-DPO Hard:  liked vs neutral)
```

### 关键超参

| 参数 | 值 |
|------|-----|
| rejected 数量 | 20 / sample |
| SP-DPO 阶段 | 3 (Easy→Medium→Hard) |
| RF-DPO 阶段 | 2 (Easy→Hard) |
| Loss | Softmax-DPO (支持 1 chosen vs N rejected) |

### 消融结果

| 方法 | Recall@10 | vs baseline |
|------|-----------|-------------|
| Softmax-DPO (普通) | 0.1295 | — |
| SP-DPO (无渐进) | 0.1356 | +4.7% |
| SP-DPO (有渐进) | 0.1396 | +7.8% |
| + RF-DPO (有渐进) | **0.1442** | **+11.4%** |
