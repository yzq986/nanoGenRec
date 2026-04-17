# 002: 从 SP-DPO 到 ECPO —— RL 对齐的完整递进路径

**Date**: 2026-04-17
**Context**: 承接 001 的 SP-DPO 讨论，向上追问：RF-DPO、GRPO、ECPO 各解决什么问题？

---

## 背景

[001](001-sp-dpo-vs-sft-vs-contrastive.md) 澄清了 SP-DPO 和 SFT / 对比学习的本质区别。
本篇继续往上走：SP-DPO → RF-DPO → GRPO → ECPO，每一步都解决前一步留下的问题。

---

## SP-DPO 留下了什么问题？

SP-DPO 的 chosen 是 ground truth，rejected 是模型自己的生成。
评判标准是**纯结构性的** —— prefix n-gram 匹配度。

问题：**它只知道"预测对不对"，不知道"推荐好不好"**。

举例：模型生成了候选 B，虽然不是 ground truth，但用户可能也喜欢（只是没曝光）。
SP-DPO 把 B 当负样本压低了 —— 这是冤枉的。
反过来，ground truth 可能是用户随便点的低质量内容，SP-DPO 却把它当正样本推高。

> SP-DPO 对齐的是"预测准确性"，而不是"推荐质量"。

---

## RF-DPO：引入真实用户反馈

**来源**: Align³GR (快手, AAAI 2026 Oral, arxiv 2511.11255)

**核心改进**：用**用户真实行为**替换 SP-DPO 的结构性信号。

```
SP-DPO:  chosen = ground truth,  rejected = 模型生成的（按 prefix 匹配打分）
RF-DPO:  chosen = 用户喜欢的,    rejected = 用户不喜欢的（真实反馈）
```

用户反馈分三档：

| 反馈 | 信号 | 例子 |
|------|------|------|
| Liked | 显式正向 | 点赞、购买、收藏 |
| Neutral | 隐式负向 | 曝光未点击 |
| Disliked | 显式负向 | 点了"不感兴趣" |

RF-DPO 同样分两阶段渐进：

```
Easy 阶段:  chosen = liked,  rejected = disliked   （容易区分）
Hard 阶段:  chosen = liked,  rejected = neutral    （微妙区分：为什么曝光了却没点？）
```

### 跟 SP-DPO 的关键区别

| | SP-DPO | RF-DPO |
|--|--------|--------|
| 信号来源 | 模型自博弈 | 用户真实行为 |
| 评判标准 | 预测准不准 | 用户喜不喜欢 |
| 对齐目标 | 生成准确性 | **业务目标**（CTR/转化/留存） |
| 负样本质量 | 可能误伤好 item | 由用户行为定义，更可靠 |

### 消融结果 (Align³GR, Instruments 数据集)

| 方法 | Recall@10 | vs SP-DPO |
|------|-----------|-----------|
| Progressive SP-DPO | 0.1396 | — |
| + RF-DPO (无渐进) | 0.1414 | +1.3% |
| + RF-DPO (有渐进) | **0.1442** | **+3.3%** |

### RF-DPO 的局限

RF-DPO 还是 DPO —— 它只看 pairwise 偏好（chosen vs rejected），
每个样本只有 1 个 chosen 对 20 个 rejected。
如果模型生成了 500 个候选，DPO 只用了其中 21 个，其余 479 个的信息全丢了。

---

## GRPO：从 pairwise 到 group-wise

**来源**: OneMall (arxiv 2601.21770), OneRec (arxiv 2506.13695v4)

### DPO 的局限

每次只比较一对（chosen vs rejected），信息效率低。

### GRPO 的做法：对整组候选计算相对优势

```
DPO:   1 个 chosen vs N 个 rejected → pairwise 比较
GRPO:  G 个候选，每个都有 reward 分数 → group-wise 相对排序
```

### GRPO 流程

```
Step 1: 对每个用户，模型 beam search 生成 G 个候选（G=512）
Step 2: reward model 对每个候选打分 → r_1, r_2, ..., r_G
Step 3: 组内归一化，计算 advantage:
        A_i = (r_i - mean(r)) / std(r)
Step 4: 用 clipped surrogate loss 优化（跟 PPO 一样的 clip 机制）
```

### 跟 DPO 的核心区别

| | DPO | GRPO |
|--|-----|------|
| 比较方式 | 1 chosen vs N rejected（二分法） | G 个候选连续排序（相对优势） |
| 信息利用 | 只用最好和最差 | **全部候选都产生梯度** |
| Reward 来源 | 隐式（偏好对蕴含的） | **显式 reward model 打分** |
| 优化目标 | 拉开 chosen/rejected 的概率差距 | 按 advantage 加权调整所有候选的概率 |

### 具体例子

模型生成了 5 个候选，reward model 打分：

| 候选 | Reward | Advantage (归一化后) | GRPO 做什么 |
|------|--------|---------------------|------------|
| A | 0.9 | +1.5 | 大幅推高概率 |
| B | 0.7 | +0.5 | 适度推高 |
| C | 0.5 | 0.0 | 几乎不动 |
| D | 0.3 | -0.5 | 适度压低 |
| E | 0.1 | -1.5 | 大幅压低 |

DPO 只会看到 A（chosen）vs E（rejected），中间的 B/C/D 完全被忽略。
GRPO 给 5 个候选都分配了不同强度的梯度信号 —— **信息效率高得多**。

### GRPO 的新前置依赖：Reward Model

DPO 不需要 reward model（偏好对本身隐式定义了 reward），GRPO 需要。

工业实践中的 reward model：
- **OneMall**: 线上排序模型（使用全量 user/item/cross features），输出 CTR/CVR/EGPM 预测
- **OneRec**: 专用 **P-Score 模型** — 多塔结构，每塔学一个目标（ctr/lvtr/ltr/vtr），最终 MLP 融合

我们的选项（无线上排序模型）：
- 行为信号：`clicked=1, bought=5, exposed_not_clicked=0`
- 离线 CTR 预估模型
- Embedding similarity（简单但弱）

### GRPO 的关键发现 (OneMall)

- **GRPO > DPO**: 在所有候选段 (Top10/100/500) 均优于 DPO
- RL loss weight = 0.5，过大会降低 SID accuracy
- 仅用 2% 训练样本做 RL
- 对全部 768 个采样候选计算 normalized advantage

---

## ECPO：修复 GRPO 的训练稳定性

**来源**: OneRec (arxiv 2506.13695v4)

### GRPO 的工程问题：负 advantage 梯度爆炸

标准 GRPO 用 PPO 风格的 clip：

```
ratio = π_θ(o|u) / π_old(o|u)
loss = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

- 当 A > 0（好候选）：clip 防止 ratio 超过 1+ε → **稳定**
- 当 A < 0（坏候选）：clip 只防止 ratio 低于 1-ε，**但不防止 ratio 变得很大** → **梯度爆炸**

直觉理解：对于一个坏候选，模型在训练中可能突然把它的概率降得很低（π_θ → 0），
导致 ratio = π_θ/π_old 变得极大，乘以负的 A，产生巨大的负梯度。

### ECPO 的修复：Early Clipping

```
π'_old = max( sg(π_θ) / (1+ε+δ),  π_old )

当 A < 0 时，用 π'_old 替换 π_old 作为分母
→ 保证 ratio 永远不超过 1+ε+δ
```

δ = 0.1，意味着坏候选的 ratio 最多到 1+ε+0.1，不会爆炸。

另外，ECPO **去掉了 KL 散度项**：因为 OneRec 同时训练 RSFT（监督微调）和 RL，
SFT loss 本身就起到了防止模型跑偏的作用，不需要额外的 KL 约束。

### Format Reward：保证生成合法性

RL 训练后模型可能生成非法 SID（不对应真实 item），合法率从 >95% 掉到 <50%。
原因是 **squeezing effect**：负 advantage 压缩分布时，合法 token 的概率被压到跟非法 token 差不多。

解决方案：

```
对随机采样的 K=5 个候选（从 128 个中随机选）：
  合法 SID → advantage = 1（奖励）
  非法 SID → advantage = 0（不惩罚，只是不奖励）
```

为什么用**随机采样**而不是 top-k？
- top-k 只让排名最高的几个变合法，其余照样非法 — 窄修复
- 随机采样让整个分布都趋向合法，最终合法率稳定在 95%

Format reward 的在线效果：**+0.13% App Stay Time, +0.30% Watch Time**。

### ECPO 关键超参 (OneRec)

| 参数 | 值 | 说明 |
|------|-----|------|
| δ (early clip margin) | 0.1 | 控制负 advantage 的最大 ratio |
| Group size (G) | 512 | ~4× inference Pass@K |
| RL 用户比例 | 1% | 从 RSFT 数据流中随机抽取 |
| Format reward K | 5 | 从 128 候选中随机选 |
| 采样策略 | Beam search | 优于 top-k/top-p（因为 SID 是 trie 结构） |
| Reference model | On-policy | 优于 off-policy（离线），但在线差异小 |

### Group Size 消融 (OneRec, 在线)

| Group Size | Watch Time | App Stay Time |
|-----------|------------|---------------|
| 0 (no RL) | +4.61% | +1.11% |
| 128 | +5.22% | +1.49% |
| 512 | +5.73% | **+1.82%** |
| 2048 | +5.84% | +1.78% |

128→512 跳跃显著，512→2048 边际递减。经验法则：group size ≈ 4× inference Pass@K。

---

## 完整谱系总结

```
SFT           "把正确答案概率推高"
 │            ✗ 不知道模型犯了什么错
 │            ✗ 不区分"好推荐"和"准确预测"
 ▼
SP-DPO        "把正确答案推高，同时压低模型自己的错误"
 │            ✓ 针对模型当前的混淆区域
 │            ✗ 只对齐预测准确性，不对齐业务目标
 │            ✗ 可能误伤好 item
 ▼
RF-DPO        "用真实用户反馈定义好/坏"
 │            ✓ 对齐业务目标（用户喜好）
 │            ✗ 只看 pairwise，信息效率低
 │            ✗ 不需要 reward model（直接用行为信号）
 ▼
GRPO          "对整组候选按 reward 排序，group-wise 优化"
 │            ✓ 全部候选都产生梯度，信息效率高
 │            ✓ 连续 reward 信号，不只是二分法
 │            ✗ 需要 reward model
 │            ✗ 负 advantage 候选梯度可能爆炸
 ▼
ECPO          "GRPO + early clip 修复稳定性 + format reward 保合法性"
              ✓ 训练稳定
              ✓ 生成合法性有保证
              ✗ 工程复杂度最高（reward model + 采样 + clip + format reward）
```

---

## 对我们项目的实操路径

| 阶段 | 方法 | 前置依赖 | 复杂度 | 新增模块 |
|------|------|---------|--------|---------|
| 1 | SP-DPO | 只需当前 NTP 模型 | 低 | DPO trainer + preference pair 构造 |
| 2 | RF-DPO | 需要用户行为标签（clicked/not-clicked） | 低-中 | 行为反馈 pair 构造 |
| 3 | GRPO | 需要训练 reward model | 中-高 | reward model + group sampling |
| 4 | ECPO | 在 GRPO 基础上改 clip + 加 format reward | 高 | early clip + format reward + RSFT |

**当前建议**：从 SP-DPO 起步（前置依赖最少），验证 DPO 框架后逐步升级。
