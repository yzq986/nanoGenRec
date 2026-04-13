# RL Alignment (强化学习对齐)

用 RL/DPO/GRPO 将生成式推荐模型与业务目标对齐。属于 NTP 模型训练稳定后的进阶优化，前置依赖重。

**影响范围**: 新增 `model/rl_trainer.py`, `metrics/sid_prediction.py`

---

## 演进路径

```
NTP 纯监督学习 (当前 baseline)
├── IDEA-onemall-2: GRPO/DPO (OneMall 方案)
│   └── GRPO > DPO, 768 候选 normalized advantage
├── IDEA-oneloc-2: DPO + 双目标奖励 (OneLoc 方案)
│   └── popularity + diversity 双目标
└── IDEA-gr4ad-3: RSPO (GR4AD 方案)
    └── list-wise Lambda-weighted, NDCG-inspired
    └── 最强但前置依赖最重
```

---

## IDEA-onemall-2: GRPO/DPO 强化学习对齐

**优先级**: P1
**来源**: OneMall §3.3 Reinforcement Learning Policy
**状态**: 待讨论

### 核心思想

用 RL 将检索模型 (generative NTP) 与排序模型对齐。具体做法:
1. **Reward Model**: 线上排序模型（使用全量 user/item/cross features）作为 reward model，输出 CTR/CVR/EGPM 预测
2. **Reference Model**: 定期从 policy model 同步参数，用 beam search 采样候选集
3. **Policy Optimization**: GRPO 或 DPO 优化 policy model

OneMall 关键发现:
- **GRPO > DPO**: GRPO 在所有候选段 (Top10/100/500) 均优于 DPO
- GRPO 对全部 768 个采样候选计算 normalized advantage，DPO 仅用 pairwise
- RL loss weight = 0.5，过大会降低 SID accuracy
- 仅用 2% 训练样本做 RL

### 与当前项目的关联

- 当前 **完全没有 RL 相关代码**，属于全新能力建设
- NTP 模型已有 beam search 基础设施 (`BeamSearchModule`)，可复用于候选采样
- 没有线上排序模型，需要构造 proxy reward:
  - 方案 A: 离线 CTR 预估模型作为 reward
  - 方案 B: 基于行为数据的 reward (clicked=1, bought=5, exposed_not_clicked=0)
  - 方案 C: embedding 相似度作为 reward (简单但弱)
- **依赖 NTP 模型先达到合理基线性能**，否则 RL fine-tuning 无意义

### 实验设计草案

**阶段 1: Offline Reward Model (简化版)**

构造 reward 函数:
```
r(user, item) = α * is_clicked + β * is_bought + γ * embedding_sim
```

**阶段 2: GRPO Implementation**

新增 `model/rl_trainer.py`:
1. Reference model: 冻结的 NTP model checkpoint
2. Policy model: 当前训练中的 NTP model
3. 每个 user query: beam search 采样 N 个候选 (N=64~256，受限于 GPU 内存)
4. 对每个候选计算 reward → normalize → advantage
5. GRPO loss: clipped importance-weighted advantage (clip ratio 1±0.2)

**Joint loss**:
```
L = L_NTP + 0.5 * L_GRPO + α * L_contrastive
```

**基线**: NTP-only (no RL)

**评估**: 离线 Recall@K + reward score 分布变化

### 关键问题

1. **Reward model 质量是根本瓶颈**: 没有线上排序模型，proxy reward 可能引入 bias
2. **采样成本高**: 每个 query 做 beam search 采样 N 个候选 → 训练速度可能下降 10x+
3. **Reference model 同步频率**: OneMall 未详细说明，需要实验确定
4. **建议先完成 IDEA-onemall-0 (contrastive loss)，建立更强的 NTP 基线后再做 RL**

---

## IDEA-oneloc-2: DPO 对齐 + 双目标奖励函数

**优先级**: P1
**来源**: OneLoc §2.5 Reinforcement Learning
**状态**: 待讨论

### 核心思想

预训练的 NTP 模型只拟合曝光数据，无法做细粒度多目标平衡。OneLoc 用 DPO 做后对齐:
1. 用预训练模型 beam search 生成 N 个候选
2. 用奖励函数 (地理距离 + GMV) 对候选打分
3. 取最高分为 positive、最低分为 negative，构造 preference pair
4. DPO loss 联合 NTP loss 训练: `L = L_ntp + λ·L_dpo`

### 与当前项目的关联

- **当前项目零 RL/DPO 代码**，这是全新的模块
- ARCHITECTURE.md 中 OneRec V1 paper 也用了 RL alignment，说明这不是 OneLoc 特有的
- 对我们的意义: **用 DPO 来对齐 NTP 模型到业务目标**，例如:
  - 奖励 1: item popularity / CTR 预估分 (替代 OneLoc 的 GMV)
  - 奖励 2: diversity / category coverage (替代 OneLoc 的地理距离)
- DPO 比 PPO 简单得多，不需要 critic 网络，只需要 preference pairs

### 实验设计草案

**Step 1: 构造奖励函数**
- `R_popularity(v)`: item 的历史 CTR 或 interaction count (已有 `data/export_behavior.py`)
- `R_diversity(v, S)`: 推荐 item 与历史序列的 category 差异度

**Step 2: 生成 preference pairs**
- 用训练好的 NTP 模型 beam search 生成 top-N (N=50) 候选
- 对每个候选计算 reward score
- 选 top-1 为 positive, bottom-1 为 negative

**Step 3: DPO 训练**
- 在 `model/train.py` 或新文件中实现 DPO loss
- 关键超参: λ (DPO 权重, OneLoc 用 0.05), β (DPO temperature)
- 训练: 先 NTP 预训练 → 冻结 reference model → NTP + DPO 联合训练

**评估**:
- 预训练 only vs DPO-aligned: recall, NDCG
- DPO-aligned 的 reward 分布变化 (推荐的 item 是否更符合目标)

### 关键问题

1. **前置依赖**: 需要 NTP 模型先训练到足够好 (当前 `AutoregressiveNTPModel` 可能还需要架构升级)
2. 奖励函数设计: 用什么替代 GMV 和地理距离? 需要与业务目标对齐
3. 负样本质量: beam search 的 bottom-1 是否真的是 "bad" recommendation? 可能需要更精细的 pair 构造
4. 计算成本: 每个训练样本需要一次 beam search → N 个候选 → reward scoring，训练速度可能大幅下降
5. **优先级判断**: RL alignment 是"锦上添花"，应在基础 NTP 模型和量化方案稳定后再做

---

## IDEA-gr4ad-3: RSPO 排序优化 (Ranking-Guided Softmax Preference Optimization)

**优先级**: P2
**来源**: GR4AD §RSPO, Table 1
**状态**: 待讨论

### 核心思想

GR4AD 提出 list-wise RL 方法 RSPO: 将 beam search 产出的候选列表按 eCPM 排序，用 NDCG-inspired Lambda 权重做偏好优化。相比 DPO (+0.70%) 和 GRPO (+0.65%)，RSPO 带来 +1.06% 增量。核心创新: (1) Lambda 权重 ℳᵢⱼ 关注排序位置交换的 NDCG 收益；(2) Reference gating Cᵢⱼ 在参考模型不可靠时自动关闭 KL 约束。

### 与当前项目的关联

- 当前 NTP 模型只做监督学习，没有任何 RL/preference optimization
- **前置依赖重**: 需要先有 (1) 合理的 reward signal（IDEA-gr4ad-2 的价值 token）；(2) 足够好的 beam search 产出多个候选（当前 beam=5 候选太少）
- 实现复杂度高: 需要 reference model、reward model、Lambda NDCG 计算、online learning pipeline
- 更适合作为系统成熟后的进阶优化

### 实验设计草案

**简化版 — Offline DPO 起步**:
1. 用当前 NTP 模型 beam search 产出 top-K 候选
2. 按行为数据构造偏好对: 被点击的 item > 未被点击的 item
3. 先实现 DPO loss 验证框架，再升级到 RSPO

**进阶版 — RSPO**:
- 在 DPO 基础上替换 pairwise loss 为 list-wise Lambda-weighted softmax loss
- 加入 reference gating 机制

**评估**: Hit@K, NDCG@K, 与纯 SL 模型对比

### 关键问题

1. **数据要求高**: 需要同一 context 下多个候选的真实反馈，当前 demo 数据不一定有
2. 训练稳定性: RL 方法调参困难，reference model 需要定期更新
3. 收益依赖于 beam search 质量 — 如果 beam search 本身不够好（候选同质化），排序优化价值有限
4. 建议优先级在 IDEA-gr4ad-0/gr4ad-1/gr4ad-2 之后

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P1 | IDEA-onemall-2 | GRPO/DPO 强化学习 | 战略重要但依赖强基线 + reward model |
| P1 | IDEA-oneloc-2 | DPO + 双目标奖励 | DPO 比 PPO 简单，可作为 RL 入门 |
| P2 | IDEA-gr4ad-3 | RSPO 排序优化 | 收益最大但前置依赖最重 |
