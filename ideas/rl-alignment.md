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
├── IDEA-align3-0: Progressive DPO (Align³GR, 快手, AAAI 2026 Oral)
│   └── SP-DPO (自博弈) → RF-DPO (真实反馈), 无需外部 reward model
├── IDEA-rankgr-0: Listwise DPO + Rescore (RankGR, 淘宝)
│   └── IAP (listwise DPO 解码) + RSP (轻量 rescore), 近万 QPS
├── IDEA-uni-0: Search Preference Optimization (UniSearch, 快手)
│   └── reward model + user feedback, 搜索场景
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

## IDEA-align3-0: Progressive DPO (SP-DPO → RF-DPO 三层对齐)

**优先级**: P1
**来源**: Align³GR (Kuaishou, arxiv 2511.11255, Nov 2025, AAAI 2026 Oral)
**状态**: 待讨论

### 核心思想

Align³GR 提出统一三层对齐框架:

1. **Token-level Alignment**: Dual tokenization 融合语义和协同信号 (与 IDEA-pit-0 有关联)
2. **Behavior Modeling-level Alignment**: 双向语义对齐增强行为建模
3. **Preference-level Alignment**: **Progressive DPO** — 先 SP-DPO (自博弈) 再 RF-DPO (真实反馈):
   - **SP-DPO (Self-Play DPO)**: 模型自己生成候选集，按 reward 排序构造 preference pairs → 不需要外部 reward model
   - **RF-DPO (Real-Feedback DPO)**: 用真实用户行为反馈替换自博弈 reward → 更准确的对齐信号

**结果**: Recall@10 +17.8%, NDCG@10 +20.2% (offline). 快手工业部署在线 A/B 显著提升。AAAI 2026 Oral。

### 与当前项目的关联

- 直接强化 IDEA-onemall-2 (GRPO/DPO): Progressive DPO 是一种更稳定的 DPO 训练策略
- **SP-DPO 解决了"没有外部 reward model"的痛点**: 用自博弈替代外部 reward
- 当前项目没有线上排序模型做 reward，SP-DPO 是最实际的 RL 入门方案
- RF-DPO 阶段可以用行为数据 (clicked > not-clicked) 替代线上反馈

### 实验设计草案

**Phase 1 — SP-DPO**:
1. 训练好 NTP baseline
2. 用 NTP 模型 beam search 生成 top-K 候选
3. 用简单 reward (embedding similarity to ground truth) 对候选排序
4. Top-1 = positive, Bottom-1 = negative → DPO loss

**Phase 2 — RF-DPO**:
1. 用行为数据: clicked item = positive, exposed-but-not-clicked item = negative
2. 替换 SP-DPO 的 preference pairs

### 关键问题

1. 与 IDEA-onemall-2 (GRPO) 的关系: Progressive DPO 和 GRPO 哪个更好? 可以做对比实验
2. SP-DPO 的 reward 设计: 用什么作为 "自博弈" 的评判标准

---

## IDEA-rankgr-0: Listwise DPO + Two-Phase Decode-Rescore

**优先级**: P1
**来源**: RankGR (Alibaba/Taobao, arxiv 2602.08575, Feb 2026)
**状态**: 待讨论

### 核心思想

RankGR 将生成式检索分为两阶段:

1. **Initial Assessment Phase (IAP)**: 在自回归解码中注入 **listwise DPO**，让模型理解候选间的偏序关系
2. **Refined Scoring Phase (RSP)**: 对 IAP 的 top-λ 候选，用轻量评分模块重新打分 (建模候选与输入序列的交互)

两阶段在统一 GR 模型中联合优化。淘宝"猜你喜欢"在线验证 + 近万 QPS 实时服务。

### 与当前项目的关联

- **直接增强 IDEA-gr4ad-3 (RSPO)**: RankGR 的 listwise DPO 是 RSPO 的工业验证版本
- RSP (rescore 阶段) 是新技术: 不需要外部 reranking 模型，在 GR 模型内部加一个轻量 scorer
- 可以与 IDEA-gr4ad-4 (Dynamic Beam Search) 配合: IAP 用小 beam 快速筛选，RSP 对 top candidates 精细打分

### 实验设计草案

**Phase 1 — Listwise DPO in NTP**:
- 训练 NTP 时，对同一 user 的 beam search 结果按行为信号排序
- 构造 listwise preference → DPO loss

**Phase 2 — RSP Module**:
- 在 NTP decoder 输出端加一个 cross-attention scorer
- 输入: user 行为序列 + 候选 SID → 输出: 精细分数
- 用分数 rerank top-K 候选

### 关键问题

1. RSP 的计算开销: 对每个 top-λ 候选做 cross-attention，推理延迟增加多少
2. 与 IDEA-gr4ad-4 (Dynamic Beam) 的配合: beam 产出 → RSP rescore → 最终 top-K

---

## IDEA-uni-0: Search Preference Optimization (SPO)

**优先级**: P2
**来源**: UniSearch (Kuaishou, arxiv 2509.06887, Sep 2025)
**状态**: 待讨论

### 核心思想

UniSearch 用 **Search Preference Optimization (SPO)** 将 reward model 和用户真实反馈融入生成式搜索:

1. 训练 reward model 对生成候选打分
2. 用真实用户反馈 (点击、停留时长) 作为额外信号
3. 将 reward 信号通过 preference optimization 注入生成器

快手直播搜索部署: **近年最大单次实验提升**。

### 与当前项目的关联

- SPO 本质上是 GRPO/DPO 的搜索场景特化版本
- 与 IDEA-onemall-2 (GRPO) 和 IDEA-align3-0 (Progressive DPO) 有重叠
- 独特价值: reward model 的训练方法和 user feedback 的融合方式可以借鉴
- **优先级低**: 当前不做搜索场景，核心 RL 技术已被其他 idea 覆盖

### 关键问题

1. 与 IDEA-onemall-2 / IDEA-align3-0 的去重: SPO 的独特贡献是什么?
2. 当前无搜索场景，价值有限

---

## IDEA-onerec-3: ECPO (Early Clipped GRPO) + Format Reward

**优先级**: P1
**来源**: OneRec (arxiv 2506.13695v4) §ECPO + §Format Reward
**状态**: 待讨论

### 核心思想

OneRec 对 GRPO 做了两个关键改进:

**1. ECPO (Early Clipped GRPO)**:
标准 GRPO 的 clipping 对负 advantage 样本不够激进 — policy 仍然可能给坏样本分配较高概率。ECPO 引入 **early clipping**: 当样本 advantage 为负时，用更紧的 clip 上界压制 policy ratio:

$$\pi_{\theta_{old}}'(o_i|u) = \max\left(\frac{\text{sg}(\pi_\theta(o_i|u))}{1+\epsilon+\delta}, \pi_{\theta_{old}}(o_i|u)\right)$$

$\delta=0.1$，让坏样本更快被压制。同时因为 RSFT 和 RL 并行训练，移除了 KL 散度项。

**2. Format Reward (合法性奖励)**:
RL 训练中模型可能生成非法 SID token（不在 codebook 中）。Format Reward 给合法输出 advantage=1，非法输出 advantage=0，作为独立的 reward 信号:

$$A_i = \begin{cases} 1 & \text{if } o_i \in I_{\text{legal}} \\ 0 & \text{if } o_i \notin I_{\text{legal}} \end{cases}$$

关键发现: 用 **random sampling** 而非 top-k 选择候选做 format reward，否则合法性先升后降。

### 与当前项目的关联

- IDEA-onemall-2 已有 GRPO 基础设计，ECPO 是直接升级
- Format Reward 解决了一个实际问题: beam search 生成无效 SID（IDEA-static-0 的 CSR 约束解码也解决此问题，但在训练侧而非推理侧）
- OneRec 实验: group_size=512 最优 (+1.82% App Stay Time)，约为推理 Pass@K 的 4 倍

### 关键问题

1. 依赖 NTP 模型足够好才有意义 — 当前阶段先不做
2. RSFT 与 RL 并行训练的工程复杂度高

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P1 | IDEA-onemall-2 | GRPO/DPO 强化学习 | 战略重要但依赖强基线 + reward model |
| P1 | IDEA-oneloc-2 | DPO + 双目标奖励 | DPO 比 PPO 简单，可作为 RL 入门 |
| P1 | IDEA-align3-0 | Progressive DPO (SP→RF) | SP-DPO 解决无 reward model 痛点，AAAI 2026 Oral |
| P1 | IDEA-rankgr-0 | Listwise DPO + Rescore | 淘宝验证，RSP 模块是新技术 |
| P1 | IDEA-onerec-3 | ECPO + Format Reward | OneRec 生产验证，GRPO 的直接升级 |
| P2 | IDEA-gr4ad-3 | RSPO 排序优化 | 收益最大但前置依赖最重 |
| P2 | IDEA-uni-0 | SPO 搜索偏好优化 | 与 GRPO/DPO 重叠，无搜索场景 |
