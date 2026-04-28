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
│   └── EXP-017 SP-DPO ✅, EXP-018 RF-DPO pure DPO ❌ (forgetting), EXP-020 joint NTP+DPO ✅ SOTA 66.2%, EXP-037 SP-DPO features 进行中
├── IDEA-rpo-0: RPO — SFT Loss as Adversarial Regularizer (NeurIPS 2024)
│   └── L = L_DPO + ηβ·L_SFT, 理论证明 SFT 正则防 overoptimization
│   └── 我们的 joint NTP+DPO (EXP-019) 本质是 RPO
├── IDEA-spot-0: Elastic Tether — DPO reward 的隐式正则化 (HKU 2026)
│   └── β 控制 tether 松紧度: β 大 → tether 紧 → 防遗忘
├── IDEA-rankgr-0: Listwise DPO + Rescore (RankGR, 淘宝)
│   └── IAP (listwise DPO 解码) + RSP (轻量 rescore), 近万 QPS
├── ~~IDEA-uni-0~~: Search Preference Optimization (UniSearch, 快手) ❌ 关闭
│   └── reward model + user feedback, 搜索场景 (无搜索场景，技术被 align3-0 覆盖)
├── IDEA-gr4ad-3: RSPO (GR4AD 方案)
│   └── list-wise Lambda-weighted, NDCG-inspired
│   └── 最强但前置依赖最重
└── IDEA-sgrec-0: A2PO + Personalized Semantic Judge (S-GRec, 腾讯)
    └── 语义-业务不对称门控, GMV +1.19%
```

---

## IDEA-onemall-2: GRPO/DPO 强化学习对齐

**优先级**: P1
**来源**: OneMall §3.3 Reinforcement Learning Policy
**状态**: 框架已实现 — EXP-026 实现了 GRPO 基础设施 (`rl/grpo.py`, `rl/reward.py`, `rl/trainer.py`); EXP-037 正在运行 SP-DPO (features 链路) 验证 DPO 路线

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

**优先级**: ~~P1~~ → 已被 align3-0 覆盖
**来源**: OneLoc §2.5 Reinforcement Learning
**状态**: 已被 IDEA-align3-0 完全覆盖 — align3-0 的 SP-DPO + RF-DPO 链路即为本 IDEA 的超集。EXP-017/018/019/020 已完整验证 DPO 对齐框架；EXP-037 正在 features 链路上复现。oneloc-2 的"双目标奖励"概念可在 RF-DPO 阶段的 reward 设计中参考，不需要单独实验。

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
**状态**: 实验进行中 — 完整 NTP baseline 链路:
- EXP-017 SP-DPO ✅ R@10 15.4%
- EXP-018 RF-DPO pure DPO ❌ forgetting (β ablation 解释见 IDEA-spot-0)
- EXP-019/020 joint NTP+DPO ✅ SOTA: R@500=66.2%, PPL=16.3 (exp020-hard-lam03)
- Features 链路 (EXP-036→037→038→039): EXP-036 SFT 起点 R@500=59.0%，EXP-037 SP-DPO 进行中，EXP-038 RF-DPO 待跑，EXP-039 ECPO 待跑

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

**优先级**: ~~P2~~ → ❌ 关闭
**来源**: UniSearch (Kuaishou, arxiv 2509.06887, Sep 2025)
**状态**: ❌ 关闭 — 当前无搜索场景，核心 RL 技术已被 IDEA-align3-0 (Progressive DPO) 和 IDEA-onemall-2 (GRPO) 完全覆盖

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

## IDEA-gpr-0: HEPO — Hierarchy Enhanced Policy Optimization

**优先级**: P2
**来源**: GPR (Tencent/Weixin Channels, arxiv 2511.10138, Nov 2025)
**状态**: 待讨论

### 核心思想

GPR 是腾讯微信视频号广告的 one-model 生成式推荐框架，其 RL 组件 **HEPO (Hierarchy Enhanced Policy Optimization)** 利用 SID 的层级结构做对齐:

- 结合 MTP (Multi-Token Prediction), Value-Aware Fine-Tuning, 和 HEPO 三阶段联合训练
- HEPO 利用 SID 的 coarse-to-fine 层级信息设计 reward (不同层级的预测正确性给不同 reward)
- 统一 interest modeling + value alignment + policy optimization

微信视频号广告全量部署: **GMV 和 CTCVR 显著提升** (具体数字在论文正文)。

### 与当前项目的关联

- HEPO 利用 SID 层级结构做 hierarchical reward 是新思路 — 区别于 GRPO/DPO 的 flat reward
- 例: 预测对了 L1 但错了 L2/L3 → 部分 reward (比完全预测错好)
- 与 IDEA-onerec-3 (ECPO + Format Reward) 互补: ECPO 关注 clipping，HEPO 关注 hierarchical reward structure

### 关键问题

1. 依赖 RL 基础设施成熟后实施
2. 论文全文细节需要补充 HEPO 的具体算法
3. 适合在 GRPO/DPO 基础方案验证后作为进阶改进

---

## IDEA-sgrec-0: A2PO + Personalized Semantic Judge (Asymmetric Advantage)

**优先级**: P1
**来源**: S-GRec (Tencent, arxiv 2025)
**状态**: 待讨论

### 核心思想

S-GRec 发现标准 GRPO/DPO 在生成式推荐中存在 **语义-业务目标不对称** 问题：语义上相似的候选在业务价值上可能差异巨大（如两个相似视频，一个高 GMV 一个低 GMV），但标准 RL 给它们相似的 advantage。S-GRec 提出两个机制：

1. **A2PO (Asymmetric Advantage Policy Optimization)**: 对 positive 和 negative 候选使用不同的 advantage 计算方式。Positive 候选用标准 normalized advantage，negative 候选额外乘以一个 **semantic gating factor** — 语义上离正样本越近但业务价值越差的负样本，penalty 越重
2. **Personalized Semantic Judge**: 训练一个轻量判别器，输入用户历史 + 候选 SID，输出语义匹配度和业务价值的联合评分。作为 RL 的 reward model，替代纯业务指标 reward

核心 insight: **在语义空间中距离近但业务价值差的 hard negative 是最有信息量的训练信号**。

腾讯在线 A/B: **GMV +1.19%, 用户留存 +0.8%**。

### 与当前项目的关联

- 与 IDEA-onemall-2 (GRPO) 直接兼容：A2PO 是 GRPO 的 advantage 计算改进，不改变整体框架
- 与 IDEA-align3-0 (Progressive DPO) 互补：align3-0 解决训练稳定性 (SP→RF progressive)，sgrec-0 解决 advantage 质量
- Personalized Semantic Judge 可以复用 SID embedding 空间中的距离信息，实现成本适中
- **语义-业务不对称门控**是新思路：利用 SID 的层级结构判断语义距离（L1 相同 = coarse 相似，L1/L2/L3 全同 = 高度相似）

### 实验设计草案

**Phase 1 — A2PO (在 GRPO 基础上改进)**:
1. 对 GRPO 的 advantage 计算，加入 semantic gating:
   - `gated_adv = adv * semantic_gate(candidate, positive)`
   - `semantic_gate = sigmoid(α * (1 - cosine_sim(sid_embed_candidate, sid_embed_positive)))`
2. 只对 negative advantage 应用 gating，positive 保持不变（asymmetric）

**Phase 2 — Personalized Semantic Judge**:
1. 轻量 MLP: `[user_repr, candidate_sid_embed] → score`
2. 训练数据: 用户行为中的 positive (clicked) 和 negative (exposed-not-clicked) pair
3. 用 judge score 替代简单 reward 信号

### 关键问题

1. 前置依赖 GRPO 基础设施 (IDEA-onemall-2)
2. Semantic gating 的 α 超参敏感性：过大会完全忽略远样本，过小退化为标准 GRPO
3. Personalized Semantic Judge 的训练数据需要曝光未点击样本

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P1 (框架已实现) | IDEA-onemall-2 | GRPO/DPO 强化学习 | EXP-026 GRPO 框架已实现，EXP-037 DPO 实验中 |
| P1 | IDEA-oneloc-2 | DPO + 双目标奖励 | DPO 比 PPO 简单，可作为 RL 入门 |
| **P1 (进行中)** | **IDEA-align3-0** | **Progressive DPO (SP→RF)** | **EXP-020 ✅ SOTA 66.2%; EXP-037 SP-DPO features 链路进行中** |
| P1 (待 EXP-037 完成) | IDEA-onerec-3 | ECPO + Format Reward | EXP-039 planned，需先完成 EXP-037/038 |
| P1 | IDEA-rankgr-0 | Listwise DPO + Rescore | 淘宝验证，RSP 模块是新技术 |
| P1 | IDEA-sgrec-0 | A2PO + Semantic Judge | 腾讯 +1.19% GMV, 语义-业务不对称门控 |
| P1 | IDEA-genrec-2 | GRPO-SR + Hybrid Rewards | JD 验证, NLL 正则 + relevance gating 防 reward hacking |
| **P1 (已验证)** | **IDEA-rpo-0** | **RPO: SFT Loss as DPO Regularizer** | **NeurIPS 2024 理论证明, EXP-019/020 joint NTP+DPO = RPO ✅** |
| **P1 (已验证)** | **IDEA-spot-0** | **Elastic Tether: β 自适应正则化** | **HKU 2026, 解释 EXP-018 β ablation 结果 ✅** |
| P2 | IDEA-gr4ad-3 | RSPO 排序优化 | 收益最大但前置依赖最重 |
| ❌ 关闭 | ~~IDEA-uni-0~~ | ~~SPO 搜索偏好优化~~ | 无搜索场景，技术已被 align3-0/onemall-2 覆盖 |
| P2 | IDEA-gpr-0 | HEPO Hierarchical Policy Opt | 腾讯微信广告部署, 层级 reward 新思路 |

---

## IDEA-genrec-2: GRPO-SR + Hybrid Rewards (防 Reward Hacking 的 RL 对齐)

**优先级**: P1
**来源**: GenRec, JD.com (arxiv 2604.14878, SIGIR 2026)
**状态**: 待讨论

### 核心思想

JD GenRec 在 GRPO 基础上提出两个关键改进防止 reward hacking:

1. **Hybrid Reward + Relevance Gate**: 用 dense reward model (SIM-based) 估计偏好分数 r_pref，但加一个 relevance gate G = I(sim > τ) 过滤语义不相关的高 reward 候选。不满足 gate 的候选 reward 置零。同时对已知正样本 (用户实际点击/购买) 强制赋予组内最高 reward，校正 reward model 的估计偏差。

2. **NLL Supervised Regularization (SR)**: 在 GRPO 目标中加入 NLL 正则项 (对正样本的负对数似然)，将 policy 锚定到真实用户行为分布，替代标准 KL 散度惩罚。

消融实验: 去掉 Gate G 后，HR@50 从 0.74 降到 0.70, HaR 从 2.68% 涨到 1.96% — 看似幻觉降了但 HR 大降 → reward hacking 现象明确。在线: SFT + GRPO-SR 比纯 SFT 再提升 +1% click, +1.4% transaction。

### 与当前项目的关联

- 直接增强 IDEA-onemall-2 (GRPO/DPO): GenRec 的 GRPO-SR 是 GRPO 的工业验证改进版
- Relevance Gate 对 SID 体系特别重要: SID 空间中可能存在 "有效但语义无关" 的组合
- NLL 正则替代 KL 散度 → 实现更简单, 不需要 reference model 的完整推理
- Reward calibration (正样本强制赋 max reward) 是实用技巧

### 实验设计草案

**Phase 1 — GRPO-SR on NTP baseline**:
- 在 NTP 基线上接 GRPO-SR: rollout 生成 G 个候选, 用 reward model 打分
- Reward model 初期可用 SID embedding cosine similarity 替代 SIM
- 加入 relevance gate: cosine_sim(generated_sid, positive_sid) > τ
- NLL 正则: α * (-log P(positive_item | history))
- 评估: Recall@K, reward distribution, HaR

**Phase 2 — Dense Reward Model**:
- 训练专门的 reward model (SIM-based 或 user preference model)
- 加入 positive calibration: 正样本的 reward = max(group rewards)

### 关键问题

1. 前置依赖: NTP SFT baseline + GRPO 基础设施 (IDEA-onemall-2)
2. Reward model 的选择: SIM-based (需要训练) vs 简单 embedding similarity (可快速启动)
3. Gate 阈值 τ 的调参: 过高则过度过滤, 过低则不起作用
4. JD 的 improvement (+1% over SFT) 在 GRPO 基础上是 marginal → 优先做好 SFT

---

## IDEA-rpo-0: RPO — SFT Loss as Adversarial Regularizer for DPO

**优先级**: P1
**来源**: RPO (ByteDance + Northwestern + Stanford, arxiv 2405.16436, NeurIPS 2024)
**状态**: **已验证** — EXP-019 joint NTP+DPO 本质是 RPO

### 核心思想

RPO 理论证明：在 DPO 训练中加入 SFT loss 作为正则项可以 **provably mitigate reward overoptimization**:

```
L_RPO = L_DPO + ηβ · L_SFT(chosen)
```

关键理论发现：
1. DPO 内置的 β KL 约束 **只控制梯度 scale，不控制方向** — 不足以防止 overoptimization
2. SFT loss 额外修正梯度方向，将 policy 锚定到高质量 response 分布
3. 从 adversarial reward model 角度推导：SFT loss 等价于对最差情况 reward model 的惩罚

实验：RPO 在 Zephyr-7b-beta 和 Zephyr-7b-gemma 上一致优于 DPO，有效防止 chosen response 概率在训练中下降（overoptimization 的典型症状）。

### 与当前项目的关联

**直接理论支持我们的 joint NTP+DPO 设计**:
- 我们的 `total_loss = ntp_loss + λ * dpo_loss` 就是 RPO 的 `L_RPO = L_SFT + (1/ηβ) * L_DPO`
- 我们的 λ 对应 RPO 的 1/(ηβ)
- RPO 论文用 η=1，即 SFT 和 DPO 权重量级相当 → 支持我们 λ=0.1~0.5 的搜索范围

**解释 EXP-018 结果**:
- Pure DPO (无 SFT 正则) → catastrophic forgetting → PPL 爆炸
- β=0.5 退化最轻但仍不够 → RPO 理论: β 只控制 scale 不够, 需要 SFT loss 控制 direction
- EXP-019 加入 NTP (=SFT) 正则 → 预期修复 forgetting

### 实验设计

**已在 EXP-019 中实现**:
- Config 2 (λ=0.1), Config 3 (λ=0.5), Config 4 (λ=0.01) → 验证 RPO 的 ηβ 权重选择
- 预期: λ 过小 → 回到 EXP-018 的 forgetting; λ 过大 → NTP 主导冲掉 DPO signal
- Sweet spot 预计在 λ=0.1~0.5

### 关键问题

1. RPO 用 chosen response 做 SFT, 我们用 NTP 训练数据 → 分布可能不同, 但正则化效果类似
2. Multi-epoch NTP 问题: RPO 理论没有讨论 SFT 数据重复 → 我们的 ~4.5 epoch 可能导致 NTP overfitting
3. 如果 EXP-019 验证了 joint NTP+DPO 有效, 下一步可以试 RPO 原始形式: 只在 DPO chosen sample 上做 SFT (而非整个 NTP dataset)

---

## IDEA-spot-0: Elastic Tether — DPO Reward 公式的隐式正则化

**优先级**: P1
**来源**: SPoT (HKU, arxiv 2603.01683, Mar 2026)
**状态**: **解释了 EXP-018 β ablation 结果**

### 核心思想

SPoT 揭示了 DPO reward 公式 `r(x,y) = β log(π/π_ref)` 自带的 **Elastic Tether** 正则化效应:

梯度缩放系数 λ = 1 - σ(r_θ(x, y+)):
- **Acquisition Mode** (π 接近 π_ref): r ≈ 0 → λ ≈ 0.5 → 正常学习
- **Saturation Mode** (π 远离 π_ref): r → ∞ → λ → 0 → 梯度消失，自动停止更新

当 r_θ = 10 时, λ = 4.5×10⁻⁵ — 比 SFT 的常数 gradient 小 22,000 倍。

对照实验:
- SFT+ (proximal data) → forgetting (IFEval 持续下降)
- DPO (同样数据) → 不 forgetting (IFEval 稳定)
- **Reward-SFT (只有 chosen, 无 rejected, 但用 reward 公式)** → 也不 forgetting!

结论: **正则化来自 reward 公式本身（log π/π_ref 的 tethering effect），而非负样本**。β 越大 tether 越紧。

### 与当前项目的关联

**直接解释 EXP-018 的 β ablation**:

| Config | β | PPL | R@10 | Elastic Tether 解释 |
|--------|-----|-----|------|-------------------|
| hard | 0.1 | 50,694 | 8.3% | β 小 → tether 松 → policy 漂移远 → catastrophic forgetting |
| prog-beta01 | 0.01 | 2.4B | 6.0% | β 极小 → 几乎无 tether → 完全 forgetting |
| prog-beta50 | 0.5 | 404.9 | 10.2% | β 较大 → tether 紧 → 退化最轻但仍不够 |

807 步的 hard DPO 训练中, 即使 β=0.5 (最紧 tether), r_θ 也会逐渐增大导致 tether 完全松弛 (λ → 0)。此时模型处于 "zero gradient" 状态但已偏离太远 — tether 只能减速不能回拉。**需要外部正则化 (NTP/SFT loss) 提供持续的 gradient 信号将 policy 拉回**。

### 实验设计

**EXP-019 已覆盖核心验证**。

进一步实验方向:
1. **Reward-SFT baseline**: 只用 chosen preference pair 做 reward-based SFT (无 negative), 对比 DPO → 验证 tethering 是否在推荐场景也成立
2. **β schedule**: 训练初期用小 β (快速学习), 后期逐渐增大 β (收紧 tether 防 forgetting)
3. **Monitor r_θ during training**: 记录 implicit reward 的增长轨迹, 验证 tether 何时松弛

### 关键问题

1. SPoT 的实验是在 LLM (Qwen3-8B) 上做的, 推荐模型 (45.8M) 行为可能不同
2. Elastic Tether 是对单步梯度的分析, 807 步的累积效应可能不同
3. 与 RPO (IDEA-rpo-0) 互补: RPO 加 SFT loss 修正方向, Elastic Tether 解释 β 的 scale 控制作用
