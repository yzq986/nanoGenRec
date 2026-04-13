# OneLoc: 地理感知生成式推荐

**来源**: OneLoc: Geo-Aware Generative Recommender Systems for Local Life Service (Kuaishou, arxiv 2508.14646v1)
**日期**: 2026-04-13

---

## IDEA-005: Geo-aware Self-attention (位置上下文注入注意力)

**优先级**: P2
**来源**: OneLoc §2.3.3 Geo-aware Self-attention
**状态**: 待讨论

### 核心思想

在 transformer self-attention 中加入一个 additive 的位置上下文相似度项，并用 user 实时位置做 gate 控制输出。具体: `A = Softmax(QK^T/√d + E_lc · E_lc^T)`，然后用 `g = 2·Sigmoid(MLP(concat(e_u, e_i)))` 作为 (0, 2) 的缩放因子，放大或衰减与用户位置相关/不相关的注意力输出。

### 与当前项目的关联

- 当前 `metrics/sid_prediction.py:CausalTransformerLayer` 只有标准 causal self-attention
- 如果项目未来引入 side information (如类目、品牌、地理)，这种 additive attention + gate 是一种低成本的方式
- 但 **当前项目无地理信息需求** (通用推荐，非 LBS 场景)，直接照搬意义不大
- 更通用的启发: **任何 side information 都可以用 additive similarity + gating 注入 attention**，不只是地理

### 实验设计草案

**适用场景**: 如果未来有多模态/多信号融合需求
- 将 item 的某种 context embedding (类目 embed、品牌 embed) 作为 E_lc
- 在 self-attention score 中加入 context similarity 项
- 用 user profile embedding 作为 gate query

**评估**: 对比 vanilla attention vs context-augmented attention 的 NTP recall

### 关键问题

1. 当前项目是纯内容推荐 (text embedding → semantic ID)，没有用户行为序列建模，此技术暂无落地场景
2. 需要先有 encoder-decoder 架构 (ARCHITECTURE.md 中 TODO) 才有实际意义
3. 如果只是提升 NTP 模型，更应该优先做 ARCHITECTURE.md 中的 "Context Processor" (OneRec V2 lazy decoder-only)

---

## IDEA-006: Neighbor-aware Prompt (邻域交叉注意力提示)

**优先级**: P2
**来源**: OneLoc §2.4.1 Neighbor-aware Prompt
**状态**: 待讨论

### 核心思想

在 decoder 输入中引入 "邻域提示": 以用户位置为 query，对周围 8 个 GeoHash block 的 context embedding 做 cross-attention，聚合局部信息 (周围品牌、热销品等) 作为生成的引导信号。

### 与当前项目的关联

- 当前 decoder (`AutoregressiveNTPModel`) 没有任何 prompt/prefix 机制
- 这个技术的**泛化形式**是: 在生成 semantic ID 之前，先通过 cross-attention 聚合某种 "上下文提示"
- 对我们有启发的不是地理邻域，而是 **用户兴趣邻域** 或 **类目邻域**: 比如用 user embedding 去 attend 到 top-k 相似类目的 prototype embedding
- 但需要先有 encoder-decoder 架构

### 实验设计草案

**泛化版本: Category-aware Prompt**
- 维护 category centroids (类目级别的 embedding 均值)
- User 的近期行为 embedding 均值作为 query
- Cross-attention 到 top-k 相关类目 centroids → 得到 prompt token
- 将 prompt token 作为 decoder 的第一个输入

**评估**: 对比有/无 category prompt 的 NTP beam search recall

### 关键问题

1. 同 IDEA-005: 当前无 encoder-decoder 架构，无法直接落地
2. 需要先完成 "Context Processor" 或 encoder-decoder 重构
3. 类目信息的获取: 当前 item metadata 是否包含类目? 需要检查数据 pipeline

---

## IDEA-007: DPO 对齐 + 双目标奖励函数

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

## IDEA-008: Geo-aware Semantic ID (地理信号注入残差量化)

**优先级**: P1
**来源**: OneLoc §2.2 Geo-aware Semantic IDs
**状态**: 待讨论

### 核心思想

OneLoc 在残差量化的初始 embedding 中融合地理上下文: `r_0 = concat(e_video, e_location_context)`，使得生成的 semantic ID 本身就编码了地理语义。地理上下文由多模态大模型从 GeoHash 的品牌、类目、热销品信息中提取。

### 与当前项目的关联

- 当前 `model/rkmeans.py` 的输入是纯 Qwen3 文本 embedding
- 这与 **IDEA-002 (协同信号增强 Embedding)** 思路一致: 在量化之前将额外信号注入 embedding
- 泛化形式: **任何 side information 都可以在量化前 concat/fuse 到 embedding 中**
- 具体对我们的启发: 除了协同信号 (IDEA-002)，还可以注入:
  - 类目层级 embedding
  - 价格区间 embedding
  - 热度/新鲜度 signal
- 本质上是 **量化输入空间的丰富化**

### 实验设计草案

**方案 A: concat + MLP fusion**
- 输入: `concat(qwen3_embed_1024d, side_info_embed_128d)` → MLP → 1024d
- 对 fused embedding 做 RKMeans (或 OPQ)

**方案 B: 加权残差**
- 在 RKMeans 第一层输入中加权融合: `r_0 = α·e_content + (1-α)·e_side`

**评估**: 对比 pure content embed vs fused embed 在量化指标和 NTP recall 上的表现

### 关键问题

1. **与 IDEA-002 重叠**: 协同信号增强也是改 embedding 输入。应统一为 "embedding enrichment" 框架，避免重复实验
2. 当前有什么 side information 可以用? 需要检查 item metadata 中除文本外还有什么字段
3. Concat 后维度增加对量化质量的影响 — 高维可能让 KMeans 更难收敛
4. 如果走 IDEA-001 (OPQ) 路线，side info 可以分配到独立子向量，天然适合并行量化

---

## IDEA-009: Scaling Law — 序列长度 >> 模型大小

**优先级**: P0
**来源**: OneLoc §4.4 Hyperparameter Experiments
**状态**: 待讨论

### 核心思想

OneLoc 的 scaling 实验揭示了一个关键发现: **序列长度的收益远大于模型大小的收益**。模型从 0.05B 扩到 0.3B，recall/NDCG 平均提升 7%；但序列长度从 100 扩到 300，recall 提升 13%、NDCG 提升 51%。这意味着在资源有限时，应优先增加序列长度而非模型参数。

### 与当前项目的关联

- 当前 `AutoregressiveNTPModel` S-tier config: 6 layers, 256 embed_dim, ~39.5M params
- ARCHITECTURE.md 定义了 M/L tier 但未实现
- **关键问题**: 我们尚未做过 NTP 模型的 scaling 实验
- 更直接的启发: 在 NTP 训练中，user 行为序列的长度是否比模型大小更重要?
- 当前行为序列处理: `data/export_behavior.py` 中导出的序列长度是多少? 是否足够长?

### 实验设计草案

**实验矩阵**:

| 维度 | 小 | 中 | 大 |
|------|-----|-----|-----|
| 模型参数 | S-tier (39.5M) | M-tier (~150M) | L-tier (~500M) |
| 序列长度 | 50 | 100 | 200 |

**设计**:
- 固定量化方案 (RKMeans 3x1024 或 OPQ)
- 3x3 grid: 模型大小 x 序列长度
- 每组训练 NTP 模型到收敛
- 记录 recall@5/10/20, NDCG@5/10/20

**评估**:
- 绘制 scaling curve: recall vs 模型参数 (固定序列长度)
- 绘制 scaling curve: recall vs 序列长度 (固定模型参数)
- 验证 OneLoc 的结论是否在我们的场景复现

### 关键问题

1. **前置依赖**: 需要先有稳定的 NTP 训练 pipeline + 稳定的量化方案
2. 当前 NTP 训练是否已经可以 end-to-end run? 需要确认 `model/train.py` → NTP 的完整流程
3. 行为数据量: 序列长度 200 需要足够的用户行为数据
4. 计算成本: 9 组实验，每组可能需要数小时训练
5. **为什么 P0**: 这个实验的结论直接决定资源分配策略 — 是花钱买更大 GPU 还是花钱采集更多行为数据

---

## IDEA-010: Multi-behavior Sequence 融合

**优先级**: P1
**来源**: OneLoc §2.3.1 Multi-behavior Sequence
**状态**: 待讨论

### 核心思想

OneLoc 区分三种行为序列: watch (浏览), click (点击), pay (购买)，每种行为的序列长度不同 (256/32/10)。三种序列 concat 后统一输入 encoder。不同行为代表不同强度的兴趣信号。

### 与当前项目的关联

- 当前 `data/export_behavior.py` 导出行为数据，但处理方式需要确认
- 当前 NTP 模型 (`metrics/sid_prediction.py`) 输入是单一序列
- 如果我们有多种行为信号 (展现/点击/购买/收藏)，分离不同行为的序列可能比混合在一起更有效
- **与 IDEA-002 (协同信号增强) 有交集**: 行为序列本身就是协同信号的来源

### 实验设计草案

**前提**: 需要行为数据包含行为类型标注

**方案**:
- 按行为强度分离序列: `S_expose` (长), `S_click` (中), `S_purchase` (短)
- 每种序列独立 embedding → concat → 输入 encoder
- 或: 用 behavior type embedding 标注每个 item，统一序列但加入类型信号

**评估**: 单一混合序列 vs 分行为序列 的 NTP recall

### 关键问题

1. 行为数据是否包含行为类型? 需要检查 `data/export_behavior.py` 的 schema
2. 不同行为的序列长度比例如何确定 (OneLoc 用 256/32/10)
3. 实现复杂度: 需要修改数据 pipeline + 模型输入处理

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P0 | IDEA-009 | Scaling Law: 序列长度 vs 模型大小 | 直接决定资源分配策略；OneLoc 显示序列长度收益 7x > 模型大小收益；验证后可指导所有后续实验的 compute budget |
| P1 | IDEA-007 | DPO 对齐 + 双目标奖励 | RL alignment 是 OneRec 系列的核心范式，当前项目零 RL 代码；但需要 NTP 模型先稳定 |
| P1 | IDEA-008 | 多信号融合量化输入 | 与 IDEA-002 统一为 "embedding enrichment"，扩展量化输入的信息密度 |
| P1 | IDEA-010 | Multi-behavior 序列融合 | 低成本区分不同行为强度；但需要行为数据包含类型标注 |
| P2 | IDEA-005 | Context-augmented Attention | 需要先有 encoder-decoder 架构；当前场景无地理需求 |
| P2 | IDEA-006 | Category-aware Prompt | 需要先有 encoder-decoder 架构；泛化形式有价值但前置依赖多 |
