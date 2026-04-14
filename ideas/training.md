# Training (训练目标与策略)

NTP 模型的训练信号设计：辅助 loss、样本加权、多行为融合等。在不改变模型架构的情况下提升训练质量。

**影响范围**: `metrics/sid_prediction.py`, `model/train.py`, `data/export_behavior.py`

---

## 演进路径

```
纯 CE loss + MoE aux loss (当前 baseline)
├── IDEA-onemall-0: In-Batch Contrastive Loss (连续语义监督)
├── IDEA-sid-4: Token-Space MTP 辅助 Loss (细粒度 token CE)
│   └── 与 onemall-0 互补: token-level vs embedding-level
├── IDEA-sid-5: Codebook Embedding 聚合 (item 表示)
│   └── 依赖 IDEA-sid-0 Phase 2 (OPQ 长 ID)
├── IDEA-gr4ad-2: Value-Aware 训练 (eCPM token + 样本加权)
│   └── 引入业务价值信号
├── IDEA-oneloc-5: Multi-behavior 序列融合
│   └── 区分 click/buy/expose 不同行为强度
└── IDEA-plum-0: LLM Continued Pre-Training (Google/YouTube)
    └── 预训练 LLM → CPT on SID 语料 → Fine-tune, 数十亿用户验证
```

---

## IDEA-onemall-0: In-Batch Contrastive Auxiliary Loss for NTP Model

**优先级**: P0
**来源**: OneMall §3.2 Supervised Objectives
**状态**: 待讨论

### 核心思想

在 NTP 自回归训练的同时，加一个 two-tower 风格的 in-batch contrastive loss 作为辅助目标。具体做法: 最后一个 SID token 的隐层表示 s₃^L（已编码完整 SID 序列信息）与目标 item embedding f_item 做 InfoNCE 对比学习。OneMall 报告该任务达到 **98% accuracy@1**，说明 s₃^L 已经高质量编码了 item 信息。

辅助对比 loss 的作用:
- 为 Transformer 提供 embedding 空间的连续监督信号（NTP 只有离散 token CE loss）
- 防止 SID 表示退化为只关心 token 分类而丢失语义连续性
- 正则化效果，改善泛化

### 与当前项目的关联

- NTP 模型在 `metrics/sid_prediction.py:AutoregressiveNTPModel`，当前仅有 `CE_loss + 0.01 * aux_loss(MoE balance)`
- item embedding 已有现成的 Qwen3 embedding（`model/encode.py`），训练时可直接加载
- 实现成本极低: 在 s₃ 位置加一个 MLP projection head → InfoNCE with in-batch negatives
- **与 IDEA-sid-1 (协同信号 embedding) 正交**: IDEA-sid-1 改善 embedding 本身，本 IDEA 改善 NTP 模型训练

### 实验设计草案

**修改 `metrics/sid_prediction.py`**:
1. 新增 `ContrastiveHead`: MLP(embed_dim → 128) 投影到对比空间
2. 取 s₃ 位置的隐层输出 → ContrastiveHead → l2_normalize
3. 目标 item embedding → MLP(1024 → 128) → l2_normalize
4. InfoNCE loss (temperature=0.05, in-batch negatives)

**训练 loss**:
```
L = L_NTP + 0.01 * L_moe_balance + α * L_contrastive
```

**变量**:
- α ∈ {0.01, 0.1, 0.5, 1.0}
- projection dim ∈ {64, 128, 256}
- temperature ∈ {0.05, 0.07, 0.1}

**基线**: 当前 NTP-only 训练 (EXP-001 final config: 3 layers x 1024 clusters)

**评估指标**: beam search Recall@{10,50,100,500}, SID accuracy@{1,2,3}, 训练收敛速度

### 关键问题

1. batch size 需要足够大以提供足量 in-batch negatives — 当前 batch size 是多少？可能需要增大
2. s₃ 隐层是否需要 stop-gradient (asymmetric design) 还是两边都 backprop
3. 训练早期 contrastive loss 可能主导梯度，需要 warmup 策略（先纯 NTP 若干 epoch 再加 contrastive）

---

## IDEA-sid-4: Token-Space MTP 辅助 Loss (适用于自回归模型)

**优先级**: P1
**来源**: RPG (KDD'25, arxiv 2506.05781) §2.2.1 Multi-Token Prediction
**状态**: 待讨论

### 核心思想

RPG 的 MTP loss 将 item 预测分解为各 token 独立的 CE loss 之和: ℒ = -Σⱼ log P(c_j | s)。这比传统 item-level CE 有两个关键优势:
1. **细粒度语义学习**: 在 token 空间（M 个类）而非 item 空间（N >> M 个类）优化，模型学到的是 sub-item 级别的语义特征
2. **冷启动友好**: 低频 item 与高频 item 共享 token，通过 token 共现获得充分训练信号。RPG 在所有频次桶 ([0,5] 到 [16,20]) 均显著优于 TIGER

**关键洞察**: 这个 loss 不要求并行预测 — 可以作为辅助目标加到任何 SID 模型上。即使在自回归模型中，最后一个 token 的隐层表示 h_L 编码了完整序列信息，可以对 h_L 施加 MTP loss 来强化语义理解。

### 与当前项目的关联

- 当前 NTP 模型 (`metrics/sid_prediction.py:AutoregressiveNTPModel`) 只有逐 token CE loss + MoE aux loss
- **与 IDEA-onemall-0 (In-Batch Contrastive Loss) 互补**: onemall-0 用 item embedding 做对比，本 IDEA 用 token-level CE 做细粒度监督
- 即使最终走自回归路线 (不用 RPG 的并行预测)，MTP 辅助 loss 也是有价值的正则化
- 如果走 IDEA-sid-0 (OPQ 并行 ID)，MTP 就是 primary loss

### 实验设计草案

**方案 A — 作为自回归模型的辅助 loss**:
1. 取最后一个 SID token 位置的隐层表示 h_3^L
2. 对 h_3^L 加 m 个独立 MLP projection heads (m = SID token 数)
3. 每个 head 输出 M 维 logits → CE loss
4. 总 loss: `L_NTP + α * L_MTP + 0.01 * L_moe`

**方案 B — 直接作为 parallel prediction primary loss** (= IDEA-sid-0 Phase 2):
1. 用户序列 → Transformer encoder → s
2. s → m 个 MLP heads → m 个 softmax → MTP loss
3. 推理: graph-constrained decoding

**变量** (方案 A):
- α ∈ {0.1, 0.5, 1.0}
- 是否与 IDEA-onemall-0 (contrastive loss) 叠加

**评估**: SID accuracy, beam search Recall@K, 冷启动 item 子集的 Recall

### 关键问题

1. 方案 A 需要最后位置的隐层同时编码 "下一个 item 的所有 token" 信息 — 是否与自回归训练的 teacher forcing 冲突？(teacher forcing 时 h_3 已经看到了 target 的前 3 个 token)
2. 如果用 BOS 位置的隐层 h_0（只编码用户序列，没看到 target token），是否更合理？
3. 与 IDEA-onemall-0 的关系: 两者都在同一个隐层位置施加额外 loss，可能有梯度冲突

---

## IDEA-sid-5: SID Codebook Embedding 聚合作为 Item 表示

**优先级**: P2
**来源**: RPG (KDD'25) §2.1.2 Semantic ID Embedding Aggregation
**状态**: 待讨论

### 核心思想

RPG 用 SID 的 codebook embedding 的 mean/max pooling 作为 item 表示，替代原始高维 embedding。每个 codebook j 有一个可学习 embedding table E_j ∈ ℝ^(M×d)。item 的 SID = (c_1, ..., c_m)，其表示为:

`v_item = Pool(E_1[c_1], E_2[c_2], ..., E_m[c_m])`

这样 item 表示的维度 = d（与 token embedding 维度相同），与 item 总数 N 无关。所有 item 共享 m 个大小为 M 的 codebook，总 embedding 参数 = m × M × d（远小于 N × d 的全 embedding table）。

### 与当前项目的关联

- 当前 NTP 模型的 item embedding 是 SID token 的 lookup + positional encoding，已经隐式用了类似的 codebook embedding
- RPG 的聚合方式更显式: mean pooling 所有 codebook embedding → 单向量表示
- 可用于: (1) item retrieval (2) item 冷启动 (3) 作为 ranking model 的 item feature
- 但当前 NTP 模型只有 3 个 token (RKMeans)，聚合收益不大。如果切换到 OPQ (16~64 token)，聚合方式变得重要

### 实验设计草案

**前置: IDEA-sid-0 Phase 2 (OPQ + 并行预测模型)**

**验证**:
1. 训练好并行预测模型后，提取 codebook embeddings
2. 对每个 item 做 mean/max pooling → item vector
3. 用 item vector 做 ANN retrieval → 对比 graph decoding 的 recall
4. 分析: pooled embedding 是否保留了足够的语义区分度？

**评估**: cosine similarity 分布, retrieval recall@K, t-SNE 可视化

### 关键问题

1. mean pooling 是否会丢失 token 间的交互信息？RPG 论文没有对 mean vs max 做消融
2. 只有在 OPQ (长 ID) 场景才有意义 — 3 个 token 的 mean pooling 太粗糙
3. 与 FAISS 检索的关系: 如果 pooled embedding 质量足够好，可以用传统 ANN 代替 graph decoding

---

## IDEA-gr4ad-2: Value-Aware 训练目标 (VSL + eCPM Token)

**优先级**: P1
**来源**: GR4AD §VSL
**状态**: 待讨论

### 核心思想

GR4AD 在 NTP 训练中引入两个价值感知机制: (1) eCPM Token Prediction — 在语义 ID 序列末尾追加一个离散化的 eCPM token，让模型同时预测"推什么"和"值多少钱"；(2) Value-Aware Sample Weighting — 按用户长期价值和行为深度（购买 > 点击）加权训练样本。

### 与当前项目的关联

- `metrics/sid_prediction.py` 当前训练目标是纯 CE loss，所有样本等权
- 我们的数据中有行为类型（点击、购买、收藏等），在 `data/export_behavior.py` 中已定义
- eCPM token 的思想可以泛化为 **任意业务价值 token** — 比如 item 热度桶、CTR 桶等
- **与 IDEA-sid-1 (协同信号增强) 互补**: IDEA-sid-1 改进 embedding 表示，本 IDEA 改进训练信号

### 实验设计草案

**变量 1 — 价值 token 追加**:
- 将 item 的某个连续指标（如行为频次、热度）离散化为 N 个桶
- 语义 ID 从 `"L1_L2_L3"` 扩展为 `"L1_L2_L3_V"`，V ∈ {0, ..., N-1}
- NTP 模型在预测 L3 后继续预测 V token
- 推理时: V token 的 logits 可作为辅助排序信号（类似 GR4AD 用 eCPM 做 reranking）

**变量 2 — 样本加权**:
- 购买样本 weight=3.0, 收藏 weight=2.0, 点击 weight=1.0（需根据数据分布调参）
- 在 `sid_prediction.py` 训练循环中加 sample weight

**评估**: Hit@K (基础), weighted Hit@K (高价值 item 权重更高), 价值 token 预测准确率

### 关键问题

1. 我们的 demo 数据中业务价值信号是否充分？如果只有点击数据，sample weighting 退化为等权
2. 价值 token 增加序列长度 → 推理成本增加，但只增加 1 个 token，可接受
3. 离散化桶数 N 的选择: 太少信息量不够，太多导致长尾稀疏

---

## IDEA-oneloc-5: Multi-behavior Sequence 融合

**优先级**: P1
**来源**: OneLoc §2.3.1 Multi-behavior Sequence
**状态**: 待讨论

### 核心思想

OneLoc 区分三种行为序列: watch (浏览), click (点击), pay (购买)，每种行为的序列长度不同 (256/32/10)。三种序列 concat 后统一输入 encoder。不同行为代表不同强度的兴趣信号。

### 与当前项目的关联

- 当前 `data/export_behavior.py` 导出行为数据，但处理方式需要确认
- 当前 NTP 模型 (`metrics/sid_prediction.py`) 输入是单一序列
- 如果我们有多种行为信号 (展现/点击/购买/收藏)，分离不同行为的序列可能比混合在一起更有效
- **与 IDEA-sid-1 (协同信号增强) 有交集**: 行为序列本身就是协同信号的来源

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

## IDEA-plum-0: LLM Continued Pre-Training for Generative Recommendation

**优先级**: P1
**来源**: PLUM (Google/YouTube, arxiv 2510.07784, Oct 2025)
**状态**: 待讨论

### 核心思想

PLUM 是 YouTube 大规模部署的 LLM-based 生成式推荐框架，核心是三阶段训练:

1. **Item Tokenization via Semantic IDs**: 视频 → SID 映射
2. **Continued Pre-Training (CPT)**: 在推荐域数据上继续预训练 LLM，让模型学会 SID 词表和用户行为模式
3. **Task-Specific Fine-Tuning**: 直接训练模型根据用户上下文生成推荐 item 的 SID

关键发现:
- CPT 是将通用 LLM 适配为推荐模型的关键步骤
- 相比 YouTube 已高度优化的生产模型 (大规模 embedding table)，PLUM 实现了 **substantial improvements**
- 已部署到 **数十亿 YouTube 用户**

### 与当前项目的关联

- 当前 NTP 模型是从零训练的 39.5M 小模型，没有利用预训练 LLM 的知识
- PLUM 证明了: 即使在推荐这样的非自然语言任务中，LLM 预训练知识 (world knowledge + sequence modeling) 仍然有价值
- **潜在实验**: 用 Qwen3-0.5B 做 CPT → fine-tune 替代当前从零训练的 `AutoregressiveNTPModel`
- 与 IDEA-oneloc-4 (Scaling Law) 直接相关: LLM backbone 自带参数量 scaling，只需研究 CPT 数据量和序列长度

### 实验设计草案

**方案 A (轻量 — LoRA CPT)**:
1. 基座: Qwen3-0.5B (与当前 embedding 模型同系列)
2. 扩展词表: 加入 SID vocab (每层 1024 tokens → 总 3072 新 token)
3. CPT 数据: 用户行为序列 SID 化 → 构造 "user_seq → next_item_sid" 样本
4. LoRA fine-tune (rank=64), 8xA100, ~数小时
5. 评估: Qwen3-0.5B-CPT vs 当前 AutoregressiveNTPModel 的 Recall@K

**方案 B (重量 — Full CPT)**:
- Full fine-tune Qwen3-0.5B on SID 语料
- 更大计算成本，但上限更高

### 关键问题

1. 0.5B 模型做 CPT 的计算成本: 8xA100 能否在合理时间 (< 1天) 完成
2. SID vocab 扩展: 新 token 的 embedding 初始化策略 (随机 vs 语义初始化)
3. 与当前 39.5M 模型的公平对比: 参数量差 10x+，需要同时对比 FLOPS

---

## IDEA-onerec-1: RSFT (Reject Sampling Fine-Tuning — 过滤低质量训练样本)

**优先级**: P1
**来源**: OneRec (arxiv 2506.13695v4) §Post-training
**状态**: 待讨论

### 核心思想

OneRec 的 post-training 阶段不是直接在全量数据上继续训练，而是先用 **Reject Sampling** 过滤: 按用户播放时长将曝光 session 排序，**丢弃后 50% 低质量 session**，只在高质量数据上做 NTP fine-tune。

这解决了"曝光≠喜欢"的问题 — 用户被推荐但没看完的内容不应该作为正样本训练。

### 与当前项目的关联

- 当前 NTP 训练 (`sid_prediction.py`) 使用所有 `action > 0` 的行为作为正样本，包含大量低质量交互 (点了但没看完)
- RSFT 本质是**训练数据质量控制**，不改模型架构，实现成本极低
- 可以在 EXP-007 (contrastive fine-tune) 中也应用: 只用高质量 pair 训练

### 实验设计草案

- 按 `event_cnt` 或行为强度 (like/share > click > view) 过滤训练数据
- 对比: 全量 vs top 50% 高质量数据 的 NTP recall

### 关键问题

1. 我们的行为数据有 `event_cnt` 但没有播放时长 — 需要找替代质量指标
2. 过滤比例 50% 可能太激进，需要调参

---

## IDEA-onerec-2: SID 替代 VID 作为 Encoder 输入 (消除稀疏 Embedding Table)

**优先级**: P2
**来源**: OneRec (arxiv 2506.13695v4) §Semantic ID vs VID Input
**状态**: 待讨论

### 核心思想

OneRec 在 2.6B 规模实验中发现: 用 Semantic ID token 直接作为 encoder 的 item 输入（而非传统的 VID sparse embedding），性能相当甚至更好 (P-score +1.74%)。好处是**消除了巨大的 sparse embedding table** (N × d 参数)，替换为极小的 SID codebook embedding (L × K × d 参数)。

### 与当前项目的关联

- 当前 NTP 模型用 SID tokens 作为输入，已经是这个范式
- 但 OneRec 证明了: 在更大模型规模下，SID 输入不会损失性能，且带来巨大的参数效率提升
- 对我们的启发: 不需要维护 item embedding table，SID codebook embedding 够用

### 关键问题

1. 在小模型 (5M probe) 下可能没有差异
2. 在 LLM backbone (IDEA-plum-0) 场景下更有价值

---

## IDEA-dualgr-0: Exposure-Aware NTP Loss (ENTP-Loss)

**优先级**: P1
**来源**: DualGR (Kuaishou, arxiv 2511.12518, Nov 2025, WWW 2026)
**状态**: 待讨论

### 核心思想

DualGR 发现标准 NTP loss 只学"用户点了什么"，但忽略了"用户看了但没点什么"这个强负信号。ENTP-Loss 引入 **exposure-aware 负样本**:

- 把 **unclicked exposures** 作为 **coarse-level hard negatives**
- 在 SID 第一层 (coarse level) 用这些负样本增强学习信号
- 效果: 模型更快识别用户兴趣衰退 (timely interest fade-out)

DualGR 还提出:
1. **Dual-Branch Long/Short-Term Router (DBR)**: 分离长短期兴趣，selective activation
2. **Search-based SID Decoding (S2D)**: 限制 fine-level 解码在 coarse bucket 内

快手短视频在线 A/B: **video views +0.527%, watch time +0.432%**。WWW 2026。

### 与当前项目的关联

- 当前 NTP 训练 (`sid_prediction.py`) 只用正样本 (用户行为序列)，没有负信号
- ENTP-Loss 是 **零架构改动** 的训练改进: 只需在 loss 计算中加入曝光未点击的 SID 作为负样本
- 与 IDEA-onerec-1 (RSFT) 互补: RSFT 过滤低质量正样本，ENTP 引入高质量负样本
- 与 IDEA-onemall-0 (contrastive loss) 正交: onemall-0 在 embedding 空间做对比，ENTP 在 NTP 的 CE loss 中引入负信号

### 实验设计草案

**实现**:
1. 对每个训练样本，收集同 session 的曝光未点击 item SIDs
2. 在 NTP loss 中，对 coarse level (第一层 SID token) 的 softmax 概率:
   - 降低对 unclicked-exposure SID token 的概率
   - 具体: 在 CE loss 中加入 margin/penalty 项
3. 变量: penalty weight, 只在 L1 还是全层都用负信号

**评估**: NTP recall@K, 特别关注"兴趣变化"场景 (用户最近行为转向新类目)

### 关键问题

1. 需要行为数据包含"曝光未点击"信息 — 当前 `export_behavior.py` 是否已覆盖?
2. Hard negative 太强可能导致模型过于保守 (偏向热门 item)
3. Long/short-term 分支 (DBR) 需要更大的架构改动，可以独立拆分

---

## IDEA-stamp-0: Semantic Adaptive Pruning + Multi-step Auxiliary Prediction (STAMP)

**优先级**: P1
**来源**: STAMP (Alibaba, arxiv 2604.05329, Apr 2026)
**状态**: 待讨论

### 核心思想

STAMP 发现高粒度 SID 存在 **Semantic Dilution Effect**: SID 越长越精细，冗余 token 越多，稀释了学习信号 → 训练效率下降 + 性能不单调波动。

双端优化:
1. **Semantic Adaptive Pruning (SAP)** — 输入端: 前向传播中动态过滤冗余 SID token，将 noisy 序列压缩为 compact 信息密集表示
2. **Multi-step Auxiliary Prediction (MAP)** — 输出端: 用 multi-token prediction 目标替代 single-token NTP，densify 监督信号

**结果**: **1.23-1.38x 训练加速, 17.2%-54.7% VRAM 减少**，性能不降。

### 与当前项目的关联

- 当前 3 层 SID 短序列下不存在 semantic dilution 问题
- **但如果切到 OPQ (16-64 token)**，semantic dilution 会成为关键问题:
  - 64 个 SID token 中很多可能是冗余的 (低信息量子向量)
  - STAMP 的 SAP 可以动态剪掉冗余 token → 解决 OPQ 长 SID 的训练效率问题
- MAP (multi-step prediction) 与 IDEA-sid-4 (MTP auxiliary loss) 方向一致，但 STAMP 更聚焦于作为 **SID 稀疏信号的补偿**
- 1.23-1.38x 训练加速对 8xA100 环境有实际价值

### 实验设计草案

**前置: IDEA-sid-0 Phase 2 (OPQ 长 SID)**

**Phase 1 — MAP (可立即实验)**:
- 在当前 NTP 模型中，除了预测下一个 token，同时预测未来 2-3 个 token
- 增加 2-3 个 projection heads，multi-token CE loss
- L = L_NTP + α * L_MAP

**Phase 2 — SAP (OPQ 后)**:
- 对 OPQ 长 SID 序列，训练 gating module 动态选择信息密集的 token
- 被 prune 的 token 不参与后续 attention 计算

**评估**: 训练时间, VRAM 用量, Recall@K

### 关键问题

1. 当前 3 token SID 太短，pruning 没意义 → 主要价值在 OPQ 路线
2. MAP 与 IDEA-sid-4 (MTP) 重叠，但 STAMP 的动机不同 (densify signal vs cold-start)

---

## IDEA-tbg-0: Next Session Prediction (NSP) — 替代 Item-by-Item 自回归

**优先级**: P1
**来源**: TBGRecall (Alibaba, arxiv 2508.11977, Aug 2025)
**状态**: 待讨论

### 核心思想

标准 GR 逐 item 自回归生成 (A→B→C→D)，存在强序列依赖。TBGRecall 提出 **Next Session Prediction (NSP)**: 将行为划分为多个 session，每个 session 有一个 session token + 多个 item token:

```
[S1] item1 item2 item3 [S2] item4 item5 [S3] → predict [S4] item6 item7
```

session 内 item 无序 (消除 positional bias)，session 间有序 (保留时间依赖)。

另一关键发现: **data recency > data volume** — 用少量最近数据训练 > 用大量历史数据训练。

在公开数据集和阿里工业数据集上均展示 **clear scaling law trend**。

### 与当前项目的关联

- 当前 NTP 模型逐 item 预测，每个 item 的 SID 序列是独立自回归的
- NSP 提供了 **更高层的抽象**: 预测"下一个 session"而非"下一个 item"
- **data recency insight** 直接可用: 训练时给近期行为更高权重，或只用最近 N 天数据
- 与 IDEA-onerec-1 (RSFT) 互补: RSFT 过滤低质量样本，NSP 改变建模粒度

### 实验设计草案

**Phase 1 — Data Recency 验证 (零成本)**:
- 在当前 NTP 训练中，对比: 全量历史 vs 最近 30 天 vs 最近 7 天
- 如果 recency > volume，可以大幅降低训练成本

**Phase 2 — Session-Level Prediction**:
- 在用户行为中划分 session (按时间间隔 > 30 min)
- 在 NTP 输入中插入 [SESSION] token
- session 内 item 随机打乱 (去除位置 bias)

### 关键问题

1. Session 划分规则: 按时间间隔? 按行为类型?
2. Session 内无序可能丢失 fine-grained 时间信号
3. 当前行为数据是否有 timestamp 支持 session 划分

---

## IDEA-hstu1b-0: Task Decomposition for Scaling (Feedback + Next-Item 分离)

**优先级**: P1
**来源**: Scaling Recommender Transformers to 1B (arxiv 2507.15994, Jul 2025, KDD 2026)
**状态**: 待讨论

### 核心思想

在 HSTU/Generative Recommenders 框架上，将自回归学习 **分解为两个子任务**:

1. **Feedback Prediction**: 预测用户对已展示 item 的反馈 (like/dislike/skip)
2. **Next-Item Prediction**: 预测用户接下来会交互的 item

这个分解在 176M → 1B 参数范围内保持有效 scaling。

音乐流媒体平台部署: **listening time +2.26%, user likes +6.37%** — 作者声称是该平台深度学习系统历史上最大单次提升。KDD 2026。

### 与当前项目的关联

- 当前 NTP 模型只做 next-item prediction (任务 2)，完全没有 feedback prediction (任务 1)
- Task decomposition 的 insight: **用户反馈本身是有价值的监督信号**，不仅仅是"预测下一个 item"
- 实现简单: 在用户序列中加入 feedback token (liked/skipped/watched_full)，让模型同时预测 feedback + next item
- 与 IDEA-oneloc-5 (Multi-behavior) 有关联但不同: oneloc-5 区分行为类型作为输入，本 IDEA 把 feedback 作为预测目标

### 实验设计草案

**方案 — Dual-Task NTP**:
1. 用户序列: `item1 [FEEDBACK:like] item2 [FEEDBACK:skip] item3 → predict [FEEDBACK:?] item4`
2. 模型同时预测 feedback token 和 next-item SID
3. `L = L_next_item + α * L_feedback`
4. 变量: α ∈ {0.1, 0.5, 1.0}

**评估**: NTP recall@K (core metric) + feedback prediction accuracy (auxiliary metric)

### 关键问题

1. 需要行为数据包含反馈类型 (like/skip/watch_full 等)
2. 当前 39.5M 小模型下分解是否有价值? 论文在 >176M 才看到 scaling 效果
3. Feedback token 增加序列长度 ~2x → 训练成本增加

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P0 | IDEA-onemall-0 | NTP In-Batch Contrastive Loss | 实现简单，OneMall 标配，为后续 RL 建立更强基线 |
| P1 | IDEA-sid-4 | Token-Space MTP 辅助 Loss | RPG 证明 token-space CE > item-space CE，冷启动友好 |
| P1 | IDEA-gr4ad-2 | Value-Aware 训练 | 丰富训练信号，与 IDEA-sid-1 互补 |
| P1 | IDEA-oneloc-5 | Multi-behavior 序列融合 | 低成本区分不同行为强度 |
| P1 | IDEA-plum-0 | LLM Continued Pre-Training | YouTube 数十亿用户验证，利用预训练知识 |
| P1 | IDEA-onerec-1 | RSFT 过滤低质量训练样本 | 零成本数据质量提升，OneRec 标配 |
| P1 | IDEA-dualgr-0 | Exposure-Aware NTP Loss | 快手 WWW 2026, 零架构改动引入负信号 |
| P1 | IDEA-stamp-0 | Semantic Pruning + MTP | 解决 OPQ 长 SID 的训练效率, 1.23x 加速 |
| P1 | IDEA-tbg-0 | Next Session Prediction + Data Recency | 阿里验证 scaling law, data recency > volume |
| P1 | IDEA-hstu1b-0 | Task Decomposition (Feedback + Next-Item) | KDD 2026, 历史最大提升, 1B 参数 scaling |
| P2 | IDEA-sid-5 | Codebook Embedding 聚合 | 依赖 IDEA-sid-0 Phase 2，短 ID 下收益不大 |
| P2 | IDEA-onerec-2 | SID 替代 VID 输入 | 大模型场景下有价值，当前无需 |
