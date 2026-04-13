# 语义 ID 构造方法

**来源**: 知乎综述文章 3.1 节 + Meta RPG (KDD'25, arxiv 2506.05781)
**日期**: 2026-04-13
**更新**: 2026-04-13 (RPG 论文完整消融 + 新 IDEA 补充)

---

## IDEA-sid-0: OPQ 并行语义 ID

**优先级**: P0
**来源**: 3.1.2.2 (Meta RPG, KDD'25), Kaiming OPQ
**状态**: 已采纳 → EXP-004 (Phase 1 intrinsic), Phase 2 待定
**参考代码**: github.com/facebookresearch/RPG_KDD2025

### 核心思想

用 Optimized Product Quantization 替代残差编码，实现并行语义 ID。先学正交旋转矩阵 R，再将旋转后向量切分子向量独立量化。

### RPG 论文关键细节

**量化**: OPQ > RQ（消融实验 Table 3 证实）。每个 digit 的 codebook 是独立词表（C⁽¹⁾={1,...,M}, C⁽²⁾={M+1,...,2M}），不共享。

**模型架构**:
- Transformer encoder 对用户行为序列编码 → s ∈ ℝᵈ
- **每个 digit 一个独立 MLP projection head** g_j(s) → M 维 logits
- MTP loss = Σ CE_j（各 digit 条件独立）
- 单次 forward 出所有 token logits，非自回归
- 独立 head >> shared head（消融证实）

**推理 — Graph-Constrained Decoding（不是 beam search 也不是笛卡尔积）**:
- **beam search 在 OPQ 上完全失败**（Table 3 recall 全是 0.0000）
- **纯笛卡尔积组合也不行** — 256^32 空间太稀疏，大部分组合是无效 SID
- 构造 SID 相似度图: 节点=有效 SID, 边=embedding 相似度 top-k neighbors
- 推理: 随机采样 b=10 种子 → 沿图边扩展 → 打分 top-b → 迭代 q 轮
- 复杂度 O(Mmd + bqkm)，与 item 总数 N 无关
- 最终只访问 ~10-25% 的 item pool

**最优配置** (RPG 论文):
- m=16~64 (数据集越大越长), M=256, b=10, k=50~500, q=2~5
- τ=0.03 (温度), 2-layer Transformer, d=448, ~13M params
- 训练: <2 GPU hours (RTX 3090)

**关键消融结论**:
- OPQ > RQ (NDCG +2~8%)
- 长 ID (16~64) >> 短 ID (4)，且 TIGER (自回归+RQ) 在长 ID 下 OOM
- 独立 projection head >> shared head >> no head
- Graph decoding 比无图约束好 3x

### 与当前项目的关联

- `ARCHITECTURE.md` 已明确: "必须用平行 tokenizer，不能用残差编码"
- 直接解决 "残差编码永远不可能思考" 问题
- FAISS 有现成 `faiss.OPQMatrix` + `faiss.ProductQuantizer` 实现
- RPG 开源代码可直接参考

### 实验设计草案

**分两阶段实验**:

#### 阶段 1: OPQ 量化质量 (intrinsic metrics only)

验证 OPQ 在我们 5M item / 1024D embedding 上的量化质量。

**配置** (1024D embedding, 对标 RPG 论文主配置):

| 方案 | 子向量维度 | token 数 m | 词表大小 M | 编码空间 |
|------|-----------|-----------|-----------|----------|
| A | 128D | 8 | 256 | 256^8 |
| B | 64D | 16 | 256 | 256^16 |
| C | 32D | 32 | 256 | 256^32 |

**对比基线**: RKMeans 3 层 x 1024 clusters (EXP-001 final config, collision=1.75%)

**评估指标**: recon_loss, collision_rate, exclusivity, entropy, cluster_balance

**注意**: collision_rate 在 OPQ 下可能意义不同 — 8~32 个 token 的 ID 空间远大于 3 token，collision 应该极低。重点看 recon_loss 是否优于 RKMeans。

#### 阶段 2: 并行预测模型 + Graph Decoding

需要新实现:
1. 并行预测模型: 替换当前 AutoregressiveNTPModel，每 digit 独立 MLP head
2. Graph 构造: SID 相似度图 (top-k neighbors per node)
3. Graph decoding: 种子采样 → 图传播 → 打分 → 迭代

### 关键问题

1. **词表大小 M 的选择**: RPG 固定 M=256。我们的 RKMeans 用 1024 per layer。需要验证 M=256 vs M=1024 在 OPQ 下的差异。
2. **Graph 构造成本**: 5M items 的 SID 相似度图构造需要 O(N²) 或近似 ANN，需要评估内存和时间。
3. **与 OneRec 的关系**: OneRec 用 3 token x 8192 parallel tokenizer，RPG 用 16~64 token x 256。两种 parallel 方案的 tradeoff 需要理清 — OneRec 适合自回归 (短序列)，RPG 适合并行预测 (长序列+图解码)。

---

## IDEA-sid-1: 协同信号增强 Embedding

**优先级**: P1
**来源**: 3.1.1 (OneRec-V1 技术报告)
**状态**: 待讨论

### 核心思想

当前直接用 Qwen3 文本 embedding 做量化，但推荐系统需要的语义相似还包含协同行为信号。通过 Item Pair 对比学习将协同信号注入 embedding。

### 与当前项目的关联

- 已有 `data/export_behavior.py` 导出行为数据
- 已有 `eval/behavior.py` 行为指标评估框架
- 对 embedding 本身的改进，不管后续用什么量化方案都受益
- 与量化方法实验 (EXP-003, IDEA-sid-0) 正交，可并行推进

### 实验设计草案

**Item Pair 构造**:
- 方式 1: 用户点击 target item + 最近正向行为 item
- 方式 2: Swing I2I 高分 item pair

**训练方案**:
- 方案 A (轻量): 冻结 Qwen3，只训练 projection head，对比学习 loss
- 方案 B (重量): 微调 Qwen3-0.6B，对比学习 loss + 文本 loss

**评估**: 原始 Qwen3 embed vs 增强 embed → 同一 RKMeans config → 对比 collision / exclusivity / behavior metrics

### 关键问题

1. Item Pair 样本量是否足够（需要检查行为数据覆盖率）
2. 方案 A (projection head) 是否足够，还是需要 finetune 整个模型
3. 对比学习的负样本策略: in-batch negatives? hard negatives?

---

## IDEA-sid-2: Balanced KMeans

**优先级**: P1
**来源**: 3.1.2.1 (OneRec Paper 提到)
**状态**: 待讨论

### 核心思想

用平衡 KMeans 替代标准 KMeans，强制各 cluster 大小均匀，提高码本利用率。

### 与当前项目的关联

- EXP-001 中 cluster_balance (Gini) 仍有优化空间
- 实现成本极低，可快速验证
- 文中还提到 "原向量和残差向量都可以做归一化" — 当前只 normalize layer 0 input

### 实验设计草案

**变量**:
- 标准 KMeans vs Balanced KMeans
- 残差归一化: 仅 L0 (当前) vs 每层都 normalize

**注意**: EXP-001 结论 "normalize_residuals 只对 layer 0" 可能需要重新验证，因为当时可能没用 balanced assignment

### 关键问题

1. FAISS 不直接支持 balanced KMeans，需要用 `faiss-contrib` 或自实现
2. 每层 normalize 残差是否会与 EXP-001 结论冲突

---

## IDEA-sid-3: 多模态语义 ID (ESANS 粗细粒度)

**优先级**: P2
**来源**: 3.1.3 (阿里 WWW'25 ESANS)
**状态**: 待讨论

### 核心思想

多模态表征不是简单 concat 再量化，而是:
- L1 (粗粒度): 多模态表征均值做聚类
- L2 (细粒度): 各模态残差 concat 后做聚类

### 与当前项目的关联

- 当前只用 `qwen3-0.6b` 单文本模态
- Config 中已有 `qwen3-vl-8b` / `qwen3-vl-2b` 多模态模型
- 需要先完成多模态 embedding 基建

### 前置依赖

1. 多模态 embedding 生成 pipeline (用 qwen3-vl)
2. 多模态表征对齐 (CLIP-style 或 ESANS encoder)
3. 数据侧: 需要 item 的图片/视频数据

### 关键问题

1. 多模态 embed 的存储和计算成本 (qwen3-vl-8b 的 4096D)
2. 模态对齐训练的复杂度
3. 是否先在小规模数据上验证粗细粒度方案的收益

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

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P0 | IDEA-sid-0 | OPQ 并行语义 ID (已 → EXP-004) | ARCHITECTURE.md 核心方向，RPG 完整验证，代码已就绪 |
| P1 | IDEA-sid-1 | 协同信号增强 Embedding | 与量化方案正交，改善 embedding 质量对所有下游实验受益 |
| P1 | IDEA-sid-2 | Balanced KMeans | 低成本改进码本利用率 |
| P1 | IDEA-sid-4 | Token-Space MTP 辅助 Loss | RPG 证明 token-space CE > item-space CE，冷启动友好，可叠加到自回归模型 |
| P2 | IDEA-sid-3 | 多模态语义 ID (ESANS) | 需要多模态 embedding 基建 |
| P2 | IDEA-sid-5 | SID Codebook Embedding 聚合 | 依赖 IDEA-sid-0 Phase 2 (OPQ 长 ID)，短 ID 下收益不大 |
