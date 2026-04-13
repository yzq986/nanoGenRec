# 语义 ID 构造方法

**来源**: 知乎综述文章 3.1 节 + Meta RPG (KDD'25, arxiv 2506.05781)
**日期**: 2026-04-13
**更新**: 2026-04-13 (RPG 论文细节补充)

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
