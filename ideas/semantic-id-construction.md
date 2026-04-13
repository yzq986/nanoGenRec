# 语义 ID 构造方法

**来源**: 知乎综述文章 3.1 节（语义ID的构造）
**日期**: 2026-04-13

---

## IDEA-001: OPQ 并行语义 ID

**优先级**: P0
**来源**: 3.1.2.2 (Meta RPG, Kaiming OPQ)
**状态**: 待讨论

### 核心思想

用 Optimized Product Quantization 替代残差编码，实现并行语义 ID。先学正交旋转矩阵 R，再将旋转后向量切分子向量独立量化。

### 与当前项目的关联

- `ARCHITECTURE.md` 已明确: "必须用平行 tokenizer，不能用残差编码"
- 直接解决 "残差编码永远不可能思考" 问题
- FAISS 有现成 `faiss.OPQMatrix` 实现

### 实验设计草案

**配置候选** (1024D embedding):

| 方案 | 子向量维度 | token 数 | 词表大小 | 编码空间 |
|------|-----------|----------|----------|----------|
| A | 128D | 8 | 1024 | 1024^8 |
| B | 64D | 16 | 256 | 256^16 |
| C | 32D | 32 | 256 | 256^32 |

**对比基线**: RKMeans 3 层 x 1024 clusters (EXP-001 final config)

**评估指标**: recon_loss, collision_rate, exclusivity, NTP beam search recall

### 关键问题

1. token 数量 vs 词表大小的 tradeoff — OneRec 用 3 token x 8192, OPQ 倾向多 token x 小词表
2. NTP 模型需要适配更长的 token 序列，beam search 成本增加
3. 需要验证: 并行 ID 的 NTP 预测是否确实比残差 ID 更好

---

## IDEA-002: 协同信号增强 Embedding

**优先级**: P1
**来源**: 3.1.1 (OneRec-V1 技术报告)
**状态**: 待讨论

### 核心思想

当前直接用 Qwen3 文本 embedding 做量化，但推荐系统需要的语义相似还包含协同行为信号。通过 Item Pair 对比学习将协同信号注入 embedding。

### 与当前项目的关联

- 已有 `data/export_behavior.py` 导出行为数据
- 已有 `eval/behavior.py` 行为指标评估框架
- 对 embedding 本身的改进，不管后续用什么量化方案都受益
- 与量化方法实验 (EXP-003, IDEA-001) 正交，可并行推进

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

## IDEA-003: Balanced KMeans

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

## IDEA-004: 多模态语义 ID (ESANS 粗细粒度)

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
