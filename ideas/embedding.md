# Embedding (表征增强)

量化前的 embedding 质量决定了语义 ID 的上限。涵盖协同信号注入、多模态融合、属性增强等方向，与量化方法正交，改善 embedding 对所有下游实验受益。

**影响范围**: `model/encode.py`, `model/embedders.py`, `data/export_behavior.py`

---

## 演进路径

```
Qwen3 纯文本 embedding (1024D, 当前 baseline)
├── IDEA-sid-1: 协同信号增强 (I2I 对比学习)
├── IDEA-onemall-3: 属性增强 (category/price/shop 对比学习)
├── IDEA-sid-3: 多模态 ESANS (粗细粒度多模态融合)
└── IDEA-oneloc-3: Side-info 融合 (量化输入空间丰富化)
    └── 与 IDEA-sid-1 统一为 "embedding enrichment" 框架
```

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

## IDEA-onemall-3: Tokenizer Auxiliary Contrastive Loss (属性增强)

**优先级**: P1
**来源**: OneMall §4.5 Component Analyses (Aux Loss row)
**状态**: 待讨论

### 核心思想

在 tokenizer 的 embedding backbone 训练中，加入 item 属性 (category, price, shop) 作为辅助信号。OneMall 将这些属性 feed 进 item tower，用对比学习 loss 训练，在 HR@50/100/500 上分别提升 +1.5%/+1.7%/+1.7%。

这与 IDEA-sid-1 (协同信号增强 embedding) 互补:
- IDEA-sid-1: 用用户行为 I2I 对注入协同信号
- 本 IDEA: 用 item 属性注入结构化商业语义

### 与当前项目的关联

- 当前 embedding 纯粹来自 Qwen3 文本编码，没有结构化属性注入
- item 元数据 (category, brand, price) 应该在行为数据中可获取
- 可以在 `model/embedders.py` 的 `Qwen3TextEmbedder` 基础上加 attribute projection head
- **与 EXP-003 (Learned FSQ) 方向一致**: 都是在量化前改善 embedding 质量

### 实验设计草案

**方案 A (轻量 — 推荐先做)**:
- 冻结 Qwen3，加 `AttributeProjectionHead`: MLP(attr_features → 128)
- item text embedding (1024D) + attribute embedding (128D) → concat → MLP → 最终 embedding
- 对比学习: 同 category 的 item pair 做正样本，不同 category 做负样本

**方案 B (重量)**:
- 与 IDEA-sid-1 合并: I2I 协同信号 + 属性信号同时注入

**评估**: 原始 Qwen3 embed vs 属性增强 embed → 同一 RKMeans config → collision / exclusivity / behavior metrics

### 关键问题

1. 需要确认数据中有哪些可用的 item 属性字段
2. category 层级结构 (一级/二级/三级分类) 如何编码
3. 连续属性 (price) 的离散化/归一化策略

---

## IDEA-oneloc-3: Geo-aware Semantic ID (Side-info 融合量化输入)

**优先级**: P1
**来源**: OneLoc §2.2 Geo-aware Semantic IDs
**状态**: 待讨论

### 核心思想

OneLoc 在残差量化的初始 embedding 中融合地理上下文: `r_0 = concat(e_video, e_location_context)`，使得生成的 semantic ID 本身就编码了地理语义。地理上下文由多模态大模型从 GeoHash 的品牌、类目、热销品信息中提取。

### 与当前项目的关联

- 当前 `model/rkmeans.py` 的输入是纯 Qwen3 文本 embedding
- 这与 **IDEA-sid-1 (协同信号增强 Embedding)** 思路一致: 在量化之前将额外信号注入 embedding
- 泛化形式: **任何 side information 都可以在量化前 concat/fuse 到 embedding 中**
- 具体对我们的启发: 除了协同信号 (IDEA-sid-1)，还可以注入:
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

1. **与 IDEA-sid-1 重叠**: 协同信号增强也是改 embedding 输入。应统一为 "embedding enrichment" 框架，避免重复实验
2. 当前有什么 side information 可以用? 需要检查 item metadata 中除文本外还有什么字段
3. Concat 后维度增加对量化质量的影响 — 高维可能让 KMeans 更难收敛
4. 如果走 IDEA-sid-0 (OPQ) 路线，side info 可以分配到独立子向量，天然适合并行量化

---

## IDEA-onerec-0: Caption Generation Loss (防止协同微调遗忘语义)

**优先级**: P1
**来源**: OneRec (arxiv 2506.13695v4) §Tokenizer Training
**状态**: 待讨论 — 直接关联 EXP-007

### 核心思想

OneRec 在 tokenizer 的对比学习训练中同时加入 **caption generation loss**:
- 对比 loss (`L_I2I`): 拉近协同 pair
- Caption loss (`L_caption_gen`): 给定 item 的多模态表示，预测 item 的文本 caption (next-token prediction)

Caption loss 的作用是 **"prevents hallucination by performing next-token prediction on video captions"** — 防止对比学习过度拟合协同信号而丢失内容语义。

### 与当前项目的关联

- **EXP-007 目前只有 InfoNCE loss**，没有语义保持机制。如果 3 epoch 训练后 embedding 丢失了文本语义（cosine_similarity 分布变差），说明需要加 caption loss
- Qwen3-Embedding-0.6B 是 encoder 模型，不直接支持 causal LM generation
- **替代方案**: 用 contrastive loss 保持语义 — 同 item 微调前后的 embedding 做正样本 (anchor preservation)，或加一个轻量 text reconstruction head

### 实验设计草案

**方案 A — Embedding Anchor Preservation (推荐，最简单)**:
```
L = L_InfoNCE + β * L_anchor
L_anchor = 1 - cos(embed_finetuned, embed_original)
```
冻结一份原始 Qwen3 作为 anchor，微调后的 embedding 不能离原始太远。

**方案 B — Text Reconstruction Head**:
- 在 Qwen3 encoder 输出上加一个轻量 decoder head
- 预测 item title tokens
- `L = L_InfoNCE + β * L_text_recon`

**评估**: 对比有/无 caption loss 的 embedding 在 `embedding_hit_rate` + `cosine_similarity` 上的变化

### 关键问题

1. β 的选择: 太大压制协同学习，太小没效果
2. 方案 A 的 anchor preservation 可能过于保守 — 限制了 embedding 空间的移动幅度
3. EXP-007 结果出来后看 `cosine_similarity` 是否退化，决定是否需要加

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P1 | IDEA-sid-1 | 协同信号增强 Embedding | 与量化方案正交，改善 embedding 质量对所有下游实验受益 |
| P1 | IDEA-onemall-3 | Tokenizer 属性增强 Contrastive | OneMall +1.5% HR，与 IDEA-sid-1 互补 |
| P1 | IDEA-onerec-0 | Caption Loss (防遗忘语义) | EXP-007 的补充，OneRec 标配 |
| P1 | IDEA-oneloc-3 | Side-info 融合量化输入 | 与 IDEA-sid-1 统一为 "embedding enrichment" |
| P2 | IDEA-sid-3 | 多模态语义 ID (ESANS) | 需要多模态 embedding 基建 |
