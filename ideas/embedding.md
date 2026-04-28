# Embedding (表征增强)

量化前的 embedding 质量决定了语义 ID 的上限。涵盖协同信号注入、多模态融合、属性增强等方向，与量化方法正交，改善 embedding 对所有下游实验受益。

**影响范围**: `model/encode.py`, `model/embedders.py`, `data/export_behavior.py`

---

## 演进路径

```
Qwen3 纯文本 embedding (1024D, 当前 baseline)
├── IDEA-sid-1: 直接 fine-tune Qwen3 (I2I 对比学习)
│   └── EXP-007 验证: full/LoRA fine-tune 均无效, HR@50 卡在 ~0.02
├── IDEA-onerec-3: QFormer Tokenizer (冻结 Qwen3 + cross-attention 压缩)  ★ 推荐
│   └── OneRec/BLIP-2 方案, 信息瓶颈 + 梯度集中, 解决 EXP-007 的根本问题
├── IDEA-onerec-0: Caption Loss (防遗忘语义, 已实现 --cap_loss_weight)
├── IDEA-onemall-3: 属性增强 (category/price/shop 对比学习)
├── IDEA-sid-3: 多模态 ESANS (粗细粒度多模态融合)
└── IDEA-oneloc-3: Side-info 融合 (量化输入空间丰富化)
    └── 与 IDEA-sid-1 统一为 "embedding enrichment" 框架
```

---

## IDEA-sid-1: 协同信号增强 Embedding

**优先级**: ~~P1~~ → ❌ 关闭
**来源**: 3.1.1 (OneRec-V1 技术报告)
**状态**: ❌ 关闭 — EXP-007 全量 fine-tune + LoRA 多种 lr/τ 全部失败 (HR@50 卡在 ~0.02)；EXP-009 冻结底座 + QFormer 同样 HR@50=0.0216 几乎无改善。根因：I2I contrastive 信号不足以弥补 semantic embedding 与行为空间的 gap。embedding fine-tune 路线关闭。

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

## IDEA-onerec-3: QFormer Tokenizer (冻结底座 + Cross-Attention 压缩)

**优先级**: ~~P0~~ → P2 暂缓
**来源**: OneRec (arxiv 2506.13695v4) §Tokenizer + BLIP-2 QFormer
**状态**: 暂缓 — EXP-007 验证了直接 fine-tune 无效；QFormer 是理论上的正确方向，但当前 NTP 已靠 MLP-FSQ + RL 对齐取得进展，embedding 改进路线 ROI 不明确。优先完成 RL 链路 (EXP-037→039) 后再评估

### 核心思想

EXP-007 的根本问题: 直接在 Qwen3-0.6B 上做 contrastive fine-tune（不管全量还是 LoRA）都推不动模型——cap_loss 纹丝不动，HR@50 卡在 ~0.02。

OneRec 的解决方案: **不动底座，在上面加一个可训练的 QFormer**。

```
OneRec 架构:
  miniCPM-V-8B (frozen, 8B) → 1280 tokens × 512d
      ↓
  QFormer (trainable, 4 layers, 4 query tokens)
      ↓
  4 tokens × 512d (压缩后的 item 表征)
      ↓
  L_I2I (InfoNCE) + L_caption_gen (next-token prediction)
      ↓
  RQ-KMeans → 3 层 SID

我们的适配:
  Qwen3-Embedding-0.6B (frozen) → S tokens × 1024d (last hidden states)
      ↓
  QFormer (trainable, N layers, M query tokens)
      ↓
  M tokens × D (压缩后的 item 表征)
      ↓
  L_I2I (InfoNCE, 已有) + L_caption (已实现 --cap_loss_weight)
      ↓
  OPQ 量化 → SID
```

### 为什么 QFormer 能解决 EXP-007 的问题

| 问题 | 直接 fine-tune (EXP-007) | QFormer |
|------|---|---|
| 梯度信号稀释 | I2I 梯度摊到 600M 参数，约等于没有 | 梯度集中在 QFormer (~30-50M)，底座冻结 |
| 语义遗忘 | cap_loss 监控不变 = 模型没动 | 底座冻结 = 天然保持语义 |
| 信息瓶颈 | 无，1024d 全部传递 | S×1024 → M×D 强制压缩，学会提取协同相关信息 |
| 优化目标 | "微调整个表征空间" (太大) | "学会从丰富表征中选择什么" (更直接) |

### QFormer 关键设计 (来自 BLIP-2 + OneRec)

**Learnable Query Tokens**: M 个可训练的 query 向量，通过 cross-attention "询问" frozen encoder 的 hidden states。

**Cross-Attention 机制**:
```
Q = learnable_queries          (M × D)
K, V = encoder_hidden_states   (S × 1024)
Output = CrossAttn(Q, K, V)    (M × D)
```

**关键超参**:
- M (query tokens): OneRec 用 4，OneMall 用 10/type。我们可以从 {4, 8, 16} 搜索
- QFormer layers: OneRec 4 层。从 {2, 4} 开始
- Output dim D: 与 OPQ 子向量维度对齐（当前 m=8, sub_dim=128 → D=1024 或压缩到 512）
- 最终 embedding: mean-pool M 个 query token → 单个向量 → OPQ

### 实验设计草案

**Phase 1 — 最小验证 (验证梯度能否流动)**:
- M=4 query tokens, 2 层 QFormer, D=1024
- 冻结 Qwen3, 只训练 QFormer
- L_I2I only, 500K pairs, lr=1e-4
- 关注: cap_loss 是否开始变化，HR@50 是否突破 0.02

**Phase 2 — 加 Caption Loss**:
- L = L_I2I + λ * L_caption (--cap_loss_weight)
- 对比有/无 caption loss 的 HR@50 差异

**Phase 3 — 超参搜索**:
- M ∈ {4, 8, 16}
- QFormer layers ∈ {2, 4}
- lr ∈ {1e-4, 5e-4, 1e-3}

**评估**: HR@50 (与 EXP-007 直接对比), cap_loss 变化量

### 实现要点

1. **新建 `model/qformer.py`**: QFormer 模块 (cross-attention + FFN + learnable queries)
2. **修改 `model/contrastive_finetune.py`**:
   - `--use_qformer` flag
   - 冻结 Qwen3，取 `last_hidden_state` (不只是最后一个 token)
   - QFormer 处理 hidden states → 得到压缩表征
   - 压缩表征做 InfoNCE + caption loss
3. **修改 `model/encode.py`**: 推理时加载 QFormer，生成压缩 embedding
4. **量化 pipeline**: OPQ 输入维度可能变化，需适配

### 与 architecture.md IDEA-onemall-1 的区别

| | IDEA-onemall-1 (architecture.md) | IDEA-onerec-3 (本 IDEA) |
|---|---|---|
| 层面 | NTP ranking 阶段 | Embedding/Tokenizer 阶段 |
| 压缩什么 | 用户行为序列 (1205→160 tokens) | Item 多模态/文本表征 (S→M tokens) |
| 目的 | 减少 NTP decoder FLOP | 产出用于量化的 item embedding |
| 训练信号 | NTP next-token loss | I2I contrastive + caption loss |

两者是 QFormer 在不同阶段的应用，互不冲突，可以共存。

### 关键问题

1. **输出格式**: QFormer 输出 M 个 token，OPQ 期望单个向量。需要 pooling (mean/cls) 或展开为更长向量
2. **Qwen3-Embedding 是 encoder**: hidden states 是双向的 (非 causal)，QFormer 的 cross-attention 可以利用全部 context
3. **训练成本**: QFormer ~30-50M 参数，比 LoRA 略大但远小于全量 fine-tune
4. **推理变化**: encode 时需要多跑一个 QFormer forward，增加 ~5% 推理时间

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| ~~P0~~ P2 暂缓 | ~~IDEA-onerec-3~~ | ~~QFormer Tokenizer~~ | 暂缓 — NTP+RL 路线已取得进展，embedding 改线 ROI 不明确，RL 链路完成后再评估 |
| P1 | IDEA-onerec-0 | Caption Loss (联合训练) | 已实现 `--cap_loss_weight`, 与 QFormer 配合使用 |
| P1 | IDEA-onemall-3 | Tokenizer 属性增强 Contrastive | OneMall +1.5% HR，可在 QFormer 基础上叠加 |
| P1 | IDEA-oneloc-3 | Side-info 融合量化输入 | QFormer 输入端可融合 side-info |
| ~~P1~~ ❌ | ~~IDEA-sid-1~~ | ~~直接 fine-tune 协同信号~~ | ❌ EXP-007 full/LoRA + EXP-009 QFormer 全败，HR@50 卡在 0.02 |
| P2 | IDEA-sid-3 | 多模态语义 ID (ESANS) | 需要多模态 embedding 基建 |
| P1 | IDEA-marc-0 | Mid-Layer 选择 + Modular Compression | MARC SIGIR 2026 eCPM +2.82% A/B；Phase 1 (layer sweep) 几乎零成本，直接影响 SID 质量上限 |

---

## IDEA-marc-0: Mid-Layer Representation Advantage + Modular Compression

**优先级**: P1
**来源**: MARC (Huawei Noah + SJTU, arxiv 2604.18146, SIGIR 2026)
**状态**: 待讨论 — Phase 1 (layer sweep) 可立即执行, 风险低

### 核心思想

MARC 系统性地研究了"将 LLM 表征用于推荐时应该从哪一层取 embedding"这个问题，得出两个重要发现：

**1. Mid-layer Representation Advantage (MRA) — 反直觉现象**

在 Llama3-8B / Qwen2-7B / Qwen2-1.5B 上做 CTR 微调（对比学习、MRL、LARR、next-token、cosine-sim 多种 proxy 任务），**中间层表征的下游 CTR AUC 始终优于最终层**，且无论用什么 proxy loss 都一致出现。作者在 MovieLens-1M 和 Yelp 上都复现了这一现象。

**2. 模块化理论解释**

LLM 在微调期间自动形成功能分工：
- **Representation Learning Module (早期到中间层)**: 提取通用的语义特征，保留丰富信息
- **Task Adaptation Module (最后几层)**: 被 proxy loss 强制塌陷为任务特化头，过滤掉对 CTR 有用但 proxy 任务"不需要"的多样性信息

最终层其实是一个 **unintended information bottleneck** — 把对推荐有用的信号挤出去了。这解释了为什么取中间层反而更好。

**3. MARC 框架（显式模块化）**

三个组件解耦：
- **LLM backbone**: 只做 representation learning，不被强加 task head 责任
- **Compression Network**: 独立的轻量网络做维度压缩（从 LLM 隐藏维度 → 推荐用维度）
- **User-Item Matching Network**: 独立网络做 CTR-style 匹配/预测

加 **HSIC (Hilbert-Schmidt Independence Criterion)** 作为约束：最大化压缩前后表征的互信息，同时强制 compression 和 matching 模块的输出彼此独立。

**4. 实验结果**

- MARC 的最终层表征超越所有基线的最佳中间层（MARC 修复了 MRA）
- 在线 A/B 测试 **eCPM +2.82%**（Huawei 商业搜索广告场景）

### 与当前项目的关联

**这是对我们 Qwen3-0.6B → MLP-FSQ 管线的直接挑战**：

- 当前 `Qwen3TextEmbedder` 取的是 final layer (EOS token pooling 或 last hidden state)，1024D → MLP-FSQ
- MARC 暗示: **中间层的 embedding 可能产生更好的 SID** — 保留了更多语义多样性，减少 tokenizer 重建压力
- 我们 EXP-007/009 的 fine-tune 路线 (sid-1) 失败是因为强加 CF proxy 反而压坏 final layer — MARC 的理论恰好解释了这个现象
- **Phase 1 实验零成本**: 不改 fine-tune 逻辑，只改 "取哪一层 hidden state" → 重跑 tokenizer → 重算 semantic_neighbor_HR

与 EXP-007/009 的关系：
- EXP-007/009 结论 ("fine-tune 路线不通") 仍然成立
- MRA 提供了新的解释: final-layer 在微调时总会退化，与"用什么 proxy loss" 几乎无关
- 新路径: 保持 Qwen3 **不 fine-tune**，但**换个 layer 取 embedding** — 可能绕过 fine-tune 失败并拿到更好 SID

与 IDEA-onerec-3 (QFormer Tokenizer) 的关系：
- QFormer 思路: 冻结底座 + Cross-Attention 从多层聚合 → MARC 是 QFormer 的理论佐证（多层聚合比单纯 final layer 更优）
- MARC Phase 1 (单层选择) 比 QFormer (多层聚合) 轻得多，可作为前置 ablation

### 实验设计草案

**Phase 1 — Qwen3 Layer Sweep (极低成本, ~1 天)**:

1. 在 `model/embedder.py::Qwen3TextEmbedder` 增加 `hidden_layer: int = -1` 参数
2. 用 `output_hidden_states=True` + `hidden_states[hidden_layer]` 取指定层
3. Qwen3-0.6B 有 ~28 层，sweep `{2, 7, 14, 21, 27}` (early, mid-early, mid, mid-late, final)
4. 对每层：
   - 重算 embedding cache (~几小时)
   - 重训 MLP-FSQ tokenizer (~30 分钟)
   - 评估 `semantic_neighbor_hit_rate@50` (秒级)

**预期**: 如果 MRA 在 Qwen3 上也成立（论文在 Llama3/Qwen2 上都复现），应该观察到 mid-layer (第 14-21 层) semantic_neighbor_HR 明显优于 final layer (27)。

| 假想结果 | 解读 | 下一步 |
|---------|------|-------|
| mid > final 显著 | MRA 在我们场景成立 | 切换默认 layer, 重训 NTP，看端到端 R@K 提升 |
| mid ≈ final | 我们的 embedding 路径不受 MRA 影响（Qwen3 预训练 + 无微调可能不符合 MARC 假设） | 跳过 Phase 2，但至少 ruled out 一个变量 |
| mid < final | MRA 反向 | 意外，需深入分析；可能我们的 MLP-FSQ 已经隐式补偿了 |

**Phase 2 — MARC 完整框架 (高成本, 需 fine-tune)**:

如果 Phase 1 显示 mid-layer 有优势，但单层选择收益有限，考虑完整 MARC：
- 引入独立的 Compression Network (当前已有 MLP-FSQ encoder, 可视为已具备)
- 引入 User-Item Matching Network (独立小 MLP 做交互)
- 加 HSIC 约束（在 tokenizer 训练阶段）
- 对 Qwen3 做 task-aware fine-tune（但加 HSIC 保护 final layer）

Phase 2 改动大，且 MARC 是 CTR 排序场景，不是生成式推荐 — 需要谨慎评估是否直接移植。我们的场景里 "Matching Network" 对应的是 NTP 模型本身，不是独立小网络。

**Phase 3 — 多层聚合 (IDEA-onerec-3 QFormer 的一个实例化)**:

如果 Phase 1 显示 mid > final，还可以尝试：
- 取中间层 hidden states + final layer hidden states, concat → linear projection → 送 MLP-FSQ
- 或用 learned attention over layers (mini-QFormer) 做加权平均

### 关键问题

1. **MRA 是否在无微调的 Qwen3 上出现？** 论文实验都是"微调后"的 LLM。我们的 Qwen3 是冻结的预训练模型，可能不存在 final-layer 退化问题。但即便如此，中间层对推荐任务可能更合适（final layer 被预训练时的 LM head 拉偏）
2. **Qwen3 的 EOS/last-token pooling 只在 final layer 有意义**: 如果取中间层，pooling 策略要同步改（mean pooling over non-pad tokens 更合适）
3. **HSIC 实现复杂度**: Phase 2 的 HSIC 约束需要计算核矩阵，在 large batch 上是 O(B²) 内存，与当前 tokenizer 训练的 multi-GPU 模式不直接兼容
4. **与 VL embedder 的一致性**: 如果文字 embedder 切到 mid-layer, 图像侧 `Qwen3VLEmbedder` 是否也应切？VL 的多模态融合层通常在中-后段，可能不能一致

### 相关 idea

- IDEA-sid-1 (CF 微调增强): ❌ 失败 — MRA 理论刚好解释了为什么 CF 微调 final layer 会退化
- IDEA-onerec-3 (QFormer Tokenizer): 多层聚合路径, Phase 3 是其简化版
- IDEA-forge-0 (Proxy Metrics): `semantic_neighbor_HR` 正是本实验的评估指标
- IDEA-snap-0 (Snapchat SIDs): Snapchat 的多模态 embedding 融合 + STE 处理 codebook collapse，和本 idea 都指向 "embedding 端对 SID 质量至关重要"
