# Experiment Log

按时间倒序记录。每次实验链接到 `experiments/` 下的结果目录。

---

## Template

<!--
复制以下模板创建新实验记录。编号递增，最新的放在最上面。

## EXP-NNN: (实验标题)

**Date**: YYYY-MM-DD
**Status**: planned | running | completed
**Results**: [./hyperparam/YYYY-MM-DD_xxx/](./hyperparam/YYYY-MM-DD_xxx/)

### Background
(当前状态、要解决的问题)

### Hypothesis
(预期结果及原因)

### Design
- **Variable**: ...
- **Fixed**: ...
- **Metric**: ...
- **Data**: ...

### Results
(跑完后填写，含表格)

### Analysis
(结果解读)

### Next Steps
(下一步计划)
-->

---

## EXP-011: Codebook Size 消融 — 等大 1024/4096 + OPQ 对照

**Date**: 2026-04-15
**Status**: planned
**Results**: TBD

### Background

EXP-010 NTP baseline 效果极差 (L1 acc=0.7%, item_recall@50=0.0008)，根因之一是当前 SID 配置 **L1=1024, L2=1024, L3=4096 不等大**，NTP 模型用全局 max=4027 作为统一 vocab。

查阅 OneMall 原文发现其生产配置是 **三层等大 4096×4096×4096**，FSQ 层使用 "binary 16-bit MLP"。需要确定我们的最优 codebook 配置。

### Hypothesis

1. 三层等大配置 (1024×3 或 4096×3) 的 semantic_neighbor_HR 不低于当前 1024×1024×4096
2. Binary FSQ ([2,...,2]) 与 multi-level FSQ ([4,...,4]) 在相同 codebook size 下效果相当
3. OPQ 3×N (等 token 数对照) 仍然输层级结构 MLP-FSQ（延续 EXP-008 结论）

### Design

| Config | L1 (KMeans) | L2 (KMeans) | L3 (FSQ) | FSQ Levels | Bits | 对标 |
|--------|-------------|-------------|----------|------------|------|------|
| A (EXP-008) | 1024 | 1024 | 4096 | [4,4,4,4,4,4] | 32 | 已有 baseline |
| E | 1024 | 1024 | 1024 | [4,4,4,4,4] | 30 | 等大 1024, multi-level |
| F | 1024 | 1024 | 1024 | [2]×10 | 30 | 等大 1024, binary |
| G | 4096 | 4096 | 4096 | [4,4,4,4,4,4] | 36 | OneMall 配置 |
| H | 4096 | 4096 | 4096 | [2]×12 | 36 | OneMall binary |
| I | OPQ 3×1024 | — | — | — | 30 | 等 bits 对照 E/F |
| J | OPQ 3×4096 | — | — | — | 36 | 等 bits 对照 G/H |

- **Fixed**: Qwen3-0.6B 1024D embedding (cached), behavior_data 7d, MLP hidden=64, 50 epochs
- **Metric**: semantic_neighbor_hit_rate (核心), collision_rate, cluster_balance (Gini)

### Run

`bash experiments/scripts/exp-011.sh`

### Results

TBD

### Analysis

TBD

### Next Steps

TBD

---

## EXP-010: NTP Baseline — MLP-FSQ SID 端到端 Recall

**Date**: 2026-04-15
**Status**: completed (效果极差，需诊断)
**Results**: [./hyperparam/2026-04-15_exp010-ntp-baseline/](./hyperparam/2026-04-15_exp010-ntp-baseline/)

### Background

Tokenizer 阶段结束，MLP-FSQ h=64 确认为赢家 (EXP-008, semantic_neighbor_HR=0.078)。现在需要第一个端到端 NTP 数字：用当前 2 层 Transformer probe (~5M params) 在 MLP-FSQ SID 上训练，拿到 item Recall@K baseline。

当前 NTP probe 参数:
- 2 层 causal Transformer decoder, embed_dim=256, n_heads=4, ffn_dim=512
- **1 epoch** (代码 bug: 缺少 epoch 外循环), AdamW lr=3e-3, CosineAnnealing
- 行为序列 n_items=10, beam_size=50
- SID: 3 tokens (L1=1024, L2=1024, **L3=4096** ← 与 L1/L2 不等大)

### Hypothesis

- Perplexity 应在 50~150 范围（good~acceptable）
- Item Recall@50 应显著高于 embedding_hit_rate (0.0047)，因为 NTP 利用了行为序列信息
- 这个数字作为所有后续 NTP 改进（architecture/training/scaling）的 baseline

### Design

- **Variable**: 无（单配置 baseline）
- **Fixed**: MLP-FSQ h=64, 2 层 probe, 1 epoch, n_items=10, beam_size=50
- **Metric**: Perplexity, Depth Accuracy, Item Recall@{10,50,100,500}
- **Data**: 7 天行为数据, 19.1M samples (train=15.3M, eval=50K)

### Run

`bash experiments/scripts/exp-010.sh`

### Results

| 指标 | 值 |
|------|-----|
| Train loss | 1.70 → 0.47 (3741 steps) |
| Eval perplexity | 5.34 |
| Depth acc beam (L1/L2/L3) | 0.007 / 0.000 / 0.000 |
| **Depth hit@10 (L1/L2/L3)** | **1.000 / 1.000 / 0.401** |
| Item recall@50 | 0.0008 |
| Item recall@500 | 0.0008 |

### Analysis

**效果极差，但 teacher-forced hit@10 表明模型学到了。核心问题在 beam search：**

1. **L1/L2/L3 不等大 vocab 共享单一 output head (Linear(256, 4027))**: L1/L2 只有 1024 个合法 token，但 softmax 在 4027 维上做，75% 的概率空间是噪声。Beam search 可能选到 L3 范围的 token 作为 L1 预测
2. **Teacher-forced hit@10 = 100%**: 说明模型在看到正确上下文时，正确 token 在 top-10 中。但 beam search 一旦 L1 选错，后续全部偏移
3. **只训练 1 epoch**: train loss 还在下降 (0.47 且斜率明显)，未收敛
4. **Train-eval gap 大**: train CE ≈ 0.47, eval CE ≈ 1.68 (PPL 5.34)，时间序列切分导致分布偏移

**根因: SID 配置 1024×1024×4096 不等大 + NTP 模型未做 per-layer vocab 处理。**

### Next Steps

1. **EXP-011**: 确定正确的 codebook 配置 (等大 1024×3 或 4096×3)
2. **修复 NTP 模型**: per-layer output head 或统一 vocab + layer embedding + beam search mask
3. **增加 epoch**: 1 → 5-10
4. 修复后重跑 NTP baseline

---

## EXP-009: QFormer Tokenizer — 冻结 Qwen3 + Cross-Attention 压缩

**Date**: 2026-04-14 ~ 2026-04-15
**Status**: completed
**IDEA**: IDEA-onerec-3
**Results**: [./hyperparam/2026-04-14_exp009-qformer/](./hyperparam/2026-04-14_exp009-qformer/)

### Background

EXP-007 证明直接 fine-tune Qwen3-0.6B（全量/LoRA，多种 lr/τ）完全无法推动模型——cap_loss 纹丝不动，HR@50 卡在 ~0.02。根本原因: I2I 梯度稀释在 600M 参数中。

OneRec 的核心方案: 冻结底座，在上面加一个可训练的 QFormer (cross-attention + learnable queries)。梯度集中在 ~30-50M 参数的 QFormer 上，底座天然保持语义。BLIP-2 QFormer 已被 OneRec (miniCPM-V-8B + 4-layer QFormer) 验证有效。

### Hypothesis

1. QFormer 训练时 cap_loss 会明显下降（不同于 EXP-007 的纹丝不动），证明梯度可以有效流动
2. HR@50 显著突破 EXP-007 的 0.02 baseline（预期 > 0.05）
3. 信息压缩 (S tokens → M tokens) 迫使 QFormer 学会提取协同相关信息而非照搬语义

### Design

**Phase 1 — 最小验证 (梯度能否流动)**:

| Config | QFormer Layers | Query Tokens (M) | lr | Loss |
|--------|---------------|-------------------|------|------|
| A | 2 | 4 | 1e-4 | L_I2I only |
| B | 2 | 4 | 5e-4 | L_I2I only |
| C | 4 | 4 | 1e-4 | L_I2I only |

- **Variable**: QFormer depth × learning rate
- **Fixed**: Qwen3-0.6B frozen, M=4 query tokens, D=1024, τ=0.05, batch_size=32, grad_accum=8, max_pairs=500K, 1 epoch, 8xA100 DDP
- **Metric**:
  - **Primary**: HR@50 (InlineHRMonitor, 与 EXP-007 baseline 直接对比)
  - **Diagnostic**: cap_loss 变化量 (W&B)、I2I loss 收敛速度
  - **Secondary**: OPQ intrinsic (collision, recon_loss) on QFormer embeddings
- **Data**: 行为数据 7 天, ~5M items

### Run
`bash experiments/scripts/exp-009.sh`

### Results

| Config | QFormer Layers | Queries (M) | lr | Final HR@50 | Final Loss | 训练时间 |
|--------|---------------|-------------|------|------------|-----------|---------|
| BL (raw Qwen3) | — | — | — | 0.0106 | — | — |
| EXP-007 best (全量FT) | — | — | 1e-5 | 0.0197 | 2.90 | 6756s |
| A | 2 | 4 | 1e-4 | 0.0211 | 4.46 | 4460s |
| B | 2 | 4 | 5e-4 | 0.0214 | 4.41 | 4458s |
| **C (best)** | **4** | **4** | **1e-4** | **0.0216** | **4.42** | **4549s** |

实际训练数据: 3,074,342 pairs (max_pairs=5M, swing 实际产出 ~3M), 12,000 steps/epoch, effective batch 2048。

### Analysis

**1. QFormer 未突破 0.02 天花板:**
- 最佳 Config C: HR@50 = 0.0216，仅比 EXP-007 best (0.0197) 高 10%，远未达到 hypothesis 预期的 >0.05
- 三组 config 差异极小 (0.0211 ~ 0.0216)，QFormer depth/lr 不是瓶颈

**2. Hypothesis 验证:**
- ✅ H1 (梯度流动): loss 从 5.5 降到 ~4.4，确实在下降（EXP-007 cap_loss 纹丝不动），证明梯度可以流过 QFormer
- ❌ H2 (HR@50 突破): 0.0216 vs 预期 >0.05，差距巨大
- ❌ H3 (信息压缩): QFormer 的 4 query tokens 并未迫使模型学到更好的协同表示

**3. HR@50 曲线特征:**
- 全程缓慢单调上升，未见明显 plateau
- 但斜率持续递减 (step 0~4000: +0.006, step 4000~8000: +0.003, step 8000~12000: +0.002)
- 更多 epoch 可能有微小提升，但趋势已极平，不可能突破 0.03

**4. 根因重新判断:**
- EXP-007 结论 "梯度稀释" 被部分推翻——QFormer 集中梯度后 loss 确实在降，但 HR@50 仍卡住
- **真正瓶颈不在模型结构，而在 I2I contrastive 信号本身**: in-batch negatives + 行为共现正样本的监督信号强度不足以将 embedding 推到行为空间中有意义的位置
- 或者说: **Qwen3 的 semantic embedding 空间与行为空间的 gap 远大于 contrastive learning 能弥补的程度**

### Next Steps

EXP-007 + EXP-009 两轮实验证明: **I2I contrastive fine-tune (无论全量/LoRA/QFormer) 都无法有效改善 embedding 的行为质量**。需要重新审视 embedding 端的策略:

1. **放弃 embedding fine-tune 路线**, 回归 "好的 tokenizer 比好的 embedding 更重要" 的架构哲学
2. 聚焦 **EXP-008 (FORGE proxy 对比)** — 用现有 Qwen3 embedding 对比 MLP-FSQ vs OPQ 的行为质量，决定 tokenizer 路线
3. 如果仍需改善 embedding，考虑完全不同的方案: multi-task learning、graph embedding、或直接用行为 embedding (collaborative filtering) 替代文本 embedding

---

## EXP-008: FORGE Proxy 对比 — MLP-FSQ vs OPQ 最优解

**Date**: 2026-04-14 ~ 2026-04-15
**Status**: completed
**Results**: [./hyperparam/2026-04-15_exp008-mlpfsq-h64/](./hyperparam/2026-04-15_exp008-mlpfsq-h64/), [./hyperparam/2026-04-15_exp008-opq-m4/](./hyperparam/2026-04-15_exp008-opq-m4/), [./hyperparam/2026-04-15_exp008-opq-m8/](./hyperparam/2026-04-15_exp008-opq-m8/)

### Background

EXP-003 最优 (MLP-FSQ h=64, collision=0.041) 和 EXP-004 最优 (OPQ 8×256, collision=0.0037) 只有 intrinsic metrics，缺行为层面验证。已实现的 FORGE proxy metrics 无需训练 NTP 就能评估 SID 质量：
- `embedding_hit_rate`: embedding I2I 邻居共现率（所有方案相同，作为 baseline）
- `semantic_neighbor_hit_rate`: SID 前缀邻居共现率（区分 tokenizer，核心指标）

目标：快速对比两条路线，决定哪条进入 NTP 阶段。

### Hypothesis

1. OPQ 8×256 的 `semantic_neighbor_hit_rate` 应显著高于 MLP-FSQ h=64，因为更低的 collision (0.0037 vs 0.041) 意味着更精细的 SID 分区
2. `embedding_hit_rate` 三组相同（相同 embedding，只是 tokenizer 不同）
3. OPQ 4×256 (等 bits 对照) 的 `semantic_neighbor_hit_rate` 介于 MLP-FSQ 和 OPQ 8×256 之间

### Design

| Config | Tokenizer | Tokens | Bits | 已知 collision |
|--------|-----------|--------|------|---------------|
| A | MLP-FSQ h=64 (6d_4096) | 3 | 32 | 0.0411 |
| B | OPQ 4×256 (等 bits 对照) | 4 | 32 | 0.1063 |
| C | OPQ 8×256 (最优) | 8 | 64 | 0.0037 |

- **Fixed**: Qwen3-0.6B 1024D embedding (cached), behavior_data 7d
- **Metric**:
  - **Primary**: `semantic_neighbor_hit_rate` — SID 前缀邻居在行为图中的共现率
  - **Baseline**: `embedding_hit_rate` — embedding 空间 I2I 邻居共现率（三组应相同）
  - **Secondary**: intrinsic metrics (collision, recon_loss, entropy)

### Run
`bash experiments/scripts/exp-008.sh`

### Results

数据: 554,754 exposed items (从 5,162,650 总 embedding 中过滤), 行为数据 7 天 (03-24 ~ 03-30)

| Config | Tokenizer | Tokens | Bits | collision | recon_loss | embedding_HR | **semantic_neighbor_HR** | 训练时间 |
|--------|-----------|--------|------|-----------|------------|-------------|------------------------|---------|
| **A** | **MLP-FSQ h=64** | **3** | **32** | **0.1074** | **0.3668** | **0.0047** | **0.0780** | 106s |
| B | OPQ 4×256 | 4 | 32 | 0.0351 | 0.3760 | 0.0047 | 0.0502 | 73s |
| C | OPQ 8×256 | 8 | 64 | 0.0006 | 0.3408 | 0.0043 | 0.0326 | 99s |

### Analysis

**结果与 hypothesis 完全相反 — MLP-FSQ 大幅领先 OPQ:**

**1. Hypothesis 验证:**
- ❌ H1: OPQ 8×256 的 semantic_neighbor_HR (0.033) **远低于** MLP-FSQ (0.078)，collision 低 180 倍却输了 58%
- ✅ H2: embedding_HR 三组几乎相同 (~0.0047)，符合预期
- ❌ H3: OPQ 4×256 (0.050) 介于两者之间，但方向反了——不是 MLP-FSQ < OPQ 4×256 < OPQ 8×256，而是 MLP-FSQ > OPQ 4×256 > OPQ 8×256

**2. collision 越低 ≠ 行为质量越好:**
- OPQ 8×256 追求极低 collision (0.06%)，将 embedding 空间切成 ~553K 个几乎不重叠的 bin
- 但过度细分破坏了语义邻域结构——SID 前缀相近的 item 不再是行为上的邻居
- MLP-FSQ 的 collision 10.7% 看似"差"，但保留了层级聚集结构，SID 前缀邻居的行为共现率反而更高

**3. 层级结构 > 扁平结构:**
- MLP-FSQ: 3 层层级 (KMeans → KMeans → FSQ)，每层逐步细化，前缀天然编码粗到细的语义聚类
- OPQ: 8 个并行子向量独立量化，token 间无层级关系，前缀邻居不具有语义含义

**4. 等 bits 对照 (32 bits):**
- MLP-FSQ (0.078) vs OPQ 4×256 (0.050)，MLP-FSQ 赢 56%
- 相同信息量下，层级残差编码的 SID 前缀邻域比并行 PQ 的前缀邻域更有行为意义

**5. 注意: MLP-FSQ 不使用行为数据训练:**
- MLP 仅优化残差重建 loss (||residual - Decoder(FSQ(Encoder(residual)))||²)，纯无监督
- 行为质量的优势完全来自层级结构对 embedding 邻域的保持，而非学习行为信号

### Next Steps

**MLP-FSQ h=64 确认为 tokenizer 路线赢家**，进入 NTP 阶段:
1. 用 MLP-FSQ 生成全量 SID，训练 NTP 预测模型
2. 端到端评估 Recall@K
3. 考虑是否需要更大的 FSQ codebook (当前 4096) 或更多 KMeans 层

---

## EXP-007: Collaborative Signal Enhanced Embedding (Qwen3-0.6B Full Fine-tune)

**Date**: 2026-04-13 ~ 2026-04-14
**Status**: completed
**IDEA**: IDEA-sid-1
**Results**: [./hyperparam/2026-04-13_exp007-collab-embed/](./hyperparam/2026-04-13_exp007-collab-embed/)

### Background

当前直接用 Qwen3-0.6B 纯文本 embedding (1024D) 做量化。这些 embedding 只编码了语义相似性（文本内容相近的 item 距离近），但推荐需要的是**行为相似性**（被同一用户群喜欢的 item 距离近）。EXP-004 的 embedding_hit_rate 指标可以量化当前 embedding 在行为维度的质量。

本实验通过 **I2I 对比学习** 全量 fine-tune Qwen3-0.6B，将协同行为信号注入 embedding，提升量化上限。与量化方案 (OPQ/RKMeans) 正交，改善 embedding 质量对所有下游实验受益。

### Hypothesis

1. 对比学习后的 embedding 在 `embedding_hit_rate` 上显著优于原始 Qwen3 embedding（预期 HR@50 提升 50%+）
2. 下游 OPQ 量化指标（collision, recon_loss）也会改善，因为行为相似的 item 在 embedding 空间更聚集
3. 全量 fine-tune 0.6B 在 8xA100 上训练时间可控（预期 < 4 小时）

### Design

- **Variable**: 训练方案 × 温度参数
  - **Baseline**: 原始 Qwen3-0.6B embedding（已缓存，无需重跑）
  - **Config A**: 全量 fine-tune, InfoNCE, τ=0.05, 3 epochs
  - **Config B**: 全量 fine-tune, InfoNCE, τ=0.07, 3 epochs
  - **Config C**: 全量 fine-tune, InfoNCE, τ=0.05, 5 epochs
- **Fixed**:
  - 模型: Qwen3-0.6B (全量参数更新, FP16, 8xA100 DDP)
  - 正样本: 同一用户 7 天内正向行为 (action_bitmap > 0) 的 item pair
  - 负样本: in-batch negatives (batch_size=512 per GPU, effective 4096)
  - Optimizer: AdamW, lr=1e-5, warmup 10%, cosine decay
  - 文本: item title (已有 Qwen3 tokenizer)
- **Metric**:
  - **Primary**: `embedding_hit_rate` (HR@10/50/100/500) — FORGE proxy，不需要训练 NTP
  - **Secondary**: OPQ intrinsic (collision, recon_loss, entropy) — 用 EXP-004 相同 OPQ config (m=8, M=256) 量化后评估
  - **Sanity**: `cosine_similarity` 分布, `embedding_behavior_correlation`
- **Data**: 行为数据 7 天 (2026-03-24 ~ 2026-03-31), ~5M items

### Run
`bash experiments/scripts/exp-007.sh`

### Results

**Baseline**: HR@50 = 0.0106 (原始 Qwen3-0.6B embedding, 50,008 items)

**Round 1 — 基础超参搜索 (全量 fine-tune)**:

| Config | τ | lr | max_pairs | HR@50 | Loss plateau | 训练时间 |
|--------|------|------|-----------|-------|-------------|---------|
| BL (baseline) | — | — | — | **0.0106** | — | — |
| A | 0.05 | 1e-5 | 2M | **0.0197** | ~step 800 | 6756s (~1h53m) |
| B | 0.07 | 1e-5 | 1M | 0.0148 | ~step 800 | killed early |
| C | 0.05 | 3e-5 | 500K | 0.0192 | ~step 400 | 1912s (~32min) |

**Round 2 — 激进学习率 (cap_loss 在 R1 纹丝不动)**:

| Config | τ | lr | 状态 |
|--------|------|------|------|
| D | 0.05 | 1e-4 | 脚本就绪，未产出超越 R1 的结果 |
| E | 0.05 | 3e-4 | 同上 |
| F | 0.05 | 1e-3 | 同上 |

**Round 3 — LoRA (冻结底座，梯度集中在 adapter)**:

| Config | Method | lr | 状态 |
|--------|--------|------|------|
| G | LoRA r=16 | 1e-4 | 脚本就绪，未产出超越 R1 的结果 |
| H | LoRA r=16 | 5e-4 | 同上 |
| I | LoRA r=64 | 1e-4 | 同上 |

### Analysis

**1. HR@50 天花板 ~0.02，较 baseline 0.0106 提升约 86%，但远未达到 hypothesis 预期的 50%+ 绝对提升:**
- 最佳 Config A: 0.0197，仍处于 poor 级别（阈值 < 0.02）
- 三组 round 1 config HR@50 收敛到同一天花板 (~0.02)，超参调优空间有限

**2. 温度不是瓶颈**: τ=0.07 (Config B) 全面劣于 τ=0.05 (Config A)

**3. 学习率影响收敛速度不影响上限**: Config C (lr=3e-5) 用 1/4 数据、1/3 时间达到同等效果

**4. Loss 快速 plateau**: 所有 config 在 ~200K pairs 后 loss 稳定在 ~2.5-2.7，cap_loss 完全不动——说明 I2I 梯度稀释在 600M 参数中

**5. Hypothesis 验证:**
- ❌ HR@50 提升 86% (0.0106→0.0197)，但绝对值仍极低，未达到 "显著优于" 的预期
- ❌ 下游量化改善未验证（HR@50 本身太低，OPQ 评估意义有限）
- ✅ 训练时间可控（最快 Config C 仅 32 分钟）

**6. 根因**: 直接 fine-tune 600M 参数的 Qwen3 底座，I2I contrastive 的梯度被稀释，模型几乎不学习。无论全量 fine-tune 还是 LoRA，都无法有效将协同信号注入 embedding。

### Next Steps

EXP-007 证明 "直接 fine-tune 底座" 路线不可行，需要方法论变更:
- **EXP-009 (已规划)**: 冻结 Qwen3 底座 + QFormer cross-attention，梯度集中在 ~30-50M 参数的 QFormer 上（OneRec 验证有效的方案）

---

## EXP-004: OPQ Parallel Semantic IDs — Intrinsic Metrics

**Date**: 2026-04-13
**Status**: completed
**IDEA**: IDEA-sid-0 (Phase 1)
**Reference**: Meta RPG (KDD'25, arxiv 2506.05781)
**Results**: [./hyperparam/2026-04-13_exp004-opq/](./hyperparam/2026-04-13_exp004-opq/), [./hyperparam/2026-04-13_exp004-opq-m4/](./hyperparam/2026-04-13_exp004-opq-m4/)

### Background
当前 RKMeans (3 层 x 1024 clusters) 使用残差编码，各层串行依赖。ARCHITECTURE.md 已明确需要切换到并行 tokenizer。RPG 论文证明 OPQ (Optimized Product Quantization) 在生成式推荐中优于 RQ，且支持并行预测。

本实验验证 OPQ 在我们 5M item / 1024D Qwen3-0.6b embedding 上的量化质量（intrinsic metrics），不涉及 NTP 预测模型。

### Hypothesis
OPQ 将 1024D embedding 切分为 m 个独立子向量分别量化，编码空间远大于 RKMeans (256^8 >> 1024^3)，collision 应显著更低。recon_loss 需要实测验证 — PQ 的独立子空间假设可能不如 RQ 的残差逼近。

### Design
- **Variable**: n_subvectors (m=4, 8, 16, 32), n_clusters_per_sub (M=256)
- **Fixed**: normalize_input=True, OPQ rotation training (FAISS default)
- **Metric**: collision_rate, exclusivity, reconstruction_loss, entropy, cluster_balance
- **Data**: 5M items, qwen3-0.6b 1024D embedding (cached)

**Comparison matrix**:

| Config | Quantizer | Tokens | Vocab/token | Bits | 子向量维度 |
|--------|-----------|--------|-------------|------|-----------|
| Baseline (EXP-001) | RKMeans 3x1024 | 3 | 1024 | 30 | N/A (residual) |
| **OPQ-4x256** | **OPQ** | **4** | **256** | **32** | **256D (等 bits 对照)** |
| OPQ-8x256 | OPQ | 8 | 256 | 64 | 128D |
| OPQ-16x256 | OPQ | 16 | 256 | 128 | 64D |
| OPQ-32x256 | OPQ | 32 | 256 | 256 | 32D |

### Run
`bash experiments/scripts/exp-004.sh`

### Results

| Config | Tokens | Bits | collision | entropy | Gini | recon_loss | time(s) |
|--------|--------|------|-----------|---------|------|------------|---------|
| **RKMeans 3×1024** (EXP-001) | **3** | **30** | **0.1634** | **0.7211** | **0.2091** | **0.3524** | — |
| **OPQ 4×256** | **4** | **32** | **0.1063** | **0.9681** | **0.1896** | **0.3772** | 125 |
| OPQ 8×256 | 8 | 64 | 0.0037 | 0.9971 | 0.0128 | 0.3429 | 160 |
| OPQ 16×256 | 16 | 128 | 0.0029 | 0.9993 | 0.0052 | 0.3026 | 220 |
| OPQ 32×256 | 32 | 256 | 0.0027 | 0.9995 | 0.0043 | 0.2522 | 338 |

### Analysis

**1. 等 bits 对照 — OPQ 4×256 (32bit) vs RKMeans 3×1024 (30bit):**
- collision: 10.6% vs 16.3% — OPQ 低 35%，相同信息量下碰撞率显著更低
- entropy: 0.968 vs 0.721 — OPQ codebook 利用率远更均匀
- recon_loss: 0.377 vs 0.352 — OPQ 略差 7%，PQ 子空间独立假设的代价
- 结论：等 bits 下 OPQ **赢 collision、输 recon**，trade-off 合理

**2. m=8 是 sweet spot:**
- collision 从 m=4 的 10.6% 骤降到 0.37%（仅多 1 倍 bits）
- recon_loss 0.3429 已优于 RKMeans 0.3524
- 8 token 并行预测成本可控

**3. m≥16 收益递减:**
- collision: 0.37% → 0.29% → 0.27%，几乎无差异
- recon_loss 持续下降但 token 数翻倍 → NTP 预测成本翻倍
- 除非下游任务对 recon 极度敏感，否则不值得

**4. Hypothesis 验证:**
- ✅ collision 显著更低（符合预期，编码空间 256^m >> 1024^3）
- ✅ recon_loss 在 m≥8 时优于 RKMeans（PQ 独立子空间假设没有严重损害重建质量）
- ✅ entropy/Gini 近乎完美，无 cluster collapse

### Next Steps
OPQ Phase 1 验证通过，推荐 **m=8** 进入 Phase 2:
1. 并行预测 NTP 模型 — per-digit independent MLP heads + MTP loss
2. Graph-Constrained Decoding — 替代 beam search（RPG 论文证明 beam search 在 OPQ 上 recall=0.0000）
3. 端到端评估 — Recall@K on downstream retrieval task

---

## EXP-003: Learned FSQ — MLP projection + straight-through training

**Date**: 2026-04-13
**Status**: completed
**Results**: [./hyperparam/2026-04-13_exp003-mlp64/](./hyperparam/2026-04-13_exp003-mlp64/), [./hyperparam/2026-04-13_exp003-mlp128/](./hyperparam/2026-04-13_exp003-mlp128/)

### Background
EXP-002 证明 PCA 线性投影 + FSQ 劣于 KMeans baseline，核心瓶颈是 PCA 在残差空间信息丢失过大（1024D→4~6D 解释方差仅 20-55%）。

OneMall (arxiv 2601.21770) 用 **learned MLP** 做投影，原始 FSQ 论文 (Mentzer 2023, arxiv 2309.15505) 将 FSQ 用在 VQ-VAE 内部，encoder 学到对量化最优的表示。关键机制：
- MLP 学习非线性投影 D→d，比 PCA 保留更多量化相关信息
- Straight-Through Estimator (STE): 前向用 round()，反向把梯度直通到 MLP 参数
- 训练目标: 重建 loss — minimize ||residual - reconstruct(FSQ(MLP(residual)))||²

### Hypothesis
Learned MLP 投影可以学到对 FSQ 量化最优的低维表示，使 FSQ 的 reconstruction quality 接近或超过 KMeans，从而在保持低 collision 的同时降低 recon_loss。

### Design
- **Variable**: 投影方式 (PCA vs MLP)，MLP 隐层宽度
- **Fixed**: 2 KMeans layers x 1024 clusters, FSQ [4,4,4,4,4,4] (6d_4096), epochs=50, lr=1e-3, AdamW
- **Metric**: collision_rate, reconstruction_loss, exclusivity, entropy

**MLP 架构** (autoencoder + STE):
```
Encoder: D → hidden → d  (d=6 for 6d_4096)
FSQ:     quantize each of 6 dims to {0,1,2,3}, STE pass-through
Decoder: d → hidden → D
Loss:    ||residual - Decoder(STE_quantize(Encoder(residual)))||²
```

**Comparison matrix**:

| Config | L3 projection | L3 codebook | Training |
|--------|---------------|-------------|----------|
| Baseline (EXP-002) | KMeans 1024 | 1024 | N/A |
| PCA-FSQ (EXP-002) | PCA 6d | 4096 | N/A |
| MLP-FSQ-64 | MLP D→64→6 | 4096 | 50 epochs |
| MLP-FSQ-128 | MLP D→128→6 | 4096 | 50 epochs |

### Run
`bash experiments/scripts/exp-003.sh`

### Results

| Config | L3 | collision | recon_loss | d3 avg_items | time(s) |
|--------|-----|-----------|------------|--------------|---------|
| **Baseline** | KMeans 1024 | 0.1634 | 0.3524 | 1.3 | 237 |
| PCA-FSQ | PCA 6d_4096 | 0.3330 | 3.1280 | 1.7 | 178 |
| **MLP-FSQ h=64** | **MLP D→64→6** | **0.0411** | **0.3619** | **1.1** | **611** |
| MLP-FSQ h=128 | MLP D→128→6 | 0.0767 | 0.3633 | 1.1 | 627 |

### Analysis

**Learned MLP 大幅超越 KMeans baseline，完全验证 hypothesis:**

1. **collision 降 75%**: MLP h=64 的 collision 0.0411 vs KMeans baseline 0.1634，FSQ 的 implicit codebook (4096 codes) + 学到的非线性投影彻底解决了碰撞问题
2. **recon_loss 与 baseline 持平**: 0.3619 vs 0.3524 (差 2.7%)，说明 MLP 学到了高质量投影，PCA 的 3.128 recon_loss 完全是线性投影的局限性
3. **h=64 优于 h=128**: collision 0.0411 vs 0.0767。更小的 hidden dim 起到正则化作用，避免 encoder 输出过于极端导致 tanh 饱和（训练中发现了 tanh 饱和导致 OOB 的 bug 并修复）
4. **训练时间翻倍但可接受**: 611s vs 237s，多出的 ~400s 是 50 epoch MLP 训练，模型仅 ~132K params

**vs PCA-FSQ (EXP-002)**: collision 从 0.333 降到 0.041 (降 88%)，recon_loss 从 3.128 降到 0.362 (降 88%)，证明非线性投影是关键。

### Next Steps
1. MLP-FSQ h=64 跑 NTP behavior 评估，确认 recall@K 指标
2. 与 OPQ (EXP-004) 在 NTP 下游对比，决定最终方案

---

## EXP-002: ResKmeansFSQ — 2 layers RKMeans + 1 layer FSQ (PCA projection)

**Date**: 2026-04-13
**Status**: completed
**Results**: [./hyperparam/2026-04-13_exp002-baseline/](./hyperparam/2026-04-13_exp002-baseline/), [./hyperparam/2026-04-13_exp002-fsq/](./hyperparam/2026-04-13_exp002-fsq/)

### Background
RKMeans 的第3层对残差做 KMeans 效果递减。OneMall (arxiv 2601.21770) 提出用 FSQ 替换第3层。本实验使用 **PCA 线性投影** 替代论文中的 learned MLP 做降维。

### Hypothesis
FSQ 的 implicit codebook 天然无 cluster collapse，可降低 collision rate。

### Design
- **Variable**: Layer 3 quantizer (KMeans vs FSQ configs)
- **Fixed**: 2 KMeans layers x 1024 clusters, niter=25, nredo=3, normalize_residuals=True
- **Metric**: conflict_rate, reconstruction_loss, entropy, exclusivity, cluster_balance

| Config | L1, L2 (KMeans) | L3 | L3 codebook |
|--------|------------------|----|-------------|
| Baseline | 1024 x 3 layers KMeans | KMeans 1024 | 1024 |
| Hybrid A | 1024 x 2 layers | FSQ [8,8,8,8] | 4096 |
| Hybrid B | 1024 x 2 layers | FSQ [7,5,5,5,5] | 4375 |
| Hybrid C | 1024 x 2 layers | FSQ [4,4,4,4,4,4] | 4096 |

### Run
`bash experiments/scripts/exp-002.sh`

### Results

| Config | L3 | conflict_rate | exclusivity | recon_loss | d3 entropy | d3 Gini | d3 unique | d3 avg_items |
|--------|-----|---------------|-------------|------------|------------|---------|-----------|--------------|
| **Baseline** | KMeans 1024 | **0.1634** | **0.6423** | **0.3524** | **0.7211** | **0.2091** | 3,963,269 | 1.3 |
| Hybrid C | FSQ 6d [4x6] | 0.3330 | 0.4015 | 3.1280 | 0.6755 | 0.3153 | 3,107,671 | 1.7 |
| Hybrid A | FSQ 4d [8x4] | 0.5688 | 0.1446 | 2.2122 | 0.6383 | 0.4693 | 1,731,222 | 3.0 |
| Hybrid B | FSQ 5d [7,5x4] | 0.8157 | 0.0089 | 0.3800 | 0.5306 | 0.6548 | 248,798 | 20.8 |

注: L1/L2 两层 KMeans 相同 (d1/d2 指标一致)，差异全部来自 L3。

### Analysis

**FSQ+PCA 全面劣于 KMeans baseline**，核心原因是 **PCA 线性投影信息丢失过大**：

1. **投影瓶颈**: 1024维残差 → 4~6维 PCA，解释方差仅 20-55%。残差空间（经两轮 KMeans 后）本就小且不规则，PCA 线性假设不适用。
2. **维度越少越差**: 5d_4375 (d=5) 的 conflict_rate 高达 0.82，几乎所有信息丢失；6d_4096 (d=6) 最好但仍 0.33 >> baseline 0.16。
3. **recon_loss 恶化**: 4d/6d 的 recon_loss 从 0.35 飙升到 2.2/3.1，说明 PCA 逆投影无法恢复原始残差。
4. **与论文差异**: OneMall 用 **learned MLP** 投影（非线性、端到端训练），可学到对量化最优的表示，而非仅保留方差最大方向。

### Next Steps
EXP-003: 将 PCA 替换为 learned MLP 投影，复现论文方案。需要：
1. 定义 MLP 架构 (D → d 维) + VQ-VAE style 重建 loss
2. 端到端训练投影网络
3. 对比 PCA vs MLP 在同一 FSQ config 下的效果

---

## EXP-001: RKMeans 训练优化 (v0→v7)

**Date**: 2026-03 ~ 2026-04
**Status**: completed
**Results**: See `config/RKMEANS_EXPERIMENT_LOG.md` for full details

### Background
RKMeans 生成 semantic_id 碰撞率极高（99%+），需要系统性优化。

### Key Findings
1. **normalize_residuals 只对 layer 0 输入做** — 残差保留原始 scale，否则 Layer 2/3 无法聚类
2. **FAISS full-batch Lloyd's 优于 SGD/MiniBatch** — 空 cluster rebalance + GPU 加速
3. **num_clusters 是唯一显著超参** — collision 与 clusters 呈 log-linear 关系，每翻倍降 50-70%
4. **nredo=3 足够，niter=25 已收敛** — nredo 1→3 关键 (-42~49%), 3→5 无意义; niter 25/50/100 无差异

### Final Config
- 3 layers × 1024 clusters, niter=25, nredo=3
- collision: 1.75%, reconstruction_loss: 0.348

---
