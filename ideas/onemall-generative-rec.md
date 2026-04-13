# OneMall: End-to-End Generative Recommender

**来源**: OneMall (arxiv 2601.21770v2) — Kuaishou E-Commerce
**日期**: 2026-04-13

---

## IDEA-010: In-Batch Contrastive Auxiliary Loss for NTP Model

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
- **与 IDEA-002 (协同信号 embedding) 正交**: IDEA-002 改善 embedding 本身，本 IDEA 改善 NTP 模型训练

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

## IDEA-011: Query-Former 长序列压缩

**优先级**: P1
**来源**: OneMall §3.2 Query Transformers
**状态**: 待讨论

### 核心思想

用 Query-Former (cross-attention with learnable query tokens) 将长用户行为序列压缩为固定数量的连续表示。OneMall 将 1205 token 压缩到 160 token (M=10 query tokens per behavior type)，FLOP 从 34.4 GFLOPs 降到 9.2 GFLOPs（**3.7x 减少**），性能仅损失 0.3-0.5% HR。

核心组件:
- learnable query tokens Q ∈ ℝ^(M×D)
- cross-attention: F = CrossAttn(Q, H_seq, H_seq)
- 每种行为序列 (click, buy, exposure) 各自一个 Query-Former

### 与当前项目的关联

- 当前 NTP 模型 (`metrics/sid_prediction.py`) 直接将 SID 序列作为输入，sequence length 受限
- ARCHITECTURE.md 计划的 M-tier/L-tier 模型需要处理更长序列
- Query-Former 是 ARCHITECTURE.md 提到的 "Context Processor (OneRec-V2 lazy decoder-only)" 的具体实现方案之一
- 实现上可参考 BLIP-2 的 Q-Former，但 OneMall 版本更简洁（纯 cross-attention，无自回归）

### 实验设计草案

**新增模块 `model/query_former.py`**:
```python
class QueryFormer(nn.Module):
    # learnable queries: (M, D)
    # N layers cross-attention
    # input: (batch, seq_len, D) → output: (batch, M, D)
```

**集成到 NTP 模型**:
- 用户序列 → QueryFormer → 压缩表示 → concat SID tokens → Decoder

**变量**:
- M (query tokens): {4, 8, 16}
- QueryFormer layers: {1, 2}
- 输入序列长度: {50, 100, 200, 500}

**基线**: 直接截断序列 (当前方案)

**评估**: Recall@K vs 序列长度, 训练/推理时间

### 关键问题

1. 当前数据中用户序列有多长？如果平均序列较短，Query-Former 收益可能不大
2. 多种行为序列 (click/buy/exposure) 需要行为数据导出支持，当前 `data/export_behavior.py` 是否已覆盖
3. Query-Former 的预训练策略: 是否需要先单独预训练再接入 NTP 模型

---

## IDEA-012: GRPO/DPO 强化学习对齐

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
4. **建议先完成 IDEA-010 (contrastive loss)，建立更强的 NTP 基线后再做 RL**

---

## IDEA-013: Tokenizer Auxiliary Contrastive Loss (属性增强)

**优先级**: P1
**来源**: OneMall §4.5 Component Analyses (Aux Loss row)
**状态**: 待讨论

### 核心思想

在 tokenizer 的 embedding backbone 训练中，加入 item 属性 (category, price, shop) 作为辅助信号。OneMall 将这些属性 feed 进 item tower，用对比学习 loss 训练，在 HR@50/100/500 上分别提升 +1.5%/+1.7%/+1.7%。

这与 IDEA-002 (协同信号增强 embedding) 互补:
- IDEA-002: 用用户行为 I2I 对注入协同信号
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
- 与 IDEA-002 合并: I2I 协同信号 + 属性信号同时注入

**评估**: 原始 Qwen3 embed vs 属性增强 embed → 同一 RKMeans config → collision / exclusivity / behavior metrics

### 关键问题

1. 需要确认数据中有哪些可用的 item 属性字段
2. category 层级结构 (一级/二级/三级分类) 如何编码
3. 连续属性 (price) 的离散化/归一化策略

---

## IDEA-014: Loss-Free MoE Load Balancing

**优先级**: P2
**来源**: OneMall §3.2 Decoder-Style Sparse MoE (引用 loss-free mechanism)
**状态**: 待讨论

### 核心思想

替换当前 Switch Transformer 的 auxiliary loss 做 MoE load balancing，改用 loss-free 机制。Loss-free balancing 通过动态调整 router 的 expert bias 来实现均衡，不需要额外 loss term 干扰主任务梯度。

核心思路 (来自 DeepSeek 系列):
- 每个 expert 维护一个 bias term b_i
- 如果 expert i 负载过高 → 降低 b_i → router 倾向选择其他 expert
- 如果 expert i 负载过低 → 提高 b_i → router 倾向选择该 expert
- bias 更新不参与梯度计算，完全基于统计量

### 与当前项目的关联

- 当前 MoE 实现在 `metrics/sid_prediction.py:SparseMoEBlock` (lines 69-143)
- 使用 `0.01 * aux_loss` (Switch Transformer style: `n_experts * sum(f_i * P_i)`)
- 替换为 loss-free 只需修改 router 逻辑，不影响整体架构
- **实现成本极低，风险极小**

### 实验设计草案

**修改 `SparseMoEBlock`**:
1. 为每个 expert 添加 `expert_bias`: nn.Parameter(zeros(n_experts), requires_grad=False)
2. router score = linear(x) + expert_bias
3. 每个 training step 后: 统计各 expert 被选中的频率 f_i
4. bias 更新: `expert_bias[i] -= lr_bias * (f_i - 1/n_experts)`
5. 移除 aux_loss

**变量**:
- lr_bias ∈ {0.001, 0.01, 0.1}
- 更新频率: 每 step / 每 N steps

**评估**: expert 利用率分布, NTP perplexity, Recall@K

### 关键问题

1. S-tier 只有 8 experts，负载不平衡问题可能不严重 — 收益可能有限
2. loss-free 在 expert 数量少时是否稳定
3. 可以与 IDEA-010 (contrastive loss) 同时实验，因为修改正交

---

## IDEA-015: OneMall 验证 EXP-003 方向 (ResKmeans + Learned FSQ)

**优先级**: P0 (已有实验计划，需加速执行)
**来源**: OneMall §3.1.3 + §4.5 Tokenizer Strategy
**状态**: 待讨论 → 应立即推进 EXP-003

### 核心思想

OneMall 的 tokenizer ablation 直接验证了我们的 EXP-003 方向:

| 方案 | Conflict Rate | Exclusive Rate | HR@50 |
|------|--------------|----------------|-------|
| 3-layer ResKmeans | 36% | 86% | 33.9% |
| 2-layer ResKmeans + 1-layer FSQ | **11%** | **95%** | **35.4%** |

关键差异: OneMall 的 FSQ 层使用 **"binary 16-bit MLP"** 量化残差 embedding 为 4096 code — 这正是我们 `LearnedFSQLayer` (MLP + STE) 的方案，而非 EXP-002 失败的 PCA 方案。

EXP-002 失败原因 (PCA 1024D→4-6D 仅保留 20-55% variance) 在 OneMall 中被隐式验证: 他们直接用 MLP 而非 PCA。

### 与当前项目的关联

- `LearnedFSQLayer` 已实现 (`model/fsq.py`)
- `ResKmeansFSQ` 已支持 `mlp` projection type (`model/rkmeans_fsq.py`)
- EXP-003 已设计但 **尚未运行**
- OneMall 结果给出了明确预期: conflict rate 应从当前 ~1.75% 进一步降低，exclusive rate 应提升

### 行动建议

**立即执行 EXP-003**，参考 OneMall 参数:
- FSQ codebook size = 4096 (与 OneMall 一致)
- MLP hidden sizes: {64, 128, 256} (已在 EXP-003 设计中)
- 训练 50 epochs with STE
- 特别关注 conflict rate 和 exclusive rate 的变化

### 关键问题

1. OneMall 的 "binary 16-bit" 具体实现细节不明 — 是否就是 16 个 binary bit 直接做 2^16=65536 codes 然后截断到 4096？还是 FSQ 风格的 multi-level quantization？
2. 我们的 FSQ level config `4d_4096: [8,8,8,8]` 产生 4096 codes，与 OneMall 一致

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P0 | IDEA-015 | 立即执行 EXP-003 (ResKmeans+LearnedFSQ) | OneMall 直接验证方向正确，代码已就绪，conflict 36%→11% |
| P0 | IDEA-010 | NTP In-Batch Contrastive Loss | 实现简单，OneMall 标配，为后续 RL 建立更强基线 |
| P1 | IDEA-013 | Tokenizer 属性增强 Contrastive | OneMall +1.5% HR，与 IDEA-002 互补 |
| P1 | IDEA-012 | GRPO/DPO 强化学习 | 战略重要但依赖强基线 + reward model，建议 IDEA-010/015 之后 |
| P1 | IDEA-011 | Query-Former 序列压缩 | 3.7x FLOP 减少，但需要更长序列场景才有意义 |
| P2 | IDEA-014 | Loss-Free MoE Balancing | 低风险低成本，但 8 experts 下收益可能有限 |
