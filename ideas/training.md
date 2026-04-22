# Training (训练目标与策略)

NTP 模型的训练信号设计：辅助 loss、样本加权、多行为融合等。在不改变模型架构的情况下提升训练质量。

**影响范围**: `metrics/sid_prediction.py`, `model/train.py`, `data/export_behavior.py`

---

## 演进路径

```
纯 CE loss + MoE aux loss (当前 baseline)
├── IDEA-mtgr-0: User-Level Sequence Packing + Causal Mask (消除滑窗)
│   └── 美团 CIKM 2025: 长序列 + dynamic mask, 训练效率大幅提升
├── IDEA-onemall-0: In-Batch Contrastive Loss (连续语义监督)
│   └── EXP-013 S-tier baseline 已建立，可直接实验
├── IDEA-sid-4: Token-Space MTP 辅助 Loss (细粒度 token CE)
│   └── 与 onemall-0 互补: token-level vs embedding-level
│   └── EXP-015: irreducible loss=2.522, 改善空间受 tokenizer 限制
├── IDEA-sid-5: Codebook Embedding 聚合 (item 表示)
│   └── 依赖 IDEA-sid-0 Phase 2 (OPQ 长 ID)
├── IDEA-gr4ad-2: Value-Aware 训练 (eCPM token + 样本加权)
│   └── 引入业务价值信号
├── IDEA-sigma-0: 指令驱动多任务 GR + 自适应概率融合
│   └── AliExpress: instruction-following + adaptive decoding distribution
├── IDEA-lemur-0: 端到端多模态 + Memory Bank 增量表征
│   └── Douyin: 联合优化 + memory bank 解缓存问题, QAUC +0.81%
├── IDEA-oneloc-5: Multi-behavior 序列融合
│   └── 区分 click/buy/expose 不同行为强度
├── IDEA-dualgr-0: Exposure-Aware NTP Loss → EXP-014 数据端完成，NTP 集成待推进
├── IDEA-tbg-0: Data Recency → EXP-016 ✅ 14d 最优，recency > volume 强力验证
└── IDEA-plum-0: LLM Continued Pre-Training (Google/YouTube)
    └── 预训练 LLM → CPT on SID 语料 → Fine-tune, 数十亿用户验证
```

---

## 当前结论 (2026-04-17)

**NTP baseline 已建立，14d 数据窗口确认最优，scaling law 揭示 tokenizer 是瓶颈。**

### 当前 config
```
模型: S-tier (17.5M active, 256d, 6L, 8E top-2 MoE)
数据: 14d 窗口 (03-17~03-31), ~130M tokens, ~1.69M users
训练: 1 epoch, batch=512, lr=1e-3, CosineAnnealing
```

### 关键实验数据

| 实验 | 发现 | 关键指标 |
|------|------|---------|
| EXP-013 | S-tier NTP baseline 建立 | PPL=27.05, R@500=58.5% |
| EXP-014 | ENTP 负样本导出完成 | 130M 正样本行, 31% 有负样本 |
| EXP-015 | Scaling law L(N)=2.522+2055/N^0.456 | irreducible loss=2.522 (PPL≈12.5), M+ loss=2.94 |
| EXP-016 | Data recency > volume, 14d 最优 | U-shaped loss curve, 更多数据反而更差 |

**核心 insight**: Tokenizer (MLP-FSQ 32-bit) 是当前瓶颈而非模型大小。M+ (101M) 相比 S (17.5M) 仅降低 loss 0.06 (2.9960→2.9371)。改善训练信号 (contrastive loss, ENTP) 比 scale up 模型更有效。

---

## IDEA-mtgr-0: User-Level Sequence Packing + Causal Mask Training

**优先级**: P0
**来源**: MTGR (Meituan, arxiv 2505.18654, CIKM 2025)
**状态**: 活跃 — 直接解决当前 `build_sequences()` 瓶颈

### 核心思想

当前 `ntp/train.py:build_sequences()` 用 Python for-loop 对 4.5M 用户做滑窗切分，生成数千万个独立 (input_30, target_3) 样本。这有两个问题：

1. **构建慢**: 纯 Python 循环，4.5M 用户 × 滑窗 = 分钟级延迟，DDP 其他 rank 等到超时
2. **信息浪费**: 滑窗之间大量重叠 token 被重复编码，同一用户的不同窗口无法共享上下文

MTGR 的核心启发：**不做滑窗切分，直接把每个用户的完整行为序列拼成一个长 SID token 序列**，用 causal mask 让模型在每个位置预测下一个 item 的 SID。

```
用户 A 完整序列: [item1_L1, item1_L2, item1_L3, item2_L1, item2_L2, item2_L3, ...]
                  ↓ causal attention ↓
                  每个位置预测下一个 token, 用 per-layer output_proj
```

**等价于所有滑窗位置同时训练，但只做一次 forward pass。**

MTGR 还提出 **Dynamic Masking** 防止信息泄露：
- 静态上下文（用户 profile）→ 对所有位置可见（双向注意力）
- 动态行为序列 → 严格 causal（只看过去）
- 如果 batch 内 pack 多个用户 → block-diagonal mask 防止跨用户注意力

### 与当前项目的关联

**直接解决痛点**：
- `build_sequences()` 消耗分钟级时间 → 改为直接构建用户级长序列，只需 numpy 拼接，秒级完成
- 数据量不变但 forward pass 更少（一个用户一次 forward vs N 个滑窗 N 次 forward）
- 训练吞吐大幅提升：同一用户的中间 hidden states 被所有位置复用

**与 MTGR 的差异**：
- MTGR 做 ranking（判别式），我们做 NTP（生成式）→ causal mask 天然适用
- MTGR 聚合 candidates（同一用户 K 个候选），我们聚合 history（同一用户 N 个历史 item）
- MTGR 用 HSTU（SiLU attention），我们用标准 Transformer → 直接可用

### 实验设计草案

**Phase 1 — 用户级长序列 + Causal Mask（核心改动）**:

替换 `build_sequences()` → `build_packed_sequences()`:
```python
def build_packed_sequences(sid_dict, behavior_data):
    """每个用户 → 一个长 SID token 序列 + causal mask"""
    # 1. 按用户分组 + 时间排序 (向量化, pandas groupby)
    # 2. 每个用户: [item1_tokens, item2_tokens, ...] → flat list
    # 3. 训练时: causal attention, 每个位置预测下一个 token
    #    target = input shifted by 1 (标准 LM 训练)
    # 4. per-layer output_proj 根据 position % n_layers 选择
```

训练循环改为标准 LM 风格：
```python
# 不再区分 input_tokens / target_tokens
# 整个序列做 causal attention, loss 在所有位置计算
logits = model(packed_sequence)  # (B, T, C_layer)
loss = CE(logits[:, :-1], packed_sequence[:, 1:])
```

**Phase 2 — Multi-User Packing（进一步提升 GPU 利用率）**:
- 短序列用户 pack 到同一个 batch 位置（类似 LLM document packing）
- Block-diagonal attention mask 防止跨用户泄露
- 需要 FlashAttention 的 varlen API 或自定义 mask

**Phase 3 — Dynamic Masking（引入 side-info）**:
- 用户 profile token → 双向注意力
- 行为序列 → 严格 causal
- 候选 item → 只看自己（如果做 ranking）

### 改动文件

1. `ntp/train.py` — `build_sequences()` → `build_packed_sequences()`, 训练循环改为 LM 风格
2. `ntp/model.py` — `NTPModel.forward()` 支持长序列 + per-position loss
3. `ntp/baseline.py` — `SIDSequenceDataset` 适配新数据格式

### 关键问题

1. **Position embedding 长度**: 当前 `pos_emb` 最大 `seq_len + n_sid_layers`。用户级长序列可能有数百个 item × 3 tokens = 上千 token → 需要扩展 pos_emb 或改用 RoPE
2. **变长序列 padding**: 不同用户序列长度差异大 → naive padding 浪费计算。需要 packing 或 bucket batching
3. **Eval 不变**: beam search 仍然用固定 n_items 窗口，只是训练方式改变
4. **Group LayerNorm (MTGR)**: 不同 token 类型（不同 SID 层）分组 LayerNorm，是否有帮助

---

## IDEA-onemall-0: In-Batch Contrastive Auxiliary Loss for NTP Model

**优先级**: P0
**来源**: OneMall §3.2 Supervised Objectives
**状态**: ❌ 已测试, 负结果 — EXP-022 全 5 config 均不优于 baseline (详见 experiments/log.md EXP-022)

> **实验结论 (2026-04-22)**: EXP-022 测试了 α∈{0.01,0.1,0.5}、dim∈{128,256}、τ∈{0.05,0.07} 共 5 个 config。最好的 α=0.01 仅 +0.7pp R@500 但 PPL 劣化 +0.84。α 越大越差。根因：SID 是离散 token，InfoNCE 连续空间对齐对离散预测无帮助。**不再追。**

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

**基线**: EXP-016 14d-S (PPL=27.05, loss=2.9960, R@500=58.5%)

**评估指标**: beam search Recall@{10,50,100,500}, SID accuracy@{1,2,3}, 训练收敛速度

### 关键问题

1. batch size 需要足够大以提供足量 in-batch negatives — 当前 batch size 是多少？可能需要增大
2. s₃ 隐层是否需要 stop-gradient (asymmetric design) 还是两边都 backprop
3. 训练早期 contrastive loss 可能主导梯度，需要 warmup 策略（先纯 NTP 若干 epoch 再加 contrastive）

---

## IDEA-sid-4: Token-Space MTP 辅助 Loss (适用于自回归模型)

**优先级**: P1
**来源**: RPG (KDD'25, arxiv 2506.05781) §2.2.1 Multi-Token Prediction
**状态**: 待讨论 — 受 irreducible loss floor 约束

> **NTP 阶段更新 (2026-04-17)**: EXP-015 scaling law 显示 irreducible loss a=2.522 (PPL≈12.5)，M+ (101M) 已达 loss=2.94，距 floor 仅 0.42。MTP 辅助 loss 仍有价值但改善空间受 tokenizer 上限限制。主要收益在冷启动 item 和 R@10 精度提升，而非大幅降低 loss。

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
**状态**: EXP-014 实验中 — PySpark 负样本导出完成，NTP 集成待验证

> **NTP 阶段更新 (2026-04-17)**: EXP-014 已完成 PySpark 端 ENTP 负样本导出和数据验证 (130M 行, 31% 有负样本)。Python 端 `load_exposure_neg_data()` 已实现。L0 层碰撞导致部分负样本与正样本共享 L1 cluster，需要在 ENTP loss 中处理。NTP 集成待下阶段推进。

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

### 实现记录

**PySpark 端 ENTP 负样本导出 (2026-04-16)**:

实现方式: `data/export_exposure.py` 新增 ENTP section，Spark SQL window function
`pos_grp = cumsum(action_bitmap > 0)` 分段，每段 non-positive 作为下一个 positive 的负样本，
COLLECT_LIST + SORT_ARRAY + SLICE 取最近 K=5 个。输出 compact parquet `feed_user_exposure_neg/`。
Python 端 `load_exposure_neg_data()` 加载 ~130M 行（秒级），`_build_sequences_from_exposure()`
只做 iid→L0 token 映射。

**数据验证 — PySpark 导出 vs 旧流式 walk 对比 (03-01~03-31)**:

| 指标 | PySpark 导出 | 旧流式 walk (对照) | 说明 |
|---|---|---|---|
| 总曝光行 | ~1.19B | 1,185,707,891 | 一致 |
| Positives (action_bitmap > 0) | 130,995,419 | 124,893,764 | +4.9% |
| Users | 4,608,606 | 3,042,069 | +51% |
| 有负样本 | 40,761,718 (31.1% row级) | 2,084,314 (68.5% user级) | 口径不同 |

差异分析:
- **Positives +4.9%**: PySpark 不过滤 SID 字典外 iid，多出 ~6M。Python 端 `_build_user_items()` 的 `iid ∈ SID` 过滤兜底，不影响最终序列
- **Users +51%**: 多出的 1.5M 用户只有 SID 字典外 iid，Python 端过滤后不足 2 个 valid item，不产出序列
- **有负样本 31% vs 68.5%**: 口径不同无矛盾。31% 是 row 级（131M 正样本行里 41M 有 neg），68.5% 是 user 级（3M 用户里 2M 有 neg）。Feed 场景用户常连续点击（同页多 item），连续 positive 之间无 non-positive → 后者拿不到 neg
- 旧流式 walk 最终 3,042,069 users / 76M items / 59M neg tokens；PySpark 导出经 Python 端过滤后应得到一致结果

**性能对比**:
- 旧流式 walk: Phase 1 读 620 文件 2917s + Phase 2 groupby 1350s = **~71 min**
- PySpark 导出: Spark 集群分钟级 + Python load_exposure_neg_data() **~30s**

### 关键问题

1. ~~需要行为数据包含"曝光未点击"信息~~ ✅ `export_exposure.py` 已有完整曝光序列
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
**状态**: Phase 1 (Data Recency) 已被 EXP-016 验证 ✅ — Phase 2 (NSP) 待实验

> **NTP 阶段更新 (2026-04-17)**: EXP-016 Data Scaling Law 实验强力验证了 **data recency > data volume**: 14d (130M tokens) 是最优训练窗口，31d/62d/90d 数据量更大但 loss 更高 (U-shaped curve)。原因: 更多天数 = 更多用户 (1.02M→6.18M) 而非更长序列，3 天曝光周期导致旧用户行为模式不适用于当前 eval 分布。Phase 1 结论已融入生产配置 (14d 数据窗口)。Phase 2 NSP 作为独立实验方向保留。

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
**状态**: 待讨论 — 受 scaling law flattening 限制

> **NTP 阶段更新 (2026-04-17)**: EXP-015 scaling law L(N)=2.522+2055/N^0.456 显示 M+ (101M active) 已达 loss=2.94，距 irreducible floor (2.522) 仅 0.42。论文在 176M→1B 才看到 task decomposition scaling 效果，而我们当前场景 scaling law 在 ~100M 已趋平（tokenizer 是瓶颈而非模型大小）。优先级维持 P1 但实际收益可能有限。

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

## IDEA-mbgr-0: Multi-Business Generative Recommendation (BID + MBP + LDR)

**优先级**: P2
**来源**: MBGR (Meituan, arxiv 2025, WWW 2026)
**状态**: 待定

> **P2 原因**: 多业务扩展是部署期需求，当前单业务 NTP baseline 尚未建立。核心技术 (Business-aware SID, MBP heads, Label Dynamic Routing) 在多业务扩展时直接参考。

### 核心思想

MBGR 是美团在多业务场景 (外卖、酒旅、到店等) 的生成式推荐部署。核心挑战：不同业务的 item 共享同一 SID 空间导致业务间干扰。三个关键技术：

1. **Business-aware SID (BID)**: 在 SID 序列前追加一个 Business ID token，让模型区分不同业务的 item 空间。SID 从 `"L1_L2_L3"` 变为 `"BIZ_L1_L2_L3"`
2. **Multi-Business Prediction (MBP)**: 每个业务一个独立的 prediction head，共享 encoder 但 head 独立。类似 multi-task learning 但在 SID 空间
3. **Label Dynamic Routing (LDR)**: 动态调整不同业务的训练样本权重，解决业务间数据量不均衡（外卖数据 >> 酒旅数据）

美团在线 A/B: 多业务联合训练 > 单业务独立训练，**全业务平均 CTR +1.2%，长尾业务 CTR +3.5%**。WWW 2026。

### 与当前项目的关联

- 当前项目只有单一推荐场景，BID 暂不需要
- **MBP heads 的思想可泛化**: 如果要区分不同行为类型 (click/buy/share)，可以每种行为一个 prediction head
- LDR 与 IDEA-onerec-1 (RSFT) 和 IDEA-gr4ad-2 (Value-Aware) 相关：都是训练样本加权策略
- **多业务扩展时的首选参考方案**: 当推荐系统需要服务多个业务线时，BID + MBP 是最低成本的扩展方案

### 实验设计草案

**当前不实验，作为多业务扩展参考**。

如果需要多业务扩展:
1. 在 SID 前加 BIZ token (vocab 按业务数扩展)
2. Encoder 共享，prediction head 按业务拆分
3. LDR: 按业务 loss 动态调权 (类似 GradNorm)

### 关键问题

1. 当前单业务场景无需 BID
2. MBP 的 head 数量随业务增长 → 参数膨胀
3. LDR 的动态路由策略需要充分的多业务数据验证

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P0 | IDEA-mtgr-0 | User-Level Packing + Causal Mask | 消除滑窗瓶颈, 训练吞吐大幅提升, 美团 CIKM 2025 |
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
| P2 | IDEA-mbgr-0 | Multi-Business Prediction + BID | 美团部署, 多业务扩展时参考 |
| P1 | IDEA-sigma-0 | 指令驱动多任务 GR + 自适应融合 | AliExpress 在线验证, 多任务扩展方向 |
| P1 | IDEA-lemur-0 | 端到端多模态 + Memory Bank | Douyin QAUC +0.81%, Memory Bank 低成本可先验证 |
| P1 | IDEA-genrec-0 | Page-wise NTP 多标签页面级监督 | JD SIGIR 2026, +9.5% click, 幻觉率降 50%, 推理不变 |
| P1 | IDEA-rclrec-0 | 反向课程学习稀疏转化 | Alibaba +2.09% revenue, decoder prefix 额外监督 |

---

## IDEA-sigma-0: 指令驱动多任务生成式推荐 + 自适应概率融合

**优先级**: P1
**来源**: SIGMA, Alibaba/AliExpress (arxiv 2602.22913)
**状态**: 待讨论

### 核心思想

阿里 AliExpress 部署的 SIGMA 将生成式推荐从 "交互驱动的 next-item prediction" 扩展为 "指令驱动的多任务推荐"。三个关键设计: (1) 统一 semantic-collaborative 潜在空间 — 同时捕获语义关系和协同关系，item grounding 不依赖单一信号; (2) Hybrid item tokenization — 精确建模 + 高效生成的平衡; (3) 自适应概率融合 (Adaptive Probabilistic Fusion) — 根据任务类型 (recall/ranking/diversity) 动态校准生成分布，同一个模型用不同 instruction 服务不同推荐需求。大规模 SFT 数据集支持 instruction following。AliExpress 在线 A/B 验证有效。

### 与当前项目的关联

- 当前 NTP 只做单一 recall 任务，SIGMA 的 instruction-following 思路可以扩展模型能力
- Adaptive Probabilistic Fusion 对推理阶段有直接价值: 同一个模型可以用 "precision" 或 "diversity" instruction 控制输出分布
- Hybrid item tokenization 可能包含对 MLP-FSQ tokenizer 的改进方向
- SFT dataset 构造方法可以参考: 将现有行为数据转化为多任务 instruction 格式

### 实验设计草案

**Phase 1 — Adaptive Decoding Temperature per Task**:
- 最简实现: 不同 beam search temperature 模拟不同 task instruction
- recall task → low temperature (precision), diversity task → high temperature
- 评估: Recall@K vs Coverage@K 的 trade-off

**Phase 2 — Instruction Prefix for NTP**:
- 在行为序列前加 task instruction token (e.g., [RECALL], [DIVERSE], [SIMILAR])
- 训练时用不同 label 策略: RECALL→下一个点击, DIVERSE→随机正例, SIMILAR→same-category 正例
- 需要 multi-task SFT 数据构造

### 关键问题

1. AliExpress 的多任务需求 (recall/ranking/diversity) 在当前单一 recall 场景下价值有限
2. Instruction-following 需要更大的 backbone (当前 small decoder 难以理解复杂 instruction)
3. Adaptive Probabilistic Fusion 的实现细节需要看 full paper
4. 更适合系统成熟后扩展多任务能力时引入

---

## IDEA-lemur-0: 端到端多模态推荐 + Memory Bank 增量表征

**优先级**: P1
**来源**: LEMUR, ByteDance/Douyin (arxiv 2511.10962)
**状态**: 待讨论

### 核心思想

字节跳动在抖音搜索和广告部署的 LEMUR 是首个大规模端到端多模态推荐系统: 联合优化多模态 encoder 和推荐模型，而非 "先预训练多模态模型，再冻结表征训练推荐模型" 的两阶段方案。核心创新是 Memory Bank 机制: 训练过程中增量积累历史多模态表征，避免对用户长历史序列做完整多模态 forward pass 的巨大计算开销。抖音搜索部署一个月后: query change rate decay -0.843%, QAUC +0.81%。

### 与当前项目的关联

- 当前项目用 Qwen3 冻结 embedding → tokenizer → NTP 的三阶段 pipeline，LEMUR 验证端到端联合训练的优势
- Memory Bank 机制对长序列 NTP 有直接价值: 无需每次训练都重新计算历史 item 的 embedding，用缓存 + 增量更新
- 当前 NTP 训练中 item embedding 是 pre-computed & frozen，如果要做 end-to-end，Memory Bank 是解决计算瓶颈的关键
- 与 IDEA-onerec-3 (QFormer Tokenizer) 方向一致: 都追求打破 frozen embedding 的限制

### 实验设计草案

**Phase 1 — Embedding Memory Bank for NTP Training**:
- 在 NTP 训练中维护 item embedding memory bank (size = item pool)
- 每个 epoch 开始时用当前 encoder 更新 memory bank (或用 EMA)
- 对比: frozen embedding vs memory bank (定期更新) vs full end-to-end
- 评估: Recall@K + 训练效率 (FLOPs per epoch)

**Phase 2 — End-to-End Multimodal Training**:
- 解冻 Qwen3 embedding encoder，与 NTP 联合训练
- 用 Memory Bank 缓存中间表征，每 N steps 更新
- 前置: 需要 GPU 内存优化 (gradient checkpointing, mixed precision)

### 关键问题

1. End-to-end 训练需要远超当前的 GPU 资源
2. Memory Bank 的 staleness: 缓存表征与当前 encoder 不同步 → 需要验证影响
3. 当前 MLP-FSQ tokenizer 是在 frozen embedding 上训练的，end-to-end 意味着 tokenizer 也要重训
4. Phase 1 (Memory Bank alone) 相对低成本，值得先验证

---

## IDEA-genrec-0: Page-wise NTP 训练目标 (多标签页面级监督)

**优先级**: P1
**来源**: GenRec, JD.com (arxiv 2604.14878, SIGIR 2026)
**状态**: 待讨论

### 核心思想

JD 的 GenRec 提出 Page-wise NTP (PW-NTP): 将同一请求页面内用户的多个正交互 (点击+购买+曝光) 拼接成一个 target 序列做自回归训练，而非 vanilla NTP 对每个正样本独立建模。解决了工业分页机制导致的 "相同输入、多个有效输出" 的 one-to-many 歧义问题。实验显示 PW-NTP 比 vanilla NTP: (1) 收敛更快; (2) HR@50 从 0.62 提升到 0.72; (3) 幻觉率降低 50% (7.8% → 4.96%)。JD 在线 A/B: 点击 +9.5%, 成交 +8.7%。推理时仍用标准 point-wise beam search，训练-推理不对称是有意设计。

### 与当前项目的关联

- 当前 NTP 训练是 point-wise: 一个 (history, target_item) pair 一条训练样本
- PW-NTP 可以直接在现有数据上实现: 把同一 session 内多个正样本拼接成 target 序列
- 与 IDEA-onemall-0 (Contrastive Loss) 互补: PW-NTP 改善 SFT 阶段, Contrastive 是额外辅助 loss
- 训练-推理不对称设计意味着**推理端完全不需要改动**
- 幻觉率降低对 SID 体系特别有价值: 减少生成无效 SID 组合

### 实验设计草案

**Phase 1 — Session-level Multi-Target NTP**:
- 数据构造: 将同一 session (或同一天) 内用户的多个正交互 item SID 拼接为 target
- 按交互强度排序: buy > click > expose
- 训练: 标准自回归 loss 但 target 是多 item 序列
- Baseline: 当前 point-wise NTP
- 评估: HR@K, NDCG@K, HaR (幻觉率)

**Phase 2 — 加入行为类型 token**:
- 在 target 序列中插入行为类型 token: `<buy> SID1 SID2 SID3 <click> SID4 SID5 SID6 ...`
- 探索不同排序策略对性能的影响

### 关键问题

1. 数据格式变化: 当前 dataloader 需要改造，支持变长 target 序列
2. 推理不变但训练 batch 内序列长度增加 → GPU 内存压力
3. 与 IDEA-dualgr-0 (ENTP-Loss) 有部分重叠: 都关注多行为训练，需要比较或融合

---

## IDEA-rclrec-0: 反向课程学习解决稀疏转化建模

**优先级**: P1
**来源**: RCLRec, Alibaba International (arxiv 2603.28124)
**状态**: 待讨论

### 核心思想

阿里国际电商 RCLRec 提出 Reverse Curriculum Prefix Module (RCPM): 对每个转化目标，从用户历史中反向选择 k 个与转化最相关的行为，将其 SID token 作为 decoder prefix，与目标转化 token 拼接做 teacher forcing。核心 insight: 转化行为前通常有一组聚类的相关行为 (同品类浏览/比较)，直接提取这些关键子序列作为额外监督。加入 curriculum quality-aware loss 确保选出的 prefix 确实提升转化预测。在线 A/B: 广告收入 +2.09%, 订单 +1.86%。

### 与当前项目的关联

- 当前 NTP 训练不区分行为类型，转化行为极稀疏 (通常 <2% interactions)
- RCPM 可以作为 NTP 训练的增强模块: 对高价值目标 (购买) 提供额外的 decoder-side supervision
- 与 IDEA-genrec-0 (PW-NTP) 互补: PW-NTP 解决 one-to-many 歧义, RCLRec 解决 conversion sparsity
- 与 IDEA-oneloc-5 (Multi-behavior 序列) 互补: oneloc-5 是 encoder 端行为融合, RCLRec 是 decoder 端课程注入
- 需要 encoder-decoder 架构 (当前是 decoder-only) → 可能需要适配

### 实验设计草案

**Phase 1 — Decoder Prefix for High-Value Targets**:
- 对训练样本中的购买行为, 从 encoder hidden states 中选 top-k 相关历史 item
- 将选出的 k 个 item SID 作为 decoder prefix, 拼接在目标 SID 前
- 用 scaled dot-product 做 relevance scoring
- Baseline: 标准 NTP (无 prefix)
- 评估: Recall@K (conversion items), NDCG@K

**Phase 2 — Quality-Aware Loss**:
- 加入 hinge loss: 确保有 prefix 的转化 NLL < 无 prefix 的 NLL + margin
- 调节 margin 和 loss weight

### 关键问题

1. 当前是 decoder-only 架构, RCLRec 需要 encoder-decoder → 需要改造或用 prefix-LM 方式适配
2. RCPM 的 top-k 选择需要可微化 (IBQ straight-through estimator)
3. 前置: 需要多行为训练数据 (目前数据中是否有行为类型标注?)
4. k=4 是论文推荐值, 需要在我们的数据上调参

---

## IDEA-lac-0: Lagged Action Conditioning (动作延迟条件化)

**优先级**: P1
**来源**: The Layout Is the Model (Roblox, arxiv 2510.16804)
**状态**: 直接关联 EXP-023/025 的 action_level 泄漏问题

### 核心思想

GR 中同时建模 item 和 action (行为类型/观看时长等) 时，token layout 决定了信息泄漏和条件化关系。论文提出三个设计原则：
1. **最大化 item/action 信号** (input 和 output 都要用)
2. **保持 "action given item" 的条件方向** (先看到 item 再预测 action)
3. **无信息泄漏** (预测 item 时不能看到该 item 的 action)

**Lagged Action Conditioning (LAC)**:
- 非交错布局: 每个 token 只是 item SID (不单独给 action 一个 token)
- **Lag**: item_i 的 action 作为 item_{i+1} 的输入 feature（延迟一个 item）
- 这样预测 item_{i+1} 时能看到 item_i 的 action（有信息增量），但看不到 item_{i+1} 自己的 action（无泄漏）
- 推理时: 生成 item_{i+1} 的 SID 后，action_{i+1} 未知，用最后已知的 action_i 的值

### 与我们 EXP-023/024/025 的关系

**这就是我们在 EXP-024 中尝试的 shift 方案的理论版本！** 但我们的 shift 实现在 flat token 级别操作，LAC 是在 item 级别做 lag:
- 我们的 EXP-024 shift: 每个 item 的 3 token 用上一个 item 的 features → **等价于 LAC**
- EXP-024 结果不佳的原因: **不是 LAC 思路错，而是 beam search incremental path 没传 features**
- EXP-025 修复了 beam search feature passing → 应重新评估 LAC (shift) + beam passes 的组合

### 实验设计

重新评估 EXP-024 shift + EXP-025 beam passes 的组合:
1. 用 shifted 数据 (exp024-14d-shifted) 训练
2. Beam search 传入 `gen_action_level = last context item's action` (即 LAC 的推理逻辑)
3. 对比 EXP-025 beam-passes (不 shift, 传真值)

### 关键问题

1. LAC 在论文中用的是 explicit action token (watchtime 等)，我们的 action_level 是更粗的 bitmap 分桶
2. 论文 backbone 是 85M 参数，我们 17.4M active → 小模型是否有足够容量利用 action 信号
3. 需要验证: shift + beam_passes vs 不 shift + beam_passes 哪个更好

---

## IDEA-onelive-0: BOS 全局时间注入 + Gated Attention

**优先级**: P2
**来源**: OneLive (Kuaishou, arxiv 2602.08612)
**状态**: 待讨论

### 核心思想

OneLive 在快手直播推荐中部署的两个时间建模技术:

1. **BOS 时间注入**: 在 [BOS] token 注入 multi-granular temporal features
   ```
   x_BOS = x_BOS + MLP(Concat(x_Hour, x_Day, x_Week))
   ```
   用当前时刻的 hour-of-day / day-of-week / week 做 embedding，通过 MLP 融合后加到 BOS token
   - 优点: 极简实现，不改 attention 结构
   - 缺点: 只编码 "当前请求时刻"，不编码序列内各 item 的时间

2. **Gated Attention 时间感知**: 在 attention 输出上加 element-wise gate
   ```
   Score(X) = σ(X · W_θ)
   O = MultiHeadAttn(Q, K, V) ⊙ Score(X) · W_O
   ```
   让模型学习抑制时间上不相关的 context

### 对我们的适配

BOS 时间注入方案可以最低成本验证"全局时间是否有用":
1. 在第一个 token 位置（或新增一个 [BOS] token）注入 hour/dayofweek embedding
2. 不改位置编码，不改 attention
3. 如果有效 → 再升级到 TO-RoPE (feat-5)

### 关键问题

1. 我们的 NTP 序列没有 [BOS] token — 需要新增还是注入到第一个 item 的第一个 SID token
2. 直播场景的时间敏感度远高于内容推荐 — 效果可能打折扣
3. 快手论文没有单独 ablation BOS 时间注入的贡献

---

## IDEA-tca-0: Token-level CF Soft Label Alignment (CF 信号注入 NTP Loss)

**优先级**: P1
**来源**: TCA4Rec, USTC + Ant Group (arxiv 2601.18457, WWW 2026)
**状态**: 待讨论

### 核心思想

TCA4Rec 解决 NTP 模型缺乏 collaborative filtering 信号的核心问题。核心 insight: CF 模型做 **item-level** 排序，NTP 做 **token-level** 预测，两者优化粒度不匹配 → 之前的方法只能把 CF 作为 soft prompt 或 representation bias 被动注入。

TCA4Rec 提出 **显式 token-level CF alignment**:

1. **Collaborative Tokenizer**: 从预训练 CF 模型 (如 SASRec) 获取 item-level logits (z_u,i = dot(e_u, e_i))，通过三步变换为 token-level 分布:
   - Step 1: 收集当前 decode position 的 valid items (前缀匹配)
   - Step 2: Softmax 归一化为概率分布 π_u,i
   - Step 3: 按 next token 聚合 (共享同一 next token 的 items 概率求和)
   
2. **Soft Label Alignment**: 将 CF token-level 分布与 one-hot 标签融合: ỹ_j(v) = (1-α)·1_{v=y_j} + α·p_u(v|y_{<j})
3. **Soft NTP Loss**: L_soft = -Σ log(Σ ỹ_j(v)·P(v|x_u,y_{<j}))

α=0 退化为标准 NTP, α=1 完全跟 CF。最优 α ≈ 0.01~0.05 (CF 信号作为 gentle regularizer)。

**与 Auxiliary KL Loss 的关键区别**: Soft NTP 的梯度是 **adaptive** (权重 q_j 依赖模型当前预测 P_j)，而 KL 用固定权重 ỹ_j。Adaptive 权重使模型能平衡 CF 信号和自身世界知识。

**核心结果**:
- 在 4 种 LLM-based 推荐架构上一致提升 (TallRec, LLaRA, CoLLM, MSL)
- MSL+TCA on Toys: NDCG@5 0.0145→0.0332 (+129%), H@5 0.0204→0.0452 (+121%)
- 对 SID-based 方法也有效: TIGER+TCA, LETTER+TCA
- Collaborative Consistency 随 α 单调增加，但性能先升后降 (α 过大引入 CF 噪声)
- Model-agnostic + plug-and-play: 不改模型架构，只改 loss

### 与当前项目的关联

- **直接回应 NTP 缺乏 cross-user signal 的问题**: CF 模型的 logits 天然包含 cross-user collaborative 信号，TCA 通过 loss 层面注入
- 我们的 NTP 模型用 SID (不是 item title text)，Collaborative Tokenizer 需要适配:
  - SID token 空间 (L1=1024, L2=1024, L3=4096) vs LLM vocab
  - 前缀匹配变成 SID 前缀匹配 (L1 → L1+L2 → L1+L2+L3)
  - 概率聚合按 SID token group 而非 text token
- 前置条件: 需要一个预训练的 CF 模型 (SASRec) — 与 IDEA-flexcode-0 共享此依赖
- 与 IDEA-onemall-0 (In-Batch Contrastive Loss) 互补: onemall-0 在 representation 层加 CL loss，TCA 在 output token distribution 层加 soft label
- **Zero model architecture change** — 只改 loss function → 实验成本极低

### 实验设计草案

**Phase 1 — 预训练 SASRec CF 模型**:
- 在相同用户行为序列上训练 SASRec → 获取 user/item embedding
- 为每个训练样本计算 CF logits (user dot-product all items)

**Phase 2 — Collaborative Tokenizer for SID**:
- 对每个 decode position j (L1/L2/L3):
  - 根据已生成的 SID 前缀筛选 valid items
  - Softmax normalize CF logits
  - 按 SID layer-j token 聚合概率
- 产出: 每个训练样本每个 decode position 的 token-level CF 分布

**Phase 3 — Soft NTP Training**:
- 修改 NTP loss: (1-α)·CE + α·CF_soft_label
- 超参搜索 α ∈ {0.001, 0.005, 0.01, 0.05, 0.1}

### 关键问题

1. **效率**: 每个训练样本需要计算 CF logits (dot product with all items) — 5M items 时是否可行? 可能需要 ANN 近似 top-K
2. SASRec CF 模型的质量是否足够? 论文用的是 academic datasets (19K users)，我们规模大很多
3. SID token 空间远小于 LLM vocab → 前缀匹配的 valid item set 在 L1 层可能很大 (每个 L1 token 对应 ~5000 items)
4. 与 IDEA-flexcode-0 的 CF 模型可以共享，但 TCA 的 CF 信号注入方式更轻量 (只改 loss)
