# Architecture (模型架构)

NTP 模型的架构设计：解码策略、序列压缩、注意力机制、专家路由等。影响推理效率和模型容量。

**影响范围**: `metrics/sid_prediction.py`, `model/train.py`

---

## 演进路径

```
AutoregressiveNTPModel (当前 6-layer decoder, beam=5)
├── IDEA-gr4ad-1: LazyAR (前 K 层非 AR，后 L-K 层 AR)
│   └── 推理吞吐翻倍，beam 共享 KV cache
├── IDEA-onemall-1: Query-Former (长序列 cross-attention 压缩)
│   └── 1205→160 token, 3.7x FLOP 减少
├── IDEA-onemall-4: Loss-Free MoE (动态 bias 替代 aux loss)
│   └── 低风险改进 MoE load balancing
├── IDEA-glide-0: Soft Prompt Injection (user embedding → prefix)
│   └── Spotify 验证, 非惯常收听 +5.4%, 新发现 +14.3%
├── IDEA-oneloc-0: Context-augmented Attention (side-info 注入)
│   └── additive similarity + gating，需 encoder-decoder 架构
├── IDEA-oneloc-1: Category Prompt (邻域 cross-attention 提示)
│   └── 泛化为 interest/category prompt prefix
├── IDEA-oxygen-0: Fast-Slow Thinking (近线 LLM + 实时 GR)
│   └── LLM 推理蒸馏为指令，IGR 意图过滤，SA-GCPO 多场景 RL
└── IDEA-llada-0: Discrete Diffusion (替代自回归)
    └── 双向注意力 + 自适应生成顺序，解决错误累积
```

---

## IDEA-gr4ad-1: LazyAR 解码器

**优先级**: P1
**来源**: GR4AD §LazyAR, Table 1
**状态**: 待讨论

### 核心思想

GR4AD 将 L 层 decoder 分为两部分: 前 K 层（非自回归）只依赖位置编码和 context，不依赖前一个 token；后 L-K 层才引入自回归依赖。关键洞察: 前 K 层的输出可以对所有 token 位置并行计算并在 beam 间共享，只有后 L-K 层需要逐 token 解码。实验显示 K=2/3·L 时性能几乎无损（-0.04%），但推理吞吐翻倍。

Fusion 机制: 在第 K 层用 gated projection 融合非自回归表示和前一 token embedding:
`Fuse(m, s) = W_f[m ⊙ (W_g · s); s]`

### 与当前项目的关联

- `metrics/sid_prediction.py` 的 `AutoregressiveNTPModel` 是纯自回归: 每层都依赖前一 token 的 embedding
- 当前只有 3 个 token 要预测，beam_size=5，推理不是瓶颈。但如果扩展到更多 token (IDEA-sid-0 OPQ 方案 B/C 有 16-32 token) 或更大 beam (生产目标 512)，LazyAR 变得关键
- **与 ARCHITECTURE.md 的 Lazy Decoder-Only 设计方向一致** — OneRec-V2 的 Context Processor 本质上也是将编码和解码分离
- 可以作为 NTP 模型升级的一部分实现

### 实验设计草案

**Phase 1 — 验证概念 (S-tier model)**:
- 当前 6 层 decoder，设 K=4（前 4 层 non-AR，后 2 层 AR）
- 在第 4 层加入 gated fusion 模块
- 对比: 原始 AR vs LazyAR，评估 perplexity / Hit@K / 训练速度

**Phase 2 — 推理加速验证**:
- beam_size 从 5 扩大到 50-500
- 测量: LazyAR 的 beam-shared KV cache 带来的推理时间节省
- 预期: K=4 时，前 4 层计算量不随 beam 增长，只有后 2 层线性增长

**改动文件**: `metrics/sid_prediction.py` — 修改 `AutoregressiveNTPModel` 的 forward 和 generate 方法

### 关键问题

1. **3 token 场景收益有限**: 当前只预测 3 个 token，beam search 的主要计算在第一层（16384 vocab softmax），LazyAR 优化的是后续层。需要量化真实推理收益
2. 论文指出 LazyAR 不适合通用 LLM（token 间依赖强且长度不固定），但推荐场景 token 少且后续层"更简单"——需要验证在我们的 3 层设定下是否成立
3. Fusion 机制的设计: gated projection vs 简单 add vs concat — 需要 ablation

---

## IDEA-onemall-1: Query-Former 长序列压缩

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

## IDEA-onemall-4: Loss-Free MoE Load Balancing

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
3. 可以与 IDEA-onemall-0 (contrastive loss) 同时实验，因为修改正交

---

## IDEA-oneloc-0: Geo-aware Self-attention (Context-augmented Attention)

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

## IDEA-oneloc-1: Neighbor-aware Prompt (Category Prompt)

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

1. 同 IDEA-oneloc-0: 当前无 encoder-decoder 架构，无法直接落地
2. 需要先完成 "Context Processor" 或 encoder-decoder 重构
3. 类目信息的获取: 当前 item metadata 是否包含类目? 需要检查数据 pipeline

---

## IDEA-glide-0: Soft Prompt Injection (用户 Embedding → Decoder Prefix)

**优先级**: P1
**来源**: GLIDE (Spotify, arxiv 2603.17540, Mar 2026)
**状态**: 待讨论

### 核心思想

GLIDE 将推荐建模为 **instruction-following** 任务。关键架构创新: 将长期用户 embedding 作为 **soft prompts** 注入 decoder，而非将用户信息编码为 token 序列。

1. **Soft Prompt Injection**: 长期用户 embedding → learned projection → 作为 decoder 的前缀 KV states
2. **Instruction Conditioning**: 短期行为序列 + 轻量用户上下文作为 "指令"，引导生成方向
3. **Semantic ID Catalog Grounding**: 用 SID 确保生成的推荐都是有效 catalog item

Spotify 大规模在线 A/B (百万级用户): **非惯常收听 +5.4%，新节目发现 +14.3%**。

### 与当前项目的关联

- 当前 NTP 模型没有任何 user representation 注入机制
- IDEA-oneloc-1 (Category Prompt) 提出了类似的 prefix 思路，但 GLIDE 更通用: 任何用户 embedding 都可以做 soft prompt
- **低成本实现**: 在 decoder 输入序列前加几个 learned prefix token (来自 user embedding projection)，不需要改变 decoder 架构
- 可以与 IDEA-sid-1 (协同信号增强 embedding) 结合: 增强后的 user embedding 做 soft prompt

### 实验设计草案

**实现**:
1. User embedding: 用用户近期行为的 item embedding 均值 (或 attention pooling)
2. Projection: `MLP(user_embed_dim → decoder_embed_dim × n_prefix)` → reshape 为 n_prefix 个 prefix token
3. Decoder 输入: `[prefix_1, ..., prefix_n, sid_1, sid_2, sid_3]`
4. n_prefix ∈ {2, 4, 8}

**评估**: 有/无 soft prompt 的 NTP Recall@K

### 关键问题

1. User embedding 从哪里来? 当前项目没有预训练的 user embedding
2. 如果用行为序列均值作为 user embedding，信息量是否足够
3. Prefix token 增加了序列长度 → 训练和推理成本上升

---

## IDEA-oxygen-0: Fast-Slow Thinking (近线 LLM 推理 + 实时生成)

**优先级**: P2
**来源**: OxygenREC (arxiv 2512.22386, Dec 2025)
**状态**: 待讨论

### 核心思想

OxygenREC 提出 **Fast-Slow Thinking** 架构解决 LLM 推理在实时推荐中不可用的问题:

1. **Slow Thinking (近线)**: LLM 离线/近线生成 **Contextual Reasoning Instructions** — 将复杂用户意图推理蒸馏为结构化指令
2. **Fast Thinking (实时)**: 高效 encoder-decoder 消费这些指令做实时 SID 生成
3. **Instruction-Guided Retrieval (IGR)**: 用指令过滤行为序列，只保留意图相关的交互
4. **SA-GCPO**: Soft Adaptive Group Clip Policy Optimization，多场景统一 RL 对齐

核心创新: 将 LLM 的深层推理能力通过"指令"传递给轻量模型，实现"train-once-deploy-everywhere"。

### 与当前项目的关联

- 当前项目没有 LLM 推理环节，NTP 模型直接从行为序列生成 SID
- Fast-Slow 架构在我们当前阶段过于复杂，但 **IGR (指令引导的行为过滤)** 思想值得借鉴:
  - 不是把用户全部行为序列都输入，而是先用轻量模型/规则过滤出与当前上下文相关的子序列
- SA-GCPO 是 GRPO 的多场景扩展，与 IDEA-onemall-2 有关联

### 关键问题

1. 当前项目是单场景，Fast-Slow + 多场景部署的价值有限
2. IGR 的"指令"从哪里来? 需要 LLM 或规则系统支持
3. 适合作为架构终极形态参考，当前阶段不适合实施

---

## IDEA-llada-0: Discrete Diffusion 替代自回归解码

**优先级**: P2
**来源**: LLaDA-Rec (arxiv 2511.06254, Nov 2025)
**状态**: 待讨论

### 核心思想

LLaDA-Rec 用 **Masked Discrete Diffusion** 替代自回归解码生成 SID，解决两个根本问题:

1. **单向约束**: causal attention 限制每个 token 只能看到前面的 token，破坏全局语义建模
2. **错误累积**: 左→右固定生成顺序让早期 token 错误传播到后续 token

技术要点:
- **Parallel Tokenization Scheme**: 专为双向注意力设计的 SID（与 RQ 的有序 SID 不同）
- **Dual Masking**: user-history level (序列间依赖) + next-item level (item 内 token 间语义)
- **Adapted Beam Search**: 适配 diffusion 的非固定顺序解码

### 与当前项目的关联

- 当前 NTP 模型是纯自回归 (`AutoregressiveNTPModel`)
- IDEA-sid-0 (OPQ 并行 ID) 已经在走非自回归路线 (并行预测+图解码)
- LLaDA-Rec 提供了另一种非自回归方案: diffusion。与 OPQ 的区别:
  - OPQ: 完全独立预测各 token → 图解码约束
  - Diffusion: 迭代去噪，token 间有隐式交互 → 可能更好的全局一致性
- **目前优先级低**: IDEA-sid-0 (OPQ) 已经在实验中，先看 OPQ 结果

### 关键问题

1. Diffusion 的推理延迟: 需要多步去噪 (T=10~50 steps)，比自回归和并行预测都慢
2. 与 OPQ 并行预测的对比: 需要在相同 SID 配置下才有意义
3. 训练复杂度: diffusion training 需要噪声调度、去噪网络设计等额外工程

---

## IDEA-s2gr-0: Stepwise Reasoning Tokens in SID Generation

**优先级**: P1
**来源**: S²GR (arxiv 2601.18664, Jan 2026)
**状态**: 待讨论

### 核心思想

S²GR 在 SID 自回归生成的每一步前插入 **thinking token**，让模型在生成每个 SID code 前先"思考"。关键区别于 OneRec-Think: reasoning 不是在 SID 生成之前集中做，而是 **interleaved** — 每个 SID code 之前都有一个 thinking step。

技术要点:
1. **Thinking tokens**: 每个 SID code 之前插入 reasoning token，受 **contrastive learning** 监督 (对齐 ground-truth codebook cluster distribution)
2. **Co-occurrence codebook optimization**: 用 item 共现关系优化 codebook，加入 load balancing 和 uniformity 约束
3. **Balanced computation**: 解决 OneRec-Think "前面 reasoning 太多、后面 SID 生成太少" 的计算失衡

在线 A/B (大规模短视频平台) 确认有效。

### 与当前项目的关联

- 当前 NTP 模型直接预测 SID tokens，没有任何 "reasoning" 步骤
- **与 OneRec-Think 的区别**: OneRec-Think 在 SID 前一次性推理，S²GR 在每个 SID step 前都推理 → 更均匀的计算分配
- 实现简单: 在 SID 序列中插入 `[THINK]` token，预测 SID 前先预测 think token
- Think token 的 contrastive supervision (对齐 cluster distribution) 是新颖的: 给 think token 显式语义 (而非 free-form reasoning)

### 实验设计草案

**实现**:
1. 扩展 SID 序列: `[THINK_1] SID_L1 [THINK_2] SID_L2 [THINK_3] SID_L3`
2. Think token 的 target: codebook cluster 分布的 softmax (contrastive)
3. SID token 的 target: 原始 CE loss
4. Total loss: `L_SID + α * L_think_contrastive`

**评估**: 有/无 think tokens 的 NTP Recall@K

### 关键问题

1. 序列长度翻倍 (3→6 tokens) → 推理成本增加，但每个 token 的计算质量更高
2. Think token 的 contrastive target (cluster distribution) 如何构造
3. 与 IDEA-sid-4 (MTP) 的关系: 两者都在 token level 加入额外监督

---

## IDEA-gr2-0: LLM Reasoning Reranker with Verifiable Rewards

**优先级**: P2
**来源**: GR2 (Meta, arxiv 2602.07774, Feb 2026)
**状态**: 待讨论

### 核心思想

GR2 用 LLM 做 reranking (非 retrieval)，三阶段训练:
1. **Mid-training**: LLM 学习 SID 词表 (≥99% uniqueness)
2. **SFT**: 用更大 LLM 蒸馏 reasoning traces (rejection sampling)
3. **RL**: DAPO with conditional verifiable rewards (防止 reward hacking: LLM 倾向保持原序)

超越 OneRec-Think: **Recall@5 +2.4%, NDCG@5 +1.3%**。

### 与当前项目的关联

- Reranking 阶段的 LLM reasoning 是当前项目的远期目标
- Reasoning trace distillation (大 LLM → 小 LLM) 是实际的工程方案
- **Conditional verifiable rewards** 是对 RL 方法的重要改进 — 解决 reward hacking
- 但需要 NTP 基础模型先成熟，当前优先级低

### 关键问题

1. 无在线 A/B 结果 (仅 offline)
2. 依赖 LLM backbone (≥7B) → 推理成本高
3. 适合作为远期 reranking 方案参考

---

## IDEA-genrank-0: Architecture > Training Paradigm (GenRank Insight)

**优先级**: P1
**来源**: GenRank (Xiaohongshu, arxiv 2505.04180, May 2025)
**状态**: 待讨论

### 核心思想

GenRank 在小红书 Explore Feed (亿级用户) 的深度分析发现: **generative ranking 的提升主要来自架构设计，而非训练范式**。

这个 insight 意义重大: 说明在 NTP + RL 训练范式的讨论之前，更应该关注 **模型架构本身**。

GenRank 提出一种高效 generative ranking 架构，在 **几乎相同计算资源** 下实现用户满意度显著提升。

### 与当前项目的关联

- 直接印证我们的 "Embedding > NTP Model" 哲学的延伸: **Architecture > Training Paradigm**
- 启发: 在投入 RL/DPO 之前，先确保 NTP 架构本身足够好
- GenRank 的具体架构设计需要读论文全文
- 与 IDEA-gr4ad-1 (LazyAR), IDEA-onemall-1 (Query-Former) 相关: 都是架构改进

### 实验设计草案

需要读论文全文获取 GenRank 具体架构，然后对比:
- 当前 vanilla Transformer decoder
- LazyAR (IDEA-gr4ad-1)
- GenRank architecture
- 在相同训练设置下比较 Recall@K

### 关键问题

1. 论文全文细节待获取
2. GenRank 是 ranking 模型而非 retrieval → 是否直接适用于 SID 生成

---

## IDEA-gti-0: Grounded Token Initialization for SID Vocabulary Extension

**优先级**: P1
**来源**: GTI (LinkedIn, arxiv 2604.02324, Apr 2026)
**状态**: 待讨论

### 核心思想

当用 LLM 做 generative recommendation 时，需要将 SID tokens 加入 LLM vocabulary。标准做法是 **mean initialization** — 但 GTI 通过 spectral analysis 证明: mean init 将所有新 token 坍缩到退化子空间，fine-tuning 无法完全恢复。

**Grounded Token Initialization (GTI)**: 用 paired linguistic supervision 将每个新 SID token 映射到 **语义有意义且互相区分的位置**。这是 fine-tuning 之前的一个轻量 pre-fine-tuning 阶段。

多个 benchmark (工业+公开) 上优于 mean init 和 auxiliary-task adaptation。

### 与当前项目的关联

- **直接关联 IDEA-plum-0 (LLM CPT)**: 当用 Qwen3-0.5B 做 CPT 时，需要扩展词表加入 SID tokens
- GTI 回答了一个关键问题: **SID token embedding 怎么初始化?**
- 在 IDEA-plum-0 实验中可以直接对比: mean init vs GTI init 的效果差异
- 实现简单: 在 fine-tuning 之前加一个 alignment 阶段，用 item text 做 linguistic supervision

### 实验设计草案

**前置: IDEA-plum-0 (LLM CPT)**

**实现**:
1. 为每个 SID token 收集 "代表性 item text" (该 cluster 内 top-k item 的 title)
2. 用 LLM 本身 encode 这些 text → 得到 linguistic ground
3. Pre-fine-tuning: 对齐 SID token embedding 到 linguistic ground
4. 然后正常 CPT fine-tuning

**对比**: mean init vs random init vs GTI init

### 关键问题

1. 依赖 LLM backbone path (IDEA-plum-0)
2. 当前 39.5M 模型从零训练，不涉及 vocab extension
3. Linguistic grounding 需要每个 SID code 有对应的文本描述

---

## IDEA-higr-0: Hierarchical Slate Planning (Two-Stage Generation)

**优先级**: P2
**来源**: HiGR (Tencent, arxiv 2512.24787, Dec 2025)
**状态**: 待讨论

### 核心思想

HiGR 将推荐列表生成分为两阶段:
1. **List-level planning**: 生成 slate 的整体 intent/composition (粗粒度)
2. **Item-level decoding**: 在 plan 指导下生成具体 item SID (细粒度)

配合 **multi-objective listwise preference alignment** 优化多目标 (watch time, diversity 等)。

腾讯商业平台 (亿级用户): **watch time +1.22%, video plays +1.73%**, 推理 **5x speedup**。

### 与当前项目的关联

- 当前 NTP 模型逐 item 生成，不考虑 slate 整体 composition
- Hierarchical planning 可以视为 IDEA-tbg-0 (NSP) 的自然延伸: session 预测 → slate 规划
- 5x 推理加速来自搜索空间缩减: 先确定 slate plan，再在受限空间内解码
- 但当前 retrieval 阶段不需要 slate planning → 更适合 reranking 阶段

### 关键问题

1. 属于 reranking 而非 retrieval → 当前阶段优先级低
2. Multi-objective alignment 需要 reward 信号

---

## IDEA-mdgr-0: Masked Diffusion with Parallel Codebook (强化 llada-0)

**优先级**: P2 (与 llada-0 合并追踪)
**来源**: MDGR (Alibaba, arxiv 2601.19501, Jan 2026)
**状态**: 待讨论 — 强化 IDEA-llada-0

### 核心思想

MDGR 是 LLaDA-Rec (IDEA-llada-0) 思路的工业落地:
1. **Parallel Codebook** (非 sequential RQ): 为 diffusion 提供结构基础
2. **Adaptive Masking Training**: 时间和样本维度自适应构造 masking 信号
3. **Warm-up Two-Stage Parallel Decoding**: 先 warm-up 再并行解码

在线广告平台: **revenue +1.20%**。Offline: 超越 10 个 SOTA baselines 最高 +10.78%。

### 与当前项目的关联

- 直接强化 IDEA-llada-0: MDGR 提供了 diffusion GR 的工业验证
- Parallel codebook 与 IDEA-sid-0 (OPQ) 有交叉: 两者都是非 sequential 的 SID 方案
- revenue +1.20% 的在线结果给 diffusion 路线提供了信心
- 但仍然: OPQ + 图解码优先，diffusion 作为备选

### 更新到 IDEA-llada-0

IDEA-llada-0 的工业验证: MDGR 在广告平台上验证了 diffusion GR 的可行性 (+1.20% revenue)。

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P1 | IDEA-gr4ad-1 | LazyAR 解码器 | 与 ARCHITECTURE.md Lazy Decoder-Only 方向一致；扩展 token 数或 beam 后必需 |
| P1 | IDEA-onemall-1 | Query-Former 序列压缩 | 3.7x FLOP 减少，但需要更长序列场景 |
| P1 | IDEA-glide-0 | Soft Prompt Injection | 低成本注入用户表示，Spotify 在线验证 |
| P1 | IDEA-s2gr-0 | Stepwise Reasoning Tokens | 每步 SID 前插入 think token, 在线验证 |
| P1 | IDEA-genrank-0 | Architecture > Training Paradigm | 小红书亿级验证，架构比训练范式更重要 |
| P1 | IDEA-gti-0 | Grounded Token Initialization | LLM SID vocab extension 必需, LinkedIn 验证 |
| P2 | IDEA-onemall-4 | Loss-Free MoE Balancing | 低风险低成本，8 experts 下收益可能有限 |
| P2 | IDEA-oneloc-0 | Context-augmented Attention | 需要 encoder-decoder 架构，当前无落地场景 |
| P2 | IDEA-oneloc-1 | Category Prompt | 需要 encoder-decoder 架构，泛化形式有价值 |
| P2 | IDEA-oxygen-0 | Fast-Slow Thinking | 架构终极形态参考，当前阶段过于复杂 |
| P2 | IDEA-llada-0 / IDEA-mdgr-0 | Discrete Diffusion 解码 | 非自回归新范式，MDGR 工业验证 +1.20% revenue |
| P2 | IDEA-gr2-0 | LLM Reasoning Reranker | Meta 远期方案, 无在线 A/B |
| P2 | IDEA-higr-0 | Hierarchical Slate Planning | Tencent 验证, 属于 reranking 阶段 |
