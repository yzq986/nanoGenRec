# Architecture (模型架构)

[English](architecture.md) | [中文](architecture.zh.md)

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
├── IDEA-onemall-4: Loss-Free MoE → EXP-013 ✅ MoE 8E top-2 已实现
│   └── aux_loss=0.01 工作良好, loss-free bias 为可选微优化
├── IDEA-glide-0: Soft Prompt Injection (user embedding → prefix)
│   └── Spotify 验证, 非惯常收听 +5.4%, 新发现 +14.3%
├── IDEA-oneloc-0: Context-augmented Attention (side-info 注入)
│   └── additive similarity + gating，需 encoder-decoder 架构
├── IDEA-oneloc-1: Category Prompt (邻域 cross-attention 提示)
│   └── 泛化为 interest/category prompt prefix
├── IDEA-oxygen-0: Fast-Slow Thinking (近线 LLM + 实时 GR)
│   └── LLM 推理蒸馏为指令，IGR 意图过滤，SA-GCPO 多场景 RL
├── IDEA-llada-0: Discrete Diffusion (替代自回归)
│   └── 双向注意力 + 自适应生成顺序，解决错误累积
├── IDEA-metaidx-0: 层次化索引 + Test-Time Training (Meta)
│   └── cross-attention + RQ 学层次索引, 中间节点=高质量数据→TTT
├── IDEA-oneranker-0: 统一生成与排序 (Tencent WeiXin)
│   └── Fake Item Token + DC Loss + Value-Aware Decoupling, GMV +1.34%
├── IDEA-orec-think-0: In-Text Reasoning (快手)
│   └── Itemic Alignment + Reasoning Scaffolding + Multi-validity Reward
├── IDEA-reg4rec-0: MoE 并行量化 + 推理自反思 (阿里)
├── IDEA-sif-0: Sample-Level Tokenization + SIF-Mixer (美团)
│   └── HGAQ 237x compression + factored row/col attention, CTR +2.03%
│   └── MPQ 无序 token + PARS/MSRA/CORP 推理增强
├── IDEA-ksa-0: Summary Attention (快手 OneRec 团队)
│   └── O(n/k) KV cache，learnable summary tokens，8x compression + 正交于 GQA/MLA
└── IDEA-vista-0: Two-Stage UIH Summarization (Meta, ICLR 2026)
    └── virtual seed embeddings + QLA O(N) + generative reconstruction loss
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

**优先级**: ~~P2~~ → ✅ 完成 (MoE 基础已实现, loss-free 为可选优化)
**来源**: OneMall §3.2 Decoder-Style Sparse MoE (引用 loss-free mechanism)
**状态**: ✅ MoE 8E top-2 已实现于 EXP-013 S-tier (SparseMoEBlock + 0.01*aux_loss)

> **完成记录 (2026-04-17)**: EXP-013 S-tier 模型已包含 MoE (8 experts, top-2, Switch Transformer aux loss)。`SparseMoEBlock` 实现于 `metrics/sid_prediction.py:69-143`。当前 aux_loss=0.01 工作良好，loss-free bias 机制是可选微优化。模型 17.5M active params (总 39.5M with all experts) 已建立 baseline。

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

> **全文阅读补充 (2026-04-28)**: GLIDE 论文全文阅读后补充关键细节:
> - **Backbone**: Llama 3.2 1B (开源 LLM，~1B params)
> - **SID 配置**: R-KMeans, 4 levels × 256 codes (1024 SID tokens 加入 LLM vocab)
> - **两阶段训练**: (1) 冻结 backbone 仅训 SID token embedding → (2) 冻结 embedding 用 LoRA 微调 backbone。语义 grounding 用 bidirectional translation (SID↔text) 目标
> - **Soft prompt**: 单个 soft prompt token (user embedding → 2-layer MLP → LLM hidden dim)，插在 system instruction 后
> - **R-KMeans vs RQ-VAE**: R-KMeans HitRate@30 比 RQ-VAE 高 9.52%，intra-bucket cosine similarity 0.856 vs 0.657。**R-KMeans 在生产中更优且更稳定** — 支持我们当前 RQ-KMeans 路线
> - **Multi-task controllable discovery**: 用 familiar/unfamiliar control token 区分不同推荐目标，unfamiliar mode Recall@30 +11.8% vs single-task
> - **Beam search 必要性**: 从 sampling 切换到 beam search (30 beams) 带来 +27% Recall@30。coarse SID tokens 不依赖 beam search，但 fine-grained tokens 需要
> - **Debiasing**: cross-surface sampling + exploration upweighting + popularity capping 抑制 popularity bias
> - **21 天 A/B**: ~20M impressions/cell，GLIDE candidates 占 treatment 组 34% 推荐量

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

## IDEA-unirec-0: Chain-of-Attribute (CoA) Prefix Decoding

**优先级**: P1
**来源**: UniRec (arxiv 2604.12234, Apr 2026)
**状态**: 活跃

### 核心思想

UniRec 用贝叶斯定理证明: **有完整特征访问的生成模型 = 判别模型的表达能力**。实际差距仅来自特征覆盖不足。

**Chain-of-Attribute (CoA)**: 在 SID 序列前添加结构化属性 token 前缀 (category, seller, brand)，恢复判别模型的特征交叉能力。

数学保证: `H(s_k | s_{<k}, a) < H(s_k | s_{<k})` — 属性前缀减少每步条件熵，缩小搜索空间，稳定 beam search。

配套: **Conditional Decoding Context (CDC)** (Task-Conditioned BOS + hash-based Content Summary) + **Joint RFT + DPO**。

在线 A/B: **HR@50 +22.6%**, 高价值订单 **+15.5%**。

### 与当前项目的关联

- 与 IDEA-onemall-3 (属性增强 contrastive) 不同: onemall-3 在 embedding 训练时加属性, CoA 在解码时加属性 token 前缀
- 与 IDEA-glide-0 (Soft Prompt) 互补: glide 用用户 embedding 做 soft prompt, CoA 用 item 属性做 hard token 前缀
- 22.6% HR@50 提升极大 → 但需要 item 属性数据 (category/seller/brand)

### 关键问题

1. 属性数据可用性: 需要 category/brand/seller 结构化属性
2. 序列长度增加: 每个 item 从 3 tokens → 5-6 tokens (属性前缀 + SID)
3. 属性预测 vs 属性给定: 推理时是模型自己预测属性还是用 context 提供?

---

## IDEA-gems-0: Multi-Stream Temporal Decoder for Lifelong Sequences

**优先级**: P1
**来源**: GEMs (Kuaishou, arxiv 2602.13631, Feb 2026)
**状态**: 活跃

### 核心思想

GEMs 解决 GR 处理超长用户行为序列 (100K+ interactions) 的计算和注意力偏置问题。将用户行为按时间分为三流:

1. **Recent Stream**: 一阶段实时提取器 → 捕捉即时兴趣动态
2. **Mid-term Stream**: 轻量 indexer + cross-attention → 平衡精度和成本
3. **Lifecycle Stream**: 两阶段 offline-online 压缩模块 → 生命周期建模

三流通过 **parameter-free fusion** 策略合并。快手高并发工业环境部署, 处理 100K+ interactions/user。

### 与当前项目的关联

- 与 IDEA-onemall-1 (Query-Former 序列压缩) 互补: QFormer 压缩整体序列, GEMs 按时间粒度分治
- 直接相关 IDEA-oneloc-4 (序列长度 scaling law): GEMs 提供了如何利用超长序列的工程方案
- 当前序列较短 → 中长期价值: 当序列扩展到 1000+ 时成为关键

### 关键问题

1. 当前数据集序列较短 → 需要先扩展数据或模拟长序列
2. Lifecycle stream 的 offline-online 两阶段压缩实现复杂度高
3. 前置: IDEA-oneloc-4 (序列 scaling law) 确认长序列价值后再投入

---

## IDEA-hpgr-0: Session-Based MIM + Preference-Guided Sparse Attention

**优先级**: P1
**来源**: HPGR (Huawei, arxiv 2603.00980, Mar 2026, WWW 2026)
**状态**: 活跃

### 核心思想

HPGR 指出现有 GR (如 HSTU) 的 "flat-sequence" 假设忽略了用户行为的内在结构:
1. 无法捕捉 session-based 时间层级
2. Dense attention 引入大量噪声, 掩盖真实偏好信号

两阶段解决方案:
1. **Structure-aware Pre-training**: 用 **Session-based Masked Item Modeling (MIM)** 学习层级化 item 表示
2. **Preference-aware Fine-tuning**: **Preference-Guided Sparse Attention** 动态约束注意力到最相关的历史 item

Huawei AppGallery 工业数据集 + 在线 A/B: **超越 HSTU 和 MTGR**, WWW 2026 接收。

### 与当前项目的关联

- 与 IDEA-hstu-0 (Sparse Self-Attention) 不同: hstu 是固定 pattern sparse, HPGR 是 preference-guided 动态 sparse
- Session-based MIM pre-training 可以独立使用 → 为 NTP 提供更好的初始化

### 关键问题

1. Session 切分策略: 时间阈值选择影响大
2. 当前序列短 → preference-guided sparse attention 收益有限
3. Pre-training → fine-tuning 两阶段 pipeline 增加工程复杂度

---

## IDEA-sif-0: Sample-Level Tokenization + Factored Attention (SIF-Mixer)

**优先级**: P2 — SIF 是 ranking 模型而非生成式检索，但 HGAQ 量化和 factored attention 设计有参考价值
**来源**: SIF (Meituan, arxiv 2604.15650, Apr 2026)
**状态**: 待讨论

> **P2 原因**: SIF 的范式 (sample-level tokenization for ranking) 与我们的 SID-based NTP 生成式检索不同。但两个技术有跨范式价值: (1) HGAQ 量化方法可用于压缩丰富的 per-interaction context; (2) factored row/col attention 可用于处理带 side features 的长序列。

### 核心思想

SIF 将推荐序列从 **item-level** 升级到 **sample-level**: 不再用裸 item embedding 表示每个历史交互，而是将完整的 Raw Sample (user+item+context+cross features, 600+ 字段) 量化为 Token Sample。

1. **Sample Tokenizer (HGAQ)**: 将 600+ features 分为 4 semantic groups (user/item/context/cross)，每组自适应切分为 K_g 个 sub-tokens (B=32 fields/token)，每个 sub-token 用 M=3 层 RVQ (V=256) 编码。总压缩: 600×8×32=153,600 bits → 27×3×8=648 bits (**237x compression**)。Label-supervised codebook: 联合优化 CTR loss + VQ commitment loss
2. **SIF-Mixer**: Factored (L+1)×T attention — Token-level Mixer (intra-sample, T sub-tokens 间交互, 捕捉 user-item-context 关系) → Sample-level Mixer (inter-sample, L+1 samples 间交互, 捕捉时序模式) → Token-level FFN
3. **Scaling 行为**: SIF 与 HyFormer 的差距随序列长度 **单调增大** (L=100: +0.0013 → L=2000: +0.0102 GAUC)。item-level 方法在 L=500 饱和，SIF 持续从更多 contextualized interactions 获益

### 关键数据

| 指标 | 数值 |
|------|------|
| Online A/B (Meituan 外卖, 5% 流量, 7 天) | CTR +2.03%, CVR +1.21%, GMV/session +1.35% |
| Heavy users (L≥500) | CTR +3.12%, CVR +1.87%, GMV/session +2.06% |
| Cold users (L<10) | CTR +0.53%, CVR +0.31% (Target Token Sample 也有增益) |
| 数据规模 | 1B+ impressions, 50M+ users, 5M+ items, 600+ feature fields |
| 量化压缩 | 237x (648 bits vs 153,600 bits raw) |
| Model config | 4 SIF Blocks, 8 heads, d0=16, L=1000, T=27 sub-tokens |

### 与当前项目的关联

- **EXP-036 验证了 side features 价值** (time_gap + action_level → R@500 +3.7pp)。SIF 是这个方向的终极形态: 不止 2 个 side features，而是 600+ features 全部编码
- **HGAQ 的 group-adaptive quantization** 与我们 MLP-FSQ tokenizer 的设计理念相近: 都是用分组量化压缩高维表示
- **Factored row/col attention** 与 IDEA-ksa-0 (Summary Attention), IDEA-vista-0 (QLA) 类似: 都是通过分解 attention 降低序列建模成本
- **关键区别**: SIF 是 discriminative ranking model (预测 CTR/CVR)，我们是 generative retrieval model (NTP 生成 SID tokens)。SIF 的 SIF-Mixer 替换不了我们的 causal decoder

### 实验设计草案

**Phase 1 — Rich Context Encoding (可在当前架构内实验)**:
- 在 NTP 训练的 input sequence 中，每个 item 除了 SID tokens 外，注入更多 per-interaction features (e.g., item category, price bucket, user-item 交互频率)
- 用 HGAQ-style group quantization 压缩这些额外 features 为固定长度 tokens
- 评估: R@500 提升 vs 序列长度增加的 tradeoff

**Phase 2 — Factored Attention (远期)**:
- 如果引入多 sub-token per item (当前每个 item 是 3 SID tokens)，可用 token-level + sample-level factored attention
- 与 IDEA-ksa-0 (Summary Attention) 的 block compression 对比

### 关键问题

1. **范式差异**: SIF 是 ranking model，我们是 generative retrieval。SIF 不需要生成 SID tokens，而是预测 CTR/CVR。直接迁移不可行
2. **数据可用性**: 我们的行为数据可能没有 600+ features per interaction — 需要检查数据源
3. **序列长度**: 当前 max_seq_len=512 (~170 items)，SIF 在 L=1000-2000 时优势最大
4. Phase 1 的 per-interaction features 与 IDEA-feat-0/1/2 (time_gap/action_level) 和 IDEA-oneloc-5 (multi-behavior) 有重叠 — 可整合

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
| P1 | IDEA-unirec-0 | Chain-of-Attribute Prefix | 贝叶斯理论 + HR@50 +22.6%, 桥接生成与判别 |
| P1 | IDEA-gems-0 | Multi-Stream Temporal Decoder | 快手 100K+ 序列部署, lifelong GR 工程方案 |
| P1 | IDEA-hpgr-0 | Session-MIM + Preference Sparse Attn | Huawei WWW 2026, 超越 HSTU 的动态稀疏注意力 |
| P1 | IDEA-metaidx-0 | 层次化索引 + Test-Time Training | Meta 数十亿用户部署，hierarchical pruning + TTT |
| P1 | IDEA-oneranker-0 | 统一生成与排序 | Tencent WeiXin GMV +1.34%, DC Loss 可作辅助 loss |
| P1 | IDEA-orec-think-0 | In-Text Reasoning for GR | 快手 +0.159%, multi-validity reward 可先用于 GRPO |
| P1 | IDEA-reg4rec-0 | MoE 并行量化 + 推理自反思 | 阿里在线验证, CORP/MSRA 组件可独立落地 |
| ~~P2~~ ✅ | ~~IDEA-onemall-4~~ | ~~Loss-Free MoE Balancing~~ | ✅ MoE 已实现 (EXP-013), loss-free 为可选微优化 |
| P2 | IDEA-oneloc-0 | Context-augmented Attention | 需要 encoder-decoder 架构，当前无落地场景 |
| P2 | IDEA-oneloc-1 | Category Prompt | 需要 encoder-decoder 架构，泛化形式有价值 |
| P2 | IDEA-oxygen-0 | Fast-Slow Thinking | 架构终极形态参考，当前阶段过于复杂 |
| P2 | IDEA-llada-0 / IDEA-mdgr-0 | Discrete Diffusion 解码 | 非自回归新范式，MDGR 工业验证 +1.20% revenue |
| P2 | IDEA-gr2-0 | LLM Reasoning Reranker | Meta 远期方案, 无在线 A/B |
| P2 | IDEA-higr-0 | Hierarchical Slate Planning | Tencent 验证, 属于 reranking 阶段 |
| P1 | IDEA-genrec-1 | Asymmetric Token Merger | JD SIGIR 2026, prompt 长度减半, 一个 Linear 层, 性能无损 |
| P2 | IDEA-nsgr-0 | Next-Scale 粗到细重排序 | 美团 CTR +2.89%, 但属于 reranking 阶段 |
| P2 | IDEA-sif-0 | Sample-Level Tokenization + SIF-Mixer | 美团 CTR +2.03%, ranking 模型但 HGAQ 量化+factored attention 有参考价值 |

---

## IDEA-metaidx-0: 层次化索引 + Test-Time Training

**优先级**: P1
**来源**: Meta, Efficient Retrieval Scaling with Hierarchical Indexing (arxiv 2604.12965)
**状态**: 待讨论

### 核心思想

Meta 提出为大规模 foundation retrieval model 联合学习层次化索引: 用 cross-attention + residual quantization 构建 hierarchical index，使搜索从根到叶逐层剪枝。关键发现: 中间索引节点对应一组高质量数据子集，在推理时用该子集对模型做 fine-tune (即 "test-time training") 可显著提升 retrieval 质量。已部署于 Facebook + Instagram 的广告推荐，服务数十亿用户。

### 与当前项目的关联

- 当前 NTP beam search 在全 SID 空间上解码，随 item pool 增长 latency 线性增加
- 层次化索引可以替代/增强 prefix tree constrained decoding (IDEA-static-0 CSR 方案)，提供更 semantic-aware 的剪枝
- Test-time training 概念对冷启动 / 时效性场景有价值: 新品类上线时用对应索引节点的 "高质量子集" 做 fast adaptation
- 与 IDEA-earn-0 (Register Token 压缩) 互补: 一个优化搜索路径，一个优化 KV cache

### 实验设计草案

**Phase 1 — 验证层次化剪枝**:
- 在现有 3-token SID 上，用 KMeans 层级 (L1, L2) 作为天然 hierarchical index
- 对比: full beam search vs 逐层 top-K 剪枝 (先在 L1 选 top-32 cluster，再在 L2 展开)
- 评估: Recall@K 损失 vs latency 减少

**Phase 2 — Test-time training**:
- 对每个 L1 cluster 子集做 NTP model 的少量 fine-tune (或 LoRA adaptation)
- 评估: cluster-level Recall 提升 vs fine-tune 成本

### 关键问题

1. 3-token SID 本身层级就浅，层次化剪枝空间有限；更适合扩展到 4+ token SID 后
2. Test-time training 在推荐场景的实时性约束: Meta 用了近线 pipeline，我们需要评估 overhead
3. 与 IDEA-static-0 CSR 约束解码的关系: 互补还是替代？

---

## IDEA-oneranker-0: 统一生成与排序 (Value-Aware Generation-Ranking Integration)

**优先级**: P1
**来源**: OneRanker, Tencent WeiXin Channels (arxiv 2603.02999)
**状态**: 待讨论

### 核心思想

腾讯微信视频号广告提出 OneRanker，将生成阶段和排序阶段深度集成于一个模型: (1) Value-aware multi-task decoupling — 用 task token sequences + causal mask 在共享表示上分离兴趣覆盖和商业价值优化，减少目标冲突; (2) Coarse-to-fine collaborative target awareness — 生成阶段用 Fake Item Tokens 做隐式感知，排序阶段用 ranking decoder 做显式价值对齐; (3) KV pass-through + Distribution Consistency Loss 保证生成和排序的一致性。微信全量部署，GMV +1.34%。

### 与当前项目的关联

- 当前项目 NTP 模型纯做 recall (生成候选)，排序由下游系统完成 → 生成与排序之间存在 gap
- OneRanker 的 Fake Item Tokens 概念可以用于 NTP 训练: 在 beam search 候选上添加 ranking 信号反馈
- Distribution Consistency Loss 可视为一种新型辅助 loss (与 IDEA-onemall-0 contrastive loss 互补)
- 实现需要 ranking stage 的 label 数据 (点击后的 CTR/CVR)，当前数据 pipeline 可能需要扩展

### 实验设计草案

**Phase 1 — Distribution Consistency Loss**:
- 在 NTP 训练中加入 DC loss: 对 beam 候选的概率分布与外部 ranking score 做 KL divergence
- 需要: ranking model 的 score 作为 soft label
- 评估: Recall@K + NDCG (如果有 ranking label)

**Phase 2 — Fake Item Token Awareness**:
- 在 decoder 输入序列中随机插入 "fake item token" (从 in-batch items 采样) 作为负例
- 训练模型区分 real vs fake → 隐式引入 ranking 信号

### 关键问题

1. 当前离线实验无 ranking label，需要从行为数据构造 proxy ranking signal
2. 生成和排序的统一增加模型复杂度，可能影响 NTP 阶段的纯 recall 性能
3. 更适合系统成熟后 (有完整 pipeline) 再引入

---

## IDEA-orec-think-0: In-Text Reasoning for Generative Recommendation

**优先级**: P1
**来源**: OneRec-Think, Kuaishou (arxiv 2510.11639)
**状态**: 待讨论

### 核心思想

快手 OneRec-Think 将对话、推理和个性化推荐统一到一个生成式框架中。核心三步: (1) Itemic Alignment — 跨模态 item-textual 对齐，让模型理解 item 的语义含义; (2) Reasoning Scaffolding — 在 NTP 上下文中激活 LLM 推理能力，不仅预测下一个 SID token，还生成推理链; (3) Multi-validity reward function — 推荐场景的多正确答案特性 (多个 item 都合理) 需要特殊的 reward 设计。"Think-Ahead" 架构允许部署时做实时推理。在快手部署，App Stay Time +0.159%。

与 IDEA-s2gr-0 (Stepwise Reasoning Tokens) 的区别: s2gr-0 是轻量级 think token 插入 (token level)，本 IDEA 是完整的 reasoning chain + multi-validity reward (system level)。

### 与当前项目的关联

- 当前 NTP 是 "隐式预测器"，缺乏可解释性和可控性
- Itemic Alignment 可以直接复用 Qwen3 embedding + SID 的映射
- Multi-validity reward 对 RL 阶段尤为重要 (IDEA-onemall-2 GRPO 目前只用单 ground truth)
- 需要 LLM backbone 支持 reasoning (当前 6-layer decoder 可能不够)

### 实验设计草案

**Phase 1 — Multi-validity reward**:
- 修改现有 NTP eval: 不只看 top-1 匹配，而是 top-K 中有多少个行为相似 item (用 item category / embedding similarity 判定)
- 构造 multi-validity reward: R(generated) = max_sim(generated, {positive_set})

**Phase 2 — Reasoning Scaffolding (post-LLM upgrade)**:
- 需要更大的 backbone (≥1B) 才能支持 reasoning
- 在 SID token 前插入 reasoning prefix (如 user preference summary)
- 与 IDEA-s2gr-0 对比: full reasoning chain vs per-token think token

### 关键问题

1. 当前 6-layer small decoder 无法支持 reasoning，需要等 LLM backbone 升级
2. Multi-validity reward 是更即时可用的 idea，可以先在 GRPO 中应用
3. 部署时 reasoning chain 增加延迟，需要 "Think-Ahead" 异步架构

---

## IDEA-reg4rec-0: MoE 并行量化码本 + 推理自反思

**优先级**: P1
**来源**: REG4Rec, Alibaba (arxiv 2508.15308)
**状态**: 待讨论

### 核心思想

阿里 REG4Rec 将推理 (reasoning) 引入生成式推荐，区别于 IDEA-s2gr-0 和 IDEA-orec-think-0 的关键创新: (1) MoE-based Parallel Quantization (MPQ) — 每个 item 生成多个无序语义 token (而非有序 SID sequence)，构建更大的多样推理空间; (2) Preference Alignment for Reasoning (PARS) — 用推荐领域定制的 reward 来增强推理和反思; (3) Multi-Step Reward Augmentation (MSRA) — 引入未来多步 action 改善泛化; (4) Consistency-Oriented Self-Reflection for Pruning (CORP) — 推理时丢弃不一致的推理路径。

### 与当前项目的关联

- MPQ 的 "多个无序 token" 与当前 "有序 3-token SID" 是根本不同的 paradigm
- CORP 自反思剪枝可以增强现有 beam search: 在 beam 候选上做 consistency check
- MSRA 多步 reward 可以增强 IDEA-onemall-2 GRPO: 不仅看下一步，还看未来 N 步
- 有在线评估，证明工业可行性

### 实验设计草案

**Phase 1 — CORP-style Beam Consistency Pruning**:
- 在 beam search 完成后，对候选做 consistency check: 多次前向推理，剪掉结果不稳定的候选
- 评估: Recall@K 变化 + 生成多样性

**Phase 2 — MSRA Multi-Step Reward**:
- 在 NTP 训练中加入 future reward: L = L_NTP + β * Σ_{t+1..t+3} reward(item_t)
- 评估: 长期指标 (session-level satisfaction) vs 即时 Recall

### 关键问题

1. MPQ 无序 token 与当前有序 SID 不兼容，需要大改 tokenizer pipeline → P2
2. CORP 和 MSRA 是更容易落地的组件
3. 在线评估细节需要看 full paper

---

## IDEA-genrec-1: 非对称 Token Merger (Prompt 侧 SID 压缩)

**优先级**: P1
**来源**: GenRec, JD.com (arxiv 2604.14878, SIGIR 2026)
**状态**: 待讨论

### 核心思想

JD GenRec 提出 Asymmetric Token Merger: 在 prefilling (encoder/prompt) 侧，将每个 item 的 3-token SID 通过线性层投影合并为 1 个 latent vector，使 prompt 长度减少 ~2x；而在 decoding 侧保持原始 SID token 分辨率。这是一种 **训练-推理一致的非对称压缩**: 压缩只应用于 prompt 侧 (用户历史)，解码侧仍生成完整 SID。实验显示 Token Merger 几乎不损失性能 (HR@50: 0.7192 vs 0.7201 without merger)，但 prompt 长度减半 → 支持更长的用户历史序列。

### 与当前项目的关联

- 当前 3-token SID 在 prompt 侧 tripling 了输入长度 → 限制 max sequence length
- Token Merger 是极简的解决方案: `h = Linear(Concat(e(s1), e(s2), e(s3)))`, 一个线性层
- 与 IDEA-onemall-1 (Query-Former 压缩) 方向相同但更轻量: QFormer 需要 cross-attention, Token Merger 只需一个 Linear
- 与 IDEA-earn-0 (Register Token) 互补: EARN 压缩推理侧, Token Merger 压缩 prompt 侧
- 直接可用: 不需要改 SID tokenizer，只在模型 forward 中加一层

### 实验设计草案

**Phase 1 — Linear Token Merger**:
- 在 NTP 模型 forward 中，prompt 侧的每个 item 的 3 个 SID embedding concat → Linear → 1 个 vector
- 保留 special tokens (<sep> 等) 不压缩
- 训练: 从头训练 or fine-tune
- 评估: HR@K, NDCG@K vs baseline (无压缩), 以及训练/推理速度

**Phase 2 — 配合更长序列**:
- Token Merger 释放的序列长度空间用于扩展用户历史 (2x 历史长度)
- 与 IDEA-oneloc-4 (Scaling Law 序列长度) 配合: 验证更长序列在压缩表示下的 scaling 行为

### 关键问题

1. 线性投影可能丢失 SID 层级结构信息 (L1/L2/L3 的 hierarchical semantics)
2. 需要与 IDEA-onemall-1 (QFormer) 做 head-to-head 对比: 哪种压缩更优
3. 前置依赖: NTP 模型基础设施

---

## IDEA-nsgr-0: Next-Scale 粗到细生成式重排序

**优先级**: P2
**来源**: NSGR, Meituan (arxiv 2604.05314)
**状态**: 待讨论

### 核心思想

美团 NSGR 提出一种新的 reranking 范式: Next-Scale Generation。不同于自回归 (逐个生成) 和 one-step (一次生成全部) 的方式，NSG 采用树状粗到细策略: 从用户兴趣出发，"一生二、二生四" 逐步细化推荐列表。核心组件: (1) Next-Scale Generator (NSG) — 每步对当前子集做 priority scoring + pairwise relationship classification (竞争/互补/中性) + binary split; (2) Multi-Scale Evaluator (MSE) — 树结构评估器，在每个 scale 提供指导信号; (3) Multi-Scale Neighbor Loss — 借鉴 GRPO 思想构造相对 reward。在线 A/B (美团外卖): CTR +2.89%, GMV +3.15%。

### 与当前项目的关联

- NSGR 是 reranking 而非 retrieval → 与我们的 NTP retrieval 不在同一阶段
- 但其 **pairwise relationship modeling** (竞争/互补/中性分类) 可以启发 beam search 后的候选重排
- Multi-Scale Neighbor Loss 与 GRPO 类似的 relative reward 思路可以迁移
- SID + HSTU 作为 user interest 提取器是共性组件
- 更适合系统上线后的 reranking 阶段引入

### 实验设计草案

**Phase 1 — Pairwise Relationship Reranking**:
- 在 beam search 输出的 top-K 候选上，做 pairwise 竞争/互补分类
- 用 NSGR 的 asymmetric influence weight 公式重排
- 评估: list-wise diversity + precision

**Phase 2 — Full NSGR Pipeline**:
- 作为 retrieval (NTP) → reranking (NSGR) 的两阶段 pipeline
- 需要训练 MSE evaluator

### 关键问题

1. 当前处于 retrieval 阶段, reranking 是后续工作 → P2
2. NSGR 需要 evaluator 模型 (额外训练成本)
3. 在线 NSGR 只在 candidate set ≥20 时有显著优势, 如果 beam=50 则值得考虑

---

## IDEA-cobra-0: Cascaded Sparse-Dense 生成式检索 (SID + Dense Vector 联合生成)

**优先级**: P2 (NTP 后)
**来源**: COBRA, Baidu (arxiv 2503.02453, Mar 2025)
**状态**: 待讨论

### 核心思想

COBRA 发现纯 SID 生成存在信息损失（量化丢细粒度），纯 dense retrieval 缺乏语义结构。提出 **级联 sparse-dense 统一生成**:

1. **Cascaded Representation**: 每个 item 表示为 (sparse_ID, dense_vector)。Sparse ID 由 RQ-VAE 生成，dense vector 由 end-to-end 可训练 text encoder 生成
2. **Sequential Modeling**: Transformer decoder 的输入序列为 [e1, v1, e2, v2, ...], 每个 item 占两个 token position (SID embedding + dense vector)
3. **Probabilistic Decomposition**: P(ID_{t+1}, v_{t+1}|S_{1:t}) = P(ID_{t+1}|S_{1:t}) · P(v_{t+1}|ID_{t+1}, S_{1:t})
4. **Training**: L_sparse (CE on SID) + L_dense (contrastive on dense vector)
5. **Inference — Coarse-to-Fine**: 先 beam search 生成 M 个 SID → 每个 SID append 到序列 → 生成 dense vector → ANN 检索 top-N items
6. **BeamFusion**: 融合 beam score (SID 置信度) 和 cosine similarity (dense 精度): Φ = Softmax(τ·φ_ID) × Softmax(ψ·cos(v̂, a))

**核心结果**:
- Beauty R@10: 0.0725 (TIGER 0.0648, +12%)
- Toys R@10: 0.0781 (TIGER 0.0712, +10%)
- Industrial (Baidu Ads, 5M users, 2M ads): R@500 0.3716 (vs w/o Dense 0.2709 +37%, vs w/o ID 0.2466 +51%)
- **Online A/B**: conversion +3.60%, ARPU +4.15% (200M+ DAU)
- Dense + Sparse 互为补充: 去掉任一都大幅降低

### 与当前项目的关联

- COBRA 的核心 insight: SID (离散) 捕获 categorical/coarse 语义，dense vector (连续) 捕获 fine-grained 细节 → 两者互补
- 与我们的 beam search 推理直接相关: 当前 beam search 只返回 SID → item mapping，COBRA 额外生成 dense vector 做二次精排
- BeamFusion 机制可以应用于我们的推理: beam score × item embedding similarity
- **但架构改动大**: 需要 (1) 在每个 item token 后添加 dense vector position, (2) 新增 dense prediction head, (3) 推理时增加 ANN 步骤
- 与 IDEA-genrec-1 (Token Merger) 冲突: Merger 减少 token 数, COBRA 增加 token 数
- 200M+ DAU online A/B 是强验证 → 架构方向有长期价值

### 实验设计草案

**Phase 1 — BeamFusion (不改架构)**:
- 保持当前 NTP 生成 SID → beam search 输出多个 candidate SID
- 对每个 candidate SID，查找对应 item 的 text embedding
- Rerank: beam_score × cosine(user_embedding, item_embedding)
- 不需要训练新模型，只需推理时加 reranking 步骤

**Phase 2 — Full Cascaded Architecture**:
- 在 NTP 序列中每个 item 后添加 dense vector token
- 新增 dense prediction head + contrastive loss
- 修改 inference pipeline: SID generation → dense vector generation → ANN

### 关键问题

1. Phase 1 (BeamFusion reranking) 几乎零成本 → 可以最快验证 dense refinement 的价值
2. Full cascaded 需要序列长度翻倍 → 训练成本翻倍
3. 与 IDEA-flexcode-0 的区别: FlexCode 在 tokenizer 层融合 CF+semantic, COBRA 在 generation 层融合 sparse+dense
4. NTP 后阶段考虑 full architecture change → P2, 但 Phase 1 BeamFusion 可以更早尝试

---

## IDEA-ksa-0: Summary Attention (Kwai Summary Attention)

**优先级**: P1
**来源**: KSA Technical Report (Kuaishou OneRec Team, arxiv 2604.24432, Apr 2026)
**状态**: 待讨论

### 核心思想

Kwai Summary Attention (KSA) 是快手 OneRec 团队提出的新型注意力机制，在 Full Attention 的 O(n) KV cache 和 Linear/SWA 的 O(1)/O(w) 之间开辟了 **O(n/k) 路径** — 通过 learnable summary tokens 实现语义级别的序列压缩。

**机制**:
1. 将输入序列切分为固定大小的 chunk (默认 k=8 tokens/chunk)
2. 每个 chunk 末尾注入一个 learnable summary token
3. Text tokens 只看 local sliding chunk (相邻 chunk) + distant summary tokens
4. Summary tokens 只看当前 chunk 内的 text tokens → 蒸馏该 chunk 的语义

**Hybrid-KSA**: 3:1 混合比例 (3 层 KSA + 1 层 Full Attention)，保持全局精确注意力的同时大幅降低平均 KV cache。

### 关键实验数据

| 指标 | Full Attention | Hybrid-KSA | 提升 |
|------|---------------|-----------|------|
| RULER-128K (CPT) | 65.86 | 71.67 | +5.81 |
| RULER-128K (Scratch) | 48.75 | 65.35 | +16.60 |
| KV Cache @128K | 18.6 GB | 7.5 GB | 2.5x 减少 |
| MMLU (CPT) | 71.83 | 70.50 | -1.33 (微降) |
| GSM8K (Scratch) | 48.29 | 59.14 | +10.85 |

**核心优势**:
- **正交于 GQA/MLA**: KSA 压缩 token 数，GQA 压缩 head 数，MLA 压缩 embedding dim → 三者组合可达 8x 进一步压缩
- **保留长程依赖**: 与 SWA (完全丢弃窗口外) 和 Linear Attention (固定 state 有损压缩) 不同，summary tokens 以可解释方式保留远距信息
- **开源**: https://github.com/Kuaishou-OneRec/KSA

**CPT 训练策略**: 三阶段 (1) Summary token adaptation: 独立 Q/K/V 权重 + 多粒度蒸馏 (layer-wise MSE + distribution-wise KL + objective-wise LM loss); (2) Parameter annealing: 线性插值将独立权重融入主 LLM 权重; (3) Full parameter tuning + 序列长度扩展。

### 与当前项目的关联

- **直接适用于 GR 长序列**: OneRec 团队明确表示下一步是 "Unifying with OneRec — 构建基于 KSA 的生成式推荐基础模型，将超长用户行为序列压缩为层次化 summary tokens"
- **解决 IDEA-oneloc-4 (序列长度 scaling) 的计算瓶颈**: 当前 EXP-015 显示 scale up 模型收益递减，但序列长度 scaling 尚未验证 — KSA 可以让序列长度 8x 增长而 KV cache 只增长 1x
- **与 IDEA-hstu-0 (Sparse Attention) 互补**: HSTU 使用稀疏 attention pattern，KSA 使用 summary compression — 可以组合
- **与 IDEA-earn-0 (Register Token) 互补**: EARN 在输入首尾放 register token 减少后期层的 KV cache，KSA 在每个 chunk 放 summary token 减少全序列 KV cache — 方向正交
- **与 IDEA-gems-0 (Multi-Stream) 互补**: GEMs 用多 stream 切分超长序列，KSA 用 summary tokens 压缩 — 可以先 KSA 压缩再 multi-stream

### 实验设计草案

**Phase 1 — 在当前 NTP 模型中验证 summary attention**:
- 修改 `ntp/model.py` 的 attention layers: 将 3/4 层替换为 KSA (summary attention)，保留 1/4 为 full attention
- Chunk size k=8 (对应 8 个 SID token ≈ 2-3 个 item，合理的 item-level summarization 粒度)
- 对比: 原始 full attention vs Hybrid-KSA，在 14d 训练窗口上评估 PPL/R@500
- 关注: 短序列场景 (我们当前 avg 21-30 items/user ≈ 63-90 tokens) KSA 是否有优势或退化

**Phase 2 — 序列长度扩展验证**:
- 利用 KSA 的 KV cache 减少，将训练序列长度从当前 ~200 tokens 扩展到 1000+ tokens
- 验证 IDEA-oneloc-4 Phase 2 的假设: 更长序列是否改善 Recall

### 关键问题

1. 当前序列很短 (~90 tokens)，KSA 的优势在长序列才显现 — Phase 1 可能看不到效率收益
2. Summary token 在 SID 语义空间中是否能有效压缩? LLM 的 summary 是语义级压缩，SID 的 summary 可能是行为模式级压缩
3. CPT 三阶段训练策略是否适用于从零训练的小模型 (17.5M params)?
4. 与 Flash Attention 的兼容性 — KSA 的 mixed attention mask 需要 custom kernel
5. **优先级判断**: 在序列长度 scaling 成为瓶颈之前，KSA 的优先级偏低 → P1 但排在 RL 对齐 (EXP-037/038) 之后

---

## IDEA-vista-0: Two-Stage UIH Summarization + Quasi-Linear Attention (VISTA)

**优先级**: P1
**来源**: VISTA (Meta, arxiv 2510.22049, ICLR 2026)
**状态**: 待讨论 — 前置：序列长度 scaling (IDEA-oneloc-4 Phase 2) 成为瓶颈后再实施

### 核心思想

VISTA 将传统的 target attention（从候选 item 到全部 user history）分解为两阶段:

1. **Stage 1 — UIH Summarization**: 将超长用户交互历史（up to 1M items）压缩为 ~128 个 virtual seed embeddings
   - Virtual seeds: 随机初始化的共享参数，通过 self-attention 与 UIH 交互更新
   - **Quasi-Linear Attention (QLA)**: φ-linear attention with SiLU activation → O(N) 复杂度
     - `O[S] = φ(Q[S]) · φ(φ(K[S])^T · V[S])` — 利用结合律从 O(N²) 降到 O(N)
     - Candidates 不能互相 attend（防 label leakage）— 通过 diagonal self-attention 项解决
     - Custom Triton kernel 实现
   - **Generative Reconstruction Loss**: causal decoder 从 seeds + UIH 前缀重建 UIH 下一项 (off-by-one MSE)
     - `L_reconstruct = Σ ||v_i - u_{i+1}||²` — 迫使 seed embeddings 最大保留历史信息
   - Summarization 仅在训练时运行，embeddings 缓存到 KV store

2. **Stage 2 — Target-Aware Attention**: 标准 O(N²) transformer 在 compact summaries (~128 tokens) 上做候选-历史交互
   - 因为 summary 很短，O(N²) 完全可接受

### 关键实验数据

**Industrial-Scale (Meta production)**:
- 训练规模: O(10B) examples/day
- 序列长度: avg 7K, max 16K (deploy 12K)
- 配置: 3-layer self-attention, 3-layer target-aware, 128 seeds, 256 embedding dim

| 模型 | C-Task Eval NE | E1-Task | E2-Task | E3-Task |
|------|---------------|---------|---------|---------|
| HSTU (baseline) | — | — | — | — |
| VISTA | -0.40% | -1.19% | -2.98% | -2.23% |
| VISTA-w/o-Recon | -0.29% | -1.29% | -3.00% | -2.21% |

QLA vs softmax attention: 序列从 6K→16K，层数 3→5，QPS +5%，NE -0.13%

**Online A/B (Meta production, 5% traffic, 15 days)**:
- C-Task: **+0.5%** (main consumption)
- O1-Task: +0.2%, O2-Task: +0.04%
- **94% reduction in inference GPU resource** (通过缓存 embedding 避免重复计算)
- Embedding delivery: 2-hour cadence updates, O(100TB) storage, geo-replicated KV store

### 与当前项目的关联

- **长序列 scaling 的关键技术**: 当前序列很短 (~90 tokens)，但 IDEA-oneloc-4 Phase 2 需要扩展 → VISTA 提供了工业验证的 O(N) 方案
- **与 IDEA-ksa-0 (Summary Attention) 深度互补**: KSA 在 attention layer 内部做 summary compression (chunk-level)，VISTA 在模型外部做 UIH summarization (user-level) — 可以组合: VISTA stage-1 压缩到 128 tokens → KSA 在 stage-2 内部进一步压缩 KV cache
- **与 IDEA-onemall-1 (Query-Former) 互补**: Query-Former 用 cross-attention 压缩当前 session，VISTA 用 self-attention + seeds 压缩 lifelong history — 两者解决不同尺度问题
- **Embedding delivery system 参考价值**: 当部署 GR 时，user embedding 缓存架构 (2-hour cadence + KV store) 是生产必需
- **Generative reconstruction loss**: 一种新的自监督信号，可以增强 user representation 的信息保留

### 实验设计草案

**Phase 1 — QLA 注意力验证** (低成本):
1. 在当前 6-layer NTP 模型中，将 1-2 层 softmax attention 替换为 QLA (SiLU-based φ-linear)
2. 对比: PPL/R@500 精度影响 + 训练速度
3. 验证 QLA 在短序列下是否退化 (理论上 O(N) vs O(N²) 差异在短序列不明显)

**Phase 2 — Virtual Seed Summarization** (需要长序列数据):
1. 在 NTP 模型前增加 stage-1 summarization module (3-layer self-attention + 128 seeds)
2. 训练: 对超长用户序列 (扩展到 500+ tokens) 做 UIH summarization + reconstruction loss
3. 评估: R@500 vs baseline，summary embedding 的信息保留度

**Phase 3 — Embedding Caching** (部署阶段):
1. 将 stage-1 离线计算，缓存 user summary embeddings
2. Stage-2 在线推理仅处理 128 summary + candidate
3. 评估: 延迟/QPS 改善

### 关键问题

1. 当前序列太短 (~90 tokens)，VISTA 的优势需要 1000+ tokens 才能显现 — 依赖序列长度扩展
2. QLA 的 SiLU activation 是否在 SID token 空间有效? Meta 的验证是在 item embedding 空间
3. Virtual seeds 数量 (128) vs 我们当前用户序列长度 (21-30 items) — 可能需要调小
4. Generative reconstruction loss 在 SID 空间的适用性: SID 是离散 token，直接 MSE 不适用 → 需要在 embedding 空间做
5. 存储成本: 128 seeds × 256 dim × fp16 = 64KB/user — 千万用户级 = 640GB，需要评估可行性

---

## IDEA-glorank-0: GloRank — SID-as-Global-Action-Space for 生成式 Reranking

**优先级**: P2 (我们当前是 retrieval 阶段, 无 reranker stage)
**来源**: GloRank (Kuaishou + UCSD + CityU HK, arxiv 2604.25291, Apr 2026)
**状态**: 待讨论 — 主要作为未来 reranking stage 的参考; 其中 "global identifier + 2-stage SFT→RL" 训练范式可借鉴

### 核心思想

List-wise reranking 传统实现是从 N 个候选中按"位置索引 (k-th position)"选择 — 但这造成**语义不一致的 action space**: 同一 output logit 在不同样本下代表不同 item (取决于输入 candidate 的顺序)。作者给出严格理论分析:

**数学核心 (Proposition 2.1)**: 假设 target 固定为 item `r*`, candidate 随机排列 σ, 则每行输出参数 `w_j` 收到的 label-dependent gradient 的 "mapping-induced variance" 下界:

```
Var_σ(g_j,loc) ≥ (1/N)(1-1/N) |μ_j|²_2   > 0   (不可消除)
```

即便 hidden state 完全稳定，这个 variance 永远存在，因为"target 到 output row 的映射" 随 σ 变化。

**解决方案**:
1. **Global action space**: 用 Semantic IDs (SID) 把 item 映射到固定全局 token 词表，reranker 输出 SID token 序列而非 local index
2. **Corollary 2.2**: 在 global 空间下, `Var_σ(g_glo) = Var_σ(h^t_σ)`, 完全消除 mapping-induced variance
3. **两阶段训练**:
   - **SFT pre-training**: 用高质量 reference list 做 behavior cloning
   - **RL post-training**: 直接优化 list-wise reward
4. **Constrained decoding**: 在 candidate 集合上构建 generation trie，确保输出有效且不重复

### 与当前项目的关联

**当前项目没有独立的 reranker stage** (end-to-end generative retrieval)，所以 GloRank 主体应用场景不适用。但有三个 insight 对我们有借鉴：

1. **"Global identifier" 范式本质上是我们 retrieval 的原生设计** — NTP 输出就是全局 SID token，天然没有 mapping-induced variance 问题。这个理论验证了我们路线的正确性
2. **Two-stage SFT→RL 训练** — 和 EXP-020 (NTP+DPO joint) 思路一致; GloRank 用 list-wise reward 做 RL，对标 IDEA-rankgr-0 / IDEA-gr4ad-3
3. **Constrained generation trie over candidates** — 我们 `SIDTrie` + `constrained_beam_search` 已有等价实现; GloRank 是"在给定 N 个候选之上做受限生成"，可复用

### 实验设计草案

**当前阶段不执行**, P2 存档. 若未来引入 reranker stage:

**Phase 1**: 用 EXP-020 checkpoint 输出 beam=500 候选 → 训练小型 SID-based reranker → SFT (top-10 by reward) → RL (list-wise NDCG)

**Phase 2**: 对比 local-index vs global-SID reranker 的 gradient variance 和训练稳定性，验证论文理论

### 关键问题

1. 当前 R@500=66.2%，是否真需要 reranker stage? ROI 不明确
2. Reranker backbone 计算开销 vs 在线 latency
3. List-wise reward 在我们 NTP 数据下定义困难 (缺 dwell time / interaction rate 标注)

### 相关 idea

- IDEA-oneranker-0 (Tencent WeiXin 统一 generation+ranking): GloRank 是 rerank-only 的 SID 版本
- IDEA-rankgr-0 (淘宝 Listwise DPO + Rescore): Listwise RL 套路
- IDEA-gr4ad-3 (GR4AD RSPO): NDCG-inspired RL reward
- IDEA-nsgr-0 (Meituan Next-Scale): 另一种粗到细生成式 reranking

---

## IDEA-a2gen-0: A2Gen — Action-Aware Generative Sequence (输出用户动作序列)

**优先级**: P1
**来源**: A2Gen (Kuaishou Beijing, arxiv 2604.25834, SIGIR 2026, 400M DAU 全流量部署)
**状态**: 待讨论 — 和已完成的 IDEA-feat-1 (action 输入特征) 是两个不同方向, A2Gen 把 action 作为输出

### 核心思想

传统推荐模型把 video 当作"单一 item + binary 标签"，忽略了：

1. **短视频多片段异质**: 用户在不同片段上态度不同（Like 的是 Messi, 不是 Ronaldo）
2. **Action timing 区分意图**: Like 在视频高潮时段 → Follow 率 ↑3.3×, Collect 率 ↑1.52×; 随意早期 Like 多为噪音
3. **Action 顺序差异**: `Follow→Like` 序列 vs `Like→Follow` 用户 watch time 差 1.28×, comment rate 差 1.66×

**A2Gen 架构**: 对每个候选 item 生成完整的 **(action_type, timing)** 序列, 而非预测 binary 标签:

- **CAM (Context-aware Attention Module)**: MHA + 把 item context 融进 query + 每 head 过 gating 学 task-specific 重要度 + MLP 后处理
- **HSE (Hierarchical Sequence Encoder)**: 两层 — Action-dim (每 item 内的 action 序列) → Item-dim (用户历史 item 序列)，嵌套 CAM
- **AAG (Action-seq Autoregressive Generator)**: 自回归生成 `{(A_i, T_i)}`, T_i 用回归预测相对时间占比

**Loss**:
```
L = α·L_cls(action type 多分类) + β·L_reg(timing MSE) + γ·L_order(禁止反序 max(T_p - T_q, 0)²)
```
默认 α=1, β=1, γ=0.1.

**在线落地四个累加策略 (Kuaishou 400M DAU 全流量)**:
| 策略 | Watch Time | Interaction | LT7 |
|------|-----------|-------------|-----|
| Model Replacement (A2Gen 替换 PLE) | +0.11% | +2.1% | — |
| Action Timing Aware (靠后 Like 升权, 过滤早期随意 Like) | +0.13% | +3.5% | +0.12% |
| Action Sequence Aware (`Follow→Like` 升权) | +0.10% | +1.4% | — |
| Action Timing Distribution Aware (峰值附近样本升权) | — | +1.1% | +0.042% |
| **累计** | **+0.34%** | **+8.1%** | **+0.162%** |

### 与当前项目的关联

**这是我们 action level 研究线的重要扩展**:

- **IDEA-feat-1 (ActionType 输入特征, ✅ EXP-036)**: 把 action 作为**输入**注入 NTP (L0/L1/L2 按 behavior type 分级 + time_gap)
- **A2Gen**: 把 action 作为**输出**, 生成 action 序列而非 SID 序列

**关键差异**:
- A2Gen 是 reranking stage 模型（上游已给 N 个候选），不是 retrieval
- A2Gen 不生成 SID, 生成的是"动作序列"，输入 item ID 是原子 ID
- 我们的 NTP 是 retrieval, 直接生成 SID

**最可移植的部分**: 在线策略 2/3/4 本质是 **item-level action statistics 作为后处理 ranking signal**，可独立于 A2Gen 架构实现，不需要输出架构改动。

### 实验设计草案

**Phase 1 — Action statistics aggregation (数据端, 独立做)**:

新增 `data/item_action_stats.parquet`:
| item_id | late_like_ratio | follow_like_seq_rate | timing_peak_alignment |
|---------|----------------|---------------------|----------------------|

从 NTP 训练数据计算 (user 行为 timestamps + behavior types 我们已有)。

**Phase 2 — 输入端 action statistics 特征 (类似 feat-1 扩展)**:

1. 给 `NTPModel` 加三个 `nn.Embedding` (bucket 化的三个统计量)
2. `embed_with_features(tokens, positions, time_gaps, action_levels, late_like_bucket, follow_like_bucket, timing_peak_bucket)`
3. 对比 baseline (feat-0/1/2 only) vs baseline + A2Gen 统计特征
4. 预期收益: R@500 +0.3~0.8pp (A2Gen 单策略线上提升幅度)

**Phase 3 — 输出端 action sequence 预测 (大改动, 远期)**:

Vocab 扩展 + 分 SID-pred / action-cls / timing-reg 三个 head. 收益未必比 Phase 1+2 更大，留远期。

### 关键问题

1. **"Action" 在我们数据集里的含义**: 我们数据来自商品行为 (click/cart/purchase), 没有 Like/Follow/Collect 的对应 — 先 audit 当前 behavior 类别
2. **Action timing 是否有信号**: 我们 event 有 timestamp, 但"在 session/item 内的相对时间"不存在于当前 schema, 要确认
3. **与 IDEA-feat-1 正交性**: feat-1 是 action type embedding (L0/L1/L2 per item); A2Gen 是 item-aggregated action statistics. 两者可叠加
4. **Phase 1 工作量**: 重做数据 pipeline + 3 个聚合统计量 ≈ 1 周; 只做 Phase 2 baseline 可压到 2-3 天
5. **商品 retrieval 场景收益上限未知**: A2Gen 线上数字来自短视频, 我们场景可能较小

### 相关 idea

- IDEA-feat-1 (ActionType 输入特征, ✅ EXP-036): 输入端 action vs A2Gen 输出端, 正交
- IDEA-onelive-0 (OneLive BOS 时间注入): 也用 action + 时间, 但 feature level
- IDEA-lac-0 (LAC 延迟 action): action 作为 context token
- IDEA-mbgr-0 (MBGR 多业务 GR): 多 task 联合, Phase 3 多任务思路类似

---

## IDEA-cadet-0: Self-Gated Attention (Representation + Q/K 三级 Gating)

**优先级**: P1
**来源**: CADET (LinkedIn, arxiv 2602.11410, Feb 2026, 生产部署)
**状态**: 待讨论 — 注意力变体改动局部, 可作为 EXP-020 baseline 上的 drop-in 升级

### 核心思想

CADET 是 LinkedIn homefeed 广告 CTR 的 decoder-only transformer，**线上 A/B +11.04% CTR** 对比 LiRank baseline (DCNv2 + 序列 encoder hybrid ensemble), 在十亿 member 平台主流量部署。核心创新 5 项中，**Self-Gated Attention** 是最独立、最可移植的组件，直接针对训练不稳定和"attention sink"病态行为。

区别于已有工作的 output-level gating (HSTU / OneLive 的 gated attention 都是在 output 侧乘门控)，CADET 把 gate 放到 attention 输入端和 Q/K 投影后，形成**三级 gating 结构**:

**第一级 — Representation-level gate** (在 attention 计算之前对 token 表征做特征选择):
```
Gate(X) = σ(W_X^gate · X)
X̃ = X ⊙ Gate(X)
```
作用：抑制噪声维度, 改善 activation scaling 和 gradient 条件数。

**第二级 — Query gate** (调制 Q):
```
Gate(Q) = σ(W_Q^gate · Q)
Q̃ = Q ⊙ Gate(Q)
```

**第三级 — Key gate** (调制 K):
```
Gate(K) = σ(W_K^gate · K)
K̃ = K ⊙ Gate(K)
```

效果：约束 Q·K 的 dot-product 幅值，防止个别 dominant token 垄断注意力（即 **mitigate attention sink**）。论文报告这个设计是训练稳定性的关键。

**其他 4 个 CADET 创新 (非主推但值得了解)**:
- **Context-Conditioned Decoding Block**: K 个 prediction head, 按 ad position bucket 分桶 (k=1 position1, k=2 position2-4, k=3 position5+)，解决 post-scoring 特征 (广告位置) 在 inference 时不可知的问题. 与 IDEA-ocarm 的 feature leakage 问题类似, 但解法是 multi-head 输出而非 distillation
- **Timestamp-based RoPE**: 用 Unix timestamp 替代 sequence position 做 RoPE 旋转, θ_i 用 `φ_min/Δt_max · base^(2i/d)`, 覆盖秒到月的时间尺度. **与 IDEA-torope-0 (Roblox Rotate Both Ways) 高度重叠**, torope 保留 order RoPE 同时加时间 rotation, CADET 完全替换为时间 — 两种实现风格
- **Session-aware training mask**: 训练时 mask 掉 `t_j > t_i - Δ_delay` 的 key, 强制模型不依赖 "同 session 延迟到达的事件" (online tracking delay 场景). 对我们离线训练-离线评估没有 train-serve skew, 不直接适用
- **Production engineering**: tensor packing + sequence chunking + Flash Attention kernel for multi-item scoring

### 与当前项目的关联

- **`ntp/model.py::_transformer_forward`** 是直接改动目标. 我们当前用标准 PyTorch `nn.MultiheadAttention`, 实现 CADET self-gated attention 只需:
  1. 新增 3 个 `nn.Linear(d_model, d_model)` 作为 `W_X^gate / W_Q^gate / W_K^gate`
  2. 在 attention 计算前对 (X, Q, K) 分别做 sigmoid gate ⊙ 乘
  3. 其余 attention 保持不变
- 参数量增加: 3 × d_model² = 约 3× 单 attention 层参数 (一个 Q/K/V 投影是 d_model², 3 个 gate 多出 3 × d_model²). 对我们 45.8M baseline (d_model=384), 每层增约 0.44M, 8 层共 ~3.5M, **+8% 参数**
- 如果 CADET 的"attention sink 缓解"在我们 SID 生成场景也成立, 可能带来训练稳定性 + 最终 PPL 改善
- **与 IDEA-hstu-0 (HSTU sparse attention) 的关系**: HSTU 是一种自定义 gated attention (output-side), CADET 是 input/Q/K-side gating. 论文在 3.2.1 明确区分 "Unlike output-level gating used in prior work [14, 22]" (22 是 HSTU). 两者可能互补或重叠, 需 ablation
- **与 IDEA-onelive-0 的关系**: OneLive 用 gated attention 注入 BOS 时间信息, 是 feature-injection 用途; CADET 的 gating 是注意力稳定性用途. 正交, 可组合

### 实验设计草案

**EXP-NNN — CADET Self-Gated Attention 集成**:

**Phase 1 — 最小改动验证 (1 天实验)**:
1. 在 `ntp/model.py` 新增 `SelfGatedMultiheadAttention` 类 (继承或替代现有 attention)
2. 加 CLI 开关 `--use_self_gated_attention`
3. 在 EXP-020 最优 config (exp020-hard-lam03) 上 **relaxed re-train**: 只改 attention, 其他 hyperparameter 保持
4. 对比:
   | Config | PPL | R@10 | R@500 | Notes |
   |--------|-----|------|-------|-------|
   | baseline (exp020) | 16.3 | 14.1% | 66.2% | no gating |
   | + self-gated (rep only) | ? | ? | ? | 只加第一级 |
   | + self-gated (rep+Q+K, full) | ? | ? | ? | 完整 CADET |

**Phase 2 — 与现有 attention 变体对比**:
- baseline vanilla vs HSTU-style gated-output vs CADET 三级 gated-input
- 加 attention sink 诊断指标 (最大 attention weight / uniform attention 偏离度) 来验证 CADET 的 "mitigate attention sink" 声明

**Phase 3 (可选) — 叠加 OneLive Gated BOS**:
- 如果 CADET 的 self-gating 和 OneLive 的 output-side gating 都有正向收益, 尝试组合

### 关键问题

1. **CADET 的收益来自 CTR prediction 还是通用架构改进?** LinkedIn 场景是 binary CTR (1 个 action per impression), 我们是 SID token generation (3 token per item). 注意力稳定性应该 generic, 但幅度可能不同
2. **参数量 +8% 是否带来 overfitting?** 我们数据规模比 LinkedIn 小, 需要验证 config 和正则
3. **与 FlashAttention / SDPA 兼容**: `F.scaled_dot_product_attention` 不直接支持 Q/K 后的 gating hook. 可能需要 custom kernel 或回退到手写 attention (吞吐可能掉)
4. **训练动态对比**: CADET 声称 "reliable convergence at scale", 我们 baseline 已稳定, 增益可能主要在 PPL / 最终指标而不是训练稳定性
5. **Timestamp-based RoPE 是否一并采纳?** 我们有 IDEA-torope-0 (Roblox) 已覆盖时间 RoPE 话题, 如果未来实现 torope, 可以直接用 CADET 风格作为一个 sub-option
6. **Context-conditioned decoding block** 在我们 retrieval 场景不适用 (没有 post-scoring contextual feature 像 ad position 那样); 但如果未来扩展到 ads, 这是 ready-made

### 相关 idea

- IDEA-hstu-0 (ULTRA-HSTU Sparse Attention): HSTU output-side gating, CADET input+Q+K-side gating, 论文中明确区分
- IDEA-onelive-0 (OneLive Gated BOS 时间注入): gated attention 用于特征注入, 正交
- IDEA-ksa-0 (Summary Attention): KV cache 压缩, 不冲突
- IDEA-vista-0 (QLA): attention 形式替换, 有竞争关系
- IDEA-torope-0 (Roblox Time-and-Order RoPE): 时间 RoPE 另一实现, CADET 的 timestamp RoPE 是激进替换方案
- IDEA-ocarm-0 paper (reference only, 2604.25839): 用 distillation 解决 post-scoring signal 缺失; CADET 用 multi-head output 解决, 两种互补思路

---

## IDEA-recochain-0: 单 Transformer 里融合生成式检索 + target-aware 重排 (KV-cache 复用)

**优先级**: P2
**来源**: RecoChain — Harmonizing Generative Retrieval and Ranking in Chain-of-Recommendation (Kuaishou Jiangxia Cao, arxiv 2604.25787)
**Tier**: B (Kuaishou 通讯作者, 离线 TAOBAO-MM, 无 A/B; short paper "work in progress")
**状态**: 待讨论

### 问题: GR 的 beam candidate 难排序

OneRec / TIGER 这类 GR 通过 hierarchical beam search 生成 `K=256` 个候选 SID, 但 **next-item-agnostic** 建模方式无法精确估计 "这 256 个里哪 10 个最好"。工业两段 pipeline 的做法是:

- Retrieval (GR): `P(next_item | user_hist)` → 粗召 256 个
- Ranking (DIN/SIM/TWIN/RankMixer): `P(click | user, item_feat)` → 精排 top10

现有 GR 蒸馏 ranking 作 reward model (OneRec-V2 / Climber / ReCast 都是这条路线), 但根本的 target-item aware searched-sequence modeling (SIM 的核心) 没接到 GR 里, 所以 ranking 能力仍差。

### RecoChain 的做法: 一个 decoder 两段用

关键是 **把 SIM 的 target-aware 行为序列 retrieval 搬进 Transformer 生成路径**:

1. **Retrieval 阶段**: decoder 对 `user_hist` → hierarchical beam 生成 `K` 个 SID, 反解出 `K` 个候选 item_id
2. **Ranking 阶段**: 对每个候选 `i(c)`, 用 cosine 相似度从 **整个 user history** 里 retrieve top-M 相似 item (SIM 风格 GSU), 把这 M 个 item 的 token 拼在 beam SID 后面, 再 append 一个候选 item_id token 作 "rank token"
3. **KV cache 复用**: 上面两步都在同一 decoder 做 incremental computation, user_hist 的 KV 不重算, beam SID 部分也复用
4. **Rank head**: 在 rank token 位置接 MLP head → sigmoid → click probability
5. **Loss**: Stage-I 纯 SID generation CE; Stage-II 同时 SID CE + binary CE (正样本 = 真 target SID match, 负样本 = beam 里其他 SID)

### 离线效果 (TAOBAO-MM, 仅数据集)

- Base (beam-only) R@5=0.2384, Rerank 后 0.2459 (**+3.14%**, beam=20, seq=32, retrieval=10)
- Beam size 10→40: rerank gain 0.27%→1.08%, 即 beam 越大 rerank 收益越大
- Retrieval length 0→20: rerank gain 0.12%→3.51%, GSU 搜到的相似序列越多越好

### 与当前项目的关联

- **对标 IDEA-onerec-3 (Reward Model 集成)** / **IDEA-orecv2-0** / **IDEA-recast-0**: 这些都是 "用 ranker 信号 shape GR 生成概率"; RecoChain 是 "让 GR 自己做 ranking"。两条路根本路线不同
- **对标 IDEA-glorank-0 (GloRank Kuaishou 重排)**: GloRank 是重排模型, 但用**独立**生成式 reranker; RecoChain 是**同一个** decoder 前后两段。耦合度更高, 但也更省算力
- **Target-aware GSU 是 SIM [arxiv 2006.05639] 的核心**, 论文引用到了; 本 idea 本质是 "OneRec + SIM 端到端化"

### 实验设计草案

**当前阶段不执行。** 前置条件:

1. 我们目前只有 retrieval 训练 (NTP 单任务), 没有 ranking label (二分类 click/conv)。需要先有 RTB / 曝光日志的 pos/neg 标签, 才能训 rank head
2. 现阶段 R@500 的相对差距 (64.1% vs 70.1%) 主要在 tokenizer + sequence scaling, rank head 的相对收益未必大过这些

如果未来补充 ranking label:

- Stage A: 保持现有 NTP 训练不变, 在 checkpoint 上加 rank head, 冻 backbone 微调 (快验证)
- Stage B: joint training, 两个 loss 加权求和

实验粒度: 对比 `beam-only R@K` vs `beam+rerank R@K`, 预期 rerank 在 beam size 大时提升更明显 (论文结论)

### 关键问题

1. **KV cache 设计**: 我们 `eval.py` 的 `constrained_beam_search` 已经有 KV cache 复用, 但 rank 阶段要 append **M 个 retrieved item tokens + 1 个 candidate item_id token**, 这套 incremental 要加一条新的路径, 非 trivial
2. **item_id token 的词汇表**: RecoChain 除了 SID token 还要额外一个 item_id embedding table (每个 item 一个 id token)。我们目前是 pure-SID, 加 item_id table 会让模型侧额外花 params
3. **GSU 搜索**: cosine 相似度在整个 user history 上搜 top-M, 这是工业 SIM 标准做法, 但离线训练时要预计算好, inference 时要实时做 (M=10 量级可接受)
4. **Rank label 来源**: 我们目前没有 "next item 是否真 click" 的 binary label; beam 里不是 target 的全算负样本, 这个构造跟论文一致, 但可能过稀疏 (positive/负 = 1/beam_size)

### 为什么是 Tier B 而非 Tier A

- Kuaishou 通讯作者 Jiangxia Cao (OneRec 系列一作) → 高 industrial relevance
- 但实验只在**公开 TAOBAO-MM 数据集**, 没有 Kuaishou 内部 online A/B 或 deployment 报告
- Short paper (5 页), 论文标 "work in progress", 方法细节 (GSU 的具体实现, rank head 的 BCE 构造) 尚不完整

### 相关 idea

- IDEA-onerec-3 / IDEA-orecv2-0 / IDEA-recast-0: GR + reward model 蒸馏路线
- IDEA-glorank-0: Kuaishou 的另一个生成式 reranking 方案, 独立 reranker
- IDEA-vista-0 (VISTA Meta): target-aware UIH attention, 思想类似但不在 GR decoder 内
- IDEA-a2gen-0 (A2Gen): 扩展 GR 输出到 (action, timing), 正交
