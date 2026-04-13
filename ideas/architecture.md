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
├── IDEA-oneloc-0: Context-augmented Attention (side-info 注入)
│   └── additive similarity + gating，需 encoder-decoder 架构
└── IDEA-oneloc-1: Category Prompt (邻域 cross-attention 提示)
    └── 泛化为 interest/category prompt prefix
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

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P1 | IDEA-gr4ad-1 | LazyAR 解码器 | 与 ARCHITECTURE.md Lazy Decoder-Only 方向一致；扩展 token 数或 beam 后必需 |
| P1 | IDEA-onemall-1 | Query-Former 序列压缩 | 3.7x FLOP 减少，但需要更长序列场景 |
| P2 | IDEA-onemall-4 | Loss-Free MoE Balancing | 低风险低成本，8 experts 下收益可能有限 |
| P2 | IDEA-oneloc-0 | Context-augmented Attention | 需要 encoder-decoder 架构，当前无落地场景 |
| P2 | IDEA-oneloc-1 | Category Prompt | 需要 encoder-decoder 架构，泛化形式有价值 |
