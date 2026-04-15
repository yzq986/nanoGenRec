# NTP Architecture Evolution Roadmap

从当前 NTPProbe (2L decoder-only, 5M params) 逐步演进到 OneRec 级 encoder-decoder 架构的迭代路径。

每个 Stage 独立可测，前一个 Stage 的指标回归是下一个 Stage 的 baseline。

---

## 当前起点

`ntp/model.py` — **NTPProbe**

| 项目 | 现状 |
|------|------|
| 架构 | Decoder-only (nn.TransformerDecoder) |
| 层数 | 2 |
| d_model / heads / FFN | 256 / 4 / 512 (Dense) |
| 参数量 | ~5M |
| 用户表示 | 无 (行为序列隐式编码) |
| 输入 | 10 items × 3 SID tokens = 30 tokens |
| 解码 | Beam search (beam=5 训练, 50 eval) |
| 训练 | per-layer CE, DDP, 1 epoch |

`sid_prediction_old.py` 中有完整 S-tier 实现 (6L, MoE 8E top-2, ~39.5M) 但未迁移到新 DDP pipeline。

---

## Stage 1: S-tier Decoder + Loss-Free MoE

**目标**: 建立强 decoder baseline，后续所有改进在此基础上对比

**来源**: `sid_prediction_old.py` 迁移 + IDEA-onemall-4

**改动**: `ntp/model.py`

| 项目 | 现状 → 目标 |
|------|-------------|
| 层数 | 2 → 6 |
| Heads | 4 → 8 |
| FFN | Dense 512 → SwiGLU MoE (8E, top-2, expert_dim=1024) |
| Load balancing | N/A → Loss-Free dynamic bias (替代 Switch aux loss) |
| KV Cache | 无 → 增量解码, beam search 复用 |
| 参数量 | 5M → ~39.5M total / ~11M active |

Loss-Free MoE (IDEA-onemall-4) 改动极小:
```python
# 每个 expert 维护一个不参与梯度的 bias
expert_bias = torch.zeros(n_experts)  # requires_grad=False
router_score = linear(x) + expert_bias
# 每步统计频率，动态调整 bias
expert_bias[i] -= lr_bias * (freq_i - 1/n_experts)
```

**验收**:
- PPL 下降幅度 (预期 > 30%)
- Recall@50 / Recall@500 对比 NTPProbe
- Expert 利用率分布 (loss-free vs aux loss)

**风险**: 低。核心代码已在 `sid_prediction_old.py` 中验证。

---

## Stage 2: Soft Prompt — 用户表示注入

**目标**: 最小架构改动下验证用户表示的价值

**来源**: IDEA-glide-0 (Spotify: 非惯常收听 +5.4%, 新发现 +14.3%)

**改动**: `ntp/model.py` 新增 ~50 行

```
当前:  [sid(item_1), sid(item_2), ..., sid(item_10)] → Decoder → next_sid
                                                        (30 tokens)

Stage 2: [prefix_1, ..., prefix_n, sid(item_1), ..., sid(item_10)] → Decoder → next_sid
           ↑
           user behavior embeddings → AttentionPooling → MLP → n prefix tokens
```

**设计**:

| 组件 | 方案 |
|------|------|
| User embedding 来源 | 用户近期行为 item 的 content embedding (Qwen3-0.6B) |
| Pooling | Attention-weighted pooling (learnable query) |
| Projection | MLP(pooled_dim → embed_dim × n_prefix) → reshape |
| n_prefix | sweep {2, 4, 8} |

**训练策略**:
1. Phase A: 冻结 decoder, 只训练 prefix projection (快速收敛)
2. Phase B: 全量 joint fine-tune

**验收**:
- 有/无 soft prompt 的 Recall@K 对比
- 分 cold user (< 5 interactions) / warm user (> 20) 分析
- 如果 soft prompt 提升显著 → 用户表示是核心缺失，Stage 3 优先级提升
- 如果提升有限 → 瓶颈在解码侧或 tokenizer 侧，考虑先做 Stage 5

**风险**: 低。不改变 decoder 架构，只在输入前加 prefix。

**Open questions**:
- [ ] 用户 content embedding 是否预计算缓存? 还是在线计算?
- [ ] prefix tokens 共享 positional embedding 还是独立?

---

## Stage 3: Encoder-Decoder 分离

**目标**: 将用户行为编码与 SID 生成解耦，支持多尺度行为建模 + 推理加速

**来源**: OneRec encoder-decoder + ARCHITECTURE.md Context Processor + IDEA-gr4ad-1 (LazyAR)

**新增文件**: `ntp/encoder.py`

### 3a — Lazy Decoder-Only (轻量版，推荐先做)

参考 OneRec-V2 "Lazy Decoder-Only" + LazyAR:

```
同一个 Transformer 的 6 层分为两段:

前 4 层 (Context Processing):
  - 双向 attention (non-causal)
  - 处理用户行为序列 (30 tokens)
  - 输出 static KV pairs
  - beam search 时只算一次，所有 beam 共享

     ──── Fusion Layer (gated projection) ────

后 2 层 (SID Generation):
  - 单向 attention (causal)
  - 只处理 [BOS] + 3 个 SID target tokens
  - Cross-attend to 前 4 层的 KV pairs
  - beam search 在这里展开
```

| 优势 | 说明 |
|------|------|
| 推理加速 | beam=500 时，前 4 层不随 beam 增长，只有后 2 层线性增长 |
| 信息交互 | 前 4 层双向 attention 比纯 causal 更好地编码用户行为 |
| 实现简洁 | 不需要独立 encoder，同一套 Transformer 参数 |

Fusion 机制 (第 4 层 → 第 5 层):
```python
# m: non-AR representation, s: previous token embedding
Fuse(m, s) = W_f[m * sigmoid(W_g @ s); s]
```

### 3b — 完整 Encoder-Decoder (在 3a 验证后)

```
┌──────────────────────────────┐
│       Context Encoder        │
│    (N layers, bidirectional)  │
│                              │
│  short_term  (20 items)  ────┤
│  positive_fb (N items)   ────┤──→ Z_enc ∈ ℝ^{T_enc × d_model}
│  [user_static (optional)]────┤       │
└──────────────────────────────┘       │
                                       │ keys, values
                                       ↓
┌──────────────────────────────┐       │
│        SID Decoder           │       │
│    (M layers, causal)        │       │
│                              │       │
│  每层:                        │       │
│    1. causal self-attention   │       │
│    2. cross-attention ◄───────┼───────┘
│    3. MoE FFN (SwiGLU)       │
│                              │
│  [BOS] → sid_1 → sid_2 → sid_3│
└──────────────────────────────┘
```

| 组件 | 配置 |
|------|------|
| Encoder layers | 4 (bidirectional, dense FFN) |
| Decoder layers | 4 (causal self-attn + cross-attn + MoE FFN) |
| 多行为通道 | short-term / positive-feedback 分别嵌入后拼接 |
| Encoder 输出 | 推理时缓存，beam search 只在 decoder 展开 |

**验收**:
- 对比 Stage 2 的 Recall@K
- 推理延迟: beam=500 时 3a vs Stage 1 的加速比
- encoder 表示质量: 探针分析 (linear probe 预测 user 兴趣类目)

**风险**: 中。架构变更大，需要仔细调试训练稳定性。

**Open questions**:
- [ ] 3a 和 3b 哪个先做? 3a 更简洁但 3b 更通用
- [ ] Encoder 和 Decoder 是否共享 embedding?
- [ ] 多行为通道: 当前数据是否有 positive_feedback 独立标注?

---

## Stage 4: 长序列压缩 — Query-Former

**目标**: 支持 200+ 行为序列输入，控制计算量

**来源**: IDEA-onemall-1 (OneMall: 1205→160 tokens, 3.7x FLOP 减少) + OneRec lifelong pathway

**前置**: Stage 3 完成 (encoder 能接收变长输入)

**新增**: `ntp/query_former.py` (可复用 `model/qformer.py`)

```
用户行为序列 (变长, 最长 500+)
       │
       ▼
┌─────────────────────┐
│    Query-Former      │
│                      │
│  Q: M learnable      │
│     query tokens     │
│  KV: 行为序列 embed   │
│                      │
│  N layers cross-attn │
└─────────────────────┘
       │
       ▼
  M 个压缩 tokens (固定长度)
       │
       ▼
  concat with short-term tokens → Encoder
```

| 参数 | 搜索范围 |
|------|----------|
| M (query tokens) | {4, 8, 16} |
| QFormer layers | {1, 2} |
| 输入序列长度 | {50, 100, 200, 500} |

**分层策略** (参考 OneRec + GEMs):

| 时间尺度 | 处理方式 | Token 数 |
|----------|----------|----------|
| Short-term (≤20 items) | 直接输入, 无压缩 | 20 × 3 = 60 |
| Mid-term (20-200 items) | Query-Former 压缩 | M = 8-16 |
| Lifelong (200+ items) | 远期: hierarchical K-means + QFormer | 远期 |

**验收**:
- 固定 FLOP 预算: 不同序列长度的 Recall@K
- 压缩率 vs 性能: M=4/8/16 的 trade-off 曲线
- 对比 baseline: 直接截断到 20 items (当前方案)

**风险**: 中低。QFormer 是成熟组件。

**Open questions**:
- [ ] 当前用户平均行为序列多长? 如果 < 50，此 stage 收益有限
- [ ] 多行为类型 (click/buy/exposure) 各自一个 QFormer 还是共享?
- [ ] QFormer 是否需要单独预训练?

---

## Stage 5: 增强解码 (二选一)

**目标**: 提升 SID 生成质量，缩小 beam search 空间

**前置**: Stage 3 完成。根据 Stage 3 的 error analysis 决定选哪个。

### 选项 A — Stepwise Reasoning Tokens (IDEA-s2gr-0)

```
原始:    [BOS]  →  sid_L1  →  sid_L2  →  sid_L3          (4 tokens)
改为:    [BOS]  →  [THINK] →  sid_L1  →  [THINK] →  sid_L2  →  [THINK] →  sid_L3
                     ↑                      ↑                      ↑
                  contrastive            contrastive            contrastive
                  (对齐 cluster            (对齐 cluster            (对齐 cluster
                   分布)                    分布)                    分布)
```

- Think token 用 contrastive loss 对齐 codebook cluster 分布
- `L_total = L_SID + alpha * L_think`
- 序列 4 → 7 tokens，解码步数翻倍，但每步更精准
- **适合场景**: error analysis 显示早期 token 错误传播到后续 token

### 选项 B — Chain-of-Attribute Prefix (IDEA-unirec-0)

```
原始:    [BOS]  →  sid_L1  →  sid_L2  →  sid_L3          (4 tokens)
改为:    [BOS]  →  cat_tok →  brand_tok →  sid_L1  →  sid_L2  →  sid_L3
                     ↑          ↑
                  属性 token    属性 token
                  (类目)       (品牌)
```

- 贝叶斯保证: `H(s_k | s_{<k}, a) < H(s_k | s_{<k})` — 属性前缀减少条件熵
- 在线结果: HR@50 +22.6%, 高价值订单 +15.5%
- **适合场景**: item 有结构化属性 (category/brand/seller)
- **需要**: 属性数据 + 属性 tokenizer

### 决策依据

| 条件 | 选择 |
|------|------|
| 早期 token 错误率高, 属性数据不可用 | 选 A (Reasoning Tokens) |
| 属性数据可用, beam search 空间过大 | 选 B (CoA Prefix) |
| 两者都可以 | 选 B (理论保证更强, 线上效果更好) |

**验收**:
- Recall@K 提升
- Beam search 效率: 有效候选占比 (top-500 中有多少是合法 item)
- 逐层准确率分析 (对应 eval.py 的 prefix depth hit@10)

---

## Stage 6: RL 对齐 + 生产级优化 (远期)

**目标**: 从 "预测准" 到 "推荐好"

**前置**: Stage 1-5 中至少 1-3 完成且架构收敛

**来源**: OneRec ECPO + IDEA-oxygen-0

| 组件 | 方案 |
|------|------|
| Reward Model | Multi-tower P-Score (ctr/lvtr/ltr/vtr towers + 聚合) |
| SFT | RSFT: 过滤底部 50% sessions (按 play duration), 监督微调 |
| RL | ECPO (Early Clipped GRPO): group_size = 4× beam |
| 推理 | Beam 扩大到 Pass@512 |
| 多场景 | SA-GCPO (远期, 当前单场景) |

**这个 Stage 不急**: OneRec 论文和 GenRank (IDEA-genrank-0) 都证明了 **Architecture > Training Paradigm**。先把架构做对再做 RL。

---

## 总览

```
Stage 1          Stage 2          Stage 3a         Stage 3b          Stage 4          Stage 5          Stage 6
S-tier           Soft             Lazy             Full              Query-           Reasoning        RL
Decoder          Prompt           Dec-Only         Enc-Dec           Former           / CoA            对齐
+ MoE                                                                                Prefix
   │                │                │                │                │                │                │
   ▼                ▼                ▼                ▼                ▼                ▼                ▼
 强baseline       验证用户          推理加速          多尺度           500+序列         解码质量          线上
 39.5M           表示价值          beam共享KV        行为建模          FLOP↓3-4x        精度↑             指标
```

**关键决策点**:
- Stage 2 结果决定 Stage 3 的优先级
- Stage 3a vs 3b: 如果 3a 效果够好，可以跳过 3b
- Stage 5 A vs B: 取决于 error analysis 和数据可用性
- Stage 4 可以和 Stage 5 并行

---

## 未纳入当前路径但值得关注的 IDEA

| IDEA | 理由 | 何时考虑 |
|------|------|----------|
| IDEA-llada-0 (Discrete Diffusion) | 全新解码范式, 工程复杂度高 | Stage 5 后如果 AR 到瓶颈 |
| IDEA-oxygen-0 (Fast-Slow Thinking) | 需要 LLM 推理环节, 当前过于复杂 | Stage 6 之后 |
| IDEA-gr2-0 (LLM Reranker) | 属于 reranking, 非 retrieval | 有独立 reranking 需求时 |
| IDEA-higr-0 (Hierarchical Slate) | 属于 reranking, 5x 推理加速 | slate 推荐场景 |
| IDEA-hpgr-0 (Session-MIM) | 需要 session 切分 + 两阶段训练 | 序列足够长时 |
| IDEA-gti-0 (Grounded Token Init) | 针对 LLM vocab extension | 走 LLM CPT 路线时 |
