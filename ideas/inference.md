# Inference (推理优化)

Beam search 和解码策略的优化，提升推理吞吐和候选质量。在模型规模和 beam 扩大后变得关键。

**影响范围**: `metrics/sid_prediction.py` (beam search 逻辑)

---

## 演进路径

```
固定 beam search (当前 beam=5, 全 vocab softmax)
├── IDEA-gr4ad-4: Dynamic Beam Search
│   ├── DBW: 逐步增大 beam (128→256→512)
│   └── TopK Pre-Cut: 每 beam 先选 top-b → 全局 top-k
├── IDEA-static-0: CSR 矩阵约束解码
│   └── trie → CSR 稀疏矩阵, GPU 向量化, YouTube 948x speedup
├── IDEA-earn-0: Register Token 压缩
│   └── 前 K 层全注意力, 后 L-K 层仅 register tokens, 3.79x speedup
└── IDEA-flame-0: GR Serving 系统 (PDA/FKE/DSO)
    └── CPU-GPU 异构 + kernel fusion + 动态调度
```

---

## IDEA-gr4ad-4: Dynamic Beam Search 策略

**优先级**: P1
**来源**: GR4AD §Dynamic Beam Serving
**状态**: 待讨论

### 核心思想

GR4AD 提出两个 beam search 优化: (1) Dynamic Beam Width (DBW) — 逐步增大 beam（128→256→512 替代固定 512→512→512），因为早期层的候选质量高，不需要大 beam 来保留好候选；(2) TopK Pre-Cut — 每个 beam 内先选 bᵢ 个候选，再全局 top-k，避免在全 vocab 上排序。结果: DBW 带来 +0.31% revenue 且 QPS 提升 45%；TopK Pre-Cut 带来 +184.8% QPS。

### 与当前项目的关联

- `metrics/sid_prediction.py` 的 beam search 是固定 beam_size，每步都在全 vocab 上 softmax + top-k
- **与 IDEA-gr4ad-0 (MGMR) 强关联**: 如果用不等大码本 (16384→4096→1024)，第一层 vocab 大但只需小 beam，后面层 vocab 小但需大 beam — 天然适合 dynamic beam
- 当前 beam_size=5 没有优化空间，但 ARCHITECTURE.md 规划的生产目标是 beam=512 — 届时 dynamic beam 是必须的
- TopK Pre-Cut 可以立即实现作为通用优化

### 实验设计草案

**变量 1 — Dynamic Beam Width**:
| 配置 | Step 1 | Step 2 | Step 3 | 总 beam |
|------|--------|--------|--------|---------|
| Fixed | 50 | 50 | 50 | 50 |
| DBW-A | 10 | 25 | 50 | 50 |
| DBW-B | 5 | 15 | 50 | 50 |

**变量 2 — TopK Pre-Cut**:
- 每个 beam 先选 top-b 候选 (b << vocab_size)，再全局 top-k
- b = {32, 64, 128} 对比 full vocab

**评估**: Hit@K (质量), 推理时间 (效率)

### 关键问题

1. 当前 3 token + beam=5 下收益不明显，需要更大 beam 才能体现优势
2. DBW 的 schedule 设计: 与码本大小的关系？GR4AD 没有给出自动确定 schedule 的方法
3. 可以作为 IDEA-gr4ad-0 (MGMR) 的配套实现

---

## IDEA-static-0: CSR Matrix 约束解码 (STATIC)

**优先级**: ~~P1~~ → ❌ 关闭
**来源**: STATIC (Google/YouTube, arxiv 2602.22647, Feb 2026)
**状态**: ❌ 关闭 — 我们已有等价实现：`ntp/model.py:SIDTrie` + `constrained_beam_search()`，每步解码通过 `trie.valid_tokens(layer, prefix)` 过滤无效 token，保证输出 100% 是真实 SID。EXP-017 起所有 beam search 均使用此约束。STATIC 的 CSR 矩阵是 GPU 向量化加速版本，当前 3 层小 trie 无需此优化。

### 核心思想

STATIC 将 prefix tree (trie) 展平为 **Compressed Sparse Row (CSR) 矩阵**，将不规则的树遍历转换为完全向量化的稀疏矩阵运算，实现 GPU/TPU 上高效的约束解码。

核心问题: 生成式推荐需要约束解码 (只生成有效 SID)，传统 trie 方法在 GPU 上极慢 (指针追踪、不规则访问)。

技术:
1. **Trie → CSR 矩阵**: 将前缀树的层级结构展平为静态 CSR 矩阵
2. **向量化 transition**: 每步解码 = 稀疏矩阵乘法，天然适合 GPU 并行
3. **支持业务约束**: 如内容新鲜度、品类限制等动态约束

**YouTube 生产部署**: **948x speedup** over CPU trie, **0.033 ms per step**, 仅占推理时间 0.25%。开源: `github.com/youtube/static-constraint-decoding`。

### 与当前项目的关联

- 当前 beam search (`metrics/sid_prediction.py`) 在全 vocab 上 softmax，没有约束解码
- 如果实现有效 SID 约束 (只生成 assignment 过的 SID 组合)，可以避免生成无效 ID → 提升有效 recall
- **与 IDEA-gr4ad-4 (Dynamic Beam Search) 配合**: 先约束有效集合，再在有效集合内做 dynamic beam
- 当前 3 层 × 1024 码本，有效 SID 约 5M / (1024^3 ≈ 10^9) = 0.5% 的空间 — 约束解码能剪掉 99.5% 无效组合
- 代码开源可直接参考

### 实验设计草案

**Phase 1 — 构造有效 SID trie**:
1. 从 RKMeans assignment 中提取所有有效 SID (5M 条)
2. 构建 3 层 prefix trie
3. 转换为 CSR 矩阵

**Phase 2 — 集成到 beam search**:
1. 每步解码时，用 CSR 矩阵 mask 无效 token → 只在有效 token 上 softmax
2. 评估: 有/无约束的 Recall@K 和推理速度

### 关键问题

1. 当前 3 层 SID，trie 深度只有 3 — 约束解码的收益可能有限 (每层 1024 vocab 中有效 token 比例不低)
2. 如果切到 OPQ (16~64 层)，约束解码价值大增 — 有效组合在指数级空间中极稀疏
3. 动态约束 (如品类过滤) 需要修改 CSR 矩阵结构

---

## IDEA-flame-0: GR 推理系统优化 (FLAME)

**优先级**: P2
**来源**: FLAME (arxiv 2509.22681, Sep 2025)
**状态**: 待讨论

### 核心思想

FLAME 是 GR 模型的专用推理系统，三个核心模块:

1. **Proximal Data Accelerator (PDA)**: CPU-GPU 异构计算，feature 预处理 (CPU) 与模型推理 (GPU) 解耦 → **1.9x 吞吐, 1.7x 延迟降低**
2. **Fused Kernel Engine (FKE)**: 基于 TensorRT 的 kernel fusion → **4.6-6.1x 加速**
3. **Dynamic Stream Orchestrator (DSO)**: 动态调度并发请求 → **1.3x 吞吐, 2.3x speedup under non-uniform distribution**

核心洞察: GR 模型的 FLOP 量级 (10^9~10^11) 比传统 DLRM 高 4 个数量级，需要专门的 serving 系统。

### 与当前项目的关联

- 当前不需要生产级 serving 优化 (研究阶段)
- 但 PDA 的 CPU-GPU 解耦思想可以用于训练: feature preprocessing 用 CPU async 预处理，GPU 专注 forward/backward
- FKE 的 kernel fusion 在我们用 PyTorch 的场景下 = torch.compile / FlashAttention
- **参考价值**: 了解生产 GR serving 的延迟分解和瓶颈点

### 关键问题

1. 研究阶段优先级低，生产部署时再详细评估
2. 部分优化 (如 FlashAttention) 已内置在现代 PyTorch 中

---

## IDEA-earn-0: Register Token 压缩推理 (KV Cache 减少 80%)

**优先级**: P1
**来源**: EARN (arxiv 2507.00715, Jul 2025, KDD 2025)
**状态**: 待讨论

### 核心思想

EARN 发现 LLM-based 推荐模型的注意力模式有独特特征:

1. **Layer-wise Attention Sparsity Inversion**: 早期层注意力密集且信息丰富，后期层高度冗余
2. **Dual Attention Sinks**: 注意力分数集中在序列首尾 token

基于此提出:
- 在输入序列首尾放置 **register tokens**
- **早期层** (dense attention): 全序列正常计算，信息压缩到 register tokens
- **后期层** (sparse attention): 只计算 register tokens，跳过其余序列

**结果**: **3.79x speedup, 80.8% KV Cache reduction**, 精度不降反升 (优于一般 fine-tuning)。KDD 2025。

### 与当前项目的关联

- 当前 `AutoregressiveNTPModel` 是 6 层 decoder，序列很短 (3 SID tokens)，推理不是瓶颈
- 但如果:
  - 切到长序列 (IDEA-onemall-1 Query-Former 的输入侧)
  - 用 LLM backbone (IDEA-plum-0)
  - 扩大 beam (IDEA-gr4ad-4)
  → Register token 压缩变得关键
- **与 IDEA-gr4ad-1 (LazyAR) 思想类似**: 都是 "前面层做完整计算，后面层简化"。LazyAR 简化自回归依赖，EARN 简化注意力范围
- 两者可以组合

### 实验设计草案

**在 LLM backbone 场景下** (依赖 IDEA-plum-0):
1. 在 Qwen3-0.5B 的 input 首尾各加 n_reg 个 register tokens
2. 前 K 层全序列 attention
3. 后 L-K 层仅 attend to register tokens
4. K 的选择: EARN 发现约 1/3 处注意力开始稀疏化

**评估**: Recall@K vs 推理延迟 vs KV Cache 占用

### 关键问题

1. 当前 39.5M 小模型 + 短序列下无收益 → 依赖模型/序列扩展
2. Register tokens 的数量选择: 太少信息损失，太多压缩不够
3. 与 LazyAR 的组合设计: 两者优化的层不同 (LazyAR 优化自回归依赖，EARN 优化注意力范围)

---

## IDEA-promise-0: Process Reward Model + Test-Time Scaling for GR

**优先级**: P1
**来源**: PROMISE (Kuaishou, arxiv 2601.04674, Jan 2026)
**状态**: 待讨论 — 前置：需先完成 RL 对齐链路 (EXP-037→039)，PRM 训练数据构造依赖 beam search rollout 基础设施

### 核心思想

PROMISE 识别了 **Semantic Drift** 问题: 层级 SID 生成中，早期高层 token 的错误会不可逆地将生成轨迹引入无关语义子空间。

解决方案:
1. **轻量 PRM (Process Reward Model)**: 评估每个中间推理步骤的质量 (而非仅看最终结果的 ORM)
2. **PRM-guided Beam Search**: 用 PRM 的稠密反馈动态剪枝错误分支 (不只依赖 token probability)
3. **Test-Time Scaling Laws**: 增加推理计算可以让小模型 match 甚至超越大模型

核心洞察: **在 GR 中复现了 LLM reasoning 的 test-time scaling 规律** — 推理阶段投入更多计算 (更多 beam + PRM 评分) 可以弥补模型容量不足。

快手大规模平台在线 A/B 验证: 显著提升推荐准确率，同时保持部署效率。

### 与当前项目的关联

- 当前 beam search 仅用 token probability 排序 → PRM 可以提供更好的中间步骤质量评估
- 与 IDEA-gr4ad-4 (Dynamic Beam Search) 互补: gr4ad-4 优化 beam 效率，promise-0 优化 beam 质量
- Test-time scaling 启示: 当前 39.5M 小模型 + PRM 可能超越未来更大模型的基础 beam search
- PRM 训练需要: step-level 标注数据 (可用 Monte Carlo rollout 自动构造)

### 关键问题

1. PRM 训练数据构造: step-level 标注成本高 → 需要 Monte Carlo 自动方案
2. PRM 推理开销: 每个 beam candidate 每步都需 PRM 评分 → 延迟 tradeoff
3. 前置依赖 NTP baseline (需要先有可用的 beam search 基础)

---

## IDEA-grc-0: Generation-Reflection-Correction Decoding

**优先级**: P1
**来源**: GRC (Alibaba, arxiv 2602.23639, Feb 2026)
**状态**: 待讨论 — 前置：NTP baseline + GRPO 基础设施（已有 EXP-026），但三阶段 GRC 训练数据构造尚未设计

### 核心思想

GRC 将标准的单次解码扩展为三阶段 **Generation-Reflection-Correction** 流程:

1. **Generation**: 标准自回归生成初始 SID 序列 (draft)
2. **Reflection**: 多粒度反思 — 模型审视已生成序列的质量
3. **Correction**: 基于反思结果修正生成轨迹

关键优化:
- **GRPO-based RL**: 在整个 GRC 轨迹上做 GRPO 优化, reward 结合 token-level 和 trajectory-level 信号
- **Entropy-Guided Reflection Scheduling (EGRS)**: serving 时动态分配 reflection 预算 — 高不确定性轨迹多反思, 低不确定性直接输出

阿里大规模工业推荐: **广告收入 +1.79%**, latency 开销可控 (EGRS 只对不确定的 beam 做 reflect)。

### 与当前项目的关联

- 类似 LLM 的 self-reflection/self-correction 但在 SID token 空间操作
- 与 IDEA-s2gr-0 (Stepwise Reasoning) 互补: s2gr 在每步生成前"思考", GRC 在整体生成后"反思修正"
- 与 IDEA-promise-0 (PRM) 也互补: PRM 评估步骤质量, GRC 允许修正
- EGRS 是关键: 不是所有 beam 都做反思, 只对高 entropy 的做 → 控制延迟

### 关键问题

1. 训练数据构造: (draft, reflection, corrected) 三元组的自动生成策略
2. 序列长度膨胀: GRC 增加 ~2x tokens → 需要 EGRS 控制
3. 前置: NTP baseline + GRPO 基础设施 (IDEA-onemall-2)

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P1 | IDEA-gr4ad-4 | Dynamic Beam Search | 生产 beam=512 时必需；可与 IDEA-gr4ad-0 配套 |
| ~~P1~~ ❌ | ~~IDEA-static-0~~ | ~~CSR 约束解码~~ | ❌ 已有等价实现：SIDTrie + constrained_beam_search，EXP-017 起标配 |
| P1 | IDEA-earn-0 | Register Token 压缩 | 3.79x speedup, 与 LazyAR 互补, KDD 2025 |
| P1 | IDEA-promise-0 | PRM-guided Beam Search | 快手在线验证, test-time scaling 解锁小模型潜力 |
| P1 | IDEA-grc-0 | Generation-Reflection-Correction | 阿里 +1.79% revenue, EGRS 控制延迟, 与 GRPO 协同 |
| P1 | IDEA-orecv2-0 | FP8 PTQ 推理加速 | 快手 OneRec-V2, -49% latency +92% throughput, 0 质量损失 |
| P2 | IDEA-flame-0 | GR Serving 系统 | 生产部署参考，当前阶段优先级低 |

---

## IDEA-orecv2-0: FP8 Post-Training Quantization 推理加速

**优先级**: P1
**来源**: Quantized Inference for OneRec-V2, Kuaishou (arxiv 2603.11486)
**状态**: 待讨论

### 核心思想

快手 OneRec-V2 (4B 参数, 0.5B activated, fat-MoE 架构) 的 FP8 PTQ 推理优化。关键发现: GR 模型的 weight/activation 分布统计量远比传统推荐模型可控 (方差低 5-6 个数量级)，接近 LLM (Qwen3-8B)。因此 LLM 的量化技术可以直接迁移。具体方案:

1. **Per-channel weight quantization** (offline): Linear 层 (Attention qkvo + Dense FFN) + grouped GEMM (MoE)
2. **Per-token activation quantization** (runtime dynamic scaling)
3. **FP8 TensorCore multiply + FP32 accumulation → cast back FP16**
4. **MoE block-wise quantization**: 1×128 activation, 128×128 weight granularity

配合 infrastructure 优化 (TensorRT 直接构建, RadixTopK, attention kernel 优化, MoE TMA kernel):
- Latency: 139ms → 70ms (-49%)
- Throughput: 205 → 394 (+92%)
- 在线 A/B: 快手+快手极速版所有核心指标无劣化

### 与当前项目的关联

- 当前阶段关注模型训练，但部署时 FP8 是必经之路
- 关键 insight: **GR 模型天然适合量化** — 与传统推荐模型不同, 不需要额外的量化感知训练
- OneRec-V2 的 MoE + Transformer 架构与我们未来可能的模型架构一致
- 42% throughput gain 来自 FP8 quant alone → 对部署成本影响巨大

### 实验设计草案

**Phase 1 — Distribution Analysis**:
- 在训练好的 NTP 模型上分析 weight/activation 分布 (variance, AbsMax, AbsP99)
- 与 OneRec-V2 和传统推荐模型的数据做对比
- 判断我们的模型是否也具有 "接近 LLM" 的量化友好特性

**Phase 2 — FP8 PTQ Inference**:
- 用 PyTorch FP8 或 TensorRT FP8 做推理
- 对比 FP16 vs FP8: latency, throughput, Recall@K 差异
- 需要 H100 GPU (FP8 TensorCore)

### 关键问题

1. 当前模型规模较小 (不是 4B), FP8 加速比可能不如 OneRec-V2 显著
2. 需要 Hopper 架构 GPU (H100/H200) 支持 FP8 TensorCore
3. Phase 1 (分布分析) 零成本，可以先做
4. 更适合模型上线部署阶段，当前优先级低于训练优化
