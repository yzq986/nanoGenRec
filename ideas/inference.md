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

**优先级**: P1
**来源**: STATIC (Google/YouTube, arxiv 2602.22647, Feb 2026)
**状态**: 待讨论

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

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P1 | IDEA-gr4ad-4 | Dynamic Beam Search | 生产 beam=512 时必需；可与 IDEA-gr4ad-0 配套 |
| P1 | IDEA-static-0 | CSR 约束解码 | YouTube 开源验证，OPQ 长 ID 下价值极大 |
| P1 | IDEA-earn-0 | Register Token 压缩 | 3.79x speedup, 与 LazyAR 互补, KDD 2025 |
| P2 | IDEA-flame-0 | GR Serving 系统 | 生产部署参考，当前阶段优先级低 |
