# GR4AD: 快手大规模广告生成式推荐

**来源**: [GR4AD: Generative Recommendation for Large-Scale Advertising](https://arxiv.org/abs/2602.22732) (Kuaishou, 2026)
**日期**: 2026-04-13

---

## IDEA-005: Multi-Granularity Multi-Resolution RQ-KMeans (MGMR)

**优先级**: P0
**来源**: GR4AD §UA-SID, Table 2
**状态**: 待讨论

### 核心思想

GR4AD 提出 MGMR 编码方案：(1) Multi-Resolution — 低层用大码本捕获主导因子，高层用小码本建模低熵残差（如 16384→4096→1024）；(2) Multi-Granularity — 最后一层用非语义特征（item ID、账户 ID）的 hash 映射替代聚类，直接消除碰撞。两者结合使碰撞率从 3.54 降至 1.07，码本利用率从 0.10‰ 提升至 0.34‰。

### 与当前项目的关联

- **直接对标 EXP-001**: 当前用等大码本 3×1024，collision=1.75%。MGMR 的不等大码本（如 4096→1024→256）是零成本改进
- `model/rkmeans.py` 的 `ResidualQuantizationMultiGPU` 已支持每层独立 `n_clusters` 参数，只需传入不同值
- Multi-Granularity 的 hash 层思想与 IDEA-001 (OPQ) 互补 — 如果最后一层用 hash 保证唯一性，前面层可以放心用更粗的语义聚类
- `eval/evaluator.py` 的 collision_rate、codebook utilization 指标可直接用于评估

### 实验设计草案

**变量 1 — Multi-Resolution 码本配置**:

| 配置 | L1 | L2 | L3 | 总编码空间 |
|------|------|------|------|-----------|
| Baseline (EXP-001) | 1024 | 1024 | 1024 | 10^9 |
| MR-A | 4096 | 1024 | 256 | 10^9 |
| MR-B | 2048 | 1024 | 512 | 10^9 |
| MR-C | 4096 | 2048 | 512 | 4×10^9 |

**变量 2 — Multi-Granularity hash 层**:
- 对每个 MR 配置，测试最后一层替换为 hash(item_id) % vocab_size
- 需要在 `model/rkmeans.py` 增加 hash 层选项

**评估**: collision_rate, recon_loss, codebook_utilization, cluster_balance (Gini), sid_prediction Hit@K

**实现成本**: 低。MR 只需修改 config 参数；MG hash 层需在 `ResidualQuantizationMultiGPU` 中增加约 20 行代码

### 关键问题

1. 不等大码本下 NTP 模型的 `vocab_size` 需要每层不同 — `metrics/sid_prediction.py` 的 `AutoregressiveNTPModel` 目前假设统一 vocab_size，需适配
2. Hash 层的 vocab_size 选择: 与 item 数量的关系？太小仍有碰撞，太大稀疏
3. 与 IDEA-003 (Balanced KMeans) 的交互: 大码本 (4096+) 下 balanced assignment 更关键

---

## IDEA-006: LazyAR 解码器

**优先级**: P1
**来源**: GR4AD §LazyAR, Table 1
**状态**: 待讨论

### 核心思想

GR4AD 将 L 层 decoder 分为两部分: 前 K 层（非自回归）只依赖位置编码和 context，不依赖前一个 token；后 L-K 层才引入自回归依赖。关键洞察: 前 K 层的输出可以对所有 token 位置并行计算并在 beam 间共享，只有后 L-K 层需要逐 token 解码。实验显示 K=2/3·L 时性能几乎无损（-0.04%），但推理吞吐翻倍。

Fusion 机制: 在第 K 层用 gated projection 融合非自回归表示和前一 token embedding:
`Fuse(m, s) = W_f[m ⊙ (W_g · s); s]`

### 与当前项目的关联

- `metrics/sid_prediction.py` 的 `AutoregressiveNTPModel` 是纯自回归: 每层都依赖前一 token 的 embedding
- 当前只有 3 个 token 要预测，beam_size=5，推理不是瓶颈。但如果扩展到更多 token (IDEA-001 OPQ 方案 B/C 有 16-32 token) 或更大 beam (生产目标 512)，LazyAR 变得关键
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

## IDEA-007: Value-Aware 训练目标 (VSL + eCPM Token)

**优先级**: P1
**来源**: GR4AD §VSL
**状态**: 待讨论

### 核心思想

GR4AD 在 NTP 训练中引入两个价值感知机制: (1) eCPM Token Prediction — 在语义 ID 序列末尾追加一个离散化的 eCPM token，让模型同时预测"推什么"和"值多少钱"；(2) Value-Aware Sample Weighting — 按用户长期价值和行为深度（购买 > 点击）加权训练样本。

### 与当前项目的关联

- `metrics/sid_prediction.py` 当前训练目标是纯 CE loss，所有样本等权
- 我们的数据中有行为类型（点击、购买、收藏等），在 `data/export_behavior.py` 中已定义
- eCPM token 的思想可以泛化为 **任意业务价值 token** — 比如 item 热度桶、CTR 桶等
- **与 IDEA-002 (协同信号增强) 互补**: IDEA-002 改进 embedding 表示，本 IDEA 改进训练信号

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

## IDEA-008: RSPO 排序优化 (Ranking-Guided Softmax Preference Optimization)

**优先级**: P2
**来源**: GR4AD §RSPO, Table 1
**状态**: 待讨论

### 核心思想

GR4AD 提出 list-wise RL 方法 RSPO: 将 beam search 产出的候选列表按 eCPM 排序，用 NDCG-inspired Lambda 权重做偏好优化。相比 DPO (+0.70%) 和 GRPO (+0.65%)，RSPO 带来 +1.06% 增量。核心创新: (1) Lambda 权重 ℳᵢⱼ 关注排序位置交换的 NDCG 收益；(2) Reference gating Cᵢⱼ 在参考模型不可靠时自动关闭 KL 约束。

### 与当前项目的关联

- 当前 NTP 模型只做监督学习，没有任何 RL/preference optimization
- **前置依赖重**: 需要先有 (1) 合理的 reward signal（IDEA-007 的价值 token）；(2) 足够好的 beam search 产出多个候选（当前 beam=5 候选太少）
- 实现复杂度高: 需要 reference model、reward model、Lambda NDCG 计算、online learning pipeline
- 更适合作为系统成熟后的进阶优化

### 实验设计草案

**简化版 — Offline DPO 起步**:
1. 用当前 NTP 模型 beam search 产出 top-K 候选
2. 按行为数据构造偏好对: 被点击的 item > 未被点击的 item
3. 先实现 DPO loss 验证框架，再升级到 RSPO

**进阶版 — RSPO**:
- 在 DPO 基础上替换 pairwise loss 为 list-wise Lambda-weighted softmax loss
- 加入 reference gating 机制

**评估**: Hit@K, NDCG@K, 与纯 SL 模型对比

### 关键问题

1. **数据要求高**: 需要同一 context 下多个候选的真实反馈，当前 demo 数据不一定有
2. 训练稳定性: RL 方法调参困难，reference model 需要定期更新
3. 收益依赖于 beam search 质量 — 如果 beam search 本身不够好（候选同质化），排序优化价值有限
4. 建议优先级在 IDEA-005/006/007 之后

---

## IDEA-009: Dynamic Beam Search 策略

**优先级**: P1
**来源**: GR4AD §Dynamic Beam Serving
**状态**: 待讨论

### 核心思想

GR4AD 提出两个 beam search 优化: (1) Dynamic Beam Width (DBW) — 逐步增大 beam（128→256→512 替代固定 512→512→512），因为早期层的候选质量高，不需要大 beam 来保留好候选；(2) TopK Pre-Cut — 每个 beam 内先选 bᵢ 个候选，再全局 top-k，避免在全 vocab 上排序。结果: DBW 带来 +0.31% revenue 且 QPS 提升 45%；TopK Pre-Cut 带来 +184.8% QPS。

### 与当前项目的关联

- `metrics/sid_prediction.py` 的 beam search 是固定 beam_size，每步都在全 vocab 上 softmax + top-k
- **与 IDEA-005 (MGMR) 强关联**: 如果用不等大码本 (16384→4096→1024)，第一层 vocab 大但只需小 beam，后面层 vocab 小但需大 beam — 天然适合 dynamic beam
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
3. 可以作为 IDEA-005 (MGMR) 的配套实现

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P0 | IDEA-005 | MGMR 不等大码本 | 零成本改进，直接提升 collision/utilization；与现有 RKMeans 代码完全兼容 |
| P1 | IDEA-006 | LazyAR 解码器 | 与 ARCHITECTURE.md Lazy Decoder-Only 方向一致；扩展 token 数或 beam 后必需 |
| P1 | IDEA-007 | Value-Aware 训练 | 丰富训练信号，与 IDEA-002 互补；eCPM token 实现简单 |
| P1 | IDEA-009 | Dynamic Beam Search | 生产 beam=512 时必需；可与 IDEA-005 配套 |
| P2 | IDEA-008 | RSPO 排序优化 | 收益最大但前置依赖最重；建议系统成熟后再做 |
