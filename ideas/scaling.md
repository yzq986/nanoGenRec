# Scaling (扩展性实验)

模型规模 vs 数据规模 vs 序列长度的 scaling law 研究，直接决定资源分配策略。

**影响范围**: `metrics/sid_prediction.py`, `model/train.py`, ARCHITECTURE.md (tier 设计)

---

## 演进路径

```
S-tier (39.5M params, 当前唯一实现)
├── IDEA-oneloc-4: Scaling Law 实验
│   ├── 模型 scaling → EXP-015 ✅ L(N)=2.522+2055/N^0.456, ~100M 趋平
│   │   └── M+ 101M vs S 17.5M: loss 仅降 0.06, tokenizer 是瓶颈
│   └── 序列长度 scaling → 待实验 (当前 max_seq_len=512, ~170 items/user)
├── IDEA-kunlun-0: Rec Scaling Laws (Meta Ads)
│   └── MFU 17%→37%, GDPA + CompSkip, power-law scaling
│   └── 重要性提升: tokenizer 瓶颈突破后成为 scale up 关键
└── IDEA-hstu-0: Sparse Self-Attention Co-design (Meta)
    └── 5x 训练 / 21x 推理 scaling, 保留 self-attention 表达力
```

---

## 当前结论 (2026-04-17)

**模型参数 scaling 在 ~100M 趋平，tokenizer 32-bit 编码是当前瓶颈。序列长度 scaling 尚未验证。**

### 关键实验数据

| 模型 | Active Params | Eval Loss | PPL | R@500 |
|------|--------------|-----------|-----|-------|
| S-tier | 17.5M | 2.9960 | 27.05 | 58.5% |
| M+-tier | 101M | 2.9371 | 25.12 | 60.7% |
| Irreducible (fit) | ∞ | 2.522 | ~12.5 | — |

**核心 insight**: M+ 比 S 多 6x 参数，但 loss 仅降 0.06。Scaling law 拟合 L(N)=2.522+2055/N^0.456 显示 irreducible loss=2.522 由 tokenizer 决定。突破瓶颈需要：(1) 更高 bits 的 SID (当前 32-bit)；(2) 更长序列 (尚未验证)。

---

## IDEA-oneloc-4: Scaling Law — 序列长度 >> 模型大小

**优先级**: ~~P0~~ → 部分完成 (模型 scaling 已验证, 序列长度待测)
**来源**: OneLoc §4.4 Hyperparameter Experiments
**状态**: EXP-015 模型 scaling 完成; 序列长度 scaling 待实验

> **NTP 阶段更新 (2026-04-17)**: EXP-015 验证了模型参数 scaling law L(N)=2.522+2055/N^0.456。关键发现: M+ (101M active) 比 S (17.5M) 仅降低 loss 0.06，scaling law 在 ~100M 已严重平坦化——tokenizer 32-bit 编码是瓶颈。序列长度维度的 scaling 尚未实验 (当前 max_seq_len=512, ~170 items/user)，这是下阶段重要方向。

### 核心思想

OneLoc 的 scaling 实验揭示了一个关键发现: **序列长度的收益远大于模型大小的收益**。模型从 0.05B 扩到 0.3B，recall/NDCG 平均提升 7%；但序列长度从 100 扩到 300，recall 提升 13%、NDCG 提升 51%。这意味着在资源有限时，应优先增加序列长度而非模型参数。

### 与当前项目的关联

- 当前 `AutoregressiveNTPModel` S-tier config: 6 layers, 256 embed_dim, ~39.5M params
- ARCHITECTURE.md 定义了 M/L tier 但未实现
- **关键问题**: 我们尚未做过 NTP 模型的 scaling 实验
- 更直接的启发: 在 NTP 训练中，user 行为序列的长度是否比模型大小更重要?
- 当前行为序列处理: `data/export_behavior.py` 中导出的序列长度是多少? 是否足够长?

### 实验设计草案

**实验矩阵**:

| 维度 | 小 | 中 | 大 |
|------|-----|-----|-----|
| 模型参数 | S-tier (39.5M) | M-tier (~150M) | L-tier (~500M) |
| 序列长度 | 50 | 100 | 200 |

**设计**:
- 固定量化方案 (RKMeans 3x1024 或 OPQ)
- 3x3 grid: 模型大小 x 序列长度
- 每组训练 NTP 模型到收敛
- 记录 recall@5/10/20, NDCG@5/10/20

**评估**:
- 绘制 scaling curve: recall vs 模型参数 (固定序列长度)
- 绘制 scaling curve: recall vs 序列长度 (固定模型参数)
- 验证 OneLoc 的结论是否在我们的场景复现

### 关键问题

1. **前置依赖**: 需要先有稳定的 NTP 训练 pipeline + 稳定的量化方案
2. 当前 NTP 训练是否已经可以 end-to-end run? 需要确认 `model/train.py` → NTP 的完整流程
3. 行为数据量: 序列长度 200 需要足够的用户行为数据
4. 计算成本: 9 组实验，每组可能需要数小时训练
5. **为什么 P0**: 这个实验的结论直接决定资源分配策略 — 是花钱买更大 GPU 还是花钱采集更多行为数据

---

## IDEA-kunlun-0: Recommendation Scaling Laws (MFU 优化 + GDPA)

**优先级**: P1 — 重要性因 scaling 平坦化而提升
**来源**: Kunlun (Meta Ads, arxiv 2602.10016, Feb 2026)
**状态**: 待讨论

> **NTP 阶段更新 (2026-04-17)**: EXP-015 显示模型 scaling 在 ~100M 已趋平 (tokenizer 瓶颈)。Kunlun 的 MFU 优化和 GDPA 在当前阶段的价值不在于 scale up 模型，而在于：(1) 提高现有模型的训练效率 (更快迭代实验)；(2) 未来突破 tokenizer 瓶颈后 (如 OPQ 长 SID 或更高 bits)，GDPA 是 scale up 的关键技术。

### 核心思想

Kunlun 在大规模推荐系统中建立了类 LLM 的 **power-law scaling laws**。核心发现: 推荐模型 scaling 效率低的根本原因是 **低 MFU (Model FLOPs Utilization)** 和 **资源分配不均**。

解决方案:
1. **Generalized Dot-Product Attention (GDPA)**: 推荐专用的注意力机制
2. **Hierarchical Seed Pooling (HSP)**: 高效特征聚合
3. **Computation Skip (CompSkip)**: 选择性计算，跳过低价值路径
4. **Sliding Window Attention**: 管理用户历史序列

**结果**: MFU 从 **17% 提升到 37%** (B200 GPU), **2x scaling efficiency**, 部署到 Meta Ads 主要模型。

### 与当前项目的关联

- 当前 S-tier 模型小 (39.5M)，MFU 不是瓶颈
- 但 IDEA-oneloc-4 (Scaling Law) 和 IDEA-plum-0 (LLM CPT) 都需要 scale up → Kunlun 的经验直接适用
- **GDPA** 可能比标准 attention 更适合推荐场景: 用户行为序列与自然语言序列的模式不同
- **CompSkip** 与 IDEA-gr4ad-1 (LazyAR) 有关联: 都是选择性计算

### 实验设计草案

**Phase 1 — GDPA 替换标准 Attention**:
- 需要读 Kunlun 论文全文了解 GDPA 具体定义
- 在 `CausalTransformerLayer` 中替换 attention 模块

**Phase 2 — MFU Profiling**:
- 在 8xA100 上 profile 当前 NTP 训练的 MFU
- 识别低效模块 → targeted 优化

### 关键问题

1. GDPA 的具体实现需要论文全文
2. 当前模型太小，MFU 提升不等于训练速度提升 (可能是 memory-bound 而非 compute-bound)
3. CompSkip 需要 per-sample 路由 → 实现复杂度高

---

## IDEA-hstu-0: Sparse Self-Attention + Model-System Co-design (ULTRA-HSTU)

**优先级**: P1
**来源**: ULTRA-HSTU (Meta, arxiv 2602.16986, Feb 2026)
**状态**: 待讨论

### 核心思想

ULTRA-HSTU 通过 **end-to-end model-system co-design** 实现:

1. **Input Sequence Design**: 针对推荐场景优化输入序列构造
2. **Sparse Attention**: 保持 self-attention 的表达能力的同时避免 O(n²) 计算
3. **Model Topology**: 架构拓扑优化以配合系统效率

关键立场: cross-attention (如 IDEA-onemall-1 Query-Former) 虽然解决了 O(n²) 问题，但 **限制了 self-attention 的表达能力**。ULTRA-HSTU 通过 sparse self-attention 既保持表达能力又控制计算量。

**结果**: **5x faster training, 21x faster inference**, 服务 **数十亿用户**, **4-8% engagement improvement**。

### 与当前项目的关联

- 当前 `CausalTransformerLayer` 是 full self-attention (O(n²))，序列短时无问题
- 如果扩展到长序列 (IDEA-oneloc-4 / IDEA-onemall-1):
  - IDEA-onemall-1 选择 cross-attention (Query-Former) → 压缩表达
  - ULTRA-HSTU 选择 sparse self-attention → 保留表达
  - 两种路线的 tradeoff 值得实验对比
- **Model-System Co-design** 的理念: 不要只看模型质量，要同时优化系统效率

### 实验设计草案

**Phase 1 — Sparse Attention 替换**:
- 在 `CausalTransformerLayer` 中加入 sparse attention 选项 (如 sliding window + global tokens)
- 对比: full attention vs sparse attention vs Query-Former 在不同序列长度下的 Recall@K 和训练速度

**Phase 2 — Input Sequence Design**:
- 需要读论文全文了解 ULTRA-HSTU 的 input sequence design 细节
- 可能涉及 action 类型 encoding、时间戳 encoding 等

### 关键问题

1. 论文全文细节 (sparse attention 的具体 pattern) 需要补充
2. 当前序列短 (3 SID tokens)，sparse attention 无收益 → 依赖序列扩展
3. 与 IDEA-onemall-1 (Query-Former) 的对比实验需要统一实验框架

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| ~~P0~~ 部分完成 | IDEA-oneloc-4 | Scaling Law: 序列长度 vs 模型大小 | 模型 scaling EXP-015 ✅ (~100M 趋平); 序列长度 scaling 待验证 |
| P1 | IDEA-kunlun-0 | Rec Scaling Laws (MFU + GDPA) | Meta Ads 部署验证; tokenizer 瓶颈突破后成 scale up 关键 |
| P1 | IDEA-hstu-0 | Sparse Self-Attention Co-design | 21x inference scaling, 对比 Query-Former 路线 |
