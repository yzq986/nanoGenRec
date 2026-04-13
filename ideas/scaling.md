# Scaling (扩展性实验)

模型规模 vs 数据规模 vs 序列长度的 scaling law 研究，直接决定资源分配策略。

**影响范围**: `metrics/sid_prediction.py`, `model/train.py`, ARCHITECTURE.md (tier 设计)

---

## 演进路径

```
S-tier (39.5M params, 当前唯一实现)
└── IDEA-oneloc-4: Scaling Law 实验
    └── 序列长度收益 (13%+51%) >> 模型大小收益 (7%)
    └── 3×3 grid: 模型参数 × 序列长度
```

---

## IDEA-oneloc-4: Scaling Law — 序列长度 >> 模型大小

**优先级**: P0
**来源**: OneLoc §4.4 Hyperparameter Experiments
**状态**: 待讨论

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

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P0 | IDEA-oneloc-4 | Scaling Law: 序列长度 vs 模型大小 | 直接决定资源分配策略；OneLoc 显示序列长度收益 7x > 模型大小收益 |
