# Inference (推理优化)

Beam search 和解码策略的优化，提升推理吞吐和候选质量。在模型规模和 beam 扩大后变得关键。

**影响范围**: `metrics/sid_prediction.py` (beam search 逻辑)

---

## 演进路径

```
固定 beam search (当前 beam=5, 全 vocab softmax)
└── IDEA-gr4ad-4: Dynamic Beam Search
    ├── DBW: 逐步增大 beam (128→256→512)
    └── TopK Pre-Cut: 每 beam 先选 top-b → 全局 top-k
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

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P1 | IDEA-gr4ad-4 | Dynamic Beam Search | 生产 beam=512 时必需；可与 IDEA-gr4ad-0 配套 |
