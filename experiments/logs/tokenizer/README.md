# Tokenizer 实验

Embedding → KMeans → FSQ codebook，生成 3-token Semantic ID（SID）。

## 当前最优

| 指标 | 值 | 来源 |
|------|----|------|
| **推荐 SID cache (0.6b)** | exp049-0.6b-nc8192-h128 | EXP-049 |
| **推荐 SID cache (4b)** | exp049-4b-nc8192-h128 | EXP-049 |
| Collision Rate (0.6b) | 0.42% | EXP-049 |
| Collision Rate (4b) | 1.28% | EXP-049 |
| Gini_d2 (0.6b) | 0.2375 | EXP-049 |
| Gini_d2 (4b) | 0.2530 | EXP-049 |
| snHR (0.6b best) | 0.0919 | EXP-049 |
| snHR (4b best) | 0.1307 | EXP-049 |
| **FSQ 配置** | 4096×3 binary `[2]×12` | EXP-012 |
| **Embedding** | Qwen3-0.6B or 4B (h=128) | EXP-049 |
| **num_clusters** | 8192 | EXP-049 |

代码文档见 [`tokenizer/README.md`](../../../tokenizer/README.md)。

## ⚠️ 已知问题

- **EXP-045 num_clusters bug**：所有 exp045-* SID cache 使用 `num_clusters=1024`（应为 4096），数据不可信，已由 EXP-049 重跑修复
- **8b embedding cache**：item_id 与 behavior data 对齐率仅 2.3%，需重建
- **4b collision 对 h 不敏感**：根源是 12d_4096 codebook 容量瓶颈，需增大 FSQ levels

## Proxy Metrics（无需训 NTP 评估 codebook）

| Metric | 含义 | 越低越好 |
|--------|------|---------|
| **Gini_d2** | L1+L2 prefix 分配均匀度（FORGE） | ✅ |
| Collision Rate | 共享 SID 的 item 比例 | ✅ |
| snHR | 语义邻居保留率（FORGE embedding HR） | ❌（越高越好）|

计算 Gini：`python -c "..." metrics/cluster_balance.py`（或直接用 `metrics/cluster_balance.ClusterBalanceMetric`）

## 实验列表

| EXP | Date | Status | 结论 |
|-----|------|--------|------|
| [001](.../exp-001.md) | 2026-03 | completed | RKMeans 训练优化 v0→v7 |
| [002](../exp-002.md) | 2026-04-13 | completed | ResKmeansFSQ — 2L RKMeans + 1L FSQ |
| [003](../exp-003.md) | 2026-04-13 | completed | Learned FSQ — MLP projection + ST training |
| [004](../exp-004.md) | 2026-04-13 | completed | OPQ Parallel Semantic IDs |
| [007](../exp-007.md) | 2026-04-13 | completed | Collaborative Signal Enhanced Embedding |
| [008](../exp-008.md) | 2026-04-14 | completed | FORGE Proxy 对比 — MLP-FSQ vs OPQ；snHR 决定性 |
| [009](../exp-009.md) | 2026-04-14 | completed | QFormer Tokenizer |
| [010](../exp-010.md) | 2026-04-15 | completed | NTP Baseline — MLP-FSQ SID 端到端（效果极差） |
| [011](../exp-011.md) | 2026-04-15 | completed | Codebook Size 消融 — 1024/4096 + OPQ |
| [012](../exp-012.md) | 2026-04-15 | completed | **Grid Search — 4096×3 binary 确认为赢家** |
| [026](../exp-026.md) | 2026-04-27 | completed | **0.6B/4B/8B SID cache 构建（14d data）** |
| [045](../exp-045.md) | 2026-04-29 | ⚠️ bug | FSQ h-dim sweep — num_clusters=1024 bug，待重跑 |
| [049](../exp-049.md) | 2026-04-30 | completed | **num_clusters × h × model sweep — nc=8192 决定性，h 无差异，推荐 exp049-{0.6b,4b}-nc8192-h128** |
