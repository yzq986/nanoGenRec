# Experiment Log

按时间倒序记录。每次实验链接到 `experiments/` 下的结果目录。

---

## Template

<!--
复制以下模板创建新实验记录。编号递增，最新的放在最上面。

## EXP-NNN: (实验标题)

**Date**: YYYY-MM-DD
**Status**: planned | running | completed
**Results**: [./hyperparam/YYYY-MM-DD_xxx/](./hyperparam/YYYY-MM-DD_xxx/)

### Background
(当前状态、要解决的问题)

### Hypothesis
(预期结果及原因)

### Design
- **Variable**: ...
- **Fixed**: ...
- **Metric**: ...
- **Data**: ...

### Results
(跑完后填写，含表格)

### Analysis
(结果解读)

### Next Steps
(下一步计划)
-->

---

## EXP-001: RKMeans 训练优化 (v0→v7)

**Date**: 2026-03 ~ 2026-04
**Status**: completed
**Results**: See `config/RKMEANS_EXPERIMENT_LOG.md` for full details

### Background
RKMeans 生成 semantic_id 碰撞率极高（99%+），需要系统性优化。

### Key Findings
1. **normalize_residuals 只对 layer 0 输入做** — 残差保留原始 scale，否则 Layer 2/3 无法聚类
2. **FAISS full-batch Lloyd's 优于 SGD/MiniBatch** — 空 cluster rebalance + GPU 加速
3. **num_clusters 是唯一显著超参** — collision 与 clusters 呈 log-linear 关系，每翻倍降 50-70%
4. **nredo=3 足够，niter=25 已收敛** — nredo 1→3 关键 (-42~49%), 3→5 无意义; niter 25/50/100 无差异

### Final Config
- 3 layers × 1024 clusters, niter=25, nredo=3
- collision: 1.75%, reconstruction_loss: 0.348

---
