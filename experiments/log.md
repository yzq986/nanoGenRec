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

## EXP-003: Learned FSQ — MLP projection + straight-through training

**Date**: 2026-04-13
**Status**: planned

### Background
EXP-002 证明 PCA 线性投影 + FSQ 劣于 KMeans baseline，核心瓶颈是 PCA 在残差空间信息丢失过大（1024D→4~6D 解释方差仅 20-55%）。

OneMall (arxiv 2601.21770) 用 **learned MLP** 做投影，原始 FSQ 论文 (Mentzer 2023, arxiv 2309.15505) 将 FSQ 用在 VQ-VAE 内部，encoder 学到对量化最优的表示。关键机制：
- MLP 学习非线性投影 D→d，比 PCA 保留更多量化相关信息
- Straight-Through Estimator (STE): 前向用 round()，反向把梯度直通到 MLP 参数
- 训练目标: 重建 loss — minimize ||residual - reconstruct(FSQ(MLP(residual)))||²

### Hypothesis
Learned MLP 投影可以学到对 FSQ 量化最优的低维表示，使 FSQ 的 reconstruction quality 接近或超过 KMeans，从而在保持低 collision 的同时降低 recon_loss。

### Design
- **Variable**: 投影方式 (PCA vs MLP)，MLP 隐层宽度，训练 epochs
- **Fixed**: 2 KMeans layers x 1024 clusters, FSQ [4,4,4,4,4,4] (6d_4096, EXP-002 最优 FSQ config)
- **Metric**: conflict_rate, reconstruction_loss, exclusivity, entropy

**MLP 架构** (autoencoder style):
```
Encoder: D → hidden → d  (d=6 for 6d_4096)
FSQ:     quantize each of 6 dims to {0,1,2,3}
Decoder: d → hidden → D
Loss:    ||residual - Decoder(FSQ(Encoder(residual)))||²
```
- Straight-through estimator: forward = round(), backward = identity
- hidden sizes to try: [64, 128, 256]
- epochs: 50, lr: 1e-3, Adam

**Comparison matrix**:

| Config | L3 projection | L3 codebook | Training |
|--------|---------------|-------------|----------|
| Baseline (EXP-002) | KMeans 1024 | 1024 | N/A |
| PCA-FSQ (EXP-002) | PCA 6d | 4096 | N/A |
| MLP-FSQ-64 | MLP D→64→6 | 4096 | 50 epochs |
| MLP-FSQ-128 | MLP D→128→6 | 4096 | 50 epochs |
| MLP-FSQ-256 | MLP D→256→6 | 4096 | 50 epochs |

### Run
`bash experiments/scripts/exp-003.sh`

### Results
TBD

### Analysis
TBD

### Next Steps
TBD

---

## EXP-002: ResKmeansFSQ — 2 layers RKMeans + 1 layer FSQ (PCA projection)

**Date**: 2026-04-13
**Status**: completed
**Results**: [./hyperparam/2026-04-13_exp002-baseline/](./hyperparam/2026-04-13_exp002-baseline/), [./hyperparam/2026-04-13_exp002-fsq/](./hyperparam/2026-04-13_exp002-fsq/)

### Background
RKMeans 的第3层对残差做 KMeans 效果递减。OneMall (arxiv 2601.21770) 提出用 FSQ 替换第3层。本实验使用 **PCA 线性投影** 替代论文中的 learned MLP 做降维。

### Hypothesis
FSQ 的 implicit codebook 天然无 cluster collapse，可降低 collision rate。

### Design
- **Variable**: Layer 3 quantizer (KMeans vs FSQ configs)
- **Fixed**: 2 KMeans layers x 1024 clusters, niter=25, nredo=3, normalize_residuals=True
- **Metric**: conflict_rate, reconstruction_loss, entropy, exclusivity, cluster_balance

| Config | L1, L2 (KMeans) | L3 | L3 codebook |
|--------|------------------|----|-------------|
| Baseline | 1024 x 3 layers KMeans | KMeans 1024 | 1024 |
| Hybrid A | 1024 x 2 layers | FSQ [8,8,8,8] | 4096 |
| Hybrid B | 1024 x 2 layers | FSQ [7,5,5,5,5] | 4375 |
| Hybrid C | 1024 x 2 layers | FSQ [4,4,4,4,4,4] | 4096 |

### Run
`bash experiments/scripts/exp-002.sh`

### Results

| Config | L3 | conflict_rate | exclusivity | recon_loss | d3 entropy | d3 Gini | d3 unique | d3 avg_items |
|--------|-----|---------------|-------------|------------|------------|---------|-----------|--------------|
| **Baseline** | KMeans 1024 | **0.1634** | **0.6423** | **0.3524** | **0.7211** | **0.2091** | 3,963,269 | 1.3 |
| Hybrid C | FSQ 6d [4x6] | 0.3330 | 0.4015 | 3.1280 | 0.6755 | 0.3153 | 3,107,671 | 1.7 |
| Hybrid A | FSQ 4d [8x4] | 0.5688 | 0.1446 | 2.2122 | 0.6383 | 0.4693 | 1,731,222 | 3.0 |
| Hybrid B | FSQ 5d [7,5x4] | 0.8157 | 0.0089 | 0.3800 | 0.5306 | 0.6548 | 248,798 | 20.8 |

注: L1/L2 两层 KMeans 相同 (d1/d2 指标一致)，差异全部来自 L3。

### Analysis

**FSQ+PCA 全面劣于 KMeans baseline**，核心原因是 **PCA 线性投影信息丢失过大**：

1. **投影瓶颈**: 1024维残差 → 4~6维 PCA，解释方差仅 20-55%。残差空间（经两轮 KMeans 后）本就小且不规则，PCA 线性假设不适用。
2. **维度越少越差**: 5d_4375 (d=5) 的 conflict_rate 高达 0.82，几乎所有信息丢失；6d_4096 (d=6) 最好但仍 0.33 >> baseline 0.16。
3. **recon_loss 恶化**: 4d/6d 的 recon_loss 从 0.35 飙升到 2.2/3.1，说明 PCA 逆投影无法恢复原始残差。
4. **与论文差异**: OneMall 用 **learned MLP** 投影（非线性、端到端训练），可学到对量化最优的表示，而非仅保留方差最大方向。

### Next Steps
EXP-003: 将 PCA 替换为 learned MLP 投影，复现论文方案。需要：
1. 定义 MLP 架构 (D → d 维) + VQ-VAE style 重建 loss
2. 端到端训练投影网络
3. 对比 PCA vs MLP 在同一 FSQ config 下的效果

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
