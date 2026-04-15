# RKMeans + FSQ 超参数网格搜索结果

**生成时间**: 2026-04-15 09:25
**模型**: qwen3-0.6b (1024d)
**固定参数**: 3 layers, normalize_residuals=True
**量化器**: rkmeans_fsq (2 KMeans + 1 FSQ)
**FSQ configs**: 12d_4096
**实验数量**: 1

---

## 1. 完整结果

| # | clusters | L3 | niter | nredo | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |
|---|----------|----|-------|-------|-----------|----------|------------|--------|----------|----------|---------|
| 1 | 4096 | 12d_4096 | 25 | 3 | 0.0089 | 7.02e-06 | 0.3378 | 118.8 | 1.5 | 1.0 | 493 |

---

## 2. 最优配置 (按 collision 排序 Top 5)

**#1**: clusters=4096, niter=25, nredo=3, L3=FSQ(12d_4096), collision=0.0089, time=493s
