# RKMeans + FSQ 超参数网格搜索结果

**生成时间**: 2026-04-15 08:59
**模型**: qwen3-0.6b (1024d)
**固定参数**: 3 layers, normalize_residuals=True
**量化器**: rkmeans_fsq (2 KMeans + 1 FSQ)
**FSQ configs**: 10d_1024
**实验数量**: 1

---

## 1. 完整结果

| # | clusters | L3 | niter | nredo | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |
|---|----------|----|-------|-------|-----------|----------|------------|--------|----------|----------|---------|
| 1 | 1024 | 10d_1024 | 25 | 3 | 0.0788 | 4.13e-04 | 0.3666 | 475.4 | 3.1 | 1.1 | 100 |

---

## 2. 最优配置 (按 collision 排序 Top 5)

**#1**: clusters=1024, niter=25, nredo=3, L3=FSQ(10d_1024), collision=0.0788, time=100s
