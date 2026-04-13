# RKMeans + FSQ 超参数网格搜索结果

**生成时间**: 2026-04-13 11:29
**模型**: qwen3-0.6b (1024d)
**固定参数**: 3 layers, normalize_residuals=True
**量化器**: rkmeans_fsq (2 KMeans + 1 FSQ)
**FSQ configs**: 4d_4096, 5d_4375, 6d_4096
**实验数量**: 3

---

## 1. 完整结果

| # | clusters | L3 | niter | nredo | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |
|---|----------|----|-------|-------|-----------|----------|------------|--------|----------|----------|---------|
| 1 | 1024 | 6d_4096 | 25 | 3 | 0.3330 | 1.09e-03 | 3.1280 | 5041.6 | 22.4 | 1.7 | 178 |
| 2 | 1024 | 4d_4096 | 25 | 3 | 0.5688 | 5.79e-04 | 2.2122 | 5041.6 | 22.4 | 3.0 | 182 |
| 3 | 1024 | 5d_4375 | 25 | 3 | 0.8157 | 7.75e-05 | 0.3800 | 5041.6 | 22.4 | 20.8 | 179 |

---

## 2. 最优配置 (按 collision 排序 Top 5)

**#1**: clusters=1024, niter=25, nredo=3, L3=FSQ(6d_4096), collision=0.333, time=178s

**#2**: clusters=1024, niter=25, nredo=3, L3=FSQ(4d_4096), collision=0.5688, time=182s

**#3**: clusters=1024, niter=25, nredo=3, L3=FSQ(5d_4375), collision=0.8157, time=179s
