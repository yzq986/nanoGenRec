# RKMeans + FSQ 超参数网格搜索结果

**生成时间**: 2026-04-15 08:53
**模型**: qwen3-0.6b (1024d)
**固定参数**: 3 layers, normalize_residuals=True
**量化器**: rkmeans_fsq (2 KMeans + 1 FSQ)
**FSQ configs**: 5d_1024
**实验数量**: 1

---

## 1. 完整结果

| # | clusters | L3 | niter | nredo | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |
|---|----------|----|-------|-------|-----------|----------|------------|--------|----------|----------|---------|
| 1 | 1024 | 5d_1024 | 25 | 3 | 0.1463 | 3.78e-04 | 0.3677 | 475.4 | 3.1 | 1.2 | 97 |

---

## 2. 最优配置 (按 collision 排序 Top 5)

**#1**: clusters=1024, niter=25, nredo=3, L3=FSQ(5d_1024), collision=0.1463, time=97s
