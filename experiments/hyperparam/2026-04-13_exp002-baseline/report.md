# RKMeans 超参数网格搜索结果

**生成时间**: 2026-04-13 11:05
**模型**: qwen3-0.6b (1024d)
**固定参数**: 3 layers, normalize_residuals=True
**实验数量**: 1

---

## 1. 完整结果

| # | clusters | niter | nredo | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |
|---|----------|-------|-------|-----------|----------|------------|--------|----------|----------|---------|
| 1 | 1024 | 25 | 3 | 0.1634 | 3.69e-03 | 0.3524 | 5041.6 | 22.4 | 1.3 | 237 |

---

## 2. 最优配置 (按 collision 排序 Top 5)

**#1**: clusters=1024, niter=25, nredo=3, collision=0.1634, time=237s
