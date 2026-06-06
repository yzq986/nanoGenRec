# RKMeans hyperparameter grid search results

[English](report.md) | [Chinese](report.zh.md)

**Generated**: 2026-04-13 11:05
**Model**: qwen3-0.6b (1024d)
**Fixed parameters**: 3 layers, normalize_residuals=True
**#Experiments**: 1

---

## 1. Complete results

| # | clusters | niter | nredo | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |
|---|----------|-------|-------|-----------|----------|------------|--------|----------|----------|---------|
| 1 | 1024 | 25 | 3 | 0.1634 | 3.69e-03 | 0.3524 | 5041.6 | 22.4 | 1.3 | 237 |

---

## 2. Optimal configuration (Top 5 sorted by collision)

**#1**: clusters=1024, niter=25, nredo=3, collision=0.1634, time=237s
