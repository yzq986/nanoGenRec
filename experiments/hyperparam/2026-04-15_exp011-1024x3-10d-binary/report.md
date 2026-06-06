# RKMeans + FSQ 1024x3-10d Binary Hyperparameter Search Results

[English](report.md) | [Chinese](report.zh.md)

**Generated**: 2026-04-15 08:59
**Model**: qwen3-0.6b (1024d)
**Fixed parameters**: 3 layers, normalize_residuals=True
**Quantizer**: rkmeans_fsq (2 KMeans + 1 FSQ)
**FSQ configs**: 10d_1024
**#Experiments**: 1

---

## 1. Complete results

| # | clusters | L3 | niter | nredo | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |
|---|----------|----|-------|-------|-----------|----------|------------|--------|----------|----------|---------|
| 1 | 1024 | 10d_1024 | 25 | 3 | 0.0788 | 4.13e-04 | 0.3666 | 475.4 | 3.1 | 1.1 | 100 |

---

## 2. Optimal configuration (Top 5 sorted by collision)

**#1**: clusters=1024, niter=25, nredo=3, L3=FSQ(10d_1024), collision=0.0788, time=100s
