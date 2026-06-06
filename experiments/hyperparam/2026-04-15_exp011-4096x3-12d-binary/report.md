# RKMeans + FSQ 4096x3-12d Binary Hyperparameter Search Results

[English](report.md) | [Chinese](report.zh.md)

**Generated**: 2026-04-15 12:01
**Model**: qwen3-0.6b (1024d)
**Fixed parameters**: 3 layers, normalize_residuals=True
**Quantizer**: rkmeans_fsq (2 KMeans + 1 FSQ)
**FSQ configs**: 12d_4096
**#Experiments**: 1

---

## 1. Complete results

| # | clusters | L3 | niter | nredo | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |
|---|----------|----|-------|-------|-----------|----------|------------|--------|----------|----------|---------|
| 1 | 4096 | 12d_4096 | 25 | 3 | 0.0076 | 7.03e-06 | 0.3374 | 118.8 | 1.5 | 1.0 | 747 |

---

## 2. Optimal configuration (Top 5 sorted by collision)

**#1**: clusters=4096, niter=25, nredo=3, L3=FSQ(12d_4096), collision=0.0076, time=747s
