# RKMeans + MLP-FSQ h=64 Hyperparameter Search Results

[English](report.md) | [Chinese](report.zh.md)

**Generated**: 2026-04-15 02:51
**Model**: qwen3-0.6b (1024d)
**Fixed parameters**: 3 layers, normalize_residuals=True
**Quantizer**: rkmeans_fsq (2 KMeans + 1 FSQ)
**FSQ configs**: 6d_4096
**#Experiments**: 1

---

## 1. Complete results

| # | clusters | L3 | niter | nredo | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |
|---|----------|----|-------|-------|-----------|----------|------------|--------|----------|----------|---------|
| 1 | 1024 | 6d_4096 | 25 | 3 | 0.1074 | 1.10e-04 | 0.3668 | 475.4 | 3.1 | 1.1 | 106 |

---

## 2. Optimal configuration (Top 5 sorted by collision)

**#1**: clusters=1024, niter=25, nredo=3, L3=FSQ(6d_4096), collision=0.1074, time=106s
