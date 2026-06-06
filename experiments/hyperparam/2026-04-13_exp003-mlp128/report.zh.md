# RKMeans + Learned FSQ h=128 Hyperparameter Search Results

[English](report.md) | [中文](report.zh.md)

**Generated**: 2026-04-13 15:22
**Model**: qwen3-0.6b (1024d)
**固定参数**: 3 layers, normalize_residuals=True
**Quantizer**: rkmeans_fsq (2 KMeans + 1 FSQ)
**FSQ configs**: 6d_4096
**# Experiments**: 1

---

## 1. 完整结果

| # | clusters | L3 | niter | nredo | collision | N^L util | recon_loss | d1 avg | d2 avg | d3 avg | time(s) |
|---|----------|----|-------|-------|-----------|----------|------------|--------|----------|----------|---------|
| 1 | 1024 | 6d_4096 | 25 | 3 | 0.0767 | 1.09e-03 | 0.3633 | 5041.6 | 22.4 | 1.1 | 627 |

---

## 2. 最优配置 (按 collision 排序 Top 5)

**#1**: clusters=1024, niter=25, nredo=3, L3=FSQ(6d_4096), collision=0.0767, time=627s
