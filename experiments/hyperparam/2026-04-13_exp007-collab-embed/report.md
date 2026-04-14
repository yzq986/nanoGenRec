# EXP-007: Collaborative Signal Enhanced Embedding (Contrastive Fine-tune)

**日期**: 2026-04-13 ~ 04-14
**模型**: Qwen3-Embedding-0.6B (full fine-tune)
**方法**: I2I contrastive learning (InfoNCE), 基于用户行为共现构造正样本对
**硬件**: 8 x A100, DDP
**IDEA**: sid-1 — 将协同信号注入 embedding

---

## 实验矩阵

| Config | 温度 τ | 学习率 | max_pairs | 状态 |
|--------|--------|--------|-----------|------|
| A | 0.05 | 1e-5 | 2M | 完整跑完 |
| B | 0.07 | 1e-5 | 1M | 提前终止 (HR@50 全程低于 A) |
| C | 0.05 | 3e-5 | 500K | 完整跑完 |

## 结果对比

| 指标 | Config A | Config B | Config C |
|------|----------|----------|----------|
| Final HR@50 | **0.0197** | 0.0148 (partial) | 0.0192 |
| HR@50 items | 49,437 | 26,174 | 30,352 |
| Final avg loss | 2.9016 | — | 2.868 |
| 训练时间 | 6756s (~1h53m) | killed | 1912s (~32min) |
| Loss plateau 起点 | ~step 800 | ~step 800 | ~step 400 |

## HR@50 曲线

```
HR@50
0.020 ┤                                          ● A final (0.0197)
      │                              ○ ─ ─ ─ ○      ○ C final (0.0192)
0.018 ┤                         ○
      │                    ○
0.016 ┤               ●
      │          △
0.014 ┤     ○    △
      │     △    △
0.012 ┤     △
0.010 ┤
      └──────────────────────────────────────────
       0    400  800  1200 1600 2000 ... 7200  step

  ● Config A (τ=0.05, lr=1e-5)
  △ Config B (τ=0.07, lr=1e-5)
  ○ Config C (τ=0.05, lr=3e-5)
```

## 关键发现

1. **三组实验 HR@50 天花板一致 (~0.02)**，均处于 poor 级别 (阈值 < 0.02)
2. **温度不是瓶颈**: τ=0.07 (Config B) 全面劣于 τ=0.05 (Config A)
3. **学习率影响收敛速度但不影响上限**: Config C (lr=3e-5) 用 1/4 数据、1/3 时间达到同等效果
4. **Loss 快速 plateau**: 所有 config 在 ~200K pairs 后 loss 就稳定在 ~2.5-2.7
5. **Caption reconstruction loss**: 已集成 LM head 监控 (frozen, no BP)，供后续分析

## 结论

**Contrastive fine-tune 的超参调优空间有限，HR@50 上限约 0.02。** 问题不在温度或学习率，而可能在更根本的层面:

- **I2I pair 质量**: 基于行为共现的正样本对噪声可能过大
- **负采样策略**: in-batch negatives 可能缺少 hard negatives
- **模型适配性**: Qwen3-Embedding 的 representation 可能不适合直接 I2I contrastive

## 推荐后续方向

如需突破 0.02，需从超参调优转向方法论变更 (待讨论)。
