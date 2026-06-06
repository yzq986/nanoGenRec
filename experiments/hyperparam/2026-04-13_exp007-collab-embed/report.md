# EXP-007: Collaborative Signal Enhanced Embedding (Contrastive Fine-tune)

[English](report.md) | [Chinese](report.zh.md)

**Date**: 2026-04-13 ~ 04-14
**Model**: Qwen3-Embedding-0.6B (full fine-tune)
**Method**: I2I contrastive learning (InfoNCE), constructing positive sample pairs based on user behavior co-occurrence
**Hardware**: 8 x A100, DDP
**IDEA**: sid-1 — Inject co-signal into embedding

---

## Experiment matrix

| Config | 温度 τ | 学习率 | max_pairs | Status |
|--------|--------|--------|-----------|------|
| A | 0.05 | 1e-5 | 2M | 完整跑完 |
| B | 0.07 | 1e-5 | 1M | 提前终止 (HR@50 全程Low于 A) |
| C | 0.05 | 3e-5 | 500K | 完整跑完 |

## Result comparison

| 指标 | Config A | Config B | Config C |
|------|----------|----------|----------|
| Final HR@50 | **0.0197** | 0.0148 (partial) | 0.0192 |
| HR@50 items | 49,437 | 26,174 | 30,352 |
| Final avg loss | 2.9016 | — | 2.868 |
| Training时间 | 6756s (~1h53m) | killed | 1912s (~32min) |
| Loss plateau 起点 | ~step 800 | ~step 800 | ~step 400 |

## HR@50 Curve

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

## Key findings

1. **The HR@50 ceiling of the three groups of experiments is consistent (~0.02)**, all at the poor level (threshold < 0.02)
2. **Temperature is not a bottleneck**: τ=0.07 (Config B) is overall worse than τ=0.05 (Config A)
3. **The learning rate affects the convergence speed but does not affect the upper limit**: Config C (lr=3e-5) uses 1/4 data and 1/3 time to achieve the same effect
4. **Loss fast plateau**: All config loss will be stable at ~2.5-2.7 after ~200K pairs
5. **Caption reconstruction loss**: LM head monitoring (frozen, no BP) has been integrated for subsequent analysis

## in conclusion

**The hyperparameter tuning space of Contrastive fine-tune is limited, and the upper limit of HR@50 is about 0.02. ** The problem is not with temperature or learning rate, but probably at a more fundamental level:

- **I2I pair quality**: Positive samples based on behavioral co-occurrence may be too noisy
- **Negative sampling strategy**: in-batch negatives may lack hard negatives
- **Model Suitability**: The representation of Qwen3-Embedding may not be suitable for direct I2I contrastive

## Recommended follow-up direction

If you want to break through 0.02, you need to shift from hyperparameter tuning to methodological changes (to be discussed).
