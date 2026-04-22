---
experiment: "EXP-022"
date: "2026-04-22"
decision: discard
confidence: high
---

## Summary
IDEA-onemall-0: In-batch contrastive loss (InfoNCE) added as auxiliary to NTP CE loss. Hypothesis: embedding space alignment would improve recall.

## Results
| Config | α | τ | dim | PPL | R@10 | R@500 | vs Baseline |
|--------|-----|------|-----|-----|------|-------|-------------|
| Baseline (EXP-016) | — | — | — | 27.05 | 9.9% | 58.5% | — |
| alpha001 | 0.01 | 0.07 | 128 | 27.89 | 10.3% | 59.2% | +0.7pp R@500, +0.84 PPL |
| alpha01 | 0.1 | 0.07 | 128 | 29.22 | 9.7% | 57.9% | -0.6pp R@500 |
| alpha05 | 0.5 | 0.07 | 128 | 29.04 | 9.7% | 56.3% | -2.2pp R@500 |
| dim256 | 0.01 | 0.07 | 256 | 29.66 | 10.1% | 58.8% | +0.3pp R@500 |
| temp005 | 0.01 | 0.05 | 128 | 28.16 | 10.1% | 58.2% | -0.3pp R@500 |

## Rationale
All 5 configs either match or degrade baseline recall. The best (α=0.01) gains only +0.7pp R@500 at cost of +0.84 PPL — not a worthwhile trade. Higher α values consistently hurt both PPL and recall. Contrastive gradient competes with NTP gradient and disrupts L2 token prediction. The idea is fundamentally at odds with our discrete SID generation setup where the decoder operates in token space, not continuous embedding space.

## Next Steps
- Mark IDEA-onemall-0 as "tested, negative result"
- Do NOT pursue further contrastive variants (temperature, dimension, detached backbone)
- Focus on training objective changes that preserve the autoregressive NTP loss structure (e.g., IDEA-genrec-0 Page-wise NTP)
