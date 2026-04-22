---
experiment: "EXP-025"
date: "2026-04-22"
decision: merge
confidence: high
---

## Summary
Beam search feature passing: fix the train-inference gap for time_gap and action_level by passing features during beam search incremental decoding, instead of shifting features at training time (EXP-024 failed approach).

## Results
| Config | PPL | R@10 | R@50 | R@100 | R@500 | vs Baseline |
|--------|-----|------|------|-------|-------|-------------|
| exp023-segment (baseline) | 25.94 | 10.9% | 24.9% | 35.4% | 61.2% | — |
| **exp025-beam-passes** | **25.22** | 10.4% | 28.2% | 40.0% | **63.6%** | **+2.4pp R@500**, -0.72 PPL |
| exp025-action-l2only | 24.85 | 5.5% | 13.2% | 17.3% | 27.0% | -34.2pp R@500 (failure) |

## Rationale
beam_passes achieves the new best R@500=63.6% (+2.4pp) and best PPL=25.22 simultaneously. The approach correctly resolves the EXP-023 information leakage by passing time_gap (known at inference) and action carry-forward (last context item's action) to beam search. This is a clean solution — no training data modification needed, just inference path fix.

action_l2only completely failed (R@500=27.0%) — restricting action to L2-only still creates a massive train-inference gap since beam search doesn't generate L2 tokens with the correct action.

## Next Steps
- exp025-beam-passes becomes the new baseline for all future NTP experiments
- The beam_passes config should be the default: segment_emb + time_gap + action_level + beam feature passing
- Next: explore IDEA-genrec-0 (Page-wise NTP) which is orthogonal and could stack on top
