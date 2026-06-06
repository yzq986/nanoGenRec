## EXP-039B: ECPO on exp038b-hard-lam03-3ep-ep1 (Features RL link endpoint)

[English](exp-039b.md) | [Chinese](exp-039b.zh.md)

[English](exp-039b.md) | [Chinese](exp-039b.zh.md)

**Date**: 2026-04-29
**Status**: completed
**Results**: experiments/ntp_checkpoints/exp039b-ecpo-from-spdpo/

### Background

The final step of RL alignment link: exp036 SFT → EXP-037 SP-DPO → EXP-038B RF-DPO (ep1 best, R@500=62.1%) → ECPO of this experiment.
EXP-039 (from exp038-hard-lam03) has been skipped, starting directly from exp038b ep1 to take advantage of a better starting point.

### Hypothesis

ECPO (δ=0.1) reproduces the exp029 magnitude improvement (+4pp) on the features model, and the final R@500 is close to or exceeds exp020 SOTA (66.2%).

### Design

- **Variable**: ECPO δ=0.1, starting from exp038b-hard-lam03-3ep-ep1 (R@500=62.1%)
- **Fixed**: G=512, BehaviorReward+FormatReward, on-policy beam, grpo_weight=0.03, lr=1e-4, 8×L20X
- **Metric**: R@{10,500}, PPL
- **Data**: context pool from exp023-14d-features, behavior cache 2026-03-31

### Run
`bash experiments/scripts/exp-039b.sh --no-smoke`

### Results

| Config | R@10 | R@500 | PPL | Wall |
|--------|------|-------|-----|------|
| exp036-full-features (SFT) | - | ~53% | - | - |
| exp037-medium (SP-DPO, ref) | - | 62.1% | - | - |
| exp038b-ep1 (RF-DPO) | - | 62.1% | - | - |
| **exp039b-ecpo (this)** | **11.8%** | **65.7%** | **20.0** | **182min** |
| exp020-hard-lam03 (SOTA 无Feature) | 14.1% | 66.2% | 16.3 | - |

### Analysis

- ECPO increased from RF-DPO ep1 (62.1%) to **65.7%**, +3.6pp, consistent with exp029 improvement
- Only **0.5pp** away from featureless SOTA (66.2%), almost tied
- PPL=20.0 is higher than SOTA (16.3), indicating that there is still room for NTP quality of the features route
- behavior_mean reward from 0.574 → 0.630 (good convergence trend), coverage=98.8%
- **Conclusion**: The features RL link (SFT→DPO→ECPO) is effective, but the introduction of features brings PPL cost

### Next Steps
- EXP-040: RSFT (behavioral quality filtering) verifies whether the baseline can be improved during the SFT stage
- EXP-041: ENTP-Loss (exposure negative sample α sweep) to verify the L0 negative sample penalty effect

---
