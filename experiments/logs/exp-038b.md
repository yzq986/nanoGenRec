## EXP-038B: RF-DPO on exp037-medium — ntp_epochs=3 + mid-checkpoints

[English](exp-038b.md) | [Chinese](exp-038b.zh.md)

**Date**: 2026-04-28
**Status**: completed
**Results**: experiments/ntp_checkpoints/exp038b-hard-lam03-3ep-ep1/ (best)

### Background

After EXP-038 RF-DPO (1 epoch, 406 steps), R@500=59.6%, PPL=25.7, 2.5pp degraded compared to ref (exp037-medium 62.1%). Cause analysis: The number of steps is too few (406 steps), and the NTP:DPO ratio is not aligned with the exp019/020 design (exp020 target is 807 DPO steps ≈ NTP steps).

EXP-038B uses `--ntp_epochs 3` (total 1218 steps) and saves mid-checkpoint at each epoch boundary to compare the effect of ep1/ep2/ep3(final).

**Code implementation**: Added `ntp_epochs` parameter (`itertools.chain.from_iterable(itertools.repeat(ntp_loader, ntp_epochs))`), mid-checkpoint is saved to `{output_dir}-ep{N}` at the end of each epoch.

### Hypothesis

ep1 (406 steps) = 1 epoch of alignment to exp038, expected to be comparable to EXP-038 (~59.6%). More epochs may improve DPO alignment but risk NTP overfitting.

### Design

- **Variable**: ntp_epochs ∈ {1,2,3} (three-point comparison is achieved through mid-checkpoint)
- **Fixed**: ref=exp037-medium, λ=0.03, β=0.1, difficulty=hard, lr=1e-4, Joint NTP+DPO
- **Metric**: R@{10,500}, PPL (three epochs for each review)
- **Data**: RF-DPO pairs from exp018 real feedback (2026-03-18~03-31), 4,312 hard pairs

### Run
`bash experiments/scripts/exp-038b.sh`

### Results

| Checkpoint | Steps | R@10 | R@500 | PPL | Conclusion |
|---|---|---|---|---|---|
| exp037-medium (ref) | — | 11.2% | 62.1% | 23.0 | SP-DPO starting point |
| **ep1 (1 epoch)** | 406 | **11.2%** | **62.1%** | **23.6** | ✅ Flat ref, DPO lossless |
| ep2 (2 epochs) | 812 | 10.3% | 59.6% | 26.0 | ❌ NTP starts to overfit |
| final (3 epochs) | 1218 | 9.3% | 52.8% | 33.3 | ❌ Severe overfitting |

**Best checkpoint**: `exp038b-hard-lam03-3ep-ep1` (ep1, R@500=62.1%)

### Analysis

1. **ep1 flat ref (no degradation!)**: The reason why EXP-038 1 epoch degrades to 59.6% may be that the LR is too high or the training is unstable, while EXP-038B ep1 gets 62.1% with the same number of steps, indicating that the impact of DPO on NTP is neutral within 1 epoch.

2. **2/3 epoch NTP overfitting**: NTP loss begins to overfit after multiple cycles on the exp018 real feedback data (narrow distribution), and PPL deteriorates rapidly from 23.6 → 26.0 → 33.3.

3. **Key Lessons**: The optimal RF-DPO is 1 epoch; `--ntp_epochs` should be set to 1 (experimentally verified). Subsequent experiments used ep1 as the ECPO starting point.

### Next Steps
- EXP-039B: ECPO on ep1 (`exp038b-hard-lam03-3ep-ep1`), δ=0.1, G=512, on-policy beam

---
