## EXP-041B: ENTP-Loss v2 — Session-Level Negatives (behavior_v2 data)

[English](exp-041b.md) | [Chinese](exp-041b.zh.md)

**Date**: 2026-04-29
**Status**: completed (Conclusion: invalid, session granularity issue)
**Results**: experiments/ntp_checkpoints/exp041b-entp{005,01,02}/

### Background

EXP-041 The root cause of the failure is that the behavior data is replaced with `exposure_neg` (the user collection is different). Correct approach: Mainly use positive behavior sample sequences, and append unclicked items in the session as neg_l0. `export_behavior_v2.py` has exported this format (uid, session_id, iid, action_bitmap), n_seqs=1,745,799, has_neg_l0=True, entp_k=5.

### Hypothesis

The behavior_v2 data contains negative samples within the session. ENTP α=0.1 increases R@500 by +2~4pp from 59.0%.

### Design

- **Variable**: ENTP weight α ∈ {0.05, 0.1, 0.2}; α=0 directly quotes exp036-full-features
- **Fixed**: behavior_v2 data, time_gap+action_level+segment_emb, 4096×3 binary SID, 1 epoch
- **Baseline**: exp036-full-features (existing, no retraining)
- **Data**: feed_user_behavior_v2 (2026-03-18~03-31), n_seqs=1,745,799

### Results

| Config | R@10 | R@500 | PPL | Wall |
|--------|------|-------|-----|------|
| exp036-full-features (α=0, baseline) | 10.9% | 59.0% | 27.3 | 7min |
| exp041b-entp005 (α=0.05) | 7.8% | 44.9% | 49.7 | 1min |
| exp041b-entp01  (α=0.1)  | 7.7% | 46.5% | 51.3 | 1min |
| exp041b-entp02  (α=0.2)  | 8.0% | 46.1% | 50.4 | 1min |

### Analysis

**Conclusion: ENTP v2 is invalid, and the root cause is the wrong session granularity. **

1. **`df_4` is not session**: `exposed` CTE in `export_behavior_v2.py` uses `df_4 AS session_id`, but `df_4` is actually a single view ID of each `$AppExposure` event (an independent ID is refreshed each time an exposure is refreshed), not a user session ID.
2. **There is almost no negative sample space within the session**: Local verification shows that 98% of sessions have only 1 exposed item, 1.99% have 2, and 0 have more than 3. An exposure event = 1 item, the user clicked on that item, neg_candidates = 0.
3. **neg:pos = 1:0.01**: The coverage rate of neg_l0 in 1.75 million sequences is less than 1%, and ENTP loss is almost not triggered, which is equivalent to pure NTP training.
4. **PPL rises from 27 to 50**: The sequence quality of behavior_v2 data itself is worse than behavior (it may contain users with less behavior or join causes the sequence to change), resulting in a decrease in basic performance.

### Next Steps

- If you want to do ENTP, you need to redefine the session: aggregate multiple exposures into one session according to the time window (such as within 30 minutes), so that each session has multiple items that can distinguish positive and negative
- Or return to the original OneRec/DualGR solution: use user-level exposure to negative samples (non-session) and directly join behavior data

---
