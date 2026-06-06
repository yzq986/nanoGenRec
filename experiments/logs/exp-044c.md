## EXP-044C: TO-RoPE Item-Pos Fix + 3-dim RoPE

[English](exp-044c.md) | [Chinese](exp-044c.zh.md)

**Date**: 2026-04-29
**Status**: completed

### Background
EXP-044B best (ts=0.25) R@500=63.6%, but PPL=467. Two assumptions:
1. Position-RoPE uses token-level index (0,1,2,3,4,5…), but time-RoPE treats all tokens in the same item as simultaneous → conflict signals, which may cause PPL to be abnormally high. Fix: Use item-level position (`pos//L`).
2. SID layer index (0/1/2, i.e. pos%L) is added as the third RoPE dimension, allowing attention to directly perceive the distance between layers.

### Design
- **Fixed**: s-tier, 0.6b SID, ntp_data=exp044b-0.6b-14d (timestamps are connected)
- **Variable**: torope_time_split, torope_layer_split

| Config | Description | torope_time_split | torope_layer_split |
|--------|------|-------------------|--------------------|
| A | 2-dim + item-pos fix | 0.25 | 0.0 |
| B | 2-dim ts=0.5 + item-pos fix | 0.50 | 0.0 |
| C | 3-dim ts=0.25 layer=0.15 | 0.25 | 0.15 |
| D | 3-dim ts=0.25 layer=0.25 | 0.25 | 0.25 |

### Results

| Config | R@10 | R@500 | PPL |
|--------|------|-------|-----|
| 043 baseline (abs pos) | 8.5% | 49.5% | 52.0 |
| 044B best (ts=0.25, no pos fix) | 12.5% | **63.6%** | 467.5 |
| **044C-A**: 2-dim ts=0.25 + pos fix | 11.5% | 63.5% | 613.6 |
| **044C-B**: 2-dim ts=0.5 + pos fix | 11.8% | 63.9% | 589.2 |
| **044C-C**: 3-dim ts=0.25 layer=0.15 | 11.9% | 62.4% | 669.4 |
| **044C-D**: 3-dim ts=0.25 layer=0.25 | 12.5% | 63.7% | 706.9 |

### Analysis

1. **Item-pos fix has no significant improvement on R@500**: A vs 044B best is almost the same (63.5% vs 63.6%). The position conflict hypothesis is not verified, or the effect is too small.

2. **ts=0.5 is slightly better than ts=0.25** (B=63.9% vs A=63.5%), which is consistent with the conclusion of 044B (B=63.5% > C=62.3%, Note: 044B B/C has been fixed). More head_dim left to time-RoPE helps somewhat.

3. **3-dim RoPE did not improve, but decreased slightly**: C=62.4%, D=63.7% (D is basically the same as B). Using layer as the third RoPE dimension does not bring benefits. The possible reason is: SID layer (0/1/2) has limited information content, and the cost of occupying head_dim is greater than the benefits.

4. **PPL continues to be high and increases with layer_split**: After pos fix, PPL is higher (613 vs 467), indicating that item-level position is more difficult to optimize for the model (token-level position is more natural in language model tasks). The coexistence of high PPL and R@500 indicates that PPL is not a good proxy indicator for this task.

### Next Steps
- TO-RoPE optimal configuration: 2-dim ts=0.5 (B, R@500=63.9%), as the new baseline
- 3-dim RoPE will not be promoted yet
- Next step first: EXP-045 FSQ sweep (fix behavior_path alignment and rerun)

---
