## EXP-044C: TO-RoPE Item-Pos Fix + 3-dim RoPE

[English](exp-044c.md) | [中文](exp-044c.zh.md)

**Date**: 2026-04-29
**Status**: completed

### Background
EXP-044B best (ts=0.25) R@500=63.6%，但 PPL=467。两个假设：
1. position-RoPE 用 token-level index (0,1,2,3,4,5…)，但 time-RoPE 把同一 item 内所有 token 视为同时 → 冲突信号，可能导致 PPL 异常高。修复：用 item-level position (`pos//L`)。
2. SID layer index (0/1/2，即 pos%L) 加为第3个 RoPE 维度，让 attention 直接感知层间距离。

### Design
- **Fixed**: s-tier, 0.6b SID, ntp_data=exp044b-0.6b-14d（timestamps 已接通）
- **Variable**: torope_time_split, torope_layer_split

| Config | 描述 | torope_time_split | torope_layer_split |
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

1. **Item-pos fix 对 R@500 无显著改善**：A vs 044B best 几乎持平（63.5% vs 63.6%）。position 冲突假设未得到验证，或影响太小。

2. **ts=0.5 略优于 ts=0.25**（B=63.9% vs A=63.5%），与 044B 结论一致（B=63.5% > C=62.3%，注：044B B/C 已修复）。更多 head_dim 留给 time-RoPE 有一定帮助。

3. **3-dim RoPE 没有改善，反而略有下降**：C=62.4%，D=63.7%（D 与 B 基本持平）。将 layer 作为第 3 个 RoPE 维度没有带来收益，可能原因：SID layer（0/1/2）信息量有限，挤占 head_dim 的代价大于收益。

4. **PPL 持续偏高且随 layer_split 增加**：pos fix 后 PPL 反而更高（613 vs 467），说明 item-level position 对模型来说更难优化（语言模型任务里 token-level position 更自然）。PPL 高与 R@500 好共存，说明 PPL 不是这个任务的好代理指标。

### Next Steps
- TO-RoPE 最优配置：2-dim ts=0.5（B，R@500=63.9%），作为新 baseline
- 3-dim RoPE 暂不推进
- 下一步优先：EXP-045 FSQ sweep（修复 behavior_path 对齐后重跑）

---
