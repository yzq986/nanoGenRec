## EXP-041B: ENTP-Loss v2 — Session-Level Negatives (behavior_v2 数据)

[English](exp-041b.md) | [中文](exp-041b.zh.md)

**Date**: 2026-04-29
**Status**: completed (结论: 无效，session 粒度问题)
**Results**: experiments/ntp_checkpoints/exp041b-entp{005,01,02}/

### Background

EXP-041 失败根因是用 `exposure_neg` 替换了 behavior 数据（用户集合不同）。正确做法：以 behavior 正样本序列为主，附加 session 内未点击 item 作为 neg_l0。`export_behavior_v2.py` 已导出此格式（uid, session_id, iid, action_bitmap），n_seqs=1,745,799，has_neg_l0=True，entp_k=5。

### Hypothesis

behavior_v2 数据包含 session 内负样本，ENTP α=0.1 使 R@500 从 59.0% 提升 +2~4pp。

### Design

- **Variable**: ENTP weight α ∈ {0.05, 0.1, 0.2}；α=0 直接引用 exp036-full-features
- **Fixed**: behavior_v2 数据，time_gap+action_level+segment_emb，4096×3 binary SID，1 epoch
- **Baseline**: exp036-full-features（已有，不重训）
- **Data**: feed_user_behavior_v2 (2026-03-18~03-31)，n_seqs=1,745,799

### Results

| Config | R@10 | R@500 | PPL | Wall |
|--------|------|-------|-----|------|
| exp036-full-features (α=0, baseline) | 10.9% | 59.0% | 27.3 | 7min |
| exp041b-entp005 (α=0.05) | 7.8% | 44.9% | 49.7 | 1min |
| exp041b-entp01  (α=0.1)  | 7.7% | 46.5% | 51.3 | 1min |
| exp041b-entp02  (α=0.2)  | 8.0% | 46.1% | 50.4 | 1min |

### Analysis

**结论：ENTP v2 无效，根本原因是 session 粒度错误。**

1. **`df_4` 不是 session**：`export_behavior_v2.py` 里 `exposed` CTE 用 `df_4 AS session_id`，但 `df_4` 实际是每条 `$AppExposure` 事件的单个 view ID（每次刷新一条曝光一个独立 ID），不是用户会话 ID。
2. **session 内几乎无负样本空间**：本地验证显示 98% 的 session 只有 1 个曝光 item，1.99% 有 2 个，0 个有 3 个以上。一次曝光事件 = 1 个 item，用户点了那个 item，neg_candidates = 0。
3. **neg:pos = 1:0.01**：175 万序列中 neg_l0 覆盖率不足 1%，ENTP loss 几乎不触发，等同于纯 NTP 训练。
4. **PPL 从 27 升至 50**：behavior_v2 数据本身的序列质量比 behavior 差（可能包含了行为较少的用户或 join 导致序列发生变化），导致基础性能下降。

### Next Steps

- 若要做 ENTP，需重新定义 session：按时间窗口（如 30min 内）将多次曝光聚合为一个 session，这样每个 session 才有多个 item 可以区分正负
- 或者回归 OneRec/DualGR 原始方案：用 user-level 曝光负样本（非 session 内），直接 join behavior 数据

---
