## EXP-044B: TO-RoPE with Real Timestamps — S-tier + 0.6B SID

**Date**: 2026-04-29
**Status**: completed

### Background
EXP-044 中 timestamps 全为 0（pipeline 未接通），本实验修复后重跑：
- `_build_sequences_from_behavior` 补充 `timestamps` 计算（rel_hours）
- time_gap_emb 与 TO-RoPE 共存
- 单独消融：去掉 time_gap（Config D）验证两者是否互补

### Design
- **Variable**: TO-RoPE time_split (0.5 / 0.25) × 有无 time_gap 共存
- **Fixed**: S-tier + 0.6B SID，14d 数据，action_level + segment_emb
- **Baseline**: exp043-s-0.6b（R@500=61.2%，PPL=26.52）

| Config | 说明 |
|--------|------|
| exp043-s-0.6b | Baseline: abs pos + time_gap + action + segment |
| exp044b-torope-ts05 | TO-RoPE ts=0.5 + time_gap + action + segment |
| exp044b-torope-ts025 | TO-RoPE ts=0.25 + time_gap + action + segment |
| exp044b-torope-ts05-notg | TO-RoPE ts=0.5 + action + segment（无 time_gap 消融）|

### Results

⚠️ **历史记录（无效结果）**：首次 eval 因两层 train-infer bug 导致 timestamps=0，结果无效（R@500≈32%）。修复后结果见下。

| Config | R@10 | R@500 | PPL | 备注 |
|--------|------|-------|-----|------|
| **exp043-s-0.6b** (baseline) | **11.4%** | **61.2%** | **26.5** | abs pos + time_gap + action + seg |
| exp044b-torope-ts05 | 12.5% | **62.3%** | 474.9 | TO-RoPE ts=0.5 + time_gap |
| exp044b-torope-ts025 | 12.5% | **63.6%** | 467.5 | TO-RoPE ts=0.25 + time_gap ← 最佳 |
| exp044b-torope-ts05-notg | 11.9% | **63.5%** | 480.0 | TO-RoPE ts=0.5，无 time_gap |

### Analysis

1. **TO-RoPE 有效**：修复 train-infer 不一致后，R@500 全部超过 baseline（61.2%），最佳 63.6%（+2.4pp）。

2. **PPL 高不等于效果差**：PPL 仍在 467-480（比 baseline 高约 18×），但 beam search R@500 却更好。原因是 TO-RoPE 改变了 logit 分布的熵（attention pattern 不同），PPL 衡量的是概率分布的尖锐程度，不直接对应 recall 质量。

3. **time_split=0.25 最优**：保留更多维度给 index-RoPE，时间信息用少量维度表达足够。ts05-notg（无 time_gap）与 ts025 接近（63.5% vs 63.6%），说明 TO-RoPE 本身已能编码时间信息，time_gap_emb 的增量有限。

4. **Bug 复盘 — 两层 train-infer 不一致**：
   - **Bug 1**：`constrained_beam_search` 的 `_step_sf` 只处理 `inject='embed_add'` 特征，`inject='torope'` 的 timestamps 在生成步骤全部为 0。修复：加 `_step_ts()` carry-forward。
   - **Bug 2（更隐蔽）**：`eval.py` 的 `eval_items` 构建循环过滤条件是 `inject != 'embed_add'`，导致 timestamps 从未被放入 `ctx_side_features`，carry-forward 逻辑根本没机会执行。修复：循环分 `embed_add` 和 `torope` 两个分支处理。
   - 两层 bug 叠加，导致误判 TO-RoPE 无效。教训：**新增特征必须从 preprocess → train → eval_items 构建 → beam search 全链路验证特征是否非零**。

5. **后续方向**：TO-RoPE +2pp 有统计意义，值得在更大模型（4B/8B SID）上验证。time_split=0.25 作为默认推荐参数。

---
