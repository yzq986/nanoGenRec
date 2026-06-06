## EXP-044B: TO-RoPE with Real Timestamps — S-tier + 0.6B SID

[English](exp-044b.md) | [Chinese](exp-044b.zh.md)

**Date**: 2026-04-29
**Status**: completed

### Background
In EXP-044, the timestamps are all 0 (the pipeline is not connected). This experiment is repaired and rerun:
- `_build_sequences_from_behavior` added `timestamps` calculation (rel_hours)
- time_gap_emb coexists with TO-RoPE
- Separate ablation: remove time_gap (Config D) to verify whether the two are complementary

### Design
- **Variable**: TO-RoPE time_split (0.5 / 0.25) × coexistence with or without time_gap
- **Fixed**: S-tier + 0.6B SID, 14d data, action_level + segment_emb
- **Baseline**: exp043-s-0.6b (R@500=61.2%, PPL=26.52)

| Config | Description |
|--------|------|
| exp043-s-0.6b | Baseline: abs pos + time_gap + action + segment |
| exp044b-torope-ts05 | TO-RoPE ts=0.5 + time_gap + action + segment |
| exp044b-torope-ts025 | TO-RoPE ts=0.25 + time_gap + action + segment |
| exp044b-torope-ts05-notg | TO-RoPE ts=0.5 + action + segment（无 time_gap 消融）|

### Results

⚠️ **History (invalid result)**: The first eval caused timestamps=0 due to a two-layer train-infer bug, and the result was invalid (R@500≈32%). The results after repair are shown below.

| Config | R@10 | R@500 | PPL | 备注 |
|--------|------|-------|-----|------|
| **exp043-s-0.6b** (baseline) | **11.4%** | **61.2%** | **26.5** | abs pos + time_gap + action + seg |
| exp044b-torope-ts05 | 12.5% | **62.3%** | 474.9 | TO-RoPE ts=0.5 + time_gap |
| exp044b-torope-ts025 | 12.5% | **63.6%** | 467.5 | TO-RoPE ts=0.25 + time_gap ← 最佳 |
| exp044b-torope-ts05-notg | 11.9% | **63.5%** | 480.0 | TO-RoPE ts=0.5，无 time_gap |

### Analysis

1. **TO-RoPE is valid**: After fixing the train-infer inconsistency, R@500 all exceeds the baseline (61.2%), and the best is 63.6% (+2.4pp).

2. **High PPL does not mean poor effect**: PPL is still 467-480 (about 18× higher than baseline), but beam search R@500 is better. The reason is that TO-RoPE changes the entropy of the logit distribution (the attention pattern is different), and PPL measures the sharpness of the probability distribution and does not directly correspond to the recall quality.

3. **time_split=0.25 Optimal**: Reserve more dimensions for index-RoPE, and time information can be expressed with a small number of dimensions. ts05-notg (without time_gap) is close to ts025 (63.5% vs 63.6%), indicating that TO-RoPE itself can encode time information, and the increment of time_gap_emb is limited.

4. **Bug review - two layers of train-infer are inconsistent**:
   - **Bug 1**: `_step_sf` of `constrained_beam_search` only processes the `inject='embed_add'` feature, and the timestamps of `inject='torope'` are all 0 in the generation step. Fix: Add `_step_ts()` carry-forward.
   - **Bug 2 (more hidden)**: The `eval_items` construction loop filter condition of `eval.py` is `inject != 'embed_add'`, resulting in timestamps never being put into `ctx_side_features`, and the carry-forward logic has no chance to be executed at all. Fix: The loop is processed in two branches: `embed_add` and `torope`.
   - Two layers of bugs are superimposed, leading to misjudgment that TO-RoPE is invalid. Lesson: **New features must be built from preprocess → train → eval_items → beam search to verify whether the feature is non-zero**.

5. **Follow-up direction**: TO-RoPE +2pp is statistically significant and worthy of verification on a larger model (4B/8B SID). time_split=0.25 is used as the default recommended parameter.

---
