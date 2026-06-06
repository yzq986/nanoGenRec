---

[English](002-experiment-analysis.md) | [Chinese](002-experiment-analysis.zh.md)
date: "2026-04-22 18:35"
type: finding
priority: normal
subject: "EXP-022~025 Trend Analysis + Next Experiment Proposal"
needs_response: true
---

## 1. Overview of experimental results

Completed collection and analysis of all results from EXP-022 to EXP-025.

### Current complete leaderboard (R@500 sorting)

| Rank | Config | PPL | R@10 | R@500 | Key variables |
|------|--------|-----|------|-------|----------|
| **1** | **exp025-beam-passes** | **25.22** | 10.4% | **63.6%** | seg+time+action, beam pass features |
| 2 | exp023-segment | 25.94 | 10.9% | 61.2% | segment_emb only |
| 3 | exp023-baseline | 28.41 | 11.0% | 60.7% | EXP-016 recurrence |
| 4 | exp023-timegap | 28.78 | 10.9% | 60.1% | time_gap only (leaks) |
| 5 | exp024-seg-timegap | ~26 | — | 59.8% | seg + shifted time_gap |
| 6 | exp022-alpha001 | 27.89 | 10.3% | 59.2% | contrastive α=0.01 |
| 7 | exp022-dim256 | 29.66 | 10.1% | 58.8% | contrastive dim=256 |
| 8 | EXP-016 baseline | 27.05 | 9.9% | 58.5% | original baseline |
| 9 | exp022-temp005 | 28.16 | 10.1% | 58.2% | contrastive τ=0.05 |
| 10 | exp022-alpha01 | 29.22 | 9.7% | 57.9% | contrastive α=0.1 |
| 11 | exp022-alpha05 | 29.04 | 9.7% | 56.3% | contrastive α=0.5 |
| 12 | exp024-seg-all | ~26 | — | ~55% | seg + shifted all |
| 13 | exp024-seg-action | ~27 | — | 52.9% | seg + shifted action |
| 14 | exp023-all | 25.16 | 9.5% | 55.0% | all features (with leaks) |
| 15 | exp025-action-l2only | 24.85 | 5.5% | 27.0% | action L2 only (failed) |
| 16 | exp023-action | 27.50 | 4.9% | 28.5% | action only (serious leak) |

## 2. Key findings

### 1. Beam search feature passing is the right direction ✅
After EXP-024 (shift) failed, EXP-025 beam_passes completely solved the training-inference gap of time_gap/action:
- R@500: 61.2% → **63.6%** (+2.4pp), PPL: 25.94 → **25.22** (-0.72)
- Method: Use all features normally for training, and pass in time_gap (known true value) + action (carry-forward the previous context item) during beam search
- This proves that side features themselves are valuable. Previous failures were inference-side bugs, not useless features.

### 2. Contrastive loss is invalid ❌
EXP-022 All five configs are inferior to baseline. IDEA-onemall-0 can be closed.
- The best α=0.01 is only +0.7pp R@500, at a cost of +0.84 PPL
- The larger α is, the worse it is (0.1 → -0.6pp, 0.5 → -2.2pp)
- Root cause speculation: SID is a discrete codebook token, and embedding space alignment does not directly help discrete token prediction.

### 3. The Shift solution is completely wrong ❌
EXP-024 proves that shift not only fails to repair the gap, but also makes the signal stale and worse. The correct design direction is to repair the inference side rather than modify the training data.

### 4. Segment embedding stable and positive ✅
EXP-023 → EXP-025, segment_emb is always a forward signal and is superimposed orthogonally with beam_passes.

## 3. P0 Ideas status

| IDEA | Status | Conclusion |
|------|------|------|
| IDEA-feat-0 (time_gap) | ✅ Verified via EXP-023/025 | Valid, need to cooperate with beam_passes |
| IDEA-feat-1 (action_level) | ✅ Verified via EXP-023/025 | Valid, need to cooperate with beam_passes |
| IDEA-feat-2 (segment_emb) | ✅ Verified via EXP-023 | Confirmed forward |
| IDEA-onemall-0 (contrastive) | ❌ Tested via EXP-022 | Invalid, discard |
| IDEA-genrec-0 (PW-NTP) | 🔲 Not tested | **Highest priority next step** |

## 4. Next experimental proposal

### Proposal A: EXP-026 — Page-wise NTP (IDEA-genrec-0) ⭐ Recommended

**Why do this now**:
- Read GenRec paper (2604.14878), JD online +9.5% click, hallucination rate reduced by 50%
- Completely orthogonal to beam_passes - PW-NTP changes the training target, beam_passes changes the inference end
- **There is no need to change the inference end at all** (the training-inference asymmetry is intentionally designed)
- Modules that need to be modified: `ntp/preprocess.py` (data structure), `ntp/train.py` (dataloader)
- No need to modify `ntp/model.py` (the model structure remains unchanged)

**Design Draft**:
- Data structure: multiple positive interaction items within the same session/time window are spliced into a target sequence
- Baseline: exp025-beam-passes (R@500=63.6%)
- Comparison: point-wise NTP vs page-wise NTP (2/3/5 items per page)
- The expected training time is the same as the current one (packed sequences, the total number of tokens remains unchanged)

**Human Confirmation Required**:
1. This involves modifying two source code files, `ntp/preprocess.py` and `ntp/train.py` - is it authorized?
2. What is the standard for session segmentation in data? Split by time interval >30min? Or on a daily basis?

### Proposal B: EXP-026 — scale-up verification based on beam_passes

**Why Consider**:
- beam_passes is new best, but only verified on S-tier (17.5M)
- You can try M-tier or L-tier models to see whether the benefits of side features increase as the model becomes larger
- No source code modification required, pure config adjustment

**Disadvantages**: Lack of new ideas, more like validation than exploration

---

**My recommendation**: Prioritize **Proposal A (Page-wise NTP)**. Reason:
1. It is the highest impact idea that has not yet been tested (paper report HR@50: 0.62→0.72, a huge improvement)
2. Improved orthogonality with existing beam_passes and can be superimposed
3. The scope of changes is clear (preprocess + dataloader) and does not involve model structure.
4. Source code modification is required, so human authorization is required → Submit it as early as possible

Please reply whether you agree with Proposal A and your preference for session splitting criteria.
