# Experiment Logs

[English](README.md) | [Chinese](README.zh.md)

Human-readable experiment records, organized by research phase.

This directory is for conclusions, comparisons, and next-step planning. Code interfaces and implementation details belong in the module READMEs, while raw artifacts live under `experiments/`.

## Phase Summaries

| Phase | Scope | Current Best / Status | Summary |
|-------|-------|-----------------------|---------|
| [tokenizer/](tokenizer/README.md) | EXP-001 to EXP-012, EXP-026, EXP-045, EXP-049 | Recommended SID caches: `exp049-0.6b-nc8192-h128`, `exp049-4b-nc8192-h128` | Semantic ID codebooks, collision, Gini, and snHR. |
| [ntp/](ntp/README.md) | EXP-013 to EXP-016, EXP-036, EXP-041 to EXP-050 | M-tier R@500=70.2%; L-tier SFT R@500=64.1% | Autoregressive recommender scaling and feature ablations. |
| [rl/](rl/README.md) | EXP-017 to EXP-040 | ECPO R@500=65.7% on the S-tier pipeline | Preference data, DPO, GRPO, and ECPO alignment. |

## Experiment Entry Format

Each `exp-NNN.md` should be readable without opening training logs:

```markdown
## EXP-NNN: Short Title

**Date**: YYYY-MM-DD
**Status**: completed

### Background

What question motivated this run?

### Design

- Variable:
- Fixed:
- Baseline:

### Results

| Config | R@10 | R@500 | PPL |
|--------|------|-------|-----|

### Analysis

What changed, what did not, and why?

### Next Steps

What should be run or changed next?
```

## Maintenance Rules

- Update the phase README whenever a completed experiment changes a SOTA table, baseline, or recommendation.
- Update the root README only for headline changes that matter to new readers.
- Mark invalid or bugged experiments explicitly; do not delete them from the lineage.
- Use full eval numbers for comparisons. Inline eval during training is a health check, not a publishable baseline.

## Full Index

| EXP | Phase | Date | Status | Title |
|-----|-------|------|--------|-------|
| [001](./exp-001.md) | tokenizer | 2026-03 | completed | RKMeans training optimization v0 -> v7 |
| [002](./exp-002.md) | tokenizer | 2026-04-13 | completed | ResKmeansFSQ |
| [003](./exp-003.md) | tokenizer | 2026-04-13 | completed | Learned FSQ |
| [004](./exp-004.md) | tokenizer | 2026-04-13 | completed | OPQ Parallel Semantic IDs |
| [007](./exp-007.md) | tokenizer | 2026-04-13 | completed | Collaborative Signal Enhanced Embedding |
| [008](./exp-008.md) | tokenizer | 2026-04-14 | completed | FORGE proxy comparison: MLP-FSQ vs OPQ |
| [009](./exp-009.md) | tokenizer | 2026-04-14 | completed | QFormer Tokenizer |
| [010](./exp-010.md) | tokenizer | 2026-04-15 | completed | NTP baseline with MLP-FSQ |
| [011](./exp-011.md) | tokenizer | 2026-04-15 | completed | Codebook size ablation |
| [012](./exp-012.md) | tokenizer | 2026-04-15 | completed | Tokenizer grid search: 4096x3 binary winner |
| [013](./exp-013.md) | ntp | 2026-04-15 | completed | S-tier NTP baseline |
| [014](./exp-014.md) | ntp | 2026-04-16 | completed | ENTP loss ablation |
| [015](./exp-015.md) | ntp | 2026-04-16 | completed | NTP scaling law |
| [016](./exp-016.md) | ntp | 2026-04-17 | completed | Data scaling law |
| [017](./exp-017.md) | rl | 2026-04-17 | completed | SP-DPO |
| [018](./exp-018.md) | rl | 2026-04-18 | completed | RF-DPO |
| [019](./exp-019.md) | rl | 2026-04-20 | completed | RF-DPO joint NTP+DPO |
| [020](./exp-020.md) | rl | 2026-04-20 | completed | RF-DPO Hard lambda=0.3 |
| [021](./exp-021.md) | rl | 2026-04-20 | planned | Qwen3-4B vs 0.6B embedding |
| [022](./exp-022.md) | rl | 2026-04-20 | completed | In-batch contrastive loss |
| [023](./exp-023.md) | rl | 2026-04-21 | completed | Side features NTP |
| [024](./exp-024.md) | rl | 2026-04-21 | completed | Side feature shift fix |
| [025](./exp-025.md) | rl | 2026-04-21 | completed | Beam-search feature passing |
| [026](./exp-026.md) | tokenizer | 2026-04-27 | completed | 0.6B/4B/8B SID cache build |
| [027](./exp-027.md) | rl | 2026-04-27 | interrupted | ECPO grpo_weight sweep |
| [028](./exp-028.md) | rl | 2026-04-27 | completed | ECPO + WeightedBehaviorReward |
| [029](./exp-029.md) | rl | 2026-04-27 | completed | ECPO + on-policy beam search |
| [030](./exp-030.md) | rl | 2026-04-27 | completed | A2PO + NLL + HEPO |
| [031](./exp-031.md) | rl | 2026-04-28 | completed | Features SFT + full RL stack |
| [032](./exp-032.md) | rl | 2026-04-28 | planned | GRPO group size sweep |
| [033](./exp-033.md) | rl | 2026-04-28 | completed | Features fix validation |
| [034](./exp-034.md) | rl | 2026-04-28 | planned | Ref model alignment |
| [035](./exp-035.md) | rl | 2026-04-28 | completed | Constrained sampling |
| [036](./exp-036.md) | ntp | 2026-04-28 | completed | Features NTP |
| [037](./exp-037.md) | rl | 2026-04-28 | completed | SP-DPO on exp036 |
| [038](./exp-038.md) | rl | 2026-04-28 | completed | RF-DPO on exp037 |
| [038B](./exp-038b.md) | rl | 2026-04-28 | completed | RF-DPO ntp_epochs=3 + mid checkpoint |
| [039](./exp-039.md) | rl | 2026-04-28 | skipped | ECPO on exp038, replaced by EXP-039B |
| [039B](./exp-039b.md) | rl | 2026-04-29 | completed | ECPO on exp038b ep1 |
| [040](./exp-040.md) | rl | 2026-04-28 | planned | RSFT |
| [041](./exp-041.md) | ntp | 2026-04-29 | completed | ENTP loss v1 |
| [041B](./exp-041b.md) | ntp | 2026-04-29 | completed | ENTP loss v2 |
| [043](./exp-043.md) | ntp | 2026-04-29 | completed | Embedding x tier comparison |
| [044](./exp-044.md) | ntp | 2026-04-29 | completed | TO-RoPE vs APE with zero timestamps |
| [044B](./exp-044b.md) | ntp | 2026-04-29 | completed | TO-RoPE with real timestamps |
| [044C](./exp-044c.md) | ntp | 2026-04-29 | completed | TO-RoPE item-position fix |
| [045](./exp-045.md) | tokenizer | 2026-04-29 | bug | FSQ h-dim sweep with num_clusters bug |
| [046](./exp-046.md) | ntp | 2026-04-29 | completed | GateAttention |
| [047](./exp-047.md) | ntp | 2026-04-30 | completed | L-tier SFT baseline |
| [048](./exp-048.md) | ntp | 2026-04-30 | completed | M-tier TO-RoPE ablation |
