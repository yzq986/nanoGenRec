# NTP Experiments

Experiment summary for autoregressive recommendation over Semantic ID sequences.

Implementation details live in [ntp/README.md](../../../ntp/README.md). This file tracks the current baselines, validated improvements, and experiment lineage.

## Current Full-Eval Baselines

| Config | R@10 | R@500 | PPL | Source |
|--------|------|-------|-----|--------|
| M-tier bare, 0.6B SID | 14.5% | 70.2% | 18.54 | EXP-043 |
| M-tier, 4B SID | 14.2% | 70.4% | 16.55 | EXP-043 |
| L-tier with validated options | 12.8% | 64.1% | 20.7 | EXP-047 |
| S-tier with TO-RoPE `ts=0.5` | 11.8% | 63.9% | 22.7 | EXP-044C |
| S-tier bare | 11.4% | 61.2% | 26.52 | EXP-043 |

The current RL starting point is `exp047`, an L-tier SFT checkpoint with R@500=64.1%.

## Model Tiers

| Tier | Config Name | embed_dim | Layers | Experts | Active Params |
|------|-------------|-----------|--------|---------|---------------|
| S-tier | `scale-05` | 256 | 6 | 8 | ~17.5M |
| M-tier | `scale-06` | 512 | 8 | 8 | ~71.6M |
| L-tier | `scale-07` | 512 | 12 | 16 | ~101.1M |

## Validated Changes

| Change | R@500 Delta | Scope | Source |
|--------|-------------|-------|--------|
| `segment_emb` + `time_gap` + `action_level` | +3.7pp | S-tier | EXP-036 |
| TO-RoPE with real timestamps, `ts=0.5` | +2.4pp | S-tier | EXP-044B |
| GateAttention | +0.4pp | S-tier | EXP-046 |
| TO-RoPE 3-dim | -0.7pp | M-tier | EXP-048 |

EXP-048 indicates TO-RoPE does not currently help M-tier: 2-dim is roughly flat and 3-dim is harmful. The likely interpretation is that the larger model has enough capacity for the tested temporal signal.

## Evaluation Rule

Only full eval should be used for comparisons:

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 run.py eval-ntp \
    --checkpoint experiments/ntp_checkpoints/<name> \
    --n_recall 1000
```

Inline eval during training uses a limited candidate set and is a health check only.

## Experiment List

| EXP | Date | Status | Takeaway |
|-----|------|--------|----------|
| [013](../exp-013.md) | 2026-04-15 | completed | S-tier NTP baseline with 6-layer MoE. |
| [014](../exp-014.md) | 2026-04-16 | completed | ENTP loss ablation. |
| [015](../exp-015.md) | 2026-04-16 | completed | NTP scaling from 1M to 100M active parameters. |
| [016](../exp-016.md) | 2026-04-17 | completed | Data scaling law. |
| [036](../exp-036.md) | 2026-04-28 | completed | Time/action/segment side features improved recall. |
| [041](../exp-041.md) | 2026-04-29 | completed | ENTP loss v1 was ineffective. |
| [041B](../exp-041b.md) | 2026-04-29 | completed | ENTP loss v2 was ineffective at session granularity. |
| [043](../exp-043.md) | 2026-04-29 | completed | Embedding size x NTP tier comparison established M-tier. |
| [044](../exp-044.md) | 2026-04-29 | completed | TO-RoPE comparison was invalid because timestamps were zero. |
| [044B](../exp-044b.md) | 2026-04-29 | completed | Real timestamps made TO-RoPE useful on S-tier. |
| [044C](../exp-044c.md) | 2026-04-29 | completed | Item-position fix and 3-dim TO-RoPE follow-up. |
| [046](../exp-046.md) | 2026-04-29 | completed | GateAttention gave a small gain. |
| [047](../exp-047.md) | 2026-04-30 | completed | L-tier SFT baseline for the next RL chain. |
| [048](../exp-048.md) | 2026-04-30 | completed | M-tier TO-RoPE variants did not improve recall. |
| EXP-050 | 2026-04-30 | queued | M-tier 0.6B/4B SID, output-gate/CADET, and RoPE ablations. |
