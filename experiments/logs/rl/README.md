# RL Alignment Experiments

[English](README.md) | [Chinese](README.zh.md)

Experiment summary for preference learning and RL-style alignment on top of NTP checkpoints.

Implementation details live in [rl/README.md](../../../rl/README.md). This file tracks validated alignment paths and current results.

## Current Results

| Config | R@500 | Source |
|--------|-------|--------|
| ECPO on `exp038b` ep1 | 65.7% | EXP-039B |
| ECPO on EXP-029 pipeline | ~65% | EXP-029 |
| RF-DPO 3ep, ep1 checkpoint | 62.1% | EXP-038B |
| SP-DPO | ~55-58% | EXP-037 |
| S-tier SFT baseline | 61.2% | EXP-043 |

Next planned chain: rerun SP-DPO -> RF-DPO -> ECPO from `exp047`, the L-tier SFT checkpoint with R@500=64.1%.

## Validated Alignment Path

```text
SFT -> SP-DPO -> RF-DPO (3 epochs, mid-checkpoint selection) -> ECPO
```

| Stage | Role |
|-------|------|
| SP-DPO | Self-play preferences generated from the SFT model. |
| RF-DPO | Real-feedback preference pairs from behavior data. |
| ECPO | GRPO + BehaviorReward + A2PO + NLL + HEPO reward stack. |

## Validated Hyperparameters

| Parameter | Setting | Source |
|-----------|---------|--------|
| RF-DPO lambda | 0.3 on hard pairs | EXP-020 |
| RF-DPO epochs | 3, choose ep1 checkpoint | EXP-038B |
| ECPO delta | 0.1 | EXP-028+ |
| ECPO epsilon | 0.2 | EXP-028+ |
| GRPO group size | 512, `grpo_batch=4` | EXP-029 |
| `grpo_weight` | 0.03 | EXP-029 |

## Experiment List

| EXP | Date | Status | Takeaway |
|-----|------|--------|----------|
| [017](../exp-017.md) | 2026-04-17 | completed | First SP-DPO run. |
| [018](../exp-018.md) | 2026-04-18 | completed | First RF-DPO run. |
| [019](../exp-019.md) | 2026-04-20 | completed | RF-DPO with joint NTP+DPO loss. |
| [020](../exp-020.md) | 2026-04-20 | completed | RF-DPO hard-pair lambda sweep selected lambda=0.3. |
| [021](../exp-021.md) | 2026-04-20 | planned | Qwen3-4B vs 0.6B embedding quality. |
| [022](../exp-022.md) | 2026-04-20 | completed | In-batch contrastive loss. |
| [023](../exp-023.md) | 2026-04-21 | completed | Side feature NTP experiment. |
| [024](../exp-024.md) | 2026-04-21 | completed | Side feature shift and leakage fix. |
| [025](../exp-025.md) | 2026-04-21 | completed | Beam-search feature passing fixed train/eval mismatch. |
| [027](../exp-027.md) | 2026-04-27 | interrupted | ECPO `grpo_weight` sweep, replaced by EXP-028. |
| [028](../exp-028.md) | 2026-04-27 | completed | ECPO + WeightedBehaviorReward. |
| [029](../exp-029.md) | 2026-04-27 | completed | ECPO + on-policy beam search reached about 65% R@500. |
| [030](../exp-030.md) | 2026-04-27 | completed | A2PO + NLL + HEPO prefix scoring. |
| [031](../exp-031.md) | 2026-04-27 | completed | Features SFT + full RL stack. |
| [032](../exp-032.md) | 2026-04-28 | planned | GRPO group size x diversity sweep. |
| [033](../exp-033.md) | 2026-04-28 | completed | Feature fix validation. |
| [034](../exp-034.md) | 2026-04-28 | planned | Ref model alignment. |
| [035](../exp-035.md) | 2026-04-28 | completed | Constrained sampling. |
| [037](../exp-037.md) | 2026-04-28 | completed | SP-DPO on the full-feature NTP checkpoint. |
| [038](../exp-038.md) | 2026-04-28 | completed | RF-DPO on EXP-037. |
| [038B](../exp-038b.md) | 2026-04-28 | completed | RF-DPO 3 epochs; ep1 was best. |
| [039](../exp-039.md) | 2026-04-28 | skipped | ECPO on EXP-038, replaced by EXP-039B. |
| [039B](../exp-039b.md) | 2026-04-29 | completed | ECPO on EXP-038B ep1 reached 65.7% R@500. |
| [040](../exp-040.md) | 2026-04-28 | planned | Reject Sampling Fine-Tuning. |
