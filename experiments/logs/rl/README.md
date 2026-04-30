# RL 对齐实验

SP-DPO → RF-DPO → GRPO/ECPO 全链路强化学习对齐。

## 当前最优

| 配置 | R@500 | 来源 |
|------|-------|------|
| **ECPO on exp038b ep1** | **65.7%** | EXP-039B |
| ECPO on exp029 | ~65% | EXP-029 |
| RF-DPO 3ep ep1 | 62.1% | EXP-038B |
| SP-DPO | ~55-58% | EXP-037 |
| SFT baseline (S-tier) | 61.2% | EXP-043 |

**下一步**：以 exp047（L-tier SFT, R@500=64.1%）为起点重做 RL 链路：SP-DPO → RF-DPO (3ep) → ECPO。

## 标准 RL 链路

```
SFT → SP-DPO → RF-DPO (3ep, mid-ckpt) → ECPO
```

- **SP-DPO**：Self-Play DPO，用 SFT 自身生成 preference pair
- **RF-DPO**：Real Feedback DPO，用真实用户行为构建 pair；3 epoch + mid-checkpoint 找最优
- **ECPO**：GRPO + BehaviorReward + A2PO + NLL + HEPO 全 reward stack

## 关键超参（已验证）

| 参数 | 最优值 | 实验 |
|------|--------|------|
| RF-DPO λ | 0.3（hard） | EXP-020 |
| RF-DPO ntp_epochs | 3，取 ep1 | EXP-038B |
| ECPO δ | 0.1 | EXP-028+ |
| ECPO ε | 0.2 | EXP-028+ |
| GRPO G | 512，grpo_batch=4 | EXP-029 |
| grpo_weight | 0.03 | EXP-029 |

## 实验列表

| EXP | Date | Status | 结论 |
|-----|------|--------|------|
| [017](../exp-017.md) | 2026-04-17 | completed | SP-DPO 初版 |
| [018](../exp-018.md) | 2026-04-18 | completed | RF-DPO 初版 |
| [019](../exp-019.md) | 2026-04-20 | completed | RF-DPO Joint NTP+DPO |
| [020](../exp-020.md) | 2026-04-20 | completed | **RF-DPO Hard λ Sweep — λ=0.3 最优** |
| [021](../exp-021.md) | 2026-04-20 | planned | Qwen3-4B vs 0.6B Embedding Quality |
| [022](../exp-022.md) | 2026-04-20 | completed | In-Batch Contrastive Loss |
| [023](../exp-023.md) | 2026-04-21 | completed | Side Features NTP |
| [024](../exp-024.md) | 2026-04-21 | completed | Side Feature Shift（信息泄漏修复） |
| [025](../exp-025.md) | 2026-04-21 | completed | **Beam Search Feature Passing — train-eval 一致性** |
| [027](../exp-027.md) | 2026-04-27 | interrupted | ECPO grpo_weight Sweep（被 028 取代） |
| [028](../exp-028.md) | 2026-04-27 | completed | ECPO + WeightedBehaviorReward |
| [029](../exp-029.md) | 2026-04-27 | completed | **ECPO + On-Policy Beam Search — R@500=~65%** |
| [030](../exp-030.md) | 2026-04-27 | completed | A2PO + NLL + HEPO Prefix Scoring |
| [031](../exp-031.md) | 2026-04-27 | completed | Features SFT + Full RL Stack |
| [032](../exp-032.md) | 2026-04-28 | planned | GRPO Group Size × Diversity Sweep |
| [033](../exp-033.md) | 2026-04-28 | completed | Features 修复验证 |
| [034](../exp-034.md) | 2026-04-28 | planned | Ref Model Alignment |
| [035](../exp-035.md) | 2026-04-28 | completed | Constrained Sampling |
| [037](../exp-037.md) | 2026-04-28 | completed | SP-DPO on exp036-full-features |
| [038](../exp-038.md) | 2026-04-28 | completed | RF-DPO on exp037-medium |
| [038B](../exp-038b.md) | 2026-04-28 | completed | **RF-DPO ntp_epochs=3 + mid-ckpt — ep1=62.1% best** |
| [039](../exp-039.md) | 2026-04-28 | skipped | ECPO on exp038（被 039B 取代） |
| [039B](../exp-039b.md) | 2026-04-29 | completed | **ECPO on exp038b ep1 — R@500=65.7% SOTA** |
| [040](../exp-040.md) | 2026-04-28 | planned | RSFT — Reject Sampling Fine-Tuning |
