## EXP-039B: ECPO on exp038b-hard-lam03-3ep-ep1 (Features RL 链路终点)

**Date**: 2026-04-29
**Status**: completed
**Results**: experiments/ntp_checkpoints/exp039b-ecpo-from-spdpo/

### Background

RL 对齐链路最终步：exp036 SFT → EXP-037 SP-DPO → EXP-038B RF-DPO (ep1 best, R@500=62.1%) → 本实验 ECPO。
EXP-039（从 exp038-hard-lam03 起）已跳过，直接从 exp038b ep1 起跑以利用更好的起点。

### Hypothesis

ECPO (δ=0.1) 在 features 模型上复现 exp029 幅度的提升（+4pp），最终 R@500 接近或超越 exp020 SOTA (66.2%)。

### Design

- **Variable**: ECPO δ=0.1，从 exp038b-hard-lam03-3ep-ep1 (R@500=62.1%) 起跑
- **Fixed**: G=512, BehaviorReward+FormatReward, on-policy beam, grpo_weight=0.03, lr=1e-4, 8×L20X
- **Metric**: R@{10,500}, PPL
- **Data**: context pool from exp023-14d-features, behavior cache 2026-03-31

### Run
`bash experiments/scripts/exp-039b.sh --no-smoke`

### Results

| Config | R@10 | R@500 | PPL | Wall |
|--------|------|-------|-----|------|
| exp036-full-features (SFT) | - | ~53% | - | - |
| exp037-medium (SP-DPO, ref) | - | 62.1% | - | - |
| exp038b-ep1 (RF-DPO) | - | 62.1% | - | - |
| **exp039b-ecpo (this)** | **11.8%** | **65.7%** | **20.0** | **182min** |
| exp020-hard-lam03 (SOTA 无特征) | 14.1% | 66.2% | 16.3 | - |

### Analysis

- ECPO 从 RF-DPO ep1（62.1%）提升至 **65.7%**，+3.6pp，与 exp029 提升幅度一致
- 距离无特征 SOTA (66.2%) 仅差 **0.5pp**，几乎追平
- PPL=20.0 高于 SOTA (16.3)，说明 features 路线的 NTP 质量还有空间
- behavior_mean reward 从 0.574 → 0.630（收敛趋势良好），coverage=98.8%
- **结论**：features RL 链路（SFT→DPO→ECPO）有效，但特征引入带来 PPL 代价

### Next Steps
- EXP-040: RSFT（行为质量过滤）验证是否能在 SFT 阶段就提升基线
- EXP-041: ENTP-Loss（曝光负样本 α sweep）验证 L0 负样本惩罚效果

---
