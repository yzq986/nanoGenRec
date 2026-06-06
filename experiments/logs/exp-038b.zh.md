## EXP-038B: RF-DPO on exp037-medium — ntp_epochs=3 + mid-checkpoints

[English](exp-038b.md) | [中文](exp-038b.zh.md)

**Date**: 2026-04-28
**Status**: completed
**Results**: experiments/ntp_checkpoints/exp038b-hard-lam03-3ep-ep1/ (best)

### Background

EXP-038 RF-DPO（1 epoch，406 steps）后 R@500=59.6%，PPL=25.7，相比 ref（exp037-medium 62.1%）退化 2.5pp。原因分析：step 数太少（406 steps），NTP:DPO 配比未对齐 exp019/020 设计（exp020 目标是 807 DPO steps ≈ NTP steps）。

EXP-038B 用 `--ntp_epochs 3`（总 1218 steps），并在每个 epoch 边界保存 mid-checkpoint，对比 ep1/ep2/ep3(final) 效果。

**代码实现**：新增 `ntp_epochs` 参数（`itertools.chain.from_iterable(itertools.repeat(ntp_loader, ntp_epochs))`），mid-checkpoint 在每个 epoch 末尾保存至 `{output_dir}-ep{N}`。

### Hypothesis

ep1（406 steps）= 对齐 exp038 的 1 epoch，预期与 EXP-038 相当（~59.6%）。更多 epoch 可能改善 DPO 对齐但有 NTP 过拟合风险。

### Design

- **Variable**: ntp_epochs ∈ {1,2,3}（通过 mid-checkpoint 实现三点对比）
- **Fixed**: ref=exp037-medium, λ=0.03, β=0.1, difficulty=hard, lr=1e-4, Joint NTP+DPO
- **Metric**: R@{10,500}, PPL（三个 epoch 各评）
- **Data**: RF-DPO pairs from exp018 real feedback (2026-03-18~03-31)，4,312 hard pairs

### Run
`bash experiments/scripts/exp-038b.sh`

### Results

| Checkpoint | Steps | R@10 | R@500 | PPL | 结论 |
|---|---|---|---|---|---|
| exp037-medium (ref) | — | 11.2% | 62.1% | 23.0 | SP-DPO 起点 |
| **ep1 (1 epoch)** | 406 | **11.2%** | **62.1%** | **23.6** | ✅ 持平 ref，DPO 无损 |
| ep2 (2 epochs) | 812 | 10.3% | 59.6% | 26.0 | ❌ NTP 开始过拟合 |
| final (3 epochs) | 1218 | 9.3% | 52.8% | 33.3 | ❌ 严重过拟合 |

**最佳 checkpoint**：`exp038b-hard-lam03-3ep-ep1`（ep1，R@500=62.1%）

### Analysis

1. **ep1 持平 ref（不退化！）**：EXP-038 1 epoch 退化到 59.6% 的原因可能是 LR 过高或训练不稳定，而 EXP-038B ep1 以相同步数得到 62.1%，说明 DPO 对 NTP 的影响在 1 epoch 内是中性的。

2. **2/3 epoch NTP 过拟合**：NTP loss 在 exp018 真实反馈数据（分布窄）上多次循环后开始过拟合，PPL 从 23.6 → 26.0 → 33.3 快速劣化。

3. **关键教训**：RF-DPO 最优是 1 epoch；`--ntp_epochs` 应设为 1（已有实验验证）。后续实验用 ep1 作为 ECPO 起点。

### Next Steps
- EXP-039B: ECPO on ep1（`exp038b-hard-lam03-3ep-ep1`），δ=0.1，G=512，on-policy beam

---
