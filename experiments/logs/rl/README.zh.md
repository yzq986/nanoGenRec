# RL Alignment Experiments

[English](README.md) | [中文](README.zh.md)

基于 NTP 检查点的偏好学习和 RL 风格对齐实验总结。

实现细节见 [rl/README.md](../../../rl/README.md)。此文件跟踪已验证的对齐路径和当前结果。

## 当前结果

| 配置 | R@500 | 来源 |
|------|-------|------|
| ECPO on `exp038b` ep1 | 65.7% | EXP-039B |
| ECPO on EXP-029 流水线 | ~65% | EXP-029 |
| RF-DPO 3ep, ep1 检查点 | 62.1% | EXP-038B |
| SP-DPO | ~55-58% | EXP-037 |
| S-tier SFT 基线 | 61.2% | EXP-043 |

下一个计划链：从 `exp047`（R@500=64.1% 的 L-tier SFT 检查点）重新运行 SP-DPO -> RF-DPO -> ECPO。

## 已验证的对齐路径

```text
SFT -> SP-DPO -> RF-DPO (3 个 epoch, 中间检查点选择) -> ECPO
```

| 阶段 | 角色 |
|------|------|
| SP-DPO | 从 SFT 模型生成的自我对抗偏好。 |
| RF-DPO | 来自行为数据的真实反馈偏好对。 |
| ECPO | GRPO + BehaviorReward + A2PO + NLL + HEPO 奖励堆栈。 |

## 已验证的超参数

| 参数 | 设置 | 来源 |
|------|------|------|
| RF-DPO lambda | 0.3（hard pairs） | EXP-020 |
| RF-DPO epochs | 3，选择 ep1 检查点 | EXP-038B |
| ECPO delta | 0.1 | EXP-028+ |
| ECPO epsilon | 0.2 | EXP-028+ |
| GRPO group size | 512, `grpo_batch=4` | EXP-029 |
| `grpo_weight` | 0.03 | EXP-029 |

