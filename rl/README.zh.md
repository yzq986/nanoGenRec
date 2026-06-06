# rl/

[English](README.md) | [中文](README.zh.md)

基于 NTP 推荐器的偏好学习和 RL 风格对齐。

该模块从 SFT NTP 检查点开始，应用基于偏好或奖励的优化。已验证的顺序是 SP-DPO -> RF-DPO -> ECPO，RL 阶段使用 GRPO 风格的分组优势函数。

## 文件

| 文件 | 用途 |
|------|------|
| `preference.py` | 从 beam search 结果构建 SP-DPO 偏好对。 |
| `feedback.py` | 从行为反馈构建 RF-DPO 偏好对。 |
| `dpo.py` | Softmax-DPO 和 RF-DPO 损失，包括 SID log-prob 计算。 |
| `reward.py` | BehaviorReward、前缀回退、A2PO/HEPO 奖励堆栈。 |
| `grpo.py` | GRPO 裁剪代理和组归一化优势。 |
| `trainer.py` | 统一的 SP-DPO、RF-DPO、GRPO 和 ECPO 训练循环。 |

## 对齐路径

```text
SFT 检查点
  -> SP-DPO 偏好对
  -> RF-DPO 真实反馈
  -> ECPO 在线策略奖励优化
  -> 全量召回评估
```

## 已验证的设置

| 参数 | 当前设置 | 来源 |
|------|---------|------|
| RF-DPO lambda | 0.3（hard pairs） | EXP-020 |
| RF-DPO epochs | 3 个 epoch，选择最佳中间检查点 | EXP-038B |
| ECPO delta | 0.1 | EXP-028+ |
| ECPO epsilon | 0.2 | EXP-028+ |
| GRPO group size | 512, `grpo_batch=4` | EXP-029 |
| `grpo_weight` | 0.03 | EXP-029 |

当前阶段结果总结见 [experiments/logs/rl/README.md](../experiments/logs/rl/README.md)。

