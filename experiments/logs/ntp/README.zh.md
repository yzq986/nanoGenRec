# NTP Experiments

[English](README.md) | [中文](README.zh.md)

Semantic ID 序列上自回归推荐的实验总结。

实现细节见 [ntp/README.md](../../../ntp/README.md)。此文件跟踪当前基线、已验证的改进和实验谱系。

## 当前全量评估基线

| 配置 | R@10 | R@500 | PPL | 来源 |
|------|------|-------|-----|------|
| M-tier bare, 0.6B SID | 14.5% | 70.2% | 18.54 | EXP-043 |
| M-tier, 4B SID | 14.2% | 70.4% | 16.55 | EXP-043 |
| L-tier 含已验证选项 | 12.8% | 64.1% | 20.7 | EXP-047 |
| S-tier 含 TO-RoPE `ts=0.5` | 11.8% | 63.9% | 22.7 | EXP-044C |
| S-tier bare | 11.4% | 61.2% | 26.52 | EXP-043 |

当前 RL 起点是 `exp047`，一个 R@500=64.1% 的 L-tier SFT 检查点。

## 模型分档

| 级别 | 配置名 | embed_dim | 层数 | 专家数 | 活跃参数 |
|------|--------|-----------|------|--------|---------|
| S-tier | `scale-05` | 256 | 6 | 8 | ~17.5M |
| M-tier | `scale-06` | 512 | 8 | 8 | ~71.6M |
| L-tier | `scale-07` | 512 | 12 | 16 | ~101.1M |

## 已验证的变更

| 变更 | R@500 变化 | 范围 | 来源 |
|------|-----------|------|------|
| `segment_emb` + `time_gap` + `action_level` | +3.7pp | S-tier | EXP-036 |
| 使用真实时间戳的 TO-RoPE, `ts=0.5` | +2.4pp | S-tier | EXP-044B |
| GateAttention | +0.4pp | S-tier | EXP-046 |
| TO-RoPE 3-dim | -0.7pp | M-tier | EXP-048 |

EXP-048 表明 TO-RoPE 目前对 M-tier 无帮助：2-dim 基本持平，3-dim 有损害。可能的解释是更大的模型已有足够的容量来处理测试的时间信号。

