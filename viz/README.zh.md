# viz/

[English](README.md) | [中文](README.zh.md)

NTP 和对齐实验的训练后可视化工具。

这些工具读取 `experiments/ntp_checkpoints/` 中的 `train_meta.json` 和 `train_log.jsonl`，生成用于实验回顾的轻量级图表。

## 工具

| 工具 | 用途 | 输入 |
|------|------|------|
| `training_dynamics` | 损失、梯度范数和训练曲线。 | `train_log.jsonl` |
| `pareto` | PPL vs Recall 散点图及 Pareto 前沿。 | `train_meta.json` |
| `degradation` | 相对于基线的指标变化表。 | `train_meta.json` |
| `pipeline` | 从 NTP 到 DPO/RL 的逐阶段瀑布图。 | `train_meta.json` |

## 使用

```bash
# 比较训练动态
python -m viz.training_dynamics exp019-*

# PPL vs Recall Pareto 图
python -m viz.pareto exp019-* exp018-* exp017-fixed-medium \
    --baseline exp017-fixed-medium \
    --save pareto.png

# 退化表
python -m viz.degradation exp019-* exp018-* \
    --baseline exp017-fixed-medium

# 流水线瀑布图
python -m viz.pipeline \
    --ntp exp016-B-14d-S \
    --spdpo exp017-fixed-medium \
    --rfdpo exp019-* \
    --save pipeline.png
```

## 数据源

```text
experiments/ntp_checkpoints/<exp_name>/
├── train_meta.json
├── train_log.jsonl
└── probe.pt
```

生成的 `output_*.png` 文件被 git 忽略。绘图依赖为 `matplotlib` 和 `numpy`。

