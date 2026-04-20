# viz/ — Post-Training Visualization

从 `experiments/ntp_checkpoints/` 读取 `train_meta.json` + `train_log.jsonl`，生成后训练实验图表。

## 工具一览

| 工具 | 用途 | 输入 |
|------|------|------|
| `training_dynamics` | NTP/DPO loss + grad_norm 训练曲线 | train_log.jsonl |
| `pareto` | PPL vs Recall 散点图 (Pareto 前沿) | train_meta.json (eval) |
| `degradation` | 降级预算表：各指标 delta% vs baseline | train_meta.json (eval) |
| `pipeline` | 瀑布图：NTP → SP-DPO → RF-DPO 逐阶段指标 | train_meta.json (eval) |

## 用法

所有工具支持 shell glob 匹配实验名，`--save` 输出文件（不加则弹窗显示）。

```bash
# 训练曲线：对比多个 config 的 loss 走势
python -m viz.training_dynamics exp019-*

# Pareto 散点：PPL vs R@10，baseline 标星
python -m viz.pareto exp019-* exp018-* exp017-fixed-medium \
    --baseline exp017-fixed-medium --save pareto.png

# 降级预算表：delta% 一目了然
python -m viz.degradation exp019-* exp018-* \
    --baseline exp017-fixed-medium

# Pipeline 瀑布：逐阶段指标变化
python -m viz.pipeline \
    --ntp exp016-B-14d-S \
    --spdpo exp017-fixed-medium \
    --rfdpo exp019-* \
    --save pipeline.png
```

## 输出文件

`output_*.png` 已在 `.gitignore` 中排除，不会被提交。

## 数据源

```
experiments/ntp_checkpoints/{exp_name}/
├── train_meta.json    ← 训练配置 + eval 指标 (PPL, Recall@K)
├── train_log.jsonl    ← 逐步 loss/grad_norm 日志
└── probe.pt           ← 模型权重 (不读取)
```

## 依赖

仅 `matplotlib` + `numpy`（已在项目环境中）。
