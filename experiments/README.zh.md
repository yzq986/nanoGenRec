# experiments/

[English](README.md) | [中文](README.zh.md)

实验编排、配置、队列、检查点和生成的结果产物。

此目录是可复现运行的操作中心。用于定义实验、检查重复基线、运行变体，以及存储复现或检查结果所需的产物。结论性叙述保存在 [experiments/logs/](logs/) 中。

## 目录结构

| 路径 | 用途 |
|------|------|
| `configs/` | YAML 实验定义。新实验应从这里开始。 |
| `configs/_base.yaml` | 共享默认值。写新配置前先读此文件。 |
| `run_exp.py` | 主要实验运行器，支持配置扩展和重复运行检查。 |
| `scripts/run_config.sh` | 队列友好的 `run_exp.py --no-smoke --commit` 包装器。 |
| `queue.txt` | 仅追加的长时间运行实验队列。 |
| `queue_state.json` | 当前队列守护进程状态。 |
| `sid_cache/` | Semantic ID 缓存产物。 |
| `ntp_data/` | NTP 预处理分片。 |
| `ntp_checkpoints/` | 训练输出、`train_meta.json` 和 `train_log.jsonl`。 |
| `logs/` | 人类可读的实验记录和阶段总结。 |

## 标准流程

```bash
# 1. 检查默认值
sed -n '1,220p' experiments/configs/_base.yaml

# 2. 检查是否已有相似的运行
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --check

# 3. 运行所有变体
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --no-smoke --commit

# 4. 恢复或运行单个变体
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --only expNNN-a --no-smoke
```

排队异步执行：

```bash
echo "run_config.sh experiments/configs/exp-NNN.yaml  /tmp/expNNN.log  exp-NNN complete!" >> experiments/queue.txt
```

## 配置指南

- 从 `configs/_base.yaml` 开始；只覆盖实验相关的字段。
- 写新配置前务必从已有配置验证 `sid_cache_name`、`ntp_data_name` 和日期范围。
- 使用 `variants:` 进行受控比较。
- 不要重跑完全相同的基线；在日志中引用已有实验。
- 如果实验只更改评估代码，则对已有检查点重新评估，而不是重新训练。

