# eval/

[English](README.md) | [中文](README.zh.md)

Tokenizer 指标、行为感知指标、批量报告和全量召回 NTP 评估的评估框架。

使用此模块评估单个模型、比较多个运行，或生成可从实验日志链接的报告。

## 文件

| 文件 | 用途 |
|------|------|
| `wrapper.py` | 指标评估器使用的模型包装器。 |
| `evaluator.py` | 指标注册和执行核心。 |
| `behavior.py` | 行为感知的评估上下文和指标。 |
| `compare.py` | Markdown、JSON 和 CSV 比较报告。 |
| `batch.py` | 批量评估编排。 |
| `hyperparam.py` | Tokenizer 超参数搜索。 |

## 使用

```bash
# 单模型评估
python run.py eval --results_path s3://... --model_path s3://...

# 批量评估
python run.py eval-all --models qwen3-0.6b qwen3-4b --quick

# 比较已有结果
python run.py compare --eval_dir eval_results

# Tokenizer 超参数搜索
python run.py hyperparam --model qwen3-0.6b --skip_embedding

# 启用 NTP 的超参数搜索
python run.py hyperparam --model qwen3-0.6b --skip_embedding --run_ntp
```

全量 NTP 召回报告使用专用命令：

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 run.py eval-ntp \
    --checkpoint experiments/ntp_checkpoints/<name> \
    --n_recall 1000
```

## 指标流程

```text
wrapper.py
  -> evaluator.py
  -> behavior.py
  -> batch.py
  -> compare.py
```

## 报告规则

- 使用全量评估获取头条 Recall@K 值。
- 将训练过程中的 inline 评估仅视为健康检查。
- 将生成的报告保存在 `experiments/` 下（当其为可复现运行的一部分时）。
- 结论链接到 `experiments/logs/<phase>/exp-NNN.md`。

