# gr_demo

[English](README.md) | [中文](README.zh.md)

`gr_demo` 是一个围绕 Semantic ID、序列建模和偏好对齐构建的生成式推荐研究项目。

项目把 item embedding 转换为离散 Semantic ID，在用户行为序列上训练自回归推荐模型，并用全量召回评测和 YAML 实验配置管理结果。这个仓库的目标不是做零散脚本集合，而是提供一套可复现实验系统和 Semantic-ID-based generative recommendation 的工程参考。

## 亮点

- **端到端 Semantic ID pipeline**：Qwen3 embedding -> residual KMeans + FSQ -> 3-token item ID。
- **生成式推荐模型**：Transformer + MoE，在 SID 序列上做 next-token prediction。
- **偏好对齐链路**：SP-DPO、RF-DPO、GRPO、ECPO。
- **全量召回评测**：SID 约束 beam search、Recall@K、tokenizer proxy metrics 和对比报告。
- **可复现实验治理**：YAML config、重复实验检查、阶段日志、队列式长任务管理。

参考论文：[OneRec](https://arxiv.org/abs/2506.13695)、[OneRec-V2](https://arxiv.org/abs/2508.20900)、[GR4AD](https://arxiv.org/abs/2602.22732)、[OneMall](https://arxiv.org/abs/2601.21770)。

## 当前结果

| 方向 | 当前最好结果 | 代表实验 | 详情 |
|------|-------------|----------|------|
| Semantic ID tokenizer | 4096x3 binary `[2]x12`，snHR=0.095，CR=0.89% | EXP-012 | [tokenizer logs](experiments/logs/tokenizer/README.md) |
| Embedding scale | 4B SID snHR=0.131；0.6B SID snHR=0.092 | EXP-049 | [tokenizer logs](experiments/logs/tokenizer/README.md) |
| NTP recommender | M-tier bare R@500=70.2%；L-tier SFT R@500=64.1% | EXP-043 / EXP-047 | [NTP logs](experiments/logs/ntp/README.md) |
| RL alignment | S-tier 链路上 ECPO R@500=65.7% | EXP-039B | [RL logs](experiments/logs/rl/README.md) |

EXP-049 确认 `num_clusters=8192` 是更强 tokenizer 设置。当前推荐 SID cache 是 `exp049-0.6b-nc8192-h128` 和 `exp049-4b-nc8192-h128`。

## 项目流程

```mermaid
graph LR
    A["行为和 item 数据"] --> B["Qwen3 embeddings"]
    B --> C["Semantic ID tokenizer"]
    C --> D["NTP recommender"]
    D --> E["Preference alignment"]
    E --> F["Full-recall evaluation"]
    F --> G["Experiment logs"]
```

所有项目命令从仓库根目录运行：

```bash
python run.py <command>
```

分布式任务使用：

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 run.py <command>
```

## 快速开始

```bash
# 训练 tokenizer 并生成 Semantic IDs
python run.py train --model qwen3-0.6b

# 复用已有 embedding cache
python run.py train --model qwen3-0.6b --skip_embedding

# 构建 NTP 数据分片
python run.py preprocess-ntp \
    --sid_cache experiments/sid_cache/<sid-cache-name> \
    --output_dir experiments/ntp_data/<data-name> \
    --date_start 2026-03-18 \
    --date_end 2026-03-31

# 通过 YAML config 运行实验
python experiments/run_exp.py experiments/configs/exp-047.yaml --no-smoke --commit

# 全量评测
PYTHONPATH=. torchrun --nproc_per_node=8 run.py eval-ntp \
    --checkpoint experiments/ntp_checkpoints/<name> \
    --n_recall 1000
```

标准训练/评测环境是 `/home/dev/.conda/envs/gr`。

| Package | Version |
|---------|---------|
| Python | 3.12.13 |
| torch | 2.7.1+cu128 |
| CUDA driver | 12.8 |
| faiss-gpu | 1.14.1 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| pyarrow | 24.0.0 |

## 实验工作流

新实验统一走 `experiments/run_exp.py` + YAML config。这样可以显式继承默认参数，避免重复跑 baseline，并让结果可比较。

```bash
# 写 config 前先看默认值
sed -n '1,220p' experiments/configs/_base.yaml

# 检查是否已有相似实验
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --check

# 跑所有 variants
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --no-smoke --commit

# 只跑或恢复一个 variant
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --only expNNN-a --no-smoke
```

长任务通过队列追加：

```bash
echo "run_config.sh experiments/configs/exp-NNN.yaml  /tmp/expNNN.log  exp-NNN complete!" >> experiments/queue.txt
```

`train-ntp` 的 inline eval 只用于健康检查。正式报告数字必须来自 `run.py eval-ntp --n_recall 1000` 的全量评测。

## 目录结构

| 路径 | 作用 |
|------|------|
| [data/](data/README.md) | 数据导出、加载、embedding 同步和分布式编码。 |
| [tokenizer/](tokenizer/README.md) | Semantic ID tokenizer 和 SID 预处理。 |
| [ntp/](ntp/README.md) | 生成式推荐模型、预处理、训练和评测。 |
| [rl/](rl/README.md) | 偏好学习和 RL 对齐。 |
| [eval/](eval/README.md) | 评测 wrapper、行为指标、全量召回报告和对比。 |
| [metrics/](metrics/README.md) | tokenizer / embedding 的 intrinsic 和 behavior-aware metrics。 |
| [model/](model/README.md) | embedding wrapper、模型打包和兼容 shim。 |
| [viz/](viz/README.md) | 训练后可视化工具。 |
| [experiments/](experiments/README.md) | configs、编排、队列、checkpoints 和结果产物。 |
| [experiments/logs/](experiments/logs/README.md) | 分阶段实验记录和 SOTA 汇总。 |
| [docs/](docs/README.zh.md) | 架构、工程记录和长期文档。 |
| [ideas/](ideas/README.md) | 按改进方向组织的研究 backlog。 |

## 文档分工

| 文件 | 读者 | 内容 |
|------|------|------|
| `README.md` / `README.zh.md` | 新访客 | 项目定位、快速开始、核心结果和文档入口。 |
| `<phase>/README.md` | 改代码的人 | 模块职责、接口、数据契约、实现细节和注意事项。 |
| `experiments/logs/<phase>/README.md` | 设计实验的人 | 当前最好结果、实验列表、下一步方向和指标口径。 |
| `docs/` | 长期维护者 | 架构决策、工程记录和稳定文档。 |

## 注意事项

- 仓库根目录直接加入 `PYTHONPATH`，模块导入不带 `gr_demo.` 前缀。
- CLI 入口是 `python run.py <command>`，不是 `python -m gr_demo`。
- 独立 shell 脚本需要先把 `PYTHONPATH` 设置为仓库根目录。
