# gr_demo — RKMeans Semantic ID 工具包

基于 Qwen3 Embedding + 残差量化 (RKMeans) 生成 Semantic ID，用于生成式推荐系统。

参考: [OneRec](https://arxiv.org/abs/2506.13695) / [OneRec-V2](https://arxiv.org/abs/2508.20900)

## 流程总览

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│ 1. 数据导出  │ ──→ │ 2. Embedding │ ──→ │ 3. RKMeans   │ ──→ │ 4. 评测      │ ──→ │ 5. 部署  │
│  Hive → S3  │     │  编码        │     │  训练 + SID  │     │  选模型/调参  │     │  打包上线 │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────┘
 data/export_hive    data/encode_dist     model/train           eval/*               model/pack
                     或 train 内置编码
```

所有命令通过 `python -m gr_demo <command>` 调用。

---

### Step 1. 数据导出 (Hive → S3)

PySpark 脚本，在 cloud notebook Notebook 里执行：

- 导出曝光内容 (文本 + 图片 URL) 到 S3
- 导出用户行为数据 (20+ 交互类型 → `action_bitmap`) 到 S3

对应文件: `data/export_hive.py`

---

### Step 2. Embedding 编码

将文本/图片编码为 dense embedding 向量。两种方式：

```bash
# 方式 A: train 命令内置编码 (单机，自动缓存)
python -m gr_demo train --model qwen3-0.6b
# 首次运行会编码 → 缓存到 EFS，后续 --skip_embedding 跳过

# 方式 B: torchrun 分布式编码 (大数据量推荐)
torchrun --nproc_per_node=8 -m gr_demo.data.encode_distributed --model qwen3-0.6b
# 8 卡并行，增量缓存，OOM 自动减半 batch size
```

---

### Step 3. 训练 RKMeans + 生成 Semantic ID

```bash
# 端到端: 编码 → 训练 RKMeans → 生成 SID → 导出到 S3
python -m gr_demo train --model qwen3-0.6b

# 已有 embedding 缓存时跳过编码
python -m gr_demo train --model qwen3-0.6b --skip_embedding

# 自定义聚类参数
python -m gr_demo train --model qwen3-4b --num_clusters 2048 --niter 50 --nredo 5

# 只保留有曝光的 item 做训练
python -m gr_demo train --model qwen3-0.6b --behavior_path s3://bucket/behavior/2026-04-01

# 训练完顺便跑 intrinsic 评测
python -m gr_demo train --model qwen3-0.6b --skip_embedding --eval_intrinsic
```

---

### Step 4. 评测

#### 4a. 单模型评测

```bash
# 全量评测 (intrinsic + behavior)
python -m gr_demo eval \
    --results_path s3://bucket/rkmeans/qwen3-0.6b/results.parquet \
    --model_path s3://bucket/rkmeans/qwen3-0.6b/rkmeans.pt \
    --behavior_path s3://bucket/behavior/2026-04-01

# 只看 intrinsic 指标 (不需要行为数据，快)
python -m gr_demo eval --results_path s3://... --model_path s3://... --intrinsic_only

# 只跑特定指标
python -m gr_demo eval --results_path s3://... --metrics reconstruction_loss entropy
```

#### 4b. 批量评测 + 模型对比

```bash
# 一键跑全部模型 (qwen3-0.6b, 4b, 8b, vl-2b) + 生成对比报告
python -m gr_demo eval-all

# 快速模式 (采样 5 万)
python -m gr_demo eval-all --quick

# 只跑指定模型
python -m gr_demo eval-all --models qwen3-0.6b qwen3-4b

# 只跑 SID 预测 (NTP Transformer+MoE)
python -m gr_demo eval-all --only-sid

# 已有各模型结果，只生成对比报告
python -m gr_demo eval-all --compare-only
# 或
python -m gr_demo compare --eval_dir eval_results
```

#### 4c. 超参数搜索

```bash
# 网格搜索 num_clusters × niter × nredo
python -m gr_demo hyperparam --model qwen3-0.6b --skip_embedding

# 自定义搜索空间
python -m gr_demo hyperparam --model qwen3-0.6b --skip_embedding \
    --clusters 256 512 1024 2048 --niters 25 50 --nredos 1 3

# 断点续搜
python -m gr_demo hyperparam --model qwen3-0.6b --skip_embedding --append
```

---

### Step 5. 打包部署

```bash
# 打包 model.tar.gz (Qwen 模型 + RKMeans 权重)
python -m gr_demo pack \
    --rkmeans_s3_path s3://bucket/rkmeans/qwen3-0.6b/rkmeans.pt

# 打包 + 上传 model registry 模型仓库
python -m gr_demo pack \
    --rkmeans_s3_path s3://bucket/rkmeans/qwen3-0.6b/rkmeans.pt \
    --qwen_model Qwen/Qwen3-Embedding-0.6B \
    --upload
```

---

## 目录结构

```
gr_demo/
├── config.py          # 模型配置 (MODEL_CONFIGS) + Config dataclass
├── s3_utils.py        # S3 上传/下载/路径解析
├── cli.py             # 统一 CLI 入口 (subcommand 分发)
├── data/              # 数据加载、分布式编码、Hive 导出
├── model/             # Embedder、RKMeans、训练、打包
├── eval/              # 评测、对比、超参搜索
├── metrics/           # 指标框架 (intrinsic + behavior)
├── config/         # 敏感配置 (独立 git 仓库, .gitignore)
└── docs/              # 架构设计文档
```

## 支持的 Embedding 模型

| Key | HuggingFace Model | Dim | 多模态 | Batch Size (8xA100) |
|-----|-------------------|-----|--------|---------------------|
| `qwen3-vl-8b` | Qwen/Qwen3-VL-Embedding-8B | 4096 | Yes | 8 |
| `qwen3-vl-2b` | Qwen/Qwen3-VL-Embedding-2B | 2048 | Yes | 16 |
| `qwen3-8b` | Qwen/Qwen3-Embedding-8B | 4096 | No | 16 |
| `qwen3-4b` | Qwen/Qwen3-Embedding-4B | 2560 | No | 32 |
| `qwen3-0.6b` | Qwen/Qwen3-Embedding-0.6B | 1024 | No | 64 |

## 环境依赖

```bash
pip install -r requirements.txt
```

核心依赖: `torch`, `transformers`, `faiss-gpu`, `boto3`, `s3fs`, `pandas`, `pyarrow`
