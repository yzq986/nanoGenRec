# model/ — Embedding、RKMeans 模型与训练

Embedding 编码、残差量化 (RKMeans) 训练、Semantic ID 生成、模型打包部署。

## 文件说明

| 文件 | 说明 |
|------|------|
| `embedders.py` | Qwen3TextEmbedder + Qwen3VLEmbedder |
| `rkmeans.py` | FaissKMeansLayer + ResidualQuantizationMultiGPU |
| `semantic_ids.py` | `generate_semantic_ids()` — 从 RKMeans 模型生成离散 ID |
| `encode.py` | 批量编码流水线 (增量缓存 + OOM 重试 + S3 备份) |
| `train.py` | 端到端训练 CLI |
| `pack.py` | 打包 model.tar.gz + 上传 model registry 模型仓库 |

## 核心流程

```
文本/图片 → embedders.py (Qwen3) → encode.py (批量+缓存)
    → rkmeans.py (残差量化训练) → semantic_ids.py (生成 SID)
    → train.py (编排以上全部) → pack.py (打包部署)
```

## embedders.py

Qwen3 Embedding 模型封装，通过 `device` 参数区分运行模式：

- `device=None`: `device_map="auto"` (单进程多卡)
- `device="cuda:0"`: 显式放置到指定 GPU (torchrun 分布式)

## rkmeans.py

残差量化模型，逐层 KMeans：

1. Layer 0: 对 L2 归一化的 embedding 做 KMeans
2. Layer 1+: 对上层残差做 KMeans
3. 支持 FAISS GPU 多卡加速

## train.py

```bash
python -m gr_demo train --model qwen3-0.6b --input_path s3://... --num_clusters 1024
```

支持: embedding 缓存跳过、曝光 item 过滤、训练后自动评测、结果导出 S3
