# model/ — Embedding 编码与模型打包

Qwen3 Embedding 编码、批量处理、模型打包部署。

> **注意**：Tokenizer 代码（RKMeans、FSQ、preprocess_sid）已于 2026-04-30 迁移至 [`tokenizer/`](../tokenizer/)。
> `model/rkmeans.py`、`model/fsq.py`、`model/rkmeans_fsq.py` 保留为向后兼容 shim。

## 文件说明

| 文件 | 说明 |
|------|------|
| `embedders.py` | Qwen3TextEmbedder + Qwen3VLEmbedder |
| `encode.py` | 批量编码流水线（增量缓存 + OOM 重试 + S3 备份）|
| `train.py` | 端到端训练 CLI |
| `semantic_ids.py` | SID 工具函数 |
| `pack.py` | 打包 model.tar.gz + 上传 model registry 模型仓库 |
| `rkmeans.py` | ⚠️ shim → `tokenizer/rkmeans.py` |
| `fsq.py` | ⚠️ shim → `tokenizer/fsq.py` |
| `rkmeans_fsq.py` | ⚠️ shim → `tokenizer/rkmeans_fsq.py` |

## 核心流程

```
文本/图片 → embedders.py (Qwen3) → encode.py (批量+缓存)
    → tokenizer/ (量化训练 + SID 生成)
    → train.py (编排) → pack.py (打包部署)
```

## embedders.py

Qwen3 Embedding 模型封装，通过 `device` 参数区分运行模式：

- `device=None`: `device_map="auto"` (单进程多卡)
- `device="cuda:0"`: 显式放置到指定 GPU (torchrun 分布式)

**踩坑**：`torch_dtype` 必须显式传，否则 HF 默认 fp32，OOM。见 CLAUDE.md VL/Embedder 踩坑记录。
