# model/ — Embedding 编码与模型打包

Qwen3 Embedding 编码、批量处理、模型打包部署。

> **注意**：Tokenizer 代码（RKMeans、FSQ、preprocess_sid）已于 2026-04-30 迁移至 [`tokenizer/`](../tokenizer/)。
> `model/rkmeans.py`、`model/fsq.py`、`model/rkmeans_fsq.py` 保留为向后兼容 shim。

## 文件

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

## embedders.py 踩坑

### `torch_dtype` 必须显式传
`Qwen3TextEmbedder` 默认了 `torch_dtype=torch.float16`，`Qwen3VLEmbedder` 漏传 → HF fallback 到 fp32。2B 模型 fp32 权重 ~10GB + fp32 activations，8192 seq batch=8 就能吃满 40GB OOM。**写 embedder 包装时永远显式传 `torch_dtype` (fp16/bf16)**。

### `output_hidden_states=True` 是显存放大器
只为取 `hidden_states[-1]` 却开了这个 flag，HF 会 materialize 所有 30+ 层 hidden states。2B 模型 seq=8192 fp32 下这一项就是 20–30GB。直接从 `outputs.last_hidden_state` 拿，不要开全量 hidden_states。

### OOM 诊断
打印 `text_len + mem alloc/reserved/total`。`memory_allocated == memory_reserved == 总量 97%` 是 dtype 问题（不是碎片）。OOM skip 路径里要 `del sub_inputs + gc.collect() + empty_cache() + synchronize()`，只调 `empty_cache()` 释放不掉 Python 仍持有引用的 GPU tensor。

### VL 场景别复用 text LFU cache
相同文本 + 不同图片 → embedding 不同，text 缓存会返回错误结果。必须 `if not is_vl and ...` 分叉。

## 实验记录

见 [`experiments/logs/tokenizer/README.md`](../experiments/logs/tokenizer/README.md)。
