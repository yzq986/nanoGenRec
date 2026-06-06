# model/

[English](README.md) | [中文](README.zh.md)

Embedding 包装器、编排辅助工具、打包工具和兼容性垫片。

Tokenizer 实现已迁移到 [tokenizer/](../tokenizer/)。此目录中的旧 tokenizer 文件仅用于保留旧版导入。

## 文件

| 文件 | 用途 |
|------|------|
| `embedders.py` | Qwen3 文本和视觉语言嵌入包装器。 |
| `encode.py` | 批量编码流水线，支持缓存、OOM 重试和可选的远程备份。 |
| `train.py` | 面向嵌入、tokenizer 训练和 SID 导出的端到端 CLI 编排。 |
| `semantic_ids.py` | Semantic ID 工具函数。 |
| `pack.py` | 部署产物的打包入口。 |
| `rkmeans.py` | 到 `tokenizer/rkmeans.py` 的兼容垫片。 |
| `fsq.py` | 到 `tokenizer/fsq.py` 的兼容垫片。 |
| `rkmeans_fsq.py` | 到 `tokenizer/rkmeans_fsq.py` 的兼容垫片。 |

## 典型流程

```text
原始商品文本/图片
  -> embedders.py
  -> encode.py 缓存
  -> tokenizer/
  -> train.py 编排
  -> pack.py 部署产物
```

## 实现说明

- 在 embedder 包装器中明确传递 `torch_dtype`。静默的 fp32 回退可能导致大 OOM。
- 除非需要每一层的输出，否则避免使用 `output_hidden_states=True`。
- OOM 重试路径必须在调用 `empty_cache()` 前删除活跃的张量引用。
- 不要将纯文本缓存复用于视觉语言输入；具有不同图像的相同文本可能产生不同的嵌入。

