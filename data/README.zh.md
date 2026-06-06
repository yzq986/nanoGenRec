# data/

[English](README.md) | [中文](README.zh.md)

数据加载、导出、Embedding 同步和分布式编码工具。

该模块将原始的商品/行为数据连接到 tokenizer 和 NTP 流水线。支持本地和远程路径，但下游实验应使用 `experiments/` 下的稳定缓存目录。

## 文件

| 文件 | 用途 |
|------|------|
| `loaders.py` | 共享的 S3/本地加载和导出辅助函数。 |
| `encode_distributed.py` | 使用 `torchrun` 的多 GPU 文本嵌入。 |
| `export_content.py` | 商品文本和图片 URL 的 PySpark 导出。 |
| `export_behavior.py` | 用户行为位图的 PySpark 导出。 |
| `sync_embeddings.py` | Embedding 缓存同步辅助。 |
| `migrate_shards.py` | 分片迁移和兼容性更新工具。 |

## 数据契约

| 数据集 | 必需字段 | 使用方 |
|--------|---------|--------|
| 商品内容 | item ID, text, 可选 image URL | Embedding 和 tokenizer 训练。 |
| 行为事件 | user ID, item ID, timestamp, action bitmap | NTP 预处理和行为指标。 |
| Embedding 缓存 | item ID, dense vector, metadata | Tokenizer 训练和代理指标。 |
| SID 缓存 | item ID -> SID 映射 | NTP 预处理和受限解码。 |

NTP 数据窗口必须与 SID 缓存兼容：用于训练/评估的每个行为商品都应被 SID 缓存覆盖。

## 分布式编码

`encode_distributed.py` 专为大型商品集设计：

- rank 0 协调模型下载和缓存合并；
- 所有 rank 编码不相交的分片；
- 跳过已编码的商品 ID；
- CUDA OOM 时重试并减小 batch size；
- 最终输出合并到稳定缓存中。

## 数据说明

行为分布呈现强烈的长尾特征。在 2026-01-25 至 2026-03-31 的观测窗口内：

| 窗口 | 用户数 | 正向事件 | 均值/用户 | P50 | P95 | P99 |
|------|--------|---------|-----------|-----|-----|-----|
| 7d | 1.54M | 23.9M | 15.6 | 3 | 68 | 220 |
| 14d | 2.51M | 53.1M | 21.2 | 3 | 92 | 331 |
| 31d | 4.55M | 129.7M | 28.5 | 3 | 118 | 499 |
| 62d | 7.29M | 261.8M | 35.9 | 3 | 138 | 669 |
| 66d | 7.85M | 299.0M | 38.1 | 3 | 146 | 715 |

对于 `max_seq_len=512` 和每个商品 3 个 SID token，NTP 保留每个用户最近的 170 个商品。这只影响一小部分用户，但这些用户贡献了原始交互中的大部分。

