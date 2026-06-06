# tokenizer/

[English](README.md) | [中文](README.zh.md)

将稠密商品嵌入转换为离散商品 ID 的 Semantic ID 分词器。

Tokenizer 是项目的表示层。它接收 Qwen3 商品嵌入，量化为紧凑的 3-token Semantic ID，并写入被 NTP 推荐器消费的 SID 缓存。

## 输出流程

```text
商品嵌入
  -> L2 归一化
  -> 残差 KMeans 层 1
  -> 残差 KMeans 层 2
  -> FSQ MLP 残差量化器
  -> Semantic ID: c1_c2_c3
```

当前推荐的系列是 4096x3 binary `[2]x12` SID，使用 EXP-049 选定的 `num_clusters=8192`。

## 文件

| 文件 | 用途 |
|------|------|
| `rkmeans.py` | 使用 `DatasetAssignGPU` 的 GPU 原生 `FaissKMeansLayer`。 |
| `fsq.py` | FSQ 层、学习型 FSQ 和 `FSQ_LEVEL_CONFIGS`。 |
| `rkmeans_fsq.py` | `ResKmeansFSQ`，2xKMeans + 1xFSQ 分词器。 |
| `preprocess_sid.py` | 用于训练 tokenizer 和写入 SID 缓存的 CLI 实现。 |

## 数据契约

SID 缓存应提供：

| 产物 | 含义 |
|------|------|
| `semantic_ids.npy` | 从商品 ID 到 SID 字符串的映射。 |
| tokenizer 权重/配置 | 用于复现分配的量化器状态。 |
| 元数据 | 模型名称、聚类数、FSQ 配置和日期/数据窗口。 |

下游 NTP 预处理假设每个行为商品都能解析为 SID。构建新缓存时，验证商品覆盖率是否覆盖目标行为数据窗口。

