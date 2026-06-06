# metrics/

[English](README.md) | [中文](README.zh.md)

嵌入质量、Semantic ID 质量和行为感知代理评估的指标实现。

指标采用模块化设计。每个指标实现 `BaseMetric.compute()` 并返回一个 `MetricResult`，可供报告生成器和比较工具使用。

## 指标分组

| 分组 | 是否需要行为数据 | 用途 |
|------|-----------------|------|
| 内在指标 | 否 | 量化质量、码本健康度、分布平衡性。 |
| 行为感知 | 是 | Embedding/SID 邻域是否保留了用户行为信号。 |
| 端到端 | 是 | 通过序列建模和 beam search 的 NTP 召回。 |

## 内在指标

| 指标 | 文件 | 含义 |
|------|------|------|
| `reconstruction_loss` | `reconstruction.py` | 量化后的 L2 重构损失。 |
| `codebook_utilization` | `codebook.py` | SID/码本空间的已用比例。 |
| `entropy` | `entropy.py` | 分配分布的香农熵。 |
| `cosine_similarity` | `similarity.py` | Embedding 相似度分布。 |
| `effective_dimension` | `effective_dim.py` | PCA 风格的嵌入空间使用率。 |
| `semantic_id_collision` | `collision.py` | 共享相同 SID 的不同商品比例。 |
| `cluster_balance` | `cluster_balance.py` | 聚类大小平衡性，包括基尼系数类摘要。 |

## 行为感知指标

| 指标 | 文件 | 含义 |
|------|------|------|
| `user_semantic_consistency` | `behavior.py` | 用户喜欢的商品是否在 SID 空间中彼此接近。 |
| `semantic_neighbor_hit_rate` | `behavior.py` | SID 邻居商品是否共享行为受众。 |
| `embedding_behavior_correlation` | `behavior.py` | 嵌入相似度与行为重叠之间的相关性。 |
| `positive_negative_separation` | `behavior.py` | 正负样本之间的距离分离度。 |
| `embedding_hit_rate` | `embedding_hitrate.py` | 快速的 I2I 代理，用于行为感知的嵌入质量。 |
| `semantic_id_prediction` | `sid_prediction.py` | 通过 NTP 的端到端 SID 预测；计算量大。 |

## 实用指南

- 使用 `semantic_neighbor_hit_rate` 比较不同的嵌入模型。
- 在相同嵌入模型下，使用基尼系数和碰撞指标调试 tokenizer 结构。
- 只有当问题是关于端到端推荐器时才使用 NTP Recall@K。
- 不要比较训练时的 inline 评估与全量召回基线。

