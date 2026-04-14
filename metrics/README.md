# metrics/ — 指标框架

模块化的 Embedding 和 Semantic ID 质量评估指标。

## 架构

所有指标继承 `BaseMetric`，实现 `compute()` 方法，返回 `MetricResult`。

```python
from gr_demo.metrics import INTRINSIC_METRICS, BEHAVIOR_METRICS, AVAILABLE_METRICS
```

## Intrinsic 指标 (无需行为数据)

| 指标 | 文件 | 说明 |
|------|------|------|
| `reconstruction_loss` | `reconstruction.py` | L2 重建损失 — 量化精度 |
| `codebook_utilization` | `codebook.py` | Codebook 利用率 — SID 空间使用比例 |
| `entropy` | `entropy.py` | Shannon 熵 — SID 分布均匀度 |
| `cosine_similarity` | `similarity.py` | 余弦相似度分布 — embedding 区分度 |
| `effective_dimension` | `effective_dim.py` | 有效维度 (PCA) — embedding 空间利用率 |
| `semantic_id_collision` | `collision.py` | SID 碰撞率 — 不同 item 获得相同 SID 的比例 |
| `cluster_balance` | `cluster_balance.py` | 聚类均衡度 (Gini) — bucket 大小分布 |

## Behavior 指标 (需要用户行为数据)

| 指标 | 文件 | 说明 |
|------|------|------|
| `user_semantic_consistency` | `behavior.py` | 用户语义一致性 — 用户喜欢的内容 SID 是否相近 |
| `semantic_neighbor_hit_rate` | `behavior.py` | 语义邻居命中率 — 同 SID 前缀的 item 是否被同一用户群喜欢 |
| `embedding_behavior_correlation` | `behavior.py` | Embedding-行为相关性 — 余弦相似度 vs 用户重叠 Jaccard |
| `positive_negative_separation` | `behavior.py` | 正负样本分离度 — 喜欢 vs 不喜欢内容的 embedding 距离 |
| `embedding_hit_rate` | `embedding_hitrate.py` | Embedding 命中率 (FORGE proxy) — I2I 检索邻居与行为共现率，**默认开启** |
| `semantic_id_prediction` | `sid_prediction.py` | SID 序列预测 (NTP) — Transformer+MoE beam search 评估，需 `--run_ntp` 开启 |

## 报告生成

`report.py` 中的 `ReportGenerator` 输出三种格式:

- **JSON**: 结构化结果 + 元数据 + 状态统计
- **Markdown**: 摘要表格 + 逐指标详情 + 解读指南
- **CSV**: 扁平表格，用于跨模型对比

## embedding_hit_rate 与 NTP recall 的关系

两者测的是不同阶段的能力:

- **embedding_hit_rate (HR@50)** = 教材质量评估（知识点组织得好不好）
- **NTP recall@K** = 学生考试成绩（最终答对了多少题）

教材好，学生**有可能**考好；教材烂，学生**一定**考不好。但教材好不代表学生一定考好——还取决于学生能力（模型大小）、学习时间（训练量）、看了多少章（序列长度）。

因果链:

```
embedding 质量 (HR@50) → SID 质量 → NTP 学习难度 → NTP recall@K
```

日常用 `embedding_hit_rate`（秒级 proxy），NTP（`--run_ntp`）仅在需要端到端 recall 数字时开启。

## 质量等级

每个指标通过阈值自动判定等级: `excellent` / `good` / `acceptable` / `poor`
