# Tokenizer Experiments

[English](README.md) | [中文](README.zh.md)

Semantic ID 分词的实验总结：嵌入质量、残差 KMeans、FSQ 码本、碰撞、平衡和行为感知代理指标。

代码级文档见 [tokenizer/README.md](../../../tokenizer/README.md)。此文件跟踪结论和推荐设置。

## 当前推荐

| 项目 | 值 | 来源 |
|------|-----|------|
| 推荐的 0.6B SID 缓存 | `exp049-0.6b-nc8192-h128` | EXP-049 |
| 推荐的 4B SID 缓存 | `exp049-4b-nc8192-h128` | EXP-049 |
| FSQ 系列 | 4096x3 binary `[2]x12` | EXP-012 |
| `num_clusters` | 8192 | EXP-049 |
| 0.6B 碰撞率 | 0.42% | EXP-049 |
| 4B 碰撞率 | 1.28% | EXP-049 |
| 0.6B Gini_d2 | 0.2375 | EXP-049 |
| 4B Gini_d2 | 0.2530 | EXP-049 |
| 0.6B 最佳 snHR | 0.0919 | EXP-049 |
| 4B 最佳 snHR | 0.1307 | EXP-049 |

## 指标指南

| 指标 | 方向 | 用途 |
|------|------|------|
| `semantic_neighbor_hit_rate` / snHR | 越高越好 | 比较嵌入模型和行为感知语义质量。 |
| `Gini_d2` | 越低越好 | 在相同嵌入模型下检查 tokenizer 平衡性。 |
| 碰撞率 | 越低越好 | 检测 SID 容量压力。 |

重要 EXP-049 结论：**Gini_d2 不应用于比较不同的嵌入模型。** 4B 嵌入的 Gini_d2 比 0.6B 差，但 snHR 明显更好，这表明更强的行为感知语义邻域，而非更差的 tokenizer。

