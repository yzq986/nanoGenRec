# metrics/

[English](README.md) | [Chinese](README.zh.md)

Metric implementations for embedding quality, Semantic ID quality, and behavior-aware proxy evaluation.

Metrics are intentionally modular. Each metric implements `BaseMetric.compute()` and returns a `MetricResult` that can be consumed by report generators and comparison tools.

## Metric Groups

| Group | Requires Behavior Data | Purpose |
|-------|------------------------|---------|
| Intrinsic | no | Quantization quality, codebook health, distribution balance. |
| Behavior-aware | yes | Whether embedding/SID neighborhoods preserve user-behavior signal. |
| End-to-end | yes | NTP recall through sequence modeling and beam search. |

## Intrinsic Metrics

| Metric | File | Meaning |
|--------|------|---------|
| `reconstruction_loss` | `reconstruction.py` | L2 reconstruction loss after quantization. |
| `codebook_utilization` | `codebook.py` | Fraction of the SID/codebook space being used. |
| `entropy` | `entropy.py` | Shannon entropy of assignment distribution. |
| `cosine_similarity` | `similarity.py` | Embedding similarity distribution. |
| `effective_dimension` | `effective_dim.py` | PCA-style embedding space usage. |
| `semantic_id_collision` | `collision.py` | Fraction of distinct items sharing identical SIDs. |
| `cluster_balance` | `cluster_balance.py` | Cluster-size balance, including Gini-style summaries. |

## Behavior-Aware Metrics

| Metric | File | Meaning |
|--------|------|---------|
| `user_semantic_consistency` | `behavior.py` | Whether a user's liked items are nearby in SID space. |
| `semantic_neighbor_hit_rate` | `behavior.py` | Whether SID-neighbor items share behavior audiences. |
| `embedding_behavior_correlation` | `behavior.py` | Correlation between embedding similarity and behavior overlap. |
| `positive_negative_separation` | `behavior.py` | Distance separation between positive and negative samples. |
| `embedding_hit_rate` | `embedding_hitrate.py` | Fast I2I proxy for behavior-aware embedding quality. |
| `semantic_id_prediction` | `sid_prediction.py` | End-to-end SID prediction through NTP; expensive. |

## Practical Guidance

- Use `semantic_neighbor_hit_rate` to compare different embedding models.
- Use Gini and collision metrics to debug tokenizer structure under the same embedding model.
- Use NTP Recall@K only when the end-to-end recommender is the question.
- Do not compare train-time inline eval against full-recall baselines.

## Reports

`report.py` can emit:

- JSON for structured results;
- Markdown for human-readable experiment reports;
- CSV for cross-run comparison.
