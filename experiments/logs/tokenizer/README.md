# Tokenizer Experiments

Experiment summary for Semantic ID tokenization: embedding quality, residual KMeans, FSQ codebooks, collision, balance, and behavior-aware proxy metrics.

Code-level documentation lives in [tokenizer/README.md](../../../tokenizer/README.md). This file tracks conclusions and recommended settings.

## Current Recommendation

| Item | Value | Source |
|------|-------|--------|
| Recommended 0.6B SID cache | `exp049-0.6b-nc8192-h128` | EXP-049 |
| Recommended 4B SID cache | `exp049-4b-nc8192-h128` | EXP-049 |
| FSQ family | 4096x3 binary `[2]x12` | EXP-012 |
| `num_clusters` | 8192 | EXP-049 |
| 0.6B collision rate | 0.42% | EXP-049 |
| 4B collision rate | 1.28% | EXP-049 |
| 0.6B Gini_d2 | 0.2375 | EXP-049 |
| 4B Gini_d2 | 0.2530 | EXP-049 |
| 0.6B best snHR | 0.0919 | EXP-049 |
| 4B best snHR | 0.1307 | EXP-049 |

## Metric Guidance

| Metric | Direction | Use It For |
|--------|-----------|------------|
| `semantic_neighbor_hit_rate` / snHR | higher is better | Comparing embedding models and behavior-aware semantic quality. |
| `Gini_d2` | lower is better | Checking tokenizer balance under the same embedding model. |
| Collision rate | lower is better | Detecting SID capacity pressure. |

Important EXP-049 conclusion: **Gini_d2 should not be used to compare different embedding models.** The 4B embedding has worse Gini_d2 than 0.6B but substantially better snHR, which indicates stronger behavior-aware semantic neighborhoods rather than a worse tokenizer.

```bash
# Behavior-aware semantic neighbor hit rate
python experiments/scripts/run_snhr.py
```

## Known Invalid or Risky Runs

| Run | Issue | Status |
|-----|-------|--------|
| EXP-045 | Intended `num_clusters=4096`, but generated caches used `num_clusters=1024`. | Superseded by EXP-049. |
| EXP-026 8B cache | Item-ID alignment with behavior data was only 2.3%. | Needs rebuild before use. |
| 4B h sweep | Collision is insensitive to h because the 12d_4096 codebook is the bottleneck. | Increase FSQ capacity before retesting. |

## Experiment List

| EXP | Date | Status | Takeaway |
|-----|------|--------|----------|
| [001](../exp-001.md) | 2026-03 | completed | RKMeans training optimization v0 -> v7. |
| [002](../exp-002.md) | 2026-04-13 | completed | ResKmeansFSQ: 2-layer residual KMeans plus FSQ. |
| [003](../exp-003.md) | 2026-04-13 | completed | Learned FSQ with MLP projection and straight-through training. |
| [004](../exp-004.md) | 2026-04-13 | completed | OPQ parallel Semantic IDs. |
| [007](../exp-007.md) | 2026-04-13 | completed | Collaborative signal enhanced embedding. |
| [008](../exp-008.md) | 2026-04-14 | completed | FORGE proxy favored MLP-FSQ over OPQ. |
| [009](../exp-009.md) | 2026-04-14 | completed | QFormer tokenizer exploration. |
| [010](../exp-010.md) | 2026-04-15 | completed | End-to-end NTP with early MLP-FSQ SID was weak. |
| [011](../exp-011.md) | 2026-04-15 | completed | Codebook size ablation across 1024/4096 and OPQ. |
| [012](../exp-012.md) | 2026-04-15 | completed | 4096x3 binary SID selected as the baseline family. |
| [026](../exp-026.md) | 2026-04-27 | completed | Built 0.6B/4B/8B SID caches for the 14d data window. |
| [045](../exp-045.md) | 2026-04-29 | bug | h-dim sweep invalid because of the `num_clusters=1024` bug. |
| [049](../exp-049.md) | 2026-04-30 | completed | `num_clusters=8192` selected; h=64 and h=128 are effectively tied. |
