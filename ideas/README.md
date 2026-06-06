# Ideas

[English](README.md) | [Chinese](README.zh.md)

Research backlog distilled from papers, technical reports, experiment discussions, and completed ablations.

Ideas are grouped by improvement dimension. Each topic file keeps the detailed record for that dimension: candidate ideas, dependencies, experiment designs, priorities, and closure notes. Mature ideas should become formal experiments under [experiments/logs/](../experiments/logs/README.md).

## Summary

| Metric | Count |
|--------|------:|
| Total ideas | 114 |
| Active P0 | 0 |
| Approx. P1 | 63 |
| Approx. P2 | 40 |
| Completed or closed | 11 |

Tier labels:

| Tier | Meaning |
|------|---------|
| A | Industrial deployment evidence plus A/B result. |
| B | Industrial authorship or top-conference evidence, without A/B result. |
| C | Academic or early technical idea worth tracking. |

Closed or completed ideas include `sid-0`, `sid-1`, `onemall-0`, `onemall-4`, `onemall-5`, `forge-0`, `feat-0/1/2`, `rpo-0`, `spot-0`, `lac-0`, and `static-0`. See the per-dimension files for rationale and experiment links.

## Index

| File | Dimension | Ideas | Active P0 |
|------|-----------|------:|-----------|
| [tokenizer.md](tokenizer.md) | Quantization and Semantic ID construction | 19 | none |
| [embedding.md](embedding.md) | Representation quality and multimodal signals | 8 | none |
| [architecture.md](architecture.md) | Model architecture and retrieval/generation structure | 31 | none |
| [training.md](training.md) | Training objectives and supervision signals | 21 | none |
| [rl-alignment.md](rl-alignment.md) | Preference learning and RL alignment | 14 | none |
| [inference.md](inference.md) | Decoding, constraints, serving, and compression | 9 | none |
| [scaling.md](scaling.md) | Sequence length, distributed training, and scaling | 6 | none |
| [ntp-features.md](ntp-features.md) | NTP side-feature injection | 6 | none |

## Planning Principles

- Prefer ideas backed by industrial deployment evidence, measured ablations, or direct relevance to current bottlenecks.
- Promote an idea to experiment only when its data pipeline, baseline, metric, and expected runtime are explicit.
- Close negative ideas with the reason preserved; do not delete failed paths from the research lineage.
- Avoid rerunning an existing baseline. Reference the matching EXP record instead.

## Dependency Map

The full dependency map is maintained in the [Chinese README](README.zh.md) because it preserves the original research planning notes and paper-tracking context. The English default page intentionally keeps the high-level index compact.
