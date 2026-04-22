---
last_updated: "2026-04-22T18:40:00"
current_task: null
next_experiment_number: 26
best_result:
  experiment: exp025-beam-passes
  ppl: 25.22
  recall_500: 0.636
total_experiments_run: 0
total_papers_read: 3
---

# Research Agent Status

## Current State

Idle. Awaiting human response on outbox/002-experiment-analysis.md (Page-wise NTP proposal).

## Recent Activity

- [2026-04-22 18:05] Activated session 2
- [2026-04-22 18:10~18:14] Read 3 papers: CRAB (2604.05113), NSGR (2604.05314), GenRec (2604.14878)
- [2026-04-22 18:20] Collected ALL EXP-022 results (5 configs) → contrastive loss DISCARD
- [2026-04-22 18:22] Collected EXP-025 results → beam_passes NEW BEST (R@500=63.6%)
- [2026-04-22 18:30] Wrote decision records: 003 (EXP-022 discard), 004 (EXP-025 merge)
- [2026-04-22 18:35] Wrote outbox/002 with full analysis + experiment proposals
- Updated experiments/log.md with complete EXP-022 and EXP-025 results

## Experiment Queue

Pending ideas (by priority):
- **IDEA-genrec-0** (P1, training.md): Page-wise NTP — 最高优先级, 需人类授权源码修改
- IDEA-feat-3 (P2): Item Category Token — lower priority
- IDEA-feat-4 (P2): User Profile Prefix — lower priority

Completed/Closed:
- ✅ IDEA-feat-0/1/2: time_gap + action + segment_emb — confirmed via EXP-023/025
- ❌ IDEA-onemall-0: contrastive loss — tested, negative, closed via EXP-022

## Open Questions

1. [outbox/002] Page-wise NTP 实验是否授权修改 ntp/preprocess.py 和 ntp/train.py?
2. [outbox/002] Session 分割标准偏好? (时间间隔 >30min vs 按天)

## Blockers

- 提案 A (Page-wise NTP) 需要源码修改授权
