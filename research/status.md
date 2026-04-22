---
last_updated: "2026-04-22T20:15:00"
current_task: null
next_experiment_number: 26
best_result:
  experiment: exp025-beam-passes
  ppl: 25.22
  recall_500: 0.636
total_experiments_run: 0
total_papers_read: 12
---

# Research Agent Status

## Current State

Idle. Awaiting human response on:
- outbox/002-experiment-analysis.md (Page-wise NTP proposal)
- outbox/005-spot-dpo-finding.md (IDEA-spot-0: SPoT-BCO for SP-DPO, 需要 rl/ 源码授权)

## Recent Activity

- [2026-04-22 20:05] Activated session 5 (/loop)
- [2026-04-22 20:07~20:11] Read 3 papers: TCA4Rec (2601.18457), FlexCode (2511.20673), MetaGDPO (2511.12113)
- [2026-04-22 19:45] Activated session 4 (/loop)
- [2026-04-22 19:47~19:51] Read 3 papers: OneLive (2602.08612), OneMall (2601.21770), FuXi-Linear (2602.23671)
- [2026-04-22 19:15] Activated session 3
- [2026-04-22 18:05] Activated session 2
- [2026-04-22 18:10~18:14] Read 3 papers: CRAB (2604.05113), NSGR (2604.05314), GenRec (2604.14878)
- [2026-04-22 18:20] Collected ALL EXP-022 results → contrastive loss DISCARD
- [2026-04-22 18:22] Collected EXP-025 results → beam_passes NEW BEST (R@500=63.6%)
- [2026-04-22 18:30] Wrote decision records: 003 (EXP-022 discard), 004 (EXP-025 merge)
- [2026-04-22 18:35] Wrote outbox/002 with full analysis + experiment proposals
- [2026-04-22 19:15] Activated session 3
- [2026-04-22 19:20~19:25] Read 3 papers: RCLRec (2603.28124), OneRec-V2 Quant (2603.11486), SPOT (2603.01683)
- [2026-04-22 19:30] Wrote outbox/005 with SPOT finding + IDEA-spot-0 proposal

## Experiment Queue

Pending ideas (by priority):
- **IDEA-genrec-0** (P1, training.md): Page-wise NTP — 最高优先级, 需人类授权 ntp/ 源码修改
- **IDEA-spot-0** (P1, rl-alignment.md): SPoT-BCO 替换 SP-DPO loss — 需人类授权 rl/ 源码修改
- IDEA-mtgr-0 (P0, training.md): User-Level Packing + Causal Mask — 需人类授权 ntp/ 源码修改
- IDEA-feat-3 (P2): Item Category Token — lower priority
- IDEA-feat-4 (P2): User Profile Prefix — lower priority

Completed/Closed:
- ✅ IDEA-feat-0/1/2: time_gap + action + segment_emb — confirmed via EXP-023/025
- ❌ IDEA-onemall-0: contrastive loss — tested, negative, closed via EXP-022

## Open Questions

1. [outbox/002] Page-wise NTP 实验是否授权修改 ntp/preprocess.py 和 ntp/train.py?
2. [outbox/002] Session 分割标准偏好? (时间间隔 >30min vs 按天)
3. [outbox/005] IDEA-spot-0 (SPoT-BCO for SP-DPO) 是否授权修改 rl/ 源码?

## Blockers

- 所有高优先级实验均需源码修改授权 (ntp/ 或 rl/)
- 7 篇论文待读 (可在等待授权期间继续): 2511.11255, 2510.20455, 2510.16804, 2509.21777, 2503.02453, 2410.06682, 2405.16436
