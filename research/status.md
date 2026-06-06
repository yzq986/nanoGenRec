---

[English](status.md) | [Chinese](status.zh.md)
last_updated: "2026-04-22T20:35:00"
current_task: null
next_experiment_number: 26
best_result:
  experiment: exp025-beam-passes
  ppl: 25.22
  recall_500: 0.636
total_experiments_run: 0
total_papers_read: 15
---

# Research Agent Status

## Current State

Idle. Awaiting human response on:
- outbox/002-experiment-analysis.md (Page-wise NTP proposal)
- outbox/005-spot-dpo-finding.md (IDEA-spot-0: SPoT-BCO for SP-DPO, requires rl/ source code authorization)

## Recent Activity

- [2026-04-22 20:25] Activated session 6 (/loop)
- [2026-04-22 20:27~20:31] Read 3 papers: Align3GR (2511.11255), TO-RoPE (2510.20455), LAC (2510.16804)
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
- **IDEA-genrec-0** (P1, training.md): Page-wise NTP — highest priority, requires human authorization ntp/ source code modification
- **IDEA-spot-0** (P1, rl-alignment.md): SPoT-BCO replaces SP-DPO loss — human authorization required rl/ source code modification
- IDEA-mtgr-0 (P0, training.md): User-Level Packing + Causal Mask — requires human authorization ntp/ source code modification
- IDEA-feat-3 (P2): Item Category Token — lower priority
- IDEA-feat-4 (P2): User Profile Prefix — lower priority

Completed/Closed:
- ✅ IDEA-feat-0/1/2: time_gap + action + segment_emb — confirmed via EXP-023/025
- ❌ IDEA-onemall-0: contrastive loss — tested, negative, closed via EXP-022

## Open Questions

1. [outbox/002] Does the Page-wise NTP experiment authorize modification of ntp/preprocess.py and ntp/train.py?
2. [outbox/002] Session splitting standard preference? (Time interval >30min vs by day)
3. [outbox/005] Is IDEA-spot-0 (SPoT-BCO for SP-DPO) authorized to modify the rl/ source code?

## Blockers

- All high-priority experiments require source code modification authorization (ntp/ or rl/)
- 4 papers to be read (can be continued while waiting for authorization): 2509.21777, 2503.02453, 2410.06682, 2405.16436
