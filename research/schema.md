# Research Shared Space — Schema

[English](schema.md) | [Chinese](schema.zh.md)

This document defines formatting conventions for all files in the `research/` directory. Agents and humans alike abide by these conventions.

## File naming

All files in `inbox/`, `outbox/`, `decisions/` use:
- Format: `NNN-kebab-case-topic.md`
- Numbering **Globally incremented** (across all directories), no duplicates
- Example: `001-initial-directive.md`, `002-contrastive-loss-proposal.md`

`paper-notes/` uses arxiv ID: `ARXIV_ID.md` (e.g. `2604.14878.md`)

## inbox/ — person → Agent

Humans write, agents read.

```yaml
---
from: human
date: "2026-04-22"
priority: normal  # normal | urgent
subject: "short title"
---

Text (markdown).
```

After the Agent finishes reading, add `read: "YYYY-MM-DD HH:MM"` to the frontmatter.

## outbox/ — Agent → Person

Agents write, humans read.

```yaml
---
date: "2026-04-22 14:30"
type: question     # question | finding | proposal | error
priority: normal   # normal | urgent
subject: "short title"
needs_response: true  # true | false
---

Text (markdown).
```

Humans can respond by:
1. Directly append the `## Response` paragraph to this file
2. Or write a new inbox message referencing this outbox number

## decisions/ — Experimental decision record

```yaml
---
experiment: "EXP-026"
date: "2026-04-22"
decision: merge    # merge | discard
confidence: high   # high | medium | low
---

## Summary
Describe the experiment and hypothesis in one sentence.

## Results
| Config | PPL | R@10 | R@500 | vs Baseline |
|--------|-----|------|-------|-------------|

## Rationale
Why merge/discard (2-3 sentences).

## Next Steps
Next steps based on this result.
```

## status.md — status panel

YAML frontmatter (machine readable) + human readable narrative.

```yaml
---
last_updated: "2026-04-22T14:30:00"
current_task: null # null or {type, experiment, phase, started_at}
next_experiment_number: 26
best_result:
  experiment: "exp023-segment"
  ppl: 25.94
  recall_500: 0.612
total_experiments_run: 0
---
```

Follow with human-readable paragraphs: Current State, Recent Activity, Experiment Queue, Open Questions, Blockers.

## log.md — Timeline

Append-only. Each record:

```markdown
## [YYYY-MM-DD HH:MM] ACTION_TYPE: short description

Details (optional).
```

ACTION_TYPE enumeration:
- `STARTUP` — Agent startup
- `INBOX_READ` — read human messages
- `PAPER_READ` — finish reading a paper
- `IDEA_PROPOSED` — Propose new ideas
- `EXPERIMENT_DESIGN` — Design an experiment
- `EXPERIMENT_RUN` — start executing an experiment
- `EXPERIMENT_EVAL` — evaluate experimental results
- `DECISION` — merge/discard decision
- `OUTBOX_WRITE` — send a message to a human
- `TOOL_CREATED` — create new analysis/estimation tools
- `ERROR` — error
- `IDLE` — no task to execute

## paper-notes/ — paper abstract

```markdown
# [paper title]

- **Arxiv**: XXXX.XXXXX
- **Authors**: ...
- **Date**: YYYY-MM
- **Read date**: YYYY-MM-DD

## Core Contribution
- (3 points)

## Method
(1 paragraph summary)

## Key Results
(Key Figures/Table)

## Relevance to nanoGenRec
(Specific applicable points)

## Connections
- → ideas/training.md: IDEA-xxx-N (associated with idea)
- → ideas/architecture.md: IDEA-yyy-N
```
