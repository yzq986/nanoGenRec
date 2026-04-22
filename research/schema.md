# Research Shared Space — Schema

本文档定义 `research/` 目录下所有文件的格式约定。Agent 和人类都遵守这些约定。

## 文件命名

所有 `inbox/`、`outbox/`、`decisions/` 中的文件使用：
- 格式：`NNN-kebab-case-topic.md`
- 编号**全局递增**（跨所有目录），不重复
- 例：`001-initial-directive.md`、`002-contrastive-loss-proposal.md`

`paper-notes/` 使用 arxiv ID：`ARXIV_ID.md`（如 `2604.14878.md`）

## inbox/ — 人 → Agent

人类写入，Agent 读取。

```yaml
---
from: human
date: "2026-04-22"
priority: normal  # normal | urgent
subject: "简短标题"
---

正文（markdown）。
```

Agent 读完后在 frontmatter 添加 `read: "YYYY-MM-DD HH:MM"`。

## outbox/ — Agent → 人

Agent 写入，人类读取。

```yaml
---
date: "2026-04-22 14:30"
type: question     # question | finding | proposal | error
priority: normal   # normal | urgent
subject: "简短标题"
needs_response: true  # true | false
---

正文（markdown）。
```

人类回复时可以：
1. 直接在此文件追加 `## Response` 段落
2. 或写一条新的 inbox 消息引用此 outbox 编号

## decisions/ — 实验决策记录

```yaml
---
experiment: "EXP-026"
date: "2026-04-22"
decision: merge    # merge | discard
confidence: high   # high | medium | low
---

## Summary
一句话描述实验和假设。

## Results
| Config | PPL | R@10 | R@500 | vs Baseline |
|--------|-----|------|-------|-------------|

## Rationale
为什么 merge/discard（2-3 句）。

## Next Steps
基于此结果的下一步方向。
```

## status.md — 状态面板

YAML frontmatter（机器可读）+ 人类可读叙述。

```yaml
---
last_updated: "2026-04-22T14:30:00"
current_task: null  # null 或 {type, experiment, phase, started_at}
next_experiment_number: 26
best_result:
  experiment: "exp023-segment"
  ppl: 25.94
  recall_500: 0.612
total_experiments_run: 0
---
```

后续跟人类可读的段落：Current State、Recent Activity、Experiment Queue、Open Questions、Blockers。

## log.md — 时间线

Append-only。每条记录：

```markdown
## [YYYY-MM-DD HH:MM] ACTION_TYPE: 简短描述

详细内容（可选）。
```

ACTION_TYPE 枚举：
- `STARTUP` — Agent 启动
- `INBOX_READ` — 读取人类消息
- `PAPER_READ` — 读完一篇论文
- `IDEA_PROPOSED` — 提出新 idea
- `EXPERIMENT_DESIGN` — 设计实验
- `EXPERIMENT_RUN` — 开始执行实验
- `EXPERIMENT_EVAL` — 评估实验结果
- `DECISION` — merge/discard 决策
- `OUTBOX_WRITE` — 向人类发消息
- `TOOL_CREATED` — 创建新的分析/估算工具
- `ERROR` — 错误
- `IDLE` — 无可执行任务

## paper-notes/ — 论文摘要

```markdown
# [论文标题]

- **Arxiv**: XXXX.XXXXX
- **Authors**: ...
- **Date**: YYYY-MM
- **Read date**: YYYY-MM-DD

## Core Contribution
- （3 条要点）

## Method
（1 段摘要）

## Key Results
（关键数字/表格）

## Relevance to gr_demo
（具体可应用的点）

## Connections
- → ideas/training.md: IDEA-xxx-N（关联已有 idea）
- → ideas/architecture.md: IDEA-yyy-N
```
