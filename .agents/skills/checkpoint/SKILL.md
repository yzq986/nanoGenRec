---
name: checkpoint
description: Update ideas & experiment docs when a phase concludes — mark closed/deferred ideas with reasons, preserve all original detail, update priorities
argument-hint: [phase description, e.g. "tokenizer phase done, MLP-FSQ wins"]
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Glob, Grep
---

# /checkpoint Skill

When a research phase concludes (e.g. tokenizer selection done, NTP baseline established), update all ideas and experiment docs to reflect conclusions. **Core rule: preserve all original thinking, append why things changed.**

## Trigger

User runs `/checkpoint [phase description]` after a batch of experiments finishes and conclusions are drawn.

## Instructions

### Step 1: Gather context

1. **Read `experiments/logs/`** to identify all experiments related to this phase and their results.
2. **Read `ideas/README.md`** to get the global idea index and priority tables.
3. **Use Grep** to find all IDEA entries across `ideas/*.md` that are affected by the concluded phase.

### Step 2: Update each affected idea in its topic file

For each idea touched by the phase conclusions:

#### Closed ideas (experimentally disproven or superseded)

Update the header — **do NOT delete any original content**:

```markdown
## IDEA-{hash}-{N}: {Title}

**优先级**: ~~{old}~~ → ❌ 关闭
**来源**: {unchanged}
**状态**: ~~{old status}~~ → {experiment} 后关闭

> **关闭原因 ({date})**: {1-2 sentences explaining what experiment showed and why this idea is closed. Include key numbers.}
```

- Keep all original sections (核心思想, 实验设计草案, 关键问题, etc.) intact.
- The `> **关闭原因**` blockquote is **appended** right after the status line, before 核心思想.

#### Completed ideas (successfully validated and adopted)

```markdown
**优先级**: ~~{old}~~ → ✅ 完成
**状态**: ✅ {one-line summary of what was achieved} ({experiment IDs})

> **完成记录 ({date})**: {1-2 sentences on what was confirmed and adopted.}
```

#### Deferred ideas (still potentially valid but deprioritized)

```markdown
**优先级**: ~~{old}~~ → P2 ({when to revisit, e.g. "NTP 后"})
**状态**: 待定，降级

> **降级原因 ({date})**: {Why deprioritized now. What would trigger re-evaluation. Reference key experimental evidence.}
```

#### Active ideas unaffected by this phase

Leave unchanged.

### Step 3: Update the evolution tree

If the topic file has a `## 演进路径` section, update the tree to show experiment outcomes:

- Add `→ EXP-NNN ❌` or `→ EXP-NNN ✅` after explored branches
- Add result summaries as child nodes
- Add new branches for deferred ideas with `→ P2 (条件)`

### Step 4: Add/update a "当前结论" section

At the top of the affected topic file (after the intro, before 演进路径), add or update:

```markdown
## 当前结论 ({date})

**{One-line conclusion}**

### 当前 config
{Current best configuration in code block}

### 关键实验数据
{Summary table of decisive experiments with key metrics}

**核心 insight**: {The most important lesson learned}
```

### Step 5: Update the priority summary table

At the bottom of the topic file, update the `## 优先级总结` table:
- Closed ideas: strikethrough with status
- Completed ideas: strikethrough with ✅
- Deferred ideas: new priority with brief reason

### Step 6: Update `ideas/README.md`

1. **Update the file index table**: idea counts, P0 column
2. **Update the global priority tables** (P0/P1/P2):
   - Move closed/completed ideas to strikethrough with status
   - Move deferred ideas to the P2 section with `(条件)` suffix
3. **If a "演进记录" subsection exists** under design principles, append the new phase conclusions. If not, create one after the design principles section.
4. **Preserve all original design principles text** — only append, never rewrite original thinking.

### Step 7: Update `experiments/logs/`

For any experiments that were concluded as part of this phase but not yet marked completed:
- Update `**Status**` to `completed`
- Ensure Results, Analysis, and Next Steps sections are filled

## Anti-patterns (DO NOT)

- **DO NOT delete original idea content** (核心思想, 实验设计, 关键问题, etc.). These record the original thinking process.
- **DO NOT rewrite design principles or strategic thinking**. Append evolution, don't overwrite.
- **DO NOT change IDEA IDs or hash prefixes**. These are permanent provenance markers.
- **DO NOT collapse multiple ideas into one**. Each idea retains its own entry.
- **DO NOT remove entries from priority tables**. Use strikethrough for closed/completed items.

## Output

After all updates, produce a summary:

```
## Checkpoint: {phase description}

### 关闭 ({N})
- IDEA-xxx-N: {title} — {reason}

### 完成 ({N})
- IDEA-xxx-N: {title} — {what was adopted}

### 降级 ({N})
- IDEA-xxx-N: {title} — P2 ({condition})

### 不变 ({N})
- {list of unaffected ideas, if relevant}

### 下一阶段
{What the next research phase focuses on}
```
