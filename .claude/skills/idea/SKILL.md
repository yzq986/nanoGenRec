---
name: idea
description: Extract experiment ideas from papers/articles discussed in conversation, write structured idea entries to ideas/ folder
argument-hint: [topic title]
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Glob, Grep
---

# /idea Skill

Extract actionable experiment ideas from papers or technical articles discussed in the conversation, and write structured entries to the `ideas/` folder.

## Instructions

1. **Read existing ideas**: Use Glob to find all `ideas/*.md` files, then Read `ideas/README.md` to find existing entries.

2. **Determine filename and hash prefix**:
   - Use the argument as the topic if provided, otherwise infer from discussion
   - Convert to kebab-case for filename: `ideas/{topic}.md`
   - Derive a short hash prefix for IDEA IDs from the source (e.g., arxiv ID `2601.21770` → `onemall`, paper name → short mnemonic). The hash must be unique across all idea files.
   - If the file already exists, append new ideas (increment the per-file sequence number)
   - If the file is new, create it and add an entry to `ideas/README.md`

3. **Extract ideas from conversation context**. For each distinct idea, create an entry with:
   - **IDEA-{hash}-{N}**: where `{hash}` is the file's unique prefix and `{N}` is a 0-based sequence number within that file (e.g., `IDEA-onemall-0`, `IDEA-onemall-1`). This avoids ID collisions across files.
   - **优先级**: P0 (critical/strategic) / P1 (high value) / P2 (nice to have)
   - **来源**: Which section/paper the idea comes from
   - **状态**: 待讨论 (initial) / 已采纳 → EXP-NNN (when promoted to experiment) / 已否决 (rejected)

4. **Each idea entry must include these sections**:

   ```markdown
   ## IDEA-{hash}-{N}: {Title}

   **优先级**: P0/P1/P2
   **来源**: {paper/section reference}
   **状态**: 待讨论

   ### 核心思想
   {What the paper proposes, in 2-3 sentences}

   ### 与当前项目的关联
   {How it connects to our codebase, existing experiments, architecture decisions}

   ### 实验设计草案
   {Concrete experiment design: variables, configs, baselines, metrics}

   ### 关键问题
   {Open questions, risks, dependencies that need to be resolved before implementation}
   ```

5. **At the end, output a priority summary table** in the idea file:

   ```markdown
   ## 优先级总结

   | 优先级 | ID | 实验 | 原因 |
   |--------|-----|------|------|
   | P0 | IDEA-{hash}-0 | ... | ... |
   ```

   If appending to an existing file, update the existing summary table.

## File Format

```markdown
# {Topic Title}

**来源**: {paper/article reference}
**日期**: {YYYY-MM-DD}

---

## IDEA-{hash}-{N}: {Title}
...

---

## IDEA-{hash}-{N}+1: {Title}
...

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| ... | ... | ... | ... |
```

## Guidelines

- **Be concrete**: Ideas must have enough detail to become an experiment. Vague "we could try X" is not enough.
- **Connect to project**: Every idea must reference specific files, configs, or experiment results in our codebase.
- **Assess feasibility**: Note implementation cost (existing FAISS support? need new code? need new data?) and dependencies.
- **Prioritize ruthlessly**: P0 = addresses a known architectural limitation or strategic goal. P1 = clear value but not blocking. P2 = interesting but has prerequisites.
- **Don't duplicate**: Check existing ideas before creating new ones. If a new paper reinforces an existing idea, update the existing entry with new evidence.
