---

[English](SKILL.md) | [Chinese](SKILL.zh.md)
name: idea
description: Extract experiment ideas from papers/articles discussed in conversation, file them by improvement dimension into ideas/ topic files
argument-hint: [topic title]
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Glob, Grep
---

# /idea Skill

Extract actionable experiment ideas from papers or technical articles discussed in the conversation, and file them into the appropriate **topic file** under `ideas/`.

## Topic Files (improvement dimensions)

Ideas are organized by improvement dimension, NOT by paper source:

| File | Dimension | When to use |
|------|-----------|-------------|
| `ideas/tokenizer.md` | 量化Method | RQ/OPQ/FSQ/Balanced KMeans, collision, codebook utilization |
| `ideas/embedding.md` | 表征增强 | Collaborative signals, multimodal, attribute enrichment |
| `ideas/architecture.md` | Model架构 | Decoder design, attention, MoE, sequence compression |
| `ideas/training.md` | Training目标 | Auxiliary losses, sample weighting, multi-behavior |
| `ideas/rl-alignment.md` | RL 对齐 | DPO/GRPO/RSPO, reward design |
| `ideas/inference.md` | 推理优化 | Beam search, decoding strategies |
| `ideas/scaling.md` | 扩展性 | Scaling laws, model size vs data size |

## Instructions

1. **Scan existing ideas**: Use Glob to find all `ideas/*.md` topic files, then Read `ideas/README.md` to see the full index and existing IDs.

2. **Determine hash prefix and sequence number**:
   - Derive a short hash prefix from the paper source (e.g., arxiv `2601.21770` → `onemall`, paper name → short mnemonic). The hash identifies **provenance** (which paper), NOT which file.
   - The sequence number N is **globally unique per hash prefix** across ALL topic files (not per-file). Use Grep to find all existing `IDEA-{hash}-` entries across `ideas/*.md` and pick the next N.
   - Example: if `IDEA-onemall-0` through `IDEA-onemall-5` exist across various topic files, the next OneMall idea is `IDEA-onemall-6` regardless of which topic file it goes into.

3. **Choose the target topic file** based on which improvement dimension the idea primarily belongs to. If an idea spans multiple dimensions, file it under its primary dimension and cross-reference the others.

4. **Extract ideas from conversation context**. For each distinct idea, create an entry with:
   - **IDEA-{hash}-{N}**: where `{hash}` is the paper-origin prefix and `{N}` is the globally-incremented sequence number
   - **Priority**: P0 (critical/strategic) / P1 (high value) / P2 (nice to have)
   - **Source**: Which section/paper the idea comes from
   - **Status**: To be discussed (initial) / Adopted → EXP-NNN (when promoted to experiment) / Rejected (rejected)

5. **Each idea entry must include these sections**:

   ```markdown
   ## IDEA-{hash}-{N}: {Title}

**Priority**: P0/P1/P2
   **Source**: {paper/section reference}
   **Status**: To be discussed

### Core Idea
   {What the paper proposes, in 2-3 sentences}

### Association with the current project
   {How it connects to our codebase, existing experiments, architecture decisions}

### Experimental Design Draft
   {Concrete experiment design: variables, configs, baselines, metrics}

### Key questions
   {Open questions, risks, dependencies that need to be resolved before implementation}
   ```

6. **Update the topic file's priority summary table** at the bottom. If adding to an existing file, update the existing table.

7. **Update `ideas/README.md`**: Update the idea count in the file index table and add the new idea to the global priority table.

## File Format (topic file)

```markdown
# {Dimension Title}
{Dimension description + scope of influence}
---
## Evolution path
{Text or ASCII diagram showing evolution within this dimension}
---
## IDEA-{hash}-{N}: {Title}
...
---
## Priority summary
| 优先级 | ID | Experiment | 原因 |
|--------|-----|------|------|
| ... | ... | ... | ... |
```

## Guidelines

- **File by dimension, not by paper**: A paper may contribute ideas to multiple topic files. Each idea goes where it thematically belongs.
- **Preserve hash prefix as provenance**: The `{hash}` part of the ID traces back to the source paper. Never change it.
- **Global N per hash**: `IDEA-onemall-6` means the 7th idea from OneMall, regardless of which topic file contains it. Always grep all files before assigning N.
- **Be concrete**: Ideas must have enough detail to become an experiment. Vague "we could try X" is not enough.
- **Connect to project**: Every idea must reference specific files, configs, or experiment results in our codebase.
- **Assess feasibility**: Note implementation cost (existing FAISS support? need new code? need new data?) and dependencies.
- **Prioritize ruthlessly**: P0 = addresses a known architectural limitation or strategic goal. P1 = clear value but not blocking. P2 = interesting but has prerequisites.
- **Don't duplicate**: Check existing ideas before creating new ones. If a new paper reinforces an existing idea, update the existing entry with new evidence.
