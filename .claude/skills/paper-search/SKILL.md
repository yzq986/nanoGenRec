---
name: paper-search
description: Search arxiv for industrial generative recommendation papers, extract ideas and file them into ideas/ topic files
argument-hint: [search query or company name]
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Glob, Grep, WebSearch, WebFetch
---

# /paper-search Skill

Search arxiv for industrial-scale generative recommendation papers, evaluate them, and extract actionable ideas using the `/idea` workflow.

## Instructions

### Step 1: Search arxiv

Use WebSearch to find recent papers on generative recommendation from major industrial labs:

**Target companies**: Kuaishou, Meta, Alibaba, ByteDance, Tencent, JD, Google, Amazon, Microsoft, Baidu

**Search queries** (adapt based on argument):
- `arxiv generative recommendation {company} 2025 2026`
- `arxiv semantic ID recommendation industrial deployment`
- `arxiv generative retrieval large-scale production`

If a specific argument is given (e.g., "ByteDance"), focus the search on that company/topic.

### Step 2: Filter papers

For each paper found, evaluate against these criteria. Papers must meet **at least 2 of 3**:

1. **Online A/B results**: Paper reports real online metrics (CTR, revenue, GMV lift), not just offline
2. **Deployment scale**: Mentions serving scale (QPS, user base, item pool size > 1M)
3. **Novel technique**: Introduces a technique not already covered by existing ideas in `ideas/*.md`

Use WebFetch on the arxiv abstract page to read the paper summary. Skip papers that don't meet criteria.

### Step 3: Extract ideas

For each qualifying paper:

1. Read the paper abstract and key sections via WebFetch
2. Identify distinct, actionable ideas relevant to our generative recommendation project
3. **Determine which improvement dimension** each idea belongs to:
   - Tokenizer (quantization methods)
   - Embedding (representation enhancement)
   - Architecture (model design)
   - Training (loss functions, training strategies)
   - RL Alignment (reinforcement learning)
   - Inference (decoding optimization)
   - Scaling (scaling laws)

4. **Check for duplicates**: Grep `ideas/*.md` for similar ideas. If a new paper reinforces an existing idea, update the existing entry with new evidence instead of creating a duplicate.

5. **File each idea** following the `/idea` skill format:
   - Derive a hash prefix from the paper (e.g., arxiv ID or short name)
   - Assign globally-unique N per hash prefix (grep all `ideas/*.md` files first)
   - Append to the appropriate topic file
   - Update `ideas/README.md` index and priority tables

### Step 4: Output summary

After processing all papers, output a summary:

```markdown
## Paper Search Summary

**Query**: {search description}
**Date**: {YYYY-MM-DD}
**Papers found**: N total, M qualifying

### Processed Papers

| Paper | Company | Venue | Key Ideas | Filed To |
|-------|---------|-------|-----------|----------|
| {title} | {company} | {venue} | IDEA-{hash}-{N}, ... | tokenizer.md, training.md |
| ... | ... | ... | ... | ... |

### Skipped Papers (didn't meet criteria)

| Paper | Reason |
|-------|--------|
| {title} | No online results / Already covered by IDEA-xxx-N |
| ... | ... |

### New Ideas Added: {count}
```

## Guidelines

- **Industrial focus**: Skip pure academic papers without deployment evidence. We want techniques proven at scale.
- **Actionable ideas only**: Each idea must be concrete enough to design an experiment. "Interesting approach" is not enough.
- **Cross-reference existing work**: Always check existing ideas before filing. The value is in discovering NEW techniques, not rediscovering known ones.
- **Respect ID conventions**: Hash prefix = paper provenance, N = globally unique per prefix, filed by improvement dimension.
- **Update README.md**: After adding ideas, update the file index table (idea counts) and priority tables in `ideas/README.md`.
- **Rate limit**: Don't fetch more than 10 paper pages per search session to avoid being blocked.
