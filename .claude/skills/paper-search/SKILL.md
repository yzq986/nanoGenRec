---
name: paper-search
description: Search arxiv for industrial generative recommendation papers, download PDFs, extract text, and file ideas into ideas/ topic files
argument-hint: [search query or company name]
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Glob, Grep, Bash, WebSearch, WebFetch
---

# /paper-search Skill

Search arxiv for industrial-scale generative recommendation papers, **download PDFs and convert to text**, evaluate them, and extract actionable ideas using the `/idea` workflow.

## Instructions

### Step 0: Ensure dependencies

Run once at the start of a session:

```bash
pip install pymupdf 2>/dev/null
```

This is needed for PDF → text conversion.

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

### Step 3: Download PDF and convert to text

For each qualifying paper:

1. **Create the papers directory** if it doesn't exist:
   ```bash
   mkdir -p papers
   ```

2. **Download the PDF** from arxiv. Convert the arxiv abstract URL to a PDF URL:
   - `https://arxiv.org/abs/XXXX.XXXXX` → `https://arxiv.org/pdf/XXXX.XXXXX`
   - Use curl to download:
   ```bash
   curl -L -o papers/{arxiv_id}.pdf "https://arxiv.org/pdf/{arxiv_id}"
   ```

3. **Convert PDF to plain text** using PyMuPDF for full-text extraction:
   ```bash
   python3 -c "
   import fitz
   doc = fitz.open('papers/{arxiv_id}.pdf')
   text = '\n\n'.join(page.get_text() for page in doc)
   with open('papers/{arxiv_id}.txt', 'w') as f:
       f.write(text)
   print(f'Converted {len(doc)} pages, {len(text)} chars')
   "
   ```

4. **Verify** the text file was created and has reasonable content:
   ```bash
   wc -l papers/{arxiv_id}.txt
   ```

5. **Read the full text** using the Read tool on `papers/{arxiv_id}.txt` to understand the paper in depth. This gives much richer context than just the abstract.

### Step 4: Update papers index

After downloading, update `papers/README.md` (create if it doesn't exist) with a table of all downloaded papers:

```markdown
# Downloaded Papers

| Arxiv ID | Title | Authors | Date | PDF | Text | Ideas |
|----------|-------|---------|------|-----|------|-------|
| {id} | {title} | {first author} et al. | {date} | [pdf](/{id}.pdf) | [txt](/{id}.txt) | IDEA-{hash}-N, ... |
```

### Step 5: Extract ideas

For each qualifying paper, **read from the full text** (`papers/{arxiv_id}.txt`) instead of relying on WebFetch:

1. Read the paper's full text via `Read` tool on `papers/{arxiv_id}.txt`
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

### Step 6: Output summary

After processing all papers, output a summary:

```markdown
## Paper Search Summary

**Query**: {search description}
**Date**: {YYYY-MM-DD}
**Papers found**: N total, M qualifying

### Downloaded Papers

| Arxiv ID | Title | PDF | Text | Pages | Chars |
|----------|-------|-----|------|-------|-------|
| {id} | {title} | papers/{id}.pdf | papers/{id}.txt | {pages} | {chars} |

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

- **Always download PDFs**: Every qualifying paper must be downloaded to `papers/` as both `.pdf` and `.txt`. This enables future deep-dive discussions without re-fetching.
- **Read full text, not just abstracts**: Use the converted `.txt` file (via Read tool) for idea extraction. Full text provides experimental details, ablation results, and implementation specifics that abstracts miss.
- **Industrial focus**: Skip pure academic papers without deployment evidence. We want techniques proven at scale.
- **Actionable ideas only**: Each idea must be concrete enough to design an experiment. "Interesting approach" is not enough.
- **Cross-reference existing work**: Always check existing ideas before filing. The value is in discovering NEW techniques, not rediscovering known ones.
- **Respect ID conventions**: Hash prefix = paper provenance, N = globally unique per prefix, filed by improvement dimension.
- **Update README.md**: After adding ideas, update the file index table (idea counts) and priority tables in `ideas/README.md`.
- **Rate limit**: Don't fetch more than 10 paper pages per search session to avoid being blocked.
- **Git-ignore PDFs** (optional): If PDFs bloat the repo, add `papers/*.pdf` to `.gitignore` and only commit the `.txt` files.
