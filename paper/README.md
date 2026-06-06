# nanoGenRec Paper Draft

This directory contains a first arXiv-style technical report draft for nanoGenRec.

## Files

| File | Purpose |
|------|---------|
| `nanogenrec.tex` | Main manuscript draft. |
| `references.bib` | Verified bibliography entries used by the draft. |
| `check_paper_consistency.py` | Repository-local consistency checker for author placeholders, citations, figures, and copied experiment numbers. |
| `figures/` | Figures copied from repository experiment artifacts for a self-contained arXiv source bundle. |
| `nanogenrec-arxiv-source.tar.gz` | Minimal arXiv source bundle: `tex`, `bib`, `bbl`, and figures. |
| `ARXIV_SUBMISSION_CHECKLIST.md` | Human verification checklist before any submission. |

## Build

From this directory:

```bash
pdflatex nanogenrec
bibtex nanogenrec
pdflatex nanogenrec
pdflatex nanogenrec
```

Run the repository consistency checks from the repository root:

```bash
python3 paper/check_paper_consistency.py
```

The paper is currently a technical-report draft. Before submission, confirm the author metadata and manually inspect the compiled PDF.
