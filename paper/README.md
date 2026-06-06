# nanoGenRec Paper Draft

This directory contains a first arXiv-style technical report draft for nanoGenRec.

## Files

| File | Purpose |
|------|---------|
| `nanogenrec.tex` | Main manuscript draft. |
| `references.bib` | Verified bibliography entries used by the draft. |
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

The paper is currently a technical-report draft. Before submission, replace the author placeholder and manually verify every table value against the cited experiment logs.
