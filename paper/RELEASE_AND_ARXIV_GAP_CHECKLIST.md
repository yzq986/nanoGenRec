# Release and arXiv Gap Checklist

Last updated: 2026-06-12

## Current Status

The repository is close to a public open-source release and arXiv technical
report. The strongest new release anchor is the public MovieLens 1M Colab-T4
result: 5,950 users, 3,532 items, 348,363 train examples, R@500=72.5%, and
R@1000=85.2%.

## Must Do Before Open-Source Promotion

| Item | Status | Notes |
|------|--------|-------|
| License | Added, human confirmation needed | Root `LICENSE` uses MIT. Human owner should confirm the license choice. |
| Citation metadata | Done | Added `CITATION.cff` with author, title, repository URL, license, and release date. |
| Privacy sweep | Done, high severity = 0 | Added `scripts/privacy_scan.py`; current scan reports no high-severity findings. |
| Public quickstart check | Done | Synthetic public benchmark and public benchmark test passed in a clean temporary workspace. |
| Public benchmark artifacts | Ready | MovieLens 1M Colab-T4 result is checked in and linked. |
| Dependency clarity | Done | README now includes `python -m pip install -r requirements.txt`; `PyYAML` and `pytest` are listed. |
| Badges / metadata | Done for current links | Added MIT, Colab, and Python badges; arXiv badge should wait for an arXiv ID. |

## Must Do Before arXiv Submission

| Item | Status | Notes |
|------|--------|-------|
| Author metadata | Needs human confirmation | Current author block: Ziqing Ye, `yeziqing986@gmail.com`. |
| Numeric consistency check | Passed | `python3 paper/check_paper_consistency.py` passes. |
| PDF compile | Passed with minor warnings | Full `pdflatex -> bibtex -> pdflatex -> pdflatex` completes; one small overfull remains. |
| Source bundle | Refreshed | `paper/nanogenrec-arxiv-source.tar.gz` was regenerated after the latest edits. |
| Manual PDF inspection | Needed | Check page breaks, table placement, and figure readability in `paper/nanogenrec.pdf`. |
| Citation audit | Needed | Verify all BibTeX titles, authors, years, and arXiv IDs manually. |
| AI-use policy | Needed | arXiv itself does not require a special statement, but any venue version may. |
| Claim-boundary review | Mostly ready | Paper now clearly says framework + landing record, not new algorithm. |

## What Would Make the Paper Stronger

| Priority | Improvement | Why it matters |
|----------|-------------|----------------|
| Done | Add simple public baselines on MovieLens 1M | Popularity, last-item repeat, and ItemKNN are now checked in; neural baselines remain future work. |
| Done | Add a diagram showing public and production paths sharing interfaces | Added a paper figure showing the shared framework interface. |
| Medium | Add one more public dataset, such as Amazon Reviews subset | Reduces concern that the public result is MovieLens-specific. |
| Partial | Document runtime and GPU memory from Colab output | Notebook now writes `runtime.json`; existing Colab run did not record elapsed time, so rerun is needed to fill actual numbers. |
| Medium | Add release tags and archived artifact links | Helps citation and reproducibility. |
| Low | Polish the LaTeX overfull/underfull warnings | Not blocking, but improves final presentation. |

## Current Local Review Score

The updated local Stanford Agentic Reviewer-style score is **7.5 / 10** in
`paper/reviews/stanford_agentic_reviewer_style_v2.md`.

Interpretation:

- Strong enough for an arXiv technical report.
- Plausible as a workshop artifact/framework paper.
- Still below a strong top-conference research paper because the contribution is
  framework landing rather than algorithmic novelty, and public baselines are
  not yet included.
