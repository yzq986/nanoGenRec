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
| License | Missing | Add a root `LICENSE`; MIT or Apache-2.0 are the likely choices. |
| Citation metadata | Missing | Add `CITATION.cff` with author, title, repository URL, and version/date. |
| Privacy sweep | Needed | Re-scan for private hostnames, company identifiers, access tokens, and local paths. |
| Public quickstart check | Mostly ready | README has CPU and Colab paths; run once from a fresh clone if possible. |
| Public benchmark artifacts | Ready | MovieLens 1M Colab-T4 result is checked in and linked. |
| Dependency clarity | Partial | `requirements.txt` exists, but Colab relies mostly on preinstalled PyTorch. |
| Badges / metadata | Optional | Add license, arXiv, Colab, and Python version badges after final links exist. |

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
| High | Add public baselines on MovieLens 1M | Reviewers will ask how R@500 compares with SASRec/BERT4Rec/GRU4Rec-style baselines. |
| High | Add a diagram showing public and production paths sharing interfaces | Makes the framework contribution obvious in one glance. |
| Medium | Add one more public dataset, such as Amazon Reviews subset | Reduces concern that the public result is MovieLens-specific. |
| Medium | Document runtime and GPU memory from Colab output | Makes the free-GPU claim more operationally useful. |
| Medium | Add release tags and archived artifact links | Helps citation and reproducibility. |
| Low | Polish the LaTeX overfull/underfull warnings | Not blocking, but improves final presentation. |

## Current Local Review Score

The updated local Stanford Agentic Reviewer-style score is **7.3 / 10** in
`paper/reviews/stanford_agentic_reviewer_style_v1.md`.

Interpretation:

- Strong enough for an arXiv technical report.
- Plausible as a workshop artifact/framework paper.
- Still below a strong top-conference research paper because the contribution is
  framework landing rather than algorithmic novelty, and public baselines are
  not yet included.
