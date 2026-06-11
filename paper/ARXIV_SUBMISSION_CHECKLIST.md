# arXiv Submission Checklist

This checklist is intentionally conservative. It is meant to avoid careless AI-assisted manuscript issues and to keep the paper framed as a framework + empirical-study report rather than an unsupported new-algorithm claim.

## Required Human Verification

- [ ] Confirm the author metadata in `nanogenrec.tex`.
- [ ] Run `python3 paper/check_paper_consistency.py` from the repository root.
- [ ] Verify every numeric value against the source experiment log:
  - [ ] EXP-015 model-scaling table and fitted law.
  - [ ] EXP-016 data-scaling table.
  - [ ] EXP-029 ECPO off-policy/on-policy comparison.
  - [ ] EXP-043 full-eval baselines.
  - [ ] EXP-044C/EXP-047 full-eval rows referenced through `experiments/logs/ntp/README.md`.
  - [ ] EXP-049 tokenizer sweep table.
  - [ ] MovieLens 1M Colab-T4 public result.
- [ ] Verify every citation in `references.bib` exists and has the correct title, arXiv ID, and author spelling.
- [ ] Compile the manuscript locally and inspect the PDF manually.
- [ ] Confirm that the paper does not claim a new algorithmic contribution.
- [ ] Confirm that private data, company identifiers, and deployment-specific details are not disclosed.
- [ ] Confirm that the limitations section clearly states that production-scale results cannot be exactly reproduced without the private data.

## AI-Assisted Writing Hygiene

- [ ] Remove all drafting meta-comments, prompts, or conversational text.
- [ ] Do not include unverified references, invented baselines, invented venues, or invented author names.
- [ ] Do not submit until all listed authors have read the full manuscript and accept responsibility for its contents.
- [ ] If submitting to a venue with an AI-use disclosure policy, add the required disclosure in the acknowledgments or submission form according to that venue's instructions.

## Suggested Positioning

Use this positioning:

> nanoGenRec is an open-source reproducibility framework and production-scale empirical study for Semantic-ID Generative Recommendation.

Avoid this positioning:

> nanoGenRec proposes a new Generative Recommendation algorithm.
