# Release Automation Report

Date: 2026-06-12

This report records release-readiness tasks completed without requiring human
judgment.

## Completed Automatically

| Task | Result |
|------|--------|
| License metadata | Added root `LICENSE` using MIT License. |
| Citation metadata | Added `CITATION.cff` for GitHub citation support. |
| README metadata | Added MIT, Colab, and Python badges to English and Chinese READMEs. |
| Dependency clarity | Added `python -m pip install -r requirements.txt` quickstart step and included `PyYAML` / `pytest` in `requirements.txt`. |
| Internal remote cleanup | Replaced company/internal remotes in `AGENTS.md` and `AGENTS.md.zh.md` with the public GitHub remote. |
| Privacy scanner | Added `scripts/privacy_scan.py`. |
| Privacy scan | Passed: no high-severity privacy findings. |
| Fresh workspace quickstart | Passed from `/tmp/nanogenrec-release-check`. |
| Public baselines | Added MovieLens 1M popularity, last-item repeat, and ItemKNN baselines. |
| Shared-interface diagram | Added to the paper to show the public and production paths meeting at the same framework interfaces. |
| Colab runtime instrumentation | Notebook now writes `runtime.json` with elapsed seconds, GPU name, total memory, used memory, and command. |

## Verification Commands

```bash
python3 scripts/privacy_scan.py
pytest -q tests/test_public_movielens.py
python3 run.py public-movielens \
    --dataset synthetic \
    --output_dir /tmp/nanogenrec-release-check-smoke \
    --epochs 1 \
    --max_users 120 \
    --clusters 8,8,8 \
    --embed_dim 32 \
    --layers 1 \
    --eval_samples 10 \
    --beam_size 10 \
    --feature_source hybrid \
    --train_mode sliding \
    --min_context_items 2 \
    --device cpu
python3 paper/check_paper_consistency.py
pdflatex -interaction=nonstopmode nanogenrec.tex
python3 public_benchmarks/baselines.py --dataset ml-1m --output public_benchmarks/results/ml-1m-baselines.json
```

## Fresh Workspace Result

The release check used a clean temporary workspace copied from the working tree
without `.git`, pytest cache, public benchmark data, or generated runs:

```text
/tmp/nanogenrec-release-check
```

The synthetic public benchmark completed with:

```text
n_users=120
n_items=120
n_train_examples=1224
n_eval_examples=120
item_recall@50=0.2
target_sid_found_rate=0.2
```

## Remaining Human-Gated Items

- Confirm the license choice is acceptable.
- Confirm author metadata and email.
- Manually inspect the compiled PDF.
- Manually audit BibTeX citation correctness.
- Decide whether to add public baselines before arXiv submission.
