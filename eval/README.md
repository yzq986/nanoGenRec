# eval/

Evaluation framework for tokenizer metrics, behavior-aware metrics, batch reports, and full-recall NTP evaluation.

Use this module to evaluate one model, compare multiple runs, or generate reports that can be linked from experiment logs.

## Files

| File | Purpose |
|------|---------|
| `wrapper.py` | Model wrappers used by metric evaluators. |
| `evaluator.py` | Metric registration and execution core. |
| `behavior.py` | Behavior-aware evaluation context and metrics. |
| `compare.py` | Markdown, JSON, and CSV comparison reports. |
| `batch.py` | Batch evaluation orchestration. |
| `hyperparam.py` | Hyperparameter search for tokenizer settings. |

## Usage

```bash
# Single-model evaluation
python run.py eval --results_path s3://... --model_path s3://...

# Batch evaluation
python run.py eval-all --models qwen3-0.6b qwen3-4b --quick

# Compare existing results
python run.py compare --eval_dir eval_results

# Tokenizer hyperparameter search
python run.py hyperparam --model qwen3-0.6b --skip_embedding

# Hyperparameter search with NTP enabled
python run.py hyperparam --model qwen3-0.6b --skip_embedding --run_ntp
```

Full NTP recall reports use the dedicated command:

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 run.py eval-ntp \
    --checkpoint experiments/ntp_checkpoints/<name> \
    --n_recall 1000
```

## Metric Flow

```text
wrapper.py
  -> evaluator.py
  -> behavior.py
  -> batch.py
  -> compare.py
```

## Reporting Rules

- Use full eval for headline Recall@K values.
- Treat inline training eval as a health check only.
- Store generated reports under `experiments/` when they are part of a reproducible run.
- Link conclusions from `experiments/logs/<phase>/exp-NNN.md`.
