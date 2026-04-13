# experiments/

Structured experiment results directory. All results are committed to git for reproducibility.

## Directory Structure

```
experiments/
├── README.md                              # This file
├── log.md                                 # Experiment log (reverse chronological)
├── hyperparam/
│   └── {date}_{name}/
│       ├── results.json                   # Raw grid search data
│       └── report.md                      # Auto-generated report
└── eval/
    └── {date}_{model}/
        ├── report.json
        ├── report.md
        └── report.csv
```

## Usage

### Hyperparameter Search

```bash
python run.py hyperparam --skip_embedding --clusters 256 512 1024 --only-sid --name cluster-sweep
# Output: experiments/hyperparam/2026-04-13_cluster-sweep/
```

### Batch Evaluation

```bash
python run.py eval-all --models qwen3-0.6b --only-sid
# Output: experiments/eval/2026-04-13_qwen3-0.6b/
```

## Experiment Log

See [log.md](./log.md) for the structured experiment log following the scientific method:
Background → Hypothesis → Design → Results → Analysis → Next Steps
