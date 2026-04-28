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

### Timing Convention

All Results tables include a `训练耗时` column (training wall time) sourced from `train_meta.json`
`train.wall_time_s`. This excludes full eval time (~25min per run on 8×A100 with n_recall=1000).

Quick reference for planning new experiments:

| Experiment Type | Typical Train Time |
|----------------|-------------------|
| SFT S-tier (17.5M, 1 epoch, 14d data) | ~21min |
| RF-DPO Hard (807 steps) | ~62min |
| GRPO/ECPO (G=512, rl_ratio=1.0) | ~80min |
| Scaling M+ (101M, 1 epoch, 14d) | ~207min |

New experiment scripts must instrument each phase with `$(date +%s)` timestamps and print
`train=${TRAIN_MIN}min eval=${EVAL_MIN}min total=${TOTAL_MIN}min` at the end. See
`.claude/skills/experiment/SKILL.md` for the required template.
