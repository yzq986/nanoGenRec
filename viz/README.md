# viz/

[English](README.md) | [Chinese](README.zh.md)

Post-training visualization utilities for NTP and alignment experiments.

The tools read `train_meta.json` and `train_log.jsonl` from `experiments/ntp_checkpoints/` and produce lightweight plots for experiment review.

## Tools

| Tool | Purpose | Input |
|------|---------|-------|
| `training_dynamics` | Loss, gradient norm, and training curves. | `train_log.jsonl` |
| `pareto` | PPL vs Recall scatter plot with Pareto frontier. | `train_meta.json` |
| `degradation` | Metric delta table relative to a baseline. | `train_meta.json` |
| `pipeline` | Stage-by-stage waterfall from NTP to DPO/RL. | `train_meta.json` |

## Usage

```bash
# Compare training dynamics
python -m viz.training_dynamics exp019-*

# PPL vs Recall Pareto plot
python -m viz.pareto exp019-* exp018-* exp017-fixed-medium \
    --baseline exp017-fixed-medium \
    --save pareto.png

# Degradation table
python -m viz.degradation exp019-* exp018-* \
    --baseline exp017-fixed-medium

# Pipeline waterfall
python -m viz.pipeline \
    --ntp exp016-B-14d-S \
    --spdpo exp017-fixed-medium \
    --rfdpo exp019-* \
    --save pipeline.png
```

## Data Source

```text
experiments/ntp_checkpoints/<exp_name>/
├── train_meta.json
├── train_log.jsonl
└── probe.pt
```

Generated `output_*.png` files are ignored by git. The plotting dependencies are `matplotlib` and `numpy`.
