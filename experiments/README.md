# experiments/

Experiment orchestration, configs, queues, checkpoints, and generated result artifacts.

This directory is the operational home for reproducible runs. Use it to define experiments, check for duplicate baselines, run variants, and store artifacts that are needed to reproduce or inspect a result. Narrative conclusions live in [experiments/logs/](logs/).

## Layout

| Path | Purpose |
|------|---------|
| `configs/` | YAML experiment definitions. New experiments should start here. |
| `configs/_base.yaml` | Shared defaults. Read this before writing a new config. |
| `run_exp.py` | Main experiment runner with config expansion and duplicate-run checks. |
| `scripts/run_config.sh` | Queue-friendly wrapper around `run_exp.py --no-smoke --commit`. |
| `queue.txt` | Append-only queue for long-running experiments. |
| `queue_state.json` | Current queue daemon state. |
| `sid_cache/` | Semantic ID cache artifacts. |
| `ntp_data/` | NTP preprocessing shards. |
| `ntp_checkpoints/` | Training outputs, `train_meta.json`, and `train_log.jsonl`. |
| `logs/` | Human-readable experiment records and phase summaries. |

## Standard Workflow

```bash
# 1. Inspect defaults
sed -n '1,220p' experiments/configs/_base.yaml

# 2. Check whether a similar run already exists
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --check

# 3. Run all variants
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --no-smoke --commit

# 4. Resume or run one variant
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --only expNNN-a --no-smoke
```

Queue a run for asynchronous execution:

```bash
echo "run_config.sh experiments/configs/exp-NNN.yaml  /tmp/expNNN.log  exp-NNN complete!" >> experiments/queue.txt
```

## Config Guidelines

- Start from `configs/_base.yaml`; only override fields that are part of the experiment.
- Always verify `sid_cache_name`, `ntp_data_name`, and date ranges from existing configs before writing a new one.
- Use `variants:` for controlled comparisons.
- Do not rerun an identical baseline; reference the existing experiment in the log.
- If an experiment only changes evaluation code, re-evaluate an existing checkpoint instead of retraining.

## Timing Reference

Training wall time is recorded in `train_meta.json` as `train.wall_time_s`. Full eval is separate and is usually around 25 minutes on 8 GPUs with `n_recall=1000`.

| Experiment Type | Typical Train Time |
|-----------------|--------------------|
| SFT S-tier, 17.5M active params, 1 epoch, 14d data | ~21 min |
| RF-DPO Hard, 807 steps | ~62 min |
| GRPO/ECPO, G=512, rl_ratio=1.0 | ~80 min |
| Scaling M+, 101M active params, 1 epoch, 14d data | ~207 min |

## Reporting

After a completed experiment, update:

1. `experiments/logs/<phase>/exp-NNN.md`
2. `experiments/logs/<phase>/README.md`
3. `README.md`

Keep implementation details in the relevant module README, not in experiment summaries.
