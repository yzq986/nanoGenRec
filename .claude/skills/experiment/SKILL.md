---

[English](SKILL.md) | [Chinese](SKILL.zh.md)
name: experiment
description: Record an experiment entry in experiments/logs/ with structured format (Background → Hypothesis → Design → Results → Analysis → Next Steps), and generate a runnable .sh script
argument-hint: [experiment title]
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Glob, Grep
---

# /experiment Skill

Record a new experiment entry in `experiments/logs/` using the project's structured six-section format.

## Instructions

1. **Read the log**: Use Read tool on `experiments/logs/` to find the highest existing `EXP-NNN` number.

2. **Determine new ID**: Increment the highest EXP number by 1 (e.g., if EXP-001 exists, create EXP-002).

3. **Extract experiment info from conversation context**:
   - Title: Use the argument if provided, otherwise infer from discussion
   - Background: Current state and problem being solved
   - Hypothesis: Expected results and reasoning — **Expected direction of change must be listed metric by metric** (see Hypothesis requirements below)
   - Design: Variables, fixed params, metrics, data — **If existing metrics are not enough to verify the hypothesis, new metrics must be added**
   - Results/Analysis/Next Steps: Fill if results are available in conversation, otherwise leave placeholder

4. **Determine status**:
   - `planned` — experiment not yet run
   - `running` — experiment in progress
   - `completed` — results are available

5. **Insert the new entry** using Edit tool:
   - Location: After the `---` line that follows `## Template` block, before the first `## EXP-` entry
   - This maintains reverse chronological order (newest first)

6. **Generate a runnable shell script** using Write tool:
   - Path: `experiments/scripts/exp-{nnn}.sh` (e.g. `experiments/scripts/exp-002.sh`)
   - The script should be self-contained and one-click runnable
   - **IMPORTANT**: CLI entry point is always `python run.py`, NOT `python -m gr_demo`
   - Include all experiment run commands from the Design section
   - Add a header comment with experiment title, date, and brief description
   - Make the script executable-ready (include `#!/bin/bash` and `set -e`)
   - Add `echo` lines for progress visibility between commands
   - If the experiment has multiple configs (e.g. baseline + variants), include all of them
   - **Smoke test (Phase 0)**: The script **must** add a smoke test stage before the formal experiment, using ~1% data + very few steps to run through the complete pipeline (data loading → model forward/backward → save). After the verification is passed, start the big experiment. Training scripts should support the `--dry_run` parameter to achieve this functionality. `set -e` ensures that the entire script stops if the smoke test fails.
   - **ETA Display**: The training script **must** display the ETA (estimated time remaining) in the log. ETA is also displayed each time loss is printed, and the total remaining time is displayed at the end of epoch. Format: `ETA 2h35m` or `ETA 12m30s`.
   - **Timing of each stage**: The script **must** record the time spent in each stage (training, eval) to facilitate the estimation of time budget for subsequent experiments. Use `$(date +%s)` to record the timestamp before and after the training/eval of each config. At the end of the training, the time consumption in the format of `(${TRAIN_MIN}min)` is printed. After the eval is completed, total is printed. Example:
     ```bash
     T0=$(date +%s)
     torchrun ... run.py grpo-train ...
     T1=$(date +%s)
     TRAIN_MIN=$(( (T1 - T0) / 60 ))
     echo "Training complete (${TRAIN_MIN}min)"
     T2=$(date +%s)
     torchrun ... run.py eval-ntp ...
     T3=$(date +%s)
     EVAL_MIN=$(( (T3 - T2) / 60 ))
     TOTAL_MIN=$(( (T3 - T0) / 60 ))
     echo "Total: train=${TRAIN_MIN}min eval=${EVAL_MIN}min total=${TOTAL_MIN}min"
     ```
     The `wall_time_s` print summary should also be read from `train_meta.json` at the end of the script:
     ```bash
     python3 -c "
     import json,os
     for name in ['exp-NNN-config-a', 'exp-NNN-config-b']:
         path = 'experiments/ntp_checkpoints/' + name + '/train_meta.json'
         if os.path.exists(path):
             m = json.load(open(path))
             w = m.get('train', {}).get('wall_time_s', 0)
             print(f' {name}: train={int(w)//60}min{int(w)%60}s')
     " 2>/dev/null || true
     ```
   - **GPU Utilization Strategy**: The experimental environment is **8 x A100 (40GB)**. Choose different parallel strategies based on the type of experiment:
     - **DDP training experiments** (such as contrastive learning fine-tuning, NTP training): each config occupies all 8 cards `torchrun --nproc_per_node=8`, multiple configs are executed serially. Reason: The throughput of DDP 8 cards is doubled compared to 4 cards + the negatives of comparative learning are doubled, and the total wall time of serial is shorter.
     - **Non-DDP independent experiments** (such as hyperparameter search, quantitative evaluation): use `CUDA_VISIBLE_DEVICES` to allocate different configs to different GPUs for parallel running (`&` background + `wait`).
     - Immediately `git commit + ./push.sh` after each config result comes out (use `flock` to serialize git operations to avoid parallel conflicts).
   - The Run Commands section in log.md should reference this script: `bash experiments/scripts/exp-{nnn}.sh`
   - **At the end of the script**, add git commit + push to auto-persist results:
     ```bash
     echo ""
     echo ">>> Committing results..."
     git add experiments/
     git commit -m "EXP-{NNN} results: {short title}" || echo "Nothing to commit"
     ./push.sh
     ```

## Hypothesis requirements (mandatory)

Every time you write a hypothesis, you must use a table to list the expected direction of change and the reasons for each indicator. General text descriptions are not allowed.

Format:
```markdown
### Hypothesis

{Description of the core mechanism of the hypothesis, 1-2 sentences}

| Metric | Current Value (control) | Expected change | Reason |
|------|--------------|---------|------|
| clip rate | 95% | ↓ ~20% | sampling so that ρ≈1 by construction |
| adv_std | ≈0 | ↑ >0.3 | Candidate diversity increases, reward variance becomes larger |
| behavior_coverage | 99% | ↓ ~89% | G from 512→64, the net becomes smaller |
| behavior_mean | 0.65 | ↓ ~0.35 | coverage decrease + sparse reward |
| kl_mean | — | ≈0 initially, with Training↑ | Add new Metric, benchmark to be established |
| R@500 | 0.678 | To be determined | Depends on whether the reward signal is sufficient |
```

**If an indicator is not recorded in the existing code, you must first add the indicator to the code before writing the experiment**.

## Metrics required (mandatory)

Core RL metrics (must be logged for every experiment):
- `clip_fraction`: PPO clip rate
- `kl_mean`: KL(π_θ || π_ref), a policy drift metric comparable across experiments
- `adv_std` (advantage_std): the standard deviation of advantage, reflecting the contrast signal strength
- `behavior_coverage`: the proportion of contexts with non-zero reward
- `behavior_mean`: average behavior reward (note that it is affected by G, cross-experimental comparisons need to be marked with G)
- `R@500` (full eval): final business indicator

If the hypothesis involves new mechanisms (such as entropy, diversity, on-policy ratio, etc.), the corresponding indicators must be added to the code before the experiment, and you cannot find out afterwards that there is no data.

## Entry Format

```markdown
## EXP-{NNN}: {Title}

**Date**: {Today’s date YYYY-MM-DD}
**Status**: {planned|running|completed}
**Results**: {Result directory link, such as [./hyperparam/YYYY-MM-DD_xxx/](./hyperparam/YYYY-MM-DD_xxx/), if not, write TBD}

### Background
{Current status, problems to be solved}

### Hypothesis

{hypothetical core mechanism, 1-2 sentences}

| Metric | Current Value (control) | Expected change | Reason |
|------|--------------|---------|------|
| clip rate | ? | ↑/↓/→ ? | ... |
| kl_mean | ? | ↑/↓/→ ? | ... |
| adv_std | ? | ↑/↓/→ ? | ... |
| behavior_coverage | ? | ↑/↓/→ ? | ... |
| behavior_mean | ? | ↑/↓/→ ? | ... |
| R@500 | ? | ↑/↓/→ ? | ... |

### Design
- **Variable**: {experimental variable}
- **Fixed**: {Fixed parameters}
- **Metric**: {Evaluation indicators, if the existing ones are not enough, add new ones}
- **Data**: {data set}

### Run
`bash experiments/scripts/exp-{nnn}.sh`

### Results
{Fill it out after finishing the run, including the form; if not completed, write TBD}

### Analysis
{Interpretation of results; if not completed, write TBD}

### Next Steps
{Next step plan; if not completed, write TBD}

---
```

## Example

User discusses testing different cluster sizes (512 vs 1024 vs 2048) for NTP recall.

```
/experiment NTP Recall vs Cluster Size
```

Creates two artifacts:

### 1. `experiments/logs/` entry:

```markdown
## EXP-002: NTP Recall vs Cluster Size

**Date**: 2026-04-13
**Status**: planned
**Results**: TBD

### Background
Need to evaluate how cluster size affects NTP retrieval recall.

### Hypothesis
Larger cluster sizes should improve recall by reducing semantic ID collisions, but with diminishing returns.

### Design
- **Variable**: num_clusters (512, 1024, 2048)
- **Fixed**: 3 layers, niter=25, nredo=3
- **Metric**: NTP recall@10, collision rate
- **Data**: standard evaluation set

### Run
`bash experiments/scripts/exp-002.sh`

### Results
TBD

### Analysis
TBD

### Next Steps
TBD

---
```

### 2. `experiments/scripts/exp-002.sh`:

```bash
#!/bin/bash
set -e

# EXP-002: NTP Recall vs Cluster Size
# Date: 2026-04-13
# Variable: num_clusters (512, 1024, 2048)

echo "=========================================="
echo "EXP-002: NTP Recall vs Cluster Size"
echo "=========================================="

echo ""
echo ">>> Running cluster sweep..."
python run.py hyperparam --skip_embedding \
    --clusters 512 1024 2048 \
    --name exp002-cluster-sweep

echo ""
echo ">>> Committing results..."
git add experiments/
git commit -m "EXP-002 results: NTP Recall vs Cluster Size" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-002 complete!"
```

## After starting the experiment: Join the queue

Each time you generate an experiment script, you must append the experiment to the queue file and confirm that the daemon cron is alive:

### 1. Append to queue
```bash
echo "exp-NNN.sh /tmp/expNNN.log EXP-NNN complete!" >> experiments/queue.txt
```

If the experiment has multiple intermediate checkpoints (multi-epoch), add POST_HOOK:
```bash
echo "exp-NNN.sh /tmp/expNNN.log EXP-NNN complete! EVAL_MID_CHECKPOINTS=exp-NNN-output-name" >> experiments/queue.txt
```

### 2. If this is the first experiment (the queue is empty), you also need:
```bash
# Start experiment
nohup bash experiments/scripts/exp-NNN.sh --no-smoke > /tmp/expNNN.log 2>&1 &
EXP_PID=$!

#Initialize queue_state.json
python3 -c "
import json
state = {
  'current': 'exp-NNN.sh',
  'log': '/tmp/expNNN.log',
  'done_string': 'EXP-NNN complete!',
  'status': 'running',
  'pid': ${EXP_PID}
}
json.dump(state, open('experiments/queue_state.json', 'w'), indent=2)
"

# Confirm that the daemon cron exists (CronList), if not, use CronCreate to create it (see CLAUDE.md daemon Cron prompt)
```

### 3. If there are already experiments running in the queue, just append
The daemon cron will automatically detect new entries in queue.txt, and automatically start the newly added experiment after the previous experiment is completed, without any other operations.

### queue.txt full format reference
```
# Comment lines are ignored, blank lines are ignored
# SCRIPT LOG DONE_STRING POST_HOOK (optional)
exp-038b.sh /tmp/exp038b.log EXP-038B complete! EVAL_MID_CHECKPOINTS=exp038b-hard-lam03-3ep
exp-039b.sh /tmp/exp039b.log EXP-039B complete!
exp-040.sh /tmp/exp040.log EXP-040 complete!
```
