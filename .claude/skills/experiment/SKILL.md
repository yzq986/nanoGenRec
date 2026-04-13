---
name: experiment
description: Record an experiment entry in experiments/log.md with structured format (Background → Hypothesis → Design → Results → Analysis → Next Steps), and generate a runnable .sh script
argument-hint: [experiment title]
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Glob, Grep
---

# /experiment Skill

Record a new experiment entry in `experiments/log.md` using the project's structured six-section format.

## Instructions

1. **Read the log**: Use Read tool on `experiments/log.md` to find the highest existing `EXP-NNN` number.

2. **Determine new ID**: Increment the highest EXP number by 1 (e.g., if EXP-001 exists, create EXP-002).

3. **Extract experiment info from conversation context**:
   - Title: Use the argument if provided, otherwise infer from discussion
   - Background: Current state and problem being solved
   - Hypothesis: Expected results and reasoning
   - Design: Variables, fixed params, metrics, data
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
   - The Run Commands section in log.md should reference this script: `bash experiments/scripts/exp-{nnn}.sh`

## Entry Format

```markdown
## EXP-{NNN}: {Title}

**Date**: {今天日期 YYYY-MM-DD}
**Status**: {planned|running|completed}
**Results**: {结果目录链接，如 [./hyperparam/YYYY-MM-DD_xxx/](./hyperparam/YYYY-MM-DD_xxx/)，若无则写 TBD}

### Background
{当前状态、要解决的问题}

### Hypothesis
{预期结果及原因}

### Design
- **Variable**: {实验变量}
- **Fixed**: {固定参数}
- **Metric**: {评估指标}
- **Data**: {数据集}

### Run
`bash experiments/scripts/exp-{nnn}.sh`

### Results
{跑完后填写，含表格；未完成则写 TBD}

### Analysis
{结果解读；未完成则写 TBD}

### Next Steps
{下一步计划；未完成则写 TBD}

---
```

## Example

User discusses testing different cluster sizes (512 vs 1024 vs 2048) for NTP recall.

```
/experiment NTP Recall vs Cluster Size
```

Creates two artifacts:

### 1. `experiments/log.md` entry:

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
echo "EXP-002 complete!"
```
