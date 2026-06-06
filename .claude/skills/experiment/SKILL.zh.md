---

[English](SKILL.md) | [中文](SKILL.zh.md)
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
   - Hypothesis: Expected results and reasoning — **必须逐指标列出预期变化方向**（见下方 Hypothesis 要求）
   - Design: Variables, fixed params, metrics, data — **如现有 metrics 不足以验证假设，必须新增 metrics**
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
   - **Smoke test (Phase 0)**: 脚本**必须**在正式实验前加一个 smoke test 阶段，用 ~1% 数据 + 极少步数跑通完整 pipeline（数据加载 → 模型 forward/backward → 保存）。验证通过后再启动大实验。训练脚本应支持 `--dry_run` 参数实现此功能。`set -e` 确保 smoke test 失败时整个脚本停止。
   - **ETA 显示**: 训练脚本**必须**在日志中显示 ETA（预计剩余时间）。每次打印 loss 时同时显示 ETA，epoch 结束时显示总剩余时间。格式: `ETA 2h35m` 或 `ETA 12m30s`。
   - **每阶段计时**: 脚本**必须**记录每个阶段（训练、eval）的耗时，方便后续实验估算时间预算。每个 config 的训练/eval 前后用 `$(date +%s)` 记录时间戳，训练结束时打印 `(${TRAIN_MIN}min)` 格式的耗时，eval 结束后打印 total。示例：
     ```bash
     T0=$(date +%s)
     torchrun ... run.py grpo-train ...
     T1=$(date +%s)
     TRAIN_MIN=$(( (T1 - T0) / 60 ))
     echo "  Training complete  (${TRAIN_MIN}min)"
     T2=$(date +%s)
     torchrun ... run.py eval-ntp ...
     T3=$(date +%s)
     EVAL_MIN=$(( (T3 - T2) / 60 ))
     TOTAL_MIN=$(( (T3 - T0) / 60 ))
     echo "  Total: train=${TRAIN_MIN}min  eval=${EVAL_MIN}min  total=${TOTAL_MIN}min"
     ```
     脚本末尾还应从 `train_meta.json` 读取 `wall_time_s` 打印汇总：
     ```bash
     python3 -c "
     import json, os
     for name in ['exp-NNN-config-a', 'exp-NNN-config-b']:
         path = 'experiments/ntp_checkpoints/' + name + '/train_meta.json'
         if os.path.exists(path):
             m = json.load(open(path))
             w = m.get('train', {}).get('wall_time_s', 0)
             print(f'  {name}: train={int(w)//60}min{int(w)%60}s')
     " 2>/dev/null || true
     ```
   - **GPU 利用策略**: 实验环境是 **8 x A100 (40GB)**。根据实验类型选择不同的并行策略：
     - **DDP 训练类实验**（如对比学习微调、NTP 训练）：每个 config 占满全部 8 卡 `torchrun --nproc_per_node=8`，多个 config 串行执行。原因：DDP 8 卡比 4 卡吞吐翻倍 + 对比学习 negatives 翻倍，串行反而总 wall time 更短。
     - **非 DDP 独立实验**（如超参搜索、量化评测）：用 `CUDA_VISIBLE_DEVICES` 将不同 config 分配到不同 GPU 并行跑（`&` 后台 + `wait`）。
     - 每个 config 结果出来就立即 `git commit + ./push.sh`（用 `flock` 串行化 git 操作避免并行冲突）。
   - The Run Commands section in log.md should reference this script: `bash experiments/scripts/exp-{nnn}.sh`
   - **At the end of the script**, add git commit + push to auto-persist results:
     ```bash
     echo ""
     echo ">>> Committing results..."
     git add experiments/
     git commit -m "EXP-{NNN} results: {short title}" || echo "Nothing to commit"
     ./push.sh
     ```

## Hypothesis 要求（强制）

每次写 Hypothesis 时，**必须用表格逐指标列出预期变化方向和理由**。不允许只写笼统的文字描述。

格式：
```markdown
### Hypothesis

{假设的核心机制描述，1-2 句话}

| 指标 | 当前值（对照） | 预期变化 | 理由 |
|------|--------------|---------|------|
| clip 率 | 95% | ↓ ~20% | sampling 使 ρ≈1 by construction |
| adv_std | ≈0 | ↑ >0.3 | 候选多样性增加，reward 方差变大 |
| behavior_coverage | 99% | ↓ ~89% | G 从 512→64，撒网变小 |
| behavior_mean | 0.65 | ↓ ~0.35 | coverage 下降 + 稀疏 reward |
| kl_mean | — | ≈0 初始，随训练↑ | 新增指标，基准待建立 |
| R@500 | 0.678 | 待定 | 取决于 reward 信号是否足够 |
```

**如果某个指标在现有代码中没有被记录，必须先在代码中新增该指标，再写实验**。

## Metrics 要求（强制）

核心 RL 指标（每个实验都必须记录）：
- `clip_fraction`：PPO clip 率
- `kl_mean`：KL(π_θ || π_ref)，跨实验可比的 policy 漂移指标
- `adv_std`（advantage_std）：advantage 的标准差，反映对比信号强度
- `behavior_coverage`：有非零 reward 的 context 比例
- `behavior_mean`：平均 behavior reward（注意受 G 影响，跨实验对比需标注 G）
- `R@500`（全量 eval）：最终业务指标

如果假设涉及新的机制（如 entropy、diversity、on-policy ratio 等），**必须在实验前把对应指标加入代码**，不能事后才发现没有数据。

## Entry Format

```markdown
## EXP-{NNN}: {Title}

**Date**: {今天日期 YYYY-MM-DD}
**Status**: {planned|running|completed}
**Results**: {结果目录链接，如 [./hyperparam/YYYY-MM-DD_xxx/](./hyperparam/YYYY-MM-DD_xxx/)，若无则写 TBD}

### Background
{当前状态、要解决的问题}

### Hypothesis

{假设的核心机制，1-2 句话}

| 指标 | 当前值（对照） | 预期变化 | 理由 |
|------|--------------|---------|------|
| clip 率 | ? | ↑/↓/→ ? | ... |
| kl_mean | ? | ↑/↓/→ ? | ... |
| adv_std | ? | ↑/↓/→ ? | ... |
| behavior_coverage | ? | ↑/↓/→ ? | ... |
| behavior_mean | ? | ↑/↓/→ ? | ... |
| R@500 | ? | ↑/↓/→ ? | ... |

### Design
- **Variable**: {实验变量}
- **Fixed**: {固定参数}
- **Metric**: {评估指标，如现有不够需新增}
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

## 启动实验后：加入队列

每次生成实验脚本后，**必须**将该实验追加到队列文件，并确认守护 cron 存活：

### 1. 追加到队列
```bash
echo "exp-NNN.sh  /tmp/expNNN.log  EXP-NNN complete!" >> experiments/queue.txt
```

如果实验有多个中间 checkpoint（multi-epoch），加 POST_HOOK：
```bash
echo "exp-NNN.sh  /tmp/expNNN.log  EXP-NNN complete!  EVAL_MID_CHECKPOINTS=exp-NNN-output-name" >> experiments/queue.txt
```

### 2. 如果是第一个实验（队列为空），还需要：
```bash
# 启动实验
nohup bash experiments/scripts/exp-NNN.sh --no-smoke > /tmp/expNNN.log 2>&1 &
EXP_PID=$!

# 初始化 queue_state.json
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

# 确认守护 cron 存在（CronList），没有则用 CronCreate 创建（见 CLAUDE.md 守护 Cron prompt）
```

### 3. 如果队列已有实验在跑，只需追加
守护 cron 会自动检测 queue.txt 的新条目，上一个实验完成后自动启动新追加的实验，无需其他操作。

### queue.txt 完整格式参考
```
# 注释行忽略，空行忽略
# SCRIPT            LOG                  DONE_STRING           POST_HOOK(可选)
exp-038b.sh  /tmp/exp038b.log  EXP-038B complete!  EVAL_MID_CHECKPOINTS=exp038b-hard-lam03-3ep
exp-039b.sh  /tmp/exp039b.log  EXP-039B complete!
exp-040.sh   /tmp/exp040.log   EXP-040 complete!
```
