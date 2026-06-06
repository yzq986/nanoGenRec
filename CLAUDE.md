# Project Instructions

[English](CLAUDE.md) | [Chinese](CLAUDE.md.zh.md)

## Background task monitoring principles

**Absolutely disallows `sleep` + checking results. ** After starting the background process, directly report "Started, PID=XXX" and then end the turn. Results are checked asynchronously by the cron daemon (triggered every 2 minutes). Sleep/tail/poll in the same turn is a waste of time and redundant.

## Auto commit & push

Every time a coding task finishes (implementation complete, no more pending changes), automatically:
1. `git add` the changed files
2. `git commit` with a descriptive message
3. `./push.sh`

Do not ask for confirmation — just do it after each coding round.

## README Principles of division of labor

There are two READMEs for each stage, each performing its own duties without duplication of content:

| Location | Audience | Content |
|------|------|------|
| `<phase>/README.md` (`rl/`, `ntp/`, `tokenizer/`, `model/`) | People who change the code | FileDescription, interfaces, implementation details, verified super parameters, pitfall records |
| `experiments/logs/<phase>/README.md` | DesignExperiment people | EXP list, current SOTA, next ExperimentDirection |

The two refer to each other without duplication. After each code change or experiment is completed, the corresponding two READMEs must be updated.

## gr conda env standard configuration

`/home/dev/.conda/envs/gr` — All training /eval/preprocess tasks use this environment.

| Package | Version |
|----|------|
| Python | 3.12.13 |
| torch | 2.7.1+cu128 |
| CUDA (driver) | 12.8 |
| faiss-gpu | 1.14.1 (GPU count=8) |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| pyarrow | 24.0.0 |

**Reconstruction method** (if you need to build it from scratch):
```bash
/root/miniconda3/bin/conda create -n gr python=3.12 -y
/home/dev/.conda/envs/gr/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
/root/miniconda3/bin/conda install -n gr -c pkgs/main faiss-gpu=1.14.1 -y
/home/dev/.conda/envs/gr/bin/pip install numpy pandas pyarrow PyYAML pytest -i https://mirrors.aliyun.com/pypi/simple/
# Note: After installing faiss in conda, numpy may be downgraded to 1.x, and a forced reinstallation is required:
/home/dev/.conda/envs/gr/bin/pip install --force-reinstall "numpy>=2.0" -i https://mirrors.aliyun.com/pypi/simple/
```

## Python module resolution

The repo root (`gr-demo/`) is added directly to `sys.path`. All modules are imported without a
`gr_demo.` prefix — e.g. `from eval.batch import ...`, `from ntp.train import ...`.

For standalone scripts under `experiments/scripts/`:

```python
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)  # adds gr-demo/ itself
```

The CLI entry point is always `python run.py <command>`, NOT `python -m gr_demo`.
For DDP/torchrun, use `torchrun ... run.py <command>`.

**Shell scripts (.sh) must also set PYTHONPATH**: add at the top of the script (after `set -euo pipefail`):

```bash
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
```

Note that it is `${REPO_ROOT}` itself (i.e. `gr-demo/`), **not the parent directory**.

## New experimental standard process

**All new experiments use `run_exp.py` + YAML, and no longer write `.sh` training scripts. **

### 1. Create YAML config

Create a new `experiments/configs/exp-NNN.yaml`, **You must first Read `experiments/configs/_base.yaml`** to confirm the defaults, and then only write the parameters that need to be overwritten:

```yaml
name: exp047
description: "..."
base: _base.yaml

sid_cache_name: exp026-0.6b-14d # grep already has yaml to confirm the current standard
ntp_data_name: exp026-0.6b-14d # If reusing other experimental data, specify it explicitly

variants:
  - name: exp047-a
    torope_time_split: 0.25
  - name: exp047-b
    torope_time_split: 0.50
```

Use the `variants:` list for multi-variant comparison experiments; omit `variants:` for single-config experiments.

### 2. Parameters that must be confirmed (grep before writing YAML)

**No fabrication from memory:**

1. **`sid_cache_name`** — grep `sid_cache_name:` in `experiments/configs/`
2. **`ntp_data_name`** — When reusing existing data, grep corresponds to the exp script to confirm the directory name.
3. **`date_start` / `date_end`** — grep latest yaml confirmation date range
4. **Existing baseline** — `--check` will automatically display similar experiments and reuse them directly without retraining.

### 3. Run

```bash
# Check: display similar historical experiments for each variant (anti-retraining)
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --check

# Run all variants
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --no-smoke --commit

# Only run a certain variant (continue running from breakpoint)
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --only exp047-a --no-smoke
```

### 4. Join the queue (running in the background)

```bash
echo "run_config.sh experiments/configs/exp-NNN.yaml  /tmp/expNNN.log  exp-NNN complete!" >> experiments/queue.txt
```

`run_config.sh` is a general wrapper that internally calls `run_exp.py --no-smoke --commit`.

### 5. When a new preprocess is needed

The date ranges of `preprocess-sid` and `preprocess-ntp` must be compatible, and the item collection covered by SID must contain items in the NTP behavioral data. `--date_start/--date_end` must be specified explicitly, do not use `auto`.

## Git remotes

| Remote | URL | Purpose |
|--------|-----|---------|
| `origin` | `git@github.com:yzq986/gr_demo.git` | Public GitHub repository |

- **Push**: `./push.sh` pushes the current branch to configured public remotes.
- **Pull**: `git pull --rebase origin master`

## Distributed code trap

**It is forbidden to use `hash()` for cross-process routing**: Python 3.3+ defaults to randomization `PYTHONHASHSEED`, `torchrun`. Each rank is an independent process, `hash(key) % n` will cause items to be silently discarded or repeated (trapped: 4B embedding cache 33% data duplication). Must use `hashlib.sha256`.

## The data pipeline must be confirmed before designing experiments

**Before designing an experiment using a new feature, you must first confirm whether the data pipeline already contains the feature. **

Check order:
1. `ntp/preprocess.py`: `save_shard` / `load_shard` Whether to store/read this feature?
2. `ntp/train.py`: Is `build_unified_sequences` populated? Is `side_features_lists` passed in?
3. `ntp/model.py`: Will `embed_with_features` be used?

If any link is missing, the pipeline must be connected first before starting the experiment. Never find out after running that the features are all 0.

Trampled: EXP-044 TO-RoPE, timestamps are passed in the code but all are 0 (pipeline is not connected), TO-RoPE vs baseline comparison is invalid.

## Code quality

- **Uncertain APIs must be verified**: Confirm with Grep or WebSearch first, don't guess from memory. Trampled: `torch.cuda.get_device_properties().total_memory` (not `total_mem`).
- **Implementation optimization must not destroy mathematical semantics**: When the formula is split into multi-step implementation, the mathematical properties implicitly guaranteed by the original framework become invariants that need to be manually maintained. After writing, go back to the original formula and check it item by item.
- **Only changing the eval code does not require retraining**: When the change only affects the inference/evaluation path, directly use the existing checkpoint re-eval, and do not waste GPU retraining.
- **It is prohibited to rerun existing configs for comparison in new experiments**: The config with exactly the same parameters directly refers to the existing results, and writes `Reference EXP-NNN: R@500=xx.x%` in log.md.

## Eval alignment rules

**`train-ntp`’s inline eval ≠ full eval, cannot be directly compared with baseline. **

- inline eval: beam search only 250 items/rank (1000 total), is a quick health check, the absolute number cannot be trusted.
- full eval:
  ```bash
  torchrun --nproc_per_node=N run.py eval-ntp \
      --checkpoint experiments/ntp_checkpoints/<name> \
      --n_recall 1000
  ```
- Each time a new experiment is run, three documents are updated after completing the eval:
  1. `experiments/logs/<phase>/exp-NNN.md` — detailed record of a single experiment
  2. `experiments/logs/<phase>/README.md` — phase summary SOTA
  3. `README.md` — root directory homepage
- The eval keys in `train_meta.json` are `item_recall@10` / `item_recall@500` (with `@`, not `_`).

## Research Agent Mode

When running as an autonomous research agent (user directive "follow research/program.md" or similar):

1. **Read first `research/program.md`** — Complete Operation Manual
2. **Check `research/inbox/`** for human instructions
3. **Estimate running time before executing experiments**: `python experiments/scripts/estimate_runtime.py --active_params <N> --total_tokens <T> --gpus 8`
4. **No more than 30 minutes time budget** (unless explicitly approved by a human)
5. **No modification of source code** (`ntp/`, `rl/`, `model/`, `data/`, `eval/`) unless human approval is obtained through outbox
6. **After each action is completed** Update `research/status.md` + `research/log.md` + commit + push
7. See `research/schema.md` for all communication formats

## Experiment queuing and Cron monitoring

The queue is managed by two files:

- `experiments/queue.txt` — Experiment queue, it will take effect immediately after appending
- `experiments/queue_state.json` — the current state, cron reads this to decide what to do

**Start new experiment/append queue:**
```bash
echo "run_config.sh experiments/configs/exp-NNN.yaml /tmp/expNNN.log exp-NNN complete!" >> experiments/queue.txt
```

**First start (when queue is empty):**
```bash
nohup bash experiments/scripts/run_config.sh experiments/configs/exp-NNN.yaml > /tmp/expNNN.log 2>&1 &
cat > experiments/queue_state.json <<EOF
{"current": "run_config.sh", "log": "/tmp/expNNN.log", "done_string": "exp-NNN complete!", "status": "running", "pid": $!}
EOF
# Confirm that the daemon cron exists (CronList check), create it if not
```

**Guardian Cron parameters: interval 2 minutes, durable=true. **

**Guard Cron prompt (only one per session):**
```
Experiment queue daemon - reads experiments/queue_state.json and experiments/queue.txt to manage the experiment chain.
Every trigger:
1. Read queue_state.json and check whether done_string appears in the current experiment log.
2. If status=pending_gpu: Check whether the log file exists and has content (indicating that the GPU side has been started), if so, change status=running
3. Not completed: report progress (grep "step \|EXP-" LOG | tail -3), continue to wait
4. An error occurs (the log has Traceback/Error/exitcode: 1): inform the user, change the state to error, and stop.

**Early stop check - If any of the following conditions are found, immediately execute the early-stop process (see below) and notify the user:**
- Traceback/Error/exitcode: 1 (not SIGTERM) appears in log
- Check the step number 3 times in a row without increasing (training stuck)
- reward_mean continues to be 0 for more than 50 steps (reward signal disappears)
- advantage_mean absolute value > 50 (numerical explosion)
- clip_fraction > 0.99 for 20+ steps (strategy crashes)

**R@500 Early stop check - executed every time there is a new inline eval result: **
- Grep all `item_recall@500:` lines from the log and extract a list of values (removing adjacent duplicates, taking at most 1 value per epoch)
- Calculate the historical best value best_r500 and take the latest value latest_r500
- If best_r500 >= 0.45 and latest_r500 < best_r500 - 0.10: trigger early stop (R@500 falls more than 10pp)
- If best_r500 < 0.45 but latest_r500 < 0.30 and has run >= 2 epochs: trigger early stop (basic performance fails)
- Otherwise: report "R@500: latest=X.XX best=X.XX" and continue to wait.

bash command to implement extraction:
```bash
grep 'item_recall@500:' LOG | awk '{print $2}' | python3 -c "
importsys
vals=[float(l) for l in sys.stdin]
deduped=[]
for v in vals:
    if not deduped or abs(v-deduped[-1])>1e-9:
        deduped.append(v)
if deduped:
    print('best={:.4f} latest={:.4f} n_evals={}'.format(max(deduped),deduped[-1],len(deduped)))
"
```

**Early-stop process:**
  a. pkill -f "<current script name>" and pkill -f "torchrun.*<experiment name keyword>"
  b. Update queue_state.json status=stopped
  c. Inform user (state trigger reason and final R@500 number)
  d. Do not start the next experiment and keep cron alive.

**Effectiveness warning — inform the user but do not automatically stop:**
- reward_mean still < 0.1 after 100 steps

5. Complete:
   a. Execute post_hook (such as EVAL_MID_CHECKPOINTS: serial eval ep1/ep2, find the optimal checkpoint)
   b. Read train_meta.json and update three places: `experiments/logs/<phase>/exp-NNN.md`, `experiments/logs/<phase>/README.md` SOTA line, `README.md` root directory homepage
   c. git add experiments/ && git commit -m "EXP-XXX complete: ..." && ./push.sh
   d. Read queue.txt, find the next unfinished experiment, start nohup, and update queue_state.json
   e. If the queue is empty: change state to done, inform the user, and keep cron alive.
```

**Don't use a script chain (calling B at the end of A) to queue experiments**, because the previous experiment may have been running in the background.
