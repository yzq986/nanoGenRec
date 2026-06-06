# Project Instructions

[English](CLAUDE.md) | [中文](CLAUDE.md.zh.md)

## 后台任务监控原则

**绝对禁止 `sleep` + 检查结果。** 启动后台进程后，直接报告"已启动，PID=XXX"然后结束这个 turn。结果由 cron 守护进程（每 2 分钟触发）异步检查。在同一个 turn 里 sleep/tail/poll 是浪费时间且多余的。

## Auto commit & push

Every time a coding task finishes (implementation complete, no more pending changes), automatically:
1. `git add` the changed files
2. `git commit` with a descriptive message
3. `./push.sh`

Do not ask for confirmation — just do it after each coding round.

## README 分工原则

各阶段有两个 README，各司其职，不重复内容：

| 位置 | 受众 | 内容 |
|------|------|------|
| `<phase>/README.md`（`rl/`, `ntp/`, `tokenizer/`, `model/`）| 改代码的人 | 文件说明、接口、实现细节、已验证超参、踩坑记录 |
| `experiments/logs/<phase>/README.md` | 设计实验的人 | EXP 列表、当前 SOTA、下一步实验方向 |

两者互相引用，不重复。每次代码变更或实验完成后，对应的两个 README 都要更新。

## gr conda env 标准配置

`/home/dev/.conda/envs/gr` — 所有训练/eval/preprocess 任务使用此环境。

| 包 | 版本 |
|----|------|
| Python | 3.12.13 |
| torch | 2.7.1+cu128 |
| CUDA (driver) | 12.8 |
| faiss-gpu | 1.14.1（GPU count=8） |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| pyarrow | 24.0.0 |

**重建方法**（如需从头搭）：
```bash
/root/miniconda3/bin/conda create -n gr python=3.12 -y
/home/dev/.conda/envs/gr/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
/root/miniconda3/bin/conda install -n gr -c pkgs/main faiss-gpu=1.14.1 -y
/home/dev/.conda/envs/gr/bin/pip install numpy pandas pyarrow PyYAML pytest -i https://mirrors.aliyun.com/pypi/simple/
# 注意：conda 安装 faiss 后可能降 numpy 到 1.x，需强制重装：
/home/dev/.conda/envs/gr/bin/pip install --force-reinstall "numpy>=2.0" -i https://mirrors.aliyun.com/pypi/simple/
```

## Python module resolution

The repository root is added directly to `sys.path`. All modules are imported without a
a package prefix — e.g. `from eval.batch import ...`, `from ntp.train import ...`.

For standalone scripts under `experiments/scripts/`:

```python
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)  # adds the repository root itself
```

The CLI entry point is always `python run.py <command>`, NOT `python -m <package>`.
For DDP/torchrun, use `torchrun ... run.py <command>`.

**Shell 脚本 (.sh) 也必须设置 PYTHONPATH**：脚本顶部（`set -euo pipefail` 之后）加：

```bash
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
```

注意是 `${REPO_ROOT}` 本身，**不是父目录**。

## 新实验标准流程

**所有新实验走 `run_exp.py` + YAML，不再写 `.sh` 训练脚本。**

### 1. 创建 YAML config

新建 `experiments/configs/exp-NNN.yaml`，**必须先 Read `experiments/configs/_base.yaml`** 确认 defaults，然后只写需要覆盖的参数：

```yaml
name: exp047
description: "..."
base: _base.yaml

sid_cache_name: exp026-0.6b-14d     # grep 已有 yaml 确认当前标准
ntp_data_name: exp026-0.6b-14d      # 若复用其他实验数据，显式指定

variants:
  - name: exp047-a
    torope_time_split: 0.25
  - name: exp047-b
    torope_time_split: 0.50
```

多 variant 对比实验用 `variants:` 列表；单 config 实验省略 `variants:`。

### 2. 必须确认的参数（写 YAML 前 grep）

**禁止凭记忆编造：**

1. **`sid_cache_name`** — grep `sid_cache_name:` in `experiments/configs/`
2. **`ntp_data_name`** — 复用已有数据时，grep 对应 exp 脚本确认目录名
3. **`date_start` / `date_end`** — grep 最近 yaml 确认日期范围
4. **已有 baseline** — `--check` 会自动显示相似实验，直接复用，不要重训

### 3. 运行

```bash
# 检查：显示每个 variant 的相似历史实验（防重训）
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --check

# 运行所有 variants
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --no-smoke --commit

# 只跑某个 variant（断点续跑）
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --only exp047-a --no-smoke
```

### 4. 加入队列（后台跑）

```bash
echo "run_config.sh experiments/configs/exp-NNN.yaml  /tmp/expNNN.log  exp-NNN complete!" >> experiments/queue.txt
```

`run_config.sh` 是通用 wrapper，内部调用 `run_exp.py --no-smoke --commit`。

### 5. 需要新 preprocess 时

`preprocess-sid` 和 `preprocess-ntp` 的日期范围必须兼容，SID 覆盖的 item 集合必须包含 NTP 行为数据中的 item。必须显式指定 `--date_start/--date_end`，不要用 `auto`。

## Git remotes

| Remote | URL | Purpose |
|--------|-----|---------|
| `origin` | `git@github.com:yzq986/nanoGenRec.git` | Public GitHub repository |

- **Push**: `./push.sh` pushes the current branch to configured public remotes.
- **Pull**: `git pull --rebase origin master`

## 分布式代码陷阱

**禁止用 `hash()` 做跨进程路由**：Python 3.3+ 默认随机化 `PYTHONHASHSEED`，`torchrun` 每个 rank 是独立进程，`hash(key) % n` 会导致 item 静默丢弃或重复（已踩坑：4B embedding cache 33% 数据重复）。必须用 `hashlib.sha256`。

## 设计实验前必须确认数据 pipeline

**在设计使用新特征的实验前，必须先确认 data pipeline 是否已包含该特征。**

检查顺序：
1. `ntp/preprocess.py`：`save_shard` / `load_shard` 是否存储/读取该特征？
2. `ntp/train.py`：`build_unified_sequences` 是否填充？`side_features_lists` 是否传入？
3. `ntp/model.py`：`embed_with_features` 是否会用到？

如果任意一环缺失，必须**先接通 pipeline**，再开始实验。绝不能跑完后发现特征全程为 0。

已踩坑：EXP-044 TO-RoPE，timestamps 代码里传了但全为 0（pipeline 未接通），TO-RoPE vs baseline 对比无效。

## Code quality

- **不确定的 API 必须验证**：先用 Grep 或 WebSearch 确认，不要凭记忆猜。已踩坑：`torch.cuda.get_device_properties().total_memory`（不是 `total_mem`）。
- **实现优化不能破坏数学语义**：公式拆成多步实现时，原本框架隐式保证的数学性质变成需要手动维护的不变量。写完回到原始公式逐项核对。
- **只改 eval 代码不需要重训**：改动只影响推理/评测路径时，直接用已有 checkpoint re-eval，不要浪费 GPU 重训。
- **禁止在新实验中重跑已有 config 作对照**：参数完全一致的 config 直接引用已有结果，在 log.md 里写 `参考 EXP-NNN：R@500=xx.x%`。

## Eval 对齐规则

**`train-ntp` 的 inline eval ≠ 全量 eval，不能与 baseline 直接比较。**

- inline eval：beam search 仅 250 items/rank（1000 total），是快速健康检查，绝对数字不可信。
- 全量 eval：
  ```bash
  torchrun --nproc_per_node=N run.py eval-ntp \
      --checkpoint experiments/ntp_checkpoints/<name> \
      --n_recall 1000
  ```
- 每次新实验跑完，补全量 eval 后更新三处文档：
  1. `experiments/logs/<phase>/exp-NNN.md` — 单实验详细记录
  2. `experiments/logs/<phase>/README.md` — 阶段汇总 SOTA
  3. `README.md` — 根目录 homepage
- `train_meta.json` 里的 eval keys 是 `item_recall@10` / `item_recall@500`（带 `@`，不是 `_`）。

## Research Agent Mode

当作为自主研究 Agent 运行时（用户指示 "follow research/program.md" 或类似指令）：

1. **先读 `research/program.md`** — 完整操作手册
2. **检查 `research/inbox/`** 获取人类指令
3. **执行实验前估算运行时间**：`python experiments/scripts/estimate_runtime.py --active_params <N> --total_tokens <T> --gpus 8`
4. **不超过 30 分钟时间预算**（除非人类明确批准）
5. **不修改源码**（`ntp/`, `rl/`, `model/`, `data/`, `eval/`）除非通过 outbox 获得人类批准
6. **每个动作完成后** 更新 `research/status.md` + `research/log.md` + commit + push
7. 所有通信格式见 `research/schema.md`

## 实验排队与 Cron 监控

队列由两个文件管理：

- `experiments/queue.txt` — 实验队列，追加即生效
- `experiments/queue_state.json` — 当前状态，cron 读这个决定做什么

**启动新实验 / 追加队列：**
```bash
echo "run_config.sh experiments/configs/exp-NNN.yaml  /tmp/expNNN.log  exp-NNN complete!" >> experiments/queue.txt
```

**首次启动（队列为空时）：**
```bash
nohup bash experiments/scripts/run_config.sh experiments/configs/exp-NNN.yaml > /tmp/expNNN.log 2>&1 &
cat > experiments/queue_state.json <<EOF
{"current": "run_config.sh", "log": "/tmp/expNNN.log", "done_string": "exp-NNN complete!", "status": "running", "pid": $!}
EOF
# 确认守护 cron 存在（CronList 检查），没有就创建
```

**守护 Cron 参数：间隔 2 分钟，durable=true。**

**守护 Cron prompt（每个 session 只需一个）：**
```
实验队列守护进程 — 读取 experiments/queue_state.json 和 experiments/queue.txt 管理实验链。
每次触发：
1. 读 queue_state.json，检查当前实验 log 是否出现 done_string
2. 如果 status=pending_gpu：检查 log 文件是否存在且有内容（说明 GPU 侧已启动），如有则改 status=running
3. 未完成：报告进度（grep "step \|EXP-" LOG | tail -3），继续等待
4. 出错（log 有 Traceback/Error/exitcode : 1）：告知用户，state 改为 error，停止

**早停检查 — 如发现以下任意情况，立即执行 early-stop 流程（见下），告知用户：**
- log 中出现 Traceback / Error / exitcode : 1（非 SIGTERM）
- 连续 3 次检查 step 数字没有增加（训练卡住）
- reward_mean 持续为 0 超过 50 steps（reward 信号消失）
- advantage_mean 绝对值 > 50（数值爆炸）
- clip_fraction > 0.99 且持续 20+ steps（策略崩溃）

**R@500 早停检查 — 每次有新的 inline eval 结果时执行：**
- 从 log 中 grep 所有 `item_recall@500:` 行，提取数值列表（去重相邻重复，每 epoch 最多取 1 个值）
- 计算历史最佳值 best_r500，取最新值 latest_r500
- 如果 best_r500 >= 0.45 且 latest_r500 < best_r500 - 0.10：触发早停（R@500 下跌超过 10pp）
- 如果 best_r500 < 0.45 但 latest_r500 < 0.30 且已跑了 >= 2 个 epoch：触发早停（基础性能不及格）
- 否则：报告 "R@500: latest=X.XX best=X.XX"，继续等待

实现提取的 bash 命令：
```bash
grep 'item_recall@500:' LOG | awk '{print $2}' | python3 -c "
import sys
vals=[float(l) for l in sys.stdin]
deduped=[]
for v in vals:
    if not deduped or abs(v-deduped[-1])>1e-9:
        deduped.append(v)
if deduped:
    print('best={:.4f} latest={:.4f} n_evals={}'.format(max(deduped),deduped[-1],len(deduped)))
"
```

**Early-stop 流程：**
  a. pkill -f "<current脚本名>" 并 pkill -f "torchrun.*<实验名关键词>"
  b. 更新 queue_state.json status=stopped
  c. 告知用户（说明触发原因和最后的 R@500 数字）
  d. 不启动下一个实验，保持 cron 存活

**效果预警 — 告知用户但不自动停止：**
- reward_mean 在 100 steps 后仍 < 0.1

5. 完成：
   a. 执行 post_hook（如 EVAL_MID_CHECKPOINTS：串行 eval ep1/ep2，找最优 checkpoint）
   b. 读 train_meta.json，更新三处：`experiments/logs/<phase>/exp-NNN.md`、`experiments/logs/<phase>/README.md` SOTA 行、`README.md` 根目录 homepage
   c. git add experiments/ && git commit -m "EXP-XXX complete: ..." && ./push.sh
   d. 读 queue.txt，找下一个未完成的实验，nohup 启动，更新 queue_state.json
   e. 如队列已空：state 改为 done，告知用户，保持 cron 存活
```

**不要用脚本 chain（A 末尾调用 B）来排队实验**，因为前序实验可能已经在后台跑了。
