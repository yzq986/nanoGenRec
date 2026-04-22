# Project Instructions

## Auto commit & push

Every time a coding task finishes (implementation complete, no more pending changes), automatically:
1. `git add` the changed files
2. `git commit` with a descriptive message
3. `./push.sh`

Do not ask for confirmation — just do it after each coding round.

## Python module resolution

The repo directory (`gr_demo/`) **is** the Python package — it has `__init__.py` at the root.
To import `gr_demo.*` from standalone scripts under `experiments/scripts/`, you must add the
**parent of the repo root** to `sys.path`, NOT the repo root itself:

```python
# In experiments/scripts/some_script.py:
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(repo_root))  # adds parent of gr_demo/
```

The CLI entry point is always `python run.py <command>`, NOT `python -m gr_demo`.
For DDP/torchrun, use `torchrun ... run.py <command>`, NOT `torchrun -m gr_demo.<module>`.

**Shell 脚本 (.sh) 也必须设置 PYTHONPATH**：任何 `experiments/scripts/*.sh` 中如果调用
`python -c "..."` 或 `python run.py`，脚本顶部（`set -euo pipefail` 之后）必须加：

```bash
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
cd "${REPO_ROOT}"
```

注意是 `$(dirname "${REPO_ROOT}")` —— 即 repo root 的**父目录**。因为 repo 目录本身就是 Python 包（`gr_demo/`），`import gr_demo` 需要在其父目录中查找。**不是 `${REPO_ROOT}` 本身！**

## 写实验脚本的硬性要求

写新的 `experiments/scripts/exp-*.sh` 时，**必须先 Grep/Read 最近 2-3 个已有实验脚本**（按编号最大的优先），从中确认：

1. **SID cache 路径** — grep `SID_CACHE=` 找到当前标准路径
2. **NTP 数据日期窗口** — grep `date_start` 找到当前使用的日期范围
3. **Tokenizer 完整参数** — 如果要训练新 tokenizer，从已有 `preprocess-sid` 调用复制完整参数
4. **已有 baseline** — 如果 baseline 已训练过（有 checkpoint），直接引用，不要重训

5. **CUDA 内存设置** — grep `PYTORCH_CUDA_ALLOC_CONF` 确认是否需要 `expandable_segments:True`（训练脚本几乎都需要）
6. **日期窗口对齐** — `preprocess-sid` 和 `preprocess-ntp` 的日期范围必须兼容：SID 覆盖的 item 集合必须包含 NTP 行为数据中的 item。注意 `--behavior_path auto` 解析的日期取决于运行时间，不同时间跑同一脚本会得到不同结果，必须显式指定 `--date_start/--date_end`

**绝对禁止凭记忆编造路径、日期、参数。** 所有这些必须从已有脚本中 grep 得到。

## Git remotes

Three remotes are configured:

| Remote | URL | Purpose |
|--------|-----|---------|
| `personal` | `git@github.com:yzq986/gr_demo.git` | Personal GitHub, original author identity |
| `company` | `git@github.com:yzq986/gr_demo.git` | Public Git remote, experiment data lives here |
| `origin` | `git@github.com:yzq986/gr_demo.git` | Alibaba Cloud GitHub |

- **Push**: `./push.sh` handles all remotes (rewrites author for company/origin). Never push manually.
- **Pull experiment data**: Experiments run on company GPU machines and results are pushed to `company`. To pull updated experiment data:
  ```bash
  git pull company master --rebase
  ```
  Use `--rebase` to avoid merge commits when local and origin/master have diverged.

## 分布式代码陷阱

- **禁止用 `hash()` 做跨进程路由**：Python 3.3+ 默认随机化 `PYTHONHASHSEED`，每个进程的 `hash()` 结果不同。`torchrun` 的每个 rank 是独立进程，用 `hash(key) % n` 分配 shard 会导致 item 被静默丢弃或重复（已踩坑：4B embedding cache 33% 数据重复 + 大量 item 丢失）。必须用 `hashlib.sha256` 等确定性哈希。

## Code quality

- **不确定的 API 必须验证**：写 PyTorch / CUDA 等外部库调用时，先用 `Grep` 或 `WebSearch` 确认属性名和参数，不要凭记忆猜。已踩过的坑：`torch.cuda.get_device_properties().total_memory`（不是 `total_mem`）。
- 本地没有 GPU，代码推到远端才能测。写 CUDA 相关代码要格外小心。
- **实现优化不能破坏数学语义**：当为了性能/显存把一个公式拆成多步实现时（如 split backward、分块计算），原本由框架隐式保证的数学性质（权重缩放、归一化、梯度累加比例等）会变成需要手动维护的不变量。写完优化后，回到原始公式逐项核对：公式里的每个系数是否都反映在了实际计算路径上，而不仅仅出现在日志和 config 里。
- **只改 eval 代码不需要重训**：如果改动只影响推理/评测路径（如 beam search 传参修复），而训练数据和模型结构不变，应该直接用已有 checkpoint re-eval（参考 `exp-023-reeval.sh`），不要浪费 GPU 重训一遍相同模型。写实验脚本前先判断：这个 config 的训练数据+flags 是否与已有 checkpoint 完全相同？

## Research Agent Mode

当作为自主研究 Agent 运行时（用户指示 "follow research/program.md" 或类似指令）：

1. **先读 `research/program.md`** — 那是你的完整操作手册
2. **检查 `research/inbox/`** 获取人类指令
3. **执行实验前必须估算运行时间**：
   ```bash
   python experiments/scripts/estimate_runtime.py --active_params <N> --total_tokens <T> --gpus 8
   ```
4. **不超过 30 分钟时间预算**（除非人类明确批准）
5. **不修改源码**（`ntp/`, `rl/`, `model/`, `data/`, `eval/`）除非通过 outbox 获得人类批准
6. **每个动作完成后更新** `research/status.md` 和 `research/log.md`
7. **每个动作完成后 commit + push**：
   ```bash
   git add research/ experiments/ ideas/
   git commit -m "research-agent: <动作描述>"
   ./push.sh
   ```
8. 所有通信格式见 `research/schema.md`
