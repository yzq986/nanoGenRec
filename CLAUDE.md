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

写新的 `experiments/scripts/exp-*.sh` 时，**必须先用 Grep/Read 查看最近 2-3 个已有实验脚本**，确认以下信息后再写：

1. **SID cache 路径** — 当前标准路径是 `experiments/sid_cache/exp013-4096x3-12d-binary`（0.6B, 4096×3, FSQ 12d_4096 binary）
2. **NTP 数据窗口** — 当前最优是 14d (2026-03-18 ~ 2026-03-31)，embedding cache 覆盖到 03-31
3. **Tokenizer 参数** — 如果要训练新 tokenizer，必须从已有脚本复制完整参数（num_clusters, fsq_levels, fsq_projection, fsq_mlp_hidden, fsq_epochs）
4. **已有 baseline** — 如果 baseline 已训练过（有 checkpoint），直接引用，不要重训

**绝对禁止凭记忆编造路径、日期、参数。** 所有这些必须从已有脚本或 log.md 中复制。

## Git remotes

Two remotes are configured:

| Remote | URL | Purpose |
|--------|-----|---------|
| `personal` | `git@github.com:yzq986/gr_demo.git` | Personal GitHub, original author identity |
| `company` | `git@github.com:yzq986/gr_demo.git` | Public Git remote, experiment data lives here |

- **Push**: `./push.sh` handles both remotes (rewrites author for company). Never push manually.
- **Pull experiment data**: Experiments run on company GPU machines and results are pushed to `company`. To pull updated experiment data:
  ```bash
  git pull company master --rebase
  ```
  Use `--rebase` to avoid merge commits when local and origin/master have diverged.

## Code quality

- **不确定的 API 必须验证**：写 PyTorch / CUDA 等外部库调用时，先用 `Grep` 或 `WebSearch` 确认属性名和参数，不要凭记忆猜。已踩过的坑：`torch.cuda.get_device_properties().total_memory`（不是 `total_mem`）。
- 本地没有 GPU，代码推到远端才能测。写 CUDA 相关代码要格外小心。
- **实现优化不能破坏数学语义**：当为了性能/显存把一个公式拆成多步实现时（如 split backward、分块计算），原本由框架隐式保证的数学性质（权重缩放、归一化、梯度累加比例等）会变成需要手动维护的不变量。写完优化后，回到原始公式逐项核对：公式里的每个系数是否都反映在了实际计算路径上，而不仅仅出现在日志和 config 里。
