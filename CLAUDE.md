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
`python -c "..."` 或 `python run.py`，脚本顶部必须加：

```bash
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$(dirname "${REPO_ROOT}"):${PYTHONPATH:-}"
cd "${REPO_ROOT}"
```

这保证无论用户从哪个目录执行脚本，`gr_demo` 包都能被找到。**每次写新的 .sh 实验脚本时必须包含这三行。**

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
