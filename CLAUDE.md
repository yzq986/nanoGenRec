# Project Instructions

## Auto commit & push

Every time a coding task finishes (implementation complete, no more pending changes), automatically:
1. `git add` the changed files
2. `git commit` with a descriptive message
3. `./push.sh`

Do not ask for confirmation — just do it after each coding round.

## Python module resolution

The repo root (`gr-demo/`) is added directly to `sys.path`. All modules are imported without a
`gr_demo.` prefix — e.g. `from eval.batch import ...`, `from ntp.train import ...`.

For standalone scripts under `experiments/scripts/`:

```python
# In experiments/scripts/some_script.py:
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)  # adds gr-demo/ itself
```

The CLI entry point is always `python run.py <command>`, NOT `python -m gr_demo`.
For DDP/torchrun, use `torchrun ... run.py <command>`.

**Shell 脚本 (.sh) 也必须设置 PYTHONPATH**：任何 `experiments/scripts/*.sh` 中如果调用
`python -c "..."` 或 `python run.py`，脚本顶部（`set -euo pipefail` 之后）必须加：

```bash
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
```

注意是 `${REPO_ROOT}` 本身（即 `gr-demo/`），**不是父目录**。

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
- **禁止在新实验中重跑已有 config 作对照**：设计多 config 对比实验时，如果某个 config 的参数与已有 checkpoint 完全一致，**直接引用已有结果，不要重训**。在 log.md 里写 `参考 EXP-NNN：R@500=xx.x%` 即可。例：EXP-031B（exp020+full stack）≡ EXP-029，EXP-032A（G=512,b=4）≡ EXP-029，均为浪费。新实验只跑真正新的 config。

## Eval 对齐规则

**`train-ntp` 的 inline eval ≠ 全量 eval，不能与 baseline 直接比较。**

- `train-ntp` 训练结束后自动跑的 inline eval：beam search 仅 250 items/rank（1000 total），是快速健康检查，绝对数字不可信。
- 与 baseline 对齐必须用：
  ```bash
  torchrun --nproc_per_node=N run.py eval-ntp \
      --checkpoint experiments/ntp_checkpoints/<name> \
      --n_recall 1000
  ```
- **baseline 标准**（exp020-hard-lam03，4×L20X，n_recall=1000）：PPL=16.3，R@10=14.1%，R@500=66.2%
- 每次新实验 checkpoint 跑完，必须用上述命令补全量 eval，再更新 experiments/log.md 结论。
- `train_meta.json` 里的 eval keys 是 `item_recall@10` / `item_recall@500`（带 `@`，不是 `_`）。

## Side Features 注入架构

模型支持两类 side features：`time_gap`（时间间隔 bucket）和 `action_level`（行为级别）。
**所有路径（训练和推理）必须通过同一个入口注入特征**，否则会产生 train-infer 不一致。

### 唯一入口：`model.embed_with_features`

```python
# ntp/model.py
def embed_with_features(self, tokens, positions, time_gaps=None, action_levels=None):
    """Single source of truth for input embedding + side-feature injection."""
    x = self._embed_tokens(tokens) + self._get_pos_emb(positions)
    if time_gaps is not None and hasattr(self, 'time_gap_emb'):
        x = x + self.time_gap_emb(time_gaps)
    if action_levels is not None and hasattr(self, 'action_emb'):
        x = x + self.action_emb(action_levels)
    return x
```

**所有调用方必须使用此方法，禁止手动拼 `_embed_tokens + time_gap_emb`**：

| 路径 | 文件 | 调用方式 |
|------|------|---------|
| NTP 训练 | `ntp/model.py:_forward_packed` | `model.embed_with_features(tokens, positions, tg, al)` |
| KV-cached 推理 | `ntp/model.py:forward_cached` | `model.embed_with_features(tokens, positions, tg, al)` (cold start) |
| DPO/GRPO log-prob 计算 | `rl/dpo.py:compute_sid_logprobs` | `model.embed_with_features(full_input, positions, tg, al)` |
| Beam search | `ntp/model.py:constrained_beam_search` | 调用 `forward_cached`，透传 `ctx_time_gaps/ctx_action_levels` |

### 新增 Side Feature 的标准步骤

1. 在 `NTPModel.__init__` 里加 `nn.Embedding`（条件创建，不破坏无特征模型）
2. 在 `embed_with_features` 里加一行 `if xxx is not None and hasattr(self, 'xxx_emb')`
3. 在 `_forward_packed` 的调用处补参数（NTP 训练路径）
4. 在 `forward_cached` 的调用处补参数（推理路径）
5. 在 `compute_sid_logprobs` 的调用处补参数（DPO/GRPO 路径）
6. 在 `rl/trainer.py` 的 `context_pool` 存储结构里加字段，在 `_grpo_step` 里传递

**不需要改 `_embed_tokens`、`_transformer_forward`、或任何下游 loss 函数。**

### Context Pool 结构

```python
# rl/trainer.py — context_pool entry
(ctx_tokens: List[int], ctx_time_gaps: List[int]|None, ctx_action_levels: List[int]|None)
```

`_grpo_step` 对每个 context 做 carry-forward：`gen_action_level = ctx_al[-1]`（最后一个 context token 的 action_level），`gen_time_gap = 0`（目标 item 时间间隔未知时默认 0）。

## GRPO/ECPO 训练踩坑记录（EXP-026）

- **SIDTrie 构建**：`semantic_ids.npy` 存 `{item_id_str: sid_str}`，必须 iterate `.values()` 构建 trie，iterate `.keys()` 只得到 item id 字符串，trie 为空，beam search 返回 0 candidates，GRPO loss 永远 0。
- **BehaviorReward 覆盖率**：全 SID 精确匹配仅 ~0.16%（1788/1.09M）。必须加 prefix cascade fallback，L0 单层可覆盖 ~24%，有效 reward 信号提升 150x。
- **reward std≈0 → advantage 爆炸**：稀疏 reward 场景下，一组 candidates reward 全相同 → std≈0 → advantage 无穷大。必须加 `std < 1e-6` group skip + `adv.clamp(-5, 5)` + `log_rho.clamp(-10, 10)`。
- **step log reward metrics 不打印**：reward metrics 挂在 `_grpo_step` 触发时的 `log_entry`，但 50 步定期打印时 GRPO 不一定触发（2% 概率），导致 reward 数据永远不出现在打印行。正确做法：用累计 `reward_metric_totals / n_grpo_steps` 而非当步瞬时值。

## VL / Embedder 踩坑记录

- **`torch_dtype` 必须显式传**：`Qwen3TextEmbedder` 里默认了 `torch_dtype=torch.float16`，`Qwen3VLEmbedder` 里漏传 → HF fallback 到 **fp32**。2B 模型 fp32 权重 ~10GB + fp32 activations，8192 seq batch=8 就能吃满 40GB OOM。**写 embedder 包装时永远显式传 `torch_dtype` (fp16/bf16)**，不要依赖 HF 默认。
- **`output_hidden_states=True` 是显存放大器**：`Qwen3VLForEmbedding.forward` 里开了这个 flag 只为取 `hidden_states[-1]`，但 HF 会 materialize 所有 30+ 层的 hidden states（每层 `batch × seq × hidden × dtype_bytes`）。2B 模型 seq=8192 fp32 下这一项就是 20-30GB。如果只要最后一层，直接从 `outputs.last_hidden_state` 或 inner encoder 的默认输出拿，不要开全量 hidden_states。
- **OOM 诊断必须打印 `text_len + mem alloc/reserved/total`**：光看 "OOM at size=1" 定位不到是 (a) 显存一开始就被占满（dtype/权重问题）还是 (b) 某条超长样本。`memory_allocated == memory_reserved == 总量 97%` 是 dtype 问题的标志信号（不是碎片）。
- **OOM skip 路径里要 `del sub_inputs + gc.collect() + empty_cache() + synchronize()`**：只调 `empty_cache()` 释放不掉 Python 还持有引用的 GPU tensor，下一条几乎必然再 OOM。
- **VL 场景别复用 text LFU cache**：text 缓存 key 是文本，VL 场景下相同文本 + 不同图片 → embedding 不同，复用会出错。必须 `if not is_vl and ...` 分叉。

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

## 实验排队与 Cron 监控

**不要用脚本 chain（A 末尾调用 B）来排队实验**，因为前序实验可能已经在后台跑了。

正确做法：用 **CronCreate** 设置每分钟监控任务，在 cron prompt 里写清楚：
1. 检测完成信号（grep "EXP-NNN complete" log）
2. 收集结果（读 train_meta.json，更新 experiments/log.md）
3. 启动下一个实验（nohup bash exp-NNN.sh > /tmp/expNNN.log 2>&1 &）
4. 告知用户结果
5. CronDelete 自身

示例：
```
如果看到 "EXP-028 complete!"：
1. 收集结果并更新 log.md
2. nohup bash experiments/scripts/exp-029.sh > /tmp/exp029.log 2>&1 &
3. 告知用户
4. CronDelete 本 job
```
