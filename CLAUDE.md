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
- **Pull experiment data**: GPU-side results land on both mirrors; pull both to stay in sync:
  ```bash
  git pull company master --rebase
  git pull origin master --rebase
  ```
  Use `--rebase` to avoid merge commits when local and remote have diverged.

## 分布式代码陷阱

- **禁止用 `hash()` 做跨进程路由**：Python 3.3+ 默认随机化 `PYTHONHASHSEED`，每个进程的 `hash()` 结果不同。`torchrun` 的每个 rank 是独立进程，用 `hash(key) % n` 分配 shard 会导致 item 被静默丢弃或重复（已踩坑：4B embedding cache 33% 数据重复 + 大量 item 丢失）。必须用 `hashlib.sha256` 等确定性哈希。

## 设计实验前必须确认数据 pipeline

**在设计使用新特征的实验前，必须先确认 data pipeline 是否已包含该特征。**

检查顺序：
1. `ntp/preprocess.py`：`save_shard` / `load_shard` 是否存储 / 读取该特征？
2. `ntp/train.py`：`build_unified_sequences` 是否填充该特征？`UnifiedSequenceDataset.side_features_lists` 是否传入？
3. `ntp/model.py`：模型是否会用到该特征（`embed_with_features` / `_transformer_forward`）？

如果任意一环缺失，必须**先接通 pipeline 或明确向用户说明"该特征当前传 0"**，再开始实验。
绝对不能在实验跑完后发现特征全程为 0（等价于没加特征），却把结果当作有效对比。

已踩坑：EXP-044 TO-RoPE 实验，timestamps 在代码里传了但全为 0（pipeline 未接通），
导致 TO-RoPE vs baseline 的对比无效（两个都没有用真实时间戳）。

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

模型支持两类 side features：`time_gaps`（时间间隔 bucket）和 `action_levels`（行为级别）。
TO-RoPE 的 `timestamps`（连续实数小时）走单独路径（不加到 embedding，而是传给 attention 的 RoPE 时间分量）。
**所有路径（训练和推理）必须通过同一个入口注入特征**，否则会产生 train-infer 不一致。

### ⚠️ Train-Infer 不一致：加特征必须全链路验证，否则实验无效

**这是最高优先级的检查。训练时用了某个特征，eval 时必须同样注入，否则模型在两种完全不同的条件下运行，结果毫无意义，等于白费 GPU。**

已踩坑（代价：多次无效实验）：
- **EXP-023/024**：`time_gaps`/`action_levels` 训练时有，beam search incremental 步骤没传 → R@500 崩溃。修复：EXP-025。
- **EXP-044B（两层 bug，排查耗时数天）**：
  - Bug 1：`constrained_beam_search` 的生成步骤没传 `step_timestamp`，timestamps=0。
  - Bug 2（更隐蔽）：`eval.py` 的 `eval_items` 构建循环 `if fdef.inject != 'embed_add': continue`，把 `inject='torope'` 的 timestamps 直接过滤掉，`ctx_side_features` 里永远没有 timestamps，carry-forward 根本没有机会执行。结果仍然 32%，误判 TO-RoPE 无效。
  - 正确结果（修复后）：R@500=63.6%，比 baseline +2.4pp。

**全链路检查清单（每次新增特征必做）**：

| 环节 | 检查点 | 常见漏洞 |
|------|--------|---------|
| **Preprocess** | shard 文件里该特征是否存在且非零？ | pipeline 未接通，全为 0 |
| **Train sanity** | `[sanity]` log 打印特征样本值是否正常？ | 看到全 0 立即停止 |
| **eval_items 构建** | `eval.py` 的循环是否把该特征放进 `ctx_side_features`？ | inject 类型过滤把特征漏掉 |
| **beam search ctx** | `constrained_beam_search` 调用时 `ctx_side_features` / `ctx_timestamps` 是否传了？ | 变量存在但未传入 |
| **beam search gen** | 生成步骤（`_step_sf` / `_step_ts`）是否覆盖了该特征的 inject 路径？ | 只处理了 embed_add，漏了 torope |

**快速验证方法**：在 eval log 里加 `[sanity] eval timestamps[:3]` 打印，确认非零。如果是 0，100% 是 bug，不要继续跑。

### Side Features 统一用 dict 传递

所有 API 接受 `side_features: dict[str, Tensor]`，key 为特征名：
- `"time_gaps"` — `(B,T)` long，bucket embedding（加到 token embedding）
- `"action_levels"` — `(B,T)` long，action level embedding（加到 token embedding）
- `"timestamps"` — `(B,T)` float，连续小时（只在 `_forward_packed` / TO-RoPE 路径读取）

训练数据侧：`side_features_lists: dict[str, list[list]]`，同名 key。

### 唯一入口：`model.embed_with_features`

```python
# ntp/model.py
def embed_with_features(self, tokens, positions, side_features=None):
    """Single source of truth for input embedding + side-feature injection."""
    x = self._embed_tokens(tokens) + self._get_pos_emb(positions)
    sf = side_features or {}
    if 'time_gaps' in sf and sf['time_gaps'] is not None and hasattr(self, 'time_gap_emb'):
        x = x + self.time_gap_emb(sf['time_gaps'])
    if 'action_levels' in sf and sf['action_levels'] is not None and hasattr(self, 'action_emb'):
        x = x + self.action_emb(sf['action_levels'])
    return x
```

**所有调用方必须使用此方法，禁止手动拼 `_embed_tokens + time_gap_emb`**：

| 路径 | 文件 | 调用方式 |
|------|------|---------|
| NTP 训练 | `ntp/model.py:_forward_packed` | `model.forward(..., side_features=sf)` |
| KV-cached 推理 | `ntp/model.py:forward_cached` | `forward_cached(..., ctx_side_features=sf)` |
| DPO/GRPO log-prob 计算 | `rl/dpo.py:compute_sid_logprobs` | `compute_sid_logprobs(..., ctx_side_features=sf, gen_side_features=sf)` |
| Beam search | `ntp/model.py:constrained_beam_search` | `constrained_beam_search(..., ctx_side_features=sf, gen_side_features=sf)` |

### 新增 Side Feature 的标准步骤

1. 在 `NTPModel.__init__` 里加 `nn.Embedding`（条件创建，不破坏无特征模型）
2. 在 `embed_with_features` 里加一行 `if 'xxx' in sf and hasattr(self, 'xxx_emb')`
3. 训练侧：在 `ntp/preprocess.py` 的 `save_shard`/`load_shard` 里增加存储/读取
4. 训练侧：在 `ntp/train.py` 的 `build_unified_sequences` 里填充新字段，在 `main()` 里加入 `side_features_lists` dict
5. 推理侧：在 `ntp/eval.py` 的 eval item 构造和 `constrained_beam_search` 调用处加入 key

**不需要改 `_embed_tokens`、`_transformer_forward`、或任何下游 loss 函数。**

### Context Pool 结构（rl/trainer.py）

```python
# rl/trainer.py — context_pool entry
(ctx_tokens: List[int], ctx_side_features: dict[str, list])
```

`_grpo_step` 对每个 context 做 carry-forward：`gen_side_features['action_levels'] = ctx_sf['action_levels'][-1]`，`gen_side_features['time_gaps'] = 0`（目标 item 时间间隔未知时默认 0）。

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

### 标准方式：队列文件 + 单个守护 Cron

实验队列由两个文件管理，**无需修改 cron prompt 或重启任何东西**：

- `experiments/queue.txt` — 实验队列，追加即生效
- `experiments/queue_state.json` — 当前状态，cron 读这个决定做什么

**启动新实验 / 追加队列：**
```bash
# 当前有实验在跑，想排队下一个：
echo "exp-042.sh  /tmp/exp042.log  EXP-042 complete!" >> experiments/queue.txt
# cron 检测到队列新条目，上一个完成后自动启动
```

**首次启动（队列为空时）：**
```bash
# 1. 启动第一个实验
nohup bash experiments/scripts/exp-NNN.sh --no-smoke > /tmp/expNNN.log 2>&1 &

# 2. 写入 queue_state.json
cat > experiments/queue_state.json <<EOF
{
  "current": "exp-NNN.sh",
  "log": "/tmp/expNNN.log",
  "done_string": "EXP-NNN complete!",
  "status": "running",
  "pid": $!
}
EOF

# 3. 确认守护 cron 存在（CronList 检查），没有就创建一个（见下方 prompt）
```

**queue.txt 格式：**
```
# 注释行忽略
exp-038b.sh  /tmp/exp038b.log  EXP-038B complete!  EVAL_MID_CHECKPOINTS=exp038b-hard-lam03-3ep
exp-039b.sh  /tmp/exp039b.log  EXP-039B complete!
exp-040.sh   /tmp/exp040.log   EXP-040 complete!
```
第4列 POST_HOOK 可选，支持 `EVAL_MID_CHECKPOINTS=NAME`（自动 eval ep1/ep2 中间 checkpoint 并选最优）。

**守护 Cron 参数：间隔 2 分钟，durable=true。**

**守护 Cron prompt（每个 session 只需一个，job ID 记录在 queue_state.json）：**
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

**效果预警 — 告知用户但不自动停止（由用户决定是否继续）：**
- reward_mean 在 100 steps 后仍 < 0.1

（后续实验可在此追加新的早停/预警条件）

5. 完成：
   a. 执行 post_hook（如 EVAL_MID_CHECKPOINTS：串行 eval ep1/ep2，找最优 checkpoint）
   b. 读 train_meta.json，更新 experiments/log.md 对应 EXP 的 Results/Analysis
   c. git add experiments/ && git commit -m "EXP-XXX complete: ..." && ./push.sh
   d. 读 queue.txt，找下一个未完成的实验，nohup 启动，更新 queue_state.json
   e. 如队列已空：state 改为 done，告知用户，保持 cron 存活（等新追加）
```

**不要用脚本 chain（A 末尾调用 B）来排队实验**，因为前序实验可能已经在后台跑了。
