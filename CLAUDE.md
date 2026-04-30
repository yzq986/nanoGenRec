# Project Instructions

## Auto commit & push

Every time a coding task finishes (implementation complete, no more pending changes), automatically:
1. `git add` the changed files
2. `git commit` with a descriptive message
3. `./push.sh`

Do not ask for confirmation — just do it after each coding round.

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

## 新实验标准流程（EXP-047 起统一使用）

**所有新实验走 `run_exp.py` + YAML，不再写 `.sh` 训练脚本。**

### 1. 创建 YAML config

新建 `experiments/configs/exp-NNN.yaml`，**必须先 Read `experiments/configs/_base.yaml`** 确认 defaults，然后只写需要覆盖的参数：

```yaml
name: exp047
description: "..."
base: _base.yaml

# 覆盖 base defaults：
sid_cache_name: exp026-0.6b-14d     # grep 已有 yaml 确认当前标准
ntp_data_name: exp026-0.6b-14d      # 若复用其他实验数据，显式指定
use_segment_emb: true

variants:
  - name: exp047-a
    torope_time_split: 0.25
  - name: exp047-b
    torope_time_split: 0.50
    torope_layer_split: 0.15
```

多 variant 对比实验用 `variants:` 列表；单 config 实验省略 `variants:`。

### 2. 必须确认的参数（写 YAML 前 grep）

**禁止凭记忆编造，必须从已有 yaml 中 grep 得到：**

1. **`sid_cache_name`** — grep `sid_cache_name:` in `experiments/configs/`
2. **`ntp_data_name`** — 复用已有数据时，grep 对应 exp 脚本确认目录名
3. **`date_start` / `date_end`** — 如需新 preprocess，grep 最近 yaml 确认日期范围
4. **已有 baseline** — `--check` 会自动显示相似实验，直接复用，不要重训

### 3. 运行

```bash
# 检查：显示每个 variant 的相似历史实验（防重训）
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --check

# 运行所有 variants（自动 full eval + registry 注册）
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --no-smoke --commit

# 只跑某个 variant（断点续跑）
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --only exp047-a --no-smoke
```

### 4. 加入队列（后台跑）

```bash
# queue.txt 格式：SCRIPT  LOG  DONE_STRING
echo "run_config.sh experiments/configs/exp-NNN.yaml  /tmp/expNNN.log  exp-NNN complete!" >> experiments/queue.txt
```

`run_config.sh` 是通用 wrapper，内部调用 `run_exp.py --no-smoke --commit`。

### 5. 需要新 preprocess 时

日期窗口对齐规则不变：`preprocess-sid` 和 `preprocess-ntp` 的日期范围必须兼容，SID 覆盖的 item 集合必须包含 NTP 行为数据中的 item。必须显式指定 `--date_start/--date_end`，不要用 `auto`。

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

## FSQ Codebook 质量评估规则（EXP-045，2026-04-30）

### ⚠️ EXP-045 num_clusters bug — 所有 exp045-* 数据不可信

EXP-045 新跑的所有 preprocess-sid 使用了 `num_clusters=1024`（默认值），而正确值应为 `num_clusters=4096`（exp026 使用）。KMeans 欠约束导致 Gini_d2 从 0.33（exp026）飙升到 0.54（exp045），collision 数值也不可与 exp026 比较。**需用 `--num_clusters 4096` 重跑 EXP-045 全部数据点后才能得出可信结论。**

### 两个 Proxy Metrics（FORGE 2025）

评估 codebook 质量，不需要跑 NTP：

1. **Collision Rate (CR)**：有多少 item 共享同一 SID（`1 - N_unique / N_items`）。越低越好，但与 Gini_d3 基本等价。

2. **Gini_d2**（推荐）：L1+L2 prefix 分布的 Gini 系数，衡量 KMeans 层间负载均衡。**比 CR 更有信息量**——codebook 容量（4096³）远大于 item 数（1.1M）时，CR 趋近于 0 但 Gini_d2 仍能区分 num_clusters 设置好坏。Gini_d2 越低 = L2 层预测难度越均匀 = NTP 性能越好。

   计算：`python -c "import sys; ..." experiments/sid_cache/<name>`（见 `metrics/cluster_balance.py`）

### 当前各 SID Cache 对比（2026-04-30 实测）

| Cache | CR | Gini_d2 | num_clusters | 可信 |
|-------|-----|---------|--------------|------|
| **exp026-0.6b-14d** | **0.49%** | **0.33** | **4096** | ✅ 基准 |
| exp026-4b-14d | 2.76% | 0.35 | 4096 | ✅ |
| exp026-8b-14d | 5.44% | 0.37 | 4096 | ✅ |
| exp045-0.6b-h128 | 1.25% | 0.54 | 1024 | ❌ bug |
| exp045-0.6b-h64 | 2.21% | 0.54 | 1024 | ❌ bug |
| exp045-4b-h512 | 3.13% | 0.57 | 1024 | ❌ bug |

### 当前推荐

| Embedding | 推荐 SID cache | CR | Gini_d2 |
|-----------|--------------|-----|---------|
| Qwen3-0.6B | exp026-0.6b-14d（h=64，num_clusters=4096） | 0.49% | 0.33 |
| Qwen3-4B | exp026-4b-14d（h=64，num_clusters=4096） | 2.76% | 0.35 |

**h sweep 结论（EXP-045 bug 修复前暂定）**：0.6b 最优 h 约为 128（CR 拐点），4b CR 对 h 不敏感（FSQ levels 瓶颈，需增大 levels 而非 h）。

## Code quality

- **不确定的 API 必须验证**：写 PyTorch / CUDA 等外部库调用时，先用 `Grep` 或 `WebSearch` 确认属性名和参数，不要凭记忆猜。已踩过的坑：`torch.cuda.get_device_properties().total_memory`（不是 `total_mem`）。
- 本机有 8×L20X GPU，使用 `/home/dev/.conda/envs/gr` 环境直接运行。写 CUDA 相关代码要格外小心。
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
- 每次新实验 checkpoint 跑完，必须用上述命令补全量 eval，再更新以下三处文档：
  1. `experiments/logs/<phase>/exp-NNN.md` — 单实验详细记录（Results/Analysis）
  2. `experiments/logs/<phase>/README.md` — 阶段汇总（更新 SOTA 行 + 在列表加一行）
  3. `README.md`（根目录）— homepage，更新"当前阶段"表格的 SOTA 列

  `<phase>` = `tokenizer`（EXP-001~012,026,045）/ `ntp`（EXP-013~016,036,041+）/ `rl`（EXP-017~040）
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
# 当前有实验在跑，想排队下一个（新流程）：
echo "run_config.sh experiments/configs/exp-NNN.yaml  /tmp/expNNN.log  exp-NNN complete!" >> experiments/queue.txt
# cron 检测到队列新条目，上一个完成后自动启动
```

**首次启动（队列为空时）：**
```bash
# 1. 启动第一个实验
nohup bash experiments/scripts/run_config.sh experiments/configs/exp-NNN.yaml > /tmp/expNNN.log 2>&1 &

# 2. 写入 queue_state.json
cat > experiments/queue_state.json <<EOF
{
  "current": "run_config.sh",
  "log": "/tmp/expNNN.log",
  "done_string": "exp-NNN complete!",
  "status": "running",
  "pid": $!
}
EOF

# 3. 确认守护 cron 存在（CronList 检查），没有就创建一个（见下方 prompt）
```

**queue.txt 格式：**
```
# 注释行忽略（新流程，script = run_config.sh + yaml path）
run_config.sh experiments/configs/exp-NNN.yaml  /tmp/expNNN.log  exp-NNN complete!
run_config.sh experiments/configs/exp-MMM.yaml  /tmp/expMMM.log  exp-MMM complete!
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
   b. 读 train_meta.json，更新以下三处：`experiments/logs/<phase>/exp-NNN.md` Results/Analysis、`experiments/logs/<phase>/README.md` SOTA 行、`README.md` 根目录 homepage
   c. git add experiments/ && git commit -m "EXP-XXX complete: ..." && ./push.sh
   d. 读 queue.txt，找下一个未完成的实验，nohup 启动，更新 queue_state.json
   e. 如队列已空：state 改为 done，告知用户，保持 cron 存活（等新追加）
```

**不要用脚本 chain（A 末尾调用 B）来排队实验**，因为前序实验可能已经在后台跑了。
