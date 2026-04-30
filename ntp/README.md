# ntp/ — Next Token Prediction 模型

Transformer + MoE 自回归模型，在 SID 序列上做 Next Token Prediction。

## 文件

| 文件 | 说明 |
|------|------|
| `model.py` | NTPModel — MoE Transformer，支持 S/M/L-tier，TO-RoPE，side features |
| `train.py` | DDP 训练入口，unified sequences，joint NTP+DPO loss |
| `eval.py` | eval-only，beam search，constrained decoding via SIDTrie |
| `preprocess.py` | 数据预处理：行为序列 → SID token 序列，保存分片 |
| `features.py` | Side features：time_gap bucket、action_level embedding |
| `baseline.py` | 非神经网络 baseline（popularity、co-occurrence） |

## 模型规格

| Tier | embed_dim | Layers | Experts | top_k | Active Params | 当前状态 |
|------|-----------|--------|---------|-------|---------------|---------|
| S-tier | 256 | 6 | 8 | 2 | ~17.5M | ✅ 已验证 |
| M-tier | 512 | 8 | 8 | 2 | ~71.6M | ✅ 已验证，R@500=70.2% |
| L-tier | 512 | 12 | 16 | 2 | ~101.1M | ✅ 已验证，RL 起点 |

## 当前最优（全量 eval，n_recall=1000）

| 配置 | R@500 | PPL | 来源 |
|------|-------|-----|------|
| M-tier bare (0.6B SID) | **70.2%** | 18.54 | EXP-043 |
| M-tier + 4B SID | 70.4% | 16.55 | EXP-043 |
| L-tier + all opts | 64.1% | 20.7 | EXP-047 ← RL 起点 |

## Side Features 注入架构

模型支持三类 side features：

| Feature | 注入方式 | 实现 |
|---------|---------|------|
| `time_gaps` | embed_add | 时间间隔 bucket embedding，加到 token embedding |
| `action_levels` | embed_add | 行为强度 embedding，加到 token embedding |
| `timestamps` | TO-RoPE | 连续实数小时，注入 Q/K 旋转矩阵（不加到 embedding）|
| `segment_emb` | embed_add | 用户行为分段 embedding |

所有 API 接受 `side_features: dict[str, Tensor]`，同名 key。训练数据侧：`side_features_lists: dict[str, list[list]]`。

### 唯一入口：`model.embed_with_features`

```python
def embed_with_features(self, tokens, positions, side_features=None):
    x = self._embed_tokens(tokens) + self._get_pos_emb(positions)
    sf = side_features or {}
    if 'time_gaps' in sf and sf['time_gaps'] is not None and hasattr(self, 'time_gap_emb'):
        x = x + self.time_gap_emb(sf['time_gaps'])
    if 'action_levels' in sf and sf['action_levels'] is not None and hasattr(self, 'action_emb'):
        x = x + self.action_emb(sf['action_levels'])
    return x
```

**禁止手动拼 `_embed_tokens + time_gap_emb`**，所有调用方走此入口：

| 路径 | 文件 |
|------|------|
| NTP 训练 | `ntp/model.py:_forward_packed` |
| KV-cached 推理 | `ntp/model.py:forward_cached` |
| DPO/GRPO log-prob | `rl/dpo.py:compute_sid_logprobs` |
| Beam search | `ntp/model.py:constrained_beam_search` |

### ⚠️ Train-Infer 不一致是最高优先级 bug

训练时用了某个特征，eval 时必须同样注入，否则结果无效。

**全链路检查清单（每次新增特征必做）**：

| 环节 | 检查点 | 常见漏洞 |
|------|--------|---------|
| Preprocess | shard 文件里该特征是否存在且非零？ | pipeline 未接通，全为 0 |
| Train sanity | `[sanity]` log 打印特征样本值是否正常？ | 看到全 0 立即停止 |
| eval_items 构建 | `eval.py` 的循环是否把该特征放进 `ctx_side_features`？ | inject 类型过滤把特征漏掉 |
| beam search ctx | `constrained_beam_search` 调用时 `ctx_side_features` 是否传了？ | 变量存在但未传入 |
| beam search gen | 生成步骤是否覆盖了该特征的 inject 路径？ | 只处理了 embed_add，漏了 torope |

**快速验证**：eval log 里加 `[sanity] eval timestamps[:3]` 打印，如果是 0，100% 是 bug。

**已踩坑**：
- EXP-023/024：beam search incremental 步骤没传 `time_gaps`/`action_levels` → R@500 崩溃。修复：EXP-025。
- EXP-044B Bug 1：`constrained_beam_search` 生成步骤没传 `step_timestamp`，timestamps=0。
- EXP-044B Bug 2（更隐蔽）：`eval.py` 的 `eval_items` 构建循环 `if fdef.inject != 'embed_add': continue`，把 `inject='torope'` 的 timestamps 过滤掉，carry-forward 根本无法执行。结果仍 32%，误判 TO-RoPE 无效。正确结果（修复后）：R@500=63.6%。

### 新增 Side Feature 标准步骤

1. `NTPModel.__init__`：加条件 `nn.Embedding`（不破坏无特征模型）
2. `embed_with_features`：加一行 `if 'xxx' in sf and hasattr(self, 'xxx_emb')`
3. `ntp/preprocess.py`：`save_shard`/`load_shard` 增加存储/读取
4. `ntp/train.py`：`build_unified_sequences` 填充新字段，`main()` 加入 `side_features_lists` dict
5. `ntp/eval.py`：eval item 构造和 `constrained_beam_search` 调用处加入 key

### Context Pool 结构（rl/trainer.py）

```python
# context_pool entry: (ctx_tokens: List[int], ctx_side_features: dict[str, list])
# _grpo_step carry-forward:
gen_side_features['action_levels'] = ctx_sf['action_levels'][-1]
gen_side_features['time_gaps'] = 0  # 目标 item 时间间隔未知时默认 0
```

## 训练命令

```bash
# 预处理（单进程，多 worker，不需要 torchrun）
python run.py preprocess-ntp \
    --sid_cache experiments/sid_cache/exp049-0.6b-nc8192-h128 \
    --output_dir experiments/ntp_data/exp049-0.6b-nc8192-h128 \
    --date_start 2026-03-18 --date_end 2026-03-31 \
    --behavior_path /mnt/workspace/gr-demo-behavior-cache \
    --n_workers 64

# 训练（torchrun 多卡）
torchrun --nproc_per_node=8 run.py train-ntp \
    --config experiments/configs/exp-047.yaml

# 通过 run_exp.py（推荐，自动 eval + commit）
python experiments/run_exp.py experiments/configs/exp-047.yaml --no-smoke --commit

# 全量 eval（对齐 baseline）
torchrun --nproc_per_node=8 run.py eval-ntp \
    --checkpoint experiments/ntp_checkpoints/<name> \
    --n_recall 1000
```

## 实验记录

见 [`experiments/logs/ntp/README.md`](../experiments/logs/ntp/README.md)。
