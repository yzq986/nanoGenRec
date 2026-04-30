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
| `architecture_roadmap.md` | S/M/L/XL tier 规格 + scaling law |

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

## Side Features

| Feature | 注入方式 | 实现 |
|---------|---------|------|
| `segment_emb` | embed_add | 用户行为分段 embedding |
| `time_gap` | embed_add | 时间间隔 bucket embedding |
| `action_level` | embed_add | 行为强度 embedding |
| `timestamps` | TO-RoPE | 连续小时值，注入 Q/K 旋转矩阵 |

**⚠️ 全链路必检**：训练加了 feature，eval 和 beam search 必须同样注入，否则结果无效。见 CLAUDE.md。

## 训练命令

```bash
# 预处理
python run.py preprocess-ntp \
    --sid_cache experiments/sid_cache/exp026-0.6b-14d \
    --date_start 2026-03-18 --date_end 2026-03-31

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
