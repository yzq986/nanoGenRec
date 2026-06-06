# ntp/

[English](README.md) | [中文](README.zh.md)

基于 Semantic ID 序列的 Next-Token Prediction 推荐器。

NTP 模块训练 Transformer + MoE 模型，读取编码为 SID token 的用户行为历史，并在受限 beam search 下生成下一个商品的 SID。

## 文件

| 文件 | 用途 |
|------|------|
| `model.py` | `NTPModel`、MoE Transformer、TO-RoPE、侧特征、KV 缓存推理。 |
| `train.py` | DDP 训练入口、统一序列、SFT 和联合 NTP+DPO 损失。 |
| `eval.py` | 仅评估工具、受限 beam search、SIDTrie 解码。 |
| `preprocess.py` | 将行为序列转换为 SID token 分片。 |
| `features.py` | 侧特征定义，如时间间隔、行为等级、时间戳和分段。 |
| `baseline.py` | 非神经基线，如流行度和共现。 |

## 模型分档

| 级别 | embed_dim | 层数 | 专家数 | top_k | 活跃参数 | 状态 |
|------|-----------|------|--------|-------|---------|------|
| S-tier | 256 | 6 | 8 | 2 | ~17.5M | 已验证 |
| M-tier | 512 | 8 | 8 | 2 | ~71.6M | 已验证, R@500=70.2% |
| L-tier | 512 | 12 | 16 | 2 | ~101.1M | 已验证, RL 起点 |

## 当前全量评估基线

| 配置 | R@500 | PPL | 来源 |
|------|-------|-----|------|
| M-tier bare, 0.6B SID | 70.2% | 18.54 | EXP-043 |
| M-tier, 4B SID | 70.4% | 16.55 | EXP-043 |
| L-tier 含已验证选项 | 64.1% | 20.7 | EXP-047 |

当前阶段总结和实验谱系见 [experiments/logs/ntp/README.md](../experiments/logs/ntp/README.md)。

## 侧特征

所有侧特征通过 `side_features: dict[str, Tensor]` 传递。

| 特征 | 注入方式 | 含义 |
|------|---------|------|
| `time_gaps` | embedding add | 事件间的分桶时间间隔。 |
| `action_levels` | embedding add | 行为强度等级。 |
| `timestamps` | TO-RoPE | 用于 Q/K 旋转的连续小时时间戳。 |
| `segment_emb` | embedding add | 用户行为分段标记。 |

唯一的嵌入入口点是 `NTPModel.embed_with_features`。不要在调用方手动重建 token embedding 加特征 embedding。

## 数据契约

`preprocess.py` 写入由 `train.py` 消费的分片。对于每个新特征，验证完整路径：

| 阶段 | 需检查 |
|------|--------|
| 预处理 | `save_shard` 和 `load_shard` 存储和恢复该特征。 |
| 序列构建 | `build_unified_sequences` 填充非零值。 |
| 训练 | `side_features_lists` 将 key 传入模型。 |
| 评估 | `eval.py` 将相同的 key 转发到 beam search。 |
| 生成 | `constrained_beam_search` 在生成步骤中携带该特征。 |

训练/评估特征不匹配会使比较无效。这是 NTP bug 中风险最高的类别。

