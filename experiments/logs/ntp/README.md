# NTP 实验

Transformer + MoE 序列模型，输入用户行为序列，预测下一个 item 的 SID（3-token）。

## 当前最优

| 配置 | R@10 | R@500 | PPL | 来源 |
|------|------|-------|-----|------|
| **M-tier bare** (71.6M active) | 14.5% | **70.2%** | 18.54 | EXP-043 |
| M-tier + 4B SID | 14.2% | 70.4% | 16.55 | EXP-043 |
| L-tier + all opts (101M active) | 12.8% | 64.1% | 20.7 | EXP-047 |
| S-tier + TO-RoPE ts=0.5 | 11.8% | 63.9% | 22.7 | EXP-044C |
| S-tier bare | 11.4% | 61.2% | 26.52 | EXP-043 |

**RL 链路 SFT 起点**：exp047（L-tier, R@500=64.1%）

## 模型规格

| Tier | embed_dim | Layers | Experts | top_k | Active Params |
|------|-----------|--------|---------|-------|---------------|
| S-tier (scale-05) | 256 | 6 | 8 | 2 | ~17.5M |
| M-tier (scale-06) | 512 | 8 | 8 | 2 | ~71.6M |
| L-tier (scale-07) | 512 | 12 | 16 | 2 | ~101.1M |

## 已验证优化（S-tier 基准）

| 优化 | R@500 增益 | 实验 |
|------|-----------|------|
| segment_emb + time_gap + action_level | +3.7pp | EXP-036 |
| TO-RoPE ts=0.5（真实 timestamps） | +2.4pp | EXP-044B |
| gate_attn | +0.4pp | EXP-046 |
| TO-RoPE 3-dim (layer:0.1) | -0.7pp（有害） | EXP-048 |

⚠️ **TO-RoPE 在 M-tier 无收益**（EXP-048）：2-dim 持平（-0.1pp），3-dim -0.7pp。M-tier 容量充足，RoPE 边际效益消失。

## 实验列表

| EXP | Date | Status | 结论 |
|-----|------|--------|------|
| [013](../exp-013.md) | 2026-04-15 | completed | S-tier NTP baseline — 6L MoE |
| [014](../exp-014.md) | 2026-04-16 | completed | ENTP-Loss 消融 |
| [015](../exp-015.md) | 2026-04-16 | completed | NTP Scaling Law — 1M→100M active |
| [016](../exp-016.md) | 2026-04-17 | completed | Data Scaling Law (Chinchilla) |
| [036](../exp-036.md) | 2026-04-28 | completed | **Features NTP — time_gap + action_level + segment_emb** |
| [041](../exp-041.md) | 2026-04-29 | completed | ENTP-Loss v1（无效） |
| [041B](../exp-041b.md) | 2026-04-29 | completed | ENTP-Loss v2（无效，session 粒度） |
| [043](../exp-043.md) | 2026-04-29 | completed | **Embedding size × NTP tier 对比；M-tier 确立** |
| [044](../exp-044.md) | 2026-04-29 | completed | TO-RoPE vs APE（timestamps=0，无效对比） |
| [044B](../exp-044b.md) | 2026-04-29 | completed | **TO-RoPE 真实 timestamps — S-tier +2.4pp** |
| [044C](../exp-044c.md) | 2026-04-29 | completed | TO-RoPE Item-Pos Fix + 3-dim |
| [046](../exp-046.md) | 2026-04-29 | completed | GateAttention — +0.4pp |
| [047](../exp-047.md) | 2026-04-30 | completed | **L-tier + all opts — SFT 起点 R@500=64.1%** |
| [048](../exp-048.md) | 2026-04-30 | completed | M-tier TO-RoPE 2-dim vs 3-dim — 无收益 |
| [050](../exp-050.md) | 2026-04-30 | 🔄 queued | M-tier NTP：0.6b/4b SID × output-gate/CADET + bare+RoPE（6 variants，exp049 SID） |
