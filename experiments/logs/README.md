# 实验记录总索引

实验按阶段分目录管理：

| 阶段 | 目录 | 实验范围 | SOTA |
|------|------|---------|------|
| **Tokenizer** | [tokenizer/](tokenizer/) | EXP-001~012, 026, 045 | 4096×3 binary, CR=0.49%, Gini_d2=0.33 |
| **NTP** | [ntp/](ntp/) | EXP-013~016, 036, 041~048 | M-tier R@500=70.2%；L-tier SFT=64.1% |
| **RL** | [rl/](rl/) | EXP-017~040 | ECPO R@500=65.7% (EXP-039B) |

---

## 新实验文档流程

每次实验完成后更新三处：

1. **`experiments/logs/<phase>/exp-NNN.md`** — 单实验详细记录（Background / Design / Results / Analysis）
2. **`experiments/logs/<phase>/README.md`** — 阶段汇总表（更新 SOTA 行 + 在实验列表加一行）
3. **`README.md`（根目录）** — homepage（更新"当前阶段"表格的 SOTA 列）

---

## 模板

```markdown
## EXP-NNN: 标题

**Date**: YYYY-MM-DD
**Status**: completed

### Background
(当前状态、要解决的问题)

### Design
- **Variable**: ...
- **Fixed**: ...
- **Baseline**: exp-XXX (R@500=XX%)

### Results

| Config | R@10 | R@500 | PPL |
|--------|------|-------|-----|
| baseline | | | |
| **new** | | | |

### Analysis
1. ...

### Next Steps
- ...
```

---

## 全量索引（按 EXP 编号）

| EXP | 阶段 | Date | Status | Title |
|-----|------|------|--------|-------|
| [001](./exp-001.md) | tokenizer | 2026-03 | completed | RKMeans 训练优化 v0→v7 |
| [002](./exp-002.md) | tokenizer | 2026-04-13 | completed | ResKmeansFSQ |
| [003](./exp-003.md) | tokenizer | 2026-04-13 | completed | Learned FSQ |
| [004](./exp-004.md) | tokenizer | 2026-04-13 | completed | OPQ Parallel Semantic IDs |
| [007](./exp-007.md) | tokenizer | 2026-04-13 | completed | Collaborative Signal Enhanced Embedding |
| [008](./exp-008.md) | tokenizer | 2026-04-14 | completed | FORGE Proxy 对比 — MLP-FSQ vs OPQ |
| [009](./exp-009.md) | tokenizer | 2026-04-14 | completed | QFormer Tokenizer |
| [010](./exp-010.md) | tokenizer | 2026-04-15 | completed | NTP Baseline MLP-FSQ（效果极差） |
| [011](./exp-011.md) | tokenizer | 2026-04-15 | completed | Codebook Size 消融 |
| [012](./exp-012.md) | tokenizer | 2026-04-15 | completed | **Tokenizer Grid Search — 4096×3 binary 赢家** |
| [013](./exp-013.md) | ntp | 2026-04-15 | completed | S-tier NTP baseline |
| [014](./exp-014.md) | ntp | 2026-04-16 | completed | ENTP-Loss 消融 |
| [015](./exp-015.md) | ntp | 2026-04-16 | completed | NTP Scaling Law |
| [016](./exp-016.md) | ntp | 2026-04-17 | completed | Data Scaling Law |
| [017](./exp-017.md) | rl | 2026-04-17 | completed | SP-DPO 初版 |
| [018](./exp-018.md) | rl | 2026-04-18 | completed | RF-DPO 初版 |
| [019](./exp-019.md) | rl | 2026-04-20 | completed | RF-DPO Joint NTP+DPO |
| [020](./exp-020.md) | rl | 2026-04-20 | completed | **RF-DPO Hard λ=0.3 最优** |
| [021](./exp-021.md) | rl | 2026-04-20 | planned | Qwen3-4B vs 0.6B Embedding |
| [022](./exp-022.md) | rl | 2026-04-20 | completed | In-Batch Contrastive Loss |
| [023](./exp-023.md) | rl | 2026-04-21 | completed | Side Features NTP |
| [024](./exp-024.md) | rl | 2026-04-21 | completed | Side Feature Shift（泄漏修复） |
| [025](./exp-025.md) | rl | 2026-04-21 | completed | **Beam Search Feature Passing — train-eval 一致** |
| [026](./exp-026.md) | tokenizer | 2026-04-27 | completed | **0.6B/4B/8B SID cache 构建** |
| [027](./exp-027.md) | rl | 2026-04-27 | interrupted | ECPO grpo_weight Sweep |
| [028](./exp-028.md) | rl | 2026-04-27 | completed | ECPO + WeightedBehaviorReward |
| [029](./exp-029.md) | rl | 2026-04-27 | completed | **ECPO + On-Policy Beam Search — R@500≈65%** |
| [030](./exp-030.md) | rl | 2026-04-27 | completed | A2PO + NLL + HEPO |
| [031](./exp-031.md) | rl | 2026-04-27 | completed | Features SFT + Full RL Stack |
| [032](./exp-032.md) | rl | 2026-04-28 | planned | GRPO Group Size Sweep |
| [033](./exp-033.md) | rl | 2026-04-28 | completed | Features 修复验证 |
| [034](./exp-034.md) | rl | 2026-04-28 | planned | Ref Model Alignment |
| [035](./exp-035.md) | rl | 2026-04-28 | completed | Constrained Sampling |
| [036](./exp-036.md) | ntp | 2026-04-28 | completed | **Features NTP — +3.7pp** |
| [037](./exp-037.md) | rl | 2026-04-28 | completed | SP-DPO on exp036 |
| [038](./exp-038.md) | rl | 2026-04-28 | completed | RF-DPO on exp037 |
| [038B](./exp-038b.md) | rl | 2026-04-28 | completed | **RF-DPO ntp_epochs=3 + mid-ckpt** |
| [039](./exp-039.md) | rl | 2026-04-28 | skipped | ECPO on exp038（被 039B 取代） |
| [039B](./exp-039b.md) | rl | 2026-04-29 | completed | **ECPO on exp038b ep1 — R@500=65.7% SOTA** |
| [040](./exp-040.md) | rl | 2026-04-28 | planned | RSFT |
| [041](./exp-041.md) | ntp | 2026-04-29 | completed | ENTP-Loss v1（无效） |
| [041B](./exp-041b.md) | ntp | 2026-04-29 | completed | ENTP-Loss v2（无效） |
| [043](./exp-043.md) | ntp | 2026-04-29 | completed | **Embedding × Tier 对比；M-tier R@500=70.2%** |
| [044](./exp-044.md) | ntp | 2026-04-29 | completed | TO-RoPE vs APE（timestamps=0） |
| [044B](./exp-044b.md) | ntp | 2026-04-29 | completed | **TO-RoPE 真实 timestamps — +2.4pp (S-tier)** |
| [044C](./exp-044c.md) | ntp | 2026-04-29 | completed | TO-RoPE Item-Pos Fix + 3-dim |
| [045](./exp-045.md) | tokenizer | 2026-04-29 | ⚠️ bug | FSQ h-dim sweep（num_clusters=1024 bug） |
| [046](./exp-046.md) | ntp | 2026-04-29 | completed | GateAttention +0.4pp |
| [047](./exp-047.md) | ntp | 2026-04-30 | completed | **L-tier SFT — R@500=64.1%（RL 链路起点）** |
| [048](./exp-048.md) | ntp | 2026-04-30 | completed | M-tier TO-RoPE 2-dim/3-dim（无收益） |
