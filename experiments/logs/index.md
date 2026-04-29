

按时间倒序记录。每次实验链接到 `experiments/` 下的结果目录。

---

## Template

<!--
复制以下模板创建新实验记录。编号递增，最新的放在最上面。

## EXP-NNN: (实验标题)

**Date**: YYYY-MM-DD
**Status**: planned | running | completed
**Results**: [./hyperparam/YYYY-MM-DD_xxx/](./hyperparam/YYYY-MM-DD_xxx/)

### Background
(当前状态、要解决的问题)

### Hypothesis
(预期结果及原因)

### Design
- **Variable**: ...
- **Fixed**: ...
- **Metric**: ...
- **Data**: ...

### Results
(跑完后填写，含表格)

### Analysis
(结果解读)

### Next Steps
(下一步计划)
-->

---

## Index

| EXP | Date | Status | Title |
|-----|------|--------|-------|
| [046](./exp-046.md) | 2026-04-29 | completed | GateAttention — Sigmoid Gate on Attention Output |
| [045](./exp-045.md) | 2026-04-29 | queued | FSQ Hidden Dim Fix — 4B h=256, 8B h=512 |
| [044](./exp-044.md) | 2026-04-29 | completed | TO-RoPE vs Absolute Position Embedding — S-tier + 0.6B SID |
| [044B](./exp-044b.md) | 2026-04-29 | completed | TO-RoPE with Real Timestamps — S-tier + 0.6B SID |
| [044C](./exp-044c.md) | 2026-04-29 | completed | TO-RoPE Item-Pos Fix + 3-dim RoPE |
| [043](./exp-043.md) | 2026-04-29 | completed | Embedding Model Size Comparison — S-tier & M-tier × 0.6B/4B/8B SID |
| [041](./exp-041.md) | 2026-04-29 | completed (结论: 无效，需重设计) | ENTP-Loss — Exposure-Aware Hard Negatives for L0 (with Features) |
| [041B](./exp-041b.md) | 2026-04-29 | completed (结论: 无效，session 粒度问题) | ENTP-Loss v2 — Session-Level Negatives (behavior_v2 数据) |
| [040](./exp-040.md) | 2026-04-28 | planned | RSFT — Reject Sampling Fine-Tuning (Training Data Quality Filter) |
| [039](./exp-039.md) | 2026-04-28 | skipped (superseded by EXP-039B) | ECPO on exp038-hard-lam03 (Features RL 链路终点) |
| [039B](./exp-039b.md) | 2026-04-29 | completed | ECPO on exp038b-hard-lam03-3ep-ep1 (Features RL 链路终点) |
| [038](./exp-038.md) | 2026-04-28 | completed | RF-DPO on exp037-medium (Features 路线第三步) |
| [038B](./exp-038b.md) | 2026-04-28 | completed | RF-DPO on exp037-medium — ntp_epochs=3 + mid-checkpoints |
| [037](./exp-037.md) | 2026-04-28 | completed | SP-DPO on exp036-full-features (Features 路线第二步) |
| [036](./exp-036.md) | 2026-04-28 | completed | Clean Features NTP — From-Scratch Training with time_gap + action_level |
| [035](./exp-035.md) | 2026-04-28 | completed | Constrained Sampling — Replace Beam Search with T=1.0 Sampling |
| [034](./exp-034.md) | 2026-04-28 | planned | Ref Model Alignment — exp025 as ref_checkpoint |
| [033](./exp-033.md) | 2026-04-28 | completed | Features 修复验证 — EXP-031A Rerun with Correct Feature Injection |
| [032](./exp-032.md) | 2026-04-28 | planned | GRPO Group Size vs Context Diversity — G × batch_size Sweep |
| [031](./exp-031.md) | 2026-04-27 | completed | New SOTA — Features SFT + Full RL Stack |
| [030](./exp-030.md) | 2026-04-27 | completed | A2PO + NLL Regularization + HEPO Prefix Scoring |
| [029](./exp-029.md) | 2026-04-27 | completed | ECPO + On-Policy Beam Search |
| [028](./exp-028.md) | 2026-04-27 | completed | ECPO + WeightedBehaviorReward — Continuous Quality×Freshness Reward |
| [027](./exp-027.md) | 2026-04-27 | interrupted (replaced by EXP-028) | ECPO grpo_weight Sweep — Align with RF-DPO Training Structure |
| [026](./exp-026.md) | 2026-04-27 | completed | GRPO+ECPO — Group Relative Policy Optimization + Pluggable Reward |
| [025](./exp-025.md) | 2026-04-21 | completed | Beam Search Feature Passing — 正确消除 side feature 训练-推理 gap |
| [024](./exp-024.md) | 2026-04-21 | completed | Side Feature Shift — 消除 time_gap/action_level 信息泄漏 |
| [023](./exp-023.md) | 2026-04-21 | completed | NTP Side Information — Time Gap + Action Type + Segment Embedding |
| [022](./exp-022.md) | 2026-04-20 | completed | NTP In-Batch Contrastive Loss (IDEA-onemall-0) |
| [021](./exp-021.md) | 2026-04-20 | planned | Qwen3-4B vs 0.6B Embedding Quality for SID Tokenizer |
| [020](./exp-020.md) | 2026-04-20 | completed | RF-DPO Hard λ Sweep — Finding Optimal DPO Weight |
| [019](./exp-019.md) | 2026-04-20 | completed | RF-DPO Joint NTP+DPO — Step-Matched Training |
| [018](./exp-018.md) | 2026-04-18 | completed | RF-DPO — Real Feedback DPO Alignment |
| [017](./exp-017.md) | 2026-04-17 | completed | SP-DPO — Self-Play DPO Alignment for NTP Model |
| [016](./exp-016.md) | 2026-04-17 | completed | Data Scaling Law — 固定模型 Sweep 数据量 (Chinchilla 双变量) |
| [015](./exp-015.md) | 2026-04-16 | completed | NTP Scaling Law — Sweep Model Size from 1M to 100M Active Params |
| [014](./exp-014.md) | 2026-04-16 | running | ENTP-Loss — Exposure-Aware Hard Negatives for L0 |
| [013](./exp-013.md) | 2026-04-15 | completed | S-tier NTP Model — 6L MoE + Loss-Free Balancing |
| [012](./exp-012.md) | 2026-04-15 | completed | Tokenizer Grid Search — KMeans × FSQ Type × OPQ |
| [011](./exp-011.md) | 2026-04-15 | completed (部分，OPQ 未跑) | Codebook Size 消融 — 等大 1024/4096 + OPQ 对照 |
| [010](./exp-010.md) | 2026-04-15 | completed (效果极差，需诊断) | NTP Baseline — MLP-FSQ SID 端到端 Recall |
| [009](./exp-009.md) | 2026-04-14 | completed | QFormer Tokenizer — 冻结 Qwen3 + Cross-Attention 压缩 |
| [008](./exp-008.md) | 2026-04-14 | completed | FORGE Proxy 对比 — MLP-FSQ vs OPQ 最优解 |
| [007](./exp-007.md) | 2026-04-13 | completed | Collaborative Signal Enhanced Embedding (Qwen3-0.6B Full Fine-tune) |
| [004](./exp-004.md) | 2026-04-13 | completed | OPQ Parallel Semantic IDs — Intrinsic Metrics |
| [003](./exp-003.md) | 2026-04-13 | completed | Learned FSQ — MLP projection + straight-through training |
| [002](./exp-002.md) | 2026-04-13 | completed | ResKmeansFSQ — 2 layers RKMeans + 1 layer FSQ (PCA projection) |
| [001](./exp-001.md) | 2026-03 | completed | RKMeans 训练优化 (v0→v7) |
