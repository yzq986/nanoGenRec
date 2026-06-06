# Discussions

[English](README.md) | [中文](README.zh.md)

关键技术主题的深度问答和概念分析。

不同于 `ideas/`（方案设计）或 `experiments/`（实证结果），这里记录的是**设计选择背后的推理**：为什么某种方法有效、概念之间有什么区别、哪些 trade-off 重要。

## Index

| 文件 | 主题 | 日期 |
|------|------|------|
| [001-sp-dpo-vs-sft-vs-contrastive.md](001-sp-dpo-vs-sft-vs-contrastive.md) | SP-DPO、SFT 和 Contrastive Learning 的本质区别 | 2026-04-17 |
| [002-rf-dpo-grpo-ecpo-progression.md](002-rf-dpo-grpo-ecpo-progression.md) | 从 SP-DPO 到 ECPO 的 RL 对齐递进路径 | 2026-04-17 |
| [003-sp-dpo-training-engineering.md](003-sp-dpo-training-engineering.md) | SP-DPO 训练工程：OOM、DDP、联合优化 | 2026-04-18 |
| [004-prefix-locked-vs-paper-beam-search.md](004-prefix-locked-vs-paper-beam-search.md) | Prefix-locked 与论文版 beam search 的候选生成权衡 | 2026-04-19 |
| [005-beam-search-kv-cache.md](005-beam-search-kv-cache.md) | Beam search KV cache：三级计算复用优化 | 2026-04-19 |
| [006-why-grpo-needs-on-policy-but-dpo-doesnt.md](006-why-grpo-needs-on-policy-but-dpo-doesnt.md) | 为什么 GRPO 需要 on-policy candidates，而 DPO 不需要 | 2026-04-28 |
| [007-sid-l2-entropy-collision-floor-ppl.md](007-sid-l2-entropy-collision-floor-ppl.md) | SID L2 entropy、collision rate 与 floor PPL 的关系；FSQ bottleneck 根因分析 | 2026-04-29 |
