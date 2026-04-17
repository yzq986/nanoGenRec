# RL Alignment

NTP 模型的强化学习对齐模块。将生成式推荐从"预测准"优化到"推荐好"。

**前置依赖**: NTP 模型训练稳定（Stage 1 完成，PPL < 30, R@500 > 60%）✅

## Roadmap

```
Phase 1: SP-DPO        ← 入门，零外部依赖
Phase 2: RF-DPO        ← 引入真实用户反馈
Phase 3: GRPO          ← group-wise 优化，需要 reward model
Phase 4: ECPO          ← 训练稳定性 + 生成合法性
```

### Phase 1 — SP-DPO (Self-Play DPO)

**来源**: Align³GR (快手, AAAI 2026 Oral, arxiv 2511.11255)

**核心**: 模型自博弈生成负样本，用 prefix n-gram 匹配度定义难度，渐进式 DPO 训练。

| 项目 | 设计 |
|------|------|
| Chosen | Ground truth next item SID |
| Rejected | 模型 beam search 生成的非 ground truth 候选 |
| 难度定义 | SID prefix 匹配层数 (0/1/2 层) |
| 训练策略 | Easy → Medium → Hard 三阶段渐进 |
| Loss | Softmax-DPO (1 chosen vs 20 rejected) |
| Reference model | 每阶段结束后冻结当前模型作为下阶段 π_ref |
| 前置依赖 | 只需训练好的 NTP 模型 + beam search |

**需要实现**:
- [ ] `rl/preference.py` — preference pair 构造 (beam search → 按 prefix 匹配分 Easy/Medium/Hard)
- [ ] `rl/dpo.py` — Softmax-DPO loss
- [ ] `rl/trainer.py` — Progressive DPO 训练循环 (NTP loss + DPO loss 联合)
- [ ] `rl/eval.py` — RL 前后对比评估

**评估**: Recall@K, NDCG@K, PPL — 对比 NTP-only baseline

### Phase 2 — RF-DPO (Real-Feedback DPO)

**来源**: Align³GR

**核心**: 用真实用户行为替换自博弈信号。

| 项目 | 设计 |
|------|------|
| Chosen | 用户 liked 的 item (点击/购买/收藏) |
| Rejected | Easy: disliked; Hard: neutral (曝光未点击) |
| 训练策略 | Easy → Hard 两阶段渐进 |
| Reference model | SP-DPO Phase 3 (Hard) 的输出模型 |
| 前置依赖 | SP-DPO 完成 + 用户行为标签 (clicked/not-clicked/disliked) |

**需要实现**:
- [ ] `rl/feedback.py` — 真实反馈 pair 构造 (从行为数据提取 liked/neutral/disliked)
- [ ] 复用 `rl/dpo.py` + `rl/trainer.py`

### Phase 3 — GRPO (Group Relative Policy Optimization)

**来源**: OneMall (arxiv 2601.21770)

**核心**: 对整组候选计算 group-normalized advantage，全部候选都产生梯度。

| 项目 | 设计 |
|------|------|
| 采样 | 每用户 beam search 生成 G=512 候选 |
| Reward | 外部 reward model 打分 (CTR/CVR 或行为信号) |
| Advantage | A_i = (r_i - mean) / std（组内归一化） |
| Loss | Clipped surrogate (PPO 风格, ε=0.2) |
| Joint loss | L = L_NTP + 0.5 * L_GRPO |
| RL 数据比例 | 2% 训练样本 |
| 前置依赖 | Reward model |

**需要实现**:
- [ ] `rl/reward.py` — reward model (方案 A: 行为信号; 方案 B: 离线 CTR 模型)
- [ ] `rl/grpo.py` — GRPO loss (clipped surrogate + group-normalized advantage)
- [ ] 更新 `rl/trainer.py` — 支持 GRPO 训练循环

### Phase 4 — ECPO (Early Clipped GRPO)

**来源**: OneRec (arxiv 2506.13695v4)

**核心**: 修复 GRPO 负 advantage 梯度爆炸 + Format Reward 保证生成合法性。

| 项目 | 设计 |
|------|------|
| Early clip | δ=0.1, 负 advantage 的 ratio 上界 1+ε+δ |
| KL penalty | 移除（RSFT co-training 提供稳定性） |
| Format reward | 随机采样 K=5, 合法 SID advantage=1, 非法 advantage=0 |
| RSFT | 过滤底部 50% sessions，监督微调 |
| Group size | 512 (≈4× inference Pass@K) |
| 前置依赖 | GRPO 完成 + P-Score reward model |

**需要实现**:
- [ ] `rl/ecpo.py` — Early clipped GRPO loss + format reward
- [ ] `rl/rsft.py` — Rejection sampling fine-tuning (session 过滤)

---

## 技术讨论

深度 Q&A 记录在 [discussions/](../discussions/) 目录：
- [001: SP-DPO vs SFT vs Contrastive Learning](../discussions/001-sp-dpo-vs-sft-vs-contrastive.md)
- [002: From SP-DPO to ECPO — the full progression](../discussions/002-rf-dpo-grpo-ecpo-progression.md)

## 相关 Ideas

详细方案设计见 [ideas/rl-alignment.md](../ideas/rl-alignment.md)，覆盖 9 个方案：
- IDEA-align3-0: Progressive DPO (SP-DPO + RF-DPO)
- IDEA-onemall-2: GRPO/DPO
- IDEA-onerec-3: ECPO + Format Reward
- IDEA-sgrec-0: A2PO + Personalized Semantic Judge
- IDEA-rankgr-0: Listwise DPO + Rescore
- IDEA-oneloc-2: DPO + 双目标奖励
- IDEA-gr4ad-3: RSPO
- IDEA-uni-0: SPO
- IDEA-gpr-0: HEPO
