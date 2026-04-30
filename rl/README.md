# rl/ — 强化学习对齐模块

NTP 模型的 RL 对齐：SP-DPO → RF-DPO → GRPO/ECPO 完整链路。

## 文件

| 文件 | 说明 |
|------|------|
| `preference.py` | SP-DPO preference pair 构造（beam search → prefix 匹配分 Easy/Medium/Hard）|
| `dpo.py` | Softmax-DPO / RF-DPO loss，`compute_sid_logprobs` side features 全链路 |
| `feedback.py` | RF-DPO 真实反馈 pair 构造（行为数据 liked/neutral/disliked）|
| `reward.py` | BehaviorReward（prefix cascade fallback），A2PO/HEPO reward stack |
| `grpo.py` | GRPO loss（clipped surrogate + group-normalized advantage）|
| `trainer.py` | 统一训练循环：SP-DPO / RF-DPO / GRPO / ECPO 模式；context pool；on-policy beam search |

## 标准 RL 链路

```
SFT → SP-DPO → RF-DPO (3ep, mid-ckpt) → ECPO
```

- **SP-DPO**：自博弈生成 preference pair，Softmax-DPO，渐进式 Easy→Medium→Hard
- **RF-DPO**：真实用户行为 pair，joint NTP+DPO loss，3 epoch + mid-checkpoint 取最优 ep
- **ECPO**：GRPO + BehaviorReward + A2PO + NLL + HEPO 全 reward stack，Early clip (δ=0.1)

## 已验证超参

| 参数 | 最优值 | 实验 |
|------|--------|------|
| RF-DPO λ | 0.3 (hard) | EXP-020 |
| RF-DPO ntp_epochs | 3，取 ep1 checkpoint | EXP-038B |
| ECPO δ | 0.1 | EXP-028+ |
| ECPO ε | 0.2 | EXP-028+ |
| GRPO G | 512，grpo_batch=4 | EXP-029 |
| grpo_weight | 0.03 | EXP-029 |

## 当前 SOTA

| 配置 | R@500 | 来源 |
|------|-------|------|
| **ECPO on exp038b ep1** | **65.7%** | EXP-039B |
| RF-DPO 3ep ep1 | 62.1% | EXP-038B |
| SP-DPO | ~55–58% | EXP-037 |

**下一步**：以 exp047（L-tier SFT，R@500=64.1%）为起点重跑完整 RL 链路。

## 关键实现细节

### BehaviorReward prefix cascade
全 SID 精确匹配覆盖率仅 ~0.16%；prefix cascade fallback (L0) 覆盖率 ~24%，有效 reward 信号提升 150×。

### reward std≈0 保护
稀疏 reward 下一组 candidates reward 全同 → std≈0 → advantage 爆炸。
对策：`std < 1e-6` group skip + `adv.clamp(-5, 5)` + `log_rho.clamp(-10, 10)`。

### on-policy beam search
ECPO 必须 on-policy：每步用当前策略重新生成候选，避免 off-policy ratio 爆炸（EXP-029）。

### side features 全链路
`dpo.py:compute_sid_logprobs` 和 `trainer.py` context pool 均通过 `ctx_side_features` / `gen_side_features` dict 传递特征，与 NTP 训练路径保持一致。详见 [CLAUDE.md — Side Features 注入架构](../CLAUDE.md)。

## 实验记录

见 [`experiments/logs/rl/README.md`](../experiments/logs/rl/README.md)。
