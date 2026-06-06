# Why GRPO Needs On-Policy Candidates But DPO Doesn't

[English](006-why-grpo-needs-on-policy-but-dpo-doesnt.md) | [中文](006-why-grpo-needs-on-policy-but-dpo-doesnt.zh.md)

**Date**: 2026-04-28  
**Context**: EXP-026~029 发现 off-policy beam search 导致 GRPO clip=99%、R@500=2%；切换 on-policy 后 clip→92%、R@500=67.8%

---

## 核心发现

EXP-028 证明了一个关键结论：**reward 修复（WeightedBehaviorReward，coverage ~100%）不足以激活 GRPO 梯度**，必须同时切换到 on-policy beam search 才能让 RL 信号传回去。

这引出一个问题：RF-DPO（EXP-020）完全使用离线数据，从未需要 on-policy 采样，照样把 R@500 推到 66.2%。为什么 DPO 不需要 on-policy？

---

## DPO 不需要 On-Policy 的原因

DPO loss：

```
L = -log σ(β · (log π_θ(y_w|x) - log π_ref(y_w|x)) - β · (log π_θ(y_l|x) - log π_ref(y_l|x)))
```

**梯度方向是确定的**：无论 policy 当前分布如何，loss 永远推高 chosen 的相对 log-prob、压低 rejected 的相对 log-prob。离线 preference pairs 的标注质量决定优化方向，policy 与数据分布的偏差不影响梯度方向的正确性，只影响步长大小（通过 β 隐式控制）。

此外，DPO 的 β 项本质上是 KL 惩罚 `KL(π_θ || π_ref)`，它天然约束 policy 不偏离 ref 太远——即使数据是离线的，两者分布差异被显式控制住了。

---

## GRPO 必须 On-Policy 的原因

GRPO advantage：

```
adv_i = (r_i - mean(r)) / std(r)    over group G candidates
```

PPO surrogate loss：

```
L = -min(ρ · adv, clip(ρ, 1-ε, 1+ε) · adv)
    where ρ = π_θ(y_i|x) / π_old(y_i|x)
```

**advantage 的有效性依赖于 candidates 来自当前 policy 分布**。如果 candidates 是 ref model 生成的（off-policy）：

1. 随训练推进，policy 和 ref 分布偏离
2. importance ratio `ρ = π_θ/π_ref` 偏离 1，大量样本落在 clip 边界
3. clip=99%：几乎所有梯度都被截掉，advantage 再准确也传不回去
4. RL 训练名存实亡

EXP-028 的实证：reward coverage 从 0.16% 提升到 ~100%（WeightedBehaviorReward），clip 率仍然 99%，R@500 仍然 2%。reward 信号有了，但 importance ratio 太偏，梯度路径断了。

EXP-029 切换 on-policy beam：clip 率 99%→92%，R@500 2%→67.8%。

---

## 本质差异

| | DPO | GRPO |
|---|---|---|
| 梯度依赖 | 固定 preference pair 的相对排序 | group 内 advantage × importance ratio |
| Off-policy 影响 | 步长变化，方向不变 | importance ratio 偏离 → clip 失效 → 梯度为 0 |
| KL 约束 | β 项隐式约束，防止漂移 | 无显式约束，policy 自由漂移 |
| 数据需求 | 离线 preference pairs，一次收集永久有效 | 每步需要 policy 当前分布下的 candidates |

**比喻**：DPO 是"给你两道菜，告诉你哪道更好"——离线数据完全够用。GRPO 是"从你现在的菜单里采样，评估相对好坏"——如果菜单是别人的（ref model），评估结果对你毫无意义。

---

## 推论

- GRPO 的 on-policy 需求是**结构性的**，不能通过 reward shaping 绕过
- DPO 的离线特性是**优势**：preference data 收集一次可反复使用，无需在线采样
- 两者结合（先 DPO 再 GRPO）是合理的：DPO 用离线数据把模型推到好的起点，GRPO 在此基础上做在线 fine-tuning
- off-policy GRPO 的 clip=99% 是诊断指标：一旦看到 clip 率居高不下，首先怀疑 off-policy 问题，而非 reward 问题
