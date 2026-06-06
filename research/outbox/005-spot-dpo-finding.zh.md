---

[English](005-spot-dpo-finding.md) | [中文](005-spot-dpo-finding.zh.md)
date: "2026-04-22 19:30"
type: finding
priority: normal
subject: "SPOT 论文: SPoT-BCO 可直接替换 SP-DPO loss，无需重训 NTP"
needs_response: false
---

## 发现来源

论文 2603.01683 "Surgical Post-Training: Cutting Errors, Keeping Knowledge"（SPOT，香港大学，2026-03）

## 核心发现

### 1. DPO 的结构性缺陷

SPOT 从理论角度揭示了为什么标准 DPO 在 **rigid correctness** 任务（正确/错误明确界定）上表现不佳：

DPO 的梯度路径是 "最小化 margin M = r(y+) - r(y-)"，因此最省力的优化方式是**压低 r(y-)**而非提升 r(y+)。实验中观察到 DPO 在训练后期 chosen reward 停止上升但 rejected reward 持续下降。

这与我们的 SP-DPO 场景高度对应：
- y- = beam-search 产生的 wrong SID 序列
- y+ = teacher-forced 的 target SID 序列  
- 目标是让模型**更倾向于生成 target SID**，而非只是"比 beam 好一点"

如果 SP-DPO 优化的是 margin 而非绝对正确性，模型可能通过降低对 beam 结果的置信度来满足损失，而不是真正提升 target SID 的生成概率。

### 2. SPoT-BCO 是更好的替代

SPOT 提出将 DPO 解耦为**独立二分类**：

```
L_BCO = -E[log σ(r(x,y+) - δ) + log σ(-(r(x,y-) - δ))]
```

- **δ** = batch reward 的指数移动平均（防止 r 增大后梯度消失）
- 独立地最大化 y+ 置信度 + 最小化 y- 置信度
- 与 DPO 相比额外引入 "Elastic Tether"：当 r(y+) 足够大时，梯度自动消失，防止过度优化

SPOT 实验：SPoT-BCO 在推理任务上比 DPO 平均高 +3.5%，且不产生 catastrophic forgetting。

### 3. 数据质量过滤

LCS filtering（RLCS < 0.6）确保 y+ 和 y- 只在关键位置分歧。对应到我们的场景：

```
RLCS(beam_sid, target_sid) = 1 - |LCS(beam_tokens, target_tokens)| / |target_tokens|
```

只保留 beam 和 target 部分匹配（RLCS < 0.6）的样本：
- 完全不同的 beam（RLCS ≈ 1.0）= 信息量低，梯度集中在全局而非局部错误
- 部分匹配（RLCS 0.2~0.6）= 错误在中后层 SID token，梯度精准定位

## 实验提案 IDEA-spot-0（P1）

**假设**：将 SP-DPO 的损失函数替换为 SPoT-BCO，可以让模型真正提升 target SID 的生成概率，而非仅仅降低 beam 置信度。

**变量**：
- 损失函数：DPO → SPoT-BCO（δ = batch reward EMA）
- 数据过滤：加入 LCS 质量过滤（RLCS < 0.6）

**固定**：
- NTP checkpoint：exp025-beam-passes（R@500=63.6%）
- beam-pass 数据：已有 sp-dpo-prepare 结果
- 模型结构、学习率等

**实现工作量**：
1. 修改 `rl/dpo_loss.py`（或相应文件）的 loss function — 小改动
2. 修改 `sp-dpo-prepare` 输出加 LCS score 字段 — 中等改动
3. 在 `sp-dpo-train` 加 BCO loss 选项 — 小改动

**需要人类授权**：`rl/` 目录下的源码修改。

**预估改进**：
- SP-DPO 当前状态未知（EXP-017~018 的对齐结果需查）
- 若 SP-DPO 已有正向结果，BCO 替换预期进一步提升 R@10/R@500 约 +0.5~2%
- 若 SP-DPO 目前表现不佳，这是可能的根因修复

## 其余论文摘要（本次批次）

**2603.28124 RCLRec**（阿里巴巴）：Conversion 稀疏性问题，用 pay-conditioned query 从历史中选 k 个关键 item 作 decoder 前缀。相关性中等，我们是 decoder-only NTP，需较大架构改动才能引入。可借鉴的轻量思路：对 conversion 行为 token 加样本权重（对应 IDEA-dualgr-0）。

**2603.11486 Quantized Inference for OneRec-V2**（快手）：FP8 PTQ 实现 49% 延迟降低、92% 吞吐提升，在线无回退。当前阶段对我们无直接价值，但验证了 GR 模型（MoE 架构）与 LLM 一样可以低精度量化，是未来部署阶段的参考。

## 行动

- [x] 写 paper-notes/2603.28124.md、2603.11486.md、2603.01683.md
- [ ] IDEA-spot-0 需要人类授权后才能实验
- [ ] 建议下次 session 优先读更多未读论文（还有 13 篇）
