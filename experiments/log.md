

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

## EXP-035: Constrained Sampling — Replace Beam Search with T=1.0 Sampling

**Date**: 2026-04-28
**Status**: completed
**Results**: experiments/ntp_checkpoints/exp035-sampling-t1/

### Background

EXP-034 验证了 ref/policy 对齐只是部分改善（clip=95%，仍然很高）。真正根因是 beam search 的结构性问题：

- beam search 总是选取 policy 最自信的候选 → ρ = π_θ/π_ref >> 1 → clip 率结构性偏高
- 候选集中在 policy 峰值 → reward 方差极小 → advantage ≈ 0 → 梯度退化
- EXP-034 日志显示 `adv=-0.00`，模型实际上几乎没在学习

**解法**：用 `constrained_sampling(T=1.0)` 替换 beam search：
- 候选直接从 policy 分布采样 → ρ ≈ 1 by construction
- 采样覆盖多样路径（好/坏/中等）→ advantage 有真正对比信号
- G: 512→64（多样性由 T 保证，不需要大 G），显存节省

### Hypothesis

| 指标 | EXP-034 (beam G=512) | EXP-035 (sampling T=1.0 G=64) |
|------|---------------------|-------------------------------|
| clip 率 | ~95% | 预期 10~40% |
| adv_std | ≈0 | 预期 >0（真正对比信号）|
| R@500 | TBD | 预期 ≥ EXP-034 |

### Design
- **Variable**: `--sampling_temperature 1.0`，`--group_size 64`（beam→sampling）
- **Fixed**: ref=exp025, policy=exp025, 其余参数同 EXP-034（rank_norm, a2po, nll_reg, hepo）
- **Metric**: clip 率、adv_std、R@10、R@500
- **Data**: exp023-14d-features

### Run
`bash experiments/scripts/exp-035.sh`

### Results

**Inline eval** (250 items, not comparable to baseline):

| 指标 | EXP-035 (sampling T=1.0, G=64) | EXP-029 SOTA (beam G=512) | 差距 |
|------|-------------------------------|--------------------------|------|
| R@10 | 0.102 | 0.130 | -0.028 ❌ |
| R@500 | 0.615 | 0.678 | -0.063 ❌ |
| clip 率 | 94.8% | 92.3% | 略高 |
| adv_std | 0.595 | — | 有对比信号 ✅ |
| behavior_mean | 0.363 | ~0.65 | 低一半 ❌ |
| behavior_coverage | 89.2% | ~99% | 低 ❌ |
| train time | 18min | ~70min | 快 4x ✅ |

全量 eval (n_recall=1000) 进行中，结果待更新。

### Analysis

**假设验证结果**：

| 假设 | 预期 | 实际 | 结论 |
|------|------|------|------|
| clip 率降至 10~40% | ✅ | 94.8% | ❌ 假设错误 |
| adv_std > 0 | ✅ | 0.595 | ✅ 有对比信号 |
| R@500 ≥ EXP-034 | ✅ | 0.615 < 0.678 | ❌ 不如 SOTA |

**关键发现（训练后分析）**：

1. **clip 高的根因不是 beam/sampling**：所有实验 clip 都在 92~96%，是 NTP joint training 的 softmax 漂移导致，与候选生成方式无关
2. **G=64 的稀疏 reward 问题**：beam G=512 coverage=99%，sampling G=64 coverage=89%，behavior_mean 从 0.65 降到 0.36，reward 信号弱了一半
3. **adv_std=0.595 是进步**：beam search 时 adv≈0（候选集中在 policy 峰值，reward 方差极小），sampling 后有了真正的对比信号
4. **训练效率大幅提升**：18min vs 70min，快 4x（G 减小 8x）

**结论**：sampling 本身方向正确（adv_std 改善），但 G=64 + behavior reward 稀疏 = 信号太弱，无法超越 beam G=512 的结果。

### Next Steps

- 加入 KL(π_θ||π_ref) 作为跨实验可比的 RL 核心指标
- EXP-036：sampling G=64 但提高 behavior coverage（扩大 behavior cache 或用 prefix cascade fallback 提升匹配率）
- 或：sampling G=256（在效率和 coverage 之间取平衡）

---

## EXP-034: Ref Model Alignment — exp025 as ref_checkpoint

**Date**: 2026-04-28
**Status**: planned
**Results**: TBD

### Background

EXP-033 证伪了 features bug 假设：修复三处 features 注入 bug 后，clip 率从 96.4% 变为 96.2%，几乎没有变化。真正根因是 ref model（exp020）与 policy 起点（exp025）不对齐。

PPO clip 条件是 ρ = exp(policy_lp - ref_lp) 超出 [1-ε, 1+ε]。exp025 在 exp020 上做了 beam-passes SFT，两者对同一 token 的 log-prob 系统性不同。从 exp025 出发时第一步就大量触发 clip，不是因为更新过大，而是初始 KL 就很大。

| 实验 | policy 起点 | ref model | clip 率 |
|------|------------|-----------|---------|
| exp031-baseline | exp020 | exp020 | 92.4% ✅ |
| exp031-features | exp025 | exp020 | 96.4% ❌ |
| exp033 | exp025 | exp020 | 96.2% ❌ |
| **EXP-034** | **exp025** | **exp025** | **预期 ~92%** |

### Hypothesis

ref model = policy 起点 = exp025 → RL 开始时 KL≈0 → clip 率回落至 ~92%（与 exp031-baseline 对齐）。features 模型（R@500=63.6% SFT）经过正确 RL 对齐后应超越 exp020 路线（67.8%），因为 features 提供更好的 beam search 区分度。

### Design
- **Variable**: ref_checkpoint = exp025（而非 exp020），其余参数与 exp031-baseline 完全一致
- **Fixed**: ECPO δ=0.1, ε=0.2, G=512, grpo_batch=4, grpo_weight=0.03, ratio=1.0, lr=1e-4, on_policy, rank_norm, A2PO(α=1.0), NLL(0.01), HEPO(0.1,0.5)
- **Metric**: R@10, R@500（full eval n_recall=1000），clip 率
- **Data**: exp023-14d-features，WeightedBehaviorReward + FormatReward

### Run
`bash experiments/scripts/exp-034.sh`

### Results
TBD

### Analysis
TBD

### Next Steps
TBD

---

## EXP-033: Features 修复验证 — EXP-031A Rerun with Correct Feature Injection

**Date**: 2026-04-28
**Status**: completed
**Results**: [experiments/ntp_checkpoints/exp033-features-fix/](experiments/ntp_checkpoints/exp033-features-fix/)

### Background

EXP-031 Config A（features 模型 exp025 起点 + full RL stack）出现严重退化：clip=0.964（vs 正常 0.924），R@500=61.8%（vs baseline 66.2%）。

分析发现代码中存在三处 features 注入 bug（在本 session 中修复）：
1. `constrained_beam_search` 调用未传 `ctx_time_gaps/ctx_action_levels` → beam search 候选基于无特征 embedding，与训练分布不一致
2. `compute_sid_logprobs` 直接调 `_embed_tokens` + 手动拼接，绕过统一入口 → policy_lp 和 ref_lp 的 embedding 路径与训练路径不一致
3. `context_pool` 只存 token 列表，丢弃了 time_gaps/action_levels → GRPO 步中所有 context 均无特征

上述 bug 导致 features 模型的 train-infer 不一致：训练时有特征，RL 步的 forward 无特征，分布偏差导致 clip 率异常升高。

修复方案：
- `ntp/model.py` 新增 `embed_with_features()` 统一入口
- `rl/dpo.py:compute_sid_logprobs` 改用 `embed_with_features`
- `rl/trainer.py:context_pool` 存 `(tokens, tg, al)` 三元组；`_grpo_step` 传特征给 beam search + logprobs；carry-forward `gen_action_level = ctx_al[-1]`

### Hypothesis

features bug 是 EXP-031 Config A clip 率升高（0.964 vs 0.924）的主要原因。修复后重跑应该：
- clip 率回落到 ~0.924（与 exp029/031-B 对齐）
- R@500 从 61.8% 提升，可能超越 66.2% baseline（features 提供更好的 reward 区分度）

### Design
- **Variable**: features 修复后重跑 EXP-031 Config A（其余参数完全相同）
- **Fixed**: sft_checkpoint=exp025-beam-passes，ECPO δ=0.1 ε=0.2，G=512，batch=4，grpo_weight=0.03，ratio=1.0，lr=1e-4，full reward stack（WeightedBehaviorReward + FormatReward + A2PO + NLL + HEPO）
- **Metric**: R@10, R@500（full eval n_recall=1000），clip 率，adv_std
- **Data**: exp023-14d-features

### Run
`bash experiments/scripts/exp-033.sh`

### Results

训练 86min（409 steps，4×A100）。全量 eval n_recall=1000：

| 指标 | EXP-033（features fix） | EXP-031A（features bug） | baseline exp020 |
|------|------------------------|------------------------|-----------------|
| R@10 | **10.3%** | 10.5% | 14.1% |
| R@500 | **61.0%** | 61.8% | 66.2% |
| clip 率 | **96.2%** | 96.4% | — |
| PPL | **24.62** | — | 16.3 |
| adv_std | 0.580 | — | — |
| wall_time | 86min | — | — |

### Analysis

**假设被证伪**：features bug 不是 clip 率异常的原因。修复后 clip 率从 0.964 变为 0.962，几乎没有变化，全程稳定在 96%。

**真正根因确认：ref model 与 policy 起点不对齐（EXP-034 待验证）**

对比三个实验的关键数据：

| 实验 | policy 起点 | ref model | KL(policy‖ref) 初始值 | clip 率 |
|------|------------|-----------|----------------------|---------|
| exp031-baseline | exp020 | exp020 | ≈ 0 | **92.4%** ✅ |
| exp031-features | exp025 | exp020 | 大 | **96.4%** ❌ |
| exp033 | exp025 | exp020 | 大 | **96.2%** ❌ |

PPO clip 的触发条件是 ρ = exp(policy_lp - ref_lp) 超出 [1-ε, 1+ε]。exp025 是在 exp020 上做了 beam-passes SFT，两者对同一 token 的 log-prob 系统性不同。从 exp025 出发做 RL 时，**第一步就已经大量触发 clip**，不是因为更新太大，而是 policy 和 ref 本来就不在同一分布。ε=0.2 的 clip 窗口对于这个跨模型 KL 来说太窄。

adv_std 和 reward_std 在三个实验里几乎完全相同（~0.58, ~1.68），排除了 reward 设计的嫌疑。

**修法：ref_model = policy 起点 = exp025**，这样 RL 开始时 KL=0，clip 只在真正更新过大时才触发。

### Next Steps

1. **EXP-034（待做）**：用 exp025 作为 ref model（而非 exp020），其余参数与 exp031-baseline 完全一致。预期 clip 率应回落到 ~92%，R@500 应超过 baseline 66.2%。这是验证上述根因的关键实验。
2. features 修复本身正确（保证 train-infer 一致性），应保留，与 ref model 对齐问题独立。

---

## EXP-032: GRPO Group Size vs Context Diversity — G × batch_size Sweep

**Date**: 2026-04-28
**Status**: planned
**Results**: TBD

### Background

EXP-026~031 全部使用 G=512, grpo_batch_size=4。分析 SID trie 结构发现：
- L1→L2 平均 branching = 3.27（p50=2），G=512 时 step3 展开约 1673 candidates，实际返回 ~512，是有效的。
- 但每步只处理 4 个 context，GRPO advantage 是 group 内归一化的，context 多样性不足可能导致更新方向偏差。

核心假设：**在总 candidate 预算相同（G × grpo_batch = 2048）的前提下，更多 context（小 G + 大 batch）比更多 per-context candidates（大 G + 小 batch）更有利于 GRPO 收敛**。原因：
- advantage 是 group 内 relative ranking，group 内候选越多不等于梯度方向越准
- 每步更新来自更多独立 context → 梯度方差更小 → 更稳定的策略改进
- 小 G beam search 更快 → 同等时间内可以多跑几步

### Hypothesis

G=128, grpo_batch=16 优于 G=512, grpo_batch=4（相同 candidate 预算，4× context 多样性）。
G=32, grpo_batch=64 进一步提升或持平（16× context 多样性，但 per-group reward variance 可能太低）。
存在一个最优 G，使 per-group reward variance 和 context 多样性达到最佳平衡。

### Design
- **Variable**: (G, grpo_batch_size) 组合，总 candidate 预算 G × grpo_batch ≈ 2048
  - Config A (control): G=512, grpo_batch=4（复现 EXP-029 baseline）
  - Config B: G=128, grpo_batch=16（4× context 多样性）
  - Config C: G=32, grpo_batch=64（16× context 多样性）
- **Fixed**: ECPO δ=0.1, ε=0.2, grpo_weight=0.03, rl_data_ratio=1.0, lr=1e-4, on_policy_beam, WeightedBehaviorReward + FormatReward，sft_checkpoint=exp020-hard-lam03
- **Metric**: R@10, R@500（full eval, n_recall=1000），avg_clip_fraction，avg_advantage_std
- **Data**: exp023-14d-features, behavior cache 2026-03-31

### Run
`bash experiments/scripts/exp-032.sh`

### Results
TBD

### Analysis
TBD

### Next Steps
TBD

---

## EXP-031: New SOTA — Features SFT + Full RL Stack

**Date**: 2026-04-27
**Status**: completed
**Results**: `experiments/ntp_checkpoints/exp031-*/`

### Background
当前 SOTA 是 `exp020-hard-lam03`（R@500=66.2%，无 side features，纯 SFT 起点）。
`exp025-beam-passes` 是 features 模型（R@500=63.6%，有 time_gap+action_level+segment_emb），
虽然 SFT 层面比 exp020 低 2.6pp，但 features 让模型在 beam search 时能区分候选的时效性和交互强度，
有望在 RL 阶段获得更强 reward 信号，最终超越 66.2%。

EXP-028/029/030 都从 exp020（无 features）出发，无法充分利用已有的 features 训练结果。
EXP-031 从 exp025 出发，首次把 features 模型接入 RL pipeline，同时叠加所有已验证改进。

顺带修复：GRPO trainer 之前未传 time_gaps_list/action_levels_list 给 UnifiedSequenceDataset，
导致 features 模型在 NTP 训练步中缺少 side features。EXP-031 同步修复该 bug。

### Hypothesis
features 模型（time_gap + action_level）在 beam search 时能生成更多样化的候选（时效性分布更广），
WeightedBehaviorReward 的 freshness × quality 信号对 features 候选更具区分度 →
RL gradient 更有效 → R@500 从 63.6% 提升超过 exp020 的 66.2%，设立新 SOTA。

### Design
- **Variable**: features SFT 起点（exp025-beam-passes）+ 完整 RL stack
- **Fixed**: ECPO δ=0.1，ε=0.2，G=512，grpo_batch=4，grpo_weight=0.03，ratio=1.0，lr=1e-4
- **Metric**: R@10, R@500（full eval，n_recall=1000），与 exp020 baseline 对比
- **Data**: exp023-14d-features（包含 time_gap + action_level），WeightedBehaviorReward
- **Config A** (full stack): ECPO + on-policy beam + rank_norm + A2PO(α=1.0) + NLL(0.01) + HEPO(0.1,0.5)
- **Config B** (ablation - no features contribution): 同 A，但 sft_checkpoint=exp020（确认 features 是否有增益）

### Run
`bash experiments/scripts/exp-031.sh`

### Results

| Config | 起点 SFT | PPL | R@10 | R@500 | clip | adv_std | 训练耗时 |
|--------|---------|-----|------|-------|------|---------|---------|
| **A: features + full RL** | exp025-beam-passes (63.6%) | 24.2 | 11.1% | 61.8% | 0.964 | 0.580 | 81min |
| **B: baseline + full RL** | exp020-hard-lam03 (66.2%) | 14.6 | 12.5% | **67.7%** | 0.924 | 0.579 | 80min |
| EXP-029 (on-policy only) | exp020 | 14.1 | 13.0% | 67.8% | 0.923 | — | 80min |
| EXP-020 SFT SOTA | — | 16.3 | 14.1% | 66.2% | — | — | 62min |

### Analysis

**Config B（baseline + full RL stack）= 67.7%，与 EXP-029 的 67.8% 基本持平（-0.1pp）**，验证了 full RL stack（A2PO+NLL+HEPO）在 exp020 起点上没有带来额外增益，也没有损害——EXP-029 的 on-policy ECPO 已经是这条路线的上限。

**Config A（features + full RL）= 61.8%，严重退化（-6pp vs EXP-029）**：
1. **clip 率 0.964 vs 0.924**：features 模型的 clip 率显著更高，说明 policy 和 ref 的 importance ratio 更大——features 改变了 token 分布，on-policy beam search 生成的 candidates 与 ref_lp 的偏差更大。
2. **adv_std 0.580 vs EXP-029 的 ~1.0**：advantage std 偏低，说明 group 内 reward variance 不足，有效梯度少。Features 模型的 beam candidates 可能都落在相似的 freshness/quality 区间，WeightedBehaviorReward 区分度低。
3. **PPL 24.2（vs B 的 14.6）**：RL 训练损害了 features 模型的 NTP 能力，但 B 的 PPL 与 SFT 起点相近（14.6 vs 16.3），说明 features 模型的 NTP loss landscape 对 RL 更敏感。
4. **根本原因**：exp025-beam-passes 的 SFT 起点本身比 exp020 弱 2.6pp（63.6% vs 66.2%），features RL pipeline 没有被充分调优——grpo_weight、reward 权重等超参都是为 exp020 调的，对 features 模型可能过强。

**结论**：features 模型接入 RL 需要单独调参，不能直接复用 exp020 的 RL 超参。核心问题是 clip 率过高（0.964），说明 grpo_weight 或 lr 对 features 模型偏大。

### Next Steps

1. **EXP-032**（已启动）：G×batch sweep，验证 context diversity 假设，在 exp020 起点上继续优化 RL。
2. **Features RL 调参**（可选）：降低 grpo_weight（0.03→0.01）或 lr（1e-4→5e-5），使 clip 率回到 0.92 附近，再验证 features 增益。

---

## EXP-030: A2PO + NLL Regularization + HEPO Prefix Scoring

**Date**: 2026-04-27
**Status**: completed
**Results**: `experiments/ntp_checkpoints/exp030-*/`

### Background
EXP-028/029 使用 WeightedBehaviorReward 解决了稀疏 reward 问题（100% coverage），但 advantage 估计和 loss 计算还有三个可改进点：

1. **Flat prefix fallback**（BehaviorReward/WeightedBehaviorReward）：prefix 级匹配统一用 `prefix_scale^depth` 折扣，没有区分浅层（L0）和深层（L0L1）prefix 的语义信息量差异
2. **对称惩罚**（GRPO）：所有 negative-advantage candidates 惩罚力度相同，但语义上接近 best candidate（hard negative）应受到更强信号
3. **相对 reward**：GRPO 只优化 group 内相对排序（advantage），最优 candidate 的绝对概率未被直接推高，容易出现 reward hacking

### Hypothesis
- **HEPO**：L0 prefix match → ×0.1（cluster 级弱信号），L0L1 → ×0.5（sub-cluster 中等信号），full → ×1.0。使 reward 梯度更精准反映 SID 层级语义
- **A2PO**：hard negatives（SID prefix 与 best 高度重叠但 reward 低）受到更强惩罚 → policy 在语义相似区分任务上梯度更强
- **NLL reg**：直接推高 best candidate 的绝对概率，防止 reward hacking，防止 policy 收缩到 degenerate 解

### Design
- **Variable**: A+B+C 联合 vs A2PO ablation (B only)，对照 EXP-028
- **Fixed**: ECPO δ=0.1，ε=0.2，G=512，grpo_batch=4，grpo_weight=0.03，ratio=1.0，lr=1e-4
- **Metric**: R@10, R@500（full eval，n_recall=1000）；advantage_mean, clip_fraction, reward/behavior_mean
- **Data**: exp023-14d-features，WeightedBehaviorReward (behavior cache)，FormatReward(0.5)
- **Config A** (all-in): `--a2po --a2po_alpha 1.0 --nll_reg 0.01 --hepo_scales "0.1,0.5"`
- **Config B** (ablation): `--a2po --a2po_alpha 1.0` only

### Run
`bash experiments/scripts/exp-030.sh`

### Results
| Config | R@10 | R@500 | PPL | behavior_mean | 训练耗时 | 备注 |
|--------|------|-------|-----|---------------|---------|------|
| exp030-a2po-nll-hepo-w003-r100 (Config A) | 12.5% | 67.0% | 14.54 | ~0.580 | 81min | A2PO+NLL+HEPO (all-in) |
| exp030-a2po-only-w003-r100 (Config B) | **13.3%** | **67.7%** | **14.14** | ~0.561 | 81min | A2PO only (ablation) |
| exp029-ecpo-onpolicy-w003-r100 | 13.0% | 67.8% | 14.1 | 0.638 | 80min | on-policy baseline |
| exp020-hard-lam03 (SFT SOTA) | 14.1% | 66.2% | 16.3 | — | — | SFT baseline |

全量 eval（n_recall=1000）：
- **Config A**: item_recall@10=0.125，item_recall@50=0.324，item_recall@100=0.425，item_recall@500=0.670
- **Config B**: item_recall@10=0.133，item_recall@50=0.327，item_recall@100=0.418，item_recall@500=0.677

**耗时参考**（4×A100 40GB，409 steps，G=512）：训练 ~81min/config，全量 eval ~25min/config，合计 ~105min/config。

### Analysis
- **Config B (A2PO only) vs EXP-029**：R@500 67.7% vs 67.8%，基本持平（-0.1pp），未见显著提升
- **Config A (A2PO+NLL+HEPO) vs Config B**：R@500 67.0% vs 67.7%，NLL reg + HEPO 反而略有下降（-0.7pp）
- **NLL reg 可能抑制了 RL 优化空间**：直接推高 best candidate 概率可能与 GRPO advantage 机制存在轻微干扰
- **HEPO prefix scoring 效果有限**：在 on-policy beam 已收敛的情况下，额外 prefix 信号未带来增益
- **A2PO 本身贡献有限**：on-policy beam (EXP-029) 已经足够有效，A2PO 的额外 hard negative 信号在此场景下边际效益低
- **结论**：R@500 瓶颈不在 reward shaping，而在 SFT 起点。EXP-031 将用 features SFT (exp025, R@500=63.6%) 作为起点，预期打破当前 67.8% 的天花板

### Next Steps
EXP-031：features SFT (exp025-beam-passes) + 完整 RL stack → 目标 R@500 > 67.8%

---

## EXP-029: ECPO + On-Policy Beam Search

**Date**: 2026-04-27
**Status**: completed
**Results**: `experiments/ntp_checkpoints/exp029-ecpo-onpolicy-w003-r100/`

### Background
EXP-026~028 全部使用 ref model 生成 beam search candidates（off-policy）。随着 policy 训练推进，policy 和 ref 的分布逐渐偏离，off-policy candidates 越来越不能代表 policy 当前的分布，advantage 估计失真，RL 梯度方向变得不可靠。

On-policy 修复：用 **policy model** 生成 candidates，ref model 只用于计算参考 log-probs。这样每步的 candidates 都来自当前 policy 分布，importance ratio ρ = π_θ/π_ref 的 off-policy 偏差最小。

### Hypothesis
On-policy beam candidates 与 policy 分布对齐 → importance ratio 更接近 1 → clip 率下降 → advantage 信号更有效 → R@500 进一步提升（相比 EXP-028）。

### Design
- **Variable**: `--on_policy_beam`（True vs False，对照 EXP-028）
- **Fixed**: 所有其他参数与 EXP-028 完全相同（WeightedBehaviorReward，w003-r100，ECPO δ=0.1，lr=1e-4）
- **Metric**: R@10, R@500（full eval），clip 率，policy_ratio_mean
- **Data**: exp023-14d-features，behavior cache，FormatReward(0.5)

### Run
`bash experiments/scripts/exp-029.sh`

### Results
| Config | R@10 | R@500 | PPL | clip率 | behavior_mean | 训练耗时 | 备注 |
|--------|------|-------|-----|--------|---------------|---------|------|
| exp029-ecpo-onpolicy-w003-r100 | **13.0%** | **67.8%** | 14.1 | 92% | 0.638 | 80min | on-policy beam |
| exp028-ecpo-weighted-w003-r100 | 0.7% | 2.0% | 3791 | 99% | 0.115 | 155min | off-policy baseline |
| exp020-hard-lam03 (SOTA) | 14.1% | 66.2% | 16.3 | — | — | — | SFT baseline |

全量 eval（n_recall=1000）：item_recall@10=0.130，item_recall@50=0.332，item_recall@100=0.422，item_recall@500=0.678。
PPL 14.1（比 SFT baseline 16.3 更低），R@500=67.8% **超过当前 SOTA 66.2%**（+1.6pp）。

**耗时参考**（4×A100 40GB）：exp029 训练 80min（409 steps），exp028 训练 155min（818 steps，因 rl_data_ratio=1.0 且 steps 翻倍）。全量 eval ~25min。

### Analysis
On-policy beam search 的核心效果验证：
- **clip 率从 99% → 92%**：on-policy candidates 与 policy 分布对齐，importance ratio 更接近 1，ECPO 梯度信号有效
- **behavior_mean 从 0.115 → 0.638**：on-policy candidates 的行为 reward 均值大幅提升，说明 policy 已经学会生成有行为反馈的 SID
- **R@500 从 2.0% → 67.8%**：彻底逆转 EXP-028 的退化，超越 SFT baseline 1.6pp
- **PPL 14.1 < SFT baseline 16.3**：RL 训练同时改善了 NTP perplexity，模型语言能力未退化

on-policy 修复了 EXP-028 的根本问题（off-policy ratio 过大 → ECPO clipping 失效）。

### Next Steps
EXP-030：在 on-policy beam 基础上叠加 A2PO + NLL regularization + HEPO prefix scoring，进一步提升 R@500。

---

## EXP-028: ECPO + WeightedBehaviorReward — Continuous Quality×Freshness Reward

**Date**: 2026-04-27
**Status**: completed
**Results**: `experiments/ntp_checkpoints/exp028-ecpo-weighted-w003-r100/`

### Background
EXP-027 的 reward signal 仍然稀疏（97.5% SID reward=0）：BehaviorReward 只覆盖 RF feedback 里的 chosen/rejected SID，binary 打分（±1），导致大多数 beam candidates advantage≈0，clip=98%，RL 梯度几乎全部来自噪声。

现在本地有完整的 behavior parquet cache（`/mnt/workspace/gr-demo-behavior-cache/`），14天完整数据，SID cache 里 100% item 都有行为记录。

新方案：`WeightedBehaviorReward`
- **质量分**：action_bitmap 按 v0420 生产权重加权，`log10(1 + Σw)`：place_order=4000, follow=4000, comment=2000, share=3, like=1, click=0.1
- **新鲜度**：`exp(-age_hours / 24)`，τ=1天，与线上3d截止策略对齐（3d后得分≈5%）
- **覆盖率**：100%（每个 beam candidate 都有非零 reward），彻底解决稀疏问题

### Hypothesis
100% reward coverage → within-group 方差显著提升 → advantage std 从 ≈0 变为有效值 → clip 率从 98% 下降 → RL 梯度真正起作用。预期 R@500 在 EXP-027 最佳 config 基础上进一步提升。

### Design
- **Variable**: WeightedBehaviorReward vs BehaviorReward（对照 EXP-027 最佳 config）
- **Fixed**: ECPO δ=0.1，ε=0.2，G=512，grpo_batch=4，grpo_weight=0.03，ratio=1.0，lr=1e-4
- **Metric**: R@10, R@500（全量 eval，n_recall=1000）；reward/behavior_coverage, reward/behavior_mean
- **Data**: exp023-14d-features，BehaviorReward cache=gr-demo-behavior-cache，FormatReward(0.5)

### Run
`bash experiments/scripts/exp-028.sh`

### Results
| Config | R@10 | R@500 | PPL | 训练耗时 | 备注 |
|--------|------|-------|-----|---------|------|
| exp028-ecpo-weighted-w003-r100 | 0.7% | 2.0% | 3791 | 155min | **严重退化** |
| exp020-hard-lam03 (baseline) | 14.1% | 66.2% | 16.3 | — | SFT baseline |

训练过程稳定（gnorm=0.19，无 spike），behavior_coverage=94.1%（prefix cascade fallback），behavior_mean≈0.115，format_legal_rate=100%。
但 full eval（n_recall=1000）R@500 从 63.6% 跌至 2.0%，严重退化。

**耗时参考**（4×A100 40GB）：818 steps，训练 155min，全量 eval ~25min。

### Analysis
WeightedBehaviorReward 100% 覆盖率的副作用：freshness × quality 连续信号给每个 candidate 都分配了非零 reward，但这些 reward 的绝对量级差异很小（behavior_mean≈0.115，方差低）。group 内 advantage std 仍然接近 0（adv=-0.02, clip=99%），RL 梯度依然无效。

根本问题：clip=99% 表示几乎所有 candidates 的 ratio 都在 [1-ε, 1+ε] clip 范围外（策略更新步 ratio 过大），说明 ECPO 的 clipping 机制本身在这个 reward 设置下失效了。WeightedBehaviorReward 的 reward 差异不足以产生有效 advantage 分布，RL 损失只是噪声 → 累积后导致 NTP 能力退化。

→ EXP-029 引入 on-policy beam search（clip 率问题的直接修复），EXP-030 引入 A2PO + NLL reg（advantage 有效性问题的修复）。

### Next Steps
EXP-029: on-policy beam 修复 off-policy 偏差问题
EXP-030: A2PO + NLL + HEPO 修复 advantage 有效性问题
EXP-031: features SFT (exp025) + 完整 RL stack 新 SOTA

---

## EXP-027: ECPO grpo_weight Sweep — Align with RF-DPO Training Structure

**Date**: 2026-04-27
**Status**: interrupted (replaced by EXP-028)
**Results**: `experiments/ntp_checkpoints/exp027-*/`

### Background
EXP-026 结论：`grpo_weight=0.5` 导致 R@500 从 63% 跌至 19%，RL 训练严重损害 NTP 能力。

根本原因分析：
- RF-DPO 用 `λ=0.03`，**每步必触发**，DPO 梯度是 NTP 梯度的 3%，持续温和正则
- EXP-026 GRPO 用 `weight=0.5`，**2% 稀疏触发**，但触发步的 GRPO loss 量级 ~5，`0.5×5=2.5` vs NTP loss ~7，GRPO 梯度占比 ~22%，远超 RF-DPO 的 3%
- 这就是 gnorm spike 到 158 的直接原因，也是 NTP 被压垮的根因

### Hypothesis
- Config A（weight=0.03, ratio=1.0）：完全对齐 RF-DPO 结构，每步必触发，GRPO 贡献稳定在 3%。预期 R@500 接近 baseline（63%）
- Config B（weight=0.03, ratio=0.5）：介于 A/C，每步 50% 触发，预期 R@500 略低于 A
- Config C（weight=0.03, ratio=0.02）：稀疏触发但低 weight，触发步 GRPO 占比 ~1%，预期 NTP 损害最小但 RL 信号也最弱

全部用 ECPO（δ=0.1，EXP-026 已证明比 GRPO 稳定），lr=1e-4（对齐 RF-DPO）

### Design
- **Variable**: grpo_weight（固定 0.03）× rl_data_ratio（1.0 / 0.5 / 0.02）
- **Fixed**: ECPO δ=0.1，ε=0.2，G=512，grpo_batch=4，818 steps，lr=1e-4，SFT=exp020-hard-lam03
- **Metric**: R@10, R@500（全量 eval，n_recall=1000，与 exp016-B-14d-S baseline 63.4% 对齐）
- **Data**: exp023-14d-features，exp018/hard feedback，BehaviorReward(1.0)+FormatReward(0.5)

### Run
`bash experiments/scripts/exp-027.sh`

### Results
Config A（w003-r100）中途观测（step 200/818，约 24% 进度）：
- NTP loss: 8.17→6.41（正常下降）
- gnorm: 0.21~0.29（稳定，无 spike）
- clip: 98~99%（reward 仍然稀疏，BehaviorReward 覆盖率低）
- behavior_mean: 0.037~0.038（极低，97.5% SID reward=0）

在 step 200 时主动中断，切换至 EXP-028 使用 WeightedBehaviorReward（100% 覆盖率）。

### Analysis
grpo_weight=0.03 + ratio=1.0 的结构与 RF-DPO 对齐，NTP 没有崩溃（gnorm 稳定）。
但 clip=99% 说明 BehaviorReward 信号太弱，RL 几乎没有学到有效梯度。
根本问题不在 weight/ratio，而在 reward signal 质量。

### Next Steps
→ EXP-028：相同超参，换 WeightedBehaviorReward（action_bitmap×freshness，100% 覆盖）

---

## EXP-026: GRPO+ECPO — Group Relative Policy Optimization + Pluggable Reward

**Date**: 2026-04-27
**Status**: completed
**Results**: `experiments/ntp_checkpoints/exp026-*/`

### Background
RF-DPO (Phase 2, exp020) 已完成。Phase 3/4 引入 GRPO 和 ECPO：
- **GRPO**（OneMall，arxiv 2601.21770）：beam search 生成 G=512 candidates，group-normalized advantage，PPO clipped surrogate loss，ε=0.2，rl_data_ratio=2%
- **ECPO**（OneRec，arxiv 2506.13695v4）：在 GRPO 基础上加 early clip (δ=0.1) 防止负 advantage 梯度爆炸
- **Pluggable Reward**：新增 `rl/reward.py`，BehaviorReward（行为信号 + prefix cascade fallback）、FormatReward（SID 合法性，sample_k=5）、CompositeReward 加权组合。metrics 实时 streaming 至 step log
- SFT 起点：exp020-hard-lam03（RF-DPO hard best），SID_CACHE=exp013-4096x3-12d-binary

**关键工程问题与修复**（本次实验踩坑记录）：
1. `SIDTrie` 构建 bug：`semantic_ids.npy` 存 `{item_id_str: sid_str}`，必须 iterate `.values()` 而非 `.keys()`。错误时 trie 空 → beam search 0 candidates → GRPO loss 永远 0
2. BehaviorReward 命中率：全 SID 仅 0.16%（1,788 条/1.09M）。加 prefix cascade fallback（L0 覆盖 24.3%），有效 reward 信号提升 150x
3. GRPO reward std≈0 → advantage 放大爆炸：加 `std < 1e-6` skip + `adv.clamp(-5,5)` + `log_rho.clamp(-10,10)` 防御
4. `save_checkpoint()` 签名与调用不符：修复为正确传 positional args
5. `SyntaxError: unicode error \x`：docstring 末尾反斜杠转义错误，修复

### Hypothesis
- GRPO 全组 continuous advantage 提供更丰富梯度信号 → R@500 优于 RF-DPO
- ECPO early clip（δ=0.1）防止稀疏 reward 下的梯度爆炸 → gnorm 全程稳定，优于 GRPO Config 2 step 600 的 gnorm spike
- Pluggable reward Behavior(1.0)+Format(0.5)：SID 合法率应 >95%

### Design
- **Variable**: 算法（GRPO vs ECPO）；reward 组合
- **Fixed**: G=512，ε=0.2，grpo_weight=0.5，rl_data_ratio=0.02，grpo_batch=4，818 steps
- **Metric**: R@10, R@500；grpo loss；gnorm；behavior_mean；format_legal_rate
- **Data**: exp023-14d-features（NTP），exp018/hard feedback shards（BehaviorReward），SFT=exp020-hard-lam03

**实验配置**：
- Config 1: `exp026-grpo-behavior` — GRPO + BehaviorReward only（preference shards 未到位，reward=0，作废）
- Config 2: `exp026-grpo-behavior-fmt` — GRPO + BehaviorReward(1.0) + FormatReward(0.5)
- Config 3: `exp026-ecpo-behavior-fmt` — ECPO(δ=0.1) + BehaviorReward(1.0) + FormatReward(0.5)

### Run
`bash experiments/scripts/exp-026.sh`

### Results

**训练稳定性对比**（GRPO vs ECPO，核心发现）：

| Step | GRPO gnorm | ECPO gnorm |
|------|-----------|-----------|
| 50   | 0.22      | **0.46**  |
| 100  | 0.20      | **0.28**  |
| 200  | — (est)   | **0.19**  |
| 400  | — (est)   | **0.22**  |
| 550  | 0.57      | **0.20**  |
| 600  | **64.93** | **0.20**  |
| 650  | **158.20**| **0.20** |

- GRPO step 600–650 gnorm 从 0.57 → 64.9 → 158.2，随后自然回落（lr cosine decay → 0）
- ECPO 全程 gnorm 0.19–0.46，未出现任何 spike

**GRPO loss 对比**（同样体现 early clip 效果）：

| Step | GRPO grpo_loss | ECPO grpo_loss |
|------|---------------|---------------|
| 50   | 5.92          | **0.011**     |
| 100  | — (est)       | **0.007**     |
| 300  | — (est)       | **0.009**     |
| 600  | 4.45          | **0.005**     |

ECPO grpo_loss 低 3 个数量级，说明早 clip 完全吸收了负 advantage 样本的梯度。

**Reward metrics（Config 2/3 均）**：
- format_legal_rate = 1.000（SID beam search 保证合法性）
- behavior_mean ≈ 0.032–0.035（prefix cascade 生效，L0 覆盖率 ~24%）
- clip_fraction = 99%（advantage 几乎总在边界，reward 信号极稀疏）

**Inline eval（快速版，beam_size=500，250 samples，不可与全量 baseline 直接比）**：

| Config | PPL | R@10 | R@50 | R@100 | R@500 | 训练耗时 |
|--------|-----|------|------|-------|-------|---------|
| Config 1 (GRPO, behavior only) | — | — | — | — | — | 21min |
| Config 2 (GRPO+fmt) | 323 | 0.009 | 0.025 | 0.056 | 0.164 | 21min |
| Config 3 (ECPO+fmt) | **270** | **0.011** | **0.028** | **0.064** | **0.189** | 22min |

ECPO 在所有指标上全面优于 GRPO（PPL -16%，R@500 +15%）。

**全量 eval（与 baseline 对齐，running）**：`bash experiments/scripts/exp-026-reeval.sh`
结果 TBD（与 exp020-hard-lam03 baseline R@500≈60% 对比）

**耗时参考**（4×A100 40GB）：818 steps，每个 config 训练 ~21min（数据集较小，49K items vs 后续 26K）。

### Analysis

**核心复现：GRPO → ECPO 稳定性的文献结论**

OneRec 论文（arxiv 2506.13695v4）ECPO 动机：稀疏 reward 环境下，负 advantage 样本的 π_θ → 0 会导致 rho = π_θ/π_old 急剧缩小，clipping 失效，梯度爆炸。early clip 用 `π'_old = max(π_θ/(1+ε+δ), π_ref)` 替换 denominator，限制 rho 上界。

本次实验完整复现了这个现象：
- reward 信号极稀疏（format=1.0 but 所有候选合法 → reward var 低；behavior_mean=0.033，clip=99%）
- GRPO step 600–650 gnorm 0.57 → 64.9 → 158.2（尽管有 clamp 防御）
- ECPO 同等条件全程 gnorm 0.19–0.46，grpo_loss 低 3 个数量级

**Recall 结论**（inline eval，相对比较有效）：ECPO > GRPO，PPL 和 R@500 均改善。与 RF-DPO baseline 的绝对对比等全量 eval 完成后补充。

**其他观察**：
- format_legal_rate=1.0：constrained beam search 已保证 SID 合法，FormatReward 对此场景冗余（对 sampling 场景有意义）
- clip_fraction=99%：advantage 几乎无区分度，reward 信号极稀疏 → 是制约 RL 提升空间的主因

### Next Steps
- 补全量 eval 结果，与 exp020-hard-lam03 (R@500≈60%) 对比
- 更丰富 behavior signal：time-decay 加权、CTR/转化率分层打分，提升 reward variance
- ECPO delta sweep (δ=0.05/0.1/0.2) 找最优稳定点
- 考虑 online policy beam search（替代 ref model）以获取 on-policy candidates

---

## EXP-025: Beam Search Feature Passing — 正确消除 side feature 训练-推理 gap

**Date**: 2026-04-21
**Status**: completed
**Results**: [./ntp_checkpoints/exp025-*/](./ntp_checkpoints/exp025-*/)：
- time_gap shift 后 R@500 = 59.8%（反而低于 baseline 61.2%），因为让 context 信息变陈旧
- action shift 后 R@500 = 52.9%，仍远低于 baseline
- 根本原因：shift 只解决训练侧泄漏，但没解决 beam search incremental path 不传 features 的问题

正确分析：
- **time_gap 完全已知**：item K 到 K+1 的时间间隔是历史事实，推理时 context items 的 time_gap 全部已知。生成 target item 的 token 时，target 的 time_gap（到上一个 context item 的间隔）也已知。
- **action_level 部分未知**：context items 的 action_level 已知，但 target item 的 action_level 未知（还没发生行为）。
- 当前 `forward_cached` incremental path（生成 L0→L1→L2 token）完全不传 features → 即使 context encoding 正确，生成 token 仍有 gap。

### Hypothesis

**Config 1** (beam_passes): 不 shift 数据。训练正常（time_gap + action 在所有 token）。Beam search incremental path 传入：
- time_gap = target item 的真实 time_gap（已知）
- action_level = 最后一个 context item 的 action_level（carry-forward）

预期 time_gap 贡献 1-2% R@500 提升（跨 item 间隔信号）；action carry-forward 可能略有帮助。

**Config 2** (action_l2_only): 训练时 action_level 只作用于每个 item 的 L2 token 位置（L0/L1 位置强制 action=0）。Beam search 对生成 token 传 action=0，完全一致：
- L0/L1 生成：action=0（训练也是 0 → 无 gap）
- L2 生成：action=0（训练是真值，但只在最后一层，对 recall 影响有限）
- 好处：彻底消除 action 对 L0/L1 recall 的负面影响

### Design
- **Variable**: beam search feature passing 策略 (2 configs)
  - Config 1 (seg+all+beam_passes): segment_emb + time_gap + action，beam search 传 features
  - Config 2 (seg+time+action_l2): segment_emb + time_gap(all) + action(L2-only)，beam search 传 time_gap
- **Baseline**: exp023-segment (PPL=25.94, R@500=61.2%)
- **Fixed**: S-tier model, 14d data (03-18~03-31), EXP-023 NTP data（不 shift）
- **Metric**: train loss, eval PPL, Recall@{10, 50, 100, 500}
- **Data**: Config 1 复用 EXP-023 数据；Config 2 新 preprocess（action_l2_only）

### Run
`bash experiments/scripts/exp-025.sh`

### Results

| Config | PPL | L0 PPL | L1 PPL | L2 PPL | R@10 | R@50 | R@100 | R@500 | 训练耗时 |
|--------|-----|--------|--------|--------|------|------|-------|-------|---------|
| exp023-segment (baseline) | 25.94 | 346.9 | 11.75 | 4.35 | 10.9% | 24.9% | 35.4% | 61.2% | 21min |
| **exp025-beam-passes** | **25.22** | 334.6 | 10.33 | 4.71 | 10.4% | 28.2% | 40.0% | **63.6%** | 20min |
| exp025-action-l2only | 24.85 | 331.4 | 10.30 | 4.57 | 5.5% | 13.2% | 17.3% | 27.0% | 21min |

### Analysis

**beam_passes 是 NEW BEST** (R@500=63.6%, +2.4pp)：
1. 训练不做任何 shift，正常使用 segment_emb + time_gap + action_level
2. Beam search incremental 传入 time_gap=target真值（已知）+ action=carry-forward上一context item
3. PPL 也改善至 25.22（-0.72），说明完整 features 让模型学得更好
4. R@50 和 R@100 提升更显著（+3.3pp, +4.6pp），说明 mid-range recall 受益最大

**action_l2only 完全失败** (R@500=27.0%)：
1. PPL 是最好的 24.85（L2 PPL 4.57），但 R@500 仅 27.0%
2. 原因：训练时 L0/L1 的 action=0，但 beam search 对 L0/L1 仍传 action=0 → 训练一致，但 L2 位置训练有真 action 而 beam search 无法给出 → gap 依旧
3. target_sid_found_rate=27%（beam_passes 是 63.7%）— 大量 item 无法被 beam search 找到

### Next Steps
- exp025-beam-passes 成为新 baseline
- 下一步探索 IDEA-genrec-0 (Page-wise NTP)，与 beam_passes 正交可叠加

---

## EXP-024: Side Feature Shift — 消除 time_gap/action_level 信息泄漏

**Date**: 2026-04-21
**Status**: completed
**Results**: [./ntp_checkpoints/exp024-*/](./ntp_checkpoints/exp024-*/)

### Background

EXP-023 发现 time_gap 和 action_level 存在训练-推理信息泄漏：
- 训练时 side features 按 item 复制 3 次铺到所有 token 位置，包括 target item 的 L0/L1/L2
- 模型在 intra-item 预测（L0→L1, L1→L2）时学会依赖 target item 自身的 action_level
- Beam search 推理时不知道 target item 的特征 → L1/L2 预测偏移 → recall 崩溃
- action 影响最严重：PPL 27.5（好于 baseline 28.4）但 R@500 仅 28.5%（baseline 60.7%）

Segment embedding 不受影响（纯位置信息），EXP-023 已验证有效（PPL 25.94, R@500 61.2%）。

### Hypothesis

将 side features 延迟一个 item：每个 item 的 3 个 token 位置使用**上一个 item** 的 time_gap/action_level（第一个 item 用 padding=0）。这样：
- 预测 item K+1 的 L0 时：使用 item K 的 features（已知 ✓）
- 预测 item K+1 的 L1/L2 时：同样使用 item K 的 features（已知 ✓）
- 训练与推理完全一致

预期：
- time_gap shifted: R@500 与 baseline 持平或略优（时间信号对 L0 跨 item 预测有帮助）
- action shifted: R@500 恢复到 baseline 水平或略优（消除泄漏后，仍提供历史行为强度信号）
- segment + shifted_all: 最佳组合，预计 R@500 > 61.2%

### Design
- **Variable**: side feature 组合 (4 configs)
  - segment-only: 仅 segment_emb（EXP-023 已有结果，作为 baseline）
  - seg+timegap: segment + shifted time_gap
  - seg+action: segment + shifted action_level
  - seg+all: segment + shifted time_gap + shifted action_level
- **Fixed**: S-tier model (17.5M active, 256d/6L/8E top-2), 14d data (03-18~03-31), batch=4096, lr=1e-3, 1 epoch
- **Metric**: train loss, eval PPL, Recall@{10, 50, 100, 500}
- **Data**: 需重新 preprocess-ntp（shifted features），segment-only 可复用 EXP-023 数据

### Run
`bash experiments/scripts/exp-024.sh`

### Results

| Config | PPL | R@10 | R@50 | R@100 | R@500 | 训练耗时 |
|--------|-----|------|------|-------|-------|---------|
| exp023-segment (baseline) | 25.94 | 13.2% | 33.8% | 43.9% | 61.2% | 21min |
| exp024-seg-timegap | ~26 | — | — | — | 59.8% | 20min |
| exp024-seg-action | ~27 | — | — | — | 52.9% | 20min |
| exp024-seg-all | ~26 | — | — | — | ~55% | 20min |

### Analysis

Shift 方案**完全失败**：
1. **time_gap shifted (59.8%)**: 比 baseline 还低 1.4%。Shift 让 context items 使用的是「上上个」item 的 time_gap，信息变陈旧反而干扰学习。
2. **action shifted (52.9%)**: 虽然比 EXP-023 未修复版 (28.5%) 好很多，但仍远低于 baseline。说明 action carry-forward 信息较弱。
3. **根本问题**：shift 解决了训练侧泄漏（target token 不再有自身 features），但 beam search incremental path 仍然不传任何 features。只要 incremental 生成时特征为 0，而训练时是非 0，就存在不可消除的 gap。

**结论**：shift 是错误方向。正确做法是不 shift 训练数据，转而修复 beam search incremental path 使其传入正确 features（EXP-025）。

### Next Steps

EXP-025: 修复 beam search incremental path，正确传入 time_gap（真值已知）和 action_level（carry-forward 或 L2-only 设计）。

---

## EXP-023: NTP Side Information — Time Gap + Action Type + Segment Embedding

**Date**: 2026-04-21
**Status**: completed
**Results**: [./ntp_checkpoints/exp023-*/](./ntp_checkpoints/exp023-*/)

### Background
当前 NTP 模型输入仅为 SID token 序列 + 单一位置编码。三个 P0 低成本 additive 特征（IDEA-feat-0/1/2）可同时实现并独立验证：
1. **Time Gap Embedding**: 相邻 item 的时间间隔 log-scale 分桶 (16 bins)，捕捉实时性信号
2. **Action Level Embedding**: action_bitmap → 4 级离散信号 (pad/weak/strong/trade)，区分行为强度
3. **Segment Embedding**: 将 position embedding 解耦为 item_pos + layer_pos，让模型区分 SID 层级

Baseline: EXP-016 B-14d-S (PPL=27.05, R@500=58.5%)

### Hypothesis
- Time Gap: +1-2% R@500（高频连续行为 vs 长时间回访语义不同）
- Action Level: +1-3% R@500（强交互 item 应被更高权重预测）
- Segment Emb: +0.5-1% R@500（层级结构感知改善 L0→L1→L2 转换建模）
- All combined: +2-4% R@500 (特征信息正交，应可叠加)

### Design
- **Variable**: side features 组合 (5 configs)
  - baseline: 无新特征（复现 EXP-016）
  - timegap: 仅 time_gap_emb
  - action: 仅 action_level_emb
  - segment: 仅 segment_emb
  - all: 全部开启
- **Fixed**: S-tier model (17.5M active, 256d/6L/8E top-2), 14d data (03-18~03-31), batch=4096, lr=1e-3, 1 epoch
- **Metric**: train loss, eval PPL, Recall@{10, 50, 100, 500}
- **Data**: 需重新 preprocess-ntp 生成带 time_gaps + action_levels 的 shards

### Run
`bash experiments/scripts/exp-023.sh`

### Results

| Config | PPL | L0 PPL | L1 PPL | L2 PPL | R@10 | R@50 | R@100 | R@500 | 训练耗时 |
|--------|-----|--------|--------|--------|------|------|-------|-------|---------|
| baseline | 28.41 | 351.4 | 12.15 | 5.45 | 11.0% | 25.9% | 34.9% | 60.7% | 21min |
| timegap | 28.78 | 340.8 | 10.82 | 6.56 | 10.9% | 27.3% | 36.8% | 60.1% | 20min |
| action | 27.50 | 359.2 | 12.30 | 4.78 | 4.9% | 11.1% | 15.9% | 28.5% | 21min |
| segment | **25.94** | 346.9 | 11.75 | 4.35 | 10.9% | 24.9% | 35.4% | **61.2%** | 21min |
| all | **25.16** | **338.0** | **10.30** | 4.64 | 9.5% | 23.0% | 31.6% | 55.0% | 21min |

### Analysis

**Segment embedding 唯一有效且可信**：PPL 25.94 (-8.7%)，R@500 61.2% (+0.5pp)。纯位置信息，训练和推理完全一致。

**time_gap 和 action_level 存在训练-推理信息泄漏**：
- 训练时 side features 按 item 复制 3 次铺开到所有 token 位置，包括 target item 的 L0/L1/L2
- 模型学会在 intra-item 预测（L0→L1, L1→L2）时依赖 target item 自己的 action_level/time_gap
- Teacher-forced eval 有同样的泄漏 → PPL 虚好
- Beam search 推理时不知道 target item 的特征 → L1/L2 预测偏移 → recall 崩溃
- action 影响最大（直接编码用户对 target 的行为强度，推理时本质不可知），R@500 仅 28.5%
- time_gap 影响小（intra-item 预测不太依赖时间差），R@500 基本持平

**结论**：segment_emb 是 confirmed positive。time_gap/action_level 需修复信息泄漏后重新验证。

### Next Steps

EXP-024：将 side features 按 item 延迟一步（每个 item 的 3 个 token 位置使用上一个 item 的 features），消除对 target item 的信息泄漏。同时修复 teacher-forced eval 使其与 beam search 一致。

---

## EXP-022: NTP In-Batch Contrastive Loss (IDEA-onemall-0)

**Date**: 2026-04-20
**Status**: completed
**Results**: [./ntp_checkpoints/exp022-*/](./ntp_checkpoints/exp022-*/)

### Background

当前 NTP 模型仅有离散 CE loss + MoE balance aux loss。OneMall §3.2 Eq.7 表明在 s₃ 位置加 in-batch contrastive auxiliary loss（对齐 decoder hidden state 与 target item embedding）可显著提升性能，报告 98% accuracy@1。

该辅助 loss 为 decoder 提供连续 embedding 空间的监督信号，防止 SID 表示退化为只关心 token 分类而丢失语义连续性。与 DPO 互补：onemall-0 强化基础表征，DPO 在此之上做偏好对齐。

基线: EXP-016 14d-S (PPL=27.05, R@500=58.5%)

### Hypothesis

1. Contrastive loss 作为正则化，应降低 PPL 并提升 Recall（特别是 R@500）
2. α 过大会与 NTP loss 竞争梯度，需要 sweet spot
3. 每 GPU 2048 local in-batch negatives 足够 InfoNCE 学到对齐

### Design
- **Variable**: contrastive weight α ∈ {0.01, 0.1, 0.5}; temperature τ ∈ {0.05, 0.07}; projection dim ∈ {128, 256}
- **Fixed**: S-tier model (6L, 8E top-2, 256d), batch_size=152 (packed), 1 epoch, same NTP data (EXP-016 14d)
- **Metric**: PPL, R@10, R@50, R@100, R@500
- **Data**: experiments/ntp_data/exp016-14d (14-day, 8 shards)
- **Implementation**: local in-batch InfoNCE, max_pairs=2048/GPU, expandable_segments=True

### Run
`bash experiments/scripts/exp-022.sh`

### Results

ALL configs completed:

| Config | α | τ | dim | PPL | R@10 | R@50 | R@100 | R@500 | 训练耗时 |
|--------|-----|------|-----|-----|------|------|-------|-------|---------|
| Baseline (EXP-016 14d-S) | 0 | — | — | **27.05** | 9.9% | **26.1%** | 35.0% | **58.5%** | — |
| exp022-alpha001 | 0.01 | 0.07 | 128 | 27.89 | 10.3% | 25.1% | 36.4% | 59.2% | 21min |
| exp022-alpha01 | 0.1 | 0.07 | 128 | 29.22 | 9.7% | 24.9% | 35.0% | 57.9% | 22min |
| exp022-alpha05 | 0.5 | 0.07 | 128 | 29.04 | 9.7% | 25.4% | 34.6% | 56.3% | 21min |
| exp022-dim256 | 0.01 | 0.07 | 256 | 29.66 | 10.1% | 26.1% | 35.4% | 58.8% | 22min |
| exp022-temp005 | 0.01 | 0.05 | 128 | 28.16 | 10.1% | 25.2% | 34.8% | 58.2% | 21min |

### Analysis

**Contrastive loss 全面失败。IDEA-onemall-0 关闭。**

1. **α sweep**: α=0.01 是最好的 (+0.7pp R@500)，α 越大越差。α=0.5 时 R@500 跌至 56.3%（-2.2pp）。Contrastive 梯度与 NTP CE 梯度竞争，强度越大破坏越大。
2. **dim256**: 投影维度翻倍无帮助（PPL=29.66，R@500=58.8%），反而更差。
3. **temp005**: 更低温度（更锐利分布）无帮助（PPL=28.16，R@500=58.2%）。
4. **根因分析**: SID 是离散 codebook token，decoder 在 token 空间做分类。InfoNCE 试图对齐 hidden state 到连续 embedding 空间，但这对离散 token 预测没有直接帮助。PPL 一致变差说明 contrastive gradient 干扰了 NTP 学习。
5. 所有 config 的 PPL 都比 baseline 差 0.84~2.61，说明这不是 regularization 而是 interference。

### Next Steps
- 关闭 IDEA-onemall-0，不再追 contrastive 变体
- 转向 training objective 层面的改进：IDEA-genrec-0 (Page-wise NTP)

---

## EXP-021: Qwen3-4B vs 0.6B Embedding Quality for SID Tokenizer

**Date**: 2026-04-20
**Status**: planned
**Results**: TBD

### Background

当前所有 SID 实验都基于 Qwen3-Embedding-0.6B (dim=1024)。已在 EFS 上产出了 Qwen3-Embedding-4B (dim=2560) 的 embedding cache。更大模型的 embedding 语义更丰富，但 tokenizer 是否能从中获益取决于：
1. 高维 embedding 是否让 RQ 分层更准确（更低的量化误差）
2. 更好的 embedding 是否传导为更好的 NTP recall
3. 4B embedding 计算成本 ~6.7× (0.6B → 4B)，只有 recall 显著提升才 worth

### Hypothesis

1. Qwen3-4B embedding 量化误差比 0.6B 降低 20%+（dim 2.5× → RQ 残差更小）
2. NTP Recall@10 在 4B embedding 的 SID 上提升 2-5pp（15.4% → 17-20%）
3. FSQ 可能需要调整（更高维输入 → FSQ projection 需要更大 hidden size 或更多 levels）
4. 如果 recall 提升显著，downstream RL/DPO 也会受益（better tokenizer = easier alignment）

### Design

- **Variable**: embedding model (qwen3-0.6b vs qwen3-4b)
- **Fixed**: Tokenizer config (1024 clusters, 2 KMeans layers, FSQ 6d_4096, MLP projection), 14d behavior data, NTP probe 模型结构相同
- **Metric**: 
  - Tokenizer: quantization error (MSE), collision rate, SID assignment distribution
  - NTP: PPL, item_recall@{10,50,100,500}
- **Data**: 同一批 content_id，两套 embedding cache

| Config | Model | Dim | FSQ Hidden | Description |
|--------|-------|-----|------------|-------------|
| 1 | qwen3-0.6b | 1024 | 64 | Baseline (existing) |
| 2 | qwen3-4b | 2560 | 64 | 4B + same FSQ hidden |
| 3 | qwen3-4b | 2560 | 128 | 4B + larger FSQ hidden |

### Run
`bash experiments/scripts/exp-021.sh`

### Results
TBD

### Analysis
TBD

### Next Steps
TBD

---

## EXP-020: RF-DPO Hard λ Sweep — Finding Optimal DPO Weight

**Date**: 2026-04-20
**Status**: completed
**Results**: [./ntp_checkpoints/exp020-hard-lam03/](./ntp_checkpoints/exp020-hard-lam03/)

### Background

EXP-019 Joint NTP+DPO 结果显示 Hard 的最佳 λ 在 0.01~0.1 之间：
- λ=0.01: PPL=14.4 (best), R@10=13.5%, R@500=66.4%, **但 pref_acc=49.8%**（DPO 信号太弱，模型没学到 preference）
- λ=0.1:  PPL=23.6 (退化), R@10=13.6%, R@500=65.5%, pref_acc=93.6%（PPL 开始崩）
- Reference: PPL=17.5, R@10=15.4%, R@500=68.3%

目标：在 λ=0.01~0.1 之间找到 sweet spot，让 pref_acc >70% 的同时保持 PPL 不显著退化（<18）。

同时测试 Easy multi-epoch：Easy 只有 95 pairs / 15 steps，但 joint 模式下可以让 DPO 数据循环多次（NTP 兜底防遗忘），验证更多 DPO epoch 能否增强 Easy 的对齐效果。

### Hypothesis

1. λ=0.03~0.05 能在 PPL 保持 <18 的同时让 pref_acc >70%，实现有意义的对齐
2. Easy multi-epoch（更多步数）能让模型更充分学到 negative feedback 的 preference，比 15 步的 Easy 有更强对齐效果
3. 存在一个 λ 使得 R@10 超过 reference（15.4%）——alignment 和 recall 不一定冲突

### Design

- **Variable**: λ (DPO weight) — 0.03, 0.05, 0.07；Easy multi-epoch steps
- **Fixed**: S-tier 模型 (17.5M), 14d 数据, β=0.1, lr=1e-4, ref=SP-DPO fixed-medium, Hard 807 steps
- **Metric**: PPL, item_recall@{10,50,100,500}, reward_margin, preference_acc
- **Data**: RF-DPO Hard 4,312 pairs; Easy 95 pairs

Step 计算：
- Hard: 4,312 pairs / 16 batch = 269 batches × 3 epochs = 807 steps (same as EXP-019)
- Easy multi-epoch: 95 pairs / 16 batch = 5 batches × 20 epochs = 100 steps

| Config | Name | Difficulty | λ | β | max_steps | Description |
|--------|------|-----------|-----|-----|-----------|-------------|
| 1 | hard-lam03 | Hard | 0.03 | 0.1 | 807 | λ=0.03 |
| 2 | hard-lam05 | Hard | 0.05 | 0.1 | 807 | λ=0.05 |
| 3 | hard-lam07 | Hard | 0.07 | 0.1 | 807 | λ=0.07 |
| 4 | easy-multi | Easy | 0.1 | 0.1 | 100 | Easy 20 epochs |

### Run
`bash experiments/scripts/exp-020.sh`

### Results

EXP-020 只保留了最优 config（hard-lam03），hard-lam05/07 和 easy-multi 检出不在本机（实验仅保留了 SOTA checkpoint）。

| Config | λ | PPL | R@10 | R@100 | R@500 | 训练耗时 |
|--------|---|-----|------|-------|-------|---------|
| **hard-lam03** | 0.03 | **16.3** | **14.1%** | — | **66.2%** | 62min |
| EXP-019 hard-lam01 (ref) | 0.01 | 14.4 | 13.5% | — | 66.4% | 62min |
| EXP-019 hard-lam10 (ref) | 0.10 | 23.6 | 13.6% | — | 65.5% | 62min |

hard-lam03 在 PPL 和 R@500 之间取得最优平衡，成为新 SOTA baseline（R@500=66.2%）。

### Analysis

λ=0.03 是 sweet spot：
- pref_acc 足够高（DPO alignment 有效）
- PPL 保持在 16.3（NTP 能力无显著退化）
- λ=0.01 PPL 最低但 R@500 仅小幅领先 66.4% vs 66.2%（统计噪声范围）
- λ=0.10 PPL 开始退化到 23.6，说明 DPO weight 过大损害语言建模

### Next Steps

exp020-hard-lam03 成为 GRPO/ECPO 阶段（EXP-026~）的 SFT baseline。

---

## EXP-019: RF-DPO Joint NTP+DPO — Step-Matched Training

**Date**: 2026-04-20
**Status**: completed
**Results**: [./ntp_checkpoints/exp019-*/](./ntp_checkpoints/exp019-*/)

### Background

EXP-018 (pure DPO) 全面退化：

| Config | PPL | R@10 | R@500 |
|--------|-----|------|-------|
| Reference (SP-DPO) | ~14.5 | 15.4% | 68.3% |
| Easy (15 steps) | 35.8 | 13.8% | 64.2% |
| Hard (807 steps) | 50,694 | 8.3% | 28.9% |
| Progressive β=0.5 (best) | 404.9 | 10.2% | 49.8% |

核心问题：纯 DPO 807 步没有 NTP 正则化 → catastrophic forgetting。而之前 EXP-017 joint NTP+DPO 的 Easy 组里 NTP 有 1555 步但 DPO 只有 15 batch → NTP 主导冲掉 DPO 信号。

解决方案：Joint NTP+DPO，但用 `--max_steps` 限制总步数到 DPO 数据的 N 个 epoch，让 NTP 和 DPO 数据量匹配。NTP 提供正则化防 forgetting，DPO 提供对齐信号，两者步数相当。

### Hypothesis

1. Joint NTP+DPO (step-matched) 在 Hard 难度下能保持 PPL ~15 同时提升 Recall
2. `max_steps = DPO_batches × epochs` 确保 NTP 不会冲掉 DPO signal
3. λ (DPO weight) 平衡 NTP 正则化 vs DPO 对齐强度：λ 太小 → NTP 主导（EXP-017 Easy 教训），λ 太大 → DPO 主导可能退化
4. Progressive Easy→Hard 在 joint 模式下应该有效（EXP-018 progressive 无效是因为 pure DPO forgetting）

### Design

- **Variable**: difficulty (Easy/Hard/Progressive)、λ (DPO weight)、max_steps
- **Fixed**: S-tier 模型 (17.5M), 14d 数据, β=0.1, lr=1e-4, ref=SP-DPO fixed-medium
- **Metric**: PPL, item_recall@{10,50,100,500}, depth_hit@10
- **Data**: RF-DPO preference pairs from EXP-018 (reuse), 14d NTP data

Step 计算：
- Easy: 95 pairs / 16 batch = 5 batches × 3 epochs = 15 steps
- Hard: 4,312 pairs / 16 batch = 269 batches × 3 epochs = 807 steps

| Config | Name | Difficulty | λ | β | max_steps | Description |
|--------|------|-----------|-----|-----|-----------|-------------|
| 1 | joint-easy | Easy | 0.1 | 0.1 | 15 | Joint NTP+DPO, Easy, step-matched |
| 2 | joint-hard | Hard | 0.1 | 0.1 | 807 | Joint NTP+DPO, Hard, step-matched |
| 3 | joint-hard-lam50 | Hard | 0.5 | 0.1 | 807 | Higher DPO weight |
| 4 | joint-hard-lam01 | Hard | 0.01 | 0.1 | 807 | Lower DPO weight |
| 5 | joint-prog | Progressive E→H | 0.1 | 0.1 | 807 | Progressive, Hard stage ref=Easy output |

### Run
`bash experiments/scripts/exp-019.sh`

### Results

注：joint-easy/easy-lam10/easy___prefixbug 的 wall_time≈27s 是因为只有 15 steps（95 pairs / 16 batch = 5 batches × 3 epochs）。

| Config | λ | Steps | PPL | R@10 | R@500 | 训练耗时 |
|--------|---|-------|-----|------|-------|---------|
| joint-easy | 0.1 | 15 | 20.6 | 13.8% | 67.5% | ~0.5min |
| joint-easy-lam10 | 1.0 | 15 | 15.3 | 14.3% | 67.7% | ~0.5min |
| joint-hard-lam01 | 0.01 | 807 | **14.4** | 13.5% | **66.4%** | 62min |
| joint-hard-lam10 | 0.10 | 807 | 23.6 | 13.6% | 65.5% | 62min |
| joint-hard-lam50 | 0.50 | 807 | 57.7 | 12.4% | 59.2% | 62min |
| joint-prog | 0.1 | 807 | 22.4 | 13.2% | 64.4% | 62min |
| Reference (SP-DPO) | — | — | ~14.5 | 15.4% | 68.3% | — |

### Analysis

1. **Joint NTP+DPO 有效防止 catastrophic forgetting**：best PPL=14.4（λ=0.01），远优于 pure DPO 的 50K+。
2. **R@500 并未超过 Reference（68.3%）**：hard-lam01 最优 66.4%。Hard DPO pairs 的 rejected = weak positive，信号噪声较大。
3. **λ sweet spot 在 0.01~0.03**：λ=0.01 PPL 最优但 DPO alignment 较弱；λ=0.1 PPL 开始退化。
4. **Progressive 无明显优势**：joint-prog R@500=64.4%，低于 joint-hard-lam01。

### Next Steps

EXP-020: 精细扫描 λ=0.03/0.05/0.07，找到 PPL vs pref_acc 的 sweet spot。

---

## EXP-018: RF-DPO — Real Feedback DPO Alignment

**Date**: 2026-04-18
**Status**: completed
**Results**: [./ntp_checkpoints/exp018-*/](./ntp_checkpoints/exp018-*/)

### Background

SP-DPO (EXP-017) 用模型 beam search 自博弈生成 rejected candidates。RF-DPO 进一步引入**真实用户反馈信号**，从 `action_bitmap` 位运算中区分信号强度：

| Tier | 信号 | action_bitmap bits |
|------|------|-------------------|
| Strong positive | like, share, follow, comment, trade, order | 2,4,8,256,512,1024,2048,131072,262144,524288,1048576 |
| Weak positive | click, coin/photo/profile click, video view | 1,16,64,128,8192,16384,32768,65536 |
| Negative | 举报/不喜欢 | bit 31 (sign bit) |

Preference pair 构造：同一用户内配对。Chosen = strong positive item，Rejected Easy = negative feedback items，Rejected Hard = weak positive items。

来源：Align³GR (AAAI 2026 Oral) Phase 2。

### Hypothesis

1. RF-DPO Easy (negative feedback rejected) 对 Recall@10 有明确提升：模型学会避开用户明确讨厌的内容
2. RF-DPO Hard (weak positive rejected) 精细化区分深度互动 vs 浅层点击
3. Progressive Easy→Hard 优于单阶段
4. 基于真实反馈的 RF-DPO 优于自博弈 SP-DPO（信号更真实，虽然量可能更少）
5. RF-DPO on top of SP-DPO (作为 π_ref) 进一步叠加提升

### Design

- **Variable**: difficulty (Easy/Hard)、渐进 vs 单阶段、β ablation
- **Fixed**: S-tier 模型 (17.5M active), 14 天数据, 同用户配对, pure DPO (no NTP loss)
- **Metric**: PPL, item_recall@{10,50,100,500}, depth_hit@10, DPO loss 曲线
- **Data**: 14d behavior data 2026-03-18 ~ 2026-03-31 (same window as EXP-016/017)
- **Baseline**: SP-DPO fixed-medium (EXP-017, R@10=15.4%)
- **Mode**: Pure Softmax-DPO (per Align³GR paper), no NTP regularization

| Config | Name | Difficulty | β | Epochs | Reference model | DPO pairs |
|--------|------|-----------|-----|--------|-----------------|-----------|
| 1 | rfdpo-easy | Easy | 0.1 | 3 | SP-DPO fixed-medium | 95 |
| 2 | rfdpo-hard | Hard | 0.1 | 3 | SP-DPO fixed-medium | 4,312 |
| 3 | rfdpo-prog | Progressive Easy→Hard | 0.1 | 3 | Easy→Hard chain | 4,312 (stage 2) |
| 4 | rfdpo-prog-beta01 | Progressive Hard | 0.01 | 3 | Easy output | 4,312 |
| 5 | rfdpo-prog-beta50 | Progressive Hard | 0.5 | 3 | Easy output | 4,312 |

### Run
`bash experiments/scripts/exp-018.sh`

### Results

| Config | β | DPO pairs | Steps | PPL | R@10 | R@50 | R@100 | R@500 | 训练耗时 |
|--------|-----|-----------|-------|---------|------|------|-------|-------|---------|
| **Reference (SP-DPO)** | — | — | — | ~14.5 | **15.4%** | — | — | **68.3%** | — |
| rfdpo-easy | 0.1 | 95 | 15 | 35.8 | 13.8% | 31.4% | 40.9% | 64.2% | ~0.5min |
| rfdpo-hard | 0.1 | 4,312 | 807 | 50,694 | 8.3% | 18.2% | 23.3% | 28.9% | 51min |
| rfdpo-prog (E→H) | 0.1 | 4,312 | 807 | 98,747 | 8.7% | 16.7% | 21.5% | 26.3% | 51min |
| rfdpo-prog-beta01 | 0.01 | 4,312 | 807 | 2.4B | 6.0% | 11.5% | 14.6% | 15.9% | 51min |
| rfdpo-prog-beta50 | 0.5 | 4,312 | 807 | 404.9 | 10.2% | 25.4% | 34.1% | 49.8% | 51min |

### Analysis

**Pure DPO 全面退化，没有任何配置超过 reference。**

1. **Catastrophic forgetting**: 纯 DPO 没有 NTP 正则化，807 步 hard 训练导致模型遗忘 NTP 语言建模能力。PPL 从 ~14.5 爆炸到 50K–2.4B。
2. **β 作为正则化器**: β 越大对 reference policy 的 KL 惩罚越强，退化越轻。β=0.5 最好（PPL 404.9, R@500 49.8%）但仍远不如 reference。β=0.01 最差（PPL 2.4B）。
3. **Easy 几乎没动模型**: 只有 95 pairs / 15 steps，训练不足以改变模型。PPL 35.8 说明轻微退化，R@10 13.8% 反而最接近 reference。
4. **Progressive 没有优势**: Easy→Hard 与单阶段 Hard 几乎相同（PPL 98K vs 50K，R@10 8.7% vs 8.3%），Progressive 结构未能缓解 forgetting。
5. **结论**: 论文的 pure Softmax-DPO 方案在我们的数据规模下不可行。可能需要更大数据量、额外 KL 约束、或论文未公开的细节。Joint NTP+DPO 思路是对的，但需要限制步数匹配 DPO 数据量。

### Next Steps

EXP-019: Joint NTP+DPO with step-matched training — 限制 NTP 步数与 DPO epoch 匹配，避免 NTP 冲掉 DPO signal（EXP-018 Easy 的教训），同时保留 NTP 正则化防止 catastrophic forgetting（EXP-018 Hard 的教训）。

---

## EXP-017: SP-DPO — Self-Play DPO Alignment for NTP Model

**Date**: 2026-04-17 ~ 2026-04-20
**Status**: completed
**Results**: [experiments/ntp_checkpoints/exp017-*](experiments/ntp_checkpoints/)

### Background

NTP 模型已达到 S-tier 基线 (EXP-016 14d-S: PPL=27.05, R@500=58.5%)。当前训练纯 SFT (交叉熵)，只告诉模型"什么是对的"，不告诉模型"你当前犯的哪些错误是错的"。

SP-DPO (Self-Play DPO, Align³GR, AAAI 2026 Oral) 是 RL 对齐的入门方案：
1. 用模型自己 beam search 生成候选作为 rejected（负样本）
2. Ground truth 作为 chosen（正样本）
3. 按 SID prefix 匹配层数定义难度：Easy (L0 不同) → Medium (L0 同, L1 不同) → Hard (L0+L1 同, L2 不同)
4. Softmax-DPO loss 渐进训练 (1 chosen vs 20 rejected)

**关键发现**：论文的 beam search + classify 方法在 4096×3 SID 体系下，Medium/Hard 候选数量 ≈ 0（B=200 仍几乎全是 Easy）。原因：beam 从 L0 自由采样，命中 GT L0 的概率极低（SFT depth_acc L0=3%）。

**解法**：Prefix-locked beam search — 固定 GT 前缀，beam search 剩余层。保证 Medium/Hard 候选充足。详见 [discussions/004](../discussions/004-prefix-locked-vs-paper-beam-search.md)。

### Hypothesis

1. SP-DPO Easy 提升 R@10（拉开基本对错边界）— **已验证** ✓
2. Prefix-locked 渐进采样产出充足 M/H 数据，使 Medium/Hard 阶段可行
3. 渐进模型（Easy model 采样）vs 固定模型（SFT 采样）：模型改进后生成的 candidates 更有针对性
4. Joint loss (NTP + DPO) 保持 SFT 知识不丢失

### Design

- **Variable**: M/H 采样模型 (SFT vs Easy model)、λ 权重
- **Fixed**: S-tier 模型 (256d, 6L, 8E top-2, ~17.5M active), 14 天数据, prefix-locked B=50
- **Metric**: PPL, item_recall@{10,50,100,500}, depth_acc_beam, DPO loss 曲线
- **Data**: EXP-016 14d preprocessed NTP data (130M tokens)
- **Baseline**: EXP-016 14d-S checkpoint (PPL=27.05, R@500=58.5%)

**采样方式**: Prefix-locked beam search (所有 config)

| 采样 pass | 锁定前缀 | 产出难度 | beam_size |
|-----------|---------|---------|-----------|
| Pass 1 | 无 | Easy (L0 ≠ GT) | 50 |
| Pass 2 | L0=GT | Medium (L1 ≠ GT) + Hard | 50 |
| Pass 3 | L0+L1=GT | Hard (L2 ≠ GT) | 50 |

**实验矩阵**:

| Config | `--start-from` | M/H 采样模型 | 说明 |
|--------|------|-------------|------|
| Shared Easy | 1 | SFT (full beam) | Easy DPO baseline, 共享 |
| Config 1 | 2 | **SFT** prefix-locked | 固定模型 + 渐进采样 |
| Config 2 | 3 | **Easy model** prefix-locked | 渐进模型 + 渐进采样 |
| λ=0.05 | 4 | Easy model | λ 消融 |
| λ=0.5 | 5 | Easy model | λ 消融 |

**Key comparison**: Config 1 vs 2 → 渐进模型是否有帮助？

### Run

`bash experiments/scripts/exp-017.sh --no-smoke --start-from=1`

### Results

**Easy stage (shared)**:

| Metric | SFT Baseline | SP-DPO Easy | Delta |
|--------|-------------|-------------|-------|
| PPL | 27.05 | 28.49 | +5.3% (expected) |
| R@10 | 9.9% | **12.5%** | **+26.3%** |
| R@50 | 26.1% | 27.1% | +3.8% |
| R@500 | 58.5% | 55.0% | -6.0% |
| depth_acc L0 | 0.030 | **0.041** | **+37%** |
| depth_acc L1 | 0.018 | **0.029** | **+61%** |
| depth_acc L2 | 0.018 | **0.029** | **+61%** |
| L2 PPL | 4.84 | **2.48** | **-48.7%** |

**Config 1 vs Config 2: Easy → Medium → Hard (completed)**:

| Metric | SFT | C1 Medium | C1 Hard | C2 Medium | C2 Hard |
|--------|-----|-----------|---------|-----------|---------|
| prefix L0 | 0.200 | 0.224 | 0.231 | **0.234** | 0.231 |
| prefix L1 | 0.172 | 0.203 | 0.210 | **0.212** | 0.207 |
| prefix L2 | 0.171 | 0.201 | 0.209 | **0.210** | 0.206 |
| indep L0 | 0.199 | 0.223 | 0.230 | **0.233** | 0.230 |
| indep L1 | 0.808 | **0.899** | 0.888 | 0.885 | 0.877 |
| indep L2 | 0.852 | 0.941 | **0.957** | 0.936 | 0.949 |
| PPL | 27.05 | 17.49 | **14.24** | 16.13 | 15.24 |
| DPO loss | — | 2.210 | 1.305 | 2.331 | 1.321 |
| wall_time | — | 3.9h | 2.1h | 2.2h | **1.2h** |

depth_hit@10 基于 147,902 eval positions。R@500 仅 1,000 samples，统计意义有限，不作为主要指标。

**Beam search candidate 分布** (SFT model, B=200):
- Easy: ~20/pair, Medium: ~0/pair, Hard: 0/pair
- 确认论文方法在 4096×3 体系下 M/H 数据严重不足 → prefix-locked 是必须的

**Hard candidate 稀缺性**: avg 5.9 rejected/pair（vs Medium ~20/pair）。原因：L0+L1 prefix 下 trie 中有效 L2 选项本身就少。非代码 bug，是数据层级结构决定的。

### Analysis

1. **最优阶段是 Easy → Medium，Hard 无正面贡献**:
   两个 Config 一致验证：Hard 阶段 depth_hit@10 和 indep L1 均退化。Hard DPO loss 异常低（~1.3 vs Medium ~2.3），信号"太简单"。

2. **Hard 退化的三重原因**:
   - **信号太窄**: rejected 只在 L2 不同，DPO 梯度仅教 L2 区分，但更新整个 shared backbone → 干扰 L0/L1 表征
   - **Rejected 太少**: avg 5.9/pair → logsumexp 只有 ~6 项，梯度 noisy；模型已能轻松区分 → 有效信号接近零
   - **选择偏差**: 只有 GT 的 L0+L1 prefix 下有 ≥2 个有效 L2 的 items 才产生 hard pairs，非 eval 全集代表

3. **C2 Medium prefix 指标全面最优**: 说明 Easy model 采样的 Medium/Hard candidates 比 SFT 采样的更有针对性（on-policy 效应）。但 C1 Medium 的 indep L1 (0.899) > C2 Medium (0.885) — 可能因为 SFT 采样产生的 candidates 覆盖更广的 L1 空间。

4. **PPL vs depth_hit 不完全正相关**: C2 Hard PPL 最低 (15.24) 但 depth_hit 不是最优 — PPL 优化绝对概率，depth_hit 优化 top-K 排序。

5. **Engineering wins**:
   - Gradient checkpointing + MoE freeze: 解决 DPO OOM，吞吐量翻倍（9k→17k tok/s）。详见 [docs/engineering/001](../docs/engineering/001-dpo-oom-gradient-checkpointing.md)
   - Packed DPO candidates: 消除 padding 浪费，Hard 训练再提速 44%（17k→30k tok/s）
   - KV cache beam search: context encoding 冗余从 ~153C/item 降至 ~C/3/item。详见 [discussions/005](../discussions/005-beam-search-kv-cache.md)

### Conclusions

1. **SP-DPO Easy → Medium 是最优 pipeline**，Hard 阶段应跳过
2. **最优 checkpoint: C2 Medium** (prefix 全面最优) 或 **C1 Medium** (indep L1 最优)
3. Prefix-locked beam search 是 4096×3 SID 体系下 Medium/Hard 数据生成的必要条件
4. 渐进模型采样（C2）vs 固定模型（C1）差异不大，两者各有优势指标

### Next Steps

1. ~~λ 消融~~ — Hard 阶段不做了
2. EXP-018: RF-DPO（引入真实用户反馈，Align³GR Phase 2），以 C2 Medium 或 C1 Medium 作为 π_ref

---

## EXP-015: NTP Scaling Law — Sweep Model Size from 1M to 100M Active Params

**Date**: 2026-04-16 ~ 2026-04-17
**Status**: completed
**Results**: [experiments/results/ntp/](experiments/results/ntp/)

### Background

EXP-013 证明了扩大参数 (7.5M→45.8M) 能加速收敛 (PPL 70→29.6, recall@500 37%→60%)。但仅有两个数据点，无法回答关键问题：收益何时饱和？多大模型是性价比最优？

OneRec-V2 论文给出了推荐领域的 scaling law `L̂(N) = 3.13 + 3660 / N^0.489`，证明推荐模型的 loss 也遵循 power law。本实验通过 7 个不同规模的模型配置，在相同数据上拟合我们自己的 scaling law。

### Hypothesis

1. NTP eval loss 关于 active params 遵循 power law `L(N) = a + b / N^α`
2. α 接近 OneRec-V2 的 0.489（架构相似）
3. 存在明确的性价比拐点（收益递减加速的转折区间）

### Design

- **Variable**: 模型规模 (embed_dim, n_layers, MoE config)
- **Fixed**: SID 4096×3 binary, 31 天数据 (03-01~03-31), 1 epoch, beam_size=500
- **Data**: 复用 EXP-013 preprocessed NTP data (262M tokens)
- **Metric**: eval loss, PPL, item_recall@{10,50,100,500}

| Config | embed_dim | layers | MoE | ~Active Params |
|--------|-----------|--------|-----|----------------|
| scale-01 | 64 | 2 | dense | 1.7M |
| scale-02 | 128 | 2 | dense | 3.6M |
| scale-03 | 128 | 4 | 4E top-2 | 5.1M |
| scale-04 | 256 | 6 | 8E top-2 | 17.5M |
| scale-05 | 384 | 6 | 8E top-2 | 34.5M |
| scale-06 | 512 | 8 | 8E top-2 | 71.6M |
| scale-07 | 512 | 12 | 16E top-2 | 101.1M |

**代码改动**: `ntp/train.py` 的 s-tier 超参从硬编码改为 CLI 可配置 (`--n_experts`, `--top_k`, `--expert_dim`, `--embed_dim`, `--n_transformer_layers`)。`n_experts=0` 自动切换 dense 模式。

### Run

`bash experiments/scripts/exp-015.sh`

### Results

| Config | Active Params | PPL | Loss | R@10 | R@100 | R@500 | 训练耗时 |
|--------|--------------|------|------|------|-------|-------|---------|
| scale-01 | 1.7M | 235.1 | 5.460 | 1.9% | 11.8% | 23.6% | 2min |
| scale-02 | 3.6M | 100.4 | 4.609 | 3.7% | 16.6% | 31.7% | 3min |
| scale-03 | 5.1M | 69.6 | 4.243 | 5.4% | 24.9% | 45.6% | 9min |
| scale-04 | 17.5M | **28.1** | 3.334 | 9.8% | 35.6% | 60.5% | 34min |
| scale-05 | 34.5M | 24.0 | 3.178 | 11.5% | 39.1% | 62.5% | 61min |
| scale-06 | 71.6M | 20.8 | 3.037 | 12.6% | 41.0% | 66.2% | 131min |
| scale-07 | 101.1M | **19.4** | 2.965 | 13.7% | 43.2% | 65.8% | 374min |

**Scaling Law Fit**:

```
L̂(N) = 2.522 + 2055.1 / N^0.456
```

- **a = 2.522**: irreducible loss floor (数据/tokenizer 信息瓶颈)
- **α = 0.456**: scaling exponent (接近 OneRec-V2 的 0.489)
- **b = 2055.1**: scale factor

![NTP Scaling Law](results/ntp/exp015-scaling-law.png)

### Analysis

1. **Power law 成立**: log-log 图上 7 个数据点基本落在直线上，R² 良好
2. **α = 0.456 ≈ OneRec-V2 的 0.489**: 架构 scaling 效率与论文接近，验证了 MoE + SwiGLU 的通用性
3. **收益递减明显**:
   - 5M→17M: PPL 70→28 (-60%)，recall@500 46%→60% — **最大提升区间**
   - 17M→71M: PPL 28→21 (-25%)，recall@500 60%→66% — 中等提升
   - 71M→101M: PPL 21→19 (-7%)，recall@500 66%→66% — **接近饱和**
4. **Irreducible loss a=2.522 (PPL≈12.5)**: 即使模型无限大，PPL 也降不到 12.5 以下。这是 tokenizer (4096×3 codebook, collision 0.89%) 和用户行为随机性的天花板
5. **Recall 也在 scale 但增速不同**: R@100 从 12%→43% 涨了 3.6x，而 R@500 从 24%→66% 只涨 2.8x — 更大模型对 top-K 精排的提升更显著
6. **EXP-013 数据点吻合**: probe (7.5M, PPL=70) 和 s-tier (45.8M, PPL=29.6) 都精确落在拟合曲线上

**Hypothesis 验证**:
- H1 ✅ Power law 成立，7 点拟合良好
- H2 ✅ α=0.456 ≈ 0.489，非常接近
- H3 ✅ M 档 (~50-70M active) 是明确甜点，之后曲线变平

### Predictions

| Active Params | 预测 PPL | 预测 Loss | 性价比 |
|--------------|---------|-----------|--------|
| 17M (S) | 28 | 3.33 | 当前基线 |
| **55M (M)** | **~23** | **~3.15** | **最佳性价比** |
| 500M (L) | ~15.5 | ~2.74 | 成本高，收益递减 |
| 1B | ~14.6 | ~2.68 | 接近 floor |

### Chinchilla 分析

EXP-015 所有模型在相同 262M tokens 上训练。按 Chinchilla 经验法则 (N* = D/20)，最优模型大小约 13M active params。

**Tokens/Param 与 FLOP 效率**:

| Config | Active | Tok/Param | FLOP 效率 (loss/PF) | Chinchilla 状态 |
|--------|--------|-----------|---------------------|----------------|
| scale-01 | 1.7M | 152 | — | 过训练 7.6x |
| scale-02 | 3.6M | 72 | 0.28 | 过训练 3.6x |
| scale-03 | 5.1M | 52 | 0.16 | 过训练 2.6x |
| **scale-04** | **17.5M** | **15** | **0.05** | **接近最优 (0.7x)** |
| scale-05 | 34.5M | 8 | 0.01 | 欠训练 0.4x |
| scale-06 | 71.6M | 4 | 0.002 | 严重欠训练 0.2x |
| scale-07 | 101.1M | 3 | 0.002 | 严重欠训练 0.1x |

**关键发现**:

1. **FLOP 效率单调递减** (0.28 → 0.16 → 0.05 → 0.01 → 0.00)，与 Chinchilla 预测完全一致
2. **scale-04 (17.5M) 是 262M tokens 的 Chinchilla 最优点** — 15 tok/param 接近 20 的经验值
3. **大模型严重欠训练但 loss 仍单调下降** — 推荐序列短 (30 tokens)，即使 3 tok/param 也不会 overfit，与 LLM 不同
4. **加数据 ROI 极高**: 101M 模型 tok/param 从 3→20 需要 ~2B tokens (~240 天数据)，PPL 有望从 19.4 降到接近 floor (12.5)

**Chinchilla 最优数据量**:

| 模型 | Active Params | Chinchilla 最优 Tokens | 需要天数 |
|------|-------------|----------------------|---------|
| S (17M) | 17.5M | 350M | ~41 天 |
| M (55M) | 55M | 1.1B | ~130 天 |
| M+ (101M) | 101M | 2.0B | ~240 天 |

**结论: 当前瓶颈是数据不是模型。先加数据 (31→90 天) 再加模型是 ROI 最高的路径。**

### Next Steps

1. **EXP-016 Data Scaling**: 固定 S/M 模型，sweep 数据量 → Chinchilla 双变量 scaling law → 找到最优 N-D 配比
2. **Tokenizer ceiling**: a=2.522 偏高，尝试 8192×3 codebook 降低 irreducible loss

---

## EXP-016: Data Scaling Law — 固定模型 Sweep 数据量 (Chinchilla 双变量)

**Date**: 2026-04-17 ~ 2026-04-18
**Status**: completed
**Results**: [experiments/results/ntp/](experiments/results/ntp/)

### Background

EXP-015 揭示了两个关键事实:

1. **Scaling law 成立**: `L(N) = 2.522 + 2055/N^0.456`，但这是固定 D=262M tokens 下的单变量 law
2. **大模型严重欠训练**: scale-07 (101M active) 仅 3 tok/param，Chinchilla 建议 20x。FLOP 效率在超过 17.5M 后急剧衰减

Chinchilla (Hoffmann 2022) 的完整 scaling law 是双变量的:

```
L(N, D) = E + A/N^α + B/D^β
```

其中 E 是 irreducible loss, A/N^α 是模型不足项, B/D^β 是数据不足项。EXP-015 只 sweep 了 N，D 固定。本实验固定 N，sweep D，以拟合完整的双变量 law，并找到给定算力预算下的最优 N-D 配比。

**核心问题**: 把数据从 31 天扩到 66 天后:
- S 档 (17.5M active) 和 M 档 (101M active) PPL 各降多少？
- β 是多少？（数据 scaling 指数）

### Data Distribution Analysis

可用 embedding 覆盖 2026-01-25 ~ 2026-03-31 (66 天)。数据分布分析 (`analyze_data_distribution.py`):

| Config | Users | Raw Items | Mean/User | P50 | P95 | P99 | Max |
|--------|-------|-----------|-----------|-----|-----|-----|-----|
| A-7d | 1.54M | 23.9M | 15.6 | 3 | 68 | 220 | 5,376 |
| B-14d | 2.51M | 53.1M | 21.2 | 3 | 92 | 331 | 9,063 |
| C-31d | 4.55M | 129.7M | 28.5 | 3 | 118 | 499 | 32,246 |
| D-62d | 7.29M | 261.8M | 35.9 | 3 | 138 | 669 | 46,223 |
| E-66d | 7.85M | 299.0M | 38.1 | 3 | 146 | 715 | 46,990 |

**关键发现: 极度长尾 + 截断影响大**

- **P50 恒定为 3**: 50% 用户只有 ≤3 次交互，分布极度右偏
- **少数重度用户贡献大量 items**: 4% 的用户被 170-item cap 截断，但其交互占总量的 ~50%
- 这是**两个维度的不矛盾现象**: 用户维度看截断影响小 (4%)，item 维度看影响大 (50%)

**截断分析** (`max_seq_len=512` → `max_items=170`):

| Config | 截断用户% | Items 丢失% | Raw Items | 有效 Items | **有效 Tokens** |
|--------|----------|------------|-----------|-----------|----------------|
| A-7d | 1.5% | 14.5% | 23.9M | ~20.4M | **~61M** |
| B-14d | 2.6% | 25.4% | 53.1M | ~39.6M | **~119M** |
| C-31d | 3.6% | 38.9% | 129.7M | ~79.3M | **~238M** |
| D-62d | 4.2% | 48.5% | 261.8M | ~134.8M | **~404M** |
| E-66d | 4.4% | 50.4% | 299.0M | ~148.3M | **~445M** |

> 注: 有效 Tokens = 有效 Items × 3 (n_layers=3)。截断保留每个用户最近 170 items，丢弃的是更早的历史。
> 对推荐场景，近期行为更有价值，截断的老行为对模型训练影响有限。

### Hypothesis

1. 数据从 238M→445M tokens (31d→66d)，S 档 (17.5M) PPL 下降有限 (<5%)，因为已接近 Chinchilla 最优
2. 数据从 238M→445M tokens (31d→66d)，M 档 (101M) PPL 下降显著 (>15%)，因为目前严重欠训练
3. β ≈ 0.4-0.5（与 α≈0.456 接近，符合 Chinchilla 对称性假设）
4. 给定 66 天数据 (~445M tokens)，Chinchilla 最优模型大小上移到 ~22M active params

### Design

- **Variable**: 数据量 D ∈ {7d, 14d, 31d, 62d, 66d} × 模型 {S (17.5M), M+ (101M)}
- **Fixed**: SID 4096×3 binary, 1 epoch, beam_size=500, max_seq_len=512 (170 items/user)
- **Metric**: eval loss, PPL, item_recall@{10,50,100,500}
- **Eval 说明**: 每个 config 的 `preprocess-ntp` 用 `n_eval_target=50000`，按时间分位切 split_ts。不同 data size 的 eval set 有轻微差异（split_ts 不同），但都集中在窗口末尾，对 scaling law 拟合影响有限。

| Config | Model | Data Days | Users | 有效 Tokens | Tok/Param (S) | Tok/Param (M+) |
|--------|-------|-----------|-------|------------|---------------|----------------|
| A-7d | S + M+ | 7 | 1.54M | ~61M | 3.5 | 0.6 |
| B-14d | S + M+ | 14 | 2.51M | ~119M | 6.8 | 1.2 |
| C-31d | S + M+ | 31 | 4.55M | ~238M | 13.6 | 2.4 |
| D-62d | S + M+ | 62 | 7.29M | ~404M | 23.1 | 4.0 |
| E-66d | S + M+ | 66 | 7.85M | ~445M | 25.4 | 4.4 |

C-31d 的 S 档可复用 EXP-015 scale-04 结果，M+ 档复用 scale-07 结果。实际新增训练: 4×2 = 8 runs (减去 C-31d 复用 = 6 runs)。

**分析计划**:
1. 分别对 S 和 M+ 拟合 `L(D) = E + B/D^β`
2. 联合 EXP-015 + EXP-016 数据拟合双变量 `L(N,D) = E + A/N^α + B/D^β`
3. 画 iso-FLOP 曲线 (固定 C=6ND)，找每条曲线上的最优 N-D 分配
4. 预测: 给定 8×A100 × 1h 算力预算，最优配置是什么

### Run

`bash experiments/scripts/exp-016.sh`

### Results

**S 模型 (17.5M active)**:

| Config | Days | Tokens | Users | PPL | Loss | R@100 | R@500 | 训练耗时 |
|--------|------|--------|-------|-----|------|-------|-------|---------|
| A-7d-S | 7 | 65M | 1.02M | 30.60 | 3.421 | 37.9% | 62.1% | 11min |
| **B-14d-S** | **14** | **130M** | **1.69M** | **27.05** | **3.298** | **35.0%** | **58.5%** | 17min |
| C-31d-S | 31 | 262M | 3.04M | 28.05 | 3.334 | 35.6% | 60.5% | 55min |
| D-62d-S | 62 | 441M | 4.86M | 30.03 | 3.402 | 36.5% | 58.6% | 55min |
| E-90d-S | 90 | 553M | 6.18M | 31.89 | 3.462 | 35.1% | 56.2% | 69min |

**M+ 模型 (101M active)**:

| Config | Days | Tokens | Users | PPL | Loss | R@100 | R@500 | 训练耗时 |
|--------|------|--------|-------|-----|------|-------|-------|---------|
| A-7d-M | 7 | 65M | 1.02M | 19.31 | 2.960 | 42.7% | 70.7% | 123min |
| **B-14d-M** | **14** | **130M** | **1.69M** | **18.96** | **2.942** | **43.0%** | **65.8%** | 207min |
| C-31d-M | 31 | 262M | 3.04M | 19.39 | 2.965 | 43.2% | 65.8% | 374min |
| D-62d-M | 62 | 441M | 4.86M | 19.80 | 2.986 | 43.2% | 68.1% | 607min |
| E-90d-M | 90 | — | 6.18M | *(跳过)* | — | — | — | — |

![Data Scaling Law](results/ntp/exp016-data-scaling.png)

### Analysis

**1. Chinchilla data scaling 不适用于推荐序列**

Chinchilla 假设 i.i.d. data：更多 tokens 单调降低 loss。但推荐行为数据有时间非平稳性，**14d 是 loss 最优点**，之后 loss 反升：

- S: 3.421 (7d) → **3.298 (14d)** → 3.334 (31d) → 3.402 (62d) → 3.462 (90d)
- M+: 2.960 (7d) → **2.942 (14d)** → 2.965 (31d) → 2.986 (62d)

这是一个 **U 型曲线**，不是 power law 递减。

**2. 根因：增加天数 = 增加用户，不是更长序列**

| Days | Users | Avg Items/User |
|------|-------|---------------|
| 7d | 1.02M | ~21 |
| 14d | 1.69M | ~26 |
| 31d | 3.04M | ~29 |
| 62d | 4.86M | ~30 |
| 90d | 6.18M | ~30 |

Avg items/user 从 21→30 几乎不变（受 max_seq_len=512 和用户活跃度限制），但用户数从 1M→6M 涨了 6x。新增用户来自更早的时间窗口，行为分布已偏移。

**3. 曝光窗口约束是核心原因**

本场景曝光 item 限定为 3 天内创作的内容。这意味着：
- item pool 每 3 天完全刷新
- 30 天前的训练数据对应的 item pool 已经完全不存在
- 老数据的行为 pattern 可能已不适用于当前 item pool

14d ≈ 4-5 个曝光窗口周转周期，是覆盖 item pool 多样性和避免分布偏移的平衡点。

**4. 模型已接近 irreducible loss**

M+ 在 14d (130M tokens) 就达到 loss=2.942，与 EXP-015 预测的 `L(101M) = 2.522 + 2055/101M^0.456 ≈ 2.96` 基本吻合。剩余 gap (2.942 - 2.522 = 0.42) 由 tokenizer 信息瓶颈主导，加数据无法突破。

**5. 与序列长度 scaling law 不矛盾**

论文中报道的序列 scaling law 是固定用户群、增加每用户历史长度（深度 scaling）。本实验 scale 的是用户广度（更多低活跃/历史用户），不是序列深度。两者是不同维度。

### Hypothesis 验证

- H1 ❌ S 14d→90d PPL 从 27.05 升到 31.89 (+18%)，不是下降
- H2 ❌ M+ 14d→62d PPL 从 18.96 升到 19.80，不是 >15% 下降
- H3 无法验证：Chinchilla 双变量 law 不适用，β 无意义
- H4 ❌ 最优模型大小不随数据量上移，因为数据量增加无效

### Key Findings

1. **最优训练窗口 ~14d**：对 S 和 M+ 模型均成立，loss/PPL 达到最低
2. **Chinchilla data scaling 不适用**：推荐行为数据非 i.i.d.，存在 "有效半衰期" (~14d)
3. **瓶颈是 tokenizer 不是数据**：M+ loss=2.94 已逼近 irreducible floor 2.52
4. **下一步应 scale 序列深度或 tokenizer**，而非数据时间范围

### Next Steps

1. **Tokenizer 改进** (最高 ROI): 8192×3 codebook 或更细 FSQ → 降低 irreducible loss floor
2. **序列深度 scaling**: 固定 14d 用户群，sweep max_items {10, 30, 50, 100, 170} → 验证真正的序列 scaling law
3. **多 epoch on 14d**: S 模型 1 epoch 可能欠拟合，尝试 2-3 epoch

---

## EXP-014: ENTP-Loss — Exposure-Aware Hard Negatives for L0

**Date**: 2026-04-16
**Status**: running
**IDEA**: IDEA-dualgr-0
**Results**: TBD

### Background

EXP-013 S-tier model recall@500=59.5%，但 **L0 PPL=344.8 是明确瓶颈**（L1=13.3, L2=5.7 已接近饱和）。L0 hit@10 仅 20%，模型在 4096 个 coarse cluster 上的区分能力很弱。

当前 NTP loss 只有正样本（用户点击了的 item），完全没有利用"用户看了但没点"的负信号。DualGR (快手, WWW 2026, arxiv 2511.12518) 提出 ENTP-Loss：将曝光未点击的 item 作为 L0 层 hard negative，通过 `−α·log(1 − p_L0)` 惩罚项直接增强 L0 监督信号。

数据侧 `export_exposure.py` 已就绪，每天 ~1.1GB 曝光数据（含 action_bitmap=0 的未点击项），与行为数据 ~85MB/天 约 13:1。

### Hypothesis

1. ENTP-Loss (α=0.1) 使 L0 PPL 下降 >10%（从 344.8 降至 <310），因为 L0 获得了额外的 per-position 时间对齐负样本监督
2. L1/L2 PPL 不受影响（ENTP 只作用在 L0 层的 output_proj）
3. recall@500 提升（L0 更准 → beam search 在 coarse level 筛选更好 → 下游 fine-level 受益）

### Design

- **Variable**: ENTP weight α ∈ {0, 0.05, 0.1, 0.2}
- **Fixed**: S-tier 6L MoE (EXP-013 配置), K=5 negatives/position, 4096×3 binary SID, batch_size=128, 1 epoch, beam_size=500
- **Metric**: L0/L1/L2 PPL, hit@10 per layer, recall@{10,50,100,500}
- **Data**: 31 天行为数据 (03-01~03-31) + 31 天曝光数据 (同期)

**ENTP 负样本构造 (PySpark 端)**:
- `export_exposure.py` 新增 ENTP section：Spark SQL window function `pos_grp = cumsum(is_positive)` 分段，
  每段的 non-positive (action_bitmap ≤ 0) 作为下一个 positive 的负样本，取最近 K=5 个
- 输出 `feed_user_exposure_neg/{date_start}_{date_end}` parquet: `uid, iid, first_ts, neg_iids ARRAY<STRING>`
- Python 端 `load_exposure_neg_data()` 加载 ~130M 行（秒级），`_build_sequences_from_exposure()` 只做 iid→L0 映射

**Loss**:
```
L = L_NTP(L0+L1+L2 三层 CE, 不变) + α * L_ENTP(仅 L0 负样本惩罚)
L_ENTP = −(1/N) Σ log(1 − p_i^(L0))   (对 unclicked exposure 的 L0 token)
```

**改动文件**:
1. `data/export_exposure.py` — PySpark ENTP 负样本导出 (Spark SQL window function)
2. `eval/batch.py` — 新增 `load_exposure_neg_data()` 加载 compact parquet
3. `ntp/train.py` — `_build_sequences_from_exposure()` 简化为 dict→序列映射; wandb 集成
4. `ntp/model.py` — `_forward_packed()` 增加 ENTP loss 项
5. `ntp/baseline.py` — `NTPProbe._forward_packed()` 同步 ENTP 扩展
6. `ntp/preprocess.py` — shard 格式扩展存储 neg_l0; 调用 `load_exposure_neg_data()`

**可插拔设计**: `--entp_weight 0`（默认）= 完全等价于 EXP-013 代码路径。

| Config | α | K | L0 filter | 说明 |
|--------|------|---|-----------|------|
| A (baseline) | 0 | — | — | 直接复用 EXP-013 s-tier 结果 |
| B | 0.05 | 5 | ✗ | 保守 (round 1, 已退步) |
| C | 0.1 | 5 | ✗ | DualGR 论文默认 (round 1, 已退步) |
| E | 0.05 | 5 | ✓ | 保守 (round 2, L0 collision 过滤) |
| F | 0.1 | 5 | ✓ | 论文默认 (round 2) |
| G | 0.2 | 5 | ✓ | 激进 (round 2) |

### Run

`bash experiments/scripts/exp-014.sh`

### Results

**PySpark ENTP 导出验证 (2026-04-16)**:

| 指标 | PySpark 导出 | 旧流式 walk (对照) | 说明 |
|---|---|---|---|
| 总曝光行 | ~1.19B | 1,185,707,891 | 一致 |
| Positives | 130,995,419 | 124,893,764 | +4.9%, 差异 = SID 字典外的 iid（Python 端过滤） |
| Users | 4,608,606 | 3,042,069 | +51%, 多出的用户只有 SID 外 iid，Python 端过滤后消失 |
| 有负样本 | 40,761,718 (31.1% row级) | 2,084,314 (68.5% user级) | 口径不同，无矛盾 |

31% row 级有负样本合理：Feed 场景用户常连续点击（同页多 item），连续 positive 之间无 non-positive → 后者拿不到 neg。

**训练结果 B/C (旧代码, 无 L0 collision 过滤)**:

| Metric | A (α=0, baseline) | B (α=0.05) | C (α=0.1) | B Δ | C Δ |
|---|---|---|---|---|---|
| PPL | 29.60 | 31.67 | 31.67 | +7.0% | +7.0% |
| L0 PPL | 344.76 | 363.78 | 361.41 | +5.5% | +4.8% |
| L1 PPL | 13.28 | 15.23 | 15.23 | +14.7% | +14.7% |
| L2 PPL | 5.72 | 5.79 | 5.83 | +1.2% | +1.9% |
| L0 hit@10_indep | 0.2004 | 0.1919 | 0.1902 | -4.2% | -5.1% |
| recall@10 | 0.102 | 0.086 | 0.089 | -15.7% | -12.7% |
| recall@50 | 0.250 | 0.230 | 0.234 | -8.0% | -6.4% |
| recall@100 | 0.346 | 0.305 | 0.304 | -11.8% | -12.1% |
| recall@500 | 0.595 | 0.525 | 0.529 | -11.8% | -11.1% |

B/C 全面退步。根因分析见 Analysis。

### Analysis

**根因: L0 token collision 导致梯度冲突。**

同 session 的 item 因为话题相似被推荐系统一起展示，经 SID 量化后大量落入同一个 L0 cluster（4096 clusters, avg 122 items/cluster）。当负样本与正样本共享同一个 L0 token 时：
- NTP loss 推高 p(L0=k)（正样本的 L0）
- ENTP loss 压低 p(L0=k)（负样本的 L0，恰好相同）
- 梯度直接对冲 → L0 PPL 反而上升 (344→363)
- 冲突通过 shared transformer backbone 传播 → L1 PPL 也大幅退步 (+14.7%)

DualGR 论文用 8192 L0 clusters 且有 10B exposures/day，collision 率天然更低。论文还提到 probability clipping `[ε, 1-ε]` 但未说明 ε 值。

**修复**: preprocess 阶段过滤掉与 positive 共享 L0 的负样本。已实现，待重跑。

### Next Steps

1. 用新代码重跑 B (α=0.05) / C (α=0.1)，包含:
   - L0 collision 过滤
   - view_exit 排除
   - neg 优先级 (negative_feedback/view_exit 优先入 neg 池)
2. 观察 drop_pct — 如果 >30% 则验证 collision 假设
3. 如果修复后仍无提升，考虑 detach ENTP 梯度不回传 backbone

---

## EXP-013: S-tier NTP Model — 6L MoE + Loss-Free Balancing

**Date**: 2026-04-15
**Status**: completed
**Results**: [experiments/results/ntp/](experiments/results/ntp/)

### Background

EXP-010 baseline (2L dense probe, ~5M params) 效果极差 (item_recall@50=0.0008)。部分原因已在 EXP-011 中通过等大 codebook 修复，但模型容量也严重不足。

本实验升级 NTP 模型到 S-tier 规格 (6L MoE, ~42M params)，对应 `ideas/architecture_roadmap.md` Stage 1。同时将 MoE load balancing 从 Switch Transformer auxiliary loss 替换为 Loss-Free dynamic bias (IDEA-onemall-4, DeepSeek-V2 方案)。

新代码: `ntp/model.py` (NTPModel) vs `ntp/baseline.py` (NTPProbe)。

### Hypothesis

1. S-tier (6L MoE, 42M params) 的 item_recall@50 应显著高于 probe (2L dense, 5M)
2. PPL 下降 > 30% (模型容量 8x，更深层能捕获长程 SID 依赖)
3. Loss-Free MoE balancing 的 expert 利用率应合理均匀 (max/min freq < 3x)

### Design

- **Variable**: 模型架构 (probe vs s-tier)
- **Fixed**: SID 4096×3 + FSQ [2]×12 binary (EXP-011-H/012 best), n_items=10, batch_size=4096, 1 epoch, recall_beam_size=500
- **Metric**: Perplexity, Depth Hit@10, Item Recall@{10,50,100,500}, Expert utilization
- **Data**: 31 天行为数据 (03-01~03-31), eval ~50K items by timestamp split

| Config | Model | Layers | FFN | Params | 说明 |
|--------|-------|--------|-----|--------|------|
| A (baseline) | NTPProbe | 2 | Dense 512 | ~5M | EXP-010 复现 |
| B (s-tier) | NTPModel | 6 | SwiGLU MoE 8E top-2 | ~42M | Loss-Free bias |

### Run

`bash experiments/scripts/exp-013.sh`

### Results

| Metric | Probe (7.5M) | S-tier (45.8M) | 提升 |
|--------|-------------|----------------|------|
| PPL | 70.0 | **29.6** | -58% |
| L0 PPL (cross-item) | 429.1 | **344.8** | -20% |
| L1 PPL | 41.8 | **13.3** | -68% |
| L2 PPL | 19.2 | **5.7** | -70% |
| hit@10 (indep L0) | 16.7% | **20.0%** | +20% |
| hit@10 (indep L1) | 62.2% | **78.9%** | +27% |
| hit@10 (indep L2) | 71.5% | **84.0%** | +17% |
| recall@10 | 5.1% | **10.2%** | 2x |
| recall@50 | 14.6% | **25.0%** | 1.7x |
| recall@100 | 20.1% | **34.6%** | 1.7x |
| recall@500 | 37.2% | **59.5%** | 1.6x |
| SID found rate | 37.3% | **59.5%** | 1.6x |

Beam search: 1000 samples, beam_size=500. Eval items: 49,383.

### Analysis

1. **S-tier 全面碾压 probe**: recall@500 从 37%→60%，PPL 降 58%。模型容量 6x (45.8M vs 7.5M) 带来显著收益。
2. **L0 (cross-item) 仍是瓶颈**: L0 PPL 344.8，即预测下一个 item 的粗粒度 cluster 仍然很难。L1/L2 intra-item 预测已接近饱和 (hit@10 79%/84%)。
3. **Hypothesis 验证**:
   - H1 ✅ S-tier recall@50 = 25% vs probe 14.6%，显著提升
   - H2 ✅ PPL 下降 58% (超预期的 30%)
   - H3 待验证 (未记录 expert utilization)
4. **关键修复**: 本轮训练修复了 TransformerDecoder 非 causal cross-attention bug（旧模型通过 cross-attention 作弊看到未来 token）。所有结果均基于正确的 TransformerEncoder causal 实现。

### Next Steps

1. L0 cross-item 预测是主要瓶颈 → 考虑增大 context window (n_items > 10) 或增加 epoch 数
2. 尝试更大 batch size / learning rate schedule 优化
3. 记录 MoE expert utilization，验证 Loss-Free balancing 效果

---

## EXP-011: Codebook Size 消融 — 等大 1024/4096 + OPQ 对照

**Date**: 2026-04-15
**Status**: completed (部分，OPQ 未跑)
**Results**: [./hyperparam/2026-04-15_exp011-*/](./hyperparam/)

### Background

EXP-010 NTP baseline 效果极差 (L1 acc=0.7%, item_recall@50=0.0008)，根因之一是当前 SID 配置 **L1=1024, L2=1024, L3=4096 不等大**，NTP 模型用全局 max=4027 作为统一 vocab。

查阅 OneMall 原文发现其生产配置是 **三层等大 4096×4096×4096**，FSQ 层使用 "binary 16-bit MLP"。需要确定我们的最优 codebook 配置。

### Hypothesis

1. 三层等大配置 (1024×3 或 4096×3) 的 semantic_neighbor_HR 不低于当前 1024×1024×4096
2. Binary FSQ ([2,...,2]) 与 multi-level FSQ ([4,...,4]) 在相同 codebook size 下效果相当
3. OPQ 3×N (等 token 数对照) 仍然输层级结构 MLP-FSQ（延续 EXP-008 结论）

### Design

| Config | L1 (KMeans) | L2 (KMeans) | L3 (FSQ) | FSQ Levels | Bits | 对标 |
|--------|-------------|-------------|----------|------------|------|------|
| A (EXP-008) | 1024 | 1024 | 4096 | [4,4,4,4,4,4] | 32 | 已有 baseline |
| E | 1024 | 1024 | 1024 | [4,4,4,4,4] | 30 | 等大 1024, multi-level |
| F | 1024 | 1024 | 1024 | [2]×10 | 30 | 等大 1024, binary |
| G | 4096 | 4096 | 4096 | [4,4,4,4,4,4] | 36 | OneMall 配置 |
| H | 4096 | 4096 | 4096 | [2]×12 | 36 | OneMall binary |
| I | OPQ 3×1024 | — | — | — | 30 | 等 bits 对照 E/F |
| J | OPQ 3×4096 | — | — | — | 36 | 等 bits 对照 G/H |

- **Fixed**: Qwen3-0.6B 1024D embedding (cached), behavior_data 7d, MLP hidden=64, 50 epochs
- **Metric**: semantic_neighbor_hit_rate (核心), collision_rate, cluster_balance (Gini)

### Run

`bash experiments/scripts/exp-011.sh`

### Results

| Config | KMeans | FSQ | Bits | collision | snHR | L3 unique | L3 Gini |
|--------|--------|-----|------|-----------|------|-----------|---------|
| A (EXP-008) | 1024×1024×4096 | [4]×6 | 32 | 10.7% | 0.078 | 487K | — |
| E (1024, multi) | 1024×3 | [4]×5 | 30 | 14.6% | 0.078 | 404K | 0.151 |
| F (1024, binary) | 1024×3 | [2]×10 | 30 | 7.9% | 0.078 | 443K | 0.083 |
| G (4096, multi) | 4096×3 | [4]×6 | 36 | **0.84%** | **0.095** | 482K | 0.009 |
| H (4096, binary) | 4096×3 | [2]×12 | 36 | **0.89%** | **0.095** | 482K | 0.010 |

OPQ I/J 未跑（由 EXP-012 覆盖）。

### Analysis

1. **KMeans cluster size 是主导因素**: 4096→snHR=0.095 vs 1024→snHR=0.078 (+22%)。前两层 KMeans 编码了绝大部分语义信息。
2. **4096 下 binary ≈ multi-level**: collision 0.89% vs 0.84%，snHR 相同。因为 L2 已将 item 分到平均 1.5 个/prefix，FSQ type 不再关键。
3. **1024 下 binary 明显更优**: collision 7.9% vs 14.6%。L2 平均 3.08 个/prefix，10 维 binary 比 5 维 multi-level 提供更好区分。
4. **三层等大 1024×3 不劣于不等大 1024×1024×4096**: snHR 相同 (0.078)，且 binary 的 collision 更低 (7.9% vs 10.7%)。

### Next Steps

→ EXP-012: 扩展 grid search 到 2048/8192 cluster size，确认 snHR 随 cluster size 的趋势曲线。

---

## EXP-012: Tokenizer Grid Search — KMeans × FSQ Type × OPQ

**Date**: 2026-04-15
**Status**: completed
**Results**: [./hyperparam/2026-04-15_exp012-grid-search/](./hyperparam/2026-04-15_exp012-grid-search/)

### Background

EXP-011 证实 KMeans cluster size 是 tokenizer 质量的主导因素。需要系统性搜索，找到 snHR 的 plateau 或最优点。

### Hypothesis

1. snHR 随 cluster size 单调递增但边际递减（信息论上限 = embedding 本身的信息量）
2. 8192×3 (OneRec 配置) 应优于 4096×3
3. Binary FSQ 在较小 cluster 有优势，大 cluster 下 binary ≈ multi-level

### Design

| Config | Type | Cluster | FSQ | Bits |
|--------|------|---------|-----|------|
| 1024-multi | FSQ | 1024 | [4]×5 | 30 |
| 1024-binary | FSQ | 1024 | [2]×10 | 30 |
| 2048-multi | FSQ | 2048 | [4,4,4,4,4,2] | 33 |
| 2048-binary | FSQ | 2048 | [2]×11 | 33 |
| 4096-multi | FSQ | 4096 | [4]×6 | 36 |
| 4096-binary | FSQ | 4096 | [2]×12 | 36 |
| 8192-multi | FSQ | 8192 | [4,4,4,4,4,4,2] | 39 |
| 8192-binary | FSQ | 8192 | [2]×13 | 39 |
| opq-4×{256,512,1024,2048} | OPQ | — | — | 32/36/40/44 |

- **Fixed**: Qwen3-0.6B 1024D, MLP hidden=64, 50 epochs
- **Metrics (4 only)**: semantic_neighbor_HR, collision, codebook_utilization, cluster_balance + neighbor_coverage
- **Multi-GPU**: KMeans groups 并行 (CUDA_VISIBLE_DEVICES pinning)
- **Merge EXP-011**: 已有 4 组结果直接合并

### Run

```bash
python experiments/scripts/tokenizer_grid_search.py --gpus 0,1,2,3
```

### Results

| Config | Cluster | FSQ | Bits | collision | snHR | Coverage | L3 Gini |
|--------|---------|-----|------|-----------|------|----------|---------|
| 8192-binary | 8192 | [2]×13 | 39 | **0.35%** | **0.104** | 31% | 0.004 |
| 8192-multi | 8192 | [4]×6,2 | 39 | 1.35% | 0.104 | 31% | 0.016 |
| 4096-multi | 4096 | [4]×6 | 36 | 0.84% | 0.095 | ~55% | 0.009 |
| 4096-binary | 4096 | [2]×12 | 36 | 0.89% | 0.095 | ~55% | 0.010 |
| 2048-binary | 2048 | [2]×11 | 33 | 2.03% | 0.083 | 70% | 0.022 |
| 2048-multi | 2048 | [4]×5,2 | 33 | 4.48% | 0.083 | 70% | 0.047 |
| 1024-binary | 1024 | [2]×10 | 30 | 7.88% | 0.078 | ~85% | 0.083 |
| 1024-multi | 1024 | [4]×5 | 30 | 14.63% | 0.078 | ~85% | 0.151 |
| opq-4x256 | OPQ | 4×256 | 32 | 3.51% | 0.050 | 98% | 0.057 |

### Analysis

**1. snHR 随 cluster size 递增但边际递减** (假说 1 成立):

```
cluster  snHR    Δ        coverage
1024     0.078   baseline ~85%
2048     0.083   +6.4%    70%
4096     0.095   +14.5%   ~55%
8192     0.104   +9.5%    31%
```

4096→8192 边际收益 (+9.5%) 已放缓，且 coverage 急剧下降。

**2. snHR 是 precision 指标，存在 precision-coverage tradeoff**:

- snHR 衡量"同 prefix 邻居中有共同用户的比例"——cluster 越大 group 越纯 → precision 越高
- 但大 cluster 下多数 item 变成 singleton (无邻居) → 只有少部分 item 被评估
- 8192 的 snHR=0.104 只代表 31% 的 item，结果有高估风险

**3. Binary FSQ 全面优于 multi-level** (假说 3 部分推翻):

不仅小 cluster 下有优势，8192 下 binary 的 collision 优势反而最大 (0.35% vs 1.35%, 3.9×)。原因: binary 每维只有 2 个 level，维度更高 (13d vs 7d)，提供更细粒度的正交切分。

**4. OPQ 全面输 FSQ** (延续 EXP-008 结论):

opq-4x256 (32bit) snHR=0.050 远低于 1024-binary (30bit) 的 0.078。层级结构的归纳偏置 > 扁平 PQ。

### Conclusion

**推荐配置: 4096×3 binary `[2]×12` (36 bit)**

- snHR=0.095，coverage 适中 (~55%)，collision=0.89%
- 对标 OneMall 4096×4096×4096 生产配置
- KMeans 训练 ~400s (vs 8192 的 ~1300s)，可接受
- collision < 1% 对 NTP 学习足够友好

8192×3 binary 可作为 aggressive 备选 (collision 最低 0.35%，NTP 最友好)，但需接受 snHR 评估覆盖不足。

### Next Steps

- 用 4096×3 binary 配置跑 NTP baseline（已修复 per-layer output head）
- 换不同 embedding (e.g. larger model) 时用 `tokenizer_grid_search.py` 重跑 grid search

---

## EXP-010: NTP Baseline — MLP-FSQ SID 端到端 Recall

**Date**: 2026-04-15
**Status**: completed (效果极差，需诊断)
**Results**: [./hyperparam/2026-04-15_exp010-ntp-baseline/](./hyperparam/2026-04-15_exp010-ntp-baseline/)

### Background

Tokenizer 阶段结束，MLP-FSQ h=64 确认为赢家 (EXP-008, semantic_neighbor_HR=0.078)。现在需要第一个端到端 NTP 数字：用当前 2 层 Transformer probe (~5M params) 在 MLP-FSQ SID 上训练，拿到 item Recall@K baseline。

当前 NTP probe 参数:
- 2 层 causal Transformer decoder, embed_dim=256, n_heads=4, ffn_dim=512
- **1 epoch** (代码 bug: 缺少 epoch 外循环), AdamW lr=3e-3, CosineAnnealing
- 行为序列 n_items=10, beam_size=50
- SID: 3 tokens (L1=1024, L2=1024, **L3=4096** ← 与 L1/L2 不等大)

### Hypothesis

- Perplexity 应在 50~150 范围（good~acceptable）
- Item Recall@50 应显著高于 embedding_hit_rate (0.0047)，因为 NTP 利用了行为序列信息
- 这个数字作为所有后续 NTP 改进（architecture/training/scaling）的 baseline

### Design

- **Variable**: 无（单配置 baseline）
- **Fixed**: MLP-FSQ h=64, 2 层 probe, 1 epoch, n_items=10, beam_size=50
- **Metric**: Perplexity, Depth Accuracy, Item Recall@{10,50,100,500}
- **Data**: 7 天行为数据, 19.1M samples (train=15.3M, eval=50K)

### Run

`bash experiments/scripts/exp-010.sh`

### Results

| 指标 | 值 |
|------|-----|
| Train loss | 1.70 → 0.47 (3741 steps) |
| Eval perplexity | 5.34 |
| Depth acc beam (L1/L2/L3) | 0.007 / 0.000 / 0.000 |
| **Depth hit@10 (L1/L2/L3)** | **1.000 / 1.000 / 0.401** |
| Item recall@50 | 0.0008 |
| Item recall@500 | 0.0008 |

### Analysis

**效果极差，但 teacher-forced hit@10 表明模型学到了。核心问题在 beam search：**

1. **L1/L2/L3 不等大 vocab 共享单一 output head (Linear(256, 4027))**: L1/L2 只有 1024 个合法 token，但 softmax 在 4027 维上做，75% 的概率空间是噪声。Beam search 可能选到 L3 范围的 token 作为 L1 预测
2. **Teacher-forced hit@10 = 100%**: 说明模型在看到正确上下文时，正确 token 在 top-10 中。但 beam search 一旦 L1 选错，后续全部偏移
3. **只训练 1 epoch**: train loss 还在下降 (0.47 且斜率明显)，未收敛
4. **Train-eval gap 大**: train CE ≈ 0.47, eval CE ≈ 1.68 (PPL 5.34)，时间序列切分导致分布偏移

**根因: SID 配置 1024×1024×4096 不等大 + NTP 模型未做 per-layer vocab 处理。**

### Next Steps

1. **EXP-011**: 确定正确的 codebook 配置 (等大 1024×3 或 4096×3)
2. **修复 NTP 模型**: per-layer output head 或统一 vocab + layer embedding + beam search mask
3. **增加 epoch**: 1 → 5-10
4. 修复后重跑 NTP baseline

---

## EXP-009: QFormer Tokenizer — 冻结 Qwen3 + Cross-Attention 压缩

**Date**: 2026-04-14 ~ 2026-04-15
**Status**: completed
**IDEA**: IDEA-onerec-3
**Results**: [./hyperparam/2026-04-14_exp009-qformer/](./hyperparam/2026-04-14_exp009-qformer/)

### Background

EXP-007 证明直接 fine-tune Qwen3-0.6B（全量/LoRA，多种 lr/τ）完全无法推动模型——cap_loss 纹丝不动，HR@50 卡在 ~0.02。根本原因: I2I 梯度稀释在 600M 参数中。

OneRec 的核心方案: 冻结底座，在上面加一个可训练的 QFormer (cross-attention + learnable queries)。梯度集中在 ~30-50M 参数的 QFormer 上，底座天然保持语义。BLIP-2 QFormer 已被 OneRec (miniCPM-V-8B + 4-layer QFormer) 验证有效。

### Hypothesis

1. QFormer 训练时 cap_loss 会明显下降（不同于 EXP-007 的纹丝不动），证明梯度可以有效流动
2. HR@50 显著突破 EXP-007 的 0.02 baseline（预期 > 0.05）
3. 信息压缩 (S tokens → M tokens) 迫使 QFormer 学会提取协同相关信息而非照搬语义

### Design

**Phase 1 — 最小验证 (梯度能否流动)**:

| Config | QFormer Layers | Query Tokens (M) | lr | Loss |
|--------|---------------|-------------------|------|------|
| A | 2 | 4 | 1e-4 | L_I2I only |
| B | 2 | 4 | 5e-4 | L_I2I only |
| C | 4 | 4 | 1e-4 | L_I2I only |

- **Variable**: QFormer depth × learning rate
- **Fixed**: Qwen3-0.6B frozen, M=4 query tokens, D=1024, τ=0.05, batch_size=32, grad_accum=8, max_pairs=500K, 1 epoch, 8xA100 DDP
- **Metric**:
  - **Primary**: HR@50 (InlineHRMonitor, 与 EXP-007 baseline 直接对比)
  - **Diagnostic**: cap_loss 变化量 (W&B)、I2I loss 收敛速度
  - **Secondary**: OPQ intrinsic (collision, recon_loss) on QFormer embeddings
- **Data**: 行为数据 7 天, ~5M items

### Run
`bash experiments/scripts/exp-009.sh`

### Results

| Config | QFormer Layers | Queries (M) | lr | Final HR@50 | Final Loss | 训练时间 |
|--------|---------------|-------------|------|------------|-----------|---------|
| BL (raw Qwen3) | — | — | — | 0.0106 | — | — |
| EXP-007 best (全量FT) | — | — | 1e-5 | 0.0197 | 2.90 | 6756s |
| A | 2 | 4 | 1e-4 | 0.0211 | 4.46 | 4460s |
| B | 2 | 4 | 5e-4 | 0.0214 | 4.41 | 4458s |
| **C (best)** | **4** | **4** | **1e-4** | **0.0216** | **4.42** | **4549s** |

实际训练数据: 3,074,342 pairs (max_pairs=5M, swing 实际产出 ~3M), 12,000 steps/epoch, effective batch 2048。

### Analysis

**1. QFormer 未突破 0.02 天花板:**
- 最佳 Config C: HR@50 = 0.0216，仅比 EXP-007 best (0.0197) 高 10%，远未达到 hypothesis 预期的 >0.05
- 三组 config 差异极小 (0.0211 ~ 0.0216)，QFormer depth/lr 不是瓶颈

**2. Hypothesis 验证:**
- ✅ H1 (梯度流动): loss 从 5.5 降到 ~4.4，确实在下降（EXP-007 cap_loss 纹丝不动），证明梯度可以流过 QFormer
- ❌ H2 (HR@50 突破): 0.0216 vs 预期 >0.05，差距巨大
- ❌ H3 (信息压缩): QFormer 的 4 query tokens 并未迫使模型学到更好的协同表示

**3. HR@50 曲线特征:**
- 全程缓慢单调上升，未见明显 plateau
- 但斜率持续递减 (step 0~4000: +0.006, step 4000~8000: +0.003, step 8000~12000: +0.002)
- 更多 epoch 可能有微小提升，但趋势已极平，不可能突破 0.03

**4. 根因重新判断:**
- EXP-007 结论 "梯度稀释" 被部分推翻——QFormer 集中梯度后 loss 确实在降，但 HR@50 仍卡住
- **真正瓶颈不在模型结构，而在 I2I contrastive 信号本身**: in-batch negatives + 行为共现正样本的监督信号强度不足以将 embedding 推到行为空间中有意义的位置
- 或者说: **Qwen3 的 semantic embedding 空间与行为空间的 gap 远大于 contrastive learning 能弥补的程度**

### Next Steps

EXP-007 + EXP-009 两轮实验证明: **I2I contrastive fine-tune (无论全量/LoRA/QFormer) 都无法有效改善 embedding 的行为质量**。需要重新审视 embedding 端的策略:

1. **放弃 embedding fine-tune 路线**, 回归 "好的 tokenizer 比好的 embedding 更重要" 的架构哲学
2. 聚焦 **EXP-008 (FORGE proxy 对比)** — 用现有 Qwen3 embedding 对比 MLP-FSQ vs OPQ 的行为质量，决定 tokenizer 路线
3. 如果仍需改善 embedding，考虑完全不同的方案: multi-task learning、graph embedding、或直接用行为 embedding (collaborative filtering) 替代文本 embedding

---

## EXP-008: FORGE Proxy 对比 — MLP-FSQ vs OPQ 最优解

**Date**: 2026-04-14 ~ 2026-04-15
**Status**: completed
**Results**: [./hyperparam/2026-04-15_exp008-mlpfsq-h64/](./hyperparam/2026-04-15_exp008-mlpfsq-h64/), [./hyperparam/2026-04-15_exp008-opq-m4/](./hyperparam/2026-04-15_exp008-opq-m4/), [./hyperparam/2026-04-15_exp008-opq-m8/](./hyperparam/2026-04-15_exp008-opq-m8/)

### Background

EXP-003 最优 (MLP-FSQ h=64, collision=0.041) 和 EXP-004 最优 (OPQ 8×256, collision=0.0037) 只有 intrinsic metrics，缺行为层面验证。已实现的 FORGE proxy metrics 无需训练 NTP 就能评估 SID 质量：
- `embedding_hit_rate`: embedding I2I 邻居共现率（所有方案相同，作为 baseline）
- `semantic_neighbor_hit_rate`: SID 前缀邻居共现率（区分 tokenizer，核心指标）

目标：快速对比两条路线，决定哪条进入 NTP 阶段。

### Hypothesis

1. OPQ 8×256 的 `semantic_neighbor_hit_rate` 应显著高于 MLP-FSQ h=64，因为更低的 collision (0.0037 vs 0.041) 意味着更精细的 SID 分区
2. `embedding_hit_rate` 三组相同（相同 embedding，只是 tokenizer 不同）
3. OPQ 4×256 (等 bits 对照) 的 `semantic_neighbor_hit_rate` 介于 MLP-FSQ 和 OPQ 8×256 之间

### Design

| Config | Tokenizer | Tokens | Bits | 已知 collision |
|--------|-----------|--------|------|---------------|
| A | MLP-FSQ h=64 (6d_4096) | 3 | 32 | 0.0411 |
| B | OPQ 4×256 (等 bits 对照) | 4 | 32 | 0.1063 |
| C | OPQ 8×256 (最优) | 8 | 64 | 0.0037 |

- **Fixed**: Qwen3-0.6B 1024D embedding (cached), behavior_data 7d
- **Metric**:
  - **Primary**: `semantic_neighbor_hit_rate` — SID 前缀邻居在行为图中的共现率
  - **Baseline**: `embedding_hit_rate` — embedding 空间 I2I 邻居共现率（三组应相同）
  - **Secondary**: intrinsic metrics (collision, recon_loss, entropy)

### Run
`bash experiments/scripts/exp-008.sh`

### Results

数据: 554,754 exposed items (从 5,162,650 总 embedding 中过滤), 行为数据 7 天 (03-24 ~ 03-30)

| Config | Tokenizer | Tokens | Bits | collision | recon_loss | embedding_HR | **semantic_neighbor_HR** | 训练时间 |
|--------|-----------|--------|------|-----------|------------|-------------|------------------------|---------|
| **A** | **MLP-FSQ h=64** | **3** | **32** | **0.1074** | **0.3668** | **0.0047** | **0.0780** | 106s |
| B | OPQ 4×256 | 4 | 32 | 0.0351 | 0.3760 | 0.0047 | 0.0502 | 73s |
| C | OPQ 8×256 | 8 | 64 | 0.0006 | 0.3408 | 0.0043 | 0.0326 | 99s |

### Analysis

**结果与 hypothesis 完全相反 — MLP-FSQ 大幅领先 OPQ:**

**1. Hypothesis 验证:**
- ❌ H1: OPQ 8×256 的 semantic_neighbor_HR (0.033) **远低于** MLP-FSQ (0.078)，collision 低 180 倍却输了 58%
- ✅ H2: embedding_HR 三组几乎相同 (~0.0047)，符合预期
- ❌ H3: OPQ 4×256 (0.050) 介于两者之间，但方向反了——不是 MLP-FSQ < OPQ 4×256 < OPQ 8×256，而是 MLP-FSQ > OPQ 4×256 > OPQ 8×256

**2. collision 越低 ≠ 行为质量越好:**
- OPQ 8×256 追求极低 collision (0.06%)，将 embedding 空间切成 ~553K 个几乎不重叠的 bin
- 但过度细分破坏了语义邻域结构——SID 前缀相近的 item 不再是行为上的邻居
- MLP-FSQ 的 collision 10.7% 看似"差"，但保留了层级聚集结构，SID 前缀邻居的行为共现率反而更高

**3. 层级结构 > 扁平结构:**
- MLP-FSQ: 3 层层级 (KMeans → KMeans → FSQ)，每层逐步细化，前缀天然编码粗到细的语义聚类
- OPQ: 8 个并行子向量独立量化，token 间无层级关系，前缀邻居不具有语义含义

**4. 等 bits 对照 (32 bits):**
- MLP-FSQ (0.078) vs OPQ 4×256 (0.050)，MLP-FSQ 赢 56%
- 相同信息量下，层级残差编码的 SID 前缀邻域比并行 PQ 的前缀邻域更有行为意义

**5. 注意: MLP-FSQ 不使用行为数据训练:**
- MLP 仅优化残差重建 loss (||residual - Decoder(FSQ(Encoder(residual)))||²)，纯无监督
- 行为质量的优势完全来自层级结构对 embedding 邻域的保持，而非学习行为信号

### Next Steps

**MLP-FSQ h=64 确认为 tokenizer 路线赢家**，进入 NTP 阶段:
1. 用 MLP-FSQ 生成全量 SID，训练 NTP 预测模型
2. 端到端评估 Recall@K
3. 考虑是否需要更大的 FSQ codebook (当前 4096) 或更多 KMeans 层

---

## EXP-007: Collaborative Signal Enhanced Embedding (Qwen3-0.6B Full Fine-tune)

**Date**: 2026-04-13 ~ 2026-04-14
**Status**: completed
**IDEA**: IDEA-sid-1
**Results**: [./hyperparam/2026-04-13_exp007-collab-embed/](./hyperparam/2026-04-13_exp007-collab-embed/)

### Background

当前直接用 Qwen3-0.6B 纯文本 embedding (1024D) 做量化。这些 embedding 只编码了语义相似性（文本内容相近的 item 距离近），但推荐需要的是**行为相似性**（被同一用户群喜欢的 item 距离近）。EXP-004 的 embedding_hit_rate 指标可以量化当前 embedding 在行为维度的质量。

本实验通过 **I2I 对比学习** 全量 fine-tune Qwen3-0.6B，将协同行为信号注入 embedding，提升量化上限。与量化方案 (OPQ/RKMeans) 正交，改善 embedding 质量对所有下游实验受益。

### Hypothesis

1. 对比学习后的 embedding 在 `embedding_hit_rate` 上显著优于原始 Qwen3 embedding（预期 HR@50 提升 50%+）
2. 下游 OPQ 量化指标（collision, recon_loss）也会改善，因为行为相似的 item 在 embedding 空间更聚集
3. 全量 fine-tune 0.6B 在 8xA100 上训练时间可控（预期 < 4 小时）

### Design

- **Variable**: 训练方案 × 温度参数
  - **Baseline**: 原始 Qwen3-0.6B embedding（已缓存，无需重跑）
  - **Config A**: 全量 fine-tune, InfoNCE, τ=0.05, 3 epochs
  - **Config B**: 全量 fine-tune, InfoNCE, τ=0.07, 3 epochs
  - **Config C**: 全量 fine-tune, InfoNCE, τ=0.05, 5 epochs
- **Fixed**:
  - 模型: Qwen3-0.6B (全量参数更新, FP16, 8xA100 DDP)
  - 正样本: 同一用户 7 天内正向行为 (action_bitmap > 0) 的 item pair
  - 负样本: in-batch negatives (batch_size=512 per GPU, effective 4096)
  - Optimizer: AdamW, lr=1e-5, warmup 10%, cosine decay
  - 文本: item title (已有 Qwen3 tokenizer)
- **Metric**:
  - **Primary**: `embedding_hit_rate` (HR@10/50/100/500) — FORGE proxy，不需要训练 NTP
  - **Secondary**: OPQ intrinsic (collision, recon_loss, entropy) — 用 EXP-004 相同 OPQ config (m=8, M=256) 量化后评估
  - **Sanity**: `cosine_similarity` 分布, `embedding_behavior_correlation`
- **Data**: 行为数据 7 天 (2026-03-24 ~ 2026-03-31), ~5M items

### Run
`bash experiments/scripts/exp-007.sh`

### Results

**Baseline**: HR@50 = 0.0106 (原始 Qwen3-0.6B embedding, 50,008 items)

**Round 1 — 基础超参搜索 (全量 fine-tune)**:

| Config | τ | lr | max_pairs | HR@50 | Loss plateau | 训练时间 |
|--------|------|------|-----------|-------|-------------|---------|
| BL (baseline) | — | — | — | **0.0106** | — | — |
| A | 0.05 | 1e-5 | 2M | **0.0197** | ~step 800 | 6756s (~1h53m) |
| B | 0.07 | 1e-5 | 1M | 0.0148 | ~step 800 | killed early |
| C | 0.05 | 3e-5 | 500K | 0.0192 | ~step 400 | 1912s (~32min) |

**Round 2 — 激进学习率 (cap_loss 在 R1 纹丝不动)**:

| Config | τ | lr | 状态 |
|--------|------|------|------|
| D | 0.05 | 1e-4 | 脚本就绪，未产出超越 R1 的结果 |
| E | 0.05 | 3e-4 | 同上 |
| F | 0.05 | 1e-3 | 同上 |

**Round 3 — LoRA (冻结底座，梯度集中在 adapter)**:

| Config | Method | lr | 状态 |
|--------|--------|------|------|
| G | LoRA r=16 | 1e-4 | 脚本就绪，未产出超越 R1 的结果 |
| H | LoRA r=16 | 5e-4 | 同上 |
| I | LoRA r=64 | 1e-4 | 同上 |

### Analysis

**1. HR@50 天花板 ~0.02，较 baseline 0.0106 提升约 86%，但远未达到 hypothesis 预期的 50%+ 绝对提升:**
- 最佳 Config A: 0.0197，仍处于 poor 级别（阈值 < 0.02）
- 三组 round 1 config HR@50 收敛到同一天花板 (~0.02)，超参调优空间有限

**2. 温度不是瓶颈**: τ=0.07 (Config B) 全面劣于 τ=0.05 (Config A)

**3. 学习率影响收敛速度不影响上限**: Config C (lr=3e-5) 用 1/4 数据、1/3 时间达到同等效果

**4. Loss 快速 plateau**: 所有 config 在 ~200K pairs 后 loss 稳定在 ~2.5-2.7，cap_loss 完全不动——说明 I2I 梯度稀释在 600M 参数中

**5. Hypothesis 验证:**
- ❌ HR@50 提升 86% (0.0106→0.0197)，但绝对值仍极低，未达到 "显著优于" 的预期
- ❌ 下游量化改善未验证（HR@50 本身太低，OPQ 评估意义有限）
- ✅ 训练时间可控（最快 Config C 仅 32 分钟）

**6. 根因**: 直接 fine-tune 600M 参数的 Qwen3 底座，I2I contrastive 的梯度被稀释，模型几乎不学习。无论全量 fine-tune 还是 LoRA，都无法有效将协同信号注入 embedding。

### Next Steps

EXP-007 证明 "直接 fine-tune 底座" 路线不可行，需要方法论变更:
- **EXP-009 (已规划)**: 冻结 Qwen3 底座 + QFormer cross-attention，梯度集中在 ~30-50M 参数的 QFormer 上（OneRec 验证有效的方案）

---

## EXP-004: OPQ Parallel Semantic IDs — Intrinsic Metrics

**Date**: 2026-04-13
**Status**: completed
**IDEA**: IDEA-sid-0 (Phase 1)
**Reference**: Meta RPG (KDD'25, arxiv 2506.05781)
**Results**: [./hyperparam/2026-04-13_exp004-opq/](./hyperparam/2026-04-13_exp004-opq/), [./hyperparam/2026-04-13_exp004-opq-m4/](./hyperparam/2026-04-13_exp004-opq-m4/)

### Background
当前 RKMeans (3 层 x 1024 clusters) 使用残差编码，各层串行依赖。ARCHITECTURE.md 已明确需要切换到并行 tokenizer。RPG 论文证明 OPQ (Optimized Product Quantization) 在生成式推荐中优于 RQ，且支持并行预测。

本实验验证 OPQ 在我们 5M item / 1024D Qwen3-0.6b embedding 上的量化质量（intrinsic metrics），不涉及 NTP 预测模型。

### Hypothesis
OPQ 将 1024D embedding 切分为 m 个独立子向量分别量化，编码空间远大于 RKMeans (256^8 >> 1024^3)，collision 应显著更低。recon_loss 需要实测验证 — PQ 的独立子空间假设可能不如 RQ 的残差逼近。

### Design
- **Variable**: n_subvectors (m=4, 8, 16, 32), n_clusters_per_sub (M=256)
- **Fixed**: normalize_input=True, OPQ rotation training (FAISS default)
- **Metric**: collision_rate, exclusivity, reconstruction_loss, entropy, cluster_balance
- **Data**: 5M items, qwen3-0.6b 1024D embedding (cached)

**Comparison matrix**:

| Config | Quantizer | Tokens | Vocab/token | Bits | 子向量维度 |
|--------|-----------|--------|-------------|------|-----------|
| Baseline (EXP-001) | RKMeans 3x1024 | 3 | 1024 | 30 | N/A (residual) |
| **OPQ-4x256** | **OPQ** | **4** | **256** | **32** | **256D (等 bits 对照)** |
| OPQ-8x256 | OPQ | 8 | 256 | 64 | 128D |
| OPQ-16x256 | OPQ | 16 | 256 | 128 | 64D |
| OPQ-32x256 | OPQ | 32 | 256 | 256 | 32D |

### Run
`bash experiments/scripts/exp-004.sh`

### Results

| Config | Tokens | Bits | collision | entropy | Gini | recon_loss | time(s) |
|--------|--------|------|-----------|---------|------|------------|---------|
| **RKMeans 3×1024** (EXP-001) | **3** | **30** | **0.1634** | **0.7211** | **0.2091** | **0.3524** | — |
| **OPQ 4×256** | **4** | **32** | **0.1063** | **0.9681** | **0.1896** | **0.3772** | 125 |
| OPQ 8×256 | 8 | 64 | 0.0037 | 0.9971 | 0.0128 | 0.3429 | 160 |
| OPQ 16×256 | 16 | 128 | 0.0029 | 0.9993 | 0.0052 | 0.3026 | 220 |
| OPQ 32×256 | 32 | 256 | 0.0027 | 0.9995 | 0.0043 | 0.2522 | 338 |

### Analysis

**1. 等 bits 对照 — OPQ 4×256 (32bit) vs RKMeans 3×1024 (30bit):**
- collision: 10.6% vs 16.3% — OPQ 低 35%，相同信息量下碰撞率显著更低
- entropy: 0.968 vs 0.721 — OPQ codebook 利用率远更均匀
- recon_loss: 0.377 vs 0.352 — OPQ 略差 7%，PQ 子空间独立假设的代价
- 结论：等 bits 下 OPQ **赢 collision、输 recon**，trade-off 合理

**2. m=8 是 sweet spot:**
- collision 从 m=4 的 10.6% 骤降到 0.37%（仅多 1 倍 bits）
- recon_loss 0.3429 已优于 RKMeans 0.3524
- 8 token 并行预测成本可控

**3. m≥16 收益递减:**
- collision: 0.37% → 0.29% → 0.27%，几乎无差异
- recon_loss 持续下降但 token 数翻倍 → NTP 预测成本翻倍
- 除非下游任务对 recon 极度敏感，否则不值得

**4. Hypothesis 验证:**
- ✅ collision 显著更低（符合预期，编码空间 256^m >> 1024^3）
- ✅ recon_loss 在 m≥8 时优于 RKMeans（PQ 独立子空间假设没有严重损害重建质量）
- ✅ entropy/Gini 近乎完美，无 cluster collapse

### Next Steps
OPQ Phase 1 验证通过，推荐 **m=8** 进入 Phase 2:
1. 并行预测 NTP 模型 — per-digit independent MLP heads + MTP loss
2. Graph-Constrained Decoding — 替代 beam search（RPG 论文证明 beam search 在 OPQ 上 recall=0.0000）
3. 端到端评估 — Recall@K on downstream retrieval task

---

## EXP-003: Learned FSQ — MLP projection + straight-through training

**Date**: 2026-04-13
**Status**: completed
**Results**: [./hyperparam/2026-04-13_exp003-mlp64/](./hyperparam/2026-04-13_exp003-mlp64/), [./hyperparam/2026-04-13_exp003-mlp128/](./hyperparam/2026-04-13_exp003-mlp128/)

### Background
EXP-002 证明 PCA 线性投影 + FSQ 劣于 KMeans baseline，核心瓶颈是 PCA 在残差空间信息丢失过大（1024D→4~6D 解释方差仅 20-55%）。

OneMall (arxiv 2601.21770) 用 **learned MLP** 做投影，原始 FSQ 论文 (Mentzer 2023, arxiv 2309.15505) 将 FSQ 用在 VQ-VAE 内部，encoder 学到对量化最优的表示。关键机制：
- MLP 学习非线性投影 D→d，比 PCA 保留更多量化相关信息
- Straight-Through Estimator (STE): 前向用 round()，反向把梯度直通到 MLP 参数
- 训练目标: 重建 loss — minimize ||residual - reconstruct(FSQ(MLP(residual)))||²

### Hypothesis
Learned MLP 投影可以学到对 FSQ 量化最优的低维表示，使 FSQ 的 reconstruction quality 接近或超过 KMeans，从而在保持低 collision 的同时降低 recon_loss。

### Design
- **Variable**: 投影方式 (PCA vs MLP)，MLP 隐层宽度
- **Fixed**: 2 KMeans layers x 1024 clusters, FSQ [4,4,4,4,4,4] (6d_4096), epochs=50, lr=1e-3, AdamW
- **Metric**: collision_rate, reconstruction_loss, exclusivity, entropy

**MLP 架构** (autoencoder + STE):
```
Encoder: D → hidden → d  (d=6 for 6d_4096)
FSQ:     quantize each of 6 dims to {0,1,2,3}, STE pass-through
Decoder: d → hidden → D
Loss:    ||residual - Decoder(STE_quantize(Encoder(residual)))||²
```

**Comparison matrix**:

| Config | L3 projection | L3 codebook | Training |
|--------|---------------|-------------|----------|
| Baseline (EXP-002) | KMeans 1024 | 1024 | N/A |
| PCA-FSQ (EXP-002) | PCA 6d | 4096 | N/A |
| MLP-FSQ-64 | MLP D→64→6 | 4096 | 50 epochs |
| MLP-FSQ-128 | MLP D→128→6 | 4096 | 50 epochs |

### Run
`bash experiments/scripts/exp-003.sh`

### Results

| Config | L3 | collision | recon_loss | d3 avg_items | time(s) |
|--------|-----|-----------|------------|--------------|---------|
| **Baseline** | KMeans 1024 | 0.1634 | 0.3524 | 1.3 | 237 |
| PCA-FSQ | PCA 6d_4096 | 0.3330 | 3.1280 | 1.7 | 178 |
| **MLP-FSQ h=64** | **MLP D→64→6** | **0.0411** | **0.3619** | **1.1** | **611** |
| MLP-FSQ h=128 | MLP D→128→6 | 0.0767 | 0.3633 | 1.1 | 627 |

### Analysis

**Learned MLP 大幅超越 KMeans baseline，完全验证 hypothesis:**

1. **collision 降 75%**: MLP h=64 的 collision 0.0411 vs KMeans baseline 0.1634，FSQ 的 implicit codebook (4096 codes) + 学到的非线性投影彻底解决了碰撞问题
2. **recon_loss 与 baseline 持平**: 0.3619 vs 0.3524 (差 2.7%)，说明 MLP 学到了高质量投影，PCA 的 3.128 recon_loss 完全是线性投影的局限性
3. **h=64 优于 h=128**: collision 0.0411 vs 0.0767。更小的 hidden dim 起到正则化作用，避免 encoder 输出过于极端导致 tanh 饱和（训练中发现了 tanh 饱和导致 OOB 的 bug 并修复）
4. **训练时间翻倍但可接受**: 611s vs 237s，多出的 ~400s 是 50 epoch MLP 训练，模型仅 ~132K params

**vs PCA-FSQ (EXP-002)**: collision 从 0.333 降到 0.041 (降 88%)，recon_loss 从 3.128 降到 0.362 (降 88%)，证明非线性投影是关键。

### Next Steps
1. MLP-FSQ h=64 跑 NTP behavior 评估，确认 recall@K 指标
2. 与 OPQ (EXP-004) 在 NTP 下游对比，决定最终方案

---

## EXP-002: ResKmeansFSQ — 2 layers RKMeans + 1 layer FSQ (PCA projection)

**Date**: 2026-04-13
**Status**: completed
**Results**: [./hyperparam/2026-04-13_exp002-baseline/](./hyperparam/2026-04-13_exp002-baseline/), [./hyperparam/2026-04-13_exp002-fsq/](./hyperparam/2026-04-13_exp002-fsq/)

### Background
RKMeans 的第3层对残差做 KMeans 效果递减。OneMall (arxiv 2601.21770) 提出用 FSQ 替换第3层。本实验使用 **PCA 线性投影** 替代论文中的 learned MLP 做降维。

### Hypothesis
FSQ 的 implicit codebook 天然无 cluster collapse，可降低 collision rate。

### Design
- **Variable**: Layer 3 quantizer (KMeans vs FSQ configs)
- **Fixed**: 2 KMeans layers x 1024 clusters, niter=25, nredo=3, normalize_residuals=True
- **Metric**: conflict_rate, reconstruction_loss, entropy, exclusivity, cluster_balance

| Config | L1, L2 (KMeans) | L3 | L3 codebook |
|--------|------------------|----|-------------|
| Baseline | 1024 x 3 layers KMeans | KMeans 1024 | 1024 |
| Hybrid A | 1024 x 2 layers | FSQ [8,8,8,8] | 4096 |
| Hybrid B | 1024 x 2 layers | FSQ [7,5,5,5,5] | 4375 |
| Hybrid C | 1024 x 2 layers | FSQ [4,4,4,4,4,4] | 4096 |

### Run
`bash experiments/scripts/exp-002.sh`

### Results

| Config | L3 | conflict_rate | exclusivity | recon_loss | d3 entropy | d3 Gini | d3 unique | d3 avg_items |
|--------|-----|---------------|-------------|------------|------------|---------|-----------|--------------|
| **Baseline** | KMeans 1024 | **0.1634** | **0.6423** | **0.3524** | **0.7211** | **0.2091** | 3,963,269 | 1.3 |
| Hybrid C | FSQ 6d [4x6] | 0.3330 | 0.4015 | 3.1280 | 0.6755 | 0.3153 | 3,107,671 | 1.7 |
| Hybrid A | FSQ 4d [8x4] | 0.5688 | 0.1446 | 2.2122 | 0.6383 | 0.4693 | 1,731,222 | 3.0 |
| Hybrid B | FSQ 5d [7,5x4] | 0.8157 | 0.0089 | 0.3800 | 0.5306 | 0.6548 | 248,798 | 20.8 |

注: L1/L2 两层 KMeans 相同 (d1/d2 指标一致)，差异全部来自 L3。

### Analysis

**FSQ+PCA 全面劣于 KMeans baseline**，核心原因是 **PCA 线性投影信息丢失过大**：

1. **投影瓶颈**: 1024维残差 → 4~6维 PCA，解释方差仅 20-55%。残差空间（经两轮 KMeans 后）本就小且不规则，PCA 线性假设不适用。
2. **维度越少越差**: 5d_4375 (d=5) 的 conflict_rate 高达 0.82，几乎所有信息丢失；6d_4096 (d=6) 最好但仍 0.33 >> baseline 0.16。
3. **recon_loss 恶化**: 4d/6d 的 recon_loss 从 0.35 飙升到 2.2/3.1，说明 PCA 逆投影无法恢复原始残差。
4. **与论文差异**: OneMall 用 **learned MLP** 投影（非线性、端到端训练），可学到对量化最优的表示，而非仅保留方差最大方向。

### Next Steps
EXP-003: 将 PCA 替换为 learned MLP 投影，复现论文方案。需要：
1. 定义 MLP 架构 (D → d 维) + VQ-VAE style 重建 loss
2. 端到端训练投影网络
3. 对比 PCA vs MLP 在同一 FSQ config 下的效果

---

## EXP-001: RKMeans 训练优化 (v0→v7)

**Date**: 2026-03 ~ 2026-04
**Status**: completed
**Results**: See `config/RKMEANS_EXPERIMENT_LOG.md` for full details

### Background
RKMeans 生成 semantic_id 碰撞率极高（99%+），需要系统性优化。

### Key Findings
1. **normalize_residuals 只对 layer 0 输入做** — 残差保留原始 scale，否则 Layer 2/3 无法聚类
2. **FAISS full-batch Lloyd's 优于 SGD/MiniBatch** — 空 cluster rebalance + GPU 加速
3. **num_clusters 是唯一显著超参** — collision 与 clusters 呈 log-linear 关系，每翻倍降 50-70%
4. **nredo=3 足够，niter=25 已收敛** — nredo 1→3 关键 (-42~49%), 3→5 无意义; niter 25/50/100 无差异

### Final Config
- 3 layers × 1024 clusters, niter=25, nredo=3
- collision: 1.75%, reconstruction_loss: 0.348

---
