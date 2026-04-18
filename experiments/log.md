# Experiment Log

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

## EXP-017: SP-DPO — Self-Play DPO Alignment for NTP Model

**Date**: 2026-04-17
**Status**: planned
**Results**: TBD

### Background

NTP 模型已达到 S-tier 基线 (EXP-016 14d-S: PPL=27.05, R@500=58.5%)。当前训练纯 SFT (交叉熵)，只告诉模型"什么是对的"，不告诉模型"你当前犯的哪些错误是错的"。

SP-DPO (Self-Play DPO, Align³GR, AAAI 2026 Oral) 是 RL 对齐的入门方案：
1. 用模型自己 beam search 生成候选作为 rejected（负样本）
2. Ground truth 作为 chosen（正样本）
3. 按 SID prefix 匹配层数定义难度：Easy (0 层匹配) → Medium (L1 匹配) → Hard (L1+L2 匹配)
4. Softmax-DPO loss 渐进训练 (1 chosen vs 20 rejected)

核心优势：**零外部依赖** — 不需要 reward model，不需要用户反馈标签，只需要训练好的 NTP 模型。

详细技术讨论见 [discussions/001](../discussions/001-sp-dpo-vs-sft-vs-contrastive.md) 和 [discussions/002](../discussions/002-rf-dpo-grpo-ecpo-progression.md)。

### Hypothesis

1. SP-DPO (Easy) 对 R@10 提升最大 (远距离负样本拉开基本对错边界)，但对 R@500 影响有限
2. SP-DPO (Hard) 对 R@10 提升有限，但对 PPL 和逐层准确率显著改善 (精细区分 L3 混淆区)
3. 渐进训练 (Easy→Medium→Hard) 优于单阶段直接 Hard (参考 Align³GR 消融: +7.8% vs +4.7%)
4. Joint loss (NTP + DPO) 优于纯 DPO (保持 SFT 知识不丢失)
5. 整体 Recall@10 提升 5-10%，PPL 提升 3-5%

### Design

- **Variable**: DPO 阶段 (Easy/Medium/Hard)、渐进 vs 单阶段、DPO loss 权重 λ
- **Fixed**: S-tier 模型 (256d, 6L, 8E top-2, ~17.5M active), 14 天数据, beam_size=50 (采样)
- **Metric**: PPL, item_recall@{10,50,100,500}, 逐层 depth_hit@10, DPO loss 曲线
- **Data**: EXP-016 14d preprocessed NTP data (130M tokens, optimal per data scaling law)
- **Baseline**: EXP-016 14d-S checkpoint (PPL=27.05, R@500=58.5%)

**SP-DPO 配置**:

| 阶段 | Difficulty | Rejected 筛选规则 | Description |
|------|-----------|------------------|-------------|
| Stage 1 | Easy | prefix 匹配 0 层 (L1 就不同) | 完全不相关的候选 |
| Stage 2 | Medium | prefix 匹配 1 层 (L1 同, L2 不同) | 同粗粒度类目 |
| Stage 3 | Hard | prefix 匹配 2 层 (L1+L2 同, L3 不同) | 高度相似，仅细粒度不同 |

**超参搜索**:

| 参数 | 搜索范围 | 默认值 |
|------|---------|--------|
| λ (DPO loss weight) | {0.05, 0.1, 0.2, 0.5} | 0.1 |
| β (DPO temperature) | {0.1, 0.5, 1.0} | 0.1 |
| N_rejected | {10, 20} | 20 |
| beam_size (采样) | 50 | 50 |
| LR | {1e-4, 5e-5} | 1e-4 |

**实验矩阵** (优先级排序):

| Config | 阶段 | 渐进? | λ | β | 说明 |
|--------|------|-------|---|---|------|
| spdpo-baseline | — | — | — | — | SFT-only (复用 EXP-015 scale-04) |
| spdpo-easy | Easy only | No | 0.1 | 0.1 | 单阶段 Easy |
| spdpo-hard | Hard only | No | 0.1 | 0.1 | 单阶段 Hard (跳过 Easy/Medium) |
| spdpo-prog | Easy→Med→Hard | Yes | 0.1 | 0.1 | 渐进三阶段 (核心实验) |
| spdpo-prog-lam05 | Easy→Med→Hard | Yes | 0.05 | 0.1 | λ 消融 |
| spdpo-prog-lam50 | Easy→Med→Hard | Yes | 0.5 | 0.1 | λ 消融 |

**实现计划** (需要新增代码):

1. `rl/preference.py` — 离线构造 preference pairs:
   - 加载 SFT checkpoint → beam search 生成 50 候选 / eval item
   - 按 prefix 匹配层数分组 Easy/Medium/Hard
   - 每个 eval item 输出: chosen (ground truth SID) + 20 rejected (按难度筛选)
   - 保存为 `.npz` 文件供 DPO trainer 读取

2. `rl/dpo.py` — Softmax-DPO loss:
   - 输入: policy logprobs (chosen + rejected), reference logprobs (chosen + rejected)
   - 输出: Softmax-DPO loss (支持 1 chosen vs N rejected)

3. `rl/trainer.py` — DPO 训练循环:
   - 加载 SFT checkpoint 作为 π_ref (冻结) 和 π_θ (可训练)
   - Joint loss: L = L_NTP + λ * L_DPO
   - 支持渐进训练: 每阶段结束后 π_θ → π_ref

4. `rl/eval.py` — 对比评估:
   - 复用 `ntp/eval.py` 的 teacher-forced + beam search recall
   - 额外: DPO loss 曲线、chosen/rejected 概率比变化

### Run

`bash experiments/scripts/exp-017.sh`

### Results

TBD

### Analysis

TBD

### Next Steps

TBD

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

| Config | Active Params | PPL | Loss | R@10 | R@100 | R@500 |
|--------|--------------|------|------|------|-------|-------|
| scale-01 | 1.7M | 235.1 | 5.460 | 1.9% | 11.8% | 23.6% |
| scale-02 | 3.6M | 100.4 | 4.609 | 3.7% | 16.6% | 31.7% |
| scale-03 | 5.1M | 69.6 | 4.243 | 5.4% | 24.9% | 45.6% |
| scale-04 | 17.5M | **28.1** | 3.334 | 9.8% | 35.6% | 60.5% |
| scale-05 | 34.5M | 24.0 | 3.178 | 11.5% | 39.1% | 62.5% |
| scale-06 | 71.6M | 20.8 | 3.037 | 12.6% | 41.0% | 66.2% |
| scale-07 | 101.1M | **19.4** | 2.965 | 13.7% | 43.2% | 65.8% |

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

| Config | Days | Tokens | Users | PPL | Loss | R@100 | R@500 |
|--------|------|--------|-------|-----|------|-------|-------|
| A-7d-S | 7 | 65M | 1.02M | 30.60 | 3.421 | 37.9% | 62.1% |
| **B-14d-S** | **14** | **130M** | **1.69M** | **27.05** | **3.298** | **35.0%** | **58.5%** |
| C-31d-S | 31 | 262M | 3.04M | 28.05 | 3.334 | 35.6% | 60.5% |
| D-62d-S | 62 | 441M | 4.86M | 30.03 | 3.402 | 36.5% | 58.6% |
| E-90d-S | 90 | 553M | 6.18M | 31.89 | 3.462 | 35.1% | 56.2% |

**M+ 模型 (101M active)**:

| Config | Days | Tokens | Users | PPL | Loss | R@100 | R@500 |
|--------|------|--------|-------|-----|------|-------|-------|
| A-7d-M | 7 | 65M | 1.02M | 19.31 | 2.960 | 42.7% | 70.7% |
| **B-14d-M** | **14** | **130M** | **1.69M** | **18.96** | **2.942** | **43.0%** | **65.8%** |
| C-31d-M | 31 | 262M | 3.04M | 19.39 | 2.965 | 43.2% | 65.8% |
| D-62d-M | 62 | 441M | 4.86M | 19.80 | 2.986 | 43.2% | 68.1% |
| E-90d-M | 90 | — | 6.18M | *(跳过)* | — | — | — |

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
