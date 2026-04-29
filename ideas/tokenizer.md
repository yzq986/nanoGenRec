# Tokenizer (量化方法)

语义 ID 的核心：如何将高维 embedding 离散化为短序列 token。涵盖 RQ/OPQ/FSQ/Balanced KMeans 等量化方案，直接决定 collision rate、codebook utilization 和下游 NTP 模型的上限。

**影响范围**: `model/rkmeans.py`, `model/fsq.py`, `model/rkmeans_fsq.py`, `eval/evaluator.py`

---

## 当前结论 (2026-04-15)

**MLP-FSQ h=64 确认为 tokenizer 路线赢家，进入 NTP 阶段。**

### 当前 tokenizer config

```
架构: 2 层 KMeans (1024 clusters) + 1 层 MLP-FSQ (h=64, [4,4,4,4,4,4], codebook=4096)
SID:  3 tokens (L1_cluster, L2_cluster, L3_fsq_code)
Bits: 10 + 10 + 12 = 32 bits
Collision: 10.7%
semantic_neighbor_HR: 0.078
```

### 关键实验数据

| 实验 | 方案 | semantic_neighbor_HR | collision | 结论 |
|------|------|---------------------|-----------|------|
| EXP-008 A | **MLP-FSQ h=64 (3 token, 32 bit)** | **0.0780** | 0.1074 | **赢家** |
| EXP-008 B | OPQ 4×256 (4 token, 32 bit) | 0.0502 | 0.0351 | 等 bits 对照，输 36% |
| EXP-008 C | OPQ 8×256 (8 token, 64 bit) | 0.0326 | 0.0006 | collision 最低但行为最差 |

**核心 insight**: collision 越低 ≠ 行为质量越好。层级结构 (KMeans→KMeans→FSQ) 比扁平结构 (OPQ 并行子向量) 更好地保留 embedding 邻域，SID 前缀邻居的行为共现率更高。

**NTP 阶段补充 (2026-04-17)**: EXP-015 scaling law 拟合 irreducible loss a=2.522 (PPL≈12.5)，该 floor 由 tokenizer 32-bit 编码决定。M+ (101M) 已达 loss=2.94，距 floor 仅 0.42。**tokenizer 是当前系统瓶颈**——模型 scale up 收益递减，突破需要更高 bits SID 或更好的量化结构。

---

## 演进路径

```
RKMeans 3×1024 (EXP-001 baseline, collision=1.75%)
├── IDEA-sid-0: OPQ 并行语义 ID → EXP-004 → EXP-008 ❌ 行为质量不如 MLP-FSQ
│   └── collision 极低 (0.06%) 但 semantic_neighbor_HR 仅 0.033
├── IDEA-onemall-5: RKMeans + Learned FSQ → EXP-003 → EXP-008 ✅ 赢家
│   └── MLP-FSQ h=64: collision 10.7%, semantic_neighbor_HR 0.078 (最优)
├── IDEA-sid-1: 协同信号增强 embedding → EXP-007 + EXP-009 ❌ 死路
│   └── 全量FT/LoRA/QFormer 全部卡在 HR@50 ~0.02
├── IDEA-forge-0: SID Proxy Metrics → ✅ 已实现
│   └── semantic_neighbor_hit_rate 是决定性指标，EXP-008 靠它选出赢家
├── IDEA-sid-2: Balanced KMeans → P2 (NTP 后)
├── IDEA-gr4ad-0: MGMR 不等大码本 → P2 (NTP 后)
├── IDEA-pit-0: Co-generative 动态 Tokenizer → P2 (NTP 后)
├── IDEA-quasid-0: Hamming Repulsion → P2 (NTP 后)
├── IDEA-adasid-0: Adaptive Collision Regulation → P2 (NTP 后, extends quasid-0)
├── IDEA-dos-0: Dual-Flow Orthogonal RQ (上下文感知 SID + 正交量化) → P2 (NTP 后)
├── IDEA-r3vae-0: Reference Vector SID → P2 (NTP 后)
├── IDEA-geogr-0: 地理感知 SID (Co-visited Contrastive) → P2 (需泛化)
├── IDEA-onevision-0: VRQ 视觉对齐 RQ + 动态剪枝 → P2 (需视觉模态)
├── IDEA-mmq-0: 共享-专有多模态混合量化 → P2 (需多模态数据)
└── IDEA-rqgmm-0: GMM + 残差量化 (概率建模 → 更高码本利用率) → P2 (NTP 后)
```

---

## IDEA-sid-0: OPQ 并行语义 ID

**优先级**: ~~P0~~ → ❌ 关闭
**来源**: 3.1.2.2 (Meta RPG, KDD'25), Kaiming OPQ
**状态**: ~~已采纳 → EXP-004~~ → EXP-008 对比后关闭
**参考代码**: github.com/facebookresearch/RPG_KDD2025

> **关闭原因 (2026-04-15)**: EXP-008 等 bits 对比中，OPQ 4×256 (32 bit) semantic_neighbor_HR=0.050 输 MLP-FSQ (0.078) 36%；OPQ 8×256 (64 bit) collision 极低 (0.06%) 但 semantic_neighbor_HR 仅 0.033，更差。扁平子向量结构不如层级结构保留 embedding 邻域。Phase 2 (并行预测模型 + Graph Decoding) 不再推进。

### 核心思想

用 Optimized Product Quantization 替代残差编码，实现并行语义 ID。先学正交旋转矩阵 R，再将旋转后向量切分子向量独立量化。

### RPG 论文关键细节

**量化**: OPQ > RQ（消融实验 Table 3 证实）。每个 digit 的 codebook 是独立词表（C⁽¹⁾={1,...,M}, C⁽²⁾={M+1,...,2M}），不共享。

**模型架构**:
- Transformer encoder 对用户行为序列编码 → s ∈ ℝᵈ
- **每个 digit 一个独立 MLP projection head** g_j(s) → M 维 logits
- MTP loss = Σ CE_j（各 digit 条件独立）
- 单次 forward 出所有 token logits，非自回归
- 独立 head >> shared head（消融证实）

**推理 — Graph-Constrained Decoding（不是 beam search 也不是笛卡尔积）**:
- **beam search 在 OPQ 上完全失败**（Table 3 recall 全是 0.0000）
- **纯笛卡尔积组合也不行** — 256^32 空间太稀疏，大部分组合是无效 SID
- 构造 SID 相似度图: 节点=有效 SID, 边=embedding 相似度 top-k neighbors
- 推理: 随机采样 b=10 种子 → 沿图边扩展 → 打分 top-b → 迭代 q 轮
- 复杂度 O(Mmd + bqkm)，与 item 总数 N 无关
- 最终只访问 ~10-25% 的 item pool

**最优配置** (RPG 论文):
- m=16~64 (数据集越大越长), M=256, b=10, k=50~500, q=2~5
- τ=0.03 (温度), 2-layer Transformer, d=448, ~13M params
- 训练: <2 GPU hours (RTX 3090)

**关键消融结论**:
- OPQ > RQ (NDCG +2~8%)
- 长 ID (16~64) >> 短 ID (4)，且 TIGER (自回归+RQ) 在长 ID 下 OOM
- 独立 projection head >> shared head >> no head
- Graph decoding 比无图约束好 3x

### 与当前项目的关联

- `ARCHITECTURE.md` 已明确: "必须用平行 tokenizer，不能用残差编码"
- 直接解决 "残差编码永远不可能思考" 问题
- FAISS 有现成 `faiss.OPQMatrix` + `faiss.ProductQuantizer` 实现
- RPG 开源代码可直接参考

### 实验设计草案

**分两阶段实验**:

#### 阶段 1: OPQ 量化质量 (intrinsic metrics only)

验证 OPQ 在我们 5M item / 1024D embedding 上的量化质量。

**配置** (1024D embedding, 对标 RPG 论文主配置):

| 方案 | 子向量维度 | token 数 m | 词表大小 M | 编码空间 |
|------|-----------|-----------|-----------|----------|
| A | 128D | 8 | 256 | 256^8 |
| B | 64D | 16 | 256 | 256^16 |
| C | 32D | 32 | 256 | 256^32 |

**对比基线**: RKMeans 3 层 x 1024 clusters (EXP-001 final config, collision=1.75%)

**评估指标**: recon_loss, collision_rate, exclusivity, entropy, cluster_balance

**注意**: collision_rate 在 OPQ 下可能意义不同 — 8~32 个 token 的 ID 空间远大于 3 token，collision 应该极低。重点看 recon_loss 是否优于 RKMeans。

#### 阶段 2: 并行预测模型 + Graph Decoding

需要新实现:
1. 并行预测模型: 替换当前 AutoregressiveNTPModel，每 digit 独立 MLP head
2. Graph 构造: SID 相似度图 (top-k neighbors per node)
3. Graph decoding: 种子采样 → 图传播 → 打分 → 迭代

### 关键问题

1. **词表大小 M 的选择**: RPG 固定 M=256。我们的 RKMeans 用 1024 per layer。需要验证 M=256 vs M=1024 在 OPQ 下的差异。
2. **Graph 构造成本**: 5M items 的 SID 相似度图构造需要 O(N²) 或近似 ANN，需要评估内存和时间。
3. **与 OneRec 的关系**: OneRec 用 3 token x 8192 parallel tokenizer，RPG 用 16~64 token x 256。两种 parallel 方案的 tradeoff 需要理清 — OneRec 适合自回归 (短序列)，RPG 适合并行预测 (长序列+图解码)。

---

## IDEA-fsq-scale-0: FSQ Hidden 自适应 — 匹配 Embedding 维度

**优先级**: P1 — EXP-043 发现的直接 actionable 改进
**状态**: 待实验，计划为 EXP-045 或独立 tokenizer 实验

### 背景与根因

EXP-043 熵分析揭示：FSQ MLP hidden=64 是为 Qwen3-0.6B (1024D) embedding 设计的。当 embedding 维度增大时，残差向量维度同步增大，但 FSQ bottleneck 不变，导致 L2 层信息严重损失：

| Embedding | Residual Dim | FSQ hidden | L2 entropy | FSQ 有效槽位 | Collision |
|-----------|-------------|-----------|-----------|------------|---------|
| Qwen3-0.6B | ~1024D | 64 | 10.58 bits (91.2%) | ~1500 | **0.49%** |
| Qwen3-4B | ~2560D | 64 | 8.10 bits (78.7%) | ~275 | 2.76% |
| Qwen3-8B | ~4096D | 64 | 7.17 bits (71.6%) | ~145 | 5.44% |

通过 S-tier + M-tier 两点 scaling law 反推，各 SID 的 irreducible floor PPL：
- 0.6B: 12.46，4B: **11.78（最优）**，8B: 12.26（差于 4B，L2 坍缩所致）

### 核心假设

FSQ hidden 应与 embedding 维度成比例扩大，使 L2 entropy 保持在 ≥90% 利用率：
```
h_optimal ≈ k × emb_dim^α   （经验公式，待拟合）
```
初始猜测：0.6B(1024D)→h=64，4B(2560D)→h=128~160，8B(4096D)→h=256

### 实验设计

**变量**：FSQ hidden h ∈ {64, 128, 256, 512} × embedding model ∈ {0.6B, 4B, 8B}

但全量 3×4=12 组成本太高（每组需重建 SID cache + 重训 NTP）。**建议经济方案**：

**Phase 1 — 纯 tokenizer 评测（无需 NTP，成本极低）**：
- 固定 0.6B SID，sweep h ∈ {32, 64, 128, 256}：确认 h=64 已是 0.6B 的最优点
- 固定 4B SID，sweep h ∈ {64, 128, 256}：找到 L2 entropy ≥90% 的最小 h
- 固定 8B SID，sweep h ∈ {64, 128, 256, 512}：同上
- 评测指标：L2 entropy（bits + 利用率）、collision rate、FSQ 有效槽位数

**Phase 2 — NTP 端到端验证（仅跑 Phase 1 找到的最优 h）**：
- 4B SID with h=最优 → S-tier NTP → 与 exp043-s-4b (R@500=64.3%) 对比
- 8B SID with h=最优 → S-tier NTP → 与 exp043-s-8b (R@500=64.7%) 对比
- 目标：验证 L2 entropy 恢复后 floor PPL 是否真正下降，R@500 是否超越 4B 当前最优

### 经验公式目标

通过 Phase 1 数据，拟合：
```
h_min_for_L2_util_90% = f(emb_dim)
```
若线性：`h ≈ emb_dim / 16`（0.6B: 64, 4B: 160, 8B: 256）
若根号：`h ≈ 2 × sqrt(emb_dim)`（0.6B: 64, 4B: 101, 8B: 128）

最终给出一个跨 embedding 规模通用的 h 选取公式，避免逐个调参。

### 改动文件

- `model/rkmeans.py` / `model/fsq.py` — `fsq_mlp_hidden` 参数已支持，只需改 config 值
- `experiments/scripts/exp-026-sid.sh`（或新建 `exp-045-fsq-scale.sh`）

---

## IDEA-sid-2: Balanced KMeans

**优先级**: ~~P1~~ → P2 (NTP 后)
**来源**: 3.1.2.1 (OneRec Paper 提到)
**状态**: 待定，降级

> **降级原因 (2026-04-15)**: 前两层 KMeans Gini=0.31，balanced assignment 可改善码本利用率，collision 可能从 10.7% 降到 ~7-8%。但 EXP-008 证明 collision 不是核心指标（OPQ collision 0.06% 反而行为最差），收益不确定。等 NTP 端到端 Recall@K 出来后再决定是否投入。
>
> **NTP 阶段更新 (2026-04-17)**: EXP-015 scaling law 显示 irreducible loss a=2.522 由 tokenizer 32-bit 编码决定。提升码本利用率 (Balanced KMeans) 可能微降 collision 但无法突破 bit 数瓶颈。长期价值在突破 32-bit 上限后 (更多 token/更大码本) 再评估。

### 核心思想

用平衡 KMeans 替代标准 KMeans，强制各 cluster 大小均匀，提高码本利用率。

### 与当前项目的关联

- EXP-001 中 cluster_balance (Gini) 仍有优化空间
- 实现成本极低，可快速验证
- 文中还提到 "原向量和残差向量都可以做归一化" — 当前只 normalize layer 0 input

### 实验设计草案

**变量**:
- 标准 KMeans vs Balanced KMeans
- 残差归一化: 仅 L0 (当前) vs 每层都 normalize

**注意**: EXP-001 结论 "normalize_residuals 只对 layer 0" 可能需要重新验证，因为当时可能没用 balanced assignment

### 关键问题

1. FAISS 不直接支持 balanced KMeans，需要用 `faiss-contrib` 或自实现
2. 每层 normalize 残差是否会与 EXP-001 结论冲突

---

## IDEA-gr4ad-0: Multi-Granularity Multi-Resolution RQ-KMeans (MGMR)

**优先级**: ~~P1~~ → P2 (NTP 后)
**来源**: GR4AD §UA-SID, Table 2
**状态**: 待定，降级

> **降级原因 (2026-04-15)**: MLP-FSQ 已确认为赢家，不等大码本 (L1=4096 L2=1024) 是对前两层 KMeans 的微调，收益中等。实现成本极低（`ResidualQuantizationMultiGPU` 已支持每层独立 `n_clusters`），但优先推进 NTP。

### 核心思想

GR4AD 提出 MGMR 编码方案：(1) Multi-Resolution — 低层用大码本捕获主导因子，高层用小码本建模低熵残差（如 16384→4096→1024）；(2) Multi-Granularity — 最后一层用非语义特征（item ID、账户 ID）的 hash 映射替代聚类，直接消除碰撞。两者结合使碰撞率从 3.54 降至 1.07，码本利用率从 0.10‰ 提升至 0.34‰。

### 与当前项目的关联

- **直接对标 EXP-001**: 当前用等大码本 3×1024，collision=1.75%。MGMR 的不等大码本（如 4096→1024→256）是零成本改进
- `model/rkmeans.py` 的 `ResidualQuantizationMultiGPU` 已支持每层独立 `n_clusters` 参数，只需传入不同值
- Multi-Granularity 的 hash 层思想与 IDEA-sid-0 (OPQ) 互补 — 如果最后一层用 hash 保证唯一性，前面层可以放心用更粗的语义聚类
- `eval/evaluator.py` 的 collision_rate、codebook utilization 指标可直接用于评估

### 实验设计草案

**变量 1 — Multi-Resolution 码本配置**:

| 配置 | L1 | L2 | L3 | 总编码空间 |
|------|------|------|------|-----------|
| Baseline (EXP-001) | 1024 | 1024 | 1024 | 10^9 |
| MR-A | 4096 | 1024 | 256 | 10^9 |
| MR-B | 2048 | 1024 | 512 | 10^9 |
| MR-C | 4096 | 2048 | 512 | 4×10^9 |

**变量 2 — Multi-Granularity hash 层**:
- 对每个 MR 配置，测试最后一层替换为 hash(item_id) % vocab_size
- 需要在 `model/rkmeans.py` 增加 hash 层选项

**评估**: collision_rate, recon_loss, codebook_utilization, cluster_balance (Gini), sid_prediction Hit@K

**实现成本**: 低。MR 只需修改 config 参数；MG hash 层需在 `ResidualQuantizationMultiGPU` 中增加约 20 行代码

### 关键问题

1. 不等大码本下 NTP 模型的 `vocab_size` 需要每层不同 — `metrics/sid_prediction.py` 的 `AutoregressiveNTPModel` 目前假设统一 vocab_size，需适配
2. Hash 层的 vocab_size 选择: 与 item 数量的关系？太小仍有碰撞，太大稀疏
3. 与 IDEA-sid-2 (Balanced KMeans) 的交互: 大码本 (4096+) 下 balanced assignment 更关键

---

## IDEA-onemall-5: OneMall 验证 EXP-003 方向 (ResKmeans + Learned FSQ)

**优先级**: ~~P1~~ → ✅ 完成
**来源**: OneMall §3.1.3 + §4.5 Tokenizer Strategy
**状态**: ✅ MLP-FSQ h=64 确认为赢家 (EXP-003 → EXP-008)

> **完成记录 (2026-04-15)**: EXP-003 验证了 MLP-FSQ 方案可行，EXP-008 通过 FORGE proxy metrics 在等 bits 条件下与 OPQ 对比，MLP-FSQ h=64 的 semantic_neighbor_HR=0.078 决定性胜出。此方案成为当前 tokenizer baseline。

### 核心思想

OneMall 的 tokenizer ablation 直接验证了我们的 EXP-003 方向:

| 方案 | Conflict Rate | Exclusive Rate | HR@50 |
|------|--------------|----------------|-------|
| 3-layer ResKmeans | 36% | 86% | 33.9% |
| 2-layer ResKmeans + 1-layer FSQ | **11%** | **95%** | **35.4%** |

关键差异: OneMall 的 FSQ 层使用 **"binary 16-bit MLP"** 量化残差 embedding 为 4096 code — 这正是我们 `LearnedFSQLayer` (MLP + STE) 的方案，而非 EXP-002 失败的 PCA 方案。

EXP-002 失败原因 (PCA 1024D→4-6D 仅保留 20-55% variance) 在 OneMall 中被隐式验证: 他们直接用 MLP 而非 PCA。

### 与当前项目的关联

- `LearnedFSQLayer` 已实现 (`model/fsq.py`)
- `ResKmeansFSQ` 已支持 `mlp` projection type (`model/rkmeans_fsq.py`)
- EXP-003 已设计但 **尚未运行**
- OneMall 结果给出了明确预期: conflict rate 应从当前 ~1.75% 进一步降低，exclusive rate 应提升

### 行动建议

**立即执行 EXP-003**，参考 OneMall 参数:
- FSQ codebook size = 4096 (与 OneMall 一致)
- MLP hidden sizes: {64, 128, 256} (已在 EXP-003 设计中)
- 训练 50 epochs with STE
- 特别关注 conflict rate 和 exclusive rate 的变化

### 关键问题

1. OneMall 的 "binary 16-bit" 具体实现细节不明 — 是否就是 16 个 binary bit 直接做 2^16=65536 codes 然后截断到 4096？还是 FSQ 风格的 multi-level quantization？
2. 我们的 FSQ level config `4d_4096: [8,8,8,8]` 产生 4096 codes，与 OneMall 一致

---

## IDEA-pit-0: Co-generative 动态 Tokenizer (PIT)

**优先级**: ~~P1~~ → P2 (NTP 后)
**来源**: PIT (Kuaishou, arxiv 2602.08530, Feb 2026)
**状态**: 待定，降级

> **降级原因 (2026-04-15)**: PIT 的核心是 tokenizer+NTP 联合训练，前置条件是先有 NTP baseline。MLP-FSQ 当前是纯无监督（重建 loss），加入行为信号联合训练可能进一步提升，但复杂度高。等 NTP baseline 出来后再评估是否值得投入。

### 核心思想

PIT 提出 **Co-generative Architecture**: tokenizer 和 NTP 模型不再分阶段训练，而是通过 **协同信号对齐 (Collaborative Signal Alignment)** 和 **共进化学习 (Co-evolution Learning)** 实现端到端联合训练。核心创新:

1. **Collaborative Signal Alignment**: 将协同过滤信号直接注入 tokenization 过程，使生成的 SID 自带行为语义
2. **Co-evolution Learning**: tokenizer 和 recommender 在统一训练循环中互相增强，避免 "先建索引再训练" 的两阶段断裂
3. **One-to-Many Beam Index**: 每个 item 可分配多个 SID token 序列，提升 recall 和鲁棒性

快手大规模在线 A/B: **App Stay Time +0.402%**。

### 与当前项目的关联

- 当前 tokenizer (RKMeans/OPQ) 是完全 offline、与 NTP 模型解耦的
- Co-evolution 的核心问题: 每次 tokenizer 更新，所有 SID 都变了，NTP 模型要重新学习 — PIT 声称解决了这个稳定性问题
- **One-to-Many Beam Index** 对我们有直接启发: 一个 item 映射到多个 SID 可以缓解 collision 问题
- 与 IDEA-sid-1 (协同信号增强) 有关联: 两者都注入协同信号，但 PIT 是在 tokenizer 训练中动态注入，sid-1 是预训练 embedding 后注入

### 实验设计草案

**Phase 1 — One-to-Many SID 映射**:
- 当前每个 item 只有一个 SID。允许 OPQ/RKMeans 给每个 item 分配 top-k (k=2~5) 最近的码字组合
- NTP 训练时，target 是 k 个 SID 中任一即算正确 (multi-label CE)
- 评估 recall 提升 + collision 缓解

**Phase 2 — Co-evolution** (高复杂度):
- 在 NTP 训练过程中定期 (每 N epoch) 重新运行 tokenizer
- 用 NTP 模型的隐层表示作为 tokenizer 输入信号之一
- 需要设计稳定性机制避免 SID 剧烈变化

### 关键问题

1. One-to-Many 映射增加了 NTP 训练的歧义性 — 一个 item 有多个"正确答案"，模型如何收敛?
2. Co-evolution 的计算成本: 每 N epoch 重新 tokenize 5M items 的 overhead
3. SID 变化的稳定性: 如何保证新旧 SID 之间的连续性

---

## IDEA-forge-0: SID Proxy Evaluation Metrics + Offline Pretraining

**优先级**: ~~P1~~ → ✅ 完成
**来源**: FORGE (Alibaba/Taobao, arxiv 2509.20904, Sep 2025)
**状态**: ✅ semantic_neighbor_hit_rate 已实现并验证为决定性指标

> **完成记录 (2026-04-15)**: `eval/evaluator.py` 已实现 semantic_neighbor_hit_rate。EXP-008 靠此指标选出 MLP-FSQ 赢家，证明 proxy metric 无需训练 NTP 即可有效评估 SID 质量。Offline Pretraining 部分留待 NTP 阶段。

### 核心思想

FORGE 发布了 Taobao 14B 交互 + 250M 商品的大规模 benchmark，并提出两个关键技术:

1. **SID Proxy Metrics**: 两个新指标与下游推荐性能正相关，**无需训练 GR 模型即可评估 SID 质量**。这解决了"每次改 tokenizer 都要跑完整 NTP 训练才知道好不好"的痛点
2. **Offline Pretraining Schema**: 用离线预训练将在线收敛时间减半

Taobao "猜你喜欢" 在线验证: **交易量 +0.35%**。

### 与当前项目的关联

- 当前评估 SID 质量需要: (1) 跑 RKMeans/OPQ → (2) 训练 NTP → (3) 看 Hit@K。整个流程耗时数小时
- 如果有 proxy metrics，可以在 step (1) 之后直接评估，**加速 tokenizer 超参搜索 10x+**
- 与 EXP-004 (OPQ) 直接相关: 快速评估不同 m/M 配置的 SID 质量
- `eval/evaluator.py` 已有 collision_rate、exclusivity 等 intrinsic metrics，可以在此基础上加入 FORGE 的 proxy metrics

### 实验设计草案

- 获取 FORGE 论文定义的 proxy metrics (需要读论文全文)
- 在 `eval/evaluator.py` 中实现
- 验证: proxy metrics 是否与 NTP Hit@K 相关 (在已有 EXP-001/004 数据上回测)

### 关键问题

1. FORGE 的 proxy metrics 具体定义需要读论文全文才能获取
2. 我们的数据规模 (5M items) 远小于 FORGE (250M items)，proxy metrics 的相关性是否仍然成立

---

## IDEA-quasid-0: Collision-Qualified SID Learning (Hamming-Guided Repulsion)

**优先级**: ~~P1~~ → P2 (NTP 后)
**来源**: QuaSID (Kuaishou E-commerce, arxiv 2603.00632, Feb 2026)
**状态**: 待定，降级

> **降级原因 (2026-04-15)**: MLP-FSQ collision 10.7%，Hamming repulsion 可能降低有害碰撞。但 EXP-008 证明低 collision 不等于高行为质量（OPQ collision 0.06% 反而最差），需要先确认有害碰撞占比。前置: Phase 1 分析——碰撞 pair 的语义距离分布。等 NTP 端到端数据出来再决定。
>
> **NTP 阶段更新 (2026-04-17)**: EXP-014 ENTP 负样本导出发现 L0 层碰撞问题——部分负样本与正样本共享 L1 cluster token，导致 ENTP loss 在 coarse level 失效。这验证了 QuaSID 的核心 premise (有害碰撞存在且影响训练信号)。但优先级仍为 P2: 先推进 ENTP loss 集成，再决定是否需要 Hamming repulsion 从 tokenizer 端解决。

### 核心思想

QuaSID 发现 SID collision 问题并非同质的: 有些碰撞是真正有害的"语义冲突" (semantically unrelated items get identical SIDs)，有些是良性的 (data redundancy)。QuaSID 提出两个机制:

1. **Hamming-guided Margin Repulsion**: 用 SID 间 Hamming distance 作为碰撞严重度指标，将低 Hamming 距离的冲突 item pair 在 encoder 空间推开。推力与碰撞严重度成正比
2. **Conflict-Aware Valid Pair Masking**: 自动过滤"良性碰撞"(protocol-induced benign overlaps)，只对真正有害的碰撞施加 repulsion

额外: 加入 **dual-tower contrastive objective** 在 tokenization 中注入协同信号。

**Plug-and-play**: repulsion loss 可以增强任何 SID 学习框架。

快手电商在线 A/B (5% traffic): **ranking GMV-S2 +2.38%, 冷启动订单 +6.42%**。

### 与当前项目的关联

- 当前 EXP-001 的 collision_rate = 1.75%，我们已经在 `eval/evaluator.py` 中追踪碰撞
- QuaSID 的 insight 是: **不是所有碰撞都应该被同等对待**。当前我们只看 collision count，不区分碰撞的严重程度
- Hamming distance 计算零成本 — SID 已经是离散码，直接比较
- **Plug-and-play**: 可以直接加到 EXP-007 (contrastive embedding fine-tune) 的训练中
- 与 IDEA-sid-1 (协同信号增强) 互补: sid-1 改善 embedding 本身，QuaSID 在 SID 空间施加碰撞约束

### 实验设计草案

**Phase 1 — Collision 严重度分析**:
- 对现有 SID assignment (EXP-001/EXP-004)，计算所有碰撞 pair 的语义距离 (embedding cosine)
- 分类: 高 cosine = 良性碰撞 (语义本来就近), 低 cosine = 有害碰撞
- 量化: 有害碰撞占比多少? 如果很低 → 收益有限

**Phase 2 — Hamming Repulsion Loss**:
- 在 embedding fine-tune (EXP-007 流程) 中加入 repulsion loss
- 对 Hamming distance < threshold 的 item pair 施加 margin loss
- L_repulsion = max(0, margin - cosine(e_i, e_j)) * severity_weight(hamming_dist)

**评估**: collision_rate, 有害碰撞比例, embedding_hit_rate

### 关键问题

1. 在 RKMeans (3 层 x 1024) 下碰撞率已经只有 1.75%，repulsion 收益可能有限
2. OPQ (8~32 token x 256) 碰撞率更低 → 此 idea 可能对 RKMeans 路线更有价值
3. Repulsion 和 contrastive loss 的梯度冲突: 一个要推开，一个要拉近

> **后续工作 (2026-04-28)**: 同一快手团队发表 AdaSID (arxiv 2604.23522)，将固定 Hamming 阈值升级为两阶段自适应调控 (语义门控 + 负载自适应 + 进度调度)，在 Toys/Beauty 全面超越 QuaSID。详见 IDEA-adasid-0。

---

## IDEA-r3vae-0: Reference Vector-Guided SID Generation (稳定训练 + 评估指标)

**优先级**: ~~P1~~ → P2 (NTP 后)
**来源**: R3-VAE (arxiv 2604.11440, Apr 2026)
**状态**: 待定，降级

> **降级原因 (2026-04-15)**: MLP-FSQ 没有 codebook collapse 问题，Reference Vector 的训练稳定性价值有限。Semantic Cohesion + Preference Discrimination 指标可补充 FORGE proxy，但主要价值在评估而非改进。等 NTP 后再考虑。

### 核心思想

R3-VAE 解决 VQ-based SID 生成的两个根本问题:

1. **训练不稳定**: STE (straight-through estimator) 梯度传播不足 + 初始化敏感 → **Reference Vector** 作为语义锚点稳定训练
2. **评估代价高**: 评估 SID 质量需要训练完整 GR 模型 + A/B test → 提出 **Semantic Cohesion** 和 **Preference Discrimination** 两个 standalone metrics，可在 SID 生成后直接评估

Reference vector + dot-product rating 机制还能防止 **codebook collapse** (死码本问题)。

新闻推荐平台在线 A/B: **MRR +1.62%**。作为 CTR 模型 item ID 替代: **冷启动 +15.36%**。

### 与当前项目的关联

- 当前 RKMeans 训练没有 codebook collapse 问题 (每层都用 KMeans)，但切到 VQ-VAE 或 learned quantization 时会遇到
- **Semantic Cohesion + Preference Discrimination 指标** 与 IDEA-forge-0 (SID Proxy Metrics) 异曲同工 — 都是无需训练 NTP 即可评估 SID 质量
- 如果整合两者，可以建立 **完整的 SID 质量评估工具包**: 训练前 (FORGE proxy) + 训练后 (R3-VAE metrics)
- 冷启动 +15.36% 的结果对我们有启发: SID 可以作为 CTR 模型的 item feature

### 实验设计草案

**Phase 1 — 实现 R3-VAE 评估指标**:
- 在 `eval/evaluator.py` 中实现 Semantic Cohesion 和 Preference Discrimination
- 在已有 EXP-001/EXP-004 SID assignments 上回测
- 验证: 这两个指标是否与 NTP recall 相关

### 关键问题

1. 具体指标定义需要读论文全文
2. 与 IDEA-forge-0 的 proxy metrics 去重/合并

---

## IDEA-unirec-1: Capacity-Constrained SID (Exposure-Weighted RQ Penalties)

**优先级**: P2 (NTP 后)
**来源**: UniRec (Alibaba, arxiv 2025, KDD 2025)
**状态**: 待定，降级

> **降级原因 (2026-04-15)**: MLP-FSQ 已确认为 tokenizer 赢家。Capacity constraint 主要解决 token collapse（少数 codebook entry 垄断大量 item），与 IDEA-sid-2 (Balanced KMeans) 目标一致，可合并评估。等 NTP 端到端 Recall@K 出来后再决定。

### 核心思想

UniRec 在 RQ tokenizer 训练中发现严重的 **token collapse** 问题：部分 codebook entry 被过度分配（高曝光 item 主导聚类中心），导致长尾 item 的 SID 表示质量差。提出 **Capacity-Constrained SID Learning**:

1. **Exposure-Weighted Assignment Penalty**: 对已分配大量高曝光 item 的 codebook entry 施加惩罚，迫使 tokenizer 更均匀地使用码本
2. **Residual Capacity Tracking**: 每个 codebook entry 维护一个 capacity 计数器，超过阈值后 assignment cost 线性增加
3. **两阶段训练**: 先标准 RQ 训练收敛，再加 capacity constraint fine-tune

效果: 码本利用率从 ~60% 提升到 ~95%，长尾 item 的 SID 区分度显著改善。

### 与当前项目的关联

- 当前 MLP-FSQ 的前两层 KMeans Gini=0.31，存在码本不均匀问题
- Capacity constraint 与 Balanced KMeans (IDEA-sid-2) 目标一致但方法不同：sid-2 在聚类时强制均衡，unirec-1 在 loss 中施加软约束
- 实现成本低：在 RQ 训练的 assignment 步骤中加 penalty term
- 与 IDEA-quasid-0 (Hamming Repulsion) 互补：quasid-0 处理有害碰撞，unirec-1 处理码本利用率

### 实验设计草案

**与 IDEA-sid-2 合并评估**:
- Balanced KMeans (硬约束) vs Capacity Penalty (软约束) vs 两者结合
- 评估: codebook utilization (Gini), collision_rate, semantic_neighbor_HR
- 在 MLP-FSQ 架构上验证：对前两层 KMeans 施加 capacity constraint

### 关键问题

1. MLP-FSQ 的第三层 FSQ 天然均匀分布，constraint 主要针对前两层 KMeans
2. 与 Balanced KMeans 的冗余：两者都解决码本利用率，可能只需选一个
3. Exposure weighting 需要曝光数据，当前 item metadata 是否包含曝光量

---

## 优先级总结

| 优先级 | ID | 方向 | 状态 |
|--------|-----|------|------|
| ~~P0~~ | ~~IDEA-sid-0~~ | ~~OPQ 并行语义 ID~~ | ❌ 关闭 (EXP-008: semantic_neighbor_HR 输 MLP-FSQ) |
| ~~P1~~ | ~~IDEA-onemall-5~~ | ~~RKMeans+FSQ~~ | ✅ 完成，MLP-FSQ h=64 确认赢家 |
| ~~P1~~ | ~~IDEA-forge-0~~ | ~~SID Proxy Metrics~~ | ✅ 完成，semantic_neighbor_hit_rate 已实现 |
| P2 | IDEA-sid-2 | Balanced KMeans | 待定，NTP 后 (collision 非核心指标) |
| P2 | IDEA-gr4ad-0 | MGMR 不等大码本 | 待定，NTP 后 (微调收益，优先推 NTP) |
| P2 | IDEA-quasid-0 | Hamming Repulsion | 待定，NTP 后 (需先确认有害碰撞占比) |
| P2 | IDEA-pit-0 | Co-gen Tokenizer | 待定，NTP 后 (前置: NTP baseline) |
| P2 | IDEA-r3vae-0 | Reference Vector SID | 待定，NTP 后 (主要价值在评估指标) |
| P2 | IDEA-unirec-1 | Capacity-Constrained SID | 待定，NTP 后 (与 sid-2 合并评估) |
| P2 | IDEA-flexcode-0 | 双码本 CF+Semantic + MoE 分配 | 待定，NTP 后 (需 CF model) |
| P2 | IDEA-crab-0 | Codebook Rebalancing 去偏 | 待定，NTP 后 (post-hoc 方法) |

---

## IDEA-geogr-0: 地理感知 SID Tokenization (Co-visited POI 对比学习)

**优先级**: P2
**来源**: GeoGR, Alibaba/AMAP (arxiv 2602.10411)
**状态**: 待讨论

### 核心思想

阿里高德地图的 GeoGR 针对 POI 推荐提出 geo-aware SID tokenization: 用地理约束的 co-visited POI pairs 做对比学习，加上迭代 refinement，生成捕获时空协同语义的 SID。关键 insight: POI 的语义不仅取决于内容 (餐厅/商店)，还取决于地理位置和时间模式 (午餐时段的附近餐厅 vs 周末的远郊景点)。配合多阶段 LLM 训练 (template-based CPT + autoregressive SFT) 实现端到端 POI 生成。在高德部署，服务数百万用户。

### 与当前项目的关联

- 与 IDEA-oneloc-3 (side-info 融合) 理念一致但更具体: 不是泛化 side-info，而是专门针对时空信号
- Co-visited POI 对比学习可以泛化为 "co-consumed item contrastive learning": 用共同消费的 item pairs 强化 SID 的行为语义
- 当前 MLP-FSQ tokenizer 基于 text embedding，缺乏行为协同信号 → co-consumed contrastive 可能弥补这一 gap
- Multi-stage LLM training (CPT + SFT) 与 IDEA-plum-0 一致

### 实验设计草案

**Phase 1 — Co-consumed Item Contrastive Loss for Tokenizer**:
- 在 tokenizer 训练 (或 embedding fine-tune) 中加入: 经常被同一用户消费的 item pair 的 SID 应该共享更多前缀
- 实现: 在 RQ/FSQ 训练的 assignment 步骤中加入 co-visit affinity penalty
- 评估: semantic_neighbor_HR (本身就测量行为邻域保留)

### 关键问题

1. 当前数据无地理信息，需要泛化为 co-consumption 信号
2. 与 IDEA-sid-1 (协同信号增强 embedding) 部分重叠，但 sid-1 是直接 fine-tune embedding，本 IDEA 是在 tokenizer 层注入
3. NTP 后阶段再考虑 tokenizer 改进 → P2

---

## IDEA-onevision-0: 视觉对齐残差量化 (VRQ) + 动态剪枝

**优先级**: P2
**来源**: OneVision, Kuaishou (arxiv 2510.05759)
**状态**: 待讨论

### 核心思想

快手 OneVision 针对视觉搜索 (visual search) 提出 VRQ (Vision-aligned Residual Quantization): 跨多视角对齐同一物体的差异巨大的视觉表征，同时保留产品独特特征，生成用于生成式检索的 semantic ID。配合多阶段语义对齐 (保留视觉相似先验 + 融入用户个性化偏好) 和动态剪枝 (推理效率提升 21%)。在线 A/B: CTR +2.15%, CVR +2.27%, 订单量 +3.12%。

### 与当前项目的关联

- VRQ 的多视角对齐思路可以泛化到多模态: 同一 item 的文本描述、标题、评论可能语义差异大，需要对齐后再量化
- 动态剪枝 (21% 效率提升) 在推理端有直接价值: 根据输入难度动态调整 SID 序列长度
- 当前项目是文本 embedding → SID，OneVision 是视觉 embedding → SID，核心 pipeline 相同
- 在线 A/B 效果显著 (CTR +2.15%)，验证了端到端生成式搜索架构的可行性

### 实验设计草案

需要视觉模态数据，当前项目暂不适用。但动态剪枝思路 (IDEA-stamp-0 也涉及) 可以共同参考。

### 关键问题

1. 当前项目无视觉模态 → VRQ 本身不直接可用
2. 动态剪枝的思路已被 IDEA-stamp-0 覆盖
3. 主要价值在于验证 "生成式搜索端到端架构" 的在线效果

---

## IDEA-mmq-0: 共享-专有多模态混合量化 Tokenizer

**优先级**: P2
**来源**: MMQ, Alibaba (arxiv 2508.15281, WSDM 2026)
**状态**: 待讨论

### 核心思想

阿里 MMQ 提出两阶段多模态 tokenizer: (1) Shared-Specific Tokenizer — multi-expert 架构，modality-specific experts 捕获各模态独特信息，modality-shared experts 捕获跨模态共性，加正交正则化; (2) Behavior-Aware Fine-Tuning — 用下游推荐目标动态适配 SID 表征，同时用多模态重建 loss 保持模态信息不丢失。支持 generative retrieval 和 discriminative ranking 两种下游任务。WSDM 2026 + 在线 A/B 验证。

### 与当前项目的关联

- 当前 tokenizer 只用 text embedding (Qwen3)，MMQ 的多模态框架提供了扩展路线
- Shared-Specific Expert 架构可以泛化: 将 "模态" 替换为 "信号类型" (semantic signal vs collaborative signal)
- Behavior-Aware Fine-Tuning 与 IDEA-onemall-3 (属性增强 contrastive) 思路一致: 用下游任务信号反过来调整 tokenizer
- 正交正则化防止 expert 退化，对 MoE 相关 IDEA (IDEA-onemall-4) 有参考价值

### 实验设计草案

**Applicable when multimodal data is available:**
- Shared expert: 学跨模态共性 (text + image 共同描述 item 语义)
- Specific expert: 学单模态独特信息 (text 的属性描述 vs image 的视觉风格)
- Behavior-aware fine-tune: 在冻结 expert 后用 NTP recall 目标微调量化层

### 关键问题

1. 当前无多模态数据 → 无法直接实验
2. Shared-Specific 思路可在单模态下测试 (semantic vs collaborative 双 expert)，但价值未验证
3. NTP 后阶段再考虑 tokenizer 扩展 → P2

---

## IDEA-flexcode-0: 双码本 (CF + Semantic) + MoE 动态分配

**优先级**: P2 (NTP 后)
**来源**: FlexCode, Roblox (arxiv 2511.20673, Nov 2025)
**状态**: 待讨论

### 核心思想

FlexCode 发现单一码本同时编码语义和协同信号导致"表示纠缠"——head item 被语义稀释，tail item 被噪声协同信号主导。提出 **双码本 + 自适应分配**:

1. **Semantic Codebook (C_SEM)**: RQ-VAE 量化 text/visual embedding，捕获内容语义
2. **Collaborative Codebook (C_CF)**: SASRec-style 对 co-purchase/co-view 序列学习协同 embedding，再 RQ-VAE 量化
3. **Cross-Codebook Alignment (CCA)**: InfoNCE 对比 loss 对齐两个码本的重建 embedding，防止空间漂移
4. **MoE Router**: 基于 item 统计特征 (log(popularity), age, sparsity, uncertainty) 路由，head items → 更多 CF tokens, tail items → 更多 semantic tokens
5. **固定总 token 预算 L**: L_CF(i) + L_SEM(i) = L，通过 sigmoid mask 实现可微分配

**核心结果**:
- KuaiRand: NDCG@10 0.0632, 比 URL +8.0%, 比 TIGER +42%
- Industrial (1.5M+ users): NDCG@10 +13.2% over SASRec baseline
- Tail items NDCG@10 +11.3% (最大提升)，Head +3.0%
- FlexCode-Fix (50/50 静态) 已经 beats baselines → 双码本本身有价值
- MoE 动态分配额外贡献 12.5% (KuaiRand)

### 与当前项目的关联

- **直接回应用户关切**: NTP 缺乏 cross-user collaborative signal → FlexCode 在 tokenizer 层注入 CF
- 当前 MLP-FSQ tokenizer 只用 text embedding (纯语义)，与 FlexCode 的 "SID Only" 对应
- 双码本方案不改 NTP 模型架构——只改 SID 的生成方式，NTP 仍然做 token 预测
- 与 IDEA-sid-1 (协同信号增强 embedding) 目标一致但方案更系统: sid-1 是直接 fine-tune text embedding，FlexCode 是独立码本 + 动态融合
- 与 IDEA-pit-0 (co-generative tokenizer) 互补: PIT 是联合训练 tokenizer+NTP，FlexCode 是独立训练双码本
- MoE Router 与 IDEA-onemall-4 (MoE Load Balancing) 相关

### 实验设计草案

**Phase 1 — CF Codebook 构造**:
- 用 SASRec-style 模型在行为序列上训练 → 得到 item collaborative embedding
- 对 CF embedding 做 RQ-VAE → 生成 CF tokens
- 与现有 semantic tokens (MLP-FSQ) concat → 双 SID

**Phase 2 — MoE Router**:
- 按 item interaction frequency 路由: 前 20% items 多分 CF tokens, 后 80% 多分 semantic tokens
- 固定总预算 L=3, 测试 {(2CF,1SEM), (1CF,2SEM), (MoE 动态)}

### 关键问题

1. CF embedding 需要训练一个 SASRec 模型 → 额外训练成本
2. 我们 5M items 中 head/tail 分布与 FlexCode 的 KuaiRand/Industrial 是否可比
3. 双 SID concat 后 NTP 输入长度翻倍 → 需要 Token Merger (IDEA-genrec-1) 配合
4. NTP 后阶段再考虑 tokenizer 改进 → P2

---

## IDEA-crab-0: 过度热门 Token 分裂去偏 (Codebook Rebalancing)

**优先级**: P2 (NTP 后)
**来源**: CRAB, Walmart (arxiv 2604.05113, Apr 2026)
**状态**: 待讨论

### 核心思想

CRAB 发现 GeneRec 的 popularity bias 根源在 **codebook 不均衡**: 语义相似的热门 items 映射到同一 token，累积交互频次后该 token 变成"过度热门 token"，模型训练偏向生成这些 token → 放大 popularity bias (比 SASRec 高 7.2%)。

提出 **post-hoc 去偏** (不需要从头重训):
1. **Token 分裂**: 识别 top-5% 热门 tokens，将其 child tokens 通过 regularized K-means 重新分配到 M 个新 parent tokens
2. **Balanced Loss**: 约束分裂后的新 tokens popularity 均匀: L_bal = Σ(P(c_k(m)) - P_avg)²
3. **Hierarchical Semantic Regularizer**: tree-structure-aware loss 促进 sibling tokens 表示一致性，同时帮助新 token 从语义邻居迁移知识
4. LoRA 高效微调 (只 1/11 训练时间)

**核心结果**:
- Industrial dataset: MGU@10 降低 16.5% (popularity bias), HR@10 持平
- 分裂中间层 (Level B) 效果最好 — "Hourglass 现象": 中间层语义过度集中
- 10% splitting ratio 最优，过度分裂破坏语义完整性
- 高效: 只需 0.28h (vs RW 3.11h, D2LR 2.75h)

### 与当前项目的关联

- 当前 MLP-FSQ 前两层 KMeans Gini=0.31，存在不均衡
- CRAB 是 **post-hoc** 方法 → 不需要改 tokenizer 训练，直接对已有码本操作
- 与 IDEA-sid-2 (Balanced KMeans) 互补: sid-2 在训练时强制均衡，CRAB 在训练后分裂修复
- 与 IDEA-unirec-1 (Capacity-Constrained SID) 目标一致但方法不同: 一个是训练时约束，一个是训练后修复
- "Hourglass 现象" insight 有价值: 我们的 3 层 SID 中间层 (L2) 是否也有过度集中

### 实验设计草案

**Phase 1 — Token Popularity 分析**:
- 统计当前 SID 各层 token 的 popularity (关联 item 的交互频次之和)
- 可视化 Gini + Top 5% token 占比
- 验证是否存在 Hourglass 现象

**Phase 2 — Token 分裂**:
- 对 top-10% popular tokens 做分裂 (M=2~3)
- 用 regularized K-means 保持 hierarchical structure
- LoRA 微调 NTP 模型适应新码本

### 关键问题

1. 分裂后 SID vocab 增大 → NTP 模型的 embedding table 需要扩展 (新 token 需要初始化)
2. 我们用 RQ-KMeans (非 RQ-VAE)，tree structure 严格 → Eq.5 直接适用
3. NTP 后阶段再做 codebook 调优 → P2

---

## IDEA-adasid-0: 自适应碰撞调控 (Adaptive Semantic-Qualified Collision Regulation)

**优先级**: P2 (NTP 后)
**来源**: AdaSID (Kuaishou E-commerce + UESTC, arxiv 2604.23522, Apr 2026)
**状态**: 待讨论

> **与 IDEA-quasid-0 的关系**: 同一快手团队的后续工作。QuaSID 用 Hamming distance 做固定阈值碰撞分级; AdaSID 升级为两阶段自适应调控——不仅判断碰撞是否有害，还根据局部拥挤度和训练进度动态调整惩罚力度。AdaSID 在 Toys/Beauty 上全面超越 QuaSID (平均 +5.2%)。

### 核心思想

AdaSID 将 SID 碰撞调控建模为 **两阶段自适应过程**:

**Stage 1 — Semantic-Adaptive Overlap Relaxation (语义自适应放松)**:
- 计算碰撞 pair 的 encoder 空间 cosine similarity
- 引入 **depth-aware semantic gate**: 碰撞越深 (overlap depth 越大)，放松阈值越严格
  - 阈值向量 η = [η₁ ≤ η₂ ≤ ... ≤ η_L]，如 [0.18, 0.24, 0.30]
  - 当 sim_ij ≥ η_{o_ij} 时，该 pair 豁免 repulsion (语义兼容的碰撞保留)
  - 当 sim_ij < η_{o_ij} 时，该 pair 保留 repulsion (有害碰撞)
- **关键 insight**: 浅层碰撞 (共享 1-2 个 token) 放松条件宽松; 深层碰撞 (几乎相同 SID) 只有极高语义相似度才允许共享

**Stage 2 — Adaptive Pressure Allocation (自适应压力分配)**:
- **Load-Adaptive Collision Strengthening (空间维度)**: 统计 mini-batch 中碰撞签名 (layer-wise overlap pattern) 的频次，拥挤区域施加更强 repulsion
  - 碰撞签名 κ_ij = [I(s¹_i=s¹_j), ..., I(s^L_i=s^L_j)]
  - 局部碰撞负载 c_ij = Σ I(κ_uv = κ_ij)
  - 负载越高 → strengthening factor 越大 (有界单调函数)
- **Progress-Adaptive Objective Rebalancing (时间维度)**: 训练早期强调 collision loss，训练后期逐渐增大 collaborative alignment loss
  - λ_col(τ) = 1 - (1 - λ_min_col) · τ (衰减)
  - λ_cf(τ) = λ_max_cf · τ (增长)
  - τ = clip((t - T_start) / (T_end - T_start), 0, 1)

**总目标**: L = L_rec + L_rq + λ_col(τ) · L_ada_col + λ_cf(τ) · L_cf

### 实验数据

| 数据集 | 方法 | Recall@3 | NDCG@3 | Recall@5 | NDCG@5 |
|--------|------|----------|--------|----------|--------|
| Toys | QuaSID | 0.0195 | 0.0157 | 0.0273 | 0.0191 |
| Toys | **AdaSID** | **0.0214** | **0.0175** | **0.0281** | **0.0202** |
| Beauty | QuaSID | 0.0201 | 0.0155 | 0.0268 | 0.0186 |
| Beauty | **AdaSID** | **0.0205** | **0.0164** | **0.0275** | **0.0190** |

消融实验 (Beauty): SeAR (语义放松) 去掉 → Recall@3 降 10.2%; PAR (进度调度) 去掉 → Recall@5 降 14.2%; LAS (负载自适应) 去掉 → 稳定但轻微降低。

**快手电商在线 A/B** (短视频检索, 千万级用户):
- **GMV +0.98%, Orders +0.91%, GPM +1.16%**
- 离线 ranking: Overall CTCVR AUC +0.05pp, Cold-start CVR AUC +0.08pp

### 与当前项目的关联

- 当前 MLP-FSQ collision 10.7% — 碰撞率不低，有优化空间
- AdaSID 的 depth-aware semantic gate 可直接应用于我们的 3 层 SID: 不同深度碰撞区别对待
- **Load-adaptive strengthening 特别有价值**: 我们的 L1 KMeans (1024 clusters) 某些 cluster 可能过度拥挤，AdaSID 自动识别拥挤区域加强惩罚
- Progress-adaptive rebalancing 可在 embedding fine-tune 阶段实现 (如果走 IDEA-sid-1 路线)
- 与 IDEA-quasid-0 可组合: 用 AdaSID 的自适应框架替换 QuaSID 的固定 Hamming 阈值

### 实验设计草案

**Phase 1 — 碰撞语义分析** (同 quasid-0 Phase 1):
- 分析所有碰撞 pair 的 cosine similarity 分布
- 按 overlap depth 分层统计，验证 "深碰撞 pair 语义相似度更高" 假设
- 计算有害碰撞占比 (sim < threshold 的 pair 数量)

**Phase 2 — Adaptive Collision Loss**:
- 在 tokenizer embedding fine-tune 中加入 AdaSID 的两阶段 loss
- 超参: depth-aware thresholds [η₁, η₂, η₃], f_max ∈ {2.0, 3.0}, schedule 起止步数
- 评估: collision_rate, codebook utilization (entropy, min perplexity), semantic_neighbor_HR

### 关键问题

1. 我们用 RQ-KMeans (离线 KMeans fit)，不是端到端训练 → AdaSID 的 loss 需要在 embedding 空间施加（先 fine-tune embedding，再重新跑 KMeans）
2. collision 10.7% 对行为质量是否真的有害? EXP-008 已证明高 collision 不等于低质量，需要先做 Phase 1
3. 在线效果 (GMV +0.98%) 虽然显著，但 QuaSID 的 GMV +2.38% 更大 — 可能因为 baseline 不同

---

## IDEA-dos-0: 双流正交残差量化 (Dual-Flow Orthogonal RQ)

**优先级**: P2 (NTP 后)
**来源**: DOS (Meituan, arxiv 2602.04460, WWW 2026)
**状态**: 待讨论

### 核心思想

DOS 针对 SID 学习的两个基本问题: (1) **Codebook-Generation Gap** — 现有方法 task-agnostic 学 SID (纯重建/聚类)，与下游生成任务脱节; (2) **量化语义损失** — 标准 RQ 的固定坐标系不适配 LLM 语义结构。

**Dual-Flow Integration (DFI)**:
- 用 **user-item 双塔** 在量化时同时编码用户行为序列和目标 item
- User 塔: Transformer Encoder 编码 click sequence 的 LLM embedding
- Item 塔: 编码 target item 的 LLM embedding
- **共享码本**: 两塔共享同一个 codebook → user interest 和 item 被映射到统一语义空间
- 训练目标: BCE (user-item 匹配) + VQ loss + Recon loss + Orth loss
- 关键: SID 码本不再是孤立学习，而是感知生成任务的上下文

**Orthogonal Residual Quantization (ORQ)**:
- 每层量化前先用可学习正交矩阵 W_orth 旋转输入 (约束 W·W^T = I)
- MLP 生成 dimension-wise weight score → **top-k masking** 选出 primary features (task-relevant)
- Primary features 做码本量化; secondary features + residual 传给下一层
- L_Mutual: 最大化 primary features 与 task label Y 的互信息
- 保证 X_pri ⊥ X_sec (正交分解)，不丢失信息

### 实验数据

**离线** (Meituan 生产数据, 24M items, 180M interactions):

| 方法 | AUC | F1-Score |
|------|-----|---------|
| RQ-KMeans | 0.8363 | 0.7641 |
| RQ-VAE | 0.8526 | 0.7739 |
| DAS | 0.8539 | 0.7869 |
| **DOS** | **0.8763** | **0.8057** |

**NTP 下游** (HSTU framework, Hit@10):

| 方法 | All | Busi_A | Busi_B | Busi_C | Busi_D |
|------|-----|--------|--------|--------|--------|
| HSTU-RQ-KMeans | 0.0410 | 0.0252 | 0.0554 | 0.0398 | 0.0421 |
| HSTU-DAS | 0.0511 | 0.0325 | 0.0672 | 0.0502 | 0.0541 |
| **HSTU-DOS** | **0.0676** | **0.0457** | **0.0797** | **0.0730** | **0.0718** |

**在线 A/B** (Meituan 生产流量 30%, 一周): **+1.15% revenue**

消融: MLP 替代 Encoder → AUC 降至 0.8462; 不共享码本 → 0.8671; 加 Decoder → 0.8626 (重建目标与 task-relevant 选择冲突)

### 与当前项目的关联

- 当前 SID 学习是 task-agnostic (RKMeans 聚类 Qwen3 embedding)，DOS 指出这导致 codebook-generation gap
- **IDEA-sid-1 失败的启示**: EXP-007/009 尝试注入协同信号到 embedding 失败; DOS 采用不同策略 — 不改 embedding，而是在量化阶段引入 user behavior context
- DFI 的共享码本思想与 IDEA-flexcode-0 (FlexCode dual codebook) 互补: FlexCode 分 CF/Semantic 两个码本，DOS 用共享码本统一 user-item 空间
- ORQ 的正交旋转与 IDEA-sid-0 (OPQ) 思想一致，但 OPQ 是静态预处理，ORQ 是端到端可学习
- **decoder 无效的发现很重要**: 重建目标与 task relevance 冲突 → 支持 "不要追求完美重建" 的直觉

### 实验设计草案

**Phase 1 — Task-Aware 量化分析**:
- 在现有 MLP-FSQ 量化后的 SID 上，测量: user click sequence 中前 N 个 item 的 SID 是否能预测 target item 的 SID (简单 BCE 模型)
- 如果预测性差 → 证实 codebook-generation gap 存在，DOS 有价值

**Phase 2 — ORQ 模块移植**:
- 在 MLP-FSQ 的 MLP head 之前加 ORQ 层 (正交旋转 + dimension masking)
- 保持 FSQ 后端不变，只改输入空间
- 评估: semantic_neighbor_HR, collision_rate, 下游 NTP Recall

### 关键问题

1. 我们用 MLP-FSQ (非 RQ-VAE)，ORQ 中的 residual 量化不直接适用; 需要适配 FSQ 的 straight-through 路径
2. 论文只有 4 页 (industry track)，技术细节有限 — 特别是 L_Mutual 的计算方式和正交约束的训练稳定性
3. 共享码本要求同时有 user sequence 和 target item — 离线 batch tokenization 时没有 "target item"，需要改为采样正例
4. 当前 tokenizer 已确认 (MLP-FSQ h=64)，改量化方案需要重训全链路 — 成本高，需要先验证 Phase 1

---

## IDEA-rqgmm-0: Gaussian Mixture Residual Quantization (RQ-GMM)

**优先级**: P2
**来源**: RQ-GMM (Tencent + Fudan University, arxiv 2602.12593)
**状态**: 待讨论 — NTP 后期，当 tokenizer 质量成为瓶颈时再评估

### 核心思想

用 **Gaussian Mixture Model** 替代 K-Means 做残差量化，引入概率建模以更好捕捉 embedding 空间的统计结构。

1. **Gaussian Mixture Quantization**: 在每个 RQ level，用 K 个高斯分布建模残差分布:
   - `p(r) = Σ π_k * N(r | μ_k, Σ_k)`，对角协方差 `Σ_k = diag(σ²_k,1, ..., σ²_k,D)`
   - 每个 codebook vector = 高斯均值 μ_k，额外存储 per-dimension 方差 σ²_k
2. **Soft Assignment (训练) + Hard Assignment (推理)**:
   - E-step: 计算后验 `γ_k = p(k|r)` (soft assignment, 用于 M-step 参数更新)
   - M-step: 更新 μ, σ², π (标准 EM)
   - 推理时: `k* = argmax_k γ_k`, `z_q = μ_{k*}` (等效于最近邻，但距离度量考虑协方差)
   - 残差传播用 hard assignment 保持与推理一致
3. **No Encoder-Decoder**: 直接在原始 embedding 空间操作 (与 RQ-KMeans 相同)，无需 VQ-VAE 的 encoder/decoder 网络

### 关键实验数据

**离线 (Amazon Review, BERT 768D embeddings, 2-level RQ, 128 codes/level)**:

| 方法 | RMSE | 码本利用率 (L1/L2) | AUC (FNN w/ Emb) |
|------|------|-------------------|------------------|
| VQ-VAE | 0.614 | 33.7% | 0.654 |
| RQ-VAE | 0.173 | 73.9%/71.8% | 0.659 |
| RQ-KMeans | 0.121 | 86.7%/87.1% | 0.667 |
| **RQ-GMM** | **0.117** | **89.5%/89.3%** | **0.678** |

- GMM vs KMeans: RMSE -3.3%, 码本利用率 +2.8pp, AUC +0.011
- GMM 收敛更快 + 更平滑 (Figure 1)

**在线 A/B (Tencent 短视频平台, 7 天, 数亿 DAU)**:

| 对比 | Advertiser Value 提升 |
|------|---------------------|
| vs 直接 embedding | **+3.600%** |
| vs RQ-VAE | **+1.502%** |
| vs RQ-KMeans | **+0.613%** |

### 与当前项目的关联

- 我们当前用 **RKMeans (2 层 K-Means) + MLP-FSQ** — RQ-GMM 可以替代 RKMeans 的前两层
- **核心价值**: 码本利用率从 86.7% → 89.5% — 解决 codebook collapse 问题
- **对 boundary samples 更优**: soft assignment 让分布边界上的 item 获得更合理的 SID，减少语义断裂
- **计算成本相当**: 与 RKMeans 同阶 O(TLNKD)，但收敛更快 (fewer iterations)
- **与 IDEA-dos-0 互补**: DOS 引入 user context 改变量化目标，RQ-GMM 改善量化算法本身
- **与 IDEA-crab-0 互补**: CRAB 通过 rebalancing 解决 codebook 不平衡，RQ-GMM 通过 mixing coefficients π_k 自动反映数据密度

### 实验设计草案

**Phase 1 — Drop-in 替换 RKMeans**:
1. 用 scikit-learn 的 `GaussianMixture` 替换 `model/rkmeans.py` 的 K-Means
2. 同样 2 层 × 1024 clusters → 比较 RMSE, 码本利用率, semantic_neighbor_HR
3. 保持 MLP-FSQ 第三层不变 → 只改前两层量化方法

**Phase 2 — 全 GMM-RQ (如果 Phase 1 有收益)**:
1. 实现自定义 RQ-GMM (因为 scikit-learn 不支持 residual quantization)
2. 3 层 RQ-GMM × 1024 clusters (不需要 FSQ 第三层)
3. 与 MLP-FSQ 端到端对比: SID 质量 + NTP Recall

### 关键问题

1. **MLP-FSQ 的 MLP head 已经做了非线性投影**，RQ-GMM 在原始空间操作 — 需要在 MLP 之后的空间做 GMM 还是之前？
2. 论文的 embedding 是 768D (BERT)，我们是 1024D (Qwen3-0.6B) — 高维对角 GMM 是否足够？
3. **概率建模的额外好处**: GMM 的 per-cluster 方差信息可以用于 confidence estimation — 高方差 cluster 的 SID 不确定性高，可以传递给 NTP 做 uncertainty-aware training
4. 当前 tokenizer 已确认 (MLP-FSQ h=64)，且 EXP-015 显示 irreducible loss 已接近 → tokenizer 改进的绝对收益可能有限
5. **优先级低于 RL 对齐** (EXP-037/038/039) — 等 RL 链路稳定后，tokenizer 改进作为下一轮突破口

---

## IDEA-coins-0: COINS — RQ + OPQ 两阶段 SID 冷启动协同转移

**优先级**: P2
**来源**: COINS (arxiv 2510.12604, WWW 2026)
**状态**: 待讨论 — 主场景为 CTR prediction 的 cold-item representation enhancement, 我们是 retrieval 端; 但 "RQ coarse 共享 + OPQ fine 差异化" 的两阶段思路可借鉴

### 核心思想

COINS 针对电商搜索中**冷启动 item 缺协同信号的 Matthew 效应**, 提出一个 RQ+OPQ 融合的 SID 表征方案。

**核心 insight**: 单纯 "content-collaborative 对齐" 忽略了 **asymmetry** — 协同信号天然是粗粒度的 (按 item popularity 分层), content 信号天然是细粒度的 (每 item 独特)。强制对齐会模糊个体差异。

**RQ-OPQ 两阶段编码**:
1. **RQ (Residual Quantization) 阶段 — 共享协同信号转移**: RQ codes 捕获 items 之间的**共性结构**, 协同信号 (从高热 item 学到) 通过共享的粗粒度 codeword propagate 给冷 item
2. **OPQ (Optimized Product Quantization) 阶段 — 差异化信息**: OPQ codes 编码**每 item 独特的精细信息**, 保留个体差异性

两阶段对齐 = 冷 item 继承协同知识 + 保持个体特征。

**在线 A/B (WWW 2026)**:
- item CTR **+1.66%**
- buyers **+1.57%**  
- order volume **+2.17%**

### 与当前项目的关联

**背景复杂性**:
- 我们 **IDEA-sid-0 (纯 OPQ SID)** 已 ❌ 关闭 — 纯 OPQ 在 behavior 质量上输给 MLP-FSQ (EXP-004)
- COINS 是 **RQ + OPQ 叠加使用**, 不是纯 OPQ。RQ 层专门给冷启动 item 做协同信号传递, 不承担全 SID 结构

**可借鉴角度**:
- 对冷 item 的 representation 增强思路: 用 shared coarse coding (我们的 L0 KMeans) 做协同转移, 用 fine-grained coding (我们的 L2 MLP-FSQ) 做差异化 — 本质上我们现有 tokenizer 已是层级的, COINS 只是命名化了这个 分层的 collaborative-vs-differentiation 角色
- **Cold-item 数据分布下的评估方法**: 单独 report cold item 子集的 Recall/CTR, 而不只是全体平均 — 我们 eval 还没做这个切分

**差异**:
- COINS 是 CTR prediction 场景 (ranking), 我们是 generative retrieval
- COINS 有独立的 collaborative embedding 源 (user-item 交互图), 我们没有

### 实验设计草案

**P2 存档, 若未来做 cold-start 验证**:

**Phase 1 — Cold item subset evaluation (零成本, 今天可做)**:
1. 从 eval 数据按 interaction count 切分: cold (<5 interactions) / warm (5-50) / hot (>50)
2. 对 EXP-020 checkpoint 分别报告 R@500 / R@10
3. 预期: 现有 MLP-FSQ tokenizer 下, cold item recall 显著低于 hot item

**Phase 2 — 只有在 Phase 1 发现 cold item 显著差距时才执行**:
- 探索在现有 RKMeans 两层 + MLP-FSQ 第三层基础上, L0 增加"冷启动专属协同 code"的方案
- 或者考虑引入 collaborative embedding 独立信号 (类似 IDEA-flexcode-0 的双码本)

### 关键问题

1. **应用场景错配**: COINS 是 CTR 排序端, 我们是 retrieval 端。CTR 场景里 item representation 与 user representation 做点积, retrieval 场景里 SID 是 generation target。角色不同, 移植成本高
2. **与 IDEA-flexcode-0 (FlexCode 双码本 CF+Semantic) 重叠**: FlexCode 已是双码本分别编 CF/semantic, 路线相似。COINS 是 RQ+OPQ 同一 SID 序列的两阶段, FlexCode 是两个独立 codebook。前者更紧凑, 后者更灵活
3. **与 IDEA-gatesid-0 (刚添加) 互补**: GateSID 在 representation 层做 per-item gating, COINS 在 tokenizer 层做 two-stage encoding。两者可组合
4. **Cold item 定义**: interaction count 阈值需要根据我们数据分布决定

### 相关 idea

- IDEA-sid-0 (纯 OPQ): ❌ 已关闭, COINS 的 RQ-OPQ 叠加是不同路线
- IDEA-flexcode-0 (FlexCode 双码本): 独立 CF + Semantic codebooks, 与 COINS 合一 SID 路线区别
- IDEA-gatesid-0 (GateSID): representation 层 gating, 可组合
- IDEA-adasid-0 (AdaSID Kuaishou): 自适应 collision 处理冷 item, 另一角度
