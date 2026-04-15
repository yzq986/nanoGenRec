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
├── IDEA-r3vae-0: Reference Vector SID → P2 (NTP 后)
├── IDEA-geogr-0: 地理感知 SID (Co-visited Contrastive) → P2 (需泛化)
├── IDEA-onevision-0: VRQ 视觉对齐 RQ + 动态剪枝 → P2 (需视觉模态)
└── IDEA-mmq-0: 共享-专有多模态混合量化 → P2 (需多模态数据)
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

## IDEA-sid-2: Balanced KMeans

**优先级**: ~~P1~~ → P2 (NTP 后)
**来源**: 3.1.2.1 (OneRec Paper 提到)
**状态**: 待定，降级

> **降级原因 (2026-04-15)**: 前两层 KMeans Gini=0.31，balanced assignment 可改善码本利用率，collision 可能从 10.7% 降到 ~7-8%。但 EXP-008 证明 collision 不是核心指标（OPQ collision 0.06% 反而行为最差），收益不确定。等 NTP 端到端 Recall@K 出来后再决定是否投入。

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
