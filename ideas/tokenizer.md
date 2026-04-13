# Tokenizer (量化方法)

语义 ID 的核心：如何将高维 embedding 离散化为短序列 token。涵盖 RQ/OPQ/FSQ/Balanced KMeans 等量化方案，直接决定 collision rate、codebook utilization 和下游 NTP 模型的上限。

**影响范围**: `model/rkmeans.py`, `model/fsq.py`, `model/rkmeans_fsq.py`, `eval/evaluator.py`

---

## 演进路径

```
RKMeans 3×1024 (EXP-001 baseline, collision=1.75%)
├── IDEA-sid-0: OPQ 并行语义 ID → EXP-004 ✅
│   └── 16~64 token × 256 码本, graph decoding, 非自回归
├── IDEA-sid-2: Balanced KMeans
│   └── 强制均匀 cluster → 提升码本利用率
├── IDEA-gr4ad-0: MGMR 不等大码本
│   └── 16384→4096→1024 multi-resolution + hash 层消除碰撞
├── IDEA-onemall-5: RKMeans + Learned FSQ → EXP-003
│   └── 2 层 RKMeans + 1 层 FSQ, conflict 36%→11%
├── IDEA-pit-0: Co-generative 动态 Tokenizer (PIT, 快手)
│   └── tokenizer+NTP 端到端联合训练, One-to-Many SID, +0.402% App Stay Time
└── IDEA-forge-0: SID Proxy Metrics (FORGE, 阿里淘宝)
    └── 无需 NTP 训练即可评估 SID 质量, 在线收敛减半
```

---

## IDEA-sid-0: OPQ 并行语义 ID

**优先级**: P0
**来源**: 3.1.2.2 (Meta RPG, KDD'25), Kaiming OPQ
**状态**: 已采纳 → EXP-004 (Phase 1 intrinsic), Phase 2 待定
**参考代码**: github.com/facebookresearch/RPG_KDD2025

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

**优先级**: P1
**来源**: 3.1.2.1 (OneRec Paper 提到)
**状态**: 待讨论

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

**优先级**: P0
**来源**: GR4AD §UA-SID, Table 2
**状态**: 待讨论

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

**优先级**: P0 (已有实验计划，需加速执行)
**来源**: OneMall §3.1.3 + §4.5 Tokenizer Strategy
**状态**: 待讨论 → 应立即推进 EXP-003

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

**优先级**: P1
**来源**: PIT (Kuaishou, arxiv 2602.08530, Feb 2026)
**状态**: 待讨论

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

**优先级**: P1
**来源**: FORGE (Alibaba/Taobao, arxiv 2509.20904, Sep 2025)
**状态**: 待讨论

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

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| P0 | IDEA-sid-0 | OPQ 并行语义 ID (已 → EXP-004) | ARCHITECTURE.md 核心方向，RPG 完整验证，代码已就绪 |
| P0 | IDEA-gr4ad-0 | MGMR 不等大码本 | 零成本改进，直接提升 collision/utilization |
| P0 | IDEA-onemall-5 | 立即执行 EXP-003 (RKMeans+FSQ) | OneMall 直接验证方向正确，代码已就绪 |
| P1 | IDEA-sid-2 | Balanced KMeans | 低成本改进码本利用率 |
| P1 | IDEA-pit-0 | Co-generative 动态 Tokenizer | 端到端联合训练 tokenizer+NTP, One-to-Many SID |
| P1 | IDEA-forge-0 | SID Proxy Metrics + Offline Pretraining | 加速 tokenizer 超参搜索 10x |
