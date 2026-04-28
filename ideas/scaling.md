# Scaling (扩展性实验)

模型规模 vs 数据规模 vs 序列长度的 scaling law 研究，直接决定资源分配策略。

**影响范围**: `metrics/sid_prediction.py`, `model/train.py`, ARCHITECTURE.md (tier 设计)

---

## 演进路径

```
S-tier (39.5M params, 当前唯一实现)
├── IDEA-oneloc-4: Scaling Law 实验
│   ├── 模型 scaling → EXP-015 ✅ L(N)=2.522+2055/N^0.456, ~100M 趋平
│   │   └── M+ 101M vs S 17.5M: loss 仅降 0.06, tokenizer 是瓶颈
│   └── 序列长度 scaling → 待实验 (当前 max_seq_len=512, ~170 items/user)
├── IDEA-kunlun-0: Rec Scaling Laws (Meta Ads)
│   └── MFU 17%→37%, GDPA + CompSkip, power-law scaling
│   └── 重要性提升: tokenizer 瓶颈突破后成为 scale up 关键
├── IDEA-hstu-0: Sparse Self-Attention Co-design (Meta)
│   └── 5x 训练 / 21x 推理 scaling, 保留 self-attention 表达力
├── IDEA-mtgenrec-0: 分布式 GR 训练系统 (Meituan)
│   └── Dynamic Hash Embedding + Sequence Batching + ID Dedup, 2.4x throughput
└── IDEA-freescale-0: Sequence Load Balancing + SM-Free 通信 (Meta, MLSys 2026)
    └── 长序列 UIH straggler 缓解 + 优先级 embedding 更新 + CPU-RDMA 零 SM 占用
```

---

## 当前结论 (2026-04-17)

**模型参数 scaling 在 ~100M 趋平，tokenizer 32-bit 编码是当前瓶颈。序列长度 scaling 尚未验证。**

### 关键实验数据

| 模型 | Active Params | Eval Loss | PPL | R@500 |
|------|--------------|-----------|-----|-------|
| S-tier | 17.5M | 2.9960 | 27.05 | 58.5% |
| M+-tier | 101M | 2.9371 | 25.12 | 60.7% |
| Irreducible (fit) | ∞ | 2.522 | ~12.5 | — |

**核心 insight**: M+ 比 S 多 6x 参数，但 loss 仅降 0.06。Scaling law 拟合 L(N)=2.522+2055/N^0.456 显示 irreducible loss=2.522 由 tokenizer 决定。突破瓶颈需要：(1) 更高 bits 的 SID (当前 32-bit)；(2) 更长序列 (尚未验证)。

---

## IDEA-oneloc-4: Scaling Law — 序列长度 >> 模型大小

**优先级**: ~~P0~~ → 部分完成 (模型 scaling 已验证, 序列长度待测)
**来源**: OneLoc §4.4 Hyperparameter Experiments
**状态**: EXP-015 模型 scaling 完成; 序列长度 scaling 待实验

> **NTP 阶段更新 (2026-04-17)**: EXP-015 验证了模型参数 scaling law L(N)=2.522+2055/N^0.456。关键发现: M+ (101M active) 比 S (17.5M) 仅降低 loss 0.06，scaling law 在 ~100M 已严重平坦化——tokenizer 32-bit 编码是瓶颈。序列长度维度的 scaling 尚未实验 (当前 max_seq_len=512, ~170 items/user)，这是下阶段重要方向。

### 核心思想

OneLoc 的 scaling 实验揭示了一个关键发现: **序列长度的收益远大于模型大小的收益**。模型从 0.05B 扩到 0.3B，recall/NDCG 平均提升 7%；但序列长度从 100 扩到 300，recall 提升 13%、NDCG 提升 51%。这意味着在资源有限时，应优先增加序列长度而非模型参数。

### 与当前项目的关联

- 当前 `AutoregressiveNTPModel` S-tier config: 6 layers, 256 embed_dim, ~39.5M params
- ARCHITECTURE.md 定义了 M/L tier 但未实现
- **关键问题**: 我们尚未做过 NTP 模型的 scaling 实验
- 更直接的启发: 在 NTP 训练中，user 行为序列的长度是否比模型大小更重要?
- 当前行为序列处理: `data/export_behavior.py` 中导出的序列长度是多少? 是否足够长?

### 实验设计草案

**实验矩阵**:

| 维度 | 小 | 中 | 大 |
|------|-----|-----|-----|
| 模型参数 | S-tier (39.5M) | M-tier (~150M) | L-tier (~500M) |
| 序列长度 | 50 | 100 | 200 |

**设计**:
- 固定量化方案 (RKMeans 3x1024 或 OPQ)
- 3x3 grid: 模型大小 x 序列长度
- 每组训练 NTP 模型到收敛
- 记录 recall@5/10/20, NDCG@5/10/20

**评估**:
- 绘制 scaling curve: recall vs 模型参数 (固定序列长度)
- 绘制 scaling curve: recall vs 序列长度 (固定模型参数)
- 验证 OneLoc 的结论是否在我们的场景复现

### 关键问题

1. **前置依赖**: 需要先有稳定的 NTP 训练 pipeline + 稳定的量化方案
2. 当前 NTP 训练是否已经可以 end-to-end run? 需要确认 `model/train.py` → NTP 的完整流程
3. 行为数据量: 序列长度 200 需要足够的用户行为数据
4. 计算成本: 9 组实验，每组可能需要数小时训练
5. **为什么 P0**: 这个实验的结论直接决定资源分配策略 — 是花钱买更大 GPU 还是花钱采集更多行为数据

---

## IDEA-kunlun-0: Recommendation Scaling Laws (MFU 优化 + GDPA)

**优先级**: P1 — 重要性因 scaling 平坦化而提升
**来源**: Kunlun (Meta Ads, arxiv 2602.10016, Feb 2026)
**状态**: 待讨论

> **NTP 阶段更新 (2026-04-17)**: EXP-015 显示模型 scaling 在 ~100M 已趋平 (tokenizer 瓶颈)。Kunlun 的 MFU 优化和 GDPA 在当前阶段的价值不在于 scale up 模型，而在于：(1) 提高现有模型的训练效率 (更快迭代实验)；(2) 未来突破 tokenizer 瓶颈后 (如 OPQ 长 SID 或更高 bits)，GDPA 是 scale up 的关键技术。

### 核心思想

Kunlun 在大规模推荐系统中建立了类 LLM 的 **power-law scaling laws**。核心发现: 推荐模型 scaling 效率低的根本原因是 **低 MFU (Model FLOPs Utilization)** 和 **资源分配不均**。

解决方案:
1. **Generalized Dot-Product Attention (GDPA)**: 推荐专用的注意力机制
2. **Hierarchical Seed Pooling (HSP)**: 高效特征聚合
3. **Computation Skip (CompSkip)**: 选择性计算，跳过低价值路径
4. **Sliding Window Attention**: 管理用户历史序列

**结果**: MFU 从 **17% 提升到 37%** (B200 GPU), **2x scaling efficiency**, 部署到 Meta Ads 主要模型。

### 与当前项目的关联

- 当前 S-tier 模型小 (39.5M)，MFU 不是瓶颈
- 但 IDEA-oneloc-4 (Scaling Law) 和 IDEA-plum-0 (LLM CPT) 都需要 scale up → Kunlun 的经验直接适用
- **GDPA** 可能比标准 attention 更适合推荐场景: 用户行为序列与自然语言序列的模式不同
- **CompSkip** 与 IDEA-gr4ad-1 (LazyAR) 有关联: 都是选择性计算

### 实验设计草案

**Phase 1 — GDPA 替换标准 Attention**:
- 需要读 Kunlun 论文全文了解 GDPA 具体定义
- 在 `CausalTransformerLayer` 中替换 attention 模块

**Phase 2 — MFU Profiling**:
- 在 8xA100 上 profile 当前 NTP 训练的 MFU
- 识别低效模块 → targeted 优化

### 关键问题

1. GDPA 的具体实现需要论文全文
2. 当前模型太小，MFU 提升不等于训练速度提升 (可能是 memory-bound 而非 compute-bound)
3. CompSkip 需要 per-sample 路由 → 实现复杂度高

---

## IDEA-hstu-0: Sparse Self-Attention + Model-System Co-design (ULTRA-HSTU)

**优先级**: P1
**来源**: ULTRA-HSTU (Meta, arxiv 2602.16986, Feb 2026)
**状态**: 待讨论

### 核心思想

ULTRA-HSTU 通过 **end-to-end model-system co-design** 实现:

1. **Input Sequence Design**: 针对推荐场景优化输入序列构造
2. **Sparse Attention**: 保持 self-attention 的表达能力的同时避免 O(n²) 计算
3. **Model Topology**: 架构拓扑优化以配合系统效率

关键立场: cross-attention (如 IDEA-onemall-1 Query-Former) 虽然解决了 O(n²) 问题，但 **限制了 self-attention 的表达能力**。ULTRA-HSTU 通过 sparse self-attention 既保持表达能力又控制计算量。

**结果**: **5x faster training, 21x faster inference**, 服务 **数十亿用户**, **4-8% engagement improvement**。

### 与当前项目的关联

- 当前 `CausalTransformerLayer` 是 full self-attention (O(n²))，序列短时无问题
- 如果扩展到长序列 (IDEA-oneloc-4 / IDEA-onemall-1):
  - IDEA-onemall-1 选择 cross-attention (Query-Former) → 压缩表达
  - ULTRA-HSTU 选择 sparse self-attention → 保留表达
  - 两种路线的 tradeoff 值得实验对比
- **Model-System Co-design** 的理念: 不要只看模型质量，要同时优化系统效率

### 实验设计草案

**Phase 1 — Sparse Attention 替换**:
- 在 `CausalTransformerLayer` 中加入 sparse attention 选项 (如 sliding window + global tokens)
- 对比: full attention vs sparse attention vs Query-Former 在不同序列长度下的 Recall@K 和训练速度

**Phase 2 — Input Sequence Design**:
- 需要读论文全文了解 ULTRA-HSTU 的 input sequence design 细节
- 可能涉及 action 类型 encoding、时间戳 encoding 等

### 关键问题

1. 论文全文细节 (sparse attention 的具体 pattern) 需要补充
2. 当前序列短 (3 SID tokens)，sparse attention 无收益 → 依赖序列扩展
3. 与 IDEA-onemall-1 (Query-Former) 的对比实验需要统一实验框架

---

## IDEA-mtgenrec-0: 高效分布式 GR 训练系统 (Dynamic Embedding + Sequence Balancing)

**优先级**: P2 — 生产部署基础设施，当前研究阶段不急需
**来源**: MTGenRec (Meituan + Wuhan Univ, arxiv 2505.12663, May 2025)
**状态**: 待讨论

> **P2 原因**: 当前训练规模小 (8 GPU, 17.5M params, 14d 数据)，TorchRec/torchrun 的效率瓶颈尚未触及。当模型 scale up 或数据量扩大到需要 100+ GPU 时，MTGenRec 的技术直接适用。

### 核心思想

MTGenRec 是美团基于 TorchRec 构建的 GR 专用分布式训练系统，解决了 GR 训练中四个工程瓶颈:

1. **Dynamic Hash Embedding Table**: 用 MurmurHash3 + grouped parallel probing 的动态哈希表替换 TorchRec 的静态 embedding table，支持实时 item 增删 (新商品上架/下架)。Key-Value 解耦存储 + chunk-based 分配，扩容时只迁移轻量 key 结构。吞吐提升 1.47-2.22x vs TorchRec MCH
2. **Two-Stage ID Deduplication**: 用户序列中特征 ID 大量重复 (同一 user/item 出现多次)。Stage 1: 本地去重后再 all-to-all 通信; Stage 2: 收到远端 ID 后再次去重。减少 embedding 通信量，吞吐提升 53%
3. **Dynamic Sequence Batching**: 用户序列长度呈长尾分布 (avg=600, max=3000)。固定 batch size → GPU 间负载严重不均 (最大差 25.8ms)。改为 target token count 模式: 二分搜索找最接近目标 token 数的 batch 切分点，GPU memory 利用率从 75%→90%。吞吐提升 26.5% (110G model, 64 GPU)
4. **Automatic Table Merging**: FeatureConfig 接口自动合并相同维度的 embedding table，减少 lookup 算子数。动态表用 bit-shift offset 避免 ID 冲突

### 关键数据

| 指标 | 数值 |
|------|------|
| 训练数据 | 200M 序列/天, avg 600 tokens, max 3000 |
| 模型规模 | GRM 4G (小) ~ 110G (大) GFLOPs |
| GPU 配置 | 8~128 × A100 80GB SXM4, NVLink 600GB/s |
| 吞吐提升 | 1.6x~2.4x vs TorchRec |
| Scaling 效率 | 128 GPU 达到 62.75%~78.5% 理想线性加速 |
| Online A/B (外卖) | +1.22% 用户下单量, +1.31% PV_CTR (vs 2 年迭代的 DRM) |
| 用户规模 | 770M 年交易用户, 日峰 98M 订单 |

### 与当前项目的关联

- **Dynamic Sequence Batching** 最直接相关: 我们用 torchrun + packed sequences 训练，不同 rank 拿到的序列长度不同，可能存在相同的 GPU 负载不均问题。当前未做 dynamic batching — 值得在 scale up 时参考
- **Dynamic Hash Embedding**: 当前不需要 (SID vocabulary 是固定的)，但如果引入 user embedding 或 item side features 作为 sparse embedding，会面临相同的动态增删问题
- **Two-Stage ID Dedup**: 当前序列短 (~170 items/user)，重复 ID 不多。Scale 到长序列时重复率会增加
- **Model Architecture**: 美团的 GRM 用 HSTU (SiLU attention) + MMoE，与我们的 CausalTransformer 不同但训练系统层面通用

### 实验设计草案

**Phase 1 — Dynamic Sequence Batching (可独立实现)**:
- 在 `ntp/data.py` 或 `data/dataset.py` 中实现 token-count-based batch 构造
- 目标: 每个 rank 的总 token 数 ≈ target_tokens (而非固定 batch_size)
- 用 cumulative sum + binary search 找切分点
- 需要修改 gradient averaging: 按各 rank 实际样本数加权 (weighted All-Reduce)
- 评估: GPU 利用率、训练吞吐、收敛曲线是否一致

**Phase 2 — 部署扩展 (远期)**:
- Dynamic hash embedding: 如果引入实时 item 更新
- ID dedup: 序列长度扩展到 1000+ 时

### 关键问题

1. 当前 8 GPU 训练，序列短 (max_seq_len=512)，GPU 间负载差异可能不大 — 需要先 profiling 确认
2. Dynamic batching 改变了每个 rank 的 batch size → gradient 需要 weighted average，实现不能破坏 DDP 的 gradient sync 正确性
3. 论文模型架构 (HSTU) 用了 SiLU attention 而非标准 softmax attention，O(n²) 的 scaling 行为可能不同

---

## 优先级总结

| 优先级 | ID | 实验 | 原因 |
|--------|-----|------|------|
| ~~P0~~ 部分完成 | IDEA-oneloc-4 | Scaling Law: 序列长度 vs 模型大小 | 模型 scaling EXP-015 ✅ (~100M 趋平); 序列长度 scaling 待验证 |
| P1 | IDEA-kunlun-0 | Rec Scaling Laws (MFU + GDPA) | Meta Ads 部署验证; tokenizer 瓶颈突破后成 scale up 关键 |
| P1 | IDEA-hstu-0 | Sparse Self-Attention Co-design | 21x inference scaling, 对比 Query-Former 路线 |
| P2 | IDEA-mtgenrec-0 | 分布式 GR 训练系统 | 美团部署, 100+ GPU scaling, dynamic batch 可先行参考 |
| P2 | IDEA-freescale-0 | Meta FreeScale: Load Balancing + SM-Free 通信 | 256×H100 验证, 90% 通信气泡削减; 当前 8 GPU 受益有限, 未来多节点扩展时核心参考 |

---

## IDEA-freescale-0: FreeScale — Sequence Load Balancing + SM-Free 通信

**优先级**: P2
**来源**: FreeScale (Meta, arxiv 2604.24073, MLSys 2026)
**状态**: 待讨论 (远期参考, 当前 4-8 GPU 单节点受益有限)

### 核心思想

FreeScale 是 Meta 为 DLRM/sequence recommendation 设计的分布式训练系统，针对三个在大规模 (100+ GPU) 训练中占主导的效率瓶颈提供系统性解决方案。在 256×H100 production 集群上实现 **90.3% exposed communication 降低**，且离线 normalized entropy 与 baseline 完全一致（不损失精度）。

**1. Sequence Load Balancing (缓解 Straggler)**

UIH (user interaction history) 长度异质性巨大：同 batch 内 2k vs 21k 样本并存，导致 rank 间计算量差异 > 20%，快 rank 空等慢 rank。FreeScale 在每次 iteration 前用三阶段 AllGather 收集 world UIH lens + candidate lens，再用 `FBS` (First-Fit-Decreasing by sequence size) 或 `VBS` (可变块) partition 算法重分发样本。关键点：
- **不能按长度预先排序**（推荐场景里 temporal ordering 对模型质量敏感）
- **在 trainer 内部做 runtime partition**（不能依赖静态数据布局，因 dynamic resource allocation）
- 21k UIH + 64 GPU 下 straggler% 从 22% 降到 2.4%

**2. Prioritized Embedding Updates**

Vanilla TorchRec 每 iteration 做两次 blocking AllToAll（IDs→lookup, result→rank）。简单 prefetch 会读到 stale embedding（下一 iter 的 lookup 在本 iter backward 之前）。FreeScale 的 insight: **真实 collision rate 仅 ~12%**（P99 = 14%），所以:
- Prefetch **所有非冲突行** → 和 forward 完全 overlap
- 只阻塞等待 collision 行 — exposed communication 变成 O(collision rate × volume)
- 结果: 8k UIH 下 TorchRec exposed comm 111 ms → FreeScale 13 ms

**3. SM-Free Communication**

GPU 上通信和计算同时发生时，NCCL 会占用 SM，导致实际 overlap 受 SM 抢占影响（10% throughput loss 即使调大 NCCL_MAX_NCHANNEL）。FreeScale 走 **CPU-RDMA**: 把 embedding 搬回 CPU 做集合通信，完全让出 GPU SM。对 sequence models (d=128, seq=8192) 观察到稳定 10% speedup，且 speedup 不随 NCCL tuning 变化。

**4. Staged Training Pipeline (不依赖 full graph trace)**

不像 CUDA Graph / `torch.compile` 那样做全图追踪（会禁用 dynamic branching / 第三方 op），FreeScale 把 train step 分为 data loading / forward / backward / opt step / metrics 五个阶段，用 PyTorch module hooks 插桩 (`named_modules()` 枚举 embedding tables)。保留模型迭代灵活性同时引入优化。

### 与当前项目的关联

- **当前不受益**: 我们的 `torchrun` 训练在 4-8 GPU 单节点，UIH 长度 ~170 items (远 < Meta 的 21k)，straggler 不是主要瓶颈
- **序列长度 scaling 验证时受益**: IDEA-oneloc-4 Phase 2 要推 max_seq_len 512→2048+，届时长尾 UIH 会出现 straggler，本方案 Phase 1 (load balancing) 可直接采纳
- **与 IDEA-mtgenrec-0 (MTGenRec) 对比**:
  - MTGenRec 的 "Sequence Batching" ≈ FreeScale 的 "FBS partition"，思路一致
  - MTGenRec 没有 "prefetch + collision-only wait" 机制
  - MTGenRec 基于 TensorFlow 生态，FreeScale 基于 PyTorch/TorchRec + Triton — 后者更贴合我们技术栈
  - **优先参考 FreeScale 而非 MTGenRec**（如果未来做分布式优化）
- **Triton kernel**: FreeScale 用 custom Triton kernel 实现 variable-length attention，我们 `ntp/model.py` 若推长序列也会需要类似优化

### 实验设计草案

**当前阶段不执行。** 等 IDEA-oneloc-4 (序列长度 scaling) 推进到 8k+ UIH 且多节点训练时再评估。届时可做：

**Phase 1 — Load Balancing (可独立实现, 风险低)**:
- 在 `ntp/data.py` 或 `data/distributed_sampler.py` 里实现 FBS partition
- 在每个 iteration 开始前 AllGather 各 rank 的 batch lengths，按 First-Fit-Decreasing 重分发样本
- 评估: rank 间 idle 时间、iteration time 方差、端到端 QPS
- **预期**: 长 UIH (>5k) + 8 GPU 场景收益显著；短 UIH 场景可能是负收益（overhead 超过 straggler）

**Phase 2 — Prioritized Embedding Updates**:
- 依赖 embedding 表足够大 + multi-node 训练，当前不适用

**Phase 3 — SM-Free Communication**:
- 依赖 CPU-RDMA 硬件支持 + NCCL 替换，工程复杂，延后

### 关键问题

1. 当前 DDP + 短 UIH 下 straggler 到底多大？需 profiling 确认，若 <5% 就别折腾
2. FBS partition 改变了每 rank 的实际 batch size → DDP gradient reduce 需要 weighted average (和 MTGenRec 一样的坑)
3. SM-Free 通信需要高速 CPU-NIC 带宽（FreeScale 实验用 8×200 Gb/s InfiniBand），我们云上环境未必匹配
4. Triton kernel 替换标准 PyTorch ops 会影响 `torch.compile`/autocast 兼容性

### 相关 idea

- IDEA-mtgenrec-0 (MTGenRec): Meituan 的同类系统, 技术重叠但 FreeScale 更成熟
- IDEA-oneloc-4: 序列长度 scaling, 是启动 FreeScale 的前置依赖
- IDEA-hstu-0: Sparse attention co-design, 减少 compute 需求 → 降低 FreeScale 必要性
