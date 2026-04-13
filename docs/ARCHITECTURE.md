# Generative Recommendation Architecture Design

## Reference

- OneRec: arxiv 2506.13695
- OneRec-V2 ("Lazy Decoder-Only"): arxiv 2508.20900
- OneRec 作者知乎文章: https://zhuanlan.zhihu.com/p/1918350919508140128
- Mixtral MoE: arxiv 2401.04088

---

## OneRec 作者核心洞察

### 1. 模型规模与部署

- **目标模型 10B**，线上部署 **1B MoE**（24 experts, top-2, 13% 激活率）
- **MFU 20-30%** — 模型结构必须简单，不能有奇怪算子，否则硬件利用率上不去
- 生成式推荐的核心优势：**解空间比判别式大得多**，才能撑满更大算力

### 2. Tokenizer 设计（最关键的设计决策）

- **Codebook 要小**: item ID 10B → codebook **8192×3**，才有 n-gram 共现可学
- **必须用平行 tokenizer，不能用残差编码**
  - 残差编码限制检索空间 — 作者原话："残差编码永远不可能思考"
  - 残差 (当前 RKMeans): L1 预测 → 决定 L2 搜索空间 → 决定 L3 搜索空间 (树形，空间受限)
  - 平行 (OneRec): L1, L2, L3 独立预测 → 组合搜索 (网格，空间更大)

### 3. 端到端系统

- 一个模型替代整条 **召回/粗排/精排/重排** pipeline
- 不是某个环节的优化，而是架构级替换

### 4. 生成式消除稀疏 Embedding Table

- 传统推荐: 每个 item 一行 embedding → 数十亿参数的稀疏表 → 需要 TorchRec DMP 分片
- 生成式: item 用 semantic ID tokens 表示 → 只需 codebook embedding (极小)
- 不需要 TorchRec，模型参数全在 transformer + MoE 里

---

## OneRec-V2 "Lazy Decoder-Only" 架构

### 核心改进：去掉 Encoder

- V1: Encoder-Decoder 结构
- V2: **无 Encoder**，改用轻量 Context Processor 生成 static KV pairs
- Decoder 只处理 3 个 target tokens → **94% 计算量削减**

### MoE 配置

- **53 routed experts + 1 shared expert, top-3**
- Load balancing: Switch Transformer style auxiliary loss

### 模型规模 (论文 Table 5)

| Config | Params | Active |
|--------|--------|--------|
| Dense 0.1B | 0.1B | 0.1B |
| Dense 1B | 1B | 1B |
| Dense 8B | 8B | 8B |
| MoE 4B | 4B | 0.5B |

### Scaling Law

```
L̂(N) = 3.13 + 3660 / N^0.489
```

### 线上部署

- 1B model, beam=512, latency=36ms, **MFU=62%**
- 3 semantic IDs per item, parallel (non-residual) tokenization

---

## 我们的模型配置

基于 OneRec 架构 + 8×A100 (80GB, NVLink) 环境：

|                     | S (eval/调参)  | M (实验)       | L (接近线上)    |
|---------------------|---------------|---------------|----------------|
| embed_dim           | 256           | 512           | 1024           |
| n_layers            | 6             | 12            | 24             |
| n_heads             | 8             | 8             | 16             |
| MoE experts         | 8             | 16            | 24             |
| MoE top-k           | 2             | 2             | 2              |
| expert FFN dim      | 1024 (4x)     | 2048 (4x)     | 4096 (4x)      |
| **每层 attention**   | 0.26M         | 1.0M          | 4.2M           |
| **每层 MoE FFN**     | 4.2M          | 33.6M         | 201M           |
| **每层 total**       | 4.5M          | 34.6M         | 205M           |
| **总参数**           | ~39.5M        | ~415M         | ~4.9B          |
| **激活参数 (top-2)** | ~11M          | ~55M          | ~420M          |
| codebook            | 256×3         | 256×3         | 8192×3         |
| seq (10 items)      | 30 tokens     | 30 tokens     | 30 tokens      |
| 8×A100 训练          | 单卡秒级       | DDP 有意义     | 需要 DDP+AMP   |
| 8×A100 推理          | 无压力         | 无压力         | 需要优化       |

> S 档实测: 39.5M total, ~11M active (SwiGLU 3-matrix 比原估算略大)

---

## 当前实现状态

### 已实现 (S 档)

- [x] `ExpertFFN`: SwiGLU FFN (w1 gate + w3 up → SiLU → w2 down)
- [x] `SparseMoEBlock`: Linear router → softmax → top-k → dispatch → weighted combine
- [x] Load balancing auxiliary loss (Switch Transformer style)
- [x] `CausalTransformerLayer`: 支持 `use_moe=True/False` 切换
- [x] `AutoregressiveNTPModel`: S-tier default config (256d, 6L, 8H, 8E top-2)
- [x] 训练: CE loss + 0.01 × aux_loss, AMP BF16, DataParallel
- [x] 推理: Batched beam search (B×beam 展平为单次 forward)
- [x] KV cache incremental decoding

### 待实现

- [ ] **平行 Tokenizer**: 替换当前残差 RKMeans → 各层独立编码
- [ ] **Context Processor** (OneRec-V2 Lazy Decoder-Only): 替代 full encoder
- [ ] M 档实验 (需要 DDP)
- [ ] L 档实验 (需要 DDP + model sharding)
- [ ] 扩大 codebook (256 → 8192) 用于 L 档

---

## 关键文件

| 文件 | 说明 |
|------|------|
| `metrics/sid_prediction.py` | NTP 模型 + MoE + 训练/评估 + beam search |
| `model/train.py` | 端到端训练 CLI (编码 → RKMeans → SID → 导出) |
| `model/rkmeans.py` | RKMeans tokenizer (当前: 残差编码) |
| `eval/batch.py` | 批量评估流程入口 |
| `eval/hyperparam.py` | 超参搜索 |
| `eval/behavior.py` | 行为指标评估框架 |

---

## 设计决策记录

### 2026-04-13: MoE 架构选型

**决策**: 基于 Mixtral 的简单 MoE 实现，不用 MegaBlocks 或 DeepSpeed MoE

**原因**:
1. OneRec 强调 MFU — 简单结构比花哨算子更重要
2. Mixtral 模式最成熟，~100 行核心代码
3. 后续可迁移到 MegaBlocks 做 block-sparse 优化

### 2026-04-13: SwiGLU vs GELU FFN

**决策**: Expert 用 SwiGLU (3 个 weight matrix)，非 MoE 用 GELU (2 个)

**原因**: SwiGLU 在 Llama/Mixtral 中验证效果更好，参数略多但单 expert 计算量可控

### 2026-04-13: 不用 DDP (S 档)

**决策**: S 档用 DataParallel 而非 DDP

**原因**: 39.5M 参数太小，单卡训练秒级完成。DDP 的进程启动开销反而更大。M 档再上 DDP。

### 2026-04-13: 残差 tokenizer 问题 (待解决)

**问题**: 当前 RKMeans 是残差编码，限制了 beam search 搜索空间

**方案**: 需要实现 parallel tokenizer — 各层独立编码，不依赖前层残差。这是下一步最重要的改动。
