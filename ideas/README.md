# Ideas

从论文/技术文章中提炼的实验想法，按**改进维度**组织。每个文件对应一个维度，包含该方向所有 idea 的演进关系、实验设计和优先级评估。

想法成熟后迁移到 `experiments/log.md` 作为正式实验。

## 文件索引

| 文件 | 维度 | Ideas 数 | P0 |
|------|------|---------|-----|
| [tokenizer.md](tokenizer.md) | 量化方法 (RQ/OPQ/FSQ/Balanced) | 4 | sid-0, gr4ad-0, onemall-5 |
| [embedding.md](embedding.md) | 表征增强 (协同/多模态/属性) | 4 | — |
| [architecture.md](architecture.md) | 模型架构 (LazyAR/QFormer/MoE/Attn) | 5 | — |
| [training.md](training.md) | 训练目标 (Contrastive/MTP/Value/Multi-beh) | 5 | onemall-0 |
| [rl-alignment.md](rl-alignment.md) | RL 对齐 (GRPO/DPO/RSPO) | 3 | — |
| [inference.md](inference.md) | 推理优化 (Dynamic Beam) | 1 | — |
| [scaling.md](scaling.md) | 扩展性 (序列长度 vs 模型大小) | 1 | oneloc-4 |

**总计: 23 ideas (5 P0 / 11 P1 / 7 P2)**

## 全局演进图

```mermaid
graph TD
    subgraph Tokenizer["🔢 Tokenizer"]
        SID0["IDEA-sid-0<br/>OPQ 并行 ID<br/>→ EXP-004 ✅"]
        SID2["IDEA-sid-2<br/>Balanced KMeans"]
        GR0["IDEA-gr4ad-0<br/>MGMR 不等大码本"]
        OM5["IDEA-onemall-5<br/>RKMeans+FSQ<br/>→ EXP-003"]
    end

    subgraph Embedding["📐 Embedding"]
        SID1["IDEA-sid-1<br/>协同信号增强"]
        SID3["IDEA-sid-3<br/>多模态 ESANS"]
        OM3["IDEA-onemall-3<br/>属性增强"]
        OL3["IDEA-oneloc-3<br/>Side-info 融合"]
    end

    subgraph Architecture["🏗️ Architecture"]
        GR1["IDEA-gr4ad-1<br/>LazyAR 解码器"]
        OM1["IDEA-onemall-1<br/>Query-Former"]
        OM4["IDEA-onemall-4<br/>Loss-Free MoE"]
        OL0["IDEA-oneloc-0<br/>Context Attn"]
        OL1["IDEA-oneloc-1<br/>Category Prompt"]
    end

    subgraph Training["🎯 Training"]
        OM0["IDEA-onemall-0<br/>Contrastive Loss"]
        SID4["IDEA-sid-4<br/>MTP 辅助 Loss"]
        SID5["IDEA-sid-5<br/>Codebook Embed 聚合"]
        GR2["IDEA-gr4ad-2<br/>Value-Aware"]
        OL5["IDEA-oneloc-5<br/>Multi-behavior"]
    end

    subgraph RL["🎮 RL Alignment"]
        OM2["IDEA-onemall-2<br/>GRPO/DPO"]
        OL2["IDEA-oneloc-2<br/>DPO+双目标"]
        GR3["IDEA-gr4ad-3<br/>RSPO"]
    end

    subgraph Inference["⚡ Inference"]
        GR4["IDEA-gr4ad-4<br/>Dynamic Beam"]
    end

    subgraph Scaling["📈 Scaling"]
        OL4["IDEA-oneloc-4<br/>序列长度>>模型大小"]
    end

    %% Cross-dimension dependencies
    SID0 -->|"OPQ 长 ID 需要"| SID5
    SID0 -->|"并行预测 = MTP primary"| SID4
    SID1 --> OM3
    OL3 -.->|"统一为 embedding enrichment"| SID1
    GR0 -->|"不等大码本适配"| GR4
    OM0 -->|"建立强基线"| OM2
    OM0 -->|"建立强基线"| OL2
    GR2 -->|"reward signal"| GR3
    OM2 -->|"升级为 list-wise"| GR3
    SID0 -->|"长 ID + 大 beam"| GR1
    OL4 -->|"指导序列长度"| OM1
    SID2 -->|"大码本更需 balanced"| GR0
```

## ID 来源追溯

| Prefix | 来源 | 论文 |
|--------|------|------|
| `sid` | 知乎综述 3.1 节 + Meta RPG (KDD'25, arxiv 2506.05781) | 语义 ID 构造方法综述 |
| `gr4ad` | GR4AD (arxiv 2602.22732) | 快手大规模广告生成式推荐 |
| `onemall` | OneMall (arxiv 2601.21770v2) | 快手电商端到端生成式推荐 |
| `oneloc` | OneLoc (arxiv 2508.14646v1) | 快手地理感知生成式推荐 |

## 全局优先级总览

### P0 — 战略方向 / 立即执行

| ID | 维度 | 实验 | 原因 |
|-----|------|------|------|
| IDEA-sid-0 | Tokenizer | OPQ 并行语义 ID → EXP-004 | ARCHITECTURE.md 核心方向，RPG 完整验证 |
| IDEA-gr4ad-0 | Tokenizer | MGMR 不等大码本 | 零成本改进 collision/utilization |
| IDEA-onemall-5 | Tokenizer | RKMeans+FSQ → EXP-003 | OneMall 验证方向正确，代码已就绪 |
| IDEA-onemall-0 | Training | NTP Contrastive Loss | OneMall 标配，为 RL 建立强基线 |
| IDEA-oneloc-4 | Scaling | 序列长度 vs 模型大小 | 直接决定资源分配策略 |

### P1 — 高价值

| ID | 维度 | 实验 |
|-----|------|------|
| IDEA-sid-1 | Embedding | 协同信号增强 |
| IDEA-sid-2 | Tokenizer | Balanced KMeans |
| IDEA-sid-4 | Training | Token-Space MTP Loss |
| IDEA-onemall-1 | Architecture | Query-Former 序列压缩 |
| IDEA-onemall-2 | RL | GRPO/DPO 对齐 |
| IDEA-onemall-3 | Embedding | 属性增强 Contrastive |
| IDEA-gr4ad-1 | Architecture | LazyAR 解码器 |
| IDEA-gr4ad-2 | Training | Value-Aware 训练 |
| IDEA-gr4ad-4 | Inference | Dynamic Beam Search |
| IDEA-oneloc-2 | RL | DPO + 双目标 |
| IDEA-oneloc-3 | Embedding | Side-info 融合 |
| IDEA-oneloc-5 | Training | Multi-behavior 序列 |

### P2 — 有前置依赖

| ID | 维度 | 实验 |
|-----|------|------|
| IDEA-sid-3 | Embedding | 多模态 ESANS |
| IDEA-sid-5 | Training | Codebook Embed 聚合 |
| IDEA-onemall-4 | Architecture | Loss-Free MoE |
| IDEA-gr4ad-3 | RL | RSPO 排序优化 |
| IDEA-oneloc-0 | Architecture | Context-augmented Attn |
| IDEA-oneloc-1 | Architecture | Category Prompt |
