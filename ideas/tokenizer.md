# Tokenizer (quantification method)

[English](tokenizer.md) | [Chinese](tokenizer.zh.md)

The core of semantic ID: how to discretize high-dimensional embedding into short sequence tokens. Covers quantitative solutions such as RQ/OPQ/FSQ/Balanced KMeans, which directly determines the upper limit of collision rate, codebook utilization and downstream NTP models.

**Scope of influence**: `model/rkmeans.py`, `model/fsq.py`, `model/rkmeans_fsq.py`, `eval/evaluator.py`

---

## Current Conclusion (2026-04-15)

**MLP-FSQ h=64 is confirmed as the tokenizer route winner and enters the NTP stage. **

### Current tokenizer config

```
Architecture: 2-layer KMeans (1024 clusters) + 1-layer MLP-FSQ (h=64, [4,4,4,4,4,4], codebook=4096)
SID:  3 tokens (L1_cluster, L2_cluster, L3_fsq_code)
Bits: 10 + 10 + 12 = 32 bits
Collision: 10.7%
semantic_neighbor_HR: 0.078
```

### Key experimental data

| Experiment | 方案 | semantic_neighbor_HR | collision | Conclusion |
|------|------|---------------------|-----------|------|
| EXP-008 A | **MLP-FSQ h=64 (3 token, 32 bit)** | **0.0780** | 0.1074 | **赢家** |
| EXP-008 B | OPQ 4×256 (4 token, 32 bit) | 0.0502 | 0.0351 | 等 bits 对照，输 36% |
| EXP-008 C | OPQ 8×256 (8 token, 64 bit) | 0.0326 | 0.0006 | collision 最Low但行为最差 |

**Core insight**: The lower the collision ≠ the better the behavior quality. Hierarchical structures (KMeans→KMeans→FSQ) preserve embedding neighborhoods better than flat structures (OPQ parallel subvectors), with higher behavior co-occurrence rates for SID prefix neighbors.

**NTP stage supplement (2026-04-17)**: EXP-015 scaling law fits irreducible loss a=2.522 (PPL≈12.5), the floor is determined by tokenizer 32-bit encoding. M+ (101M) has reached loss=2.94, which is only 0.42 from floor. **Tokenizer is the bottleneck of the current system** - model scale up has diminishing returns, and a breakthrough requires higher bits SID or better quantification structure.

---

## Evolution path

```
RKMeans 3×1024 (EXP-001 baseline, collision=1.75%)
├── IDEA-sid-0: OPQ parallel semantics ID → EXP-004 → EXP-008 ❌ Behavior quality is not as good as MLP-FSQ
│ └── collision is very low (0.06%) but semantic_neighbor_HR is only 0.033
├── IDEA-onemall-5: RKMeans + Learned FSQ → EXP-003 → EXP-008 ✅ Winner
│ └── MLP-FSQ h=64: collision 10.7%, semantic_neighbor_HR 0.078 (optimal)
├── IDEA-sid-1: Collaborative signal enhancement embedding → EXP-007 + EXP-009 ❌ Dead end
│ └── Full FT/LoRA/QFormer are all stuck at HR@50 ~0.02
├── IDEA-forge-0: SID Proxy Metrics → ✅ Implemented
│ └── semantic_neighbor_hit_rate is the decisive indicator, EXP-008 relies on it to select the winner
├── IDEA-sid-2: Balanced KMeans → P2 (after NTP)
├── IDEA-gr4ad-0: MGMR unequal codebook → P2 (after NTP)
├── IDEA-pit-0: Co-generative dynamic Tokenizer → P2 (after NTP)
├── IDEA-quasid-0: Hamming Repulsion → P2 (after NTP)
├── IDEA-adasid-0: Adaptive Collision Regulation → P2 (after NTP, extends quasid-0)
├── IDEA-dos-0: Dual-Flow Orthogonal RQ (context-aware SID + orthogonal quantization) → P2 (after NTP)
├── IDEA-r3vae-0: Reference Vector SID → P2 (after NTP)
├── IDEA-geogr-0: Geo-aware SID (Co-visited Contrastive) → P2 (needs generalization)
├── IDEA-onevision-0: VRQ visual alignment RQ + dynamic pruning → P2 (requires visual modality)
├── IDEA-mmq-0: Shared-proprietary multi-modal hybrid quantization → P2 (requires multi-modal data)
└── IDEA-rqgmm-0: GMM + residual quantization (probabilistic modeling → higher codebook utilization) → P2 (after NTP)
```

---

## IDEA-sid-0: OPQ parallel semantic ID

**Priority**: ~~P0~~ → ❌ Close
**Source**: 3.1.2.2 (Meta RPG, KDD'25), Kaiming OPQ
**Status**: ~~Adopted → EXP-004~~ → EXP-008 Closed after comparison
**Reference code**: github.com/facebookresearch/RPG_KDD2025

> **Reason for closure (2026-04-15)**: In comparison of EXP-008 and other bits, OPQ 4×256 (32 bit) semantic_neighbor_HR=0.050 loses MLP-FSQ (0.078) 36%; OPQ 8×256 (64 bit) collision is extremely low (0.06%) but semantic_neighbor_HR is only 0.033, worse. Flat subvector structures do not preserve embedding neighborhoods as well as hierarchical structures. Phase 2 (Parallel Prediction Model + Graph Decoding) is no longer advanced.

### Core Idea

Replace residual encoding with Optimized Product Quantization to implement parallel semantic ID. First learn the orthogonal rotation matrix R, and then divide the rotated vector into sub-vectors and quantize them independently.

### RPG Paper Key Details

**Quantification**: OPQ > RQ (confirmed by ablation experiments Table 3). The codebook of each digit is an independent vocabulary list (C⁽¹⁾={1,...,M}, C⁽²⁾={M+1,...,2M}) and is not shared.

**Model Architecture**:
- Transformer encoder encodes user behavior sequence → s ∈ ℝᵈ
- **An independent MLP projection head for each digit** g_j(s) → M-dimensional logits
- MTP loss = Σ CE_j (each digit is conditionally independent)
- Single forward out all token logits, non-autoregressive
- independent head >> shared head (ablation confirmed)

**Inference — Graph-Constrained Decoding (not beam search nor Cartesian product)**:
- **beam search completely fails on OPQ** (Table 3 recall is all 0.0000)
- **Pure Cartesian product combinations will not work** — 256^32 The space is too sparse, and most combinations are invalid SIDs
- Construct SID similarity graph: node=valid SID, edge=embedding similarity top-k neighbors
- Inference: Randomly sample b=10 seeds → expand along the edge of the graph → score top-b → iterate for q rounds
- Complexity O(Mmd + bqkm), independent of the total number of items N
- Finally only access ~10-25% of the item pool

**Optimal configuration** (RPG paper):
- m=16~64 (the larger the data set, the longer), M=256, b=10, k=50~500, q=2~5
- τ=0.03 (temperature), 2-layer Transformer, d=448, ~13M params
- Training: <2 GPU hours (RTX 3090)

**Key Ablation Conclusion**:
- OPQ > RQ (NDCG +2~8%)
- Long ID (16~64) >> Short ID (4), and TIGER (autoregressive+RQ) OOM under long ID
- independent projection head >> shared head >> no head
- Graph decoding is 3x better than no graph constraints

### Association with the current project

- `ARCHITECTURE.md` has been clarified: "Parallel tokenizer must be used, residual encoding cannot be used"
- Directly solve the problem of "residual coding can never be thought of"
- FAISS has ready-made `faiss.OPQMatrix` + `faiss.ProductQuantizer` implementation
- RPG open source code can be directly referenced

### Experimental Design Draft

**Two-stage experiment**:

#### Phase 1: OPQ quantitative quality (intrinsic metrics only)

Verify the quantification quality of OPQ on our 5M item / 1024D embedding.

**Configuration** (1024D embedding, the main configuration of the benchmark RPG paper):

| 方案 | 子向量维度 | token 数 m | 词表大小 M | 编码空间 |
|------|-----------|-----------|-----------|----------|
| A | 128D | 8 | 256 | 256^8 |
| B | 64D | 16 | 256 | 256^16 |
| C | 32D | 32 | 256 | 256^32 |

**Compared to baseline**: RKMeans 3 layers x 1024 clusters (EXP-001 final config, collision=1.75%)

**Evaluation metrics**: recon_loss, collision_rate, exclusivity, entropy, cluster_balance

**Note**: collision_rate may have different meanings under OPQ - the ID space of 8~32 tokens is much larger than 3 tokens, and collision should be extremely low. Focus on whether recon_loss is better than RKMeans.

#### Phase 2: Parallel Prediction Model + Graph Decoding

New implementation required:
1. Parallel prediction model: replace the current AutoregressiveNTPModel, each digit has an independent MLP head
2. Graph construction: SID similarity graph (top-k neighbors per node)
3. Graph decoding: seed sampling → graph propagation → scoring → iteration

### Key questions

1. **Selection of vocabulary size M**: RPG fixed M=256. Our RKMeans uses 1024 per layer. Need to verify the difference of M=256 vs M=1024 under OPQ.
2. **Graph construction cost**: The SID similarity graph construction of 5M items requires O(N²) or approximate ANN, and requires evaluation of memory and time.
3. **Relationship with OneRec**: OneRec uses 3 token x 8192 parallel tokenizer, and RPG uses 16~64 token x 256. The tradeoff between the two parallel schemes needs to be clarified — OneRec is suitable for autoregression (short sequences), and RPG is suitable for parallel prediction (long sequences + graph decoding).

---

## IDEA-fsq-scale-0: FSQ Hidden Adaptive — Match Embedding Dimension

**Priority**: P1 — Direct actionable improvements discovered in EXP-043
**Status**: Pending experiment, planned as EXP-045 or independent tokenizer experiment

### Background and root causes

EXP-043 entropy analysis reveals: FSQ MLP hidden=64 is designed for Qwen3-0.6B (1024D) embedding. When the embedding dimension increases, the dimension of the residual vector increases simultaneously, but the FSQ bottleneck remains unchanged, resulting in serious loss of L2 layer information:

| Embedding | Residual Dim | FSQ hidden | L2 entropy | FSQ 有效槽位 | Collision |
|-----------|-------------|-----------|-----------|------------|---------|
| Qwen3-0.6B | ~1024D | 64 | 10.58 bits (91.2%) | ~1500 | **0.49%** |
| Qwen3-4B | ~2560D | 64 | 8.10 bits (78.7%) | ~275 | 2.76% |
| Qwen3-8B | ~4096D | 64 | 7.17 bits (71.6%) | ~145 | 5.44% |

By inverting the S-tier + M-tier two-point scaling law, the irreducible floor PPL of each SID:
- 0.6B: 12.46, 4B: **11.78 (optimal)**, 8B: 12.26 (worse than 4B, caused by L2 collapse)

### Core Assumptions

FSQ hidden should be scaled proportionally to the embedding dimension so that L2 entropy remains at ≥90% utilization:
```
h_optimal ≈ k × emb_dim^α (empirical formula, to be fitted)
```
Initial guess: 0.6B(1024D)→h=64, 4B(2560D)→h=128~160, 8B(4096D)→h=256

### Experimental design

**Variable**: FSQ hidden h ∈ {64, 128, 256, 512} × embedding model ∈ {0.6B, 4B, 8B}

However, the cost of the full 3×4=12 groups is too high (each group needs to rebuild the SID cache + retrain the NTP). **Suggested Economic Plan**:

**Phase 1 — Pure tokenizer evaluation (no NTP required, very low cost)**:
- Fixed 0.6B SID, sweep h ∈ {32, 64, 128, 256}: confirm that h=64 is the optimal point of 0.6B
- Fixed 4B SID, sweep h ∈ {64, 128, 256}: find the minimum h with L2 entropy ≥90%
- Fixed 8B SID, sweep h ∈ {64, 128, 256, 512}: Same as above
- Evaluation indicators: L2 entropy (bits + utilization), collision rate, number of FSQ effective slots

**Phase 2 — NTP end-to-end verification (only run the optimal h found in Phase 1)**:
- 4B SID with h=optimal → S-tier NTP → compared with exp043-s-4b (R@500=64.3%)
- 8B SID with h=optimal → S-tier NTP → compared with exp043-s-8b (R@500=64.7%)
- Goal: Verify whether the floor PPL really drops after L2 entropy is restored, and whether R@500 exceeds the current optimal of 4B

### Experience formula target

Through Phase 1 data, fit:
```
h_min_for_L2_util_90% = f(emb_dim)
```
If linear: `h ≈ emb_dim / 16` (0.6B: 64, 4B: 160, 8B: 256)
If the root sign: `h ≈ 2 × sqrt(emb_dim)` (0.6B: 64, 4B: 101, 8B: 128)

Finally, a universal h selection formula is given across embedding scales to avoid adjusting parameters one by one.

### Change files

- `model/rkmeans.py` / `model/fsq.py` — `fsq_mlp_hidden` parameter is supported, just change the config value
- `experiments/scripts/exp-026-sid.sh` (or create new `exp-045-fsq-scale.sh`)

---

## IDEA-sid-2: Balanced KMeans

**Priority**: ~~P1~~ → P2 (after NTP)
**Source**: 3.1.2.1 (mentioned by OneRec Paper)
**Status**: Pending, downgraded

> **Reason for downgrade (2026-04-15)**: KMeans Gini=0.31 for the first two layers, balanced assignment can improve codebook utilization, and collision may drop from 10.7% to ~7-8%. But EXP-008 proves that collision is not the core indicator (OPQ collision 0.06% is the worst behavior), and the benefits are uncertain. Wait for NTP end-to-end Recall@K to be released before deciding whether to invest.
>
> **NTP phase update (2026-04-17)**: EXP-015 scaling law shows irreducible loss a=2.522 determined by tokenizer 32-bit encoding. Improving codebook utilization (Balanced KMeans) may slightly reduce collision but cannot break through the bit bottleneck. The long-term value will be re-evaluated after breaking through the 32-bit upper limit (more tokens/larger codebook).

### Core Idea

Use balanced KMeans to replace standard KMeans to force each cluster to be evenly sized and improve codebook utilization.

### Association with the current project

- There is still room for optimization in cluster_balance (Gini) in EXP-001
- Very low implementation cost and can be quickly verified
- The article also mentioned that "both the original vector and the residual vector can be normalized" — currently only normalize layer 0 input

### Experimental Design Draft

**Variables**:
- Standard KMeans vs Balanced KMeans
- Residual normalization: L0 only (currently) vs normalize for every layer

**Note**: EXP-001 conclusion "normalize_residuals only for layer 0" may need to be revalidated because balanced assignment may not be used at that time

### Key questions

1. FAISS does not directly support balanced KMeans, you need to use `faiss-contrib` or self-implementation
2. Will the normalize residuals at each layer conflict with the EXP-001 conclusion?

---

## IDEA-gr4ad-0: Multi-Granularity Multi-Resolution RQ-KMeans (MGMR)

**Priority**: ~~P1~~ → P2 (after NTP)
**Source**: GR4AD §UA-SID, Table 2
**Status**: Pending, downgraded

> **Reason for downgrade (2026-04-15)**: MLP-FSQ has been confirmed as the winner. The non-large codebook (L1=4096 L2=1024) is a fine-tuning of the first two layers of KMeans, with medium benefits. The implementation cost is extremely low (`ResidualQuantizationMultiGPU` already supports independent `n_clusters` per layer), but priority is given to promoting NTP.

### Core Idea

GR4AD proposes an MGMR coding scheme: (1) Multi-Resolution - the lower layer uses a large codebook to capture the dominant factors, and the high layer uses a small codebook to model the low-entropy residual (such as 16384→4096→1024); (2) Multi-Granularity - the last layer uses hash mapping of non-semantic features (item ID, account ID) to replace clustering and directly eliminate collisions. The combination of the two reduces the collision rate from 3.54 to 1.07 and increases the codebook utilization from 0.10‰ to 0.34‰.

### Association with the current project

- **Direct benchmarking EXP-001**: Currently using the same large codebook 3×1024, collision=1.75%. MGMR’s unequal large codebooks (such as 4096→1024→256) are a zero-cost improvement
- `ResidualQuantizationMultiGPU` in `model/rkmeans.py` already supports independent `n_clusters` parameter for each layer, just pass in different values
- The hash layer idea of Multi-Granularity is complementary to IDEA-sid-0 (OPQ) - if the last layer uses hash to ensure uniqueness, the previous layers can safely use coarser semantic clustering
- The collision_rate and codebook utilization indicators of `eval/evaluator.py` can be directly used for evaluation

### Experimental Design Draft

**Variable 1 — Multi-Resolution Codebook Configuration**:

| Config | L1 | L2 | L3 | 总编码空间 |
|------|------|------|------|-----------|
| Baseline (EXP-001) | 1024 | 1024 | 1024 | 10^9 |
| MR-A | 4096 | 1024 | 256 | 10^9 |
| MR-B | 2048 | 1024 | 512 | 10^9 |
| MR-C | 4096 | 2048 | 512 | 4×10^9 |

**Variable 2 — Multi-Granularity hash layer**:
- For each MR configuration, test the last layer replaced with hash(item_id) % vocab_size
- Need to add hash layer option in `model/rkmeans.py`

**Evaluation**: collision_rate, recon_loss, codebook_utilization, cluster_balance (Gini), sid_prediction Hit@K

**Implementation Cost**: Low. MR only needs to modify the config parameters; MG hash layer needs to add about 20 lines of code in `ResidualQuantizationMultiGPU`

### Key questions

1. The `vocab_size` of the NTP model under different large codebooks needs to be different for each layer - the `AutoregressiveNTPModel` of `metrics/sid_prediction.py` currently assumes a unified vocab_size and needs to be adapted.
2. Hash layer vocab_size selection: How is it related to the number of items? Too small and there will still be collisions, too big and sparse
3. Interaction with IDEA-sid-2 (Balanced KMeans): balanced assignment is more critical in large codebooks (4096+)

---

## IDEA-onemall-5: OneMall Validation EXP-003 Direction (ResKmeans + Learned FSQ)

**Priority**: ~~P1~~ → ✅ Complete
**Source**: OneMall §3.1.3 + §4.5 Tokenizer Strategy
**Status**: ✅ MLP-FSQ h=64 confirmed as winner (EXP-003 → EXP-008)

> **Complete Record (2026-04-15)**: EXP-003 verified that the MLP-FSQ solution is feasible. EXP-008 compared it with OPQ through FORGE proxy metrics under equal bits conditions. MLP-FSQ h=64’s semantic_neighbor_HR=0.078 decisively won. This scheme becomes the current tokenizer baseline.

### Core Idea

OneMall's tokenizer ablation directly verified our EXP-003 direction:

| 方案 | Conflict Rate | Exclusive Rate | HR@50 |
|------|--------------|----------------|-------|
| 3-layer ResKmeans | 36% | 86% | 33.9% |
| 2-layer ResKmeans + 1-layer FSQ | **11%** | **95%** | **35.4%** |

Key difference: OneMall's FSQ layer uses **"binary 16-bit MLP"** to quantize the residual embedding to 4096 codes — which is exactly what our `LearnedFSQLayer` (MLP + STE) scheme does, not the failed PCA scheme of EXP-002.

EXP-002 failure reason (PCA 1024D→4-6D only retains 20-55% variance) is implicitly verified in OneMall: they use MLP directly instead of PCA.

### Association with the current project

- `LearnedFSQLayer` implemented (`model/fsq.py`)
- `ResKmeansFSQ` already supports `mlp` projection type (`model/rkmeans_fsq.py`)
- EXP-003 Designed but **not yet operational**
- OneMall results give clear expectations: the conflict rate should be further reduced from the current ~1.75%, and the exclusive rate should be increased

### Action recommendations

**Execute EXP-003 immediately**, refer to OneMall parameters:
- FSQ codebook size = 4096 (consistent with OneMall)
- MLP hidden sizes: {64, 128, 256} (already in EXP-003 design)
- Train for 50 epochs with STE
- Pay special attention to changes in conflict rate and exclusive rate

### Key questions

1. The specific implementation details of OneMall's "binary 16-bit" are unclear - is it just 16 binary bits that are directly processed into 2^16=65536 codes and then truncated to 4096? Or FSQ-style multi-level quantization?
2. Our FSQ level config `4d_4096: [8,8,8,8]` generates 4096 codes, consistent with OneMall

---

## IDEA-pit-0: Co-generative Dynamic Tokenizer (PIT)

**Priority**: ~~P1~~ → P2 (after NTP)
**Source**: PIT (Kuaishou, arxiv 2602.08530, Feb 2026)
**Status**: Pending, downgraded

> **Reason for downgrade (2026-04-15)**: The core of PIT is tokenizer+NTP joint training, and the prerequisite is that there is NTP baseline first. MLP-FSQ is currently purely unsupervised (reconstruction loss). Adding behavioral signal joint training may further improve it, but the complexity is high. Wait for the NTP baseline to come out before you evaluate whether it is worth the investment.

### Core Idea

PIT proposed **Co-generative Architecture**: tokenizer and NTP models are no longer trained in stages, but implement end-to-end joint training through **Collaborative Signal Alignment** and **Co-evolution Learning**. Core innovation:

1. **Collaborative Signal Alignment**: Inject the collaborative filtering signal directly into the tokenization process so that the generated SID has its own behavioral semantics
2. **Co-evolution Learning**: tokenizer and recommender enhance each other in a unified training cycle to avoid the two-stage break of "build index first and then train"
3. **One-to-Many Beam Index**: Each item can be assigned multiple SID token sequences to improve recall and robustness.

Kuaishou large-scale online A/B: **App Stay Time +0.402%**.

### Association with the current project

- The current tokenizer (RKMeans/OPQ) is completely offline and decoupled from the NTP model
- The core problem of Co-evolution: every time the tokenizer is updated, all SIDs change, and the NTP model needs to be relearned - PIT claims to have solved this stability problem
- **One-to-Many Beam Index** has direct inspiration for us: mapping an item to multiple SIDs can alleviate the collision problem
- Related to IDEA-sid-1 (collaborative signal enhancement): both inject collaborative signals, but PIT is dynamically injected during tokenizer training, and sid-1 is injected after pre-training embedding

### Experimental Design Draft

**Phase 1 — One-to-Many SID Mapping**:
- Currently each item has only one SID. Allow OPQ/RKMeans to assign top-k (k=2~5) recent codeword combinations to each item
- During NTP training, if the target is any one of k SIDs, it is considered correct (multi-label CE)
- Evaluate recall improvement + collision mitigation

**Phase 2 — Co-evolution** (High Complexity):
- Rerun tokenizer periodically (every N epochs) during NTP training
- Use the hidden layer representation of the NTP model as one of the tokenizer input signals
- A stability mechanism needs to be designed to avoid drastic changes in SID

### Key questions

1. One-to-Many mapping increases the ambiguity of NTP training - there are multiple "correct answers" for an item, how does the model converge?
2. Computational cost of Co-evolution: overhead of re-tokenizing 5M items every N epochs
3. Stability of SID changes: How to ensure continuity between old and new SIDs

---

## IDEA-forge-0: SID Proxy Evaluation Metrics + Offline Pretraining

**Priority**: ~~P1~~ → ✅ Complete
**Source**: FORGE (Alibaba/Taobao, arxiv 2509.20904, Sep 2025)
**Status**: ✅ semantic_neighbor_hit_rate has been implemented and verified as a decisive indicator

> **COMPLETE RECORD (2026-04-15)**: `eval/evaluator.py` has implemented semantic_neighbor_hit_rate. EXP-008 relies on this metric to select the MLP-FSQ winner, proving that proxy metric can effectively evaluate SID quality without training NTP. The Offline Pretraining part is left to the NTP stage.

### Core Idea

FORGE released a large-scale benchmark of Taobao 14B interaction + 250M products, and proposed two key technologies:

1. **SID Proxy Metrics**: Two new metrics are positively related to downstream recommendation performance and **can assess SID quality without training a GR model**. This solves the pain point of "every time you change the tokenizer, you have to run the complete NTP training to know whether it is good or not."
2. **Offline Pretraining Schema**: Use offline pretraining to halve online convergence time

Taobao "Guess You Like" online verification: **Trading volume +0.35%**.

### Association with the current project

- Current assessment of SID quality requires: (1) running RKMeans/OPQ → (2) training NTP → (3) watching Hit@K. The whole process takes several hours
- If there are proxy metrics, they can be evaluated directly after step (1), **accelerating the tokenizer super parameter search by 10x+**
- Directly related to EXP-004 (OPQ): Quickly assess SID quality of different m/M configurations
- `eval/evaluator.py` already has intrinsic metrics such as collision_rate and exclusivity. You can add FORGE's proxy metrics on this basis.

### Experimental Design Draft

- Get the proxy metrics defined in the FORGE paper (you need to read the full text of the paper)
- Implemented in `eval/evaluator.py`
- Verify: whether proxy metrics are related to NTP Hit@K (backtested on existing EXP-001/004 data)

### Key questions

1. The specific definition of FORGE’s proxy metrics requires reading the full text of the paper to obtain it.
2. Our data size (5M items) is much smaller than FORGE (250M items). Does the correlation of proxy metrics still hold?

---

## IDEA-quasid-0: Collision-Qualified SID Learning (Hamming-Guided Repulsion)

**Priority**: ~~P1~~ → P2 (after NTP)
**Source**: QuaSID (Kuaishou E-commerce, arxiv 2603.00632, Feb 2026)
**Status**: Pending, downgraded

> **Reason for downgrade (2026-04-15)**: MLP-FSQ collision 10.7%, Hamming repulsion may reduce harmful collisions. However, EXP-008 proves that low collision does not equal high behavioral quality (OPQ collision 0.06% is the worst), and the proportion of harmful collisions needs to be confirmed first. Preliminary: Phase 1 analysis - semantic distance distribution of collision pairs. Wait for the NTP end-to-end data to come out before making a decision.
>
> **NTP stage update (2026-04-17)**: EXP-014 ENTP negative sample export found an L0 layer collision problem - some negative samples share L1 cluster tokens with positive samples, causing ENTP loss to fail at coarse level. This verifies the core premise of QuaSID (that harmful collisions exist and affect the training signal). But the priority is still P2: promote ENTP loss integration first, and then decide whether Hamming repulsion is needed to solve it from the tokenizer side.

### Core Idea

QuaSID discovered that SID collision problems are not homogeneous: some collisions are truly harmful "semantic conflicts" (semantically unrelated items get identical SIDs), and some are benign (data redundancy). QuaSID proposes two mechanisms:

1. **Hamming-guided Margin Repulsion**: Use the Hamming distance between SIDs as a collision severity indicator to push conflicting item pairs with low Hamming distance away in the encoder space. Thrust is proportional to collision severity
2. **Conflict-Aware Valid Pair Masking**: Automatically filter "benign collisions" (protocol-induced benign overlaps) and only apply repulsion to truly harmful collisions

Extra: Added **dual-tower contrastive objective** to inject synergistic signals in tokenization.

**Plug-and-play**: repulsion loss can enhance any SID learning framework.

Kuaishou e-commerce online A/B (5% traffic): **ranking GMV-S2 +2.38%, cold start orders +6.42%**.

### Association with the current project

- Currently EXP-001's collision_rate = 1.75%, we are already tracking collisions in `eval/evaluator.py`
- The insight of QuaSID is: **Not all collisions should be treated equally**. Currently we only look at the collision count and do not differentiate between the severity of the collision.
- Hamming distance calculation has zero cost - SID is already a discrete code, direct comparison
- **Plug-and-play**: Can be directly added to the training of EXP-007 (contrastive embedding fine-tune)
- Complementary to IDEA-sid-1 (Cooperative Signal Enhancement): sid-1 improves embedding itself, QuaSID imposes collision constraints in SID space

### Experimental Design Draft

**Phase 1 — Collision Severity Analysis**:
- For existing SID assignment (EXP-001/EXP-004), calculate the semantic distance (embedding cosine) of all collision pairs
- Classification: high cosine = benign collision (the semantics are close), low cosine = harmful collision
- Quantification: What is the proportion of harmful collisions? If it is low → limited benefits

**Phase 2 — Hamming Repulsion Loss**:
- Add repulsion loss to embedding fine-tune (EXP-007 process)
- Apply margin loss to item pairs with Hamming distance < threshold
- L_repulsion = max(0, margin - cosine(e_i, e_j)) * severity_weight(hamming_dist)

**Evaluation**: collision_rate, harmful collision ratio, embedding_hit_rate

### Key questions

1. The collision rate under RKMeans (3 layers x 1024) is already only 1.75%, and the repulsion benefit may be limited.
2. OPQ (8~32 token x 256) has a lower collision rate → this idea may be more valuable for the RKMeans route
3. Gradient conflicts between Repulsion and contrastive loss: one needs to be pushed away, the other needs to be brought closer

> **Follow-up work (2026-04-28)**: The same Kuaishou team published AdaSID (arxiv 2604.23522), which upgraded the fixed Hamming threshold to a two-stage adaptive regulation (semantic gating + load adaptation + progress scheduling), surpassing QuaSID in Toys/Beauty. See IDEA-adasid-0 for details.

---

## IDEA-r3vae-0: Reference Vector-Guided SID Generation (stable training + evaluation indicators)

**Priority**: ~~P1~~ → P2 (after NTP)
**Source**: R3-VAE (arxiv 2604.11440, Apr 2026)
**Status**: Pending, downgraded

> **Reason for downgrade (2026-04-15)**: MLP-FSQ does not have a codebook collapse problem, and the training stability of Reference Vector has limited value. The Semantic Cohesion + Preference Discrimination metric complements the FORGE proxy, but its primary value lies in evaluation rather than improvement. Wait until NTP and think about it.

### Core Idea

R3-VAE solves two fundamental problems of VQ-based SID generation:

1. **Unstable training**: STE (straight-through estimator) gradient propagation is insufficient + initialization sensitivity → **Reference Vector** serves as a semantic anchor for stable training
2. **High evaluation cost**: Evaluating SID quality requires training a complete GR model + A/B test → Propose two standalone metrics **Semantic Cohesion** and **Preference Discrimination**, which can be directly evaluated after SID generation

The Reference vector + dot-product rating mechanism can also prevent **codebook collapse** (dead codebook problem).

News Recommendation Platform Online A/B: **MRR +1.62%**. As CTR model item ID alternative: **Cold Start +15.36%**.

### Association with the current project

- There is currently no codebook collapse problem in RKMeans training (KMeans is used for each layer), but it will be encountered when switching to VQ-VAE or learned quantization.
- **Semantic Cohesion + Preference Discrimination Metrics** are similar to IDEA-forge-0 (SID Proxy Metrics) - both can evaluate SID quality without training NTP
- If you integrate the two, you can build a **complete SID quality assessment toolkit**: before training (FORGE proxy) + after training (R3-VAE metrics)
- The cold start +15.36% result is inspiring to us: SID can be used as the item feature of the CTR model

### Experimental Design Draft

**Phase 1 — Implementing R3-VAE evaluation metrics**:
- Implement Semantic Cohesion and Preference Discrimination in `eval/evaluator.py`
- Backtest on existing EXP-001/EXP-004 SID assignments
- Verification: Whether these two indicators are related to NTP recall

### Key questions

1. For the definition of specific indicators, you need to read the full text of the paper.
2. Deduplication/merging proxy metrics with IDEA-forge-0

---

## IDEA-unirec-1: Capacity-Constrained SID (Exposure-Weighted RQ Penalties)

**Priority**: P2 (after NTP)
**Source**: UniRec (Alibaba, arxiv 2025, KDD 2025)
**Status**: Pending, downgraded

> **Reason for downgrade (2026-04-15)**: MLP-FSQ has been confirmed as the tokenizer winner. Capacity constraint mainly solves token collapse (a small number of codebook entries monopolize a large number of items). It is consistent with the goal of IDEA-sid-2 (Balanced KMeans) and can be combined for evaluation. We will decide after NTP end-to-end Recall@K comes out.

### Core Idea

UniRec discovered a serious **token collapse** problem during RQ tokenizer training: some codebook entries were over-allocated (high-exposure items dominate the cluster center), resulting in poor quality SID representation of long-tail items. Proposed **Capacity-Constrained SID Learning**:

1. **Exposure-Weighted Assignment Penalty**: Penalize codebook entries that have been assigned a large number of high-exposure items, forcing the tokenizer to use the codebook more evenly
2. **Residual Capacity Tracking**: Each codebook entry maintains a capacity counter. After exceeding the threshold, the assignment cost increases linearly.
3. **Two-stage training**: Standard RQ training convergence first, then capacity constraint fine-tune

Effect: The codebook utilization rate increased from ~60% to ~95%, and the SID discrimination of long-tail items was significantly improved.

### Association with the current project

- The current KMeans Gini=0.31 of the first two layers of MLP-FSQ has an uneven codebook problem.
- Capacity constraint has the same goal as Balanced KMeans (IDEA-sid-2) but different methods: sid-2 enforces balance during clustering, unirec-1 imposes soft constraints in loss
- Low implementation cost: add penalty term in the assignment step of RQ training
- Complementary to IDEA-quasid-0 (Hamming Repulsion): quasid-0 handles harmful collisions, unirec-1 handles codebook utilization

### Experimental Design Draft

**Merge evaluation with IDEA-sid-2**:
- Balanced KMeans (hard constraints) vs Capacity Penalty (soft constraints) vs a combination of both
- Evaluation: codebook utilization (Gini), collision_rate, semantic_neighbor_HR
- Validation on MLP-FSQ architecture: impose capacity constraint on first two layers of KMeans

### Key questions

1. The third layer of MLP-FSQ, FSQ, is naturally evenly distributed, and the constraint is mainly targeted at the first two layers of KMeans.
2. Redundancy with Balanced KMeans: Both solve codebook utilization, you may only need to choose one
3. Exposure weighting requires exposure data. Does the current item metadata contain exposure?

---

## Priority summary

| 优先级 | ID | Direction | Status |
|--------|-----|------|------|
| ~~P0~~ | ~~IDEA-sid-0~~ | ~~OPQ 并行语义 ID~~ | ❌ 关闭 (EXP-008: semantic_neighbor_HR 输 MLP-FSQ) |
| ~~P1~~ | ~~IDEA-onemall-5~~ | ~~RKMeans+FSQ~~ | ✅ 完成，MLP-FSQ h=64 确认赢家 |
| ~~P1~~ | ~~IDEA-forge-0~~ | ~~SID Proxy Metrics~~ | ✅ 完成，semantic_neighbor_hit_rate 已实现 |
| P2 | IDEA-sid-2 | Balanced KMeans | 待定，NTP 后 (collision 非核心Metric) |
| P2 | IDEA-gr4ad-0 | MGMR 不等大码本 | 待定，NTP 后 (微调收益，优先推 NTP) |
| P2 | IDEA-quasid-0 | Hamming Repulsion | 待定，NTP 后 (需先确认有害碰撞占比) |
| P2 | IDEA-pit-0 | Co-gen Tokenizer | 待定，NTP 后 (前置: NTP baseline) |
| P2 | IDEA-r3vae-0 | Reference Vector SID | 待定，NTP 后 (主要价Value在EvaluationMetric) |
| P2 | IDEA-unirec-1 | Capacity-Constrained SID | 待定，NTP 后 (与 sid-2 合并Evaluation) |
| P2 | IDEA-flexcode-0 | 双码本 CF+Semantic + MoE 分配 | 待定，NTP 后 (需 CF model) |
| P2 | IDEA-crab-0 | Codebook Rebalancing 去偏 | 待定，NTP 后 (post-hoc Method) |

---

## IDEA-geogr-0: Geo-aware SID Tokenization (Co-visited POI comparative learning)

**Priority**: P2
**Source**: GeoGR, Alibaba/AMAP (arxiv 2602.10411)
**Status**: To be discussed

### Core Idea

Alibaba Map's GeoGR proposes geo-aware SID tokenization for POI recommendation: using geographically constrained co-visited POI pairs for comparative learning, plus iterative refinement, to generate SIDs that capture spatio-temporal collaborative semantics. Key insight: The semantics of a POI depend not only on the content (restaurant/store), but also on geography and time pattern (nearby restaurant during lunch vs. outlying attraction on weekends). Cooperate with multi-stage LLM training (template-based CPT + autoregressive SFT) to achieve end-to-end POI generation. Deployed in Amap, serving millions of users.

### Association with the current project

- Consistent with the idea of IDEA-oneloc-3 (side-info fusion) but more specific: instead of generalizing side-info, it specifically targets spatiotemporal signals
- Co-visited POI contrastive learning can be generalized to "co-consumed item contrastive learning": using jointly consumed item pairs to strengthen the behavioral semantics of SID
- The current MLP-FSQ tokenizer is based on text embedding and lacks behavioral co-signaling → co-consumed contrastive may fill this gap
- Multi-stage LLM training (CPT + SFT) is consistent with IDEA-plum-0

### Experimental Design Draft

**Phase 1 — Co-consumed Item Contrastive Loss for Tokenizer**:
- Added in tokenizer training (or embedding fine-tune): SIDs of item pairs frequently consumed by the same user should share more prefixes
- Implementation: Add co-visit affinity penalty in the assignment step of RQ/FSQ training
- Evaluation: semantic_neighbor_HR (which itself measures behavioral neighborhood retention)

### Key questions

1. The current data has no geographical information and needs to be generalized into co-consumption signals.
2. Partially overlaps with IDEA-sid-1 (collaborative signal enhancement embedding), but sid-1 is a direct fine-tune embedding, and this IDEA is injected in the tokenizer layer
3. Consider tokenizer improvements in the post-NTP stage → P2

---

## IDEA-onevision-0: Visually aligned residual quantization (VRQ) + dynamic pruning

**Priority**: P2
**Source**: OneVision, Kuaishou (arxiv 2510.05759)
**Status**: To be discussed

### Core Idea

Kuaishou OneVision proposes VRQ (Vision-aligned Residual Quantization) for visual search: aligning vastly different visual representations of the same object across multiple viewing angles, while retaining the unique features of the product, and generating semantic ID for generative retrieval. With multi-stage semantic alignment (preserving visual similarity prior + incorporating user personalized preferences) and dynamic pruning (increasing reasoning efficiency by 21%). Online A/B: CTR +2.15%, CVR +2.27%, order volume +3.12%.

### Association with the current project

- VRQ's multi-perspective alignment idea can be generalized to multi-modality: the text description, title, and comments of the same item may have large semantic differences and need to be aligned before quantification.
- Dynamic pruning (21% efficiency improvement) has direct value on the inference side: dynamically adjust the SID sequence length according to the input difficulty
- The current project is text embedding → SID, OneVision is visual embedding → SID, and the core pipeline is the same
- The online A/B effect is significant (CTR +2.15%), verifying the feasibility of the end-to-end generative search architecture

### Experimental Design Draft

Visual modality data is required and is not applicable to the current project. However, dynamic pruning ideas (also involved in IDEA-stamp-0) can be referenced together.

### Key questions

1. The current project has no visual mode → VRQ itself is not directly available
2. The idea of dynamic pruning has been covered by IDEA-stamp-0
3. The main value lies in verifying the online effect of "generative search end-to-end architecture"

---

## IDEA-mmq-0: Shared-proprietary multi-modal hybrid quantization Tokenizer

**Priority**: P2
**Source**: MMQ, Alibaba (arxiv 2508.15281, WSDM 2026)
**Status**: To be discussed

### Core Idea

Ali MMQ proposes a two-stage multi-modal tokenizer: (1) Shared-Specific Tokenizer — multi-expert architecture, modality-specific experts capture unique information of each modality, modality-shared experts capture cross-modal commonality, add orthogonal regularization; (2) Behavior-Aware Fine-Tuning — use downstream recommendation targets to dynamically adapt SID representations, while using multi-modal reconstruction loss to maintain modal information without loss. Supports two downstream tasks: generative retrieval and discriminative ranking. WSDM 2026 + online A/B validation.

### Association with the current project

- The current tokenizer only uses text embedding (Qwen3), and MMQ’s multi-modal framework provides an expansion route.
- Shared-Specific Expert architecture can be generalized: replace "modal" with "signal type" (semantic signal vs collaborative signal)
- Behavior-Aware Fine-Tuning has the same idea as IDEA-onemall-3 (attribute enhancement contrastive): use downstream task signals to in turn adjust the tokenizer
- Orthogonal regularization prevents expert degradation and has reference value for MoE-related IDEA (IDEA-onemall-4)

### Experimental Design Draft

**Applicable when multimodal data is available:**
- Shared expert: learn cross-modal commonalities (text + image jointly describe item semantics)
- Specific expert: learn unique information of single modality (attribute description of text vs visual style of image)
- Behavior-aware fine-tune: Use NTP recall target to fine-tune the quantization layer after freezing the expert

### Key questions

1. Currently there is no multimodal data → direct experiment is not possible
2. The Shared-Specific idea can be tested in a single mode (semantic vs collaborative dual expert), but its value has not been verified.
3. Consider tokenizer extension in the post-NTP stage → P2

---

## IDEA-flexcode-0: Dual codebook (CF + Semantic) + MoE dynamic allocation

**Priority**: P2 (after NTP)
**Source**: FlexCode, Roblox (arxiv 2511.20673, Nov 2025)
**Status**: To be discussed

### Core Idea

FlexCode found that a single codebook simultaneously encodes semantics and synergy signals, resulting in "representation entanglement" - head items are diluted by semantics, and tail items are dominated by noisy synergy signals. Propose **Dual Codebook + Adaptive Allocation**:

1. **Semantic Codebook (C_SEM)**: RQ-VAE quantifies text/visual embedding and captures content semantics
2. **Collaborative Codebook (C_CF)**: SASRec-style learns collaborative embedding for co-purchase/co-view sequences, and then quantizes it with RQ-VAE
3. **Cross-Codebook Alignment (CCA)**: InfoNCE compares loss to align the reconstruction embedding of the two codebooks to prevent spatial drift
4. **MoE Router**: Routing based on item statistical characteristics (log(popularity), age, sparsity, uncertainty), head items → more CF tokens, tail items → more semantic tokens
5. **Fixed total token budget L**: L_CF(i) + L_SEM(i) = L, differentiable allocation is achieved through sigmoid mask

**Core results**:
- KuaiRand: NDCG@10 0.0632, +8.0% than URL, +42% than TIGER
- Industrial (1.5M+ users): NDCG@10 +13.2% over SASRec baseline
- Tail items NDCG@10 +11.3% (maximum improvement), Head +3.0%
- FlexCode-Fix (50/50 static) already beats baselines → dual codebooks are valuable in their own right
- MoE dynamic allocation additional contribution 12.5% (KuaiRand)

### Association with the current project

- **Direct response to user concerns**: NTP lacks cross-user collaborative signal → FlexCode injects CF in the tokenizer layer
- The current MLP-FSQ tokenizer only uses text embedding (pure semantics), corresponding to FlexCode's "SID Only"
- The dual codebook solution does not change the NTP model architecture - only the SID generation method is changed, and NTP still performs token prediction.
- The same goal as IDEA-sid-1 (collaborative signal enhancement embedding) but the solution is more systematic: sid-1 is direct fine-tune text embedding, FlexCode is independent codebook + dynamic fusion
- Complementary to IDEA-pit-0 (co-generative tokenizer): PIT is joint training tokenizer+NTP, FlexCode is independent training dual codebook
- MoE Router related to IDEA-onemall-4 (MoE Load Balancing)

### Experimental Design Draft

**Phase 1 — CF Codebook Construction**:
- Train on behavior sequences with SASRec-style model → get item collaborative embedding
- Do RQ-VAE on CF embedding → generate CF tokens
- concat with existing semantic tokens (MLP-FSQ) → dual SID

**Phase 2 — MoE Router**:
- Routing by item interaction frequency: the first 20% of items are divided into CF tokens, the last 80% are divided into semantic tokens
- Fixed total budget L=3, test {(2CF,1SEM), (1CF,2SEM), (MoE dynamic)}

### Key questions

1. CF embedding requires training a SASRec model → additional training cost
2. Is the head/tail distribution in our 5M items comparable to FlexCode’s KuaiRand/Industrial?
3. After double SID concat, the NTP input length doubles → Token Merger (IDEA-genrec-1) is required.
4. Consider tokenizer improvements in the post-NTP stage → P2

---

## IDEA-crab-0: Splitting and debiasing overly popular Tokens (Codebook Rebalancing)

**Priority**: P2 (after NTP)
**Source**: CRAB, Walmart (arxiv 2604.05113, Apr 2026)
**Status**: To be discussed

### Core Idea

CRAB found that the popularity bias of GeneRec is rooted in **codebook imbalance**: popular items with similar semantics are mapped to the same token. After accumulating interaction frequency, the token becomes an "overly popular token", and model training tends to generate these tokens → amplifying the popularity bias (7.2% higher than SASRec).

Proposed **post-hoc debiasing** (no need to retrain from scratch):
1. **Token split**: Identify the top-5% popular tokens, and redistribute their child tokens to M new parent tokens through regularized K-means
2. **Balanced Loss**: Constrain the popularity of new tokens after splitting to be uniform: L_bal = Σ(P(c_k(m)) - P_avg)²
3. **Hierarchical Semantic Regularizer**: tree-structure-aware loss promotes sibling tokens to represent consistency while helping new tokens transfer knowledge from semantic neighbors
4. LoRA efficient fine-tuning (only 1/11 training time)

**Core results**:
- Industrial dataset: MGU@10 decreased by 16.5% (popularity bias), HR@10 remained the same
- Splitting the middle layer (Level B) works best — "Hourglass phenomenon": excessive concentration of semantics in the middle layer
- 10% splitting ratio is optimal, excessive splitting destroys semantic integrity
- Efficient: only 0.28h (vs RW 3.11h, D2LR 2.75h)

### Association with the current project

- The current MLP-FSQ first two layers KMeans Gini=0.31, there is imbalance
- CRAB is a **post-hoc** method → no need to change the tokenizer training, it can directly operate on the existing codebook
- Complementary to IDEA-sid-2 (Balanced KMeans): sid-2 enforces balancing during training, CRAB splits and repairs after training
- The same goal as IDEA-unirec-1 (Capacity-Constrained SID) but different methods: one is training constraints, the other is post-training repair
- "Hourglass Phenomenon" valuable insight: is our 3-layer SID middle layer (L2) also over-concentrated

### Experimental Design Draft

**Phase 1 — Token Popularity Analysis**:
- Count the popularity of tokens at each level of the current SID (the sum of the interaction frequencies of associated items)
- Visualize Gini + Top 5% token proportion
- Verify whether Hourglass phenomenon exists

**Phase 2 — Token split**:
- Split top-10% popular tokens (M=2~3)
- Use regularized K-means to maintain hierarchical structure
- LoRA fine-tunes the NTP model to adapt to the new codebook

### Key questions

1. The SID vocab increases after splitting → the embedding table of the NTP model needs to be expanded (new tokens need to be initialized)
2. We use RQ-KMeans (not RQ-VAE), the tree structure is strict → Eq.5 is directly applicable
3. Do codebook tuning in the post-NTP stage → P2

---

## IDEA-adasid-0: Adaptive Semantic-Qualified Collision Regulation

**Priority**: P2 (after NTP)
**Source**: AdaSID (Kuaishou E-commerce + UESTC, arxiv 2604.23522, Apr 2026)
**Status**: To be discussed

> **Relationship with IDEA-quasid-0**: Follow-up work of the same Kuaishou team. QuaSID uses Hamming distance for fixed threshold collision classification; AdaSID is upgraded to two-stage adaptive regulation—not only determines whether a collision is harmful, but also dynamically adjusts the penalty based on local congestion and training progress. AdaSID outperforms QuaSID across the board in Toys/Beauty (+5.2% average).

### Core Idea

AdaSID models SID collision regulation as a **two-stage adaptive process**:

**Stage 1 — Semantic-Adaptive Overlap Relaxation**:
- Calculate the encoder space cosine similarity of the collision pair
- Introducing **depth-aware semantic gate**: the deeper the collision (the greater the overlap depth), the stricter the relaxation threshold
  - Threshold vector η = [η₁ ≤ η₂ ≤ ... ≤ η_L], such as [0.18, 0.24, 0.30]
  - When sim_ij ≥ η_{o_ij}, the pair is exempt from repulsion (semantically compatible collision preservation)
  - When sim_ij < η_{o_ij}, the pair retains repulsion (harmful collision)
- **Key insight**: Shallow collisions (share 1-2 tokens) loose relaxation conditions; deep collisions (almost the same SID) only allow sharing with extremely high semantic similarity

**Stage 2 — Adaptive Pressure Allocation**:
- **Load-Adaptive Collision Strengthening (spatial dimension)**: Count the frequency of collision signatures (layer-wise overlap pattern) in mini-batch, and apply stronger repulsion in crowded areas
  - Collision signature κ_ij = [I(s¹_i=s¹_j), ..., I(s^L_i=s^L_j)]
  - Local collision load c_ij = Σ I(κ_uv = κ_ij)
  - The higher the load → the greater the strengthening factor (bounded monotonic function)
- **Progress-Adaptive Objective Rebalancing (time dimension)**: The collision loss is emphasized in the early stage of training, and the collaborative alignment loss gradually increases in the later stage of training.
  - λ_col(τ) = 1 - (1 - λ_min_col) · τ (attenuation)
  - λ_cf(τ) = λ_max_cf · τ (growth)
  - τ = clip((t - T_start) / (T_end - T_start), 0, 1)

**Total goal**: L = L_rec + L_rq + λ_col(τ) · L_ada_col + λ_cf(τ) · L_cf

### Experimental data

| 数据集 | Method | Recall@3 | NDCG@3 | Recall@5 | NDCG@5 |
|--------|------|----------|--------|----------|--------|
| Toys | QuaSID | 0.0195 | 0.0157 | 0.0273 | 0.0191 |
| Toys | **AdaSID** | **0.0214** | **0.0175** | **0.0281** | **0.0202** |
| Beauty | QuaSID | 0.0201 | 0.0155 | 0.0268 | 0.0186 |
| Beauty | **AdaSID** | **0.0205** | **0.0164** | **0.0275** | **0.0190** |

Ablation experiment (Beauty): SeAR (Semantic Relaxation) is removed → Recall@3 is reduced by 10.2%; PAR (Progress Scheduling) is removed → Recall@5 is reduced by 14.2%; LAS (Load Adaptation) is removed → Stable but slightly reduced.

**Kuaishou E-commerce Online A/B** (short video search, tens of millions of users):
- **GMV +0.98%, Orders +0.91%, GPM +1.16%**
- Offline ranking: Overall CTCVR AUC +0.05pp, Cold-start CVR AUC +0.08pp

### Association with the current project

- Current MLP-FSQ collision 10.7% — the collision rate is not low and there is room for optimization
- AdaSID's depth-aware semantic gate can be directly applied to our 3-layer SID: collisions at different depths are treated differently
- **Load-adaptive strengthening is particularly valuable**: Our L1 KMeans (1024 clusters) some clusters may be overcrowded, AdaSID automatically identifies crowded areas to strengthen penalties
- Progress-adaptive rebalancing can be implemented in the embedding fine-tune stage (if you take the IDEA-sid-1 route)
- Composable with IDEA-quasid-0: replaces QuaSID's fixed Hamming threshold with AdaSID's adaptive framework

### Experimental Design Draft

**Phase 1 — Collision semantic analysis** (same as quasid-0 Phase 1):
- Analyze the cosine similarity distribution of all collision pairs
- Stratified statistics based on overlap depth to verify the hypothesis that "deep collision pairs have higher semantic similarity"
- Calculate the proportion of harmful collisions (sim < number of pairs in threshold)

**Phase 2 — Adaptive Collision Loss**:
- Add AdaSID's two-stage loss to tokenizer embedding fine-tune
- Super parameters: depth-aware thresholds [η₁, η₂, η₃], f_max ∈ {2.0, 3.0}, schedule start and stop steps
- Evaluation: collision_rate, codebook utilization (entropy, min perplexity), semantic_neighbor_HR

### Key questions

1. We use RQ-KMeans (offline KMeans fit), not end-to-end training → AdaSID’s loss needs to be applied in the embedding space (first fine-tune embedding, then rerun KMeans)
2. Is collision 10.7% really harmful to the quality of behavior? EXP-008 has proven that high collision does not equal low quality, and Phase 1 needs to be done first
3. Although the online effect (GMV +0.98%) is significant, QuaSID’s GMV +2.38% is larger — possibly because the baseline is different

---

## IDEA-dos-0: Dual-Flow Orthogonal Residual Quantization (Dual-Flow Orthogonal RQ)

**Priority**: P2 (after NTP)
**Source**: DOS (Meituan, arxiv 2602.04460, WWW 2026)
**Status**: To be discussed

### Core Idea

DOS addresses two basic issues with SID learning: (1) **Codebook-Generation Gap** — existing methods task-agnosticly learn SID (pure reconstruction/clustering), which is disconnected from downstream generation tasks; (2) **Quantitative semantic loss** — the fixed coordinate system of standard RQ is not suitable for the LLM semantic structure.

**Dual-Flow Integration (DFI)**:
- Use **user-item twin towers** to simultaneously encode the user behavior sequence and target item when quantifying
- User Tower: Transformer Encoder encoding click sequence of LLM embedding
- Item tower: LLM embedding encoding target item
- **Shared codebook**: The two towers share the same codebook → user interest and item are mapped to a unified semantic space
- Training target: BCE (user-item matching) + VQ loss + Recon loss + Orth loss
- Key: SID codebook is no longer learned in isolation, but in the context of perceptual generation tasks

**Orthogonal Residual Quantization (ORQ)**:
- Before each layer of quantization, use the learnable orthogonal matrix W_orth to rotate the input (constraint W·W^T = I)
- MLP generates dimension-wise weight score → **top-k masking** selects primary features (task-relevant)
- Primary features are used for codebook quantization; secondary features + residual are passed to the next layer
- L_Mutual: Maximize the mutual information between primary features and task label Y
- Guarantee X_pri ⊥ X_sec (orthogonal decomposition) without losing information

### Experimental data

**Offline** (Meituan production data, 24M items, 180M interactions):

| Method | AUC | F1-Score |
|------|-----|---------|
| RQ-KMeans | 0.8363 | 0.7641 |
| RQ-VAE | 0.8526 | 0.7739 |
| DAS | 0.8539 | 0.7869 |
| **DOS** | **0.8763** | **0.8057** |

**NTP downstream** (HSTU framework, Hit@10):

| Method | All | Busi_A | Busi_B | Busi_C | Busi_D |
|------|-----|--------|--------|--------|--------|
| HSTU-RQ-KMeans | 0.0410 | 0.0252 | 0.0554 | 0.0398 | 0.0421 |
| HSTU-DAS | 0.0511 | 0.0325 | 0.0672 | 0.0502 | 0.0541 |
| **HSTU-DOS** | **0.0676** | **0.0457** | **0.0797** | **0.0730** | **0.0718** |

**Online A/B** (Meituan production traffic 30%, one week): **+1.15% revenue**

Ablation: MLP replaces Encoder → AUC drops to 0.8462; does not share codebook → 0.8671; adds Decoder → 0.8626 (reconstruction target conflicts with task-relevant selection)

### Association with the current project

- Current SID learning is task-agnostic (RKMeans clustering Qwen3 embedding), DOS pointed out that this leads to codebook-generation gap
- **Revelation from the failure of IDEA-sid-1**: EXP-007/009 failed to inject synergy signals into embedding; DOS adopts a different strategy - does not change the embedding, but introduces user behavior context in the quantization phase
- DFI's shared codebook idea is complementary to IDEA-flexcode-0 (FlexCode dual codebook): FlexCode is divided into two codebooks, CF/Semantic, and DOS uses a shared codebook to unify the user-item space.
- ORQ's orthogonal rotation is consistent with IDEA-sid-0 (OPQ), but OPQ is static preprocessing and ORQ is end-to-end learnable
- **decoder invalid discovery matters**: reconstruction goals conflict with task relevance → supports the intuition of "don't pursue perfect reconstruction"

### Experimental Design Draft

**Phase 1 — Task-Aware Quantitative Analysis**:
- Based on the existing MLP-FSQ quantified SID, measure: whether the SID of the first N items in the user click sequence can predict the SID of the target item (simple BCE model)
- If predictability is poor → confirms the existence of codebook-generation gap, DOS is valuable

**Phase 2 — ORQ module porting**:
- Add an ORQ layer (orthogonal rotation + dimension masking) before the MLP head of MLP-FSQ
- Keep the FSQ backend unchanged, only change the input space
- Evaluation: semantic_neighbor_HR, collision_rate, downstream NTP Recall

### Key questions

1. We use MLP-FSQ (non-RQ-VAE), and the residual quantification in ORQ is not directly applicable; the straight-through path of FSQ needs to be adapted.
2. The paper is only 4 pages (industry track) and has limited technical details - especially the calculation method of L_Mutual and the training stability of orthogonal constraints
3. The shared codebook requires both user sequence and target item - there is no "target item" during offline batch tokenization and needs to be changed to sampling positive examples.
4. The current tokenizer has been confirmed (MLP-FSQ h=64). Changing the quantization scheme requires retraining the entire link - the cost is high and Phase 1 needs to be verified first.

---

## IDEA-rqgmm-0: Gaussian Mixture Residual Quantization (RQ-GMM)

**Priority**: P2
**Source**: RQ-GMM (Tencent + Fudan University, arxiv 2602.12593)
**Status**: To be discussed — NTP will be re-evaluated later when tokenizer quality becomes a bottleneck

### Core Idea

Use **Gaussian Mixture Model** to replace K-Means for residual quantization, and introduce probabilistic modeling to better capture the statistical structure of the embedding space.

1. **Gaussian Mixture Quantization**: At each RQ level, use K Gaussian distributions to model the residual distribution:
   - `p(r) = Σ π_k * N(r | μ_k, Σ_k)`, diagonal covariance `Σ_k = diag(σ²_k,1, ..., σ²_k,D)`
   - Each codebook vector = Gaussian mean μ_k, additionally stores per-dimension variance σ²_k
2. **Soft Assignment (Training) + Hard Assignment (Inference)**:
   - E-step: Calculate posterior `γ_k = p(k|r)` (soft assignment, used for M-step parameter update)
   - M-step: update μ, σ², π (standard EM)
   - During inference: `k* = argmax_k γ_k`, `z_q = μ_{k*}` (equivalent to nearest neighbor, but distance metric takes into account covariance)
   - Residual propagation uses hard assignment to remain consistent with inference
3. **No Encoder-Decoder**: Operate directly in the original embedding space (same as RQ-KMeans), without the encoder/decoder network of VQ-VAE

### Key experimental data

**Offline (Amazon Review, BERT 768D embeddings, 2-level RQ, 128 codes/level)**:

| Method | RMSE | 码本利用率 (L1/L2) | AUC (FNN w/ Emb) |
|------|------|-------------------|------------------|
| VQ-VAE | 0.614 | 33.7% | 0.654 |
| RQ-VAE | 0.173 | 73.9%/71.8% | 0.659 |
| RQ-KMeans | 0.121 | 86.7%/87.1% | 0.667 |
| **RQ-GMM** | **0.117** | **89.5%/89.3%** | **0.678** |

- GMM vs KMeans: RMSE -3.3%, codebook utilization +2.8pp, AUC +0.011
- GMM converges faster + smoother (Figure 1)

**Online A/B (Tencent short video platform, 7 days, hundreds of millions of DAU)**:

| Comparison | Advertiser Value 提升 |
|------|---------------------|
| vs 直接 embedding | **+3.600%** |
| vs RQ-VAE | **+1.502%** |
| vs RQ-KMeans | **+0.613%** |

### Association with the current project

- We currently use **RKMeans (2-layer K-Means) + MLP-FSQ** — RQ-GMM can replace the first two layers of RKMeans
- **Core Value**: Codebook utilization rate from 86.7% → 89.5% — Solving the codebook collapse problem
- **Better for boundary samples**: soft assignment allows items on the distribution boundary to obtain more reasonable SIDs and reduce semantic breaks
- **Comparable computational cost**: Same order as RKMeans in O(TLNKD), but converges faster (fewer iterations)
- **Complementary to IDEA-dos-0**: DOS introduces user context to change the quantization target, and RQ-GMM improves the quantization algorithm itself
- **Complementary to IDEA-crab-0**: CRAB solves codebook imbalance through rebalancing, and RQ-GMM automatically reflects data density through mixing coefficients π_k

### Experimental Design Draft

**Phase 1 — Drop-in replacement for RKMeans**:
1. Replace K-Means in `model/rkmeans.py` with `GaussianMixture` of scikit-learn
2. Same 2 layers × 1024 clusters → compare RMSE, codebook utilization, semantic_neighbor_HR
3. Keep the third layer of MLP-FSQ unchanged → only change the quantization methods of the first two layers

**Phase 2 — Full GMM-RQ (if Phase 1 is profitable)**:
1. Implement custom RQ-GMM (because scikit-learn does not support residual quantization)
2. 3-layer RQ-GMM × 1024 clusters (no FSQ third layer required)
3. End-to-end comparison with MLP-FSQ: SID quality + NTP Recall

### Key questions

1. **The MLP head of **MLP-FSQ has done nonlinear projection**, and RQ-GMM operates in the original space—does GMM need to be done in the space after the MLP or before?
2. The paper’s embedding is 768D (BERT), ours is 1024D (Qwen3-0.6B) — is a high-dimensional diagonal GMM sufficient?
3. **Additional benefits of probabilistic modeling**: GMM’s per-cluster variance information can be used for confidence estimation — the SID of a high-variance cluster has high uncertainty and can be passed to NTP for uncertainty-aware training
4. The current tokenizer has been confirmed (MLP-FSQ h=64), and EXP-015 shows that irreducible loss is close → the absolute benefit of tokenizer improvement may be limited
5. **Lower priority than RL alignment** (EXP-037/038/039) — After the RL link is stable, tokenizer improvements will be used as a breakthrough for the next round

---

## IDEA-coins-0: COINS — RQ + OPQ two-phase SID cold start collaborative transfer

**Priority**: P2
**Source**: COINS (arxiv 2510.12604, WWW 2026)
**Status**: To be discussed - the main scenario is cold-item representation enhancement of CTR prediction, we are on the retrieval side; but the two-stage idea of "RQ coarse sharing + OPQ fine differentiation" can be used for reference

### Core Idea

COINS proposes an RQ+OPQ fusion SID representation scheme to address the Matthew effect of lack of synergy signals for cold-start items in e-commerce searches.

**Core insight**: Pure "content-collaborative alignment" ignores **asymmetry** — collaborative signals are naturally coarse-grained (stratified by item popularity), and content signals are naturally fine-grained (unique for each item). Forced alignment obscures individual differences.

**RQ-OPQ two-stage encoding**:
1. **RQ (Residual Quantization) stage - shared collaborative signal transfer**: RQ codes capture the **common structure** between items, and collaborative signals (learned from hot items) are given to cold items through shared coarse-grained codeword propagate
2. **OPQ (Optimized Product Quantization) stage - differentiated information**: OPQ codes encode **unique fine information for each item**, retaining individual differences

Two-stage alignment = cold items inherit collaborative knowledge + maintain individual characteristics.

**Online A/B (WWW 2026)**:
- item CTR **+1.66%**
- buyers **+1.57%**
- order volume **+2.17%**

### Association with the current project

**Background Complexity**:
- Our **IDEA-sid-0 (pure OPQ SID)** has been ❌ closed — pure OPQ loses to MLP-FSQ in behavior quality (EXP-004)
- COINS is a combination of **RQ + OPQ**, not pure OPQ. The RQ layer is dedicated to collaborative signal transmission for cold start items and does not assume the full SID structure.

**Points of reference**:
- Ideas for enhancing the representation of cold items: use shared coarse coding (our L0 KMeans) for collaborative transfer, and fine-grained coding (our L2 MLP-FSQ) for differentiation — essentially our existing tokenizer is already hierarchical, and COINS just names this hierarchical collaborative-vs-differentiation role
- **Evaluation method under Cold-item data distribution**: report the Recall/CTR of cold item subsets individually, instead of just the overall average - our eval has not done this segmentation yet

**Difference**:
- COINS is CTR prediction scenario (ranking), we are generative retrieval
- COINS has an independent collaborative embedding source (user-item interaction graph), we do not

### Experimental Design Draft

**P2 archive, if cold-start verification is done in the future**:

**Phase 1 — Cold item subset evaluation (zero cost, can be done today)**:
1. Split the eval data by interaction count: cold (<5 interactions) / warm (5-50) / hot (>50)
2. Report R@500 / R@10 respectively for EXP-020 checkpoint
3. Expectation: Under the existing MLP-FSQ tokenizer, cold item recall is significantly lower than hot item

**Phase 2 — Only executed if a significant gap in cold item is found in Phase 1**:
- Explore the solution of adding "cold start exclusive collaboration code" to L0 based on the existing two layers of RKMeans + the third layer of MLP-FSQ
- Or consider introducing collaborative embedding independent signals (similar to the dual codebook of IDEA-flexcode-0)

### Key questions

1. **Mismatch of application scenarios**: COINS is the sorting end of CTR, and we are the retrieval end. In the CTR scene, item representation and user representation are dot products, and in the retrieval scene, SID is the generation target. Different roles, high transplantation cost
2. **Overlapping with IDEA-flexcode-0 (FlexCode dual codebook CF+Semantic)**: FlexCode is already a dual codebook encoding CF/semantic, and the route is similar. COINS is two stages of the same SID sequence of RQ+OPQ, and FlexCode is two independent codebooks. The former is more compact, the latter is more flexible
3. **Complementary to IDEA-gatesid-0 (just added)**: GateSID does per-item gating in the representation layer, and COINS does two-stage encoding in the tokenizer layer. Both can be combined
4. **Cold item definition**: The interaction count threshold needs to be determined based on our data distribution

### Related ideas

- IDEA-sid-0 (Pure OPQ): ❌ is closed, the RQ-OPQ overlay of COINS is a different route
- IDEA-flexcode-0 (FlexCode dual codebooks): independent CF + Semantic codebooks, and COINS integrated SID route difference
- IDEA-gatesid-0 (GateSID): representation layer gating, can be combined
- IDEA-adasid-0 (AdaSID Kuaishou): Adaptive collision processing of cold items, another perspective

---

## IDEA-card-0: NU-RQ-VAE — Learnable reversible non-uniform transformation preprocessing → Re-RQ

**Priority**: P2
**Source**: CARD — Non-Uniform Quantization of Visual Semantic Unit for Generative Recommendation (UESTC, arxiv 2604.26427, SIGIR 2026)
**Tier**: C (Purely academic, offline public data set, no A/B; accepted in SIGIR 2026)
**Status**: To be discussed

### Core observation: Embedding is unevenly distributed → codeword usage is unbalanced

Figure 1 of the paper shows the 2D PCA of industrial rec embedding: **Extremely dense head items + long tail sparse diffusion**. Do an RQ-VAE directly on this distribution:

- Codewords in dense areas are over-allocated, resulting in insufficient fine-grained differentiation capabilities (many items share several codes)
- Codewords in sparse areas are idle, and codebook utilization is low
- Autoregressive decoding will further amplify this bias (frequent codes are more likely to be generated)

Our MLP-FSQ theoretically has a low-collision goal (EXP-045 is controlled by hidden dim), but the embedding itself is still unevenly distributed → items in the long tail area have poor centroid coverage even if their codes do not conflict.

### CARD’s solution: distribution correction before quantization

Between the encoder output of RQ-VAE → quantize, insert a **learnable and reversible nonlinear transformation** `T: z → z'` to approximate the skewed distribution of z into a uniform distribution; then use `T⁻¹` reverse mapping when doing RQ and reconstruct on z'.

Two transformation variants:
1. **Kumaraswamy-based (general)**: Use the CDF of Kumaraswamy distribution as `T`. Kumaraswamy CDF `F(x; a,b) = 1 − (1 − x^a)^b` Differentiable, reversible, strong parameterization → can fit various complex non-uniform distributions. (a, b) is learned during training.
2. **Logistic-Logit-based (dedicated bell-shaped)**: Use logistic function + logit dual, for bell-shaped distribution. The numerical value is more stable, but the expressive power is weaker.

Key points: The transformation is **per-dimension invertible**, RQ reconstruction loss through `T` / `T⁻¹` backpropagation training (a, b). The entire pipeline is still end-to-end VAE.

### Plugability

The paper emphasizes that NU-RQ-VAE is **plug-and-play**: it can replace RQ-VAE and can be grafted to any quantization backbone such as MLP-FSQ and RQ-KMeans.

### Association with the current project

- **Direct benchmarking with IDEA-crab-0 (CRAB Walmart)**: CRAB is codeword frequency-aware regular + split, which handles overheating tokens after the fact; CARD is ex-ante distribution correction. The two are orthogonal and can be superimposed.
- **Benchmarking IDEA-rqgmm-0 (Tencent GMM-RQ)**: GMM uses a mixture of Gaussians to model the embedding distribution, and CARD flattens the distribution through transformation. The goal is the same (dealing with non-uniformity), the means are opposite (modeling vs leveling).
- **Benchmarking IDEA-adasid-0 (Kuaishou AdaSID)**: AdaSID allows cold items to borrow the SID of hot items afterwards, and CARD reduces the codebook waste of cold items in advance. Can be combined.
- **Relationship with MLP-FSQ**: Our MLP-FSQ projection is already a nonlinear transformation, but it does not explicitly constrain the output distribution to be uniform. Adding Kumaraswamy CDF after projection output + before quantize is equivalent to "doing distribution normalization on MLP output".

### Experimental Design Draft

**Phase 1 (~0.5 days, simulation verification)**:
1. Take Qwen3-0.6B embeddings 14d
2. For each dimension apply Kumaraswamy CDF transform (a, b estimated from data or nn.Parameter learned)
3. Calculate variance / skewness / kurtosis and compare before and after transformation
4. Observe the Gini coefficient of code frequency after quantization

**Phase 2 (~2 days, integrated into MLP-FSQ)**:
1. Insert the `NUTransform` layer after the MLP projection in `model/fsq_quantizer.py` and before FSQ discretize
2. Retrain a tokenizer (4B h=256 of EXP-045 is baseline)
3. Indicators: collision rate, d2 Gini, NTP R@500 end-to-end comparison

### Key questions

1. **Bijective constraint**: Kumaraswamy CDF is bijective on (0,1). It is necessary to map the embedding to (0,1) with sigmoid first and add a nonlinearity; will it destroy the learning ability of MLP-FSQ projection?
2. **Training Stability**: Can (a, b) initialize from 1.0 (equivalent to identity transformation) converge to a non-trivial solution? The paper does not seem to mention the initialization details.
3. **reconstruction path**: FSQ itself has no decoder (no reconstruction loss), only NTP loss can provide gradient signals. Can the Kumaraswamy transformation be effectively updated by downstream loss? (RQ-VAE has reconstruction loss, but FSQ needs to design additional self-supervision)
4. **Sweep relationship with MLP-FSQ**: If the MLP hidden dim is large enough (h=1024+), MLP itself can learn an approximately uniform output distribution. Does Kumaraswamy's marginal revenue return to zero at h_max? It is recommended to compare h=32 and h=1024 before deciding.

### Why Tier C instead of Tier A

- Author of UESTC + Southwestern University of Finance and Economics, co-author of Wu Industrial Lab
- Experiment on Amazon Beauty/Toys/Sports public data sets, non-industrial daily activity level item pool
- No online A/B, no CTR / GMV lift numbers
- But the NU-RQ-VAE transformation is a **mathematically clear independent component**, a new technology recognized by SIGIR 2026, and can be verified as a plug-in on our existing tokenizer

### Related ideas

- IDEA-crab-0 (CRAB): ex-post token split, complementary to the ex-ante distribution correction of this idea
- IDEA-rqgmm-0 (RQ-GMM): another route to handle non-uniformity (Gaussian mixture modeling)
- IDEA-adasid-0 (AdaSID): Cold item borrows hot item SID, can be combined
- IDEA-quasid-0 (QuaSID): Qualification-aware SID, similar to motivation
