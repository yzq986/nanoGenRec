# Scaling (scalability experiment)

[English](scaling.md) | [Chinese](scaling.zh.md)

The scaling law study of model size vs. data size vs. sequence length directly determines the resource allocation strategy.

**Scope of influence**: `metrics/sid_prediction.py`, `model/train.py`, ARCHITECTURE.md (tier design)

---

## Evolution path

```
S-tier (39.5M params, currently the only implementation)
├── IDEA-oneloc-4: Scaling Law experiment
│ ├── Model scaling → EXP-015 ✅ L(N)=2.522+2055/N^0.456, ~100M flattening
│ │ └── M+ 101M vs S 17.5M: loss only dropped by 0.06, tokenizer is the bottleneck
│ └── Sequence length scaling → To be tested (current max_seq_len=512, ~170 items/user)
├── IDEA-kunlun-0: Rec Scaling Laws (Meta Ads)
│   └── MFU 17%→37%, GDPA + CompSkip, power-law scaling
│ └── Increased importance: tokenizer becomes the key to scale up after breaking through the bottleneck
├── IDEA-hstu-0: Sparse Self-Attention Co-design (Meta)
│ └── 5x training / 21x inference scaling, retaining self-attention expressiveness
├── IDEA-mtgenrec-0: Distributed GR training system (Meituan)
│   └── Dynamic Hash Embedding + Sequence Batching + ID Dedup, 2.4x throughput
├── IDEA-freescale-0: Sequence Load Balancing + SM-Free Communication (Meta, MLSys 2026)
│ └── Long sequence UIH straggler mitigation + priority embedding update + CPU-RDMA zero SM usage
└── IDEA-vlm-0: Versioned Late Materialization — Breaking the Fat Row Wall (Meta, arxiv 2604.24806)
└── UIH sequence single copy storage + JIT reconstruction during training, sequence length unlocked from 4K → 64K, A/B Topline +0.22%
```

---

## Current Conclusion (2026-04-17)

**Model parameter scaling levels off at ~100M, and tokenizer 32-bit encoding is the current bottleneck. Sequence length scaling has not yet been verified. **

### Key experimental data

| Model | Active Params | Eval Loss | PPL | R@500 |
|------|--------------|-----------|-----|-------|
| S-tier | 17.5M | 2.9960 | 27.05 | 58.5% |
| M+-tier | 101M | 2.9371 | 25.12 | 60.7% |
| Irreducible (fit) | ∞ | 2.522 | ~12.5 | — |

**Core insight**: M+ has 6x more parameters than S, but the loss is only 0.06 lower. Scaling law fitting L(N)=2.522+2055/N^0.456 shows irreducible loss=2.522 determined by tokenizer. Breaking through the bottleneck requires: (1) higher bit SID (currently 32-bit); (2) longer sequence (not yet verified).

---

## IDEA-oneloc-4: Scaling Law — Sequence length >> model size

**Priority**: ~~P0~~ → Partially completed (model scaling verified, sequence length to be tested)
**Source**: OneLoc §4.4 Hyperparameter Experiments
**Status**: EXP-015 model scaling completed; sequence length scaling pending experiment

> **NTP stage update (2026-04-17)**: EXP-015 verified the model parameter scaling law L(N)=2.522+2055/N^0.456. Key findings: M+ (101M active) only reduces loss by 0.06 compared to S (17.5M). The scaling law has been severely flattened at ~100M - tokenizer 32-bit encoding is the bottleneck. Scaling of the sequence length dimension has not yet been tested (currently max_seq_len=512, ~170 items/user), which is an important direction in the next stage.

### Core Idea

OneLoc's scaling experiments revealed a key finding: **The benefit of sequence length is much greater than the benefit of model size**. When the model is expanded from 0.05B ​​to 0.3B, recall/NDCG increases by 7% on average; but when the sequence length is expanded from 100 to 300, recall increases by 13% and NDCG increases by 51%. This means that when resources are limited, increasing sequence length should be prioritized over model parameters.

### Association with the current project

- Current `AutoregressiveNTPModel` S-tier config: 6 layers, 256 embed_dim, ~39.5M params
- ARCHITECTURE.md defines M/L tier but does not implement it
- **Key issue**: We have not done any scaling experiments on the NTP model yet
- More direct inspiration: In NTP training, is the length of user behavior sequence more important than model size?
- Current behavior sequence processing: What is the length of the sequence exported in `data/export_behavior.py`? Is it long enough?

### Experimental Design Draft

**Experimental Matrix**:

| Dimensions | Small | Medium | Large |
|------|-----|-----|-----|
| ModelParameter | S-tier (39.5M) | M-tier (~150M) | L-tier (~500M) |
| sequence length | 50 | 100 | 200 |

**Design**:
- Fixed quantization scheme (RKMeans 3x1024 or OPQ)
- 3x3 grid: model size x sequence length
- Train NTP model to convergence in each group
- Record recall@5/10/20, NDCG@5/10/20

**Evaluation**:
- Draw scaling curve: recall vs model parameters (fixed sequence length)
- Draw scaling curve: recall vs sequence length (fixed model parameters)
- Verify whether the conclusion of OneLoc is reproduced in our scenario

### Key questions

1. **Pre-requisites**: A stable NTP training pipeline + a stable quantization solution are required first.
2. Is the current NTP training ready for end-to-end run? You need to confirm the complete process of `model/train.py` → NTP
3. Amount of behavioral data: Sequence length 200 requires sufficient user behavior data
4. Computational cost: 9 sets of experiments, each set may require several hours of training
5. **Why P0**: The conclusion of this experiment directly determines the resource allocation strategy - whether to spend money to buy a larger GPU or to spend money to collect more behavioral data

---

## IDEA-kunlun-0: Recommendation Scaling Laws (MFU Optimization + GDPA)

**Priority**: P1 — Increased importance due to scaling flattening
**Source**: Kunlun (Meta Ads, arxiv 2602.10016, Feb 2026)
**Status**: To be discussed

> **NTP phase update (2026-04-17)**: EXP-015 shows that model scaling has leveled off at ~100M (tokenizer bottleneck). The value of Kunlun's MFU optimization and GDPA at the current stage does not lie in the scale up model, but in: (1) improving the training efficiency of the existing model (faster iteration experiments); (2) after breaking through the tokenizer bottleneck in the future (such as OPQ long SID or higher bits), GDPA is the key technology for scale up.

### Core Idea

Kunlun established LLM-like **power-law scaling laws** in large-scale recommendation systems. Core findings: The fundamental reasons for the low efficiency of recommended model scaling are **low MFU (Model FLOPs Utilization)** and **uneven resource allocation**.

Solution:
1. **Generalized Dot-Product Attention (GDPA)**: Recommended dedicated attention mechanism
2. **Hierarchical Seed Pooling (HSP)**: Efficient feature aggregation
3. **Computation Skip (CompSkip)**: Selective calculation, skipping low-value paths
4. **Sliding Window Attention**: Manage user history sequence

**Results**: MFU improved from **17% to 37%** (B200 GPU), **2x scaling efficiency**, deployed to Meta Ads main model.

### Association with the current project

- The current S-tier model is small (39.5M), MFU is not the bottleneck
- But both IDEA-oneloc-4 (Scaling Law) and IDEA-plum-0 (LLM CPT) require scale up → Kunlun’s experience is directly applicable
- **GDPA** may be more suitable than standard attention for recommendation scenarios: user behavior sequences have different patterns from natural language sequences
- **CompSkip** is related to IDEA-gr4ad-1 (LazyAR): both are selective calculations

### Experimental Design Draft

**Phase 1 — GDPA replacement standard Attention**:
- You need to read the full text of Kunlun’s paper to understand the specific definition of GDPA
- Replace attention module in `CausalTransformerLayer`

**Phase 2 — MFU Profiling**:
- Profile the current NTP trained MFU on 8xA100
- Identify inefficient modules → targeted optimization

### Key questions

1. The specific implementation of GDPA requires the full text of the paper
2. The current model is too small, and the increase in MFU does not equal the increase in training speed (it may be memory-bound rather than compute-bound)
3. CompSkip requires per-sample routing → high implementation complexity

---

## IDEA-hstu-0: Sparse Self-Attention + Model-System Co-design (ULTRA-HSTU)

**Priority**: P1
**Source**: ULTRA-HSTU (Meta, arxiv 2602.16986, Feb 2026)
**Status**: To be discussed

### Core Idea

ULTRA-HSTU is implemented through **end-to-end model-system co-design**:

1. **Input Sequence Design**: Optimize input sequence construction for recommended scenarios
2. **Sparse Attention**: Maintain the expressiveness of self-attention while avoiding O(n²) calculations
3. **Model Topology**: Architecture topology optimization to match system efficiency

Key position: cross-attention (such as IDEA-onemall-1 Query-Former) although solving the O(n²) problem, **limits the expressive power of self-attention**. ULTRA-HSTU maintains expressiveness while controlling computational effort through sparse self-attention.

**Results**: **5x faster training, 21x faster inference**, serving **billions of users**, **4-8% engagement improvement**.

### Association with the current project

- The current `CausalTransformerLayer` is full self-attention (O(n²)), and there is no problem with short sequences
- If extended to long sequences (IDEA-oneloc-4 / IDEA-onemall-1):
  - IDEA-onemall-1 select cross-attention (Query-Former) → compress expression
  - ULTRA-HSTU select sparse self-attention → preserve expression
  - The tradeoff of the two routes is worthy of experimental comparison.
- **Model-System Co-design**'s philosophy: Don't just look at model quality, but also optimize system efficiency

### Experimental Design Draft

**Phase 1 — Sparse Attention Replacement**:
- Add sparse attention option (such as sliding window + global tokens) to `CausalTransformerLayer`
- Comparison: full attention vs sparse attention vs Query-Former Recall@K and training speed under different sequence lengths

**Phase 2 — Input Sequence Design**:
- You need to read the full text of the paper to understand the input sequence design details of ULTRA-HSTU
- May involve action type encoding, timestamp encoding, etc.

### Key questions

1. The details of the full text of the paper (the specific pattern of sparse attention) need to be supplemented
2. The current sequence is short (3 SID tokens), sparse attention has no benefit → relies on sequence expansion
3. Comparative experiments with IDEA-onemall-1 (Query-Former) require a unified experimental framework

---

## IDEA-mtgenrec-0: Efficient distributed GR training system (Dynamic Embedding + Sequence Balancing)

**Priority**: P2 — Production deployment infrastructure, not urgently needed during the current research phase
**Source**: MTGenRec (Meituan + Wuhan Univ, arxiv 2505.12663, May 2025)
**Status**: To be discussed

> **P2 Reason**: The current training scale is small (8 GPU, 17.5M params, 14d data), and the efficiency bottleneck of TorchRec/torchrun has not yet been touched. When the model scales up or the data volume expands to require 100+ GPUs, MTGenRec's technology is directly applicable.

### Core Idea

MTGenRec is a GR-specific distributed training system built by Meituan based on TorchRec, which solves four engineering bottlenecks in GR training:

1. **Dynamic Hash Embedding Table**: Replace TorchRec’s static embedding table with the dynamic hash table of MurmurHash3 + grouped parallel probing, supporting real-time item additions and deletions (new products on/off the shelves). Key-Value decoupled storage + chunk-based allocation, only the lightweight key structure is migrated during capacity expansion. Throughput improvement 1.47-2.22x vs TorchRec MCH
2. **Two-Stage ID Deduplication**: There are a large number of repetitions of feature IDs in the user sequence (the same user/item appears multiple times). Stage 1: local deduplication and then all-to-all communication; Stage 2: deduplication again after receiving the remote ID. Reduce embedding communication volume and increase throughput by 53%
3. **Dynamic Sequence Batching**: The user sequence length has a long-tail distribution (avg=600, max=3000). Fixed batch size → severely uneven load across GPUs (maximum difference 25.8ms). Change to target token count mode: Binary search finds the batch split point closest to the target token number, and the GPU memory utilization changes from 75% to 90%. Throughput increased by 26.5% (110G model, 64 GPU)
4. **Automatic Table Merging**: The FeatureConfig interface automatically merges embedding tables with the same dimensions to reduce the number of lookup operators. Dynamic tables use bit-shift offset to avoid ID conflicts

### Key data

| Metric | Value |
|------|------|
| Training data | 200M sequences/day, avg 600 tokens, max 3000 |
| Model scale | GRM 4G (small) ~ 110G (large) GFLOPs |
| GPU Config | 8~128 × A100 80GB SXM4, NVLink 600GB/s |
| Throughput improvement | 1.6x~2.4x vs TorchRec |
| Scaling efficiency | 128 GPU reaches 62.75%~78.5% ideal linear acceleration |
| Online A/B (takeaway) | +1.22% user order volume, +1.31% PV_CTR (vs 2-year iteration of DRM) |
| User scale | 770M annual trading users, daily peak 98M orders |

### Association with the current project

- **Dynamic Sequence Batching** Most directly related: We use torchrun + packed sequences for training. The sequence lengths obtained by different ranks are different, and the same GPU load imbalance problem may exist. Currently not doing dynamic batching — worth referencing when scaling up
- **Dynamic Hash Embedding**: Currently not needed (SID vocabulary is fixed), but if user embedding or item side features are introduced as sparse embedding, you will face the same dynamic addition and deletion problem
- **Two-Stage ID Dedup**: The current sequence is short (~170 items/user) and there are not many duplicate IDs. The repetition rate will increase when Scale reaches long sequences.
- **Model Architecture**: Meituan’s GRM uses HSTU (SiLU attention) + MMoE, which is different from our CausalTransformer but is common at the training system level.

### Experimental Design Draft

**Phase 1 — Dynamic Sequence Batching (can be implemented independently)**:
- Implement token-count-based batch construct in `ntp/data.py` or `data/dataset.py`
- Target: total number of tokens per rank ≈ target_tokens (instead of fixed batch_size)
- Use cumulative sum + binary search to find the split points
- Need to modify gradient averaging: weighted by the actual number of samples in each rank (weighted All-Reduce)
- Evaluation: GPU utilization, training throughput, and whether the convergence curve is consistent

**Phase 2 — Deploying Extensions (Forward)**:
- Dynamic hash embedding: If real-time item updates are introduced
- ID dedup: when the sequence length is extended to 1000+

### Key questions

1. Currently 8 GPUs are used for training, the sequence is short (max_seq_len=512), and the load difference between GPUs may not be significant — profiling is required to confirm first
2. Dynamic batching changes the batch size of each rank → gradient requires weighted average, and the implementation cannot destroy the correctness of gradient sync of DDP
3. The paper model architecture (HSTU) uses SiLU attention instead of standard softmax attention, and the scaling behavior of O(n²) may be different.

---

## Priority summary

| Priority | ID | Experiment | Reason |
|--------|-----|------|------|
| ~~P0~~ Partially completed | IDEA-oneloc-4 | Scaling Law: Sequence length vs Model size | Model scaling EXP-015 ✅ (~100M flattening); Sequence length scaling to be verified |
| P1 | IDEA-kunlun-0 | Rec Scaling Laws (MFU + GDPA) | Meta Ads deployment verification; tokenizer becomes the key to scale up after breaking through the bottleneck |
| P1 | IDEA-hstu-0 | Sparse Self-Attention Co-design | 21x inference scaling, Comparison Query-Former Route |
| P2 | IDEA-mtgenrec-0 | Distributed GR Training system | Meituan deployment, 100+ GPU scaling, dynamic batch for reference |
| P2 | IDEA-freescale-0 | Meta FreeScale: Load Balancing + SM-Free Communication | 256×H100 verification, 90% communication bubble reduction; current 8 GPUs have limited benefit, core reference for future multi-node expansion |
| P2 | IDEA-vlm-0 | Meta VLM: Versioned Late Materialization | Fat Row wall @ 4K UIH, delayed materialization wall breaking pushed to 64K; A/B Platform A Topline +0.22%/Metrics-C +4.1% |

---

## IDEA-freescale-0: FreeScale — Sequence Load Balancing + SM-Free Communication

**Priority**: P2
**Source**: FreeScale (Meta, arxiv 2604.24073, MLSys 2026)
**Status**: To be discussed (for future reference, currently 4-8 GPU single node has limited benefit)

### Core Idea

FreeScale is a distributed training system designed by Meta for DLRM/sequence recommendation, providing systematic solutions to three efficiency bottlenecks that dominate large-scale (100+ GPU) training. Achieved **90.3% exposed communication reduction** on a 256×H100 production cluster, and the offline normalized entropy is exactly the same as the baseline (without loss of accuracy).

**1. Sequence Load Balancing (mitigating Straggler)**

The length heterogeneity of UIH (user interaction history) is huge: 2k vs 21k samples coexist in the same batch, resulting in > 20% difference in calculation amount between ranks, and there are empty ranks and slow ranks. FreeScale uses a three-stage AllGather to collect world UIH lens + candidate lens before each iteration, and then uses `FBS` (First-Fit-Decreasing by sequence size) or `VBS` (Variable Block) partition algorithm to redistribute samples. Key points:
- **Cannot be pre-sorted by length** (temporal ordering is sensitive to model quality in recommended scenarios)
- **Make runtime partition inside trainer** (cannot rely on static data layout due to dynamic resource allocation)
- straggler% dropped from 22% to 2.4% on 21k UIH + 64 GPU

**2. Prioritized Embedding Updates**

Vanilla TorchRec does blocking AllToAll (IDs→lookup, result→rank) twice per iteration. A simple prefetch will read the stale embedding (the next iter's lookup is before this iter's backward). FreeScale's insight: **Real collision rate is only ~12%** (P99 = 14%), so:
- Prefetch **all non-conflicting lines** → completely overlap with forward
- Only blocks waiting for collision lines — exposed communication becomes O(collision rate × volume)
- Result: TorchRec exposed comm 111 ms → FreeScale 13 ms under 8k UIH

**3. SM-Free Communication**

When communication and calculation on the GPU occur at the same time, NCCL will occupy SM, causing the actual overlap to be affected by SM preemption (10% throughput loss even if NCCL_MAX_NCHANNEL is increased). FreeScale goes **CPU-RDMA**: Move embedding back to the CPU for collective communication, completely giving up GPU SM. A stable 10% speedup is observed for sequence models (d=128, seq=8192), and the speedup does not change with NCCL tuning.

**4. Staged Training Pipeline (does not rely on full graph trace)**

Unlike CUDA Graph / `torch.compile` that does full graph tracking (which disables dynamic branching / third-party ops), FreeScale divides the train step into five stages: data loading / forward / backward / opt step / metrics, and uses PyTorch module hooks for instrumentation (`named_modules()` enumerates embedding tables). Preserve model iteration flexibility while introducing optimization.

### Association with the current project

- **Currently not beneficial**: Our `torchrun` is trained on 4-8 GPU single node, UIH length ~170 items (far < Meta's 21k), straggler is not the main bottleneck
- **Benefit from sequence length scaling verification**: IDEA-oneloc-4 Phase 2 will push max_seq_len 512→2048+, and then straggler will appear in the long-tail UIH. This solution Phase 1 (load balancing) can be directly adopted
- **Comparison with IDEA-mtgenrec-0 (MTGenRec)**:
  - MTGenRec's "Sequence Batching" ≈ FreeScale's "FBS partition", the same idea
  - MTGenRec does not have "prefetch + collision-only wait" mechanism
  - MTGenRec is based on the TensorFlow ecosystem, and FreeScale is based on PyTorch/TorchRec + Triton — the latter is more in line with our technology stack
  - **Refer to FreeScale instead of MTGenRec** (if you do distributed optimization in the future)
- **Triton kernel**: FreeScale uses custom Triton kernel to implement variable-length attention. Our `ntp/model.py` will also require similar optimization if we push long sequences.

### Experimental Design Draft

**Not executed in the current stage. ** Will be evaluated again when IDEA-oneloc-4 (sequence length scaling) is advanced to 8k+ UIH and multi-node training is performed. What you can do then:

**Phase 1 — Load Balancing (can be implemented independently, low risk)**:
- Implement FBS partition in `ntp/data.py` or `data/distributed_sampler.py`
- AllGather the batch lengths of each rank before each iteration starts, and redistribute samples according to First-Fit-Decreasing
- Evaluation: idle time between ranks, iteration time variance, end-to-end QPS
- **Expectation**: Long UIH (>5k) + 8 GPU scenarios have significant benefits; short UIH scenarios may have negative benefits (overhead exceeds straggler)

**Phase 2 — Prioritized Embedding Updates**:
- Depends on embedding table being large enough + multi-node training, currently not applicable

**Phase 3 — SM-Free Communication**:
- Depends on CPU-RDMA hardware support + NCCL replacement, complex project, delayed

### Key questions

1. How big is the straggler under current DDP + short UIH? Profiling confirmation is required. If it is <5%, don’t bother.
2. FBS partition changes the actual batch size of each rank → DDP gradient reduce requires weighted average (the same pitfall as MTGenRec)
3. SM-Free communication requires high-speed CPU-NIC bandwidth (8×200 Gb/s InfiniBand for FreeScale experiments), which may not match our cloud environment.
4. Triton kernel replacing standard PyTorch ops will affect `torch.compile`/autocast compatibility

### Related ideas

- IDEA-mtgenrec-0 (MTGenRec): A similar system to Meituan, with overlapping technologies but FreeScale is more mature
- IDEA-oneloc-4: sequence length scaling, which is a prerequisite for starting FreeScale
- IDEA-hstu-0: Sparse attention co-design, reduce compute requirements → reduce FreeScale necessity

---

## IDEA-vlm-0: Versioned Late Materialization — Strip UIH sequences from training samples

**Priority**: P2
**Source**: Versioned Late Materialization for Ultra-Long Sequence Training in Recommendation Systems at Scale (Meta, arxiv 2604.24806, 2026-04-27)
**Status**: To be discussed (bound with IDEA-oneloc-4 sequence length scaling)

### Core Issue: Fat Row Wall

The industrial RecSys standard practice is to **pre-materialize** the complete UIH sequence into each training sample ("Fat Row"):

- A request generates K training samples in the lookback window → the same UIH is copied K times → K-fold data redundancy
- Platform A actual measurement: When UIH reaches **4K items**, data infrastructure resources have exceeded GPU training resources → defined as "Fat Row Wall"
- Continue scaling above 4K, storage/IO costs > training benefits, economically unreasonable → blocking the exploration of longer UIH

### Core Insight: Physical replication is not a necessary condition for O2O consistency

Why do you need to pre-materialize before? In order to ensure **Online-to-Offline (O2O) consistency** - the features seen during training must be the exact state at the time of inference (otherwise future leakage will make the offline indicators falsely high).

Meta's argument: UIH is an **append-only, temporally ordered, immutable** sequence. In this case:

- Only requires **a standardized canonical UIH store** + **per-request lightweight version metadata** (version pointer)
- During training, use version pointer + time predicate (`t < T_request`) **JIT reconstruction** UIH state at that moment
- This is the database community's **late materialization + MVCC**, ported to the RecSys training data pipeline

### Key points in system design

1. **Bifurcated consistency protocol** — streaming training (streaming) and batch training (batch) have different time semantics
   - Streaming: Events are appended in real time, requiring snapshot read of `t ≤ T_request`
   - Batch: Historical traceback requires the version pointer to accurately point to the UIH state at that time
   - Both protocols must prevent future leakage and share a canonical store.

2. **Read-optimized immutable storage with multi-tenant projection pushdown**
   - Different model tenants require different sequence lengths (A requires 64K, B requires 16K, and C requires 4K)
   - Share a long sequence data and pushdown projection according to the length required by the tenant (similar to column storage projection pushdown)
   - Eliminate the read amplification of "short sequence tenant reads the entire long sequence" in the Fat Row paradigm

3. **Disaggregated preprocessing + pipelined I/O prefetching**
   - JIT reconstruction during training will introduce sequence lookup I/O (Model A 62.7% baseline primary read)
   - But immutable store single-tier, compaction-free → 3.4× higher per-host throughput than append-only primary storage
   - Data affinity sharding + prefetching enables lookup delays to be absorbed by the pipeline and training remains GPU-bound

4. **Data-affinity optimization**
   - Batch training time-adjacent samples of the same user are sharded to the same DPP worker
   - The same UIH lookup is reused, and the batch scene lookup bandwidth is reduced by another 60%.

### Actual measurement results

**Fat Row system efficiency (baseline = 1.0):**

| Tenant | UIH length | Primary Write ↓ | Primary Read ↓ | Lookup Read (new) | Data loading delay |
|--------|---------|----------------|---------------|------------------|-------------|
| Model A (long) | Long | **-46.2%** | **-70.3%** | +62.7% (streaming) / +24.6% (batch) | +9.7% |
| Model B (mid) | Mid | -50.9% | — | +16.2% / +6.5% | **-26.4%** |
| Model C (short) | Short | -47.7% | — | +8.7% / +3.4% | **-36.2%** |

Key: The 62.7% new lookup read of the long sequence tenant is offset by the 3.4× read density of the compaction-free immutable store; the net income of the short and medium sequence tenant is positive due to projection pushdown.

**Sequence length scaling A/B (Table 2):**

| Platform | Seq Length | Topline | Metrics-C | Metrics-E |
|----------|-----------|---------|-----------|-----------|
| Platform A | 4K → 16K | **+0.22%** | **+4.1%** | +2.3% / +4.3% |
| Platform B | 4K → 10K | **+0.14%** | +0.79% | +1.4% / +1.7% |

Platform A can achieve cumulative NE +1.2% by continuing to push from 4K to 64K (4K→64K total NE > 5%); an improvement of this magnitude under the Fat Row paradigm "within the same infrastructure envelope is impractical".

**Infrastructure as HSTU / trillion-parameter sequential transducer**: Meta's HSTU [arxiv 2402.17152] and VISTA (IDEA-vista-0) are deployed on this infrastructure.

### Association with the current project

- **IDEA-oneloc-4 sequence length scaling**: direct correspondence. Currently `max_seq_len=512`, ~170 items/user, far from hitting the wall
- **Which step makes sense**: Meta's Fat Row Wall @ 4K is caused by K-fold amplification (the user copies a copy of UIH for each request). Our current NTP training data format is **per-user single row** (day partition aggregation), and K is not a problem. The same wall is only encountered when (a) pushing to true ultra-long (>4K items) and (b) switching to request-level training samples (per-impression)
- **multi-tenant benefits**: If we have different tenant models (P2 ranker vs retriever, short sequence vs long sequence) sharing the same user behavior data set in the future, projection pushdown can amplify and eliminate the reads of short sequence tenants

### Reason for not executing in the current stage

- 4-8 L20X GPU, single node, `max_seq_len=512` is far from the bottleneck
- The data form is per-user daily partition (not per-request Fat Row), and K-fold replication problem does not exist
- A dedicated data platform team is required to implement it (MVCC + dual protocols + column storage pushdown is a complete storage layer transformation)

### Future trigger conditions

It will be re-evaluated when the following two conditions are met at the same time:

1. `max_seq_len` is pushed to 2K+ and a single training sample is too large (it has become an I/O bottleneck)
2. Switch the training data granularity from per-user daily to per-request (for example, per-impression CTR label instead of day-level NTP)

### Key questions

1. **Non-necessity argument for O2O consistency**: Meta’s argument depends on UIH being append-only + immutable. Our SID sequence also satisfies this property (item behavior is immutable once it occurs), but are user features (side information such as time_gap bucket) also purely append-only? If side features will be revised → late materialization will break O2O
2. **Version metadata overhead**: The lightweight pointer is more lightweight, and the number of bytes is not given in the paper. Estimated ~16-32 bytes/pointer vs several MB for the entire UIH row
3. **Complexity of bifurcated protocol**: The two sets of stream/batch consistency protocols are a typical design that "looks elegant and actually debugs it for a whole quarter", and it is unrealistic to promote it to small and medium-sized teams.

### Related ideas

- IDEA-oneloc-4: Sequence length scaling is the only prerequisite that triggers this idea
- IDEA-mtgenrec-0 / IDEA-freescale-0: training system optimization, this idea is data layer optimization, complementary
- IDEA-vista-0 (VISTA, Meta): deployed directly on this data infrastructure
- IDEA-hstu-0 (HSTU, Meta): Same as above, trillion-parameter transducer relies on this data layer
