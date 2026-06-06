# Inference (inference optimization)

[English](inference.md) | [Chinese](inference.zh.md)

Optimization of beam search and decoding strategies to improve inference throughput and candidate quality. Becomes critical as the model size and beam scale up.

**Scope of influence**: `metrics/sid_prediction.py` (beam search logic)

---

## Evolution path

```
Fixed beam search (current beam=5, full vocab softmax)
├── IDEA-gr4ad-4: Dynamic Beam Search
│ ├── DBW: Gradually increase the beam (128→256→512)
│ └── TopK Pre-Cut: Select top-b for each beam → global top-k
├── IDEA-static-0: CSR matrix constraint decoding
│ └── trie → CSR sparse matrix, GPU vectorization, YouTube 948x speedup
├── IDEA-earn-0: Register Token compression
│ └── Full attention in the first K layer, only register tokens in the last L-K layer, 3.79x speedup
├── IDEA-flame-0: GR Serving System (PDA/FKE/DSO)
│ └── CPU-GPU heterogeneous + kernel fusion + dynamic scheduling
└── IDEA-nezha-0: Self-Drafting Speculative Decoding
└── placeholder tokens → single prefill, draft head (logit+RNN), hash-set verification
```

---

## IDEA-gr4ad-4: Dynamic Beam Search Strategy

**Priority**: P1
**Source**: GR4AD §Dynamic Beam Serving
**Status**: To be discussed

### Core Idea

GR4AD proposes two beam search optimizations: (1) Dynamic Beam Width (DBW) — Gradually increase the beam (128→256→512 instead of fixed 512→512→512), because the candidate quality in the early layers is high, and a large beam is not needed to retain good candidates; (2) TopK Pre-Cut — bᵢ candidates are first selected in each beam, and then top-k are globally selected to avoid sorting on the full vocab. Results: DBW brought +0.31% revenue and QPS increased by 45%; TopK Pre-Cut brought +184.8% QPS.

### Association with the current project

- The beam search of `metrics/sid_prediction.py` is a fixed beam_size, and each step is softmax + top-k on the full vocab
- **Strongly related to IDEA-gr4ad-0 (MGMR)**: If you use unequal large codebooks (16384→4096→1024), the first layer vocab is large but only requires a small beam, and the subsequent layer vocab is small but requires a large beam - naturally suitable for dynamic beam
- Currently beam_size=5 has no room for optimization, but the production target planned by ARCHITECTURE.md is beam=512 - dynamic beam will be necessary by then
- TopK Pre-Cut can be implemented immediately as a universal optimization

### Experimental Design Draft

**Variable 1 — Dynamic Beam Width**:
| Configuration | Step 1 | Step 2 | Step 3 | Total beam |
|------|--------|--------|--------|---------|
| Fixed | 50 | 50 | 50 | 50 |
| DBW-A | 10 | 25 | 50 | 50 |
| DBW-B | 5 | 15 | 50 | 50 |

**Variable 2 — TopK Pre-Cut**:
- Each beam first selects top-b candidates (b << vocab_size), and then global top-k
- b = {32, 64, 128} vs. full vocab

**Evaluation**: Hit@K (quality), inference time (efficiency)

### Key questions

1. At present, the benefits under 3 token + beam=5 are not obvious, and a larger beam is needed to realize the advantages.
2. DBW’s schedule design: How is it related to codebook size? GR4AD does not provide a method to automatically determine the schedule.
3. Can be implemented as a companion to IDEA-gr4ad-0 (MGMR)

---

## IDEA-static-0: CSR Matrix constraint decoding (STATIC)

**Priority**: ~~P1~~ → ❌ Close
**Source**: STATIC (Google/YouTube, arxiv 2602.22647, Feb 2026)
**Status**: ❌ Closed - We have equivalent implementation: `ntp/model.py:SIDTrie` + `constrained_beam_search()`. Each decoding step passes `trie.valid_tokens(layer, prefix)` to filter invalid tokens, ensuring that the output is 100% real SID. All beam searches since EXP-017 use this constraint. STATIC's CSR matrix is ​​a GPU vectorization accelerated version, and the current 3-layer small trie does not require this optimization.

### Core Idea

STATIC flattens the prefix tree (trie) into a Compressed Sparse Row (CSR) matrix, converting irregular tree traversal into fully vectorized sparse matrix operations to achieve efficient constraint decoding on GPU/TPU.

Core problem: Generative recommendation requires constrained decoding (only valid SIDs are generated), and the traditional trie method is extremely slow on GPU (pointer tracking, irregular access).

Technology:
1. **Trie → CSR Matrix**: Flatten the hierarchical structure of the prefix tree into a static CSR matrix
2. **Vectorization transition**: Each step of decoding = sparse matrix multiplication, naturally suitable for GPU parallelism
3. **Support business constraints**: dynamic constraints such as content freshness, category restrictions, etc.

**YouTube production deployment**: **948x speedup** over CPU trie, **0.033 ms per step**, only 0.25% of inference time. Open source: `github.com/youtube/static-constraint-decoding`.

### Association with the current project

- Current beam search (`metrics/sid_prediction.py`) softmax on full vocab, no constrained decoding
- If effective SID constraints are implemented (only assigned SID combinations are generated), invalid IDs can be avoided → effective recall is improved
- **Cooperate with IDEA-gr4ad-4 (Dynamic Beam Search)**: first constrain the effective collection, and then do dynamic beam in the effective collection
- The current 3-layer × 1024 codebook, the effective SID is about 5M / (1024^3 ≈ 10^9) = 0.5% of the space - constrained decoding can cut out 99.5% of invalid combinations
- The code is open source for direct reference

### Experimental Design Draft

**Phase 1 — Constructing a valid SID trie**:
1. Extract all valid SIDs (5M items) from RKMeans assignment
2. Build a 3-layer prefix trie
3. Convert to CSR matrix

**Phase 2 — integrated into beam search**:
1. During each decoding step, use the CSR matrix to mask invalid tokens → softmax only on valid tokens
2. Evaluation: Recall@K and inference speed with/without constraints

### Key questions

1. Currently there are 3 layers of SID, and the trie depth is only 3 - the benefits of constrained decoding may be limited (the proportion of valid tokens in 1024 vocabs per layer is not low)
2. If you switch to OPQ (layers 16~64), the value of constraint decoding increases greatly - effective combinations are extremely sparse in exponential space
3. Dynamic constraints (such as category filtering) require modification of the CSR matrix structure

---

## IDEA-flame-0: GR inference system optimization (FLAME)

**Priority**: P2
**Source**: FLAME (arxiv 2509.22681, Sep 2025)
**Status**: To be discussed

### Core Idea

FLAME is a dedicated inference system for the GR model, with three core modules:

1. **Proximal Data Accelerator (PDA)**: CPU-GPU heterogeneous computing, decoupling feature preprocessing (CPU) and model inference (GPU) → **1.9x throughput, 1.7x latency reduction**
2. **Fused Kernel Engine (FKE)**: kernel fusion based on TensorRT → **4.6-6.1x acceleration**
3. **Dynamic Stream Orchestrator (DSO)**: Dynamically schedule concurrent requests → **1.3x throughput, 2.3x speedup under non-uniform distribution**

Core insight: The FLOP magnitude of the GR model (10^9~10^11) is 4 orders of magnitude higher than that of traditional DLRM, requiring a specialized serving system.

### Association with the current project

- No production-level serving optimization is currently required (research phase)
- But PDA’s CPU-GPU decoupling idea can be used for training: feature preprocessing uses CPU async preprocessing, and GPU focuses on forward/backward
- FKE's kernel fusion in the scenario where we use PyTorch = torch.compile / FlashAttention
- **Reference value**: Understand the delay breakdown and bottleneck points of production GR serving

### Key questions

1. The research stage has a low priority and will be evaluated in detail during production deployment.
2. Some optimizations (such as FlashAttention) are built into modern PyTorch

> **Supplement (2026-04-28)**: MTServe (arxiv 2604.22881, Meituan + Wuhan Univ + Nvidia) proposed a GR-specific hierarchical KV cache management system:
> - **Issue**: The GR model requires persisting a separate KV cache for each user (vs. LLM shared system prompt prefix). The HSTU model requires 160GB for only 1000 users × 10K tokens, far exceeding the 80GB of a single card A100
> - **Scheme**: GPU VRAM (primary) + Host RAM (backup) two-level storage, Page-Chunk dual-granularity management (GPU fine-grained paging + CPU coarse-grained chunking)
> - **Key technologies**: (1) Asynchronous offload pipeline hides I/O latency; (2) LRU replacement strategy maintains temporal locality; (3) Zero-copy eviction
> - **Results**: Meituan production data set 3.1x speedup (BS=8: 26.6ms vs 82.4ms), hit ratio 98.7%, KuaiRand-1K 3.04x speedup
> - **Complementary with FLAME**: FLAME solves CPU-GPU computing decoupling + kernel fusion, MTServe solves cross-request KV cache persistence and tiered storage
> - **Complementary to EARN**: EARN compresses KV cache size (register tokens → 80% reduction), MTServe expands KV cache capacity (GPU→RAM tiering)

---

## IDEA-earn-0: Register Token compressed inference (KV Cache reduced by 80%)

**Priority**: P1
**Source**: EARN (arxiv 2507.00715, Jul 2025, KDD 2025)
**Status**: To be discussed

### Core Idea

EARN found that the attention model of the LLM-based recommendation model has unique characteristics:

1. **Layer-wise Attention Sparsity Inversion**: Early layers are attention-intensive and information-rich, and later layers are highly redundant.
2. **Dual Attention Sinks**: The attention score is concentrated on the first and last tokens of the sequence

Based on this it is proposed:
- Place **register tokens** at the beginning and end of the input sequence
- **Early layer** (dense attention): The entire sequence is calculated normally, and the information is compressed into register tokens
- **Later layer** (sparse attention): only calculate register tokens and skip the rest of the sequence

**Result**: **3.79x speedup, 80.8% KV Cache reduction**, accuracy does not decrease but increases (better than general fine-tuning). KDD2025.

### Association with the current project

- The current `AutoregressiveNTPModel` is a 6-layer decoder, the sequence is very short (3 SID tokens), and inference is not a bottleneck
- But if:
  - Switch to long sequence (input side of IDEA-onemall-1 Query-Former)
  - Use LLM backbone (IDEA-plum-0)
  - Expand beam (IDEA-gr4ad-4)
  → Register token compression becomes critical
- **Similar to the idea of IDEA-gr4ad-1 (LazyAR)**: both "complete calculations are done in the front layer and simplified in the back layer". LazyAR simplifies autoregressive dependencies, EARN simplifies attention span
- Both can be combined

### Experimental Design Draft

**In LLM backbone scenario** (depends on IDEA-plum-0):
1. Add n_reg register tokens at the beginning and end of the input of Qwen3-0.5B
2. First K layer full sequence attention
3. The rear L-K layers only attend to register tokens
4. Selection of K: EARN found that attention began to become sparse at about 1/3

**Evaluation**: Recall@K vs Inference Latency vs KV Cache Occupancy

### Key questions

1. Currently 39.5M small model + no profit under short sequence → dependent on model/sequence expansion
2. Selection of the number of Register tokens: too few information losses, too many insufficient compression
3. Combined design with LazyAR: The optimization layers of the two are different (LazyAR optimizes autoregressive dependence, EARN optimizes attention span)

---

## IDEA-promise-0: Process Reward Model + Test-Time Scaling for GR

**Priority**: P1
**Source**: PROMISE (Kuaishou, arxiv 2601.04674, Jan 2026)
**Status**: To be discussed - Prerequisite: RL alignment link (EXP-037→039) needs to be completed first, PRM training data construction relies on beam search rollout infrastructure

### Core Idea

PROMISE identifies the **Semantic Drift** problem: In hierarchical SID generation, errors in early high-level tokens will irreversibly introduce the generated trajectory into irrelevant semantic subspaces.

Solution:
1. **Lightweight PRM (Process Reward Model)**: Evaluate the quality of each intermediate reasoning step (rather than just looking at the final result ORM)
2. **PRM-guided Beam Search**: Use PRM’s dense feedback to dynamically prune wrong branches (not just relying on token probability)
3. **Test-Time Scaling Laws**: Adding inference calculations can allow small models to match or even surpass large models

Core insight: **The test-time scaling law of LLM reasoning is reproduced in GR** — investing more calculations (more beams + PRM scores) in the reasoning phase can make up for the lack of model capacity.

Kuaishou large-scale platform online A/B verification: significantly improves recommendation accuracy while maintaining deployment efficiency.

### Association with the current project

- Current beam search only uses token probability sorting → PRM can provide better quality assessment of intermediate steps
- Complementary to IDEA-gr4ad-4 (Dynamic Beam Search): gr4ad-4 optimizes beam efficiency, promise-0 optimizes beam quality
- Test-time scaling revelation: The current 39.5M small model + PRM may surpass the basic beam search of future larger models
- PRM training requires: step-level annotated data (can be automatically constructed using Monte Carlo rollout)

### Key questions

1. PRM training data construction: step-level labeling cost is high → requires Monte Carlo automatic solution
2. PRM inference overhead: Each beam candidate requires PRM scoring at each step → delay tradeoff
3. Pre-reliance on NTP baseline (requires an available beam search foundation first)

---

## IDEA-grc-0: Generation-Reflection-Correction Decoding

**Priority**: P1
**Source**: GRC (Alibaba, arxiv 2602.23639, Feb 2026)
**Status**: To be discussed - Prerequisite: NTP baseline + GRPO infrastructure (EXP-026 already exists), but the three-stage GRC training data structure has not yet been designed

### Core Idea

GRC extends the standard single-shot decoding into a three-stage **Generation-Reflection-Correction** process:

1. **Generation**: Standard autoregressive generation of initial SID sequence (draft)
2. **Reflection**: Multi-granularity reflection — the model examines the quality of the generated sequence
3. **Correction**: Correct the generated trajectory based on the reflection results

Key optimizations:
- **GRPO-based RL**: perform GRPO optimization on the entire GRC trajectory, reward combines token-level and trajectory-level signals
- **Entropy-Guided Reflection Scheduling (EGRS)**: Dynamically allocate reflection budget when serving - high uncertainty trajectories require multiple reflections, low uncertainty direct output

Alibaba's large-scale industrial recommendation: **Advertising revenue +1.79%**, latency overhead is controllable (EGRS only reflects uncertain beams).

### Association with the current project

- Similar to LLM's self-reflection/self-correction but operates in SID token space
- Complementary to IDEA-s2gr-0 (Stepwise Reasoning): s2gr "thinks" before each step of generation, GRC "reflects and corrects" after the overall generation
- Also complementary to IDEA-promise-0 (PRM): PRM evaluates step quality, GRC allows corrections
- EGRS is the key: not all beams are reflected, only those with high entropy → control delay

### Key questions

1. Training data construction: (draft, reflection, corrected) automatic generation strategy of triples
2. Sequence length expansion: GRC increases by ~2x tokens → EGRS control is required
3. Prerequisite: NTP baseline + GRPO infrastructure (IDEA-onemall-2)

---

## Priority summary

| Priority | ID | Experiment | Reason |
|--------|-----|------|------|
| P1 | IDEA-gr4ad-4 | Dynamic Beam Search | Required when producing beam=512; can be used with IDEA-gr4ad-0 |
| ~~P1~~ ❌ | ~~IDEA-static-0~~ | ~~CSR constraint decoding~~ | ❌ There is an equivalent implementation: SIDTrie + constrained_beam_search, standard configuration starting from EXP-017 |
| P1 | IDEA-earn-0 | Register Token compression | 3.79x speedup, complementary to LazyAR, KDD 2025 |
| P1 | IDEA-promise-0 | PRM-guided Beam Search | Kuaishou online verification, test-time scaling unlocks the potential of small models |
| P1 | IDEA-grc-0 | Generation-Reflection-Correction | Alibaba +1.79% revenue, EGRS control delay, collaboration with GRPO |
| P1 | IDEA-orecv2-0 | FP8 PTQ inference acceleration | Kuaishou OneRec-V2, -49% latency +92% throughput, 0 quality loss |
| P1 | IDEA-nezha-0 | Self-Drafting Speculative Decoding | Taobao +1.2% revenue, <30ms (10x speedup), hash-set verification zero Model overhead |
| P2 | IDEA-flame-0 | GR Serving system | Production deployment reference, current Phase priority Low |

---

## IDEA-orecv2-0: FP8 Post-Training Quantization inference acceleration

**Priority**: P1
**Source**: Quantized Inference for OneRec-V2, Kuaishou (arxiv 2603.11486)
**Status**: To be discussed

### Core Idea

FP8 PTQ inference optimization of Kuaishou OneRec-V2 (4B parameters, 0.5B activated, fat-MoE architecture). Key findings: The weight/activation distribution statistics of the GR model are far more controllable than the traditional recommendation model (the variance is 5-6 orders of magnitude lower), close to LLM (Qwen3-8B). Therefore the quantitative techniques of LLM can be directly transferred. Specific plans:

1. **Per-channel weight quantization** (offline): Linear layer (Attention qkvo + Dense FFN) + grouped GEMM (MoE)
2. **Per-token activation quantization** (runtime dynamic scaling)
3. **FP8 TensorCore multiply + FP32 accumulation → cast back FP16**
4. **MoE block-wise quantization**: 1×128 activation, 128×128 weight granularity

With infrastructure optimization (TensorRT direct construction, RadixTopK, attention kernel optimization, MoE TMA kernel):
- Latency: 139ms → 70ms (-49%)
- Throughput: 205 → 394 (+92%)
- Online A/B: All core indicators of Kuaishou + Kuaishou Express Edition are not degraded

### Association with the current project

- The current stage focuses on model training, but FP8 is the only way to go when deploying
- Key insight: **GR model is naturally suitable for quantification** — unlike traditional recommendation models, no additional quantification-aware training is required
- OneRec-V2’s MoE + Transformer architecture is consistent with our possible future model architectures
- 42% throughput gain from FP8 quant alone → huge impact on deployment costs

### Experimental Design Draft

**Phase 1 — Distribution Analysis**:
- Analyze weight/activation distribution (variance, AbsMax, AbsP99) on the trained NTP model
- Compare data with OneRec-V2 and traditional recommendation models
- Determine whether our model also has "close to LLM" quantification-friendly properties

**Phase 2 — FP8 PTQ Inference**:
- Use PyTorch FP8 or TensorRT FP8 for inference
- Compare FP16 vs FP8: latency, throughput, Recall@K differences
- Requires H100 GPU (FP8 TensorCore)

### Key questions

1. The current model is small (not 4B), and the FP8 acceleration ratio may not be as significant as OneRec-V2
2. Requires Hopper architecture GPU (H100/H200) to support FP8 TensorCore
3. Phase 1 (distribution analysis) zero cost, you can do it first
4. More suitable for the model online deployment stage, and the current priority is lower than training optimization.

---

## IDEA-nezha-0: Self-Drafting Speculative Decoding for GR (NEZHA)

**Priority**: P1
**Source**: NEZHA (Alibaba + CityU HK, arxiv 2511.18793, WWW 2026)
**Status**: To be discussed — Prerequisite: NTP baseline + beam search extended to beam≥50

### Core Idea

NEZHA introduces speculative decoding into GR to solve the beam search decoding delay bottleneck (accounting for 60%+ of the total inference time). Three major innovations:

1. **Placeholder Prompt + Single Prefill**: Append L placeholder tokens `<SP_1>...<SP_L>` (L=SID length) at the end of the input. One prefill will obtain L+1 hidden states (context + each placeholder position) to avoid gradual autoregressive prefill.
2. **Autoregressive Draft Head**: A lightweight draft head is attached to the main model and contains:
   - `logit_head_l`: linear layer `[d_hidden, T_l]` (1024×512), mapping hidden state to token probability
   - `Transition_l`: RNN module, update context state `s_{l+1} = Transition(s_l, e_l)` (e_l is the embedding of the selected token)
   - Teacher-forcing training: ground-truth token as transition input
   - Beam search reasoning: select top-K tokens in each step and update context state respectively
3. **Model-Free Hash-Set Verification**: Encode multi-token SID into a unique integer (mixed-radix: `V_i = Σ t_l × Π_{j<l} T_j`), build hash set V, O(1) query verification. Only ~0.1% of valid IDs → filter 99.9% of hallucinations. draft valid ratio: 43% → 93%

### Key experimental data

| Metric | Vanilla Beam | NEZHA | Improvement |
|------|-------------|-------|------|
| Total Latency (Normalized) | 4.86 | 1.86 | **2.6x** |
| Decoding delay | 2.95 | 0.78 | **3.8x** |
| System latency | 0.91 | 0.08 | **11x** |

Public dataset (Llama-1B, beam=10): same accuracy (H@10 ≈ 0.056-0.041), latency ~10x lower (74ms → 7ms)

**Taobao Production Deployment** (Oct 2025):
- Model: 0.6B LLM, L=3, T_l=512, beam=512
- Latency: >1000ms → <30ms
- Online A/B: **+1.2% advertising revenue** (billion-level revenue)
- Services: hundreds of millions DAU
- Open source: `github.com/Applied-Machine-Learning-Lab/WWW2026_NEZHA`

### Association with the current project

- **Directly applicable**: We also use 3-token SID + beam search, and NEZHA’s placeholder prompt + draft head architecture can be directly transplanted
- **Complementary to SIDTrie**: Currently `constrained_beam_search` uses SIDTrie to filter step by step, and NEZHA's hash-set filters once in the final step. The two can be combined: draft head generation + SIDTrie step-by-step constraints + hash-set final verification
- **Cooperated with IDEA-gr4ad-4 (Dynamic Beam)**: dynamic beam width reduces the total number of beam operations, NEZHA reduces the LLM forward pass of each beam operation
- **Complementary with IDEA-promise-0 (PRM)**: NEZHA accelerates draft generation, PRM improves draft quality assessment
- Small training overhead: draft head only adds logit_head (linear layer) + RNN transition, parameter amount << main model

### Experimental Design Draft

**Phase 1 — Draft Head implementation** (on existing NTP checkpoint):
1. Attach draft head to `NTPModel`: one `nn.Linear(d_model, vocab_size)` + shared RNN transition for each layer
2. Modify training: add `<SP_1>...<SP_3>` placeholder tokens to input, teacher-forcing training draft head
3. Evaluation: draft head single token accuracy vs original autoregressive accuracy

**Phase 2 — Hash-Set Verification**:
1. Build valid SID hash set (mixed-radix encoding) from `semantic_ids.npy`
2. Replace the SIDTrie constraint in the final step of beam search with hash-set filtering
3. Comparison: valid ratio, Recall@K, delay

**Phase 3 — Federated Beam Search**:
1. Single prefill + draft head beam search (beam=50/100/500)
2. Comparison of vanilla beam search vs NEZHA: Recall@K vs delayed tradeoff
3. Combine SIDTrie (step by step) + hash-set (final) double verification

### Key questions

1. The current beam=5 is too small, and the advantages of NEZHA only appear in large beams (≥50) - the beam needs to be expanded first
2. Draft head’s RNN transition vs our available MLP/GRU options
3. Placeholder tokens require additional vocab entries — affects tokenizer
4. Integration with KV-cache inference (`forward_cached`): NEZHA does not require the gradual update of KV-cache, instead single prefill + draft head

---

## IDEA-snap-0: SID-to-Item correlation guided disambiguation + Depth>Breadth retrieval

**Priority**: P1
**Source**: Snapchat SIDs Industry Report (arxiv 2604.03949, SIGIR 2026 Industry Track)
**Status**: To be discussed

### Core Idea

Snapchat reports two key engineering decisions that led to the launch of SID as a Generative Retrieval, increasing video shares from +0.13% to +4.39% in short-form video A/B:

**1. Relevance-guided intra-bucket disambiguation**

The cardinality of SID is much smaller than the number of items, and there are usually multiple items in the same SID bucket. When the GR model beam search outputs top-N SIDs, how to expand each SID into an actual item list determines the final quality:
- **Random mapping**: Random selection within the bucket — weak online income (+0.13% views)
- **Relevance-guided mapping**: Use item-level heuristics (cumulative viewing time, freshness, CTR history, etc.) to secondary sort within buckets — online **+0.57% views / +2.54% sends / +3.55% re-posts / +4.39% shares**

The essence: Let the generative model be responsible for "semantic category selection", and the lightweight heuristic be responsible for "selection of items in the bucket", and the two stages are decoupled.

**2. Depth > Breadth retrieve allocation**

With a fixed retrieval budget (e.g. 1000 items):
- **Breadth**: Take 10 for each SID, covering 100 SIDs → neutral
- **Depth**: Take 100 for each SID, only use top-10 SID → +0.57% views

It shows that the confidence of the model on the top-rank SID is very high, and it is better to concentrate the budget into a few high-confidence buckets than to spread it over a large number of low-confidence buckets.

**3. Meta Observation: Uniqueness is not the gold standard for SID quality**

Through Amazon Beauty experiments (Table 5) and Snap internal data, Recall@10 is saturated after uniqueness exceeds ~70%, and there is no gain in continuing to pursue 90%+ uniqueness. It is instructive for us to evaluate the tokenizer: the existing MLP-FSQ (collision 10.7%, uniqueness 89%) is already in the "high enough" range, and there is no need to roll up the collision.

### Association with the current project

- **`metrics/sid_prediction.py` is directly related**: After the current beam search outputs top-N SID, use `SIDTrie` or `semantic_ids.npy` to do SID→item mapping. If one SID corresponds to multiple items, our current strategy is "take all" (prefix cascade reward) or "by id order", without business-side sorting.
- **prefix cascade semantic alignment**: EXP-026's BehaviorReward L0/L1/L2 prefix cascade is essentially what Snapchat calls the "depth" idea - when the full SID match is 0, the prefix is reduced by one level, which is equivalent to "less SID × more items"
- **No business side features**: We currently do not have item-level popularity/freshness/CTR features, but they can be used:
  - `recency`: last interaction time (exportable from NTP training data)
  - `popularity`: frequency of occurrence (item count)
  - `SID frequency`: the proportion of items in the bucket appearing in the training set
- **Collaboration with IDEA-gr4ad-4 (Dynamic Beam)**: DBW controls the beam width, and this idea controls the expanded item allocation strategy - orthogonal

### Experimental Design Draft

**Phase 1 — Relevance-guided intra-bucket mapping (eval only)**:

1. Introduce the sorting key in the beam-search-to-item expansion phase of `metrics/sid_prediction.py`:
   ```python
   bucket_items = sid_to_items[sid] # List[item_id]
   bucket_items.sort(key=lambda iid: item_priority[iid], reverse=True)
   ```
   Among them `item_priority[iid]` option:
   - `popularity`: training set interaction frequency (log-scaled)
   - `recency`: `max(timestamp)` normalization
   - `pop × recency`: combination
2. Do re-eval on EXP-020 baseline checkpoint (no need to retrain), compare:
   | Config | R@10 | R@100 | R@500 |
   |--------|------|-------|-------|
   | random mapping (baseline) | ? | ? | 66.2% |
   | popularity | ? | ? | ? |
   | recency | ? | ? | ? |
   | pop × recency | ? | ? | ? |

**Phase 2 — Depth vs Breadth budget allocation**:

Fixed total retrieval budget = 1000, scan (top-K SID, per-SID items) combination:
- (50, 20), (100, 10), (200, 5), (500, 2), (1000, 1) — breadth-first
- (10, 100), (20, 50), (5, 200) — depth-first

Report Recall@1000 and Recall@100. Snapchat's discovery predictions are (10, 100) or (20, 50) optimal.

**Phase 3 — Supervised learnable scorer (optional)**:

If the heuristic is saturated, train a lightweight MLP to perform bucket sorting on (sid, item_feats), using clicks/views in the NTP log as labels.

### Key questions

1. Currently eval uses `semantic_ids.npy` to do SID→item mapping. Is there really a bucket conflict in actual beam search? First count the average number of items in each SID bucket in the current training/eval set
2. Are the popularity / recency features redundant with the prefix cascade L0/L1 fallback? It is necessary to verify the two paths of prefix cascade eval and "full SID matching + sorting within the bucket" eval separately.
3. The SID uniqueness 70% threshold conclusion is not necessarily universal—the collision distribution of our 32-bit SID + 1.09M items is different from Amazon Beauty, which can supplement internal verification

### Related ideas

- IDEA-gr4ad-4 (Dynamic Beam): Adjust beam width → This idea adjusts the expansion strategy, orthogonal
- IDEA-adasid-0 (Adaptive Collision): Reduce collision from tokenizer side → This idea does not touch tokenizer, pure eval side optimization
- IDEA-r3vae-0 / FORGE proxy metrics: Snapchat's "uniqueness is not the gold standard" is new evidence of this line
