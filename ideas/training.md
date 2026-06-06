# Training (Training Goals and Strategies)

[English](training.md) | [Chinese](training.zh.md)

Training signal design of NTP model: auxiliary loss, sample weighting, multi-behavior fusion, etc. Improve training quality without changing the model architecture.

**Scope of influence**: `metrics/sid_prediction.py`, `model/train.py`, `data/export_behavior.py`

---

## Evolution path

```
Pure CE loss + MoE aux loss (current baseline)
├── IDEA-mtgr-0: User-Level Sequence Packing + Causal Mask (eliminate sliding window)
│ └── Meituan CIKM 2025: long sequence + dynamic mask, greatly improving training efficiency
├── IDEA-onemall-0: In-Batch Contrastive Loss (continuous semantic supervision)
│ └── EXP-013 S-tier baseline has been established and can be tested directly
├── IDEA-sid-4: Token-Space MTP auxiliary Loss (fine-grained token CE)
│ └── Complementary to onemall-0: token-level vs embedding-level
│ └── EXP-015: irreducible loss=2.522, room for improvement is limited by tokenizer
├── IDEA-sid-5: Codebook Embedding aggregation (item representation)
│ └── Depends on IDEA-sid-0 Phase 2 (OPQ long ID)
├── IDEA-gr4ad-2: Value-Aware training (eCPM token + sample weighting)
│ └── Introducing business value signals
├── IDEA-sigma-0: Instruction-driven multi-task GR + adaptive probabilistic fusion
│   └── AliExpress: instruction-following + adaptive decoding distribution
├── IDEA-lemur-0: End-to-end multi-modal + Memory Bank incremental representation
│ └── Douyin: joint optimization + memory bank to solve caching problem, QAUC +0.81%
├── IDEA-oneloc-5: Multi-behavior sequence fusion
│ └── Distinguish different behavioral intensities of click/buy/expose
├── IDEA-dualgr-0: Exposure-Aware NTP Loss → EXP-014 data end completed, NTP integration to be advanced
├── IDEA-tbg-0: Data Recency → EXP-016 ✅ 14d optimal, recency > volume strong verification
└── IDEA-plum-0: LLM Continued Pre-Training (Google/YouTube)
└── Pre-training LLM → CPT on SID corpus → Fine-tune, verified by billions of users
```

---

## Current Conclusion (2026-04-17)

**NTP baseline has been established, the 14d data window is confirmed to be optimal, and scaling law reveals that the tokenizer is the bottleneck. **

### Current config
```
Model: S-tier (17.5M active, 256d, 6L, 8E top-2 MoE)
Data: 14d window (03-17~03-31), ~130M tokens, ~1.69M users
Training: 1 epoch, batch=512, lr=1e-3, CosineAnnealing
```

### Key experimental data

| Experiment | Discovery | Key Metric |
|------|------|---------|
| EXP-013 | S-tier NTP baseline established | PPL=27.05, R@500=58.5% |
| EXP-014 | ENTP negative sample export completed | 130M positive sample rows, 31% have negative samples |
| EXP-015 | Scaling law L(N)=2.522+2055/N^0.456 | irreducible loss=2.522 (PPL≈12.5), M+ loss=2.94 |
| EXP-016 | Data recency > volume, 14d optimal | U-shaped loss curve, more data makes it worse |

**Core insight**: Tokenizer (MLP-FSQ 32-bit) is the current bottleneck rather than model size. M+ (101M) only reduces loss by 0.06 (2.9960→2.9371) compared to S (17.5M). Improving the training signal (contrastive loss, ENTP) is more effective than scaling up the model.

---

## IDEA-mtgr-0: User-Level Sequence Packing + Causal Mask Training

**Priority**: ~~P0~~ → ✅ Implemented
**Source**: MTGR (Meituan, arxiv 2505.18654, CIKM 2025)
**Status**: ✅ Implemented — `ntp/train.py:train_packed()` + `build_unified_sequences()` completed, all training has used packed mode (since EXP-013)

### Core Idea

Currently `ntp/train.py:build_sequences()` uses Python for-loop to perform sliding window segmentation on 4.5M users, generating tens of millions of independent (input_30, target_3) samples. There are two problems with this:

1. **Slow build**: Pure Python loop, 4.5M users × sliding window = minute-level delay, DDP other ranks wait until timeout
2. **Information waste**: A large number of overlapping tokens between sliding windows are repeatedly encoded, and different windows of the same user cannot share context.

The core inspiration of MTGR: **Without sliding window segmentation, directly combine each user's complete behavior sequence into a long SID token sequence**, and use causal mask to let the model predict the SID of the next item at each position.

```
User A's complete sequence: [item1_L1, item1_L2, item1_L3, item2_L1, item2_L2, item2_L3, ...]
                  ↓ causal attention ↓
Predict the next token at each position, using per-layer output_proj
```

**Equivalent to training all sliding window positions at the same time, but only doing one forward pass. **

MTGR also proposed **Dynamic Masking** to prevent information leakage:
- Static context (user profile) → visible everywhere (bidirectional attention)
- Dynamic behavioral sequences → strictly causal (only look at the past)
- If pack multiple users in batch → block-diagonal mask to prevent cross-user attention

### Association with the current project

**Directly solve pain points**:
- `build_sequences()` consumes minutes → Instead, directly build user-level long sequences, just numpy splicing, completed in seconds
- The amount of data remains the same but there are fewer forward passes (one forward for one user vs N forwards for N sliding windows)
- Training throughput is greatly improved: the middle hidden states of the same user are reused in all locations

**Differences from MTGR**:
- MTGR does ranking (discriminant), we do NTP (generative) → causal mask is naturally applicable
- MTGR aggregates candidates (K candidates from the same user), and we aggregate history (N historical items from the same user)
- MTGR uses HSTU (SiLU attention), we use standard Transformer → directly available

### Experimental Design Draft

**Phase 1 — User-level long sequence + Causal Mask (core changes)**:

Replace `build_sequences()` → `build_packed_sequences()`:
```python
def build_packed_sequences(sid_dict, behavior_data):
    """Each user → a long SID token sequence + causal mask"""
    # 1. Group by user + sort by time (vectorization, pandas groupby)
    # 2. Each user: [item1_tokens, item2_tokens, ...] → flat list
    # 3. During training: causal attention, predict the next token at each position
    # target = input shifted by 1 (standard LM training)
    # 4. per-layer output_proj is selected based on position % n_layers
```

The training loop is changed to standard LM style:
```python
# No longer distinguish input_tokens / target_tokens
# Do causal attention on the entire sequence, and loss is calculated at all positions
logits = model(packed_sequence) # (B, T, C_layer)
loss = CE(logits[:, :-1], packed_sequence[:, 1:])
```

**Phase 2 — Multi-User Packing (further improves GPU utilization)**:
- Pack short sequence users to the same batch position (similar to LLM document packing)
- Block-diagonal attention mask prevents cross-user leakage
- Requires FlashAttention's varlen API or custom mask

**Phase 3 — Dynamic Masking (introducing side-info)**:
- User profile token → Bidirectional attention
- Behavior sequence → strictly causal
- Candidate item → only look at yourself (if ranking)

### Change files

1. `ntp/train.py` — `build_sequences()` → `build_packed_sequences()`, the training loop is changed to LM style
2. `ntp/model.py` — `NTPModel.forward()` supports long sequences + per-position loss
3. `ntp/baseline.py` — `SIDSequenceDataset` adapts to new data formats

### Key questions

1. **Position embedding length**: The current `pos_emb` maximum is `seq_len + n_sid_layers`. User-level long sequences may have hundreds of items × 3 tokens = thousands of tokens → need to extend pos_emb or use RoPE instead
2. **Variable length sequence padding**: The sequence lengths of different users vary greatly → naive padding wastes calculations. Requires packing or bucket batching
3. **Eval remains unchanged**: beam search still uses a fixed n_items window, but the training method changes
4. **Group LayerNorm (MTGR)**: Different token types (different SID layers) group LayerNorm, is it helpful?

---

## IDEA-onemall-0: In-Batch Contrastive Auxiliary Loss for NTP Model

**Priority**: P0
**Source**: OneMall §3.2 Supervised Objectives
**Status**: ❌ Tested, negative results — EXP-022 All 5 configs are no better than baseline (see experiments/logs/ EXP-022 for details)

> **Experimental Conclusion (2026-04-22)**: EXP-022 tested a total of 5 configs α∈{0.01,0.1,0.5}, dim∈{128,256}, τ∈{0.05,0.07}. The best α=0.01 is only +0.7pp R@500 but PPL degrades by +0.84. The bigger α is, the worse it is. Root cause: SID is a discrete token, and InfoNCE continuous space alignment is not helpful for discrete prediction. **No more chasing. **

### Core Idea

While NTP autoregressive training, a two-tower style in-batch contrastive loss is added as an auxiliary objective. Specific method: The hidden layer representation s₃^L of the last SID token (complete SID sequence information has been encoded) is compared with the target item embedding f_item for InfoNCE comparative learning. OneMall reported that the task reached **98% accuracy@1**, indicating that s₃^L has encoded item information with high quality.

The role of auxiliary comparison loss:
- Provide continuous supervision signals in the embedding space for Transformer (NTP only has discrete token CE loss)
- Prevent SID representation from degenerating into only caring about token classification and losing semantic continuity
- Regularization effect to improve generalization

### Association with the current project

- The NTP model is in `metrics/sid_prediction.py:AutoregressiveNTPModel`, currently only `CE_loss + 0.01 * aux_loss(MoE balance)`
- item embedding already has ready-made Qwen3 embedding (`model/encode.py`), which can be loaded directly during training
- Very low implementation cost: add an MLP projection head at the s₃ position → InfoNCE with in-batch negatives
- **Orthogonal to IDEA-sid-1 (co-signal embedding)**: IDEA-sid-1 improves embedding itself, and this IDEA improves NTP model training

### Experimental Design Draft

**Modify `metrics/sid_prediction.py`**:
1. Added `ContrastiveHead`: MLP(embed_dim → 128) projected to contrast space
2. Take the hidden layer output at the s₃ position → ContrastiveHead → l2_normalize
3. Target item embedding → MLP(1024 → 128) → l2_normalize
4. InfoNCE loss (temperature=0.05, in-batch negatives)

**training loss**:
```
L = L_NTP + 0.01 * L_moe_balance + α * L_contrastive
```

**Variables**:
- α ∈ {0.01, 0.1, 0.5, 1.0}
- projection dim ∈ {64, 128, 256}
- temperature ∈ {0.05, 0.07, 0.1}

**Baseline**: EXP-016 14d-S (PPL=27.05, loss=2.9960, R@500=58.5%)

**Evaluation indicators**: beam search Recall@{10,50,100,500}, SID accuracy@{1,2,3}, training convergence speed

### Key questions

1. The batch size needs to be large enough to provide a sufficient number of in-batch negatives — what is the current batch size? may need to be increased
2. Does the s₃ hidden layer need stop-gradient (asymmetric design) or backprop on both sides?
3. In the early stage of training, contrastive loss may dominate the gradient, and a warmup strategy is required (pure NTP first for several epochs and then add contrastive)

---

## IDEA-sid-4: Token-Space MTP auxiliary Loss (applicable to autoregressive models)

**Priority**: P1
**Source**: RPG (KDD'25, arxiv 2506.05781) §2.2.1 Multi-Token Prediction
**Status**: To be discussed — subject to irreducible loss floor

> **NTP stage update (2026-04-17)**: EXP-015 scaling law shows irreducible loss a=2.522 (PPL≈12.5), M+ (101M) has reached loss=2.94, only 0.42 from floor. MTP auxiliary loss is still valuable but the room for improvement is limited by the tokenizer upper limit. The main benefit is the cold start item and R@10 accuracy improvement, rather than a significant reduction in loss.

### Core Idea

RPG's MTP loss decomposes item prediction into the sum of independent CE losses for each token: ℒ = -Σⱼ log P(c_j | s). This has two key advantages over traditional item-level CE:
1. **Fine-grained semantic learning**: Optimized in token space (M categories) rather than item space (N >> M categories), the model learns sub-item level semantic features
2. **Cold start friendly**: Low-frequency items and high-frequency items share tokens, and sufficient training signals are obtained through token co-occurrence. RPG significantly outperforms TIGER in all frequency buckets ([0,5] to [16,20])

**Key Insight**: This loss does not require parallel prediction — it can be added as an auxiliary objective to any SID model. Even in the autoregressive model, the hidden layer representation h_L of the last token encodes the complete sequence information, and MTP loss can be applied to h_L to enhance semantic understanding.

### Association with the current project

- The current NTP model (`metrics/sid_prediction.py:AutoregressiveNTPModel`) only has token-by-token CE loss + MoE aux loss
- **Complementary to IDEA-onemall-0 (In-Batch Contrastive Loss)**: onemall-0 uses item embedding for comparison, and this IDEA uses token-level CE for fine-grained supervision
- Even if you end up going the autoregressive route (parallel prediction without RPG), MTP-assisted loss is a valuable regularization
- If you use IDEA-sid-0 (OPQ parallel ID), MTP is primary loss

### Experimental Design Draft

**Option A — as an auxiliary loss for autoregressive models**:
1. Get the hidden layer representation h_3^L of the last SID token position
2. Add m independent MLP projection heads to h_3^L (m = number of SID tokens)
3. Each head outputs M-dimensional logits → CE loss
4. Total loss: `L_NTP + α * L_MTP + 0.01 * L_moe`

**Option B — directly as parallel prediction primary loss** (= IDEA-sid-0 Phase 2):
1. User sequence → Transformer encoder → s
2. s → m MLP heads → m softmax → MTP loss
3. Inference: graph-constrained decoding

**Variables** (Option A):
- α ∈ {0.1, 0.5, 1.0}
- Whether to overlap with IDEA-onemall-0 (contrastive loss)

**Evaluation**: SID accuracy, beam search Recall@K, cold start item subset Recall

### Key questions

1. Solution A requires the hidden layer at the last position to simultaneously encode "all tokens of the next item" information - does it conflict with teacher forcing of autoregressive training? (h_3 has already seen the first 3 tokens of target during teacher forcing)
2. Is it more reasonable to use the hidden layer h_0 at the BOS position (which only encodes the user sequence and does not see the target token)?
3. Relationship with IDEA-onemall-0: Both apply additional loss at the same hidden layer position, and there may be gradient conflicts.

---

## IDEA-sid-5: SID Codebook Embedding aggregation represented as Item

**Priority**: P2
**Source**: RPG (KDD'25) §2.1.2 Semantic ID Embedding Aggregation
**Status**: To be discussed

### Core Idea

RPG uses the mean/max pooling of the codebook embedding of the SID as the item representation, replacing the original high-dimensional embedding. Each codebook j has a learnable embedding table E_j ∈ ℝ^(M×d). SID of item = (c_1, ..., c_m), which is expressed as:

`v_item = Pool(E_1[c_1], E_2[c_2], ..., E_m[c_m])`

In this way, the dimension represented by the item = d (the same as the token embedding dimension), has nothing to do with the total number of items N. All items share m codebooks of size M, and the total embedding parameters = m × M × d (much smaller than the full embedding table of N × d).

### Association with the current project

- The item embedding of the current NTP model is the lookup + positional encoding of the SID token, and a similar codebook embedding has been implicitly used.
- RPG's aggregation method is more explicit: mean pooling all codebook embedding → single vector representation
- Can be used for: (1) item retrieval (2) item cold start (3) item feature as ranking model
- But the current NTP model only has 3 tokens (RKMeans), and the aggregation benefits are not large. If you switch to OPQ (16~64 tokens), the aggregation method becomes important

### Experimental Design Draft

**Prerequisite: IDEA-sid-0 Phase 2 (OPQ + parallel prediction model)**

**Verification**:
1. After training the parallel prediction model, extract the codebook embeddings
2. Do mean/max pooling → item vector for each item
3. Use item vector to do ANN retrieval → compare recall of graph decoding
4. Analysis: Does pooled embedding retain sufficient semantic distinction?

**Evaluation**: cosine similarity distribution, retrieval recall@K, t-SNE visualization

### Key questions

1. Will mean pooling lose the interaction information between tokens? The RPG paper does not ablate mean vs max
2. It only makes sense in OPQ (long ID) scenarios — the mean pooling of 3 tokens is too rough
3. Relationship with FAISS retrieval: If the quality of pooled embedding is good enough, traditional ANN can be used instead of graph decoding

---

## IDEA-gr4ad-2: Value-Aware training target (VSL + eCPM Token)

**Priority**: P1
**Source**: GR4AD §VSL
**Status**: To be discussed

### Core Idea

GR4AD introduces two value-aware mechanisms in NTP training: (1) eCPM Token Prediction - Appends a discretized eCPM token at the end of the semantic ID sequence, allowing the model to predict "what to push" and "how much it is worth" at the same time; (2) Value-Aware Sample Weighting - weights training samples according to the user's long-term value and behavioral depth (purchase > click).

### Association with the current project

- `metrics/sid_prediction.py` The current training target is pure CE loss, all samples are equally weighted
- There are behavior types (clicks, purchases, collections, etc.) in our data, which are defined in `data/export_behavior.py`
- The idea of eCPM token can be generalized to **any business value token** — such as item popularity bucket, CTR bucket, etc.
- **Complementary to IDEA-sid-1 (collaborative signal enhancement)**: IDEA-sid-1 improves embedding representation, and this IDEA improves the training signal

### Experimental Design Draft

**Variable 1 — value token appended**:
- Discretize a continuous indicator of the item (such as frequency of behavior, popularity) into N buckets
- Semantic ID expanded from `"L1_L2_L3"` to `"L1_L2_L3_V"`, V ∈ {0, ..., N-1}
- NTP model continues to predict V token after predicting L3
- During inference: The logits of V token can be used as auxiliary ranking signals (similar to GR4AD using eCPM for reranking)

**Variable 2 — Sample weighting**:
- Purchase sample weight=3.0, collect weight=2.0, click weight=1.0 (parameters need to be adjusted according to data distribution)
- Add sample weight to `sid_prediction.py` training loop

**Evaluation**: Hit@K (basic), weighted Hit@K (high-value items have higher weights), value token prediction accuracy

### Key questions

1. Are the business value signals in our demo data sufficient? If there is only click data, sample weighting degenerates into equal weighting
2. The value token increases the sequence length → the reasoning cost increases, but only 1 token is added, which is acceptable
3. Selection of the number of discretization buckets N: too few will not provide enough information, too many will lead to long tail sparseness

---

## IDEA-oneloc-5: Multi-behavior Sequence fusion

**Priority**: P1
**Source**: OneLoc §2.3.1 Multi-behavior Sequence
**Status**: To be discussed

### Core Idea

OneLoc distinguishes three behavior sequences: watch (browse), click (click), pay (purchase), and the sequence length of each behavior is different (256/32/10). The three sequences are uniformly input to the encoder after concat. Different behaviors represent interest signals of different strengths.

### Association with the current project

- Currently `data/export_behavior.py` exports behavioral data, but the processing method needs to be confirmed
- The current NTP model (`metrics/sid_prediction.py`) input is a single sequence
- If we have multiple behavioral signals (impression/click/purchase/collection), it may be more effective to separate the different behavioral sequences rather than mixing them together
- **Intersection with IDEA-sid-1 (Coordinated Signal Enhancement)**: The behavioral sequence itself is the source of the coordinated signal

### Experimental Design Draft

**Prerequisite**: Behavior data needs to contain behavior type annotations

**Plan**:
- Separate sequences by action intensity: `S_expose` (long), `S_click` (medium), `S_purchase` (short)
- Each sequence is independently embedding → concat → input encoder
- Or: Mark each item with behavior type embedding, unify the sequence but add type signals

**Evaluation**: NTP recall of a single mixed sequence vs. a branched sequence

### Key questions

1. Does the behavior data contain behavior types? You need to check the schema of `data/export_behavior.py`
2. How to determine the sequence length ratio of different behaviors (OneLoc uses 256/32/10)
3. Implementation complexity: data pipeline + model input processing needs to be modified

---

## IDEA-plum-0: LLM Continued Pre-Training for Generative Recommendation

**Priority**: P1
**Source**: PLUM (Google/YouTube, arxiv 2510.07784, Oct 2025)
**Status**: To be discussed

### Core Idea

PLUM is an LLM-based generative recommendation framework deployed on a large scale by YouTube. Its core is three-stage training:

1. **Item Tokenization via Semantic IDs**: Video → SID Mapping
2. **Continued Pre-Training (CPT)**: Continue to pre-train LLM on recommended domain data to let the model learn the SID vocabulary and user behavior patterns
3. **Task-Specific Fine-Tuning**: Directly train the model to generate the SID of recommended items based on user context.

Key findings:
- CPT is a key step in adapting a general LLM to a recommendation model
- PLUM implements **substantial improvements** compared to YouTube's highly optimized production model (large-scale embedding table)
- Deployed to **Billions of YouTube Users**

### Association with the current project

- The current NTP model is a 39.5M small model trained from scratch, without utilizing the knowledge of pre-trained LLM
- PLUM proves that LLM pre-trained knowledge (world knowledge + sequence modeling) is still valuable even in non-natural language tasks such as recommendation
- **Potential experiment**: Use Qwen3-0.5B for CPT → fine-tune to replace the current `AutoregressiveNTPModel` trained from scratch
- Directly related to IDEA-oneloc-4 (Scaling Law): LLM backbone comes with parameter scaling, you only need to study the CPT data volume and sequence length

### Experimental Design Draft

**Option A (Lightweight — LoRA CPT)**:
1. Base: Qwen3-0.5B (same series as the current embedding model)
2. Expanded vocabulary: Add SID vocab (1024 tokens per layer → 3072 new tokens in total)
3. CPT data: user behavior sequence SID → construct "user_seq → next_item_sid" sample
4. LoRA fine-tune (rank=64), 8xA100, ~several hours
5. Evaluation: Qwen3-0.5B-CPT vs current AutoregressiveNTPModel’s Recall@K

**Option B (Weight — Full CPT)**:
- Full fine-tune Qwen3-0.5B on SID corpus
- Greater computational cost, but higher ceiling

### Key questions

1. The computational cost of CPT for 0.5B model: 8xA100 Can it be completed in a reasonable time (< 1 day)
2. SID vocab extension: embedding initialization strategy for new tokens (random vs. semantic initialization)
3. A fair comparison with the current 39.5M model: the parameter difference is 10x+, and FLOPS needs to be compared at the same time

---

## IDEA-onerec-1: RSFT (Reject Sampling Fine-Tuning — filtering out low-quality training samples)

**Priority**: P1
**Source**: OneRec (arxiv 2506.13695v4) §Post-training
**Status**: To be discussed

### Core Idea

The post-training phase of OneRec does not directly continue training on the full amount of data, but first uses **Reject Sampling** to filter: sort the exposed sessions according to the user's playback time, **discard 50% of the low-quality sessions**, and only perform NTP fine-tuning on high-quality data.

This solves the "exposure ≠ likes" problem - content that users have been recommended but not finished should not be used as positive samples for training.

### Association with the current project

- The current NTP training (`sid_prediction.py`) uses all behaviors with `action > 0` as positive samples, including a large number of low-quality interactions (clicked but did not finish)
- The essence of RSFT is **training data quality control**, without changing the model architecture, and the implementation cost is extremely low
- Can also be applied in EXP-007 (contrastive fine-tune): only train with high-quality pairs

### Experimental Design Draft

- Filter training data by `event_cnt` or behavior intensity (like/share > click > view)
- Comparison: NTP recall of full volume vs top 50% high-quality data

### Key questions

1. Our behavioral data has `event_cnt` but no playback duration - we need to find alternative quality indicators
2. The filtering ratio of 50% may be too aggressive and needs to be adjusted.

---

## IDEA-onerec-2: SID replaces VID as Encoder input (eliminates sparse Embedding Table)

**Priority**: P2
**Source**: OneRec (arxiv 2506.13695v4) §Semantic ID vs VID Input
**Status**: To be discussed

### Core Idea

OneRec found in a 2.6B scale experiment that using Semantic ID token directly as the item input of the encoder (instead of traditional VID sparse embedding) has equivalent or even better performance (P-score +1.74%). The advantage is that the huge sparse embedding table is eliminated (N × d parameters) and replaced by a very small SID codebook embedding (L × K × d parameters).

### Association with the current project

- The current NTP model uses SID tokens as input, which is already this paradigm
- But OneRec has proven that: at a larger model scale, SID input will not lose performance and will bring huge parameter efficiency improvements.
- Inspiration for us: No need to maintain item embedding table, SID codebook embedding is enough

### Key questions

1. There may be no difference in small models (5M probe)
2. More valuable in LLM backbone (IDEA-plum-0) scenario

---

## IDEA-dualgr-0: Exposure-Aware NTP Loss (ENTP-Loss)

**Priority**: P1
**Source**: DualGR (Kuaishou, arxiv 2511.12518, Nov 2025, WWW 2026)
**Status**: EXP-014 Experiment in progress - PySpark negative sample export completed, NTP integration to be verified

> **NTP phase update (2026-04-17)**: EXP-014 Completed PySpark-side ENTP negative sample export and data verification (130M rows, 31% with negative samples). Python side `load_exposure_neg_data()` has been implemented. L0 layer collision causes some negative samples to share the L1 cluster with positive samples, which needs to be processed in ENTP loss. NTP integration will be advanced in the next phase.

### Core Idea

DualGR found that standard NTP loss only learned "what the user clicked", but ignored the strong negative signal of "the user looked but didn't click anything". ENTP-Loss introduces **exposure-aware negative samples**:

- treat **unclicked exposures** as **coarse-level hard negatives**
- Use these negative samples to enhance the learning signal at the SID first layer (coarse level)
- Effect: The model identifies user interest fading faster (timely interest fade-out)

DualGR also proposes:
1. **Dual-Branch Long/Short-Term Router (DBR)**: Separate long-term and short-term interests, selective activation
2. **Search-based SID Decoding (S2D)**: restrict fine-level decoding to coarse bucket

Kuaishou short video online A/B: **video views +0.527%, watch time +0.432%**. WWW 2026.

### Association with the current project

- Currently NTP training (`sid_prediction.py`) only uses positive samples (user behavior sequences), no negative signals
- ENTP-Loss is a training improvement with **zero architectural changes**: just add exposed unclicked SIDs as negative samples in the loss calculation
- Complementary to IDEA-onerec-1 (RSFT): RSFT filters low-quality positive samples, ENTP introduces high-quality negative samples
- Orthogonal to IDEA-onemall-0 (contrastive loss): onemall-0 compares in the embedding space, ENTP introduces negative signals in the CE loss of NTP

### Experimental Design Draft

**Implementation**:
1. For each training sample, collect the unclicked item SIDs of the same session.
2. In NTP loss, the softmax probability of coarse level (first layer SID token):
   - Reduce the probability of unclicked-exposure SID token
   - Specific: Add margin/penalty items to CE loss
3. Variable: penalty weight, use negative signal only in L1 or all layers

**Evaluation**: NTP recall@K, paying special attention to the "interest change" scenario (the user's recent behavior has shifted to new categories)

### Implementation record

**PySpark side ENTP negative sample export (2026-04-16)**:

Implementation method: `data/export_exposure.py` Added ENTP section, Spark SQL window function
`pos_grp = cumsum(action_bitmap > 0)` segmentation, each non-positive segment is used as the negative sample of the next positive,
COLLECT_LIST + SORT_ARRAY + SLICE takes the nearest K=5. Output compact parquet `feed_user_exposure_neg/`.
Python side `load_exposure_neg_data()` loads ~130M rows (seconds), `_build_sequences_from_exposure()`
Only do iid→L0 token mapping.

**Data verification - PySpark export vs old streaming walk comparison (03-01~03-31)**:

| Metric | PySpark Export | Old Streaming Walk (Contrast) | Description |
|---|---|---|---|
| Total Exposure Rows | ~1.19B | 1,185,707,891 | Consistent |
| Positives (action_bitmap > 0) | 130,995,419 | 124,893,764 | +4.9% |
| Users | 4,608,606 | 3,042,069 | +51% |
| There are negative samples | 40,761,718 (31.1% row level) | 2,084,314 (68.5% user level) | Caliber Different |

Difference analysis:
- **Positives +4.9%**: PySpark does not filter iids outside the SID dictionary, which is ~6M more. The `iid ∈ SID` of `_build_user_items()` on the Python side is filtered and does not affect the final sequence.
- **Users +51%**: The extra 1.5M users only have iids outside the SID dictionary. After filtering on the Python side, there are less than 2 valid items and no sequence is generated.
- **There are negative samples 31% vs 68.5%**: There is no contradiction in the caliber. 31% are row-level (41M of 131M positive sample rows have neg), and 68.5% are user-level (2M of 3M users have neg). In the feed scenario, users often click continuously (multiple items on the same page), and there is no non-positive between consecutive positives → the latter cannot get neg.
- Old streaming walk ended up with 3,042,069 users / 76M items / 59M neg tokens; PySpark export should give consistent results after filtering on the Python side

**Performance comparison**:
- Old streaming walk: Phase 1 read 620 files 2917s + Phase 2 groupby 1350s = **~71 min**
- PySpark export: Spark cluster minute level + Python load_exposure_neg_data() **~30s**

### Key questions

1. ~~The behavioral data needs to include "exposure without click" information~~ ✅ `export_exposure.py` already has a complete exposure sequence
2. Hard negative that is too strong may cause the model to be too conservative (biased towards popular items)
3. Long/short-term branch (DBR) requires larger architectural changes and can be split independently

---

## IDEA-stamp-0: Semantic Adaptive Pruning + Multi-step Auxiliary Prediction (STAMP)

**Priority**: P1
**Source**: STAMP (Alibaba, arxiv 2604.05329, Apr 2026)
**Status**: To be discussed

### Core Idea

STAMP found that high-granularity SID exists **Semantic Dilution Effect**: The longer and more refined the SID, the more redundant tokens there are, which dilutes the learning signal → training efficiency decreases + performance does not fluctuate monotonically.

Double-ended optimization:
1. **Semantic Adaptive Pruning (SAP)** — Input: dynamically filter redundant SID tokens in forward propagation, compressing noisy sequences into compact information-dense representations
2. **Multi-step Auxiliary Prediction (MAP)** — Output: replace single-token NTP with multi-token prediction target, densify supervision signal

**Results**: **1.23-1.38x training acceleration, 17.2%-54.7% VRAM reduction**, no performance degradation.

### Association with the current project

- There is no semantic dilution problem under the current 3-layer SID short sequence.
- **But if you switch to OPQ (16-64 token)**, semantic dilution will become a key issue:
  - Many of the 64 SID tokens may be redundant (low information quantum vectors)
  - STAMP's SAP can dynamically cut off redundant tokens → solve the training efficiency problem of OPQ long SID
- MAP (multi-step prediction) is in the same direction as IDEA-sid-4 (MTP auxiliary loss), but STAMP is more focused as a compensation for **SID sparse signals**
- 1.23-1.38x training speedup of real value for 8xA100 environments

### Experimental Design Draft

**Prefix: IDEA-sid-0 Phase 2 (OPQ long SID)**

**Phase 1 — MAP (can be tested immediately)**:
- In the current NTP model, in addition to predicting the next token, it also predicts 2-3 tokens in the future
- Add 2-3 projection heads, multi-token CE loss
- L = L_NTP + α * L_MAP

**Phase 2 — SAP (Post-OPQ)**:
- For OPQ long SID sequences, train the gating module to dynamically select information-dense tokens
- The prune token does not participate in subsequent attention calculations

**Evaluation**: Training time, VRAM usage, Recall@K

### Key questions

1. The current 3 token SID is too short and pruning is meaningless → the main value lies in the OPQ route
2. MAP overlaps with IDEA-sid-4 (MTP), but STAMP has different motivations (densify signal vs cold-start)

---

## IDEA-tbg-0: Next Session Prediction (NSP) — Replacement of Item-by-Item autoregression

**Priority**: P1
**Source**: TBGRecall (Alibaba, arxiv 2508.11977, Aug 2025)
**Status**: Phase 1 (Data Recency) Verified by EXP-016 ✅ — Phase 2 (NSP) To be tested

> **NTP phase update (2026-04-17)**: The EXP-016 Data Scaling Law experiment strongly verified that **data recency > data volume**: 14d (130M tokens) is the optimal training window, 31d/62d/90d has larger data volume but higher loss (U-shaped curve). Reason: More days = more users (1.02M→6.18M) instead of a longer sequence, 3-day exposure period causes old user behavior patterns not applicable to the current eval distribution. Phase 1 conclusions are integrated into the production configuration (14d data window). Phase 2 NSP is retained as an independent experimental direction.

### Core Idea

Standard GR is generated item-by-item autoregressively (A→B→C→D), with strong sequence dependence. TBGRecall proposes **Next Session Prediction (NSP)**: Divide behavior into multiple sessions, each session has a session token + multiple item tokens:

```
[S1] item1 item2 item3 [S2] item4 item5 [S3] → predict [S4] item6 item7
```

The items within the session are disordered (eliminating positional bias), and the items between sessions are ordered (retaining time dependence).

Another key finding: **data recency > data volume** — train with a small amount of recent data > train with a large amount of historical data.

**clear scaling law trend** is displayed on both public data sets and Alibaba industrial data sets.

### Association with the current project

- The current NTP model predicts item by item, and the SID sequence of each item is independently autoregressive.
- NSP provides **higher level abstraction**: predict "next session" instead of "next item"
- **data recency insight** is directly available: give more weight to recent behaviors during training, or only use the last N days of data
- Complementary to IDEA-onerec-1 (RSFT): RSFT filters low-quality samples, NSP changes the modeling granularity

### Experimental Design Draft

**Phase 1 — Data Recency Validation (Zero Cost)**:
- In the current NTP training, comparison: full history vs last 30 days vs last 7 days
- If recency > volume, the training cost can be greatly reduced

**Phase 2 — Session-Level Prediction**:
- Divide sessions among user behaviors (by time intervals > 30 min)
- Insert [SESSION] token in NTP input
- Randomly shuffle items within the session (remove position bias)

### Key questions

1. Session division rules: By time interval? By behavior type?
2. The fine-grained time signal may be lost due to disorder in the Session.
3. Whether the current behavior data has a timestamp that supports session division

---

## IDEA-hstu1b-0: Task Decomposition for Scaling (Feedback + Next-Item separation)

**Priority**: P1
**Source**: Scaling Recommender Transformers to 1B (arxiv 2507.15994, Jul 2025, KDD 2026)
**Status**: Awaiting discussion — subject to scaling law flattening

> **NTP stage update (2026-04-17)**: EXP-015 scaling law L(N)=2.522+2055/N^0.456 shows that M+ (101M active) has reached loss=2.94, which is only 0.42 from the irreducible floor (2.522). The paper only saw the effect of task decomposition scaling at 176M→1B, while the scaling law of our current scenario has leveled off at ~100M (the tokenizer is the bottleneck rather than the model size). Priority remains P1 but actual benefits may be limited.

### Core Idea

Based on the HSTU/Generative Recommenders framework, autoregressive learning is decomposed into two subtasks:

1. **Feedback Prediction**: Predict user feedback (like/dislike/skip) on displayed items
2. **Next-Item Prediction**: Predict the item that the user will interact with next

This decomposition remains valid over the parameter range 176M → 1B scaling.

Music streaming platform deployment: **listening time +2.26%, user likes +6.37%** — The author claims this is the largest single improvement in the history of the platform’s deep learning system. KDD2026.

### Association with the current project

- The current NTP model only does next-item prediction (Task 2), and no feedback prediction (Task 1) at all.
- Task decomposition insight: **User feedback itself is a valuable supervision signal**, not just "predicting the next item"
- Simple implementation: add feedback token (liked/skipped/watched_full) to the user sequence and let the model predict feedback + next item at the same time
- Related to IDEA-oneloc-5 (Multi-behavior) but different: oneloc-5 distinguishes behavior types as input, this IDEA uses feedback as the prediction target

### Experimental Design Draft

**Scheme — Dual-Task NTP**:
1. User sequence: `item1 [FEEDBACK:like] item2 [FEEDBACK:skip] item3 → predict [FEEDBACK:?] item4`
2. The model predicts feedback token and next-item SID at the same time
3. `L = L_next_item + α * L_feedback`
4. Variable: α ∈ {0.1, 0.5, 1.0}

**Evaluation**: NTP recall@K (core metric) + feedback prediction accuracy (auxiliary metric)

### Key questions

1. Behavior data needs to include feedback type (like/skip/watch_full, etc.)
2. Is decomposition valuable under the current 39.5M small model? The paper only sees the scaling effect when it is >176M
3. Feedback token increases sequence length ~2x → training cost increases

---

## IDEA-mbgr-0: Multi-Business Generative Recommendation (BID + MBP + LDR)

**Priority**: P2
**Source**: MBGR (Meituan, arxiv 2025, WWW 2026)
**Status**: Pending

> **P2 Reason**: Multi-service expansion is a requirement during the deployment period, and the current single-service NTP baseline has not yet been established. Core technologies (Business-aware SID, MBP heads, Label Dynamic Routing) are directly referenced when expanding multiple businesses.

### Core Idea

MBGR is Meituan’s generative recommendation deployment in multiple business scenarios (takeout, wine travel, in-store, etc.). Core challenge: Items from different businesses share the same SID space, causing interference between businesses. Three key technologies:

1. **Business-aware SID (BID)**: Append a Business ID token before the SID sequence to allow the model to distinguish the item spaces of different businesses. SID changed from `"L1_L2_L3"` to `"BIZ_L1_L2_L3"`
2. **Multi-Business Prediction (MBP)**: Each business has an independent prediction head, sharing the encoder but the head is independent. Similar to multi-task learning but in SID space
3. **Label Dynamic Routing (LDR)**: Dynamically adjust the training sample weights of different businesses to solve the imbalance of data volume between businesses (takeaway data >> wine and travel data)

Meituan Online A/B: Multi-business joint training > Single-business independent training, **average CTR for all businesses +1.2%, long-tail business CTR +3.5%**. WWW 2026.

### Association with the current project

- The current project only has a single recommended scenario, and the BID is not needed yet.
- **The idea of ​​MBP heads can be generalized**: If you want to distinguish different behavior types (click/buy/share), you can have one prediction head for each behavior
- LDR is related to IDEA-onerec-1 (RSFT) and IDEA-gr4ad-2 (Value-Aware): both training sample weighting strategies
- **The preferred reference solution for multi-business expansion**: When the recommendation system needs to serve multiple business lines, BID + MBP is the lowest-cost expansion solution

### Experimental Design Draft

**Currently not experimental, only used as a reference for multi-service expansion**.

If multiple business expansion is required:
1. Add BIZ token before SID (vocab is expanded according to the number of services)
2. Encoder is shared, prediction head is split according to business
3. LDR: Dynamic weight adjustment based on business loss (similar to GradNorm)

### Key questions

1. The current single business scenario does not require BID
2. The number of MBP heads increases with business → parameter expansion
3. LDR’s dynamic routing strategy requires sufficient multi-service data verification

---

## Priority summary

| Priority | ID | Experiment | Reason |
|--------|-----|------|------|
| ~~P0~~ ✅ | ~~IDEA-mtgr-0~~ | ~~User-Level Packing + Causal Mask~~ | ✅ Implemented — `train_packed()` Standard configuration starting from EXP-013 |
| ~~P0~~ ❌ | ~~IDEA-onemall-0~~ | ~~NTP In-Batch Contrastive Loss~~ | ❌ EXP-022 Negative Result: SID discrete space is incompatible with InfoNCE |
| P1 | IDEA-sid-4 | Token-Space MTP auxiliary Loss | RPG proves token-space CE > item-space CE, cold start friend Good |
| P1 | IDEA-gr4ad-2 | Value-Aware Training | Enrich training signals, complementary to IDEA-sid-1 |
| P1 | IDEA-oneloc-5 | Multi-behavior sequence fusion | Low cost distinguishes Different behavior intensity |
| P1 | IDEA-plum-0 | LLM Continued Pre-Training | YouTube billions of user verification, leveraging pre-training knowledge |
| P1 | IDEA-onerec-1 | RSFT filtering Low quality Training sample | Zero-cost data quality improvement, OneRec standard |
| P1 | IDEA-dualgr-0 | Exposure-Aware NTP Loss | Kuaishou WWW 2026, zero architectural changes introduce negative signals |
| P1 | IDEA-stamp-0 | Semantic Pruning + MTP | Solve the training efficiency of OPQ long SID, 1.23x acceleration |
| P1 | IDEA-tbg-0 | Next Session Prediction + Data Recency | Alibaba verification scaling law, data recency > volume |
| P1 | IDEA-hstu1b-0 | Task Decomposition (Feedback + Next-Item) | KDD 2026, the largest improvement in history, 1B Parameter scaling |
| P2 | IDEA-sid-5 | Codebook Embedding aggregation | Depends on IDEA-sid-0 Phase 2, little benefit from short ID |
| P2 | IDEA-onerec-2 | SID replaces VID Input | Valuable Value in large Model scenarios, currently not needed |
| P2 | IDEA-mbgr-0 | Multi-Business Prediction + BID | Meituan deployment, reference for multi-business expansion |
| P1 | IDEA-sigma-0 | Instruction-driven multi-tasking GR + adaptive fusion | AliExpress online verification, multi-tasking extension Direction |
| P1 | IDEA-lemur-0 | End-to-end multi-modal + Memory Bank | Douyin QAUC +0.81%, Memory Bank Low cost can be verified first |
| P1 | IDEA-genrec-0 | Page-wise NTP multi-label page-level supervision | JD SIGIR 2026, +9.5% click, hallucination rate reduced by 50%, inference unchanged |
| P1 | IDEA-rclrec-0 | Reverse curriculum learning sparse transformation | Alibaba +2.09% revenue, decoder prefix additional supervision |

---

## IDEA-sigma-0: Instruction-driven multi-task generative recommendation + adaptive probabilistic fusion

**Priority**: P1
**Source**: SIGMA, Alibaba/AliExpress (arxiv 2602.22913)
**Status**: To be discussed

### Core Idea

SIGMA deployed by AliExpress expands generative recommendations from "interaction-driven next-item prediction" to "instruction-driven multi-task recommendation". Three key designs: (1) Unified semantic-collaborative latent space - simultaneously captures semantic relationships and collaborative relationships, item grounding does not rely on a single signal; (2) Hybrid item tokenization - a balance of accurate modeling + efficient generation; (3) Adaptive Probabilistic Fusion - dynamically calibrate the generation distribution according to task type (recall/ranking/diversity), the same model uses different instructions to serve different recommendation needs. Large-scale SFT data sets are supported with instruction following. AliExpress online A/B verification is valid.

### Association with the current project

- Currently NTP only performs a single recall task, and SIGMA’s instruction-following idea can expand model capabilities.
- Adaptive Probabilistic Fusion has direct value for the inference stage: the same model can control the output distribution with "precision" or "diversity" instruction
- Hybrid item tokenization may include improvements to MLP-FSQ tokenizer
- For the SFT dataset construction method, please refer to: Convert existing behavioral data into multi-task instruction format

### Experimental Design Draft

**Phase 1 — Adaptive Decoding Temperature per Task**:
- The simplest implementation: different beam search temperatures simulate different task instructions
- recall task → low temperature (precision), diversity task → high temperature
- Evaluation: Recall@K vs Coverage@K trade-off

**Phase 2 — Instruction Prefix for NTP**:
- Add task instruction token (e.g., [RECALL], [DIVERSE], [SIMILAR]) before the behavior sequence
- Use different label strategies during training: RECALL→next click, DIVERSE→random positive examples, SIMILAR→same-category positive examples
- Requires multi-task SFT data structure

### Key questions

1. AliExpress’s multi-tasking requirements (recall/ranking/diversity) have limited value in the current single recall scenario
2. Instruction-following requires a larger backbone (currently small decoder has difficulty understanding complex instructions)
3. The implementation details of Adaptive Probabilistic Fusion need to be read in the full paper
4. More suitable for introduction when the system matures and expands multi-task capabilities.

---

## IDEA-lemur-0: End-to-end multi-modal recommendation + Memory Bank incremental representation

**Priority**: P1
**Source**: LEMUR, ByteDance/Douyin (arxiv 2511.10962)
**Status**: To be discussed

### Core Idea

LEMUR deployed by Bytedance in Douyin search and advertising is the first large-scale end-to-end multi-modal recommendation system: jointly optimizing the multi-modal encoder and recommendation model, instead of the two-stage solution of "pre-training the multi-modal model first, and then freezing the representation to train the recommendation model". The core innovation is the Memory Bank mechanism: during the training process, historical multi-modal representations are incrementally accumulated to avoid the huge computational overhead of doing a complete multi-modal forward pass on the user's long history sequence. One month after Douyin search deployment: query change rate decay -0.843%, QAUC +0.81%.

### Association with the current project

- The current project uses Qwen3 to freeze the three-stage pipeline of embedding → tokenizer → NTP, and LEMUR verifies the advantages of end-to-end joint training
- The Memory Bank mechanism has direct value for long sequence NTP: there is no need to recalculate the embedding of historical items for each training, use cache + incremental update
- In current NTP training, item embedding is pre-computed & frozen. If end-to-end is to be done, Memory Bank is the key to solving the computing bottleneck.
- In the same direction as IDEA-onerec-3 (QFormer Tokenizer): both pursue breaking the limitations of frozen embedding

### Experimental Design Draft

**Phase 1 — Embedding Memory Bank for NTP Training**:
- Maintain item embedding memory bank (size = item pool) during NTP training
- Update the memory bank with the current encoder at the beginning of each epoch (or use EMA)
- Comparison: frozen embedding vs memory bank (regularly updated) vs full end-to-end
- Evaluation: Recall@K + training efficiency (FLOPs per epoch)

**Phase 2 — End-to-End Multimodal Training**:
- Unfreeze Qwen3 embedding encoder and train jointly with NTP
- Use Memory Bank to cache intermediate representations, updated every N steps
- Frontend: requires GPU memory optimization (gradient checkpointing, mixed precision)

### Key questions

1. End-to-end training requires far more than current GPU resources
2. Memory Bank’s staleness: The cache representation is out of sync with the current encoder → the impact needs to be verified
3. The current MLP-FSQ tokenizer is trained on frozen embedding, and end-to-end means that the tokenizer must also be retrained.
4. Phase 1 (Memory Bank alone) is relatively low-cost and worth verifying first.

---

## IDEA-genrec-0: Page-wise NTP training objective (multi-label page-level supervision)

**Priority**: P1
**Source**: GenRec, JD.com (arxiv 2604.14878, SIGIR 2026)
**Status**: To be discussed

### Core Idea

JD's GenRec proposed Page-wise NTP (PW-NTP): splicing multiple positive interactions (clicks + purchases + exposure) of users on the same request page into a target sequence for autoregressive training, instead of vanilla NTP modeling each positive sample independently. Solve the one-to-many ambiguity problem of "same input, multiple valid outputs" caused by the industrial paging mechanism. Experiments show that PW-NTP: (1) converges faster than vanilla NTP; (2) HR@50 increases from 0.62 to 0.72; (3) the hallucination rate is reduced by 50% (7.8% → 4.96%). JD Online A/B: Clicks +9.5%, Deals +8.7%. Standard point-wise beam search is still used for inference, and the training-inference asymmetry is intentionally designed.

### Association with the current project

- The current NTP training is point-wise: a (history, target_item) pair and a training sample
- PW-NTP can be implemented directly on existing data: splicing multiple positive samples in the same session into a target sequence
- Complementary to IDEA-onemall-0 (Contrastive Loss): PW-NTP improves the SFT stage, Contrastive is an additional auxiliary loss
- Training-inference asymmetric design means that **the inference side does not need to be modified at all**
- The reduction in hallucination rate is particularly valuable for SID systems: reducing the generation of invalid SID combinations

### Experimental Design Draft

**Phase 1 — Session-level Multi-Target NTP**:
- Data structure: Splice multiple positive interaction item SIDs of users in the same session (or the same day) into a target
- Sort by interaction intensity: buy > click > expose
- Training: standard autoregressive loss but target is a multi-item sequence
- Baseline: current point-wise NTP
- Assessment: HR@K, NDCG@K, HaR (hallucination rate)

**Phase 2 — Add behavior type token**:
- Insert the behavior type token in the target sequence: `<buy> SID1 SID2 SID3 <click> SID4 SID5 SID6 ...`
- Explore the impact of different sorting strategies on performance

### Key questions

1. Data format changes: The current dataloader needs to be modified to support variable-length target sequences.
2. Inference remains unchanged but the sequence length in the training batch increases → GPU memory pressure
3. There is some overlap with IDEA-dualgr-0 (ENTP-Loss): both focus on multi-behavior training and need to be compared or integrated.

---

## IDEA-rclrec-0: Reverse curriculum learning to solve sparse transformation modeling

**Priority**: P1
**Source**: RCLRec, Alibaba International (arxiv 2603.28124)
**Status**: To be discussed

### Core Idea

Alibaba International E-commerce RCLRec proposed the Reverse Curriculum Prefix Module (RCPM): for each conversion target, k behaviors most relevant to conversion are reversely selected from the user history, and their SID tokens are used as decoder prefix, and are spliced ​​with the target conversion token for teacher forcing. Core insight: Before conversion behavior, there is usually a set of clustered related behaviors (same category browsing/comparison), and these key subsequences are directly extracted as additional supervision. Adding curriculum quality-aware loss ensures that the selected prefix actually improves conversion predictions. Online A/B: Ad revenue +2.09%, Orders +1.86%.

### Association with the current project

- Current NTP training does not distinguish between behavior types, and conversion behaviors are extremely sparse (usually <2% interactions)
- RCPM can be used as an enhancement module for NTP training: providing additional decoder-side supervision for high-value targets (purchases)
- Complementary to IDEA-genrec-0 (PW-NTP): PW-NTP resolves one-to-many ambiguity, RCLRec resolves conversion sparsity
- Complementary to IDEA-oneloc-5 (Multi-behavior sequence): oneloc-5 is encoder-side behavior fusion, RCLRec is decoder-side course injection
- Requires encoder-decoder architecture (currently decoder-only) → May require adaptation

### Experimental Design Draft

**Phase 1 — Decoder Prefix for High-Value Targets**:
- For the purchase behavior in the training sample, select the top-k related historical items from the encoder hidden states
- Use the selected k item SIDs as decoder prefix and splice them in front of the target SID
- Use scaled dot-product for relevance scoring
- Baseline: Standard NTP (no prefix)
- Assessment: Recall@K (conversion items), NDCG@K

**Phase 2 — Quality-Aware Loss**:
- Add hinge loss: ensure conversion NLL with prefix < NLL without prefix + margin
- Adjust margin and loss weight

### Key questions

1. Currently it is a decoder-only architecture, RCLRec requires encoder-decoder → needs to be modified or adapted using prefix-LM method
2. The top-k selection of RCPM needs to be differentiable (IBQ straight-through estimator)
3. Pre-requisite: Multi-behavior training data is required (Is there any behavior type annotation in the current data?)
4. k=4 is the recommended value in the paper, and the parameters need to be adjusted based on our data.

---

## IDEA-lac-0: Lagged Action Conditioning

**Priority**: ~~P1~~ → ✅ Implemented and verified
**Source**: The Layout Is the Model (Roblox, arxiv 2510.16804)
**Status**: ✅ Implemented — EXP-025 implements the shift scheme (=LAC) and fixes beam search feature passing; EXP-036 full-features adopts lag action_level, which has become the standard training configuration

### Core Idea

When modeling items and actions (behavior type/viewing duration, etc.) simultaneously in GR, token layout determines information leakage and conditional relationships. The paper proposes three design principles:
1. **Maximize item/action signal** (use both input and output)
2. **Keep the condition direction of "action given item"** (see the item first and then predict the action)
3. **No information leakage** (you cannot see the action of the item when predicting the item)

**Lagged Action Conditioning (LAC)**:
- Non-staggered layout: each token is just the item SID (no separate token is given to the action)
- **Lag**: The action of item_i is used as the input feature of item_{i+1} (delay one item)
- In this way, when predicting item_{i+1}, you can see the action of item_i (with information increment), but you cannot see the action of item_{i+1} itself (no leakage)
- During inference: After generating the SID of item_{i+1}, action_{i+1} is unknown, use the last known value of action_i

### Relationship to our EXP-023/024/025

**This is the theoretical version of the shift scheme we tried in EXP-024! ** But our shift implementation operates at the flat token level, and LAC does lag at the item level:
- Our EXP-024 shift: 3 tokens for each item use the features of the previous item → **Equivalent to LAC**
- The reason for the poor results of EXP-024: **It is not that the LAC idea is wrong, but that the beam search incremental path does not pass features**
- EXP-025 Fixed beam search feature passing → combination of LAC (shift) + beam passes should be re-evaluated

### Experimental design

Re-evaluate the combination of EXP-024 shift + EXP-025 beam passes:
1. Train with shifted data (exp024-14d-shifted)
2. Beam search passes in `gen_action_level = last context item's action` (that is, the reasoning logic of LAC)
3. Compare EXP-025 beam-passes (without shift, fax value)

### Key questions

1. LAC uses explicit action token (watchtime, etc.) in the paper, and our action_level is a thicker bitmap bucket
2. The backbone of the paper is 85M parameters, and our 17.4M active → does the small model have enough capacity to utilize the action signal?
3. Need to verify: Which one is better: shift + beam_passes vs not shift + beam_passes?

---

## IDEA-onelive-0: BOS global time injection + Gated Attention

**Priority**: P2
**Source**: OneLive (Kuaishou, arxiv 2602.08612)
**Status**: To be discussed

### Core Idea

Two temporal modeling technologies deployed by OneLive in Kuaishou live broadcast recommendations:

1. **BOS temporal injection**: Inject multi-granular temporal features into [BOS] token
   ```
   x_BOS = x_BOS + MLP(Concat(x_Hour, x_Day, x_Week))
   ```
   Use the hour-of-day / day-of-week / week of the current moment for embedding, and add it to the BOS token after fusion through MLP
   - Advantages: minimalist implementation, does not change the attention structure
   - Disadvantage: only encodes "current request time", not encoding the time of each item in the sequence

2. **Gated Attention Time Perception**: Add element-wise gate to the attention output
   ```
   Score(X) = σ(X · W_θ)
   O = MultiHeadAttn(Q, K, V) ⊙ Score(X) · W_O
   ```
   Let the model learn to suppress temporally irrelevant context

### Adaptation to us

The BOS time injection scheme can verify "whether the global time is useful" at the lowest cost:
1. Inject hour/dayofweek embedding in the first token position (or add a [BOS] token)
2. Do not change position coding or attention.
3. If valid → upgrade to TO-RoPE (feat-5)

### Key questions

1. Our NTP sequence does not have [BOS] token — does it need to be added or injected into the first SID token of the first item?
2. The time sensitivity of live broadcast scenes is much higher than that of content recommendations - the effect may be compromised
3. The Kuaishou paper does not have a single contribution of ablation BOS time injection

---

## IDEA-tca-0: Token-level CF Soft Label Alignment (CF signal injection NTP Loss)

**Priority**: P1
**Source**: TCA4Rec, USTC + Ant Group (arxiv 2601.18457, WWW 2026)
**Status**: To be discussed

### Core Idea

TCA4Rec solves the core problem of the NTP model lacking collaborative filtering signals. Core insight: The CF model does **item-level** sorting, and NTP does **token-level** prediction. The optimization granularity of the two does not match → The previous method can only passively inject CF as a soft prompt or representation bias.

TCA4Rec proposes **explicit token-level CF alignment**:

1. **Collaborative Tokenizer**: Obtain item-level logits (z_u,i = dot(e_u, e_i)) from a pre-trained CF model (such as SASRec), and transform it into a token-level distribution through three steps:
   - Step 1: Collect valid items of the current decode position (prefix matching)
   - Step 2: Softmax normalized to probability distribution π_u,i
   - Step 3: Aggregate by next token (sum of the probabilities of items sharing the same next token)

2. **Soft Label Alignment**: Fusion of CF token-level distribution with one-hot label: ỹ_j(v) = (1-α)·1_{v=y_j} + α·p_u(v|y_{<j})
3. **Soft NTP Loss**: L_soft = -Σ log(Σ ỹ_j(v)·P(v|x_u,y_{<j}))

α=0 degenerates to standard NTP, α=1 completely follows CF. Optimal α ≈ 0.01~0.05 (CF signal as gentle regularizer).

**Key difference from Auxiliary KL Loss**: Soft NTP's gradient is **adaptive** (weight q_j depends on the model's current prediction P_j), while KL uses fixed weight ỹ_j. Adaptive weights allow the model to balance the CF signal with its own world knowledge.

**Core results**:
- Consistently improved on 4 LLM-based recommendation architectures (TallRec, LLaRA, CoLLM, MSL)
- MSL+TCA on Toys: NDCG@5 0.0145→0.0332 (+129%), H@5 0.0204→0.0452 (+121%)
- Also valid for SID-based methods: TIGER+TCA, LETTER+TCA
- Collaborative Consistency increases monotonically with α, but the performance first increases and then decreases (too large α introduces CF noise)
- Model-agnostic + plug-and-play: Do not change the model architecture, only change the loss

### Association with the current project

- **Direct response to NTP's lack of cross-user signal**: The logits of the CF model naturally contain the cross-user collaborative signal, and TCA is injected through the loss layer
- Our NTP model uses SID (not item title text), and the Collaborative Tokenizer needs to be adapted:
  - SID token space (L1=1024, L2=1024, L3=4096) vs LLM vocab
  - Prefix matching becomes SID prefix matching (L1 → L1+L2 → L1+L2+L3)
  - Probability aggregation by SID token group instead of text token
- Prerequisite: requires a pretrained CF model (SASRec) — shares this dependency with IDEA-flexcode-0
- Complementary to IDEA-onemall-0 (In-Batch Contrastive Loss): onemall-0 adds CL loss to the representation layer, and TCA adds soft label to the output token distribution layer
- **Zero model architecture change** — only change the loss function → extremely low experimental cost

### Experimental Design Draft

**Phase 1 — Pretrained SASRec CF model**:
- Train SASRec on the same user behavior sequence → obtain user/item embedding
- Calculate CF logits for each training sample (user dot-product all items)

**Phase 2 — Collaborative Tokenizer for SID**:
- For each decode position j (L1/L2/L3):
  - Filter valid items based on generated SID prefix
  -Softmax normalize CF logits
  - Aggregate probability by SID layer-j token
- Output: token-level CF distribution for each decode position of each training sample

**Phase 3 — Soft NTP Training**:
- Modify NTP loss: (1-α)·CE + α·CF_soft_label
- Hyperparameter search α ∈ {0.001, 0.005, 0.01, 0.05, 0.1}

### Key questions

1. **Efficiency**: Is it feasible when each training sample needs to calculate CF logits (dot product with all items) — 5M items? ANN may be needed to approximate top-K
2. Is the quality of the SASRec CF model sufficient? The paper uses academic datasets (19K users), our scale is much larger
3. The SID token space is much smaller than the LLM vocab → the valid item set for prefix matching may be large at the L1 layer (each L1 token corresponds to ~5000 items)
4. The CF model of IDEA-flexcode-0 can be shared, but TCA’s CF signal injection method is more lightweight (only the loss is changed)

---

## IDEA-climber-0: TAMIP — Time-Aware Multi-Item Prediction + Consumption Lag Diagnosis

**Priority**: P1
**Source**: Climber-Pilot (NetEase Cloud Music, arxiv 2602.13581, Feb 2026)
**Status**: To be discussed - Consumption Lag diagnosis can be run immediately, TAMIP training paradigm needs to be changed NTP loss

### Core Idea

**1. Consumption Lag (diagnostic level insights)**

The exposure of large-scale recommendations is **batch parallel**: one request exposes m items to the user at the same time. But interaction log records these m items as **sequence** `{i_1, i_2, ..., i_m}` - this "sequence" is a mechanical product of log generation, **not a causal chain** of real user intentions.

Thesis conclusion: NTP training will learn **spurious sequential pattern** (spurious sequential pattern) on this kind of data, that is, "items with the same exposure appearing one after another" will be learned as real causal signals. This is an invisible bias in NTP training.

**2. TAMIP (Time-Aware Multi-Item Prediction)**

Two components of the correction plan:
- **Multi-branch prediction backbone**: Build K independent transformer branches on top of shared user representation `h_n`, each branch predicts `i_{n+1}, i_{n+2}, ..., i_{n+K}`. K=1 degrades to standard NTP
- **Time-Aware Masking**: train attention mask to identify item pairs that "co-occur within a narrow time window Δτ" and mask the causal attention between them (allowing non-causal parallel attention)

Training loss: `L_TAMIP = Σ_{k=1..K} L(i_{n+k} | S_n, time-mask)`

**3. CGSA (Condition-Guided Sparse Attention, SFT stage)**

In the SFT stage, condition instruction tokens (music genre/language/freshness and other business constraints) are added, and the attention mask is forced to comply with the constraints in generation.

**4. Results**: NetEase Cloud Music Online A/B **+4.24% Core Business Metrics**

### Association with the current project

**Consumption Lag observation is extremely important to us** — Our behavior data structure is the same as the paper, and the log ordering is not guaranteed to reflect the true cause and effect. It may have affected the NTP loss quality of baselines such as EXP-036/EXP-020.

**TAMIP Adaptation**:
- Change `ntp/train.py` to multi-SID prediction: predict the SIDs of the next K items
- User behavior timestamp is used for masking — EXP-044B has been connected to the `rel_hours` pipeline and can be reused

**CGSA Adaptation**: Currently no business constraint, will be adopted when controlling diversity online in the future

### Experimental Design Draft

**Phase 0 — Consumption Lag Diagnosis (0.5 days)**:
1. Count the timestamp gap of (user, item_k, item_{k+1}) in NTP training data
2. The Δt distribution has a significant peak at a very short time (<1 min) → strong Consumption Lag evidence
3. Report the proportion of "same window (Δt<5min)" among adjacent items

**Phase 1 — TAMIP K=2**:
1. `ntp/model.py::_forward_packed` adds a second prediction head
2. `L = L_{NTP, i_{n+1}} + α · L_{NTP, i_{n+2}}` (α=0.5 default)
3. Compare baseline NTP K=1 (R@500=66.2%) vs TAMIP K=2 / K=3

**Phase 2 — Time-Aware Mask** (relies on EXP-044B timestamp pipeline):
1. Mask rule: `|t_i - t_j| < Δτ` prevents causal attention
2. Δτ scan {1m, 5m, 30m, 2h}

### Key questions

1. **Data terminal**: EXP-044B `rel_hours` pipeline is connected, timestamp can be reused directly
2. **K value selection**: K=2-4, the larger K, the greater the overhead.
3. **Overlaps with IDEA-stamp-0 (MTP)**: stamp-0 is "multi-step prediction of tokens within the same item"; TAMIP is "multi-step prediction of subsequent items". The two are orthogonal and can be combined
4. **Consumption Lag may be absorbed by IDEA-feat-0 (TimeGap bucket, ✅ EXP-036)**: TimeGap indirect encoding gap, TAMIP explicit training target modification. Phase 0 Decision after diagnosis

### Related ideas

- IDEA-stamp-0 (Semantic Pruning + MTP): Orthogonal (within item vs between items)
- IDEA-feat-0 (TimeGap Bucket, ✅ EXP-036): indirect encoding lag
- IDEA-torope-0 (Time-and-Order RoPE): time difference modulation position
- IDEA-dualgr-0 (ENTP-Loss): Negative sample mechanism
- IDEA-lac-0 (LAC, ✅ EXP-025/036): action delay to avoid leakage
