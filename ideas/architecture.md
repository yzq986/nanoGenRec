# Architecture (model architecture)

[English](architecture.md) | [Chinese](architecture.zh.md)

Architectural design of NTP model: decoding strategy, sequence compression, attention mechanism, expert routing, etc. Affects inference efficiency and model capacity.

**Scope of influence**: `metrics/sid_prediction.py`, `model/train.py`

---

## Evolution path

```
AutoregressiveNTPModel (currently 6-layer decoder, beam=5)
├── IDEA-gr4ad-1: LazyAR (the first K layers are non-AR, and the last L-K layers are AR)
│ └── Doubled inference throughput, beam shares KV cache
├── IDEA-onemall-1: Query-Former (long sequence cross-attention compression)
│ └── 1205→160 token, 3.7x FLOP reduction
├── IDEA-onemall-4: Loss-Free MoE → EXP-013 ✅ MoE 8E top-2 implemented
│ └── aux_loss=0.01 works well, loss-free bias is an optional micro-optimization
├── IDEA-glide-0: Soft Prompt Injection (user embedding → prefix)
│ └── Spotify Verified, Unusual Listening +5.4%, New Discovery +14.3%
├── IDEA-oneloc-0: Context-augmented Attention (side-info injection)
│ └── additive similarity + gating, requires encoder-decoder architecture
├── IDEA-oneloc-1: Category Prompt (neighborhood cross-attention prompt)
│ └── Generalized to interest/category prompt prefix
├── IDEA-oxygen-0: Fast-Slow Thinking (nearline LLM + real-time GR)
│ └── LLM reasoning distilled into instructions, IGR intent filtering, SA-GCPO multi-scenario RL
├── IDEA-llada-0: Discrete Diffusion (substitute autoregressive)
│ └── Bidirectional attention + adaptive generation order to solve error accumulation
├── IDEA-metaidx-0: Hierarchical Index + Test-Time Training (Meta)
│ └── cross-attention + RQ learning hierarchical index, intermediate node = high-quality data → TTT
├── IDEA-oneranker-0: Unified generation and ranking (Tencent WeiXin)
│   └── Fake Item Token + DC Loss + Value-Aware Decoupling, GMV +1.34%
├── IDEA-orec-think-0: In-Text Reasoning (Kuaishou)
│   └── Itemic Alignment + Reasoning Scaffolding + Multi-validity Reward
├── IDEA-reg4rec-0: MoE parallel quantification + reasoning self-reflection (Alibaba)
├── IDEA-sif-0: Sample-Level Tokenization + SIF-Mixer (Meituan)
│   └── HGAQ 237x compression + factored row/col attention, CTR +2.03%
│ └── MPQ unordered token + PARS/MSRA/CORP reasoning enhancement
├── IDEA-ksa-0: Summary Attention (Kuaishou OneRec Team)
│ └── O(n/k) KV cache, learnable summary tokens, 8x compression + orthogonal to GQA/MLA
└── IDEA-vista-0: Two-Stage UIH Summarization (Meta, ICLR 2026)
    └── virtual seed embeddings + QLA O(N) + generative reconstruction loss
```

---

## IDEA-gr4ad-1: LazyAR decoder

**Priority**: P1
**Source**: GR4AD §LazyAR, Table 1
**Status**: To be discussed

### Core Idea

GR4AD divides the L-layer decoder into two parts: the first K layer (non-autoregressive) only relies on position coding and context, and does not rely on the previous token; only the latter L-K layer introduces autoregressive dependence. Key insight: The output of the first K layers can be calculated in parallel for all token positions and shared between beams, only the last L-K layers need to be decoded token by token. Experiments show that when K=2/3·L, the performance is almost lossless (-0.04%), but the inference throughput is doubled.

Fusion mechanism: Use gated projection to fuse the non-autoregressive representation and the previous token embedding at the Kth layer:
`Fuse(m, s) = W_f[m ⊙ (W_g · s); s]`

### Association with the current project

- `AutoregressiveNTPModel` in `metrics/sid_prediction.py` is purely autoregressive: each layer relies on the embedding of the previous token
- Currently there are only 3 tokens to predict, beam_size=5, inference is not a bottleneck. But if scaling to more tokens (IDEA-sid-0 OPQ scheme b/c has 16-32 tokens) or larger beams (production target 512), LazyAR becomes critical
- **Consistent with ARCHITECTURE.md’s Lazy Decoder-Only design direction** — OneRec-V2’s Context Processor essentially separates encoding and decoding
- Can be implemented as part of NTP model upgrade

### Experimental Design Draft

**Phase 1 — Proof of concept (S-tier model)**:
- There are currently 6 layers of decoder, assuming K=4 (the first 4 layers are non-AR, and the last 2 layers are AR)
- Add gated fusion module in layer 4
- Comparison: Original AR vs LazyAR, evaluate perplexity / Hit@K / training speed

**Phase 2 — Inference accelerated verification**:
- beam_size expanded from 5 to 50-500
- Measurement: Inference time savings brought by LazyAR’s beam-shared KV cache
- Expectation: When K=4, the calculation amount of the first 4 layers does not increase with the beam, only the last 2 layers increase linearly

**Change file**: `metrics/sid_prediction.py` — Modify the forward and generate methods of `AutoregressiveNTPModel`

### Key questions

1. **3 token scenario has limited benefits**: Currently only 3 tokens are predicted. The main calculation of beam search is in the first layer (16384 vocab softmax), and LazyAR optimizes subsequent layers. Need to quantify real reasoning benefits
2. The paper points out that LazyAR is not suitable for general-purpose LLM (there is strong dependence between tokens and the length is not fixed), but the recommended scenario has fewer tokens and the subsequent layers are "simpler" - it needs to be verified whether it is true under our 3-layer setting
3. Design of Fusion mechanism: gated projection vs simple add vs concat — requires ablation

---

## IDEA-onemall-1: Query-Former long sequence compression

**Priority**: P1
**Source**: OneMall §3.2 Query Transformers
**Status**: To be discussed

### Core Idea

Use Query-Former (cross-attention with learnable query tokens) to compress long user behavior sequences into a fixed number of continuous representations. OneMall compresses 1205 tokens to 160 tokens (M=10 query tokens per behavior type), reducing FLOPs from 34.4 GFLOPs to 9.2 GFLOPs (**3.7x reduction**), with a performance loss of only 0.3-0.5% HR.

Core components:
- learnable query tokens Q ∈ ℝ^(M×D)
- cross-attention: F = CrossAttn(Q, H_seq, H_seq)
- One Query-Former for each behavior sequence (click, buy, exposure)

### Association with the current project

- The current NTP model (`metrics/sid_prediction.py`) directly takes the SID sequence as input, and the sequence length is limited
- M-tier/L-tier models of ARCHITECTURE.md plan need to handle longer series
- Query-Former is one of the specific implementation solutions of "Context Processor (OneRec-V2 lazy decoder-only)" mentioned in ARCHITECTURE.md
- For implementation, please refer to BLIP-2’s Q-Former, but the OneMall version is simpler (pure cross-attention, no autoregression)

### Experimental Design Draft

**New module `model/query_former.py`**:
```python
class QueryFormer(nn.Module):
    # learnable queries: (M, D)
    # N layers cross-attention
    # input: (batch, seq_len, D) → output: (batch, M, D)
```

**Integrated into NTP model**:
- User sequence → QueryFormer → Compressed representation → concat SID tokens → Decoder

**Variables**:
- M (query tokens): {4, 8, 16}
- QueryFormer layers: {1, 2}
- Input sequence length: {50, 100, 200, 500}

**Baseline**: truncate the sequence directly (current solution)

**Evaluation**: Recall@K vs sequence length, training/inference time

### Key questions

1. How long is the user sequence in the current data? Query-Former may not gain much if the average series is short
2. Multiple behavior sequences (click/buy/exposure) require behavioral data export support. Is the current `data/export_behavior.py` covered?
3. Query-Former’s pre-training strategy: Is it necessary to pre-train separately before connecting to the NTP model?

---

## IDEA-onemall-4: Loss-Free MoE Load Balancing

**Priority**: ~~P2~~ → ✅ Completed (MoE basics have been implemented, loss-free is an optional optimization)
**Source**: OneMall §3.2 Decoder-Style Sparse MoE (reference loss-free mechanism)
**Status**: ✅ MoE 8E top-2 implemented in EXP-013 S-tier (SparseMoEBlock + 0.01*aux_loss)

> **Complete Record (2026-04-17)**: EXP-013 S-tier model already contains MoE (8 experts, top-2, Switch Transformer aux loss). `SparseMoEBlock` is implemented in `metrics/sid_prediction.py:69-143`. Currently aux_loss=0.01 works well and the loss-free bias mechanism is an optional micro-optimization. Model 17.5M active params (total 39.5M with all experts) baseline established.

### Core Idea

Replace the current Switch Transformer's auxiliary loss for MoE load balancing and use the loss-free mechanism instead. Loss-free balancing achieves balance by dynamically adjusting the router's expert bias without requiring additional loss terms to interfere with the main task gradient.

Core idea (from DeepSeek series):
- Each expert maintains a bias term b_i
- If the load of expert i is too high → reduce b_i → router tends to choose other experts
- If the load of expert i is too low → increase b_i → router tends to select this expert
- Bias update does not participate in gradient calculation and is entirely based on statistics

### Association with the current project

- Current MoE implementation in `metrics/sid_prediction.py:SparseMoEBlock` (lines 69-143)
- Use `0.01 * aux_loss` (Switch Transformer style: `n_experts * sum(f_i * P_i)`)
- Replacing it with loss-free only requires modifying the router logic and does not affect the overall architecture.
- **Extremely low implementation cost and minimal risk**

### Experimental Design Draft

**Modify `SparseMoEBlock`**:
1. Add `expert_bias` for each expert: nn.Parameter(zeros(n_experts), requires_grad=False)
2. router score = linear(x) + expert_bias
3. After each training step: count the frequency f_i of each expert being selected.
4. bias update: `expert_bias[i] -= lr_bias * (f_i - 1/n_experts)`
5. Remove aux_loss

**Variables**:
- lr_bias ∈ {0.001, 0.01, 0.1}
- Update frequency: every step / every N steps

**Evaluation**: expert utilization distribution, NTP perplexity, Recall@K

### Key questions

1. S-tier only has 8 experts, the load imbalance problem may not be serious - the benefits may be limited
2. Is loss-free stable when the number of experts is small?
3. Can be experimented with IDEA-onemall-0 (contrastive loss) at the same time, because the modification is orthogonal

---

## IDEA-oneloc-0: Geo-aware Self-attention (Context-augmented Attention)

**Priority**: P2
**Source**: OneLoc §2.3.3 Geo-aware Self-attention
**Status**: To be discussed

### Core Idea

Add an additive position context similarity term to the transformer self-attention, and use the user's real-time position as a gate to control the output. Specifically: `A = Softmax(QK^T/√d + E_lc · E_lc^T)`, and then use `g = 2·Sigmoid(MLP(concat(e_u, e_i)))` as the scaling factor of (0, 2) to amplify or attenuate the attention output related/irrelevant to the user's position.

### Association with the current project

- Currently `metrics/sid_prediction.py:CausalTransformerLayer` only has standard causal self-attention
- If the project introduces side information (such as category, brand, geography) in the future, this additive attention + gate is a low-cost way
- But **the current project has no geographic information requirements** (generally recommended, non-LBS scenarios), so it doesn’t make much sense to copy it directly.
- More general inspiration: **Any side information can be used to inject attention** using additive similarity + gating, not just geography

### Experimental Design Draft

**Applicable scenario**: If there are multi-modal/multi-signal fusion requirements in the future
- Use some context embedding of item (category embed, brand embed) as E_lc
- Add context similarity item to self-attention score
- Use user profile embedding as gate query

**Evaluation**: NTP recall comparing vanilla attention vs context-augmented attention

### Key questions

1. The current project is pure content recommendation (text embedding → semantic ID), without user behavior sequence modeling, and there is no implementation scenario for this technology yet.
2. The encoder-decoder architecture (TODO in ARCHITECTURE.md) is required first to be of practical significance.
3. If you only upgrade the NTP model, you should give priority to the "Context Processor" (OneRec V2 lazy decoder-only) in ARCHITECTURE.md

---

## IDEA-oneloc-1: Neighbor-aware Prompt (Category Prompt)

**Priority**: P2
**Source**: OneLoc §2.4.1 Neighbor-aware Prompt
**Status**: To be discussed

### Core Idea

Introduce "neighborhood hints" in the decoder input: use the user location as the query, perform cross-attention on the context embedding of the surrounding 8 GeoHash blocks, and aggregate local information (surrounding brands, hot-selling products, etc.) as the generated guidance signal.

### Association with the current project

- The current decoder (`AutoregressiveNTPModel`) does not have any prompt/prefix mechanism
- A **generalized form** of this technique is: aggregating some kind of "contextual hint" through cross-attention before generating a semantic ID
- What inspires us is not the geographical neighborhood, but the **user interest neighborhood** or **category neighborhood**: For example, use user embedding to attend to top-k prototype embeddings of similar categories
- But it requires an encoder-decoder architecture first

### Experimental Design Draft

**Generalized version: Category-aware Prompt**
- Maintain category centroids (mean embedding at the category level)
- The average value of User's recent behavior embedding is used as query
- Cross-attention to top-k related categories centroids → get prompt token
- Use prompt token as first input to decoder

**Evaluation**: Comparison of NTP beam search recall with/without category prompt

### Key questions

1. Same as IDEA-oneloc-0: currently there is no encoder-decoder architecture and cannot be implemented directly.
2. You need to complete the "Context Processor" or encoder-decoder reconstruction first
3. Obtaining category information: Does the current item metadata contain categories? The data pipeline needs to be checked

---

## IDEA-glide-0: Soft Prompt Injection (User Embedding → Decoder Prefix)

**Priority**: P1
**Source**: GLIDE (Spotify, arxiv 2603.17540, Mar 2026)
**Status**: To be discussed

### Core Idea

GLIDE models recommendations as **instruction-following** tasks. Key architectural innovation: Inject long-term user embeddings into the decoder as **soft prompts** instead of encoding user information into token sequences.

1. **Soft Prompt Injection**: long-term user embedding → learned projection → KV states as prefix of decoder
2. **Instruction Conditioning**: short-term behavior sequence + lightweight user context as "instructions" to guide the generation direction
3. **Semantic ID Catalog Grounding**: Use SID to ensure that the generated recommendations are valid catalog items

Spotify Online A/B at Scale (Millions of Users): **Non-usual listening +5.4%, New Show Discovery +14.3%**.

### Association with the current project

- The current NTP model does not have any user representation injection mechanism
- IDEA-oneloc-1 (Category Prompt) proposes a similar prefix idea, but GLIDE is more general: any user embedding can do soft prompt
- **Low-cost implementation**: Add several learned prefix tokens (from user embedding projection) before the decoder input sequence, no need to change the decoder architecture
- Can be combined with IDEA-sid-1 (collaborative signal enhancement embedding): enhanced user embedding for soft prompt

### Experimental Design Draft

**Implementation**:
1. User embedding: Use the average item embedding of the user’s recent behavior (or attention pooling)
2. Projection: `MLP(user_embed_dim → decoder_embed_dim × n_prefix)` → reshape to n_prefix prefix tokens
3. Decoder input: `[prefix_1, ..., prefix_n, sid_1, sid_2, sid_3]`
4. n_prefix ∈ {2, 4, 8}

**Evaluation**: NTP Recall@K with/without soft prompt

### Key questions

1. Where does User embedding come from? There is no pre-trained user embedding in the current project.
2. If the mean value of the behavior sequence is used as user embedding, is the amount of information sufficient?
3. Prefix token increases the sequence length → training and inference costs increase

> **Full text reading supplement (2026-04-28)**: Supplement key details after reading the full text of the GLIDE paper:
> - **Backbone**: Llama 3.2 1B (open source LLM, ~1B params)
> - **SID configuration**: R-KMeans, 4 levels × 256 codes (1024 SID tokens added to LLM vocab)
> - **Two-stage training**: (1) Freeze the backbone and only train SID token embedding → (2) Freeze the embedding and use LoRA to fine-tune the backbone. Semantic grounding with bidirectional translation (SID↔text) target
> - **Soft prompt**: single soft prompt token (user embedding → 2-layer MLP → LLM hidden dim), inserted after the system instruction
> - **R-KMeans vs RQ-VAE**: R-KMeans HitRate@30 is 9.52% higher than RQ-VAE, intra-bucket cosine similarity 0.856 vs 0.657. **R-KMeans is better and more stable in production** — supports our current RQ-KMeans route
> - **Multi-task controllable discovery**: Use familiar/unfamiliar control token to distinguish different recommendation targets, unfamiliar mode Recall@30 +11.8% vs single-task
> - **Beam search necessary**: Switching from sampling to beam search (30 beams) brings +27% Recall@30. coarse SID tokens do not depend on beam search, but fine-grained tokens do
> - **Debiasing**: cross-surface sampling + exploration upweighting + popularity capping to suppress popularity bias
> - **21 days A/B**: ~20M impressions/cell, GLIDE candidates account for 34% of the recommended amount in the treatment group

---

## IDEA-oxygen-0: Fast-Slow Thinking (near-line LLM reasoning + real-time generation)

**Priority**: P2
**Source**: OxygenREC (arxiv 2512.22386, Dec 2025)
**Status**: To be discussed

### Core Idea

OxygenREC proposed the **Fast-Slow Thinking** architecture to solve the problem of LLM reasoning being unavailable in real-time recommendations:

1. **Slow Thinking (near-line)**: LLM offline/near-line generation **Contextual Reasoning Instructions** — Distill complex user intent reasoning into structured instructions
2. **Fast Thinking (real-time)**: Efficient encoder-decoder consumes these instructions for real-time SID generation
3. **Instruction-Guided Retrieval (IGR)**: Use instructions to filter behavior sequences and only retain intent-related interactions
4. **SA-GCPO**: Soft Adaptive Group Clip Policy Optimization, unified RL alignment in multiple scenarios

Core innovation: Pass LLM's deep reasoning capabilities to lightweight models through "instructions" to achieve "train-once-deploy-everywhere".

### Association with the current project

- There is no LLM inference link in the current project, and the NTP model generates SID directly from the behavior sequence.
- The Fast-Slow architecture is too complex at our current stage, but the idea of **IGR (Instruction-Guided Behavior Filtering)** is worth learning from:
  - Instead of inputting all user behavior sequences, use lightweight models/rules to filter out sub-sequences related to the current context.
- SA-GCPO is a multi-scenario extension of GRPO and is related to IDEA-onemall-2

### Key questions

1. The current project is a single scenario, and the value of Fast-Slow + multi-scenario deployment is limited.
2. Where does the "instruction" of IGR come from? LLM or rule system support is required
3. Suitable as a reference for the ultimate form of architecture, but not suitable for implementation at the current stage.

---

## IDEA-llada-0: Discrete Diffusion replaces autoregressive decoding

**Priority**: P2
**Source**: LLaDA-Rec (arxiv 2511.06254, Nov 2025)
**Status**: To be discussed

### Core Idea

LLaDA-Rec uses **Masked Discrete Diffusion** instead of autoregressive decoding to generate SID, solving two fundamental problems:

1. **One-way constraint**: causal attention limits each token to only see the previous token, destroying global semantic modeling.
2. **Error accumulation**: Fixed generation order from left to right to allow early token errors to propagate to subsequent tokens

Technical points:
- **Parallel Tokenization Scheme**: SID designed for bidirectional attention (different from RQ's ordered SID)
- **Dual Masking**: user-history level (dependency between sequences) + next-item level (semantics between tokens within items)
- **Adapted Beam Search**: non-fixed order decoding of adapted diffusion

### Association with the current project

- The current NTP model is purely autoregressive (`AutoregressiveNTPModel`)
- IDEA-sid-0 (OPQ parallel ID) is already taking the non-autoregressive route (parallel prediction + graph decoding)
- LLaDA-Rec provides another non-autoregressive solution: diffusion. Differences from OPQ:
  - OPQ: predict each token completely independently → graph decoding constraints
  - Diffusion: iterative denoising, implicit interaction between tokens → possible better global consistency
- **Currently low priority**: IDEA-sid-0 (OPQ) is already in the experiment, let’s look at the OPQ results first

### Key questions

1. Diffusion’s reasoning delay: requires multi-step denoising (T=10~50 steps), slower than autoregression and parallel prediction
2. Comparison with OPQ parallel prediction: It only makes sense under the same SID configuration.
3. Training complexity: diffusion training requires additional engineering such as noise scheduling and denoising network design.

---

## IDEA-s2gr-0: Stepwise Reasoning Tokens in SID Generation

**Priority**: P1
**Source**: S²GR (arxiv 2601.18664, Jan 2026)
**Status**: To be discussed

### Core Idea

S²GR inserts **thinking token** before each step of SID autoregressive generation, allowing the model to "think" before generating each SID code. The key difference from OneRec-Think: reasoning is not done centrally before SID generation, but **interleaved** — there is a thinking step before each SID code.

Technical points:
1. **Thinking tokens**: insert reasoning token before each SID code, supervised by **contrastive learning** (aligned to ground-truth codebook cluster distribution)
2. **Co-occurrence codebook optimization**: Use item co-occurrence relationships to optimize the codebook, and add load balancing and uniformity constraints
3. **Balanced computation**: Solve the calculation imbalance of OneRec-Think "too much reasoning in the front and too little SID generation in the back"

Online A/B (large-scale short video platform) confirms validity.

### Association with the current project

- The current NTP model directly predicts SID tokens without any "reasoning" step
- **Difference from OneRec-Think**: OneRec-Think infers once before SID, S²GR infers before each SID step → more even calculation distribution
- Simple implementation: insert `[THINK]` token in the SID sequence and predict think token before predicting SID
- Think token's contrastive supervision (aligned cluster distribution) is novel: giving think token explicit semantics (instead of free-form reasoning)

### Experimental Design Draft

**Implementation**:
1. Extended SID sequence: `[THINK_1] SID_L1 [THINK_2] SID_L2 [THINK_3] SID_L3`
2. Think token target: softmax (contrastive) of codebook cluster distribution
3. Target of SID token: original CE loss
4. Total loss: `L_SID + α * L_think_contrastive`

**Evaluation**: NTP Recall@K with/without think tokens

### Key questions

1. The sequence length is doubled (3→6 tokens) → the inference cost increases, but the calculation quality of each token is higher
2. How to construct the contrastive target (cluster distribution) of Think token
3. Relationship with IDEA-sid-4 (MTP): Both add additional supervision at the token level

---

## IDEA-gr2-0: LLM Reasoning Reranker with Verifiable Rewards

**Priority**: P2
**Source**: GR2 (Meta, arxiv 2602.07774, Feb 2026)
**Status**: To be discussed

### Core Idea

GR2 uses LLM for reranking (non-retrieval), three-stage training:
1. **Mid-training**: LLM learning SID vocabulary (≥99% uniqueness)
2. **SFT**: Distilling reasoning traces (rejection sampling) with larger LLM
3. **RL**: DAPO with conditional verifiable rewards (to prevent reward hacking: LLM tends to maintain the original order)

Beyond OneRec-Think: **Recall@5 +2.4%, NDCG@5 +1.3%**.

### Association with the current project

- LLM reasoning in the Reranking stage is the long-term goal of the current project
- Reasoning trace distillation (large LLM → small LLM) is the actual engineering solution
- **Conditional verifiable rewards** are an important improvement to RL methods - solving reward hacking
- But the NTP basic model needs to mature first, and the current priority is low

### Key questions

1. No online A/B results (offline only)
2. Rely on LLM backbone (≥7B) → high reasoning cost
3. Suitable as a reference for long-term reranking plans

---

## IDEA-genrank-0: Architecture > Training Paradigm (GenRank Insight)

**Priority**: P1
**Source**: GenRank (Xiaohongshu, arxiv 2505.04180, May 2025)
**Status**: To be discussed

### Core Idea

GenRank's in-depth analysis of Xiaohongshu's Explore Feed (100 million users) found that: **The improvement in generative ranking mainly comes from the architectural design, not the training paradigm**.

This insight is of great significance: it shows that before discussing the NTP + RL training paradigm, more attention should be paid to the **model architecture itself**.

GenRank proposes an efficient generative ranking architecture that achieves significant improvements in user satisfaction with **almost the same computing resources**.

### Association with the current project

- Directly demonstrates the extension of our "Embedding > NTP Model" philosophy: **Architecture > Training Paradigm**
- Inspiration: Before investing in RL/DPO, make sure the NTP architecture itself is good enough
- The specific architecture design of GenRank requires reading the full text of the paper
- Related to IDEA-gr4ad-1 (LazyAR), IDEA-onemall-1 (Query-Former): all are architectural improvements

### Experimental Design Draft

You need to read the full text of the paper to obtain the specific structure of GenRank, and then compare:
- Current vanilla Transformer decoder
- LazyAR (IDEA-gr4ad-1)
- GenRank architecture
- Compare Recall@K under the same training settings

### Key questions

1. Full text details of the paper are to be obtained.
2. GenRank is a ranking model rather than a retrieval → is it directly applicable to SID generation?

---

## IDEA-gti-0: Grounded Token Initialization for SID Vocabulary Extension

**Priority**: P1
**Source**: GTI (LinkedIn, arxiv 2604.02324, Apr 2026)
**Status**: To be discussed

### Core Idea

When using LLM for generative recommendation, SID tokens need to be added to the LLM vocabulary. The standard approach is **mean initialization** — but GTI proves through spectral analysis: mean init collapses all new tokens into degenerate subspaces, and fine-tuning cannot fully recover.

**Grounded Token Initialization (GTI)**: Use paired linguistic supervision to map each new SID token to a **semantically meaningful and mutually distinguishable position**. This is a lightweight pre-fine-tuning stage before fine-tuning.

Outperforms mean init and auxiliary-task adaptation on multiple benchmarks (industrial + public).

### Association with the current project

- **Directly related to IDEA-plum-0 (LLM CPT)**: When using Qwen3-0.5B for CPT, you need to expand the vocabulary to add SID tokens
- GTI answered a key question: **How to initialize SID token embedding?**
- In the IDEA-plum-0 experiment, you can directly compare: the difference in effects of mean init vs GTI init
- Simple implementation: add an alignment stage before fine-tuning and use item text for linguistic supervision

### Experimental Design Draft

**Prerequisite: IDEA-plum-0 (LLM CPT)**

**Implementation**:
1. Collect "representative item text" (title of top-k items in the cluster) for each SID token
2. Use LLM itself to encode these texts → get the linguistic ground
3. Pre-fine-tuning: Align SID token embedding to linguistic ground
4. Then normal CPT fine-tuning

**Comparison**: mean init vs random init vs GTI init

### Key questions

1. Depends on LLM backbone path (IDEA-plum-0)
2. The current 39.5M models are trained from scratch and do not involve vocab extension.
3. Linguistic grounding requires a corresponding text description for each SID code.

---

## IDEA-higr-0: Hierarchical Slate Planning (Two-Stage Generation)

**Priority**: P2
**Source**: HiGR (Tencent, arxiv 2512.24787, Dec 2025)
**Status**: To be discussed

### Core Idea

HiGR divides recommendation list generation into two stages:
1. **List-level planning**: Generate the overall intent/composition of slate (coarse-grained)
2. **Item-level decoding**: Generate specific item SID under the guidance of plan (fine-grained)

Works with **multi-objective listwise preference alignment** to optimize multiple objectives (watch time, diversity, etc.).

Tencent Business Platform (100 million users): **watch time +1.22%, video plays +1.73%**, reasoning **5x speedup**.

### Association with the current project

- The current NTP model is generated item by item, regardless of the overall composition of the slate
- Hierarchical planning can be seen as a natural extension of IDEA-tbg-0 (NSP): session prediction → slate planning
- 5x inference acceleration comes from search space reduction: first determine the slate plan, and then decode within the restricted space
- But the current retrieval stage does not require slate planning → more suitable for the reranking stage

### Key questions

1. It belongs to reranking rather than retrieval → the current stage has low priority
2. Multi-objective alignment requires reward signal

---

## IDEA-mdgr-0: Masked Diffusion with Parallel Codebook (enhanced llada-0)

**Priority**: P2 (merge tracking with llada-0)
**Source**: MDGR (Alibaba, arxiv 2601.19501, Jan 2026)
**Status**: Awaiting discussion — hardening IDEA-llada-0

### Core Idea

MDGR is the industrial implementation of the LLaDA-Rec (IDEA-llada-0) idea:
1. **Parallel Codebook** (non-sequential RQ): Provides a structural basis for diffusion
2. **Adaptive Masking Training**: Adaptive construction of masking signals in time and sample dimensions
3. **Warm-up Two-Stage Parallel Decoding**: Warm-up first and then parallel decoding

Online advertising platform: **revenue +1.20%**. Offline: Surpasses 10 SOTA baselines up to +10.78%.

### Association with the current project

- Direct enhancement IDEA-llada-0: MDGR provides industrial validation of diffusion GR
- Parallel codebook overlaps with IDEA-sid-0 (OPQ): both are non-sequential SID schemes
- Online results of revenue +1.20% provide confidence in the diffusion route
- But still: OPQ + graph decoding first, diffusion as alternative

### Update to IDEA-llada-0

Industrial validation of IDEA-llada-0: MDGR validates the feasibility of diffusion GR on an advertising platform (+1.20% revenue).

---

## IDEA-unirec-0: Chain-of-Attribute (CoA) Prefix Decoding

**Priority**: P1
**Source**: UniRec (arxiv 2604.12234, Apr 2026)
**Status**: Active

### Core Idea

UniRec uses Bayes' theorem to prove: **Generative model with complete feature access = expressive ability of discriminative model**. The actual gap only comes from insufficient feature coverage.

**Chain-of-Attribute (CoA)**: Add the structured attribute token prefix (category, seller, brand) before the SID sequence to restore the feature crossover ability of the discriminant model.

Mathematical guarantee: `H(s_k | s_{<k}, a) < H(s_k | s_{<k})` — Attribute prefix reduces the condition entropy of each step, narrows the search space, and stabilizes beam search.

Package: **Conditional Decoding Context (CDC)** (Task-Conditioned BOS + hash-based Content Summary) + **Joint RFT + DPO**.

Online A/B: **HR@50 +22.6%**, high value orders **+15.5%**.

### Association with the current project

- Different from IDEA-onemall-3 (attribute enhancement contrastive): onemall-3 adds attributes during embedding training, and CoA adds attribute token prefix during decoding.
- Complementary to IDEA-glide-0 (Soft Prompt): glide uses user embedding as soft prompt, CoA uses item attribute as hard token prefix
- 22.6% HR@50 Great improvement → But item attribute data (category/seller/brand) is required

### Key questions

1. Attribute data availability: category/brand/seller structured attribute required
2. The sequence length increases: each item from 3 tokens → 5-6 tokens (attribute prefix + SID)
3. Attribute prediction vs. attribute given: During inference, does the model predict the attributes by itself or is it provided by context?

---

## IDEA-gems-0: Multi-Stream Temporal Decoder for Lifelong Sequences

**Priority**: P1
**Source**: GEMs (Kuaishou, arxiv 2602.13631, Feb 2026)
**Status**: Active

### Core Idea

GEMs solve the computational and attention bias problems of GR in processing extremely long user behavior sequences (100K+ interactions). Divide user behavior into three streams based on time:

1. **Recent Stream**: One-stage real-time extractor → Capture real-time interest dynamics
2. **Mid-term Stream**: lightweight indexer + cross-attention → balance accuracy and cost
3. **Lifecycle Stream**: two-stage offline-online compression module → life cycle modeling

The third stream is merged via the **parameter-free fusion** strategy. Kuaishou high-concurrency industrial environment deployment, processing 100K+ interactions/user.

### Association with the current project

- Complementary to IDEA-onemall-1 (Query-Former sequence compression): QFormer compresses the entire sequence, GEMs divide and conquer according to time granularity
- Directly related to IDEA-oneloc-4 (sequence length scaling law): GEMs provide engineering solutions for utilizing extremely long sequences
- Current series is shorter → Mid- to long-term value: Becomes key when the series expands to 1000+

### Key questions

1. The current data set sequence is short → you need to expand the data or simulate a long sequence first
2. The offline-online two-stage compression of Lifecycle stream is highly complex to implement.
3. Prerequisite: IDEA-oneloc-4 (sequence scaling law) Confirm the value of long sequences before investing

---

## IDEA-hpgr-0: Session-Based MIM + Preference-Guided Sparse Attention

**Priority**: P1
**Source**: HPGR (Huawei, arxiv 2603.00980, Mar 2026, WWW 2026)
**Status**: Active

### Core Idea

HPGR points out that the "flat-sequence" assumptions of existing GR (such as HSTU) ignore the intrinsic structure of user behavior:
1. Unable to capture session-based time hierarchy
2. Dense attention introduces a lot of noise and masks the real preference signal

Two-stage solution:
1. **Structure-aware Pre-training**: Use **Session-based Masked Item Modeling (MIM)** to learn hierarchical item representation
2. **Preference-aware Fine-tuning**: **Preference-Guided Sparse Attention** dynamically constrains attention to the most relevant historical items

Huawei AppGallery Industrial Dataset + Online A/B: **Beyond HSTU and MTGR**, WWW 2026 received.

### Association with the current project

- Different from IDEA-hstu-0 (Sparse Self-Attention): hstu is a fixed pattern sparse, HPGR is a preference-guided dynamic sparse
- Session-based MIM pre-training can be used independently → provides better initialization for NTP

### Key questions

1. Session segmentation strategy: Time threshold selection has a great impact
2. The current sequence is short → preference-guided sparse attention has limited benefits.
3. Pre-training → fine-tuning two-stage pipeline increases project complexity

---

## IDEA-sif-0: Sample-Level Tokenization + Factored Attention (SIF-Mixer)

**Priority**: P2 — SIF is a ranking model rather than a generative retrieval, but HGAQ quantification and factored attention design have reference value
**Source**: SIF (Meituan, arxiv 2604.15650, Apr 2026)
**Status**: To be discussed

> **P2 reason**: SIF's paradigm (sample-level tokenization for ranking) is different from our SID-based NTP generative retrieval. But two techniques have cross-paradigm value: (1) HGAQ quantization method can be used to compress rich per-interaction context; (2) factored row/col attention can be used to process long sequences with side features.

### Core Idea

SIF upgrades the recommendation sequence from **item-level** to **sample-level**: instead of using bare item embedding to represent each historical interaction, the complete Raw Sample (user+item+context+cross features, 600+ fields) is quantified into a Token Sample.

1. **Sample Tokenizer (HGAQ)**: Divide 600+ features into 4 semantic groups (user/item/context/cross), each group is adaptively divided into K_g sub-tokens (B=32 fields/token), and each sub-token is encoded with M=3 layer RVQ (V=256). Total compression: 600×8×32=153,600 bits → 27×3×8=648 bits (**237x compression**). Label-supervised codebook: joint optimization CTR loss + VQ commitment loss
2. **SIF-Mixer**: Factored (L+1)×T attention — Token-level Mixer (intra-sample, interaction between T sub-tokens, capture user-item-context relationship) → Sample-level Mixer (inter-sample, interaction between L+1 samples, capture timing pattern) → Token-level FFN
3. **Scaling behavior**: The gap between SIF and HyFormer increases **monotonically** with the sequence length (L=100: +0.0013 → L=2000: +0.0102 GAUC). The item-level approach saturates at L=500, and SIF continues to benefit from more contextualized interactions

### Key data

| Metric | 数Value |
|------|------|
| Online A/B (Meituan 外卖, 5% 流量, 7 天) | CTR +2.03%, CVR +1.21%, GMV/session +1.35% |
| Heavy users (L≥500) | CTR +3.12%, CVR +1.87%, GMV/session +2.06% |
| Cold users (L<10) | CTR +0.53%, CVR +0.31% (Target Token Sample 也有增益) |
| 数据规模 | 1B+ impressions, 50M+ users, 5M+ items, 600+ feature fields |
| 量化压缩 | 237x (648 bits vs 153,600 bits raw) |
| Model config | 4 SIF Blocks, 8 heads, d0=16, L=1000, T=27 sub-tokens |

### Association with the current project

- **EXP-036 validates side features value** (time_gap + action_level → R@500 +3.7pp). SIF is the ultimate form of this direction: not just 2 side features, but 600+ features all encoded
- **HGAQ's group-adaptive quantization** is similar to the design concept of our MLP-FSQ tokenizer: both use group quantization to compress high-dimensional representations
- **Factored row/col attention** is similar to IDEA-ksa-0 (Summary Attention), IDEA-vista-0 (QLA): both reduce sequence modeling costs by decomposing attention
- **Key difference**: SIF is a discriminative ranking model (predicting CTR/CVR), and we are a generative retrieval model (NTP generates SID tokens). SIF's SIF-Mixer cannot replace our causal decoder

### Experimental Design Draft

**Phase 1 — Rich Context Encoding (can be experimented within the current architecture)**:
- In the input sequence of NTP training, in addition to SID tokens, each item is injected with more per-interaction features (e.g., item category, price bucket, user-item interaction frequency)
- Use HGAQ-style group quantization to compress these additional features into fixed-length tokens
- Evaluation: R@500 improvement vs tradeoff with increased sequence length

**Phase 2 — Factored Attention (forward)**:
- If multiple sub-tokens per item are introduced (currently each item is 3 SID tokens), token-level + sample-level factored attention can be used
- Comparison with block compression of IDEA-ksa-0 (Summary Attention)

### Key questions

1. **Difference in paradigm**: SIF is a ranking model, and we are a generative retrieval. SIF does not need to generate SID tokens, but predicts CTR/CVR. Lift migration is not feasible
2. **Data Availability**: Our behavioral data may not have 600+ features per interaction — need to check data source
3. **Sequence length**: Currently max_seq_len=512 (~170 items), SIF has the greatest advantage when L=1000-2000
4. The per-interaction features of Phase 1 overlap with IDEA-feat-0/1/2 (time_gap/action_level) and IDEA-oneloc-5 (multi-behavior) - can be integrated

---

## Priority summary

| 优先级 | ID | Experiment | 原因 |
|--------|-----|------|------|
| P1 | IDEA-gr4ad-1 | LazyAR 解码器 | 与 ARCHITECTURE.md Lazy Decoder-Only Direction一致；扩展 token 数或 beam 后必需 |
| P1 | IDEA-onemall-1 | Query-Former 序列压缩 | 3.7x FLOP 减少，但需要更长序列场景 |
| P1 | IDEA-glide-0 | Soft Prompt Injection | Low成本注入用户表示，Spotify 在线验证 |
| P1 | IDEA-s2gr-0 | Stepwise Reasoning Tokens | 每步 SID 前插入 think token, 在线验证 |
| P1 | IDEA-genrank-0 | Architecture > Training Paradigm | 小红书亿级验证，架构比Training范式更重要 |
| P1 | IDEA-gti-0 | Grounded Token Initialization | LLM SID vocab extension 必需, LinkedIn 验证 |
| P1 | IDEA-unirec-0 | Chain-of-Attribute Prefix | 贝叶斯理论 + HR@50 +22.6%, 桥接生成与判别 |
| P1 | IDEA-gems-0 | Multi-Stream Temporal Decoder | 快手 100K+ 序列部署, lifelong GR 工程方案 |
| P1 | IDEA-hpgr-0 | Session-MIM + Preference Sparse Attn | Huawei WWW 2026, 超越 HSTU 的动态稀疏注意力 |
| P1 | IDEA-metaidx-0 | 层次化索引 + Test-Time Training | Meta 数十亿用户部署，hierarchical pruning + TTT |
| P1 | IDEA-oneranker-0 | 统一生成与排序 | Tencent WeiXin GMV +1.34%, DC Loss 可作辅助 loss |
| P1 | IDEA-orec-think-0 | In-Text Reasoning for GR | 快手 +0.159%, multi-validity reward 可先用于 GRPO |
| P1 | IDEA-reg4rec-0 | MoE 并行量化 + 推理自反思 | 阿里在线验证, CORP/MSRA 组件可独立落地 |
| ~~P2~~ ✅ | ~~IDEA-onemall-4~~ | ~~Loss-Free MoE Balancing~~ | ✅ MoE 已实现 (EXP-013), loss-free 为可选微优化 |
| P2 | IDEA-oneloc-0 | Context-augmented Attention | 需要 encoder-decoder 架构，当前无落地场景 |
| P2 | IDEA-oneloc-1 | Category Prompt | 需要 encoder-decoder 架构，泛化形式有价Value |
| P2 | IDEA-oxygen-0 | Fast-Slow Thinking | 架构终极形态参考，当前Phase过于复杂 |
| P2 | IDEA-llada-0 / IDEA-mdgr-0 | Discrete Diffusion 解码 | 非自回归新范式，MDGR 工业验证 +1.20% revenue |
| P2 | IDEA-gr2-0 | LLM Reasoning Reranker | Meta 远期方案, 无在线 A/B |
| P2 | IDEA-higr-0 | Hierarchical Slate Planning | Tencent 验证, 属于 reranking Phase |
| P1 | IDEA-genrec-1 | Asymmetric Token Merger | JD SIGIR 2026, prompt 长度减半, 一个 Linear 层, 性能无损 |
| P2 | IDEA-nsgr-0 | Next-Scale 粗到细重排序 | 美团 CTR +2.89%, 但属于 reranking Phase |
| P2 | IDEA-sif-0 | Sample-Level Tokenization + SIF-Mixer | 美团 CTR +2.03%, ranking Model但 HGAQ 量化+factored attention 有参考价Value |

---

## IDEA-metaidx-0: Hierarchical Index + Test-Time Training

**Priority**: P1
**Source**: Meta, Efficient Retrieval Scaling with Hierarchical Indexing (arxiv 2604.12965)
**Status**: To be discussed

### Core Idea

Meta proposes to jointly learn hierarchical indexes for large-scale foundation retrieval models: use cross-attention + residual quantization to build a hierarchical index, so that the search is pruned layer by layer from root to leaf. Key findings: The intermediate index nodes correspond to a set of high-quality data subsets. Using this subset to fine-tune the model (i.e. "test-time training") during inference can significantly improve the retrieval quality. Ad recommendations deployed on Facebook + Instagram, serving billions of users.

### Association with the current project

- The current NTP beam search is decoded in the entire SID space, and the latency increases linearly with the growth of the item pool.
- Hierarchical index can replace/enhance prefix tree constrained decoding (IDEA-static-0 CSR solution), providing more semantic-aware pruning
- The concept of Test-time training is valuable for cold start/timeliness scenarios: when new categories are launched, use the "high-quality subset" of the corresponding index nodes for fast adaptation
- Complementary to IDEA-earn-0 (Register Token compression): one optimized search path, one optimized KV cache

### Experimental Design Draft

**Phase 1 — Verification of hierarchical pruning**:
- Use KMeans hierarchy (L1, L2) as natural hierarchical index on existing 3-token SID
- Comparison: full beam search vs layer-by-layer top-K pruning (first select top-32 cluster in L1, then expand in L2)
- Evaluation: Recall@K loss vs latency reduction

**Phase 2 — Test-time training**:
- Do a small amount of fine-tune (or LoRA adaptation) of the NTP model for each L1 cluster subset
- Evaluation: cluster-level Recall improvement vs fine-tune cost

### Key questions

1. The 3-token SID itself has a shallow level and limited space for hierarchical pruning; it is more suitable to be expanded to 4+ token SIDs.
2. Real-time constraints of Test-time training in recommended scenarios: Meta uses a near-line pipeline, and we need to evaluate overhead
3. Relationship to IDEA-static-0 CSR constraint decoding: complementary or substitute?

---

## IDEA-oneranker-0: Unified Generation and Ranking (Value-Aware Generation-Ranking Integration)

**Priority**: P1
**Source**: OneRanker, Tencent WeiXin Channels (arxiv 2603.02999)
**Status**: To be discussed

### Core Idea

Tencent WeChat Video Account Advertising proposed OneRanker, which deeply integrates the generation phase and the ranking phase into one model: (1) Value-aware multi-task decoupling — Use task token sequences + causal mask to separate interest coverage and business value optimization on shared representation, reducing goal conflicts; (2) Coarse-to-fine collaborative target awareness — Use Fake Item Tokens for implicit perception in the generation phase, and use ranking decoder for explicit value alignment in the ranking phase; (3) KV pass-through + Distribution Consistency Loss ensures the consistency of generation and sorting. WeChat is fully deployed, GMV +1.34%.

### Association with the current project

- The NTP model of the current project is purely for recall (generating candidates), and the sorting is completed by the downstream system → there is a gap between generation and sorting
- OneRanker's Fake Item Tokens concept can be used for NTP training: adding ranking signal feedback on beam search candidates
- Distribution Consistency Loss can be regarded as a new type of auxiliary loss (complementary to IDEA-onemall-0 contrastive loss)
- To implement label data that requires ranking stage (CTR/CVR after click), the current data pipeline may need to be expanded.

### Experimental Design Draft

**Phase 1 — Distribution Consistency Loss**:
- Add DC loss to NTP training: do KL divergence on the probability distribution of beam candidates and external ranking score
- Requires: ranking model’s score as soft label
- Evaluation: Recall@K + NDCG (if there is ranking label)

**Phase 2 — Fake Item Token Awareness**:
- Randomly insert "fake item tokens" (sampled from in-batch items) into the decoder input sequence as negative examples
- Training model to distinguish real vs fake → implicitly introduce ranking signal

### Key questions

1. There is no ranking label in the current offline experiment, and a proxy ranking signal needs to be constructed from behavioral data.
2. The unification of generation and sorting increases model complexity and may affect the pure recall performance of the NTP stage.
3. More suitable for introduction after the system matures (with a complete pipeline)

---

## IDEA-orec-think-0: In-Text Reasoning for Generative Recommendation

**Priority**: P1
**Source**: OneRec-Think, Kuaishou (arxiv 2510.11639)
**Status**: To be discussed

### Core Idea

Kuaishou OneRec-Think unifies dialogue, reasoning, and personalized recommendations into a generative framework. The core three steps: (1) Itemic Alignment - cross-modal item-textual alignment, allowing the model to understand the semantic meaning of the item; (2) Reasoning Scaffolding - activating LLM reasoning capabilities in the NTP context, not only predicting the next SID token, but also generating reasoning chains; (3) Multi-validity reward function - the multi-correct answer feature of the recommendation scenario (multiple items are reasonable) requires special reward design. The "Think-Ahead" architecture allows real-time inference during deployment. Deployed on Kuaishou, App Stay Time +0.159%.

The difference from IDEA-s2gr-0 (Stepwise Reasoning Tokens): s2gr-0 is a lightweight think token insertion (token level), while this IDEA is a complete reasoning chain + multi-validity reward (system level).

### Association with the current project

- The current NTP is an "implicit predictor" and lacks interpretability and controllability
- Itemic Alignment can directly reuse the mapping of Qwen3 embedding + SID
- Multi-validity reward is particularly important for the RL stage (IDEA-onemall-2 GRPO currently only uses a single ground truth)
- Requires LLM backbone to support reasoning (current 6-layer decoder may not be enough)

### Experimental Design Draft

**Phase 1 — Multi-validity reward**:
- Modify the existing NTP eval: not only look at top-1 matches, but how many items in top-K behave similarly (determined by item category / embedding similarity)
- Construct multi-validity reward: R(generated) = max_sim(generated, {positive_set})

**Phase 2 — Reasoning Scaffolding (post-LLM upgrade)**:
- A larger backbone (≥1B) is needed to support reasoning
- Insert reasoning prefix (such as user preference summary) before SID token
- Comparison with IDEA-s2gr-0: full reasoning chain vs per-token think token

### Key questions

1. The current 6-layer small decoder cannot support reasoning and needs to wait for the LLM backbone to be upgraded.
2. Multi-validity reward is a more immediately available idea and can be applied in GRPO first.
3. Reasoning chain increases delay during deployment and requires "Think-Ahead" asynchronous architecture

---

## IDEA-reg4rec-0: MoE parallel quantization codebook + inference self-reflection

**Priority**: P1
**Source**: REG4Rec, Alibaba (arxiv 2508.15308)
**Status**: To be discussed

### Core Idea

Ali REG4Rec introduces reasoning into generative recommendation, which is different from IDEA-s2gr-0 and IDEA-orec-think-0 in key innovations: (1) MoE-based Parallel Quantization (MPQ) — each item generates multiple unordered semantic tokens (instead of ordered SID sequence) to build a larger and diverse reasoning space; (2) Preference Alignment for Reasoning (PARS) — uses rewards customized in the recommendation field to enhance reasoning and reflection; (3) Multi-Step Reward Augmentation (MSRA) — introduces future multi-step actions to improve generalization; (4) Consistency-Oriented Self-Reflection for Pruning (CORP) — discards inconsistent reasoning paths during reasoning.

### Association with the current project

- MPQ's "multiple unordered tokens" are fundamentally different paradigms from the current "ordered 3-token SID"
- CORP self-reflective pruning can enhance existing beam search: do consistency check on beam candidates
- MSRA multi-step reward can enhance IDEA-onemall-2 GRPO: not only look at the next step, but also look at the next N steps
- There is an online assessment to prove industrial feasibility

### Experimental Design Draft

**Phase 1 — CORP-style Beam Consistency Pruning**:
- After the beam search is completed, perform a consistency check on the candidates: perform multiple forward inferences and prune candidates with unstable results.
- Assessment: Recall@K Variation + Generate Diversity

**Phase 2 — MSRA Multi-Step Reward**:
- Add future reward to NTP training: L = L_NTP + β * Σ_{t+1..t+3} reward(item_t)
- Evaluation: Long-term indicators (session-level satisfaction) vs immediate Recall

### Key questions

1. MPQ unordered token is incompatible with the current ordered SID, and the tokenizer pipeline needs to be significantly modified → P2
2. CORP and MSRA are components that are easier to implement
3. Please see the full paper for online assessment details

---

## IDEA-genrec-1: Asymmetric Token Merger (Prompt side SID compression)

**Priority**: P1
**Source**: GenRec, JD.com (arxiv 2604.14878, SIGIR 2026)
**Status**: To be discussed

### Core Idea

JD GenRec proposed Asymmetric Token Merger: On the prefilling (encoder/prompt) side, the 3-token SID of each item is merged into 1 latent vector through linear layer projection, reducing the prompt length by ~2x; while on the decoding side, the original SID token resolution is maintained. This is a **train-inference consistent asymmetric compression**: compression is only applied to the prompt side (user history), the decoding side still generates the full SID. Experiments show that Token Merger has almost no performance loss (HR@50: 0.7192 vs 0.7201 without merger), but the prompt length is halved → supporting longer user history sequences.

### Association with the current project

- The current 3-token SID is tripling the input length on the prompt side → limiting max sequence length
- Token Merger is a minimalist solution: `h = Linear(Concat(e(s1), e(s2), e(s3)))`, a linear layer
- Same direction as IDEA-onemall-1 (Query-Former compression) but more lightweight: QFormer requires cross-attention, Token Merger only needs one Linear
- Complementary to IDEA-earn-0 (Register Token): EARN compresses the reasoning side, Token Merger compresses the prompt side
- Directly available: No need to change the SID tokenizer, just add a layer to the model forward

### Experimental Design Draft

**Phase 1 — Linear Token Merger**:
- In the NTP model forward, 3 SID embedding concat → Linear → 1 vector for each item on the prompt side
- Keep special tokens (<sep>, etc.) without compression
- Training: training from scratch or fine-tune
- Evaluation: HR@K, NDCG@K vs baseline (no compression), and training/inference speed

**Phase 2 — with longer sequences**:
- The sequence length space released by Token Merger is used to expand user history (2x history length)
- Cooperate with IDEA-oneloc-4 (Scaling Law sequence length): verify the scaling behavior of longer sequences under compressed representation

### Key questions

1. Linear projection may lose SID hierarchical structure information (hierarchical semantics of L1/L2/L3)
2. Need to do head-to-head comparison with IDEA-onemall-1 (QFormer): Which compression is better?
3. Pre-requisite: NTP model infrastructure

---

## IDEA-nsgr-0: Next-Scale coarse-to-fine generative reordering

**Priority**: P2
**Source**: NSGR, Meituan (arxiv 2604.05314)
**Status**: To be discussed

### Core Idea

Meituan NSGR proposes a new reranking paradigm: Next-Scale Generation. Different from the autoregressive (generating one by one) and one-step (generating all at once) methods, NSG adopts a tree-like coarse-to-fine strategy: starting from the user's interests, "one first, two, two, four" gradually refines the recommendation list. Core components: (1) Next-Scale Generator (NSG) - performs priority scoring + pairwise relationship classification (competition/complementary/neutral) + binary split on the current subset at each step; (2) Multi-Scale Evaluator (MSE) - a tree structure evaluator that provides guidance signals at each scale; (3) Multi-Scale Neighbor Loss - draws on the GRPO idea to construct relative rewards. Online A/B (Meituan Waimai): CTR +2.89%, GMV +3.15%.

### Association with the current project

- NSGR is reranking not retrieval → not in the same stage as our NTP retrieval
- But its **pairwise relationship modeling** (competitive/complementary/neutral classification) can inspire candidate rearrangement after beam search
- Multi-Scale Neighbor Loss The relative reward idea similar to GRPO can be migrated
- SID + HSTU as user interest extractor is a common component
- More suitable for introduction in the reranking stage after the system goes online

### Experimental Design Draft

**Phase 1 — Pairwise Relationship Reranking**:
- Perform pairwise competitive/complementary classification on the top-K candidates output by beam search
- Rearrange using NSGR’s asymmetric influence weight formula
- Evaluation: list-wise diversity + precision

**Phase 2 — Full NSGR Pipeline**:
- As a two-stage pipeline of retrieval (NTP) → reranking (NSGR)
- Need to train MSE evaluator

### Key questions

1. Currently in the retrieval stage, reranking is the follow-up work → P2
2. NSGR requires an evaluator model (additional training cost)
3. Online NSGR only has significant advantages when candidate set ≥20, and it is worth considering if beam=50

---

## IDEA-cobra-0: Cascaded Sparse-Dense generative retrieval (SID + Dense Vector joint generation)

**Priority**: P2 (after NTP)
**Source**: COBRA, Baidu (arxiv 2503.02453, Mar 2025)
**Status**: To be discussed

### Core Idea

COBRA found that pure SID generation suffers from information loss (quantization loses fine granularity), and pure dense retrieval lacks semantic structure. Proposed **cascade sparse-dense unified generation**:

1. **Cascaded Representation**: Each item is represented as (sparse_ID, dense_vector). Sparse ID is generated by RQ-VAE and dense vector is generated by end-to-end trainable text encoder
2. **Sequential Modeling**: The input sequence of Transformer decoder is [e1, v1, e2, v2, ...], each item occupies two token positions (SID embedding + dense vector)
3. **Probabilistic Decomposition**: P(ID_{t+1}, v_{t+1}|S_{1:t}) = P(ID_{t+1}|S_{1:t}) · P(v_{t+1}|ID_{t+1}, S_{1:t})
4. **Training**: L_sparse (CE on SID) + L_dense (contrastive on dense vector)
5. **Inference — Coarse-to-Fine**: First beam search generates M SIDs → append each SID to the sequence → generate dense vector → ANN retrieves top-N items
6. **BeamFusion**: Fusion of beam score (SID confidence) and cosine similarity (dense accuracy): Φ = Softmax(τ·φ_ID) × Softmax(ψ·cos(v̂, a))

**Core results**:
- Beauty R@10: 0.0725 (TIGER 0.0648, +12%)
- Toys R@10: 0.0781 (TIGER 0.0712, +10%)
- Industrial (Baidu Ads, 5M users, 2M ads): R@500 0.3716 (vs w/o Dense 0.2709 +37%, vs w/o ID 0.2466 +51%)
- **Online A/B**: conversion +3.60%, ARPU +4.15% (200M+ DAU)
- Dense + Sparse complement each other: removing either one will significantly reduce the

### Association with the current project

- COBRA’s core insight: SID (discrete) captures categorical/coarse semantics, dense vector (continuous) captures fine-grained details → the two are complementary
- Directly related to our beam search reasoning: currently beam search only returns SID → item mapping, COBRA additionally generates dense vectors for secondary fine sorting
- The BeamFusion mechanism can be applied to our inference: beam score × item embedding similarity
- **But the architecture has changed a lot**: It is necessary to (1) add dense vector position after each item token, (2) add dense prediction head, (3) add ANN step during inference
- Conflict with IDEA-genrec-1 (Token Merger): Merger reduces the number of tokens, COBRA increases the number of tokens
- 200M+ DAU online A/B is strong verification → the architectural direction has long-term value

### Experimental Design Draft

**Phase 1 — BeamFusion (no architecture changes)**:
- Keep current NTP generated SID → beam search outputs multiple candidate SIDs
- For each candidate SID, find the text embedding of the corresponding item
- Rerank: beam_score × cosine(user_embedding, item_embedding)
- No need to train a new model, just add a reranking step during inference

**Phase 2 — Full Cascaded Architecture**:
- Add dense vector token after each item in the NTP sequence
- Added dense prediction head + contrastive loss
- Modify inference pipeline: SID generation → dense vector generation → ANN

### Key questions

1. Phase 1 (BeamFusion reranking) almost zero cost → can quickly verify the value of dense refinement
2. Full cascaded requires doubling the sequence length → doubling the training cost
3. Differences from IDEA-flexcode-0: FlexCode integrates CF+semantic at the tokenizer layer, and COBRA integrates sparse+dense at the generation layer.
4. Consider full architecture change → P2 in the post-NTP phase, but Phase 1 BeamFusion can be tried earlier.

---

## IDEA-ksa-0: Summary Attention (Kwai Summary Attention)

**Priority**: P1
**Source**: KSA Technical Report (Kuaishou OneRec Team, arxiv 2604.24432, Apr 2026)
**Status**: To be discussed

### Core Idea

Kwai Summary Attention (KSA) is a new attention mechanism proposed by the Kuaishou OneRec team. It opens up an **O(n/k) path** between the O(n) KV cache of Full Attention and the O(1)/O(w) of Linear/SWA - achieving semantic-level sequence compression through learnable summary tokens.

**Mechanism**:
1. Divide the input sequence into fixed-size chunks (default k=8 tokens/chunk)
2. Inject a learnable summary token at the end of each chunk
3. Text tokens only look at local sliding chunk (adjacent chunk) + distant summary tokens
4. Summary tokens only look at the text tokens in the current chunk → distill the semantics of the chunk

**Hybrid-KSA**: 3:1 hybrid ratio (3 layers of KSA + 1 layer of Full Attention), which maintains global precise attention while greatly reducing the average KV cache.

### Key experimental data

| Metric | Full Attention | Hybrid-KSA | 提升 |
|------|---------------|-----------|------|
| RULER-128K (CPT) | 65.86 | 71.67 | +5.81 |
| RULER-128K (Scratch) | 48.75 | 65.35 | +16.60 |
| KV Cache @128K | 18.6 GB | 7.5 GB | 2.5x 减少 |
| MMLU (CPT) | 71.83 | 70.50 | -1.33 (微降) |
| GSM8K (Scratch) | 48.29 | 59.14 | +10.85 |

**Core Advantages**:
- **Orthogonal to GQA/MLA**: KSA compresses the number of tokens, GQA compresses the number of heads, and MLA compresses embedding dim → the combination of the three can achieve 8x further compression
- **Preserve long-range dependencies**: Unlike SWA (completely discard outside the window) and Linear Attention (fixed state lossy compression), summary tokens retain long-range information in an interpretable way
- **Open Source**: https://github.com/Kuaishou-OneRec/KSA

**CPT training strategy**: three stages (1) Summary token adaptation: independent Q/K/V weights + multi-granularity distillation (layer-wise MSE + distribution-wise KL + objective-wise LM loss); (2) Parameter annealing: linear interpolation to integrate independent weights into the main LLM weight; (3) Full parameter tuning + sequence length expansion.

### Association with the current project

- **Directly applicable to GR long sequences**: The OneRec team made it clear that the next step is "Unifying with OneRec — building a basic generative recommendation model based on KSA to compress ultra-long user behavior sequences into hierarchical summary tokens"
- **Solve the computational bottleneck of IDEA-oneloc-4 (sequence length scaling)**: The current EXP-015 shows diminishing returns for the scale up model, but sequence length scaling has not yet been verified - KSA can increase the sequence length 8x while the KV cache only grows 1x
- **Complementary to IDEA-hstu-0 (Sparse Attention)**: HSTU uses sparse attention pattern, KSA uses summary compression — can be combined
- **Complementary to IDEA-earn-0 (Register Token)**: EARN puts register tokens at the beginning and end of the input to reduce the KV cache in later layers, and KSA puts summary tokens in each chunk to reduce the full sequence KV cache — Orthogonal direction
- **Complementary to IDEA-gems-0 (Multi-Stream)**: GEMs uses multi-stream to split very long sequences, and KSA uses summary tokens to compress - you can compress KSA first and then multi-stream

### Experimental Design Draft

**Phase 1 — Validating summary attention in the current NTP model**:
- Modify the attention layers of `ntp/model.py`: replace 3/4 layers with KSA (summary attention) and retain 1/4 as full attention
- Chunk size k=8 (corresponding to 8 SID tokens ≈ 2-3 items, reasonable item-level summarization granularity)
- Comparison: original full attention vs Hybrid-KSA, evaluated on 14d training window PPL/R@500
- Focus on: short sequence scenarios (our current avg 21-30 items/user ≈ 63-90 tokens) whether KSA has advantages or degradations

**Phase 2 — Sequence Length Extension Verification**:
- Leverage KSA's KV cache reduction to expand the training sequence length from the current ~200 tokens to 1000+ tokens
- Verify the hypothesis of IDEA-oneloc-4 Phase 2: whether longer sequences improve Recall

### Key questions

1. The current sequence is very short (~90 tokens), and the advantages of KSA only appear in long sequences - Phase 1 may not see efficiency gains
2. Can Summary token be effectively compressed in the SID semantic space? The summary of LLM is semantic-level compression, and the summary of SID may be behavioral pattern-level compression.
3. Is the CPT three-stage training strategy suitable for small models trained from scratch (17.5M params)?
4. Compatibility with Flash Attention — KSA’s mixed attention mask requires custom kernel
5. **Priority Judgment**: Before sequence length scaling becomes a bottleneck, KSA has a low priority → P1 but ranks after RL alignment (EXP-037/038)

---

## IDEA-vista-0: Two-Stage UIH Summarization + Quasi-Linear Attention (VISTA)

**Priority**: P1
**Source**: VISTA (Meta, arxiv 2510.22049, ICLR 2026)
**Status**: To be discussed - Prerequisite: Sequence length scaling (IDEA-oneloc-4 Phase 2) will be implemented after it becomes a bottleneck

### Core Idea

VISTA decomposes traditional target attention (from candidate items to all user history) into two stages:

1. **Stage 1 — UIH Summarization**: compress extremely long user interaction history (up to 1M items) into ~128 virtual seed embeddings
   - Virtual seeds: randomly initialized shared parameters, updated interactively with UIH through self-attention
   - **Quasi-Linear Attention (QLA)**: φ-linear attention with SiLU activation → O(N) complexity
     - `O[S] = φ(Q[S]) · φ(φ(K[S])^T · V[S])` — Use associativity to reduce from O(N²) to O(N)
     - Candidates cannot attend each other (to prevent label leakage) - solved by the diagonal self-attention item
     - Custom Triton kernel implementation
   - **Generative Reconstruction Loss**: causal decoder reconstructs UIH next item from seeds + UIH prefix (off-by-one MSE)
     - `L_reconstruct = Σ ||v_i - u_{i+1}||²` — Force seed embeddings to retain maximum historical information
   - Summarization only runs during training, and embeddings are cached in the KV store

2. **Stage 2 — Target-Aware Attention**: Standard O(N²) transformer doing candidate-history interaction on compact summaries (~128 tokens)
   - Since summary is very short, O(N²) is perfectly acceptable

### Key experimental data

**Industrial-Scale (Meta production)**:
- Training size: O(10B) examples/day
- Sequence length: avg 7K, max 16K (deploy 12K)
- Configuration: 3-layer self-attention, 3-layer target-aware, 128 seeds, 256 embedding dim

| Model | C-Task Eval NE | E1-Task | E2-Task | E3-Task |
|------|---------------|---------|---------|---------|
| HSTU (baseline) | — | — | — | — |
| VISTA | -0.40% | -1.19% | -2.98% | -2.23% |
| VISTA-w/o-Recon | -0.29% | -1.29% | -3.00% | -2.21% |

QLA vs softmax attention: sequence from 6K→16K, number of layers 3→5, QPS +5%, NE -0.13%

**Online A/B (Meta production, 5% traffic, 15 days)**:
- C-Task: **+0.5%** (main consumption)
- O1-Task: +0.2%, O2-Task: +0.04%
- **94% reduction in inference GPU resource** (avoiding double calculations through cache embedding)
- Embedding delivery: 2-hour cadence updates, O(100TB) storage, geo-replicated KV store

### Association with the current project

- **Key technology for long sequence scaling**: The current sequence is very short (~90 tokens), but IDEA-oneloc-4 Phase 2 needs to be expanded → VISTA provides an industrially proven O(N) solution
- ** Deeply complementary to IDEA-ksa-0 (Summary Attention) **: KSA does summary compression (chunk-level) inside the attention layer, VISTA does UIH summarization (user-level) outside the model — can be combined: VISTA stage-1 compresses to 128 tokens → KSA further compresses KV cache inside stage-2
- **Complementary to IDEA-onemall-1 (Query-Former)**: Query-Former uses cross-attention to compress the current session, and VISTA uses self-attention + seeds to compress lifelong history — the two solve problems at different scales
- **Embedding delivery system reference value**: When deploying GR, user embedding caching architecture (2-hour cadence + KV store) is required for production
- **Generative reconstruction loss**: a new self-supervised signal that can enhance the information retention of user representation

### Experimental Design Draft

**Phase 1 — QLA Attention Verification** (low cost):
1. In the current 6-layer NTP model, replace the 1-2 layer softmax attention with QLA (SiLU-based φ-linear)
2. Comparison: PPL/R@500 accuracy impact + training speed
3. Verify whether QLA is degraded in short sequences (theoretically the difference between O(N) vs O(N²) is not obvious in short sequences)

**Phase 2 — Virtual Seed Summarization** (requires long sequence data):
1. Add stage-1 summarization module (3-layer self-attention + 128 seeds) before the NTP model
2. Training: do UIH summarization + reconstruction loss for very long user sequences (expanded to 500+ tokens)
3. Evaluation: R@500 vs baseline, information retention of summary embedding

**Phase 3 — Embedding Caching** (deployment phase):
1. Calculate stage-1 offline and cache user summary embeddings
2. Stage-2 online reasoning only processes 128 summary + candidate
3. Evaluation: Latency/QPS Improvement

### Key questions

1. The current sequence is too short (~90 tokens), and the advantages of VISTA require 1000+ tokens to appear - dependent on sequence length expansion
2. Is QLA’s SiLU activation valid in the SID token space? Meta’s verification is in the item embedding space
3. Number of Virtual seeds (128) vs our current user sequence length (21-30 items) — may need to be reduced
4. Applicability of Generative reconstruction loss in SID space: SID is a discrete token, direct MSE is not applicable → needs to be done in embedding space
5. Storage cost: 128 seeds × 256 dim × fp16 = 64KB/user — tens of millions of users = 640GB, feasibility needs to be evaluated

---

## IDEA-glorank-0: GloRank — SID-as-Global-Action-Space for Generative Reranking

**Priority**: P2 (We are currently in the retrieval stage, no reranker stage)
**Source**: GloRank (Kuaishou + UCSD + CityU HK, arxiv 2604.25291, Apr 2026)
**Status**: To be discussed - Mainly used as a reference for future reranking stages; among them, the "global identifier + 2-stage SFT→RL" training paradigm can be used for reference

### Core Idea

The traditional implementation of list-wise reranking is to select from N candidates by "position index (k-th position)" — but this results in a semantically inconsistent action space: the same output logit represents different items in different samples (depending on the order of input candidates). The author gives a strict theoretical analysis:

**Mathematical core (Proposition 2.1)**: Assuming that the target is fixed to item `r*` and the candidates are randomly arranged σ, then the lower bound of the "mapping-induced variance" of the label-dependent gradient received by the output parameter `w_j` in each row:

```
Var_σ(g_j,loc) ≥ (1/N)(1-1/N) |μ_j|²_2 > 0 (cannot be eliminated)
```

Even if the hidden state is completely stable, this variance will always exist because the "target to output row mapping" changes with σ.

**Solution**:
1. **Global action space**: Use Semantic IDs (SID) to map items to a fixed global token vocabulary, and the reranker outputs the SID token sequence instead of the local index
2. **Corollary 2.2**: In the global space, `Var_σ(g_glo) = Var_σ(h^t_σ)`, completely eliminating mapping-induced variance
3. **Two-stage training**:
   - **SFT pre-training**: Use high-quality reference list for behavior cloning
   - **RL post-training**: Directly optimize list-wise reward
4. **Constrained decoding**: Build a generation trie on the candidate set to ensure that the output is valid and non-duplicate

### Association with the current project

**The current project does not have an independent reranker stage** (end-to-end generative retrieval), so the main application scenario of GloRank is not applicable. But there are three insights that can teach us:

1. **The "Global identifier" paradigm is essentially the native design of our retrieval** - the NTP output is the global SID token, and naturally there is no mapping-induced variance problem. This theory verifies the correctness of our route
2. **Two-stage SFT→RL training** — The same idea as EXP-020 (NTP+DPO joint); GloRank uses list-wise reward for RL, benchmarking IDEA-rankgr-0 / IDEA-gr4ad-3
3. **Constrained generation trie over candidates** — We have equivalent implementations of `SIDTrie` + `constrained_beam_search`; GloRank is "constrained generation on given N candidates" and can be reused

### Experimental Design Draft

**Not executed in the current stage**, P2 archive. If reranker stage is introduced in the future:

**Phase 1**: Use EXP-020 checkpoint to output beam=500 candidates → train small SID-based reranker → SFT (top-10 by reward) → RL (list-wise NDCG)

**Phase 2**: Compare the gradient variance and training stability of local-index vs global-SID reranker to verify the theory of the paper

### Key questions

1. The current R@500=66.2%, is the reranker stage really necessary? The ROI is unclear
2. Reranker backbone computing overhead vs online latency
3. List-wise reward is difficult to define under our NTP data (lack of dwell time / interaction rate annotation)

### Related ideas

- IDEA-oneranker-0 (Tencent WeiXin unified generation+ranking): GloRank is a rerank-only version of SID
- IDEA-rankgr-0 (Taobao Listwise DPO + Rescore): Listwise RL routine
- IDEA-gr4ad-3 (GR4AD RSPO): NDCG-inspired RL reward
- IDEA-nsgr-0 (Meituan Next-Scale): Another coarse-to-fine generative reranking

---

## IDEA-a2gen-0: A2Gen — Action-Aware Generative Sequence (output user action sequence)

**Priority**: P1
**Source**: A2Gen (Kuaishou Beijing, arxiv 2604.25834, SIGIR 2026, 400M DAU full traffic deployment)
**Status**: To be discussed - It is two different directions from the completed IDEA-feat-1 (action input feature), A2Gen uses action as output

### Core Idea

The traditional recommendation model treats video as a "single item + binary tag" and ignores:

1. **Short videos with multiple clips are heterogeneous**: Users have different attitudes on different clips (Like Messi, not Ronaldo)
2. **Action timing distinguishes intentions**: Like during the climax of the video → Follow rate ↑3.3×, Collect rate ↑1.52×; random early Likes are mostly noise
3. **Action sequence difference**: `Follow→Like` sequence vs `Like→Follow` user watch time difference is 1.28×, comment rate difference is 1.66×

**A2Gen architecture**: Generate a complete **(action_type, timing)** sequence for each candidate item instead of predicting binary labels:

- **CAM (Context-aware Attention Module)**: MHA + integrate item context into query + learn task-specific importance of each head through gating + MLP post-processing
- **HSE (Hierarchical Sequence Encoder)**: two layers - Action-dim (action sequence within each item) → Item-dim (user history item sequence), nested CAM
- **AAG (Action-seq Autoregressive Generator)**: Autoregressive generation `{(A_i, T_i)}`, T_i uses regression to predict the relative time proportion

**Loss**:
```
L = α·L_cls(action type multi-class) + β·L_reg(timing MSE) + γ·L_order(reverse order prohibited max(T_p - T_q, 0)²)
```
Default α=1, β=1, γ=0.1.

**Online implementation of four cumulative strategies (Kuaishou 400M DAU full traffic)**:
| Strategy | Watch Time | Interaction | LT7 |
|------|-----------|-------------|-----|
| Model Replacement (A2Gen replaces PLE) | +0.11% | +2.1% | — |
| Action Timing Aware (lower Likes increase the power, filter early random Likes) | +0.13% | +3.5% | +0.12% |
| Action Sequence Aware (`Follow→Like` upgrade) | +0.10% | +1.4% | — |
| Action Timing Distribution Aware (sample weight increase near peak) | — | +1.1% | +0.042% |
| **Cumulative** | **+0.34%** | **+8.1%** | **+0.162%** |

### Association with the current project

**This is an important extension of our action level research line**:

- **IDEA-feat-1 (ActionType input feature, ✅ EXP-036)**: Inject action as **input** into NTP (L0/L1/L2 classified by behavior type + time_gap)
- **A2Gen**: use action as **output**, generate action sequence instead of SID sequence

**Key differences**:
- A2Gen is a reranking stage model (N candidates have been given by the upstream), not retrieval
- A2Gen does not generate SID, but generates "action sequence", and the input item ID is an atomic ID.
- Our NTP is retrieval, directly generating SID

**The most portable part**: The essence of online strategy 2/3/4 is **item-level action statistics as a post-processing ranking signal**, which can be implemented independently of the A2Gen architecture and does not require changes in the output architecture.

### Experimental Design Draft

**Phase 1 — Action statistics aggregation (data side, done independently)**:

Added `data/item_action_stats.parquet`:
| item_id | late_like_ratio | follow_like_seq_rate | timing_peak_alignment |
|---------|----------------|---------------------|----------------------|

Calculated from NTP training data (user behavior timestamps + behavior types we already have).

**Phase 2 — Input side action statistics feature (similar to feat-1 extension)**:

1. Add three `nn.Embedding` (three bucketed statistics) to `NTPModel`
2. `embed_with_features(tokens, positions, time_gaps, action_levels, late_like_bucket, follow_like_bucket, timing_peak_bucket)`
3. Compare baseline (feat-0/1/2 only) vs baseline + A2Gen statistical features
4. Expected return: R@500 +0.3~0.8pp (A2Gen single strategy online improvement rate)

**Phase 3 — Output action sequence prediction (big change, long term)**:

Vocab extension + points SID-pred / action-cls / timing-reg three heads. The benefits may not be greater than Phase 1+2, so stay in the future.

### Key questions

1. **The meaning of "Action" in our data set**: Our data comes from product behavior (click/cart/purchase), there is no correspondence with Like/Follow/Collect - first audit the current behavior category
2. **Action timing is there a signal**: Our event has a timestamp, but the "relative time within the session/item" does not exist in the current schema, we need to confirm
3. **Orthogonality with IDEA-feat-1**: feat-1 is action type embedding (L0/L1/L2 per item); A2Gen is item-aggregated action statistics. The two can be superimposed
4. **Phase 1 workload**: Redo data pipeline + 3 aggregate statistics ≈ 1 week; only Phase 2 baseline can be reduced to 2-3 days
5. **The upper limit of product retrieval scenario revenue is unknown**: A2Gen’s online numbers come from short videos, and our scenario may be smaller.

### Related ideas

- IDEA-feat-1 (ActionType input feature, ✅ EXP-036): input side action vs A2Gen output side, orthogonal
- IDEA-onelive-0 (OneLive BOS time injection): also uses action + time, but feature level
- IDEA-lac-0 (LAC delayed action): action as context token
- IDEA-mbgr-0 (MBGR multi-service GR): multi-task combination, Phase 3 multi-task idea is similar

---

## IDEA-cadet-0: Self-Gated Attention (Representation + Q/K Level 3 Gating)

**Priority**: P1
**Source**: CADET (LinkedIn, arxiv 2602.11410, Feb 2026, production deployment)
**Status**: To be discussed - partial changes to the attention variant, available as a drop-in upgrade on the EXP-020 baseline

### Core Idea

CADET is the decoder-only transformer of LinkedIn homefeed advertising CTR, **online A/B +11.04% CTR** compared to LiRank baseline (DCNv2 + sequence encoder hybrid ensemble), deployed on the main traffic of the billion member platform. Among the 5 core innovations, **Self-Gated Attention** is the most independent and portable component, directly targeting training instability and "attention sink" pathological behavior.

Different from the existing output-level gating (HSTU / OneLive's gated attention is multiplied by the output side gate control), CADET places the gate on the attention input end and Q/K projection to form a **three-level gating structure**:

**First level — Representation-level gate** (Feature selection of token representation before attention calculation):
```
Gate(X) = σ(W_X^gate·X)
Xə = X ⊙ Gate(X)
```
Function: Suppress noise dimensions, improve activation scaling and gradient condition numbers.

**Level 2 — Query gate** (Modulation Q):
```
Gate(Q) = σ(W_Q^gate · Q)
Qə = Q ⊙ Gate(Q)
```

**Level 3 — Key gate** (Modulation K):
```
Gate(K) = σ(W_K^gate · K)
Kə = K ⊙ Gate(K)
```

Effect: Constrain the dot-product amplitude of Q·K to prevent individual dominant tokens from monopolizing attention (ie **mitigate attention sink**). The paper reports that this design is key to training stability.

**4 other CADET innovations (not featured but worth knowing about)**:
- **Context-Conditioned Decoding Block**: K prediction heads, bucketed by ad position bucket (k=1 position1, k=2 position2-4, k=3 position5+), solving the problem that post-scoring features (ad position) are unknown during inference. Similar to IDEA-ocarm's feature leakage problem, but the solution is multi-head output instead of distillation
- **Timestamp-based RoPE**: Use Unix timestamp instead of sequence position for RoPE rotation, θ_i uses `φ_min/Δt_max · base^(2i/d)`, covering the time scale from seconds to months. **Highly overlaps with IDEA-torope-0 (Roblox Rotate Both Ways)**, torope retains order RoPE while adding time rotation, CADET is completely replaced by time — two implementation styles
- **Session-aware training mask**: The key of `t_j > t_i - Δ_delay` is removed from the mask during training, forcing the model not to rely on "events arriving with delay in the same session" (online tracking delay scenario). There is no train-serve skew for our offline training and offline evaluation, so it is not directly applicable
- **Production engineering**: tensor packing + sequence chunking + Flash Attention kernel for multi-item scoring

### Association with the current project

- **`ntp/model.py::_transformer_forward`** is a direct change target. We currently use standard PyTorch `nn.MultiheadAttention` to implement CADET self-gated attention. Just:
  1. Add 3 new `nn.Linear(d_model, d_model)` as `W_X^gate / W_Q^gate / W_K^gate`
  2. Before attention calculation, do sigmoid gate ⊙ multiplication for (X, Q, K) respectively.
  3. The rest of the attention remains unchanged
- Increased number of parameters: 3 × d_model² = about 3× single attention layer parameters (one Q/K/V projection is d_model², 3 gates are 3 × d_model² more). For our 45.8M baseline (d_model=384), each layer increases by about 0.44M, 8 layers total ~3.5M, **+8% parameters**
- If CADET's "attention sink mitigation" also holds true in our SID generation scenario, it may bring about training stability + final PPL improvement
- **Relationship with IDEA-hstu-0 (HSTU sparse attention)**: HSTU is a custom gated attention (output-side), CADET is input/Q/K-side gating. The paper clearly distinguishes in 3.2.1 "Unlike output-level gating used in prior work [14, 22]" (22 is HSTU). The two may be complementary or overlapping, requiring ablation
- **Relationship with IDEA-onelive-0**: OneLive uses gated attention to inject BOS time information, which is used for feature-injection; CADET's gating is used for attention stability. Orthogonal, can be combined

### Experimental Design Draft

**EXP-NNN — CADET Self-Gated Attention Integration**:

**Phase 1 — Minimal change verification (1 day experiment)**:
1. Add `SelfGatedMultiheadAttention` class in `ntp/model.py` (inherit or replace existing attention)
2. Add CLI switch `--use_self_gated_attention`
3. **relaxed re-train** on EXP-020 optimal config (exp020-hard-lam03): only attention is changed, other hyperparameters remain
4. Comparison:
   | Config | PPL | R@10 | R@500 | Notes |
   |--------|-----|------|-------|-------|
   | baseline (exp020) | 16.3 | 14.1% | 66.2% | no gating |
   | + self-gated (rep only) | ? | ? | ? | Only add the first level |
   | + self-gated (rep+Q+K, full) | ? | ? | ? | full CADET |

**Phase 2 — Comparison with existing attention variants**:
- baseline vanilla vs HSTU-style gated-output vs CADET three-level gated-input
- Add attention sink diagnostic indicators (maximum attention weight / uniform attention deviation) to verify CADET's "mitigate attention sink" statement

**Phase 3 (optional) — Overlay OneLive Gated BOS**:
- If both CADET's self-gating and OneLive's output-side gating have positive returns, try the combination

### Key questions

1. **Does the benefit of **CADET come from CTR prediction or general architecture improvements?** The LinkedIn scenario is binary CTR (1 action per impression), and ours is SID token generation (3 token per item). Attention stability should be generic, but the magnitude may be different
2. **Does **parameter volume +8% bring overfitting?** Our data size is smaller than LinkedIn, so we need to verify the config and regularity
3. **Compatible with FlashAttention / SDPA**: `F.scaled_dot_product_attention` does not directly support the gating hook after Q/K. You may need to customize the kernel or fall back to handwritten attention (throughput may decrease)
4. **Training Dynamic Comparison**: CADET claims "reliable convergence at scale", our baseline has been stable, the gain may be mainly in PPL / final index rather than training stability
5. Will **Timestamp-based RoPE be adopted at the same time?** We have IDEA-torope-0 (Roblox) which already covers the time RoPE topic. If torope is implemented in the future, the CADET style can be directly used as a sub-option
6. **Context-conditioned decoding block** is not applicable in our retrieval scenario (there is no post-scoring contextual feature like ad position); but if it is expanded to ads in the future, this is ready-made

### Related ideas

- IDEA-hstu-0 (ULTRA-HSTU Sparse Attention): HSTU output-side gating, CADET input+Q+K-side gating, clearly distinguished in the paper
- IDEA-onelive-0 (OneLive Gated BOS time injection): gated attention is used for feature injection, orthogonal
- IDEA-ksa-0 (Summary Attention): KV cache compression, no conflict
- IDEA-vista-0 (QLA): attention form replacement, competitive relationship
- IDEA-torope-0 (Roblox Time-and-Order RoPE): Another implementation of time RoPE, CADET's timestamp RoPE is a radical replacement
- IDEA-ocarm-0 paper (reference only, 2604.25839): Use distillation to solve the missing post-scoring signal; CADET uses multi-head output to solve the problem, two complementary ideas

---

## IDEA-recochain-0: Fusion of generative retrieval + target-aware rearrangement (KV-cache reuse) in a single Transformer

**Priority**: P2
**Source**: RecoChain — Harmonizing Generative Retrieval and Ranking in Chain-of-Recommendation (Kuaishou Jiangxia Cao, arxiv 2604.25787)
**Tier**: B (Kuaishou corresponding author, offline TAOBAO-MM, no A/B; short paper "work in progress")
**Status**: To be discussed

### Problem: GR beam candidates are difficult to sort

GRs such as OneRec / TIGER generate `K=256` candidate SIDs through hierarchical beam search, but the **next-item-agnostic** modeling method cannot accurately estimate "which 10 of these 256 are the best". The method of industrial two-stage pipeline is:

- Retrieval (GR): `P(next_item | user_hist)` → Retrieve 256 items
- Ranking (DIN/SIM/TWIN/RankMixer): `P(click | user, item_feat)` → fine ranking top10

The existing GR distillation ranking is used as a reward model (OneRec-V2 / Climber / ReCast are all this route), but the basic target-item aware searched-sequence modeling (the core of SIM) is not included in GR, so the ranking ability is still poor.

### RecoChain approach: one decoder for two stages

The key is to move SIM's target-aware behavior sequence retrieval into the Transformer generation path**:

1. **Retrieval stage**: decoder generates `K` SIDs for `user_hist` → hierarchical beam, and decodes `K` candidate item_ids
2. **Ranking stage**: For each candidate `i(c)`, use cosine similarity to retrieve top-M similar items from **the entire user history** (SIM style GSU), put the tokens of these M items behind the beam SID, and then append a candidate item_id token as "rank token"
3. **KV cache reuse**: The above two steps are performed in the same decoder for incremental computation. The KV of user_hist is not recalculated, and the beam SID part is also reused.
4. **Rank head**: Connect MLP head at rank token → sigmoid → click probability
5. **Loss**: Stage-I pure SID generation CE; Stage-II simultaneous SID CE + binary CE (positive sample = true target SID match, negative sample = other SID in beam)

### Offline effects (TAOBAO-MM, data set only)

- Base (beam-only) R@5=0.2384, Rerank 0.2459 (**+3.14%**, beam=20, seq=32, retrieval=10)
- Beam size 10→40: rerank gain 0.27%→1.08%, that is, the larger the beam, the greater the rerank gain.
- Retrieval length 0→20: rerank gain 0.12%→3.51%, the more similar sequences GSU searches, the better

### Association with the current project

- **Benchmarking IDEA-onerec-3 (Reward Model integration)** / **IDEA-orecv2-0** / **IDEA-recast-0**: These are "use ranker signal shape GR to generate probability"; RecoChain is "let GR do ranking by itself". The two roads have fundamentally different routes
- **Benchmarking IDEA-glorank-0 (GloRank Kuaishou Reranking)**: GloRank is a reranking model, but uses an **independent** generative reranker; RecoChain is the **same** decoder in both front and back sections. The degree of coupling is higher, but it also saves computing power.
- **Target-aware GSU is the core of SIM [arxiv 2006.05639]**, which is cited in the paper; the essence of this idea is "OneRec + SIM end-to-end"

### Experimental Design Draft

**Not executed in the current stage. ** Prerequisites:

1. We currently only have retrieval training (NTP single task) and no ranking label (two-class click/conv). You need to have the pos/neg tag of RTB/exposure log before you can train the rank head
2. The relative gap of R@500 at this stage (64.1% vs 70.1%) is mainly due to tokenizer + sequence scaling, and the relative benefit of rank head may not be greater than these.

If ranking label is added in the future:

- Stage A: Keep the existing NTP training unchanged, add rank head to checkpoint, and freeze backbone fine-tuning (quick verification)
- Stage B: joint training, weighted sum of two losses

Experimental granularity: Comparing `beam-only R@K` vs `beam+rerank R@K`, it is expected that rerank will improve more obviously when the beam size is large (paper conclusion)

### Key questions

1. **KV cache design**: Our `constrained_beam_search` in `eval.py` already has KV cache reuse, but the rank stage requires appending **M retrieved item tokens + 1 candidate item_id token**. This set of incremental needs to add a new path, which is not trivial.
2. **item_id token vocabulary**: In addition to SID token, RecoChain also has an additional item_id embedding table (one id token for each item). We are currently using pure-SID. Adding item_id table will cost extra params on the model side.
3. **GSU search**: cosine similarity searches top-M in the entire user history. This is a standard practice for industrial SIM, but it must be pre-calculated during offline training and done in real time during inference (M=10 is acceptable)
4. **Rank label source**: We currently do not have a binary label for "whether the next item is really clicked"; all negative samples in the beam that are not target are calculated. This structure is consistent with the paper, but it may be too sparse (positive/negative = 1/beam_size)

### Why Tier B instead of Tier A

- Kuaishou Corresponding author Jiangxia Cao (First work in the OneRec series) → High industrial relevance
- But the experiment is only on the **public TAOBAO-MM data set**, and there is no Kuaishou internal online A/B or deployment report
- Short paper (5 pages), titled "work in progress", method details (specific implementation of GSU, BCE construction of rank head) are not yet complete

### Related ideas

- IDEA-onerec-3 / IDEA-orecv2-0 / IDEA-recast-0: GR + reward model distillation route
- IDEA-glorank-0: Another generative reranking scheme by Kuaishou, independent reranker
- IDEA-vista-0 (VISTA Meta): target-aware UIH attention, similar in idea but not included in GR decoder
- IDEA-a2gen-0 (A2Gen): Extended GR output to (action, timing), orthogonal
