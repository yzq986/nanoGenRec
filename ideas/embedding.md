# Embedding (representation enhancement)

[English](embedding.md) | [Chinese](embedding.zh.md)

The quality of embedding before quantization determines the upper limit of semantic ID. Covering directions such as collaborative signal injection, multi-modal fusion, and attribute enhancement, it is orthogonal to quantification methods and improves embedding to benefit all downstream experiments.

**Scope of influence**: `model/encode.py`, `model/embedders.py`, `data/export_behavior.py`

---

## Evolution path

```
Qwen3 plain text embedding (1024D, current baseline)
├── IDEA-sid-1: direct fine-tune Qwen3 (I2I comparative learning)
│ └── EXP-007 Verification: full/LoRA fine-tune are invalid, HR@50 stuck at ~0.02
├── IDEA-onerec-3: QFormer Tokenizer (freeze Qwen3 + cross-attention compression) ★ Recommended
│ └── OneRec/BLIP-2 solution, information bottleneck + gradient concentration, solves the fundamental problem of EXP-007
├── IDEA-onerec-0: Caption Loss (anti-forgetting semantics, implemented --cap_loss_weight)
├── IDEA-onemall-3: attribute enhancement (category/price/shop comparison learning)
├── IDEA-sid-3: Multimodal ESANS (coarse and fine-grained multimodal fusion)
└── IDEA-oneloc-3: Side-info fusion (quantified input space enrichment)
└── Unified with IDEA-sid-1 into the "embedding enrichment" framework
```

---

## IDEA-sid-1: Collaborative signal enhancement Embedding

**Priority**: ~~P1~~ → ❌ Close
**Source**: 3.1.1 (OneRec-V1 Technical Report)
**Status**: ❌ Closed — EXP-007 full fine-tune + LoRA multiple lr/τ all failed (HR@50 stuck at ~0.02); EXP-009 frozen base + QFormer same HR@50=0.0216 with little improvement. Root cause: I2I contrastive signal is not enough to bridge the gap between semantic embedding and behavioral space. embedding fine-tune routes closed.

### Core Idea

Currently, Qwen3 text embedding is directly used for quantification, but the semantic similarity required by the recommendation system also includes collaborative behavior signals. Inject synergistic signals into embedding through Item Pair contrastive learning.

### Association with the current project

- There is already `data/export_behavior.py` to export behavior data
- Existing `eval/behavior.py` behavioral indicator evaluation framework
- Improvements to embedding itself will benefit no matter what quantization scheme is used later.
- Orthogonal to the quantitative method experiment (EXP-003, IDEA-sid-0) and can be advanced in parallel

### Experimental Design Draft

**Item Pair Construction**:
- Method 1: User clicks on target item + recent positive behavior item
- Method 2: Swing I2I high score item pair

**Training Plan**:
- Solution A (lightweight): freeze Qwen3, only train projection head, and compare learning loss
- Solution B (weight): fine-tuning Qwen3-0.6B, comparative learning loss + text loss

**Evaluation**: Original Qwen3 embed vs enhanced embed → same RKMeans config → comparison collision / exclusivity / behavior metrics

### Key questions

1. Is the Item Pair sample size sufficient (behavioral data coverage needs to be checked)
2. Is option A (projection head) sufficient, or does it need to finetune the entire model?
3. Negative sample strategy for contrastive learning: in-batch negatives? hard negatives?

---

## IDEA-sid-3: Multi-modal semantic ID (ESANS coarse and fine granularity)

**Priority**: P2
**Source**: 3.1.3 (Alibaba WWW'25 ESANS)
**Status**: To be discussed

### Core Idea

Multimodal representation is not a simple concat re-quantification, but:
- L1 (coarse-grained): multi-modal representation mean is used for clustering
- L2 (fine-grained): Each modal residual is concated and then clustered

### Association with the current project

- Currently only using `qwen3-0.6b` single text mode
- There are already `qwen3-vl-8b` / `qwen3-vl-2b` multi-modal models in Config
- Multimodal embedding infrastructure needs to be completed first

### Prerequisites

1. Multimodal embedding generation pipeline (using qwen3-vl)
2. Multimodal representation alignment (CLIP-style or ESANS encoder)
3. Data side: Image/video data of item is required

### Key questions

1. Storage and computational cost of multimodal embeds (4096D of qwen3-vl-8b)
2. Complexity of modal alignment training
3. Whether to first verify the benefits of coarse- and fine-grained solutions on small-scale data

---

## IDEA-onemall-3: Tokenizer Auxiliary Contrastive Loss (attribute enhancement)

**Priority**: P1
**Source**: OneMall §4.5 Component Analyses (Aux Loss row)
**Status**: To be discussed

### Core Idea

In the embedding backbone training of tokenizer, add item attributes (category, price, shop) as auxiliary signals. OneMall feeds these attributes into the item tower and uses comparative learning loss training to increase HR@50/100/500 by +1.5%/+1.7%/+1.7% respectively.

This is complementary to IDEA-sid-1 (Coordinated Signal Enhancement embedding):
- IDEA-sid-1: Inject coordinated signals using user behavior I2I pairs
- This IDEA: Use item attributes to inject structured business semantics

### Association with the current project

- The current embedding comes purely from Qwen3 text encoding, with no structured attribute injection
- item metadata (category, brand, price) should be available in behavioral data
- You can add attribute projection head based on `Qwen3TextEmbedder` in `model/embedders.py`
- **Same direction as EXP-003 (Learned FSQ)**: both improve embedding quality before quantization

### Experimental Design Draft

**Option A (Lightweight – recommended first)**:
- Freeze Qwen3, add `AttributeProjectionHead`: MLP(attr_features → 128)
- item text embedding (1024D) + attribute embedding (128D) → concat → MLP → final embedding
- Comparative learning: item pairs in the same category are used as positive samples, and items in different categories are used as negative samples

**Option B (weight)**:
- Merged with IDEA-sid-1: I2I cooperative signal + attribute signal injected at the same time

**Evaluation**: Original Qwen3 embed vs attribute-enhanced embed → same RKMeans config → collision / exclusivity / behavior metrics

### Key questions

1. Need to confirm which item attribute fields are available in the data
2. How to code the category hierarchical structure (first-level/second-level/third-level classification)
3. Discretization/normalization strategy for continuous attributes (price)

---

## IDEA-oneloc-3: Geo-aware Semantic ID (Side-info fusion quantization input)

**Priority**: P1
**Source**: OneLoc §2.2 Geo-aware Semantic IDs
**Status**: To be discussed

### Core Idea

OneLoc incorporates geographic context in the initial embedding of residual quantization: `r_0 = concat(e_video, e_location_context)`, so that the generated semantic ID itself encodes geographic semantics. Geographical context is extracted from GeoHash’s brand, category, and best-selling product information by a large multi-modal model.

### Association with the current project

- The current input to `model/rkmeans.py` is pure Qwen3 text embedding
- This is consistent with the idea of **IDEA-sid-1 (Coordinated Signal Enhancement Embedding)**: inject additional signal into the embedding before quantization
- Generalized form: **Any side information can be concat/fuse into embedding before quantization**
- Specific inspiration for us: In addition to the synergistic signal (IDEA-sid-1), you can also inject:
  - Category level embedding
  - price range embedding
  - Hotness/freshness signal
- Essentially **enrichment of quantified input space**

### Experimental Design Draft

**Option A: concat + MLP fusion**
- Input: `concat(qwen3_embed_1024d, side_info_embed_128d)` → MLP → 1024d
- do RKMeans (or OPQ) on fused embedding

**Option B: Weighted Residuals**
- Weighted fusion in RKMeans first layer input: `r_0 = α·e_content + (1-α)·e_side`

**Evaluation**: Compare the performance of pure content embed vs fused embed on quantitative indicators and NTP recall

### Key questions

1. **Overlap with IDEA-sid-1**: Collaborative signal enhancement also changes the embedding input. Should be unified into an "embedding enrichment" framework to avoid repeated experiments
2. What side information is currently available? You need to check what fields are there in the item metadata besides text.
3. The impact of increased dimensionality after Concat on quantization quality - high dimensions may make it more difficult for KMeans to converge
4. If you take the IDEA-sid-0 (OPQ) route, side info can be assigned to independent sub-vectors, which is naturally suitable for parallel quantization.

---

## IDEA-onerec-0: Caption Generation Loss (to prevent collaborative fine-tuning from forgetting semantics)

**Priority**: P1
**Source**: OneRec (arxiv 2506.13695v4) §Tokenizer Training
**Status**: Awaiting discussion — Directly related to EXP-007

### Core Idea

OneRec also adds **caption generation loss** to the contrastive learning training of tokenizer:
- Compare loss (`L_I2I`): bring the collaborative pair closer
- Caption loss (`L_caption_gen`): Given the multi-modal representation of the item, predict the text caption of the item (next-token prediction)

The role of Caption loss is to **"prevents hallucination by performing next-token prediction on video captions"** — to prevent contrastive learning from overfitting the collaborative signal and losing content semantics.

### Association with the current project

- **EXP-007 Currently only InfoNCE loss**, no semantic preservation mechanism. If embedding loses text semantics after 3 epoch training (cosine_similarity distribution becomes worse), it means that caption loss needs to be added
- Qwen3-Embedding-0.6B is an encoder model and does not directly support causal LM generation.
- **Alternative**: Use contrastive loss to maintain semantics — use the same embedding before and after item fine-tuning as positive samples (anchor preservation), or add a lightweight text reconstruction head

### Experimental Design Draft

**Option A — Embedding Anchor Preservation (recommended, the simplest)**:
```
L = L_InfoNCE + β * L_anchor
L_anchor = 1 - cos(embed_finetuned, embed_original)
```
Freeze a copy of the original Qwen3 as the anchor, and the fine-tuned embedding cannot be too far from the original.

**Option B — Text Reconstruction Head**:
- Add a lightweight decoder head to the Qwen3 encoder output
- Predict item title tokens
- `L = L_InfoNCE + β * L_text_recon`

**Evaluation**: Compare the change of embedding with/without caption loss on `embedding_hit_rate` + `cosine_similarity`

### Key questions

1. Selection of β: too large will suppress collaborative learning, too small will have no effect
2. The anchor preservation of solution A may be too conservative—limiting the movement range of the embedding space.
3. After the results of EXP-007 come out, check whether `cosine_similarity` is degraded and decide whether it needs to be added.

---

## IDEA-onerec-3: QFormer Tokenizer (Freezing Base + Cross-Attention Compression)

**Priority**: ~~P0~~ → P2 Suspended
**Source**: OneRec (arxiv 2506.13695v4) §Tokenizer + BLIP-2 QFormer
**Status**: On hold - EXP-007 has verified that direct fine-tune is ineffective; QFormer is the theoretically correct direction, but currently NTP has made progress by MLP-FSQ + RL alignment, and the embedding improvement route ROI is unclear. Prioritize completion of RL link (EXP-037→039) before evaluation

### Core Idea

The fundamental problem of EXP-007: Doing contrastive fine-tune directly on Qwen3-0.6B (regardless of full volume or LoRA) cannot push the model - cap_loss does not move at all, and HR@50 is stuck at ~0.02.

OneRec's solution: **Don't move the base, add a trainable QFormer on it**.

```
OneRec Architecture:
  miniCPM-V-8B (frozen, 8B) → 1280 tokens × 512d
      ↓
  QFormer (trainable, 4 layers, 4 query tokens)
      ↓
4 tokens × 512d (compressed item representation)
      ↓
  L_I2I (InfoNCE) + L_caption_gen (next-token prediction)
      ↓
RQ-KMeans → Layer 3 SID

Our adaptation:
  Qwen3-Embedding-0.6B (frozen) → S tokens × 1024d (last hidden states)
      ↓
  QFormer (trainable, N layers, M query tokens)
      ↓
  M tokens × D (compressed item representation)
      ↓
  L_I2I (InfoNCE, existing) + L_caption (implemented --cap_loss_weight)
      ↓
  OPQ quantification → SID
```

### Why QFormer can solve the problem of EXP-007

| Problem | Direct fine-tune (EXP-007) | QFormer |
|------|---|---|
| Gradient signal dilution | I2I Gradient is spread to 600M Parameter, which is approximately equal to none | Gradient set Medium is in QFormer (~30-50M), the base is frozen |
| Semantic forgetting | cap_loss monitoring unchanged = Model does not move | Base frozen = naturally maintained semantics |
| Information bottleneck | None, all 1024d is transmitted | S×1024 → M×D forced compression, learn to extract collaborative related information |
| Optimization goal | "Fine-tune the entire representation space" (too big) | "Learn what to choose from a rich representation medium" (more direct) |

### QFormer key design (from BLIP-2 + OneRec)

**Learnable Query Tokens**: M trainable query vectors that "query" the hidden states of the frozen encoder via cross-attention.

**Cross-Attention Mechanism**:
```
Q = learnable_queries (M × D)
K, V = encoder_hidden_states (S × 1024)
Output = CrossAttn(Q, K, V) (M × D)
```

**Key hyperparameters**:
- M (query tokens): OneRec uses 4, OneMall uses 10/type. We can search from {4, 8, 16}
- QFormer layers: OneRec 4 layers. Starting from {2, 4}
- Output dim D: aligned with OPQ subvector dimensions (currently m=8, sub_dim=128 → D=1024 or compressed to 512)
- Final embedding: mean-pool M query tokens → single vector → OPQ

### Experimental Design Draft

**Phase 1 — Minimal verification (verify that gradients can flow)**:
- M=4 query tokens, 2-layer QFormer, D=1024
- Freeze Qwen3, only train QFormer
- L_I2I only, 500K pairs, lr=1e-4
- Pay attention to: whether cap_loss starts to change, whether HR@50 breaks through 0.02

**Phase 2 — Add Caption Loss**:
- L = L_I2I + λ * L_caption (--cap_loss_weight)
- Compare the HR@50 difference with/without caption loss

**Phase 3 — Hyperparameter search**:
-M∈{4, 8, 16}
- QFormer layers ∈ {2, 4}
- lr ∈ {1e-4, 5e-4, 1e-3}

**Evaluation**: HR@50 (direct comparison with EXP-007), cap_loss change

### Implementation points

1. **New `model/qformer.py`**: QFormer module (cross-attention + FFN + learnable queries)
2. **Modify `model/contrastive_finetune.py`**:
   - `--use_qformer` flag
   - Freeze Qwen3, take `last_hidden_state` (not just the last token)
   - QFormer handles hidden states → gets compressed representation
   - Compressed representation using InfoNCE + caption loss
3. **Modify `model/encode.py`**: Load QFormer during inference and generate compressed embedding
4. **Quantification pipeline**: OPQ input dimensions may change and need to be adapted

### Differences from architecture.md IDEA-onemall-1

| | IDEA-onemall-1 (architecture.md) | IDEA-onerec-3 (this IDEA) |
|---|---|---|
| Layer | NTP ranking Phase | Embedding/Tokenizer Phase |
| What to compress | User behavior sequence (1205→160 tokens) | Item multi-modal/textual representation (S→M tokens) |
| Purpose | Reduce NTP decoder FLOP | Produce item embedding for quantification |
| Training signal | NTP next-token loss | I2I contrastive + caption loss |

The two are applications of QFormer at different stages. They do not conflict with each other and can coexist.

### Key questions

1. **Output format**: QFormer outputs M tokens, OPQ expects a single vector. Requires pooling (mean/cls) or expansion into longer vectors
2. **Qwen3-Embedding is an encoder**: hidden states are bidirectional (non-causal), and QFormer’s cross-attention can make use of all contexts
3. **Training cost**: QFormer ~30-50M parameters, slightly larger than LoRA but much smaller than full fine-tune
4. **Inference changes**: One more QFormer forward needs to be run when encoding, which increases the inference time by ~5%.

---

## Priority summary

| Priority | ID | Experiment | Reason |
|--------|-----|------|------|
| ~~P0~~ P2 Suspended | ~~IDEA-onerec-3~~ | ~~QFormer Tokenizer~~ | Suspended - NTP+RL route has made progress, embedding rerouting ROI is unclear, evaluation will be done after the RL link is completed |
| P1 | IDEA-onerec-0 | Caption Loss (joint with Training) | Implemented `--cap_loss_weight`, cooperate with QFormer Usage |
| P1 | IDEA-onemall-3 | Tokenizer attribute enhancement Contrastive | OneMall +1.5% HR, can be superimposed on QFormer |
| P1 | IDEA-oneloc-3 | Side-info fusion quantization Input | QFormer Input side can be fused side-info |
| ~~P1~~ ❌ | ~~IDEA-sid-1~~ | ~~Direct fine-tune synergy signal~~ | ❌ EXP-007 full/LoRA + EXP-009 QFormer completely failed, HR@50 stuck at 0.02 |
| P2 | IDEA-sid-3 | Multimodal Semantic ID (ESANS) | Requires multimodal embedding infrastructure |
| P1 | IDEA-marc-0 | Mid-Layer selection + Modular Compression | MARC SIGIR 2026 eCPM +2.82% A/B; Phase 1 (layer sweep) almost zero cost, directly affects the SID quality upper limit |

---

## IDEA-marc-0: Mid-Layer Representation Advantage + Modular Compression

**Priority**: P1
**Source**: MARC (Huawei Noah + SJTU, arxiv 2604.18146, SIGIR 2026)
**Status**: To be discussed — Phase 1 (layer sweep) can be executed immediately, low risk

### Core Idea

MARC systematically studied the issue of "from which layer should embedding be taken when using LLM representation for recommendation" and came to two important findings:

**1. Mid-layer Representation Advantage (MRA) — Counterintuitive Phenomenon**

When doing CTR fine-tuning on Llama3-8B / Qwen2-7B / Qwen2-1.5B (comparative learning, MRL, LARR, next-token, cosine-sim various proxy tasks), the downstream CTR AUC of the intermediate layer representation is always better than the final layer**, and it appears consistently no matter what proxy loss is used. The author replicated this phenomenon on both MovieLens-1M and Yelp.

**2. Modularity theory explanation**

LLM automatically forms functional division of labor during fine-tuning:
- **Representation Learning Module (early to middle layer)**: Extract common semantic features and retain rich information
- **Task Adaptation Module (last few layers)**: forced to collapse into a task-specific header by proxy loss, filtering out diversity information that is useful for CTR but "unnecessary" for proxy tasks

The final layer is actually an **unintended information bottleneck** — squeezing out signals useful for recommendation. This explains why it is better to take the middle layer.

**3. MARC Framework (Explicit Modularity)**

Three components are decoupled:
- **LLM backbone**: only does representation learning and is not imposed with task head responsibility
- **Compression Network**: independent lightweight network for dimension compression (hidden dimensions from LLM → recommended dimensions)
- **User-Item Matching Network**: independent network for CTR-style matching/prediction

Add **HSIC (Hilbert-Schmidt Independence Criterion)** as a constraint: maximize the mutual information between pre- and post-compression representations, while forcing the outputs of the compression and matching modules to be independent of each other.

**4. Experimental results**

- MARC's final layer representation surpasses the best intermediate layer of all baselines (MARC fixes MRA)
- Online A/B test **eCPM +2.82%** (Huawei commercial search advertising scenario)

### Association with the current project

**This is a direct challenge to our Qwen3-0.6B → MLP-FSQ pipeline**:

- Currently `Qwen3TextEmbedder` takes the final layer (EOS token pooling or last hidden state), 1024D → MLP-FSQ
- MARC hint: **embedding in the middle layer may produce better SID** — retaining more semantic diversity and reducing tokenizer reconstruction pressure
- The fine-tune route (sid-1) of our EXP-007/009 failed because imposing CF proxy actually crushed the final layer - MARC's theory exactly explains this phenomenon
- **Phase 1 experiment zero cost**: Do not change the fine-tune logic, only change "which layer of hidden state to take" → rerun tokenizer → recalculate semantic_neighbor_HR

Relation to EXP-007/009:
- EXP-007/009 conclusion ("fine-tune route is blocked") still holds
- MRA provides a new explanation: final-layer will always degrade during fine-tuning, and has almost nothing to do with "what proxy loss is used"
- New path: Keep Qwen3 **not fine-tune**, but **change the layer to get embedding** — possibly bypass fine-tune failure and get a better SID

Relationship with IDEA-onerec-3 (QFormer Tokenizer):
- QFormer idea: Freeze base + Cross-Attention from multi-layer aggregation → MARC is the theoretical support of QFormer (multi-layer aggregation is better than pure final layer)
- MARC Phase 1 (single-layer selection) is much lighter than QFormer (multi-layer aggregation) and can be used as a pre-ablation

### Experimental Design Draft

**Phase 1 — Qwen3 Layer Sweep (very low cost, ~1 day)**:

1. Add `hidden_layer: int = -1` parameter in `model/embedder.py::Qwen3TextEmbedder`
2. Use `output_hidden_states=True` + `hidden_states[hidden_layer]` to get the specified layer
3. Qwen3-0.6B has ~28 layers, sweep `{2, 7, 14, 21, 27}` (early, mid-early, mid, mid-late, final)
4. For each layer:
   - Recalculate embedding cache (~hours)
   - Retrain MLP-FSQ tokenizer (~30 minutes)
   - Evaluate `semantic_neighbor_hit_rate@50` (seconds)

**Expectation**: If MRA also holds true on Qwen3 (the paper is reproduced on Llama3/Qwen2), it should be observed that the mid-layer (layer 14-21) semantic_neighbor_HR is significantly better than the final layer (27).

| Hypothetical Result | Interpretation | Next Step |
|---------|------|-------|
| mid > final is significant | MRA is established in our scenario | Switch the default layer, retrain NTP, and see the end-to-end R@K improvement |
| mid ≈ final | Our embedding Path is not affected by MRA (Qwen3 pre-Training + no fine-tuning may not comply with MARC Hypothesis) | Skip Phase 2, but at least rule out one variable |
| mid < final | MRA reverse | Unexpected, in-depth Analysis is required; maybe our MLP-FSQ has implicitly compensated |

**Phase 2 — MARC complete framework (high cost, requires fine-tune)**:

If Phase 1 shows mid-layer advantages, but single-layer options have limited benefits, consider full MARC:
- Introducing an independent Compression Network (currently there is an MLP-FSQ encoder, which can be regarded as already available)
- Introducing User-Item Matching Network (independent small MLP for interaction)
- Add HSIC constraints (during tokenizer training phase)
- Perform task-aware fine-tune on Qwen3 (but add HSIC protection final layer)

Phase 2 has major changes, and MARC is a CTR sorting scenario, not a generative recommendation - you need to carefully evaluate whether to transplant directly. In our scenario, "Matching Network" corresponds to the NTP model itself, not an independent small network.

**Phase 3 — Multi-level aggregation (an instantiation of IDEA-onerec-3 QFormer)**:

If Phase 1 shows mid > final, you can also try:
- Take the middle layer hidden states + final layer hidden states, concat → linear projection → send MLP-FSQ
- Or use learned attention over layers (mini-QFormer) to do weighted average

### Key questions

1. Does **MRA occur on Qwen3 without fine-tuning? ** The experiments in the paper are all "after fine-tuning" LLM. Our Qwen3 is a frozen pre-trained model and may not have final-layer degradation issues. But even so, the middle layer may be more suitable for recommendation tasks (the final layer is biased by the LM head during pre-training)
2. **Qwen3’s EOS/last-token pooling is only meaningful in the final layer**: If you take the middle layer, the pooling strategy must be changed simultaneously (mean pooling over non-pad tokens is more appropriate)
3. **HSIC implementation complexity**: The HSIC constraints of Phase 2 require calculation of the kernel matrix, which is O(B²) memory on large batches and is not directly compatible with the multi-GPU mode of the current tokenizer training.
4. **Consistency with VL embedder**: If the text embedder is cut to mid-layer, should the image side `Qwen3VLEmbedder` also be cut? The multi-modal fusion layer of VL is usually in the middle and rear segments and may not be consistent

### Related ideas

- IDEA-sid-1 (CF fine-tuning enhancement): ❌ Failure - MRA theory just explains why the CF fine-tuning final layer degrades
- IDEA-onerec-3 (QFormer Tokenizer): multi-layer aggregation path, Phase 3 is its simplified version
- IDEA-forge-0 (Proxy Metrics): `semantic_neighbor_HR` is the evaluation metric of this experiment
- IDEA-snap-0 (Snapchat SIDs): Snapchat's multi-modal embedding fusion + STE processing codebook collapse, and this idea both point to "the embedding end is critical to SID quality"

---

## IDEA-gatesid-0: GateSID — Adaptive gate control semantics by item maturity - collaborative signal

**Priority**: P2 (currently no independent collaborative embedding, the scene does not completely match)
**Source**: GateSID (arxiv 2603.22916, Mar 2026)
**Status**: To be discussed - the main scenario is "semantic SID + collaborative atomic ID twin towers", we are pure SID; but Gate-Regulated Contrastive Alignment can be used as a reference for tokenizer training

### Core Idea

GateSID solves the collaborative-semantic tradeoff of cold start scenarios: popular items have reliable collaborative signals, cold start items have sparse collaborative signals, and the two extremes require different signal combinations.

**Industrial A/B (not stated explicitly by the company, large-scale e-commerce scenario)**:
- GMV **+2.6%**, CTR **+1.1%**, orders **+1.6%**, additional latency **< 5 ms**

**Architecture**:
1. **Semantic Basics**: multimodal features → RQ-VAE → hierarchical Semantic IDs
2. **Gating-Fused Shared Attention**: intra-modal attention distribution + **per-item gating weight** (from embedding + statistical features such as popularity/interaction stats) for fusion
3. **Gate-Regulated Contrastive Alignment**: The intensity of cross-modal contrastive loss is modulated by gate:
   - Cold start item → gate **tight** → force semantic-behavior alignment (to compensate for collaborative sparsity)
   - Popular item → gate **lax** → retain the collaborative characteristic structure and will not be homogenized by semantics

### Association with the current project

**The current application scenario does not exactly match**: We are a pure SID retrieval (no independent collaborative atomic ID embedding coexists with semantic SID), so the main double-tower gate control of GateSID is not directly applicable.

**Some lessons to be learned**:
- **Per-item gating by maturity** is a common technology that can be used for our tokenizer training (cold start items use stronger semantic regularization)
- **Item statistical features as gate input** (popularity, interaction count) — We already have similar features in IDEA-feat-0/1/2, which can be extended to gate signal
- If hybrid retrieval (SID + atomic ID) is introduced in the future, GateSID will be a ready-made framework

### Experimental Design Draft

**Not currently executed**, P2 archive. If cold-start special processing is introduced in the future:

**Phase 1 — Item maturity bucket injection**:
- During NTP training, buckets are divided according to item interaction count (new/medium/hot)
- The training samples of new items are weighted more heavily (existing NTP loss × maturity-weight)
- Expectation: Significant improvement in new item Recall@500

**Phase 2 — Gate-regulated tokenizer re-alignment**:
- During MLP-FSQ training, use stronger contrastive-to-semantic regularization for cold items
- Allow embedding to deviate from raw semantics for popular items to reflect collaborative signals

### Key questions

1. **Our data distribution**: What is the proportion of cold start items? If < 5%, the ROI of the GateSID solution is limited
2. **Embedding vs atomic ID dual-channel**: GateSID assumes that both coexist; we don’t have it, we have to introduce it first to have space.
3. **Comparison with IDEA-adasid-0 (AdaSID Adaptive Collision)**: AdaSID adjusts the collision strength from the tokenizer end; GateSID adjusts the semantic-collaborative weight from the embedding end. The two complement each other

### Related ideas

- IDEA-adasid-0 (AdaSID): Tokenizer side collision adaptive, comparison ideas
- IDEA-sid-1 (Coordinated signal enhancement, ❌ off): Tried I2I contrastive injection, failed; GateSID avoids excessive intervention through gate
- IDEA-feat-0/1/2 (✅ EXP-036): side features injected, maturity feature extensible
- IDEA-flexcode-0 (FlexCode): Dual codebook CF+Semantic route, similar to the GateSID twin tower idea
