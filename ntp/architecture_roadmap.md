# NTP Architecture Evolution Roadmap

[English](architecture_roadmap.md) | [Chinese](architecture_roadmap.zh.md)

An iterative path to gradually evolve from the current NTPProbe (2L decoder-only, 5M params) to the OneRec-level encoder-decoder architecture.

Each stage is independently measurable, and the indicator regression of the previous stage is the baseline of the next stage.

---

## Current starting point

`ntp/baseline.py` — **NTPProbe** (original baseline)

| Project | Current Status |
|------|------|
| Architecture | Decoder-only (nn.TransformerDecoder) |
| Number of layers | 2 |
| d_model / heads / FFN | 256 / 4 / 512 (Dense) |
| Parameter amount | ~5M |
| User Representation | None (behavior sequence implicitly encoded) |
| Input | 10 items × 3 SID tokens = 30 tokens |
| Decoding | Beam search (beam=5 Training, 50 eval) |
| Training | Sliding window (input_30→target_3), per-layer CE, DDP, 1 epoch |

---

## Stage 1: S-tier Decoder + Loss-Free MoE ✅

**Status**: Completed (model + training + evaluation full link)

**Source**: `sid_prediction_old.py` migration + IDEA-onemall-4 + IDEA-mtgr-0

### 1a — S-tier Model (`ntp/model.py`)

| Project | NTPProbe → NTPModel |
|------|--------------------------|
| Number of layers | 2 → 6 |
| Heads | 4 → 8 |
| FFN | Dense 512 → SwiGLU MoE (8E, top-2, expert_dim=1024) |
| Load balancing | N/A → Loss-Free dynamic bias (replaces Switch aux loss) |
| Parameter amount | 5M → ~39.5M total / ~11M active |

Loss-Free MoE (DeepSeek-V2 / IDEA-onemall-4):
```python
#register_buffer: does not participate in gradients, does not participate in optimizer, and is saved with checkpoint
self.register_buffer('expert_bias', torch.zeros(n_experts))

# Forward: bias injected into router logits
router_logits = self.router(x_flat) + self.expert_bias
router_probs = F.softmax(router_logits, dim=-1)

# Training: Statistical frequency, dynamically adjust bias to uniform distribution
freq = expert_mask.sum(dim=1).mean(dim=0) # (n_experts,)
self.expert_bias.add_(-bias_lr * (freq - 1.0 / self.n_experts))
```

Key design:
- `register_buffer` instead of `nn.Parameter` — bias does not participate in gradient return and does not interfere with the main task loss
- Do not return aux_loss - Completely eliminate the gradient conflict problem of Switch Transformer
- Compatible with DDP `broadcast_object_list`, buffer is automatically synchronized with state_dict

### 1b — Packed Sequence Training (`ntp/train.py`)

**Source**: IDEA-mtgr-0 (Meituan MTGR, CIKM 2025)

Old scheme (sliding window):
```
User [A, B, C, D, E] → cut out: [A,B,C]→D, [B,C,D]→E, ...
Problem: Python for-loop over ~100M interactions → ~45M samples, ~40-60 GB RAM
```

New solution (Packed sequences + causal mask):
```
User [A, B, C, D, E] → a sequence: [tokA, tokB, tokC, tokD, tokE]
Predict the next token for each position (standard LM training)
Causal masking ensures each location only sees the past
```

| Comparison | Sliding Window | Packed |
|------|----------|--------|
| Number of samples | ~45M | ~2M (1 per user) |
| Memory | ~40-60 GB | ~3-4 GB |
| Data construction | Python for-loop (slow) | numpy vectorized (fast) |
| Training signal | Each bar has only 3 target tokens | Each position generates Gradient |

Data construction optimization:
```python
# Vectorized user grouping (replaces defaultdict for-loop)
sort_idx = np.lexsort((ts_f, uids_f)) # numpy sorting
boundaries = np.where(uids_s[1:] != uids_s[:-1])[0] + 1 # Boundary detection
# pandas isin() alternative set membership check
iid_mask = pd.Index(iids_f).isin(valid_iids)
```

Right-padding + causal mask:
```python
# Variable length sequence batch: right-pad to max length
# Causal attention naturally prevents real tokens attend to padding (at the end)
# Loss mask excludes padding position
target_mask = arange < (lengths.unsqueeze(1) - 1)
loss = model(input_tokens, packed_targets=target_tokens, packed_mask=target_mask)
```

### 1c — Full-History Eval Context (`ntp/train.py` + `ntp/eval.py`)

Old scheme:
```
Training: The model looks at the complete user history (up to 512 tokens)
Evaluation: Only feed the latest 10 items (30 tokens) to do beam search
Problem: train-eval mismatch, wasting the long-distance dependency capability learned by the model
```

New solution:
```
Evaluation: User's complete behavioral history as context (up to max_seq_len tokens)
Match training distribution and make full use of long-distance information
```

Variable length eval pipeline:
- `EvalSequenceDataset` + `eval_collate_fn`: variable length input right-pad, fixed target stack
- Teacher-forced loss: forward each sample independently (avoid padding between context and target)
- Beam search: independent constrained beam search after each sample strip padding

### 1d — Trie-Constrained Beam Search (`ntp/model.py` + `ntp/eval.py`)

**Problem Analysis**:

Old beam search (unconstrained):
```
SID space: 4096^3 ≈ 68B possible combinations
Actual item: ~1M (corresponding to ~500K unique SIDs)
beam_size=500: Most of the beams hit non-existent SIDs → empty cannons, a waste of capacity

Result: Only 10-50 of 500 beams may be mapped to real items
```

Trie-constrained beam search:
```
Build SID prefix trie → only retain tokens present in the trie at each step
beam_size=500: Each beam is guaranteed to hit the real SID → zero waste

Result: 500 beams are all mapped to real items, and recall is greatly improved.
```

`SIDTrie` data structure:
```python
class SIDTrie:
    # children[layer] = {prefix_tuple → set of valid next tokens}
    # Example: children[0] = {() → {0, 1, 2, ...}} # layer 0: all valid first tokens
    # children[1] = {(0,) → {10, 11}, (1,) → {20}} # layer 1: Group by first token
    # children[2] = {(0,10) → {100,101}, ...} # layer 2: Group by the first two tokens
```

Constrained beam search core:
```python
for step in range(n_layers):
    logits = model.forward(input_exp, gen_exp)
    log_probs = F.log_softmax(logits, dim=-1)

# Build trie mask: group query by prefix to reduce redundancy dict lookup
    for each beam:
        valid_tokens = trie.valid_tokens(step, beam_prefix)
        mask[beam_idx, valid_tokens] = True

# Invalid token → -inf, topk naturally only selects valid candidates
    log_probs.masked_fill_(~mask, float('-inf'))
    topk_scores, topk_idx = flat_scores.topk(beam_size)
```

Item retrieval:
```python
# Old: Traverse 500 beams, many sid_to_items.get(sid) returns empty
# New: Each beam is guaranteed to exist → every query must be successful, and the candidate list is filled in order of score
candidates = []
for beam in sorted_beams:
    for item in sid_to_items[beam.sid]:
        candidates.append(item) # Sort by beam score, items with high-scoring SIDs are given priority
```

### Acceptance indicators

| Metric | NTPProbe (baseline) | NTPModel (expected) |
|------|--------------------|--------------------|
| PPL | baseline | Decline > 30% |
| Recall@50 | baseline | Significant improvement (trie constraint) |
| Recall@500 | baseline | Significant improvement (full beam effective) |
| Expert Utilization | N/A | Uniform (loss-free bias) |
| Eval context | 30 tokens | up to 512 tokens |
| Beam effective | ~10-20% | 100% |

### File list

| File | Change |
|------|------|
| `ntp/model.py` | ExpertFFN, SparseMoEBlock, TransformerLayer, NTPModel, SIDTrie, constrained_beam_search |
| `ntp/baseline.py` | unchanged (NTPProbe remains backwards compatible) |
| `ntp/train.py` | build_packed_sequences, PackedSequenceDataset, train_packed, EvalSequenceDataset, eval_collate_fn |
| `ntp/eval.py` | varlen eval path, SIDTrie build, constrained_beam_search call |
| `ntp/__init__.py` | export SIDTrie, constrained_beam_search |

**Risk**: Low. Each component is independently testable and backward compatible.

---

## Stage 2: Soft Prompt — User Representation Injection

**Goal**: Verify the value of user representations with minimal architectural changes

**Source**: IDEA-glide-0 (Spotify: Unusual Listening +5.4%, New Discovery +14.3%)

**Change**: `ntp/model.py` added ~50 lines

```
Current: [sid(item_1), sid(item_2), ..., sid(item_10)] → Decoder → next_sid
                                                        (30 tokens)

Stage 2: [prefix_1, ..., prefix_n, sid(item_1), ..., sid(item_10)] → Decoder → next_sid
           ↑
           user behavior embeddings → AttentionPooling → MLP → n prefix tokens
```

**design**:

| Components | Solutions |
|------|------|
| User embedding Source | Content embedding of user’s recent behavior item (Qwen3-0.6B) |
| Pooling | Attention-weighted pooling (learnable query) |
| Projection | MLP(pooled_dim → embed_dim × n_prefix) → reshape |
| n_prefix | sweep {2, 4, 8} |

**Training Strategy**:
1. Phase A: Freeze the decoder and only train prefix projection (fast convergence)
2. Phase B: full joint fine-tune

**Acceptance**:
- Recall@K comparison with/without soft prompt
- divided into cold user (< 5 interactions) / warm user (> 20) analysis
- If the soft prompt is significantly improved → the user indicates that the core is missing, and the Stage 3 priority is increased.
- If the improvement is limited → the bottleneck is on the decoding side or tokenizer side, consider doing Stage 5 first

**Risk**: Low. Do not change the decoder structure, just add prefix before input.

**Open questions**:
- [ ] Is user content embedding pre-calculated for cache? Or is it calculated online?
- [ ] Do prefix tokens share positional embedding or are independent?

---

## Stage 3: Encoder-Decoder separation

**Goal**: Decouple user behavior encoding and SID generation, support multi-scale behavior modeling + inference acceleration

**Source**: OneRec encoder-decoder + ARCHITECTURE.md Context Processor + IDEA-gr4ad-1 (LazyAR)

**New file**: `ntp/encoder.py`

### 3a — Lazy Decoder-Only (lightweight version, recommended to do first)

Reference OneRec-V2 "Lazy Decoder-Only" + LazyAR:

```
The 6 layers of the same Transformer are divided into two sections:

First 4 layers (Context Processing):
  - Bidirectional attention (non-causal)
  - Process user behavior sequences (30 tokens)
  - Output static KV pairs
  - Beam search is only counted once and is shared by all beams.

     ──── Fusion Layer (gated projection) ────

Next 2 layers (SID Generation):
  - One-way attention (causal)
  - Only handle [BOS] + 3 SID target tokens
  - Cross-attend to KV pairs of first 4 layers
  - beam search expand here
```

| Advantages | Description |
|------|------|
| Inference acceleration | When beam=500, the first 4 layers do not grow with beam, only the last 2 layers grow linearly |
| Information interaction | The first 4 layers of bidirectional attention encode user behavior better than pure causality |
| Simple implementation | No independent encoder required, the same set of Transformer Parameter |

Fusion Mechanism (Layer 4 → Layer 5):
```python
# m: non-AR representation, s: previous token embedding
Fuse(m, s) = W_f[m * sigmoid(W_g @ s); s]
```

### 3b — Complete Encoder-Decoder (after 3a validation)

```
┌──────────────────────────────┐
│       Context Encoder        │
│    (N layers, bidirectional)  │
│                              │
│  short_term  (20 items)  ────┤
│  positive_fb (N items)   ────┤──→ Z_enc ∈ ℝ^{T_enc × d_model}
│  [user_static (optional)]────┤       │
└──────────────────────────────┘       │
                                       │ keys, values
                                       ↓
┌──────────────────────────────┐       │
│        SID Decoder           │       │
│    (M layers, causal)        │       │
│                              │       │
│ Each layer: │ │
│    1. causal self-attention   │       │
│    2. cross-attention ◄───────┼───────┘
│    3. MoE FFN (SwiGLU)       │
│                              │
│  [BOS] → sid_1 → sid_2 → sid_3│
└──────────────────────────────┘
```

| Components | Config |
|------|------|
| Encoder layers | 4 (bidirectional, dense FFN) |
| Decoder layers | 4 (causal self-attn + cross-attn + MoE FFN) |
| Multi-behavior channels | short-term / positive-feedback are embedded separately and then spliced |
| Encoder Output | Cache during inference, beam search is only expanded in decoder |

**Acceptance**:
- Compare Stage 2 Recall@K
- Inference latency: speedup ratio of 3a vs Stage 1 when beam=500
- encoder representation quality: probe analysis (linear probe predicts user interest categories)

**Risk**: Medium. The architecture changes are large and training stability needs to be carefully debugged.

**Open questions**:
- [ ] Which should be done first, 3a or 3b? 3a is more concise but 3b is more general
- [ ] Do Encoder and Decoder share embedding?
- [ ] Multi-behavior channel: Does the current data have positive_feedback independent annotation?

---

## Stage 4: Long sequence compression — Query-Former

**Goal**: Support 200+ behavioral sequence inputs and control the amount of calculations

**Source**: IDEA-onemall-1 (OneMall: 1205→160 tokens, 3.7x FLOP reduction) + OneRec lifelong pathway

**Prerequisite**: Stage 3 completed (encoder can accept variable-length input)

**New**: `ntp/query_former.py` (reusable `model/qformer.py`)

```
User behavior sequence (variable length, up to 500+)
       │
       ▼
┌─────────────────────┐
│    Query-Former      │
│                      │
│  Q: M learnable      │
│     query tokens     │
│ KV: behavioral sequence embed │
│                      │
│  N layers cross-attn │
└─────────────────────┘
       │
       ▼
M compressed tokens (fixed length)
       │
       ▼
  concat with short-term tokens → Encoder
```

| Parameter | Search scope |
|------|----------|
| M (query tokens) | {4, 8, 16} |
| QFormer layers | {1, 2} |
| Input sequence length | {50, 100, 200, 500} |

**Layering Strategy** (refer to OneRec + GEMs):

| Time scale | Processing method | Token number |
|----------|----------|----------|
| Short-term (≤20 items) | Direct input, no compression | 20 × 3 = 60 |
| Mid-term (20-200 items) | Query-Former compression | M = 8-16 |
| Lifelong (200+ items) | Forward: hierarchical K-means + QFormer | Forward |

**Acceptance**:
- Fixed FLOP budget: Recall@K with different sequence lengths
- Compression ratio vs performance: trade-off curve of M=4/8/16
- Compare to baseline: truncate directly to 20 items (current plan)

**Risk**: Medium to low. QFormer is a mature component.

**Open questions**:
- [ ] How long is the average behavior sequence of the current user? If < 50, this stage will have limited benefits.
- [ ] Multiple behavior types (click/buy/exposure) each have a QFormer or are they shared?
- [ ] Does QFormer need to be pre-trained separately?

---

## Stage 5: Enhanced decoding (choose one of two)

**Goal**: Improve SID generation quality and reduce beam search space

**Prerequisite**: Stage 3 completed. Decide which one to choose based on Stage 3 error analysis.

### Option A — Stepwise Reasoning Tokens (IDEA-s2gr-0)

```
Original: [BOS] → sid_L1 → sid_L2 → sid_L3 (4 tokens)
Change to: [BOS] → [THINK] → sid_L1 → [THINK] → sid_L2 → [THINK] → sid_L3
                     ↑                      ↑                      ↑
                  contrastive            contrastive            contrastive
(align cluster (align cluster (align cluster
distribution) distribution) distribution)
```

- Think token uses contrastive loss to align codebook cluster distribution
- `L_total = L_SID + alpha * L_think`
- Sequence 4 → 7 tokens, the number of decoding steps is doubled, but each step is more accurate
- **Suitable for scenarios**: error analysis shows early token errors propagated to subsequent tokens

### Option B — Chain-of-Attribute Prefix (IDEA-unirec-0)

```
Original: [BOS] → sid_L1 → sid_L2 → sid_L3 (4 tokens)
Change to: [BOS] → cat_tok → brand_tok → sid_L1 → sid_L2 → sid_L3
                     ↑          ↑
attribute token attribute token
(category) (brand)
```

- Bayesian guarantee: `H(s_k | s_{<k}, a) < H(s_k | s_{<k})` — Attribute prefixes reduce conditional entropy
- Online results: HR@50 +22.6%, high value orders +15.5%
- **Suitable for scenarios**: item has structured attributes (category/brand/seller)
- **requires**: attribute data + attribute tokenizer

### Decision basis

| Conditions | Selection |
|------|------|
| Early token error rate is high, attribute data is unavailable | Choose A (Reasoning Tokens) |
| Attribute data is available, beam search space is too large | Select B (CoA Prefix) |
| Both are possible | Choose B (theoretical guarantee is stronger, the online effect is better) |

**Acceptance**:
- Recall@K improvement
- Beam search efficiency: Proportion of valid candidates (how many of the top-500 are legitimate items)
- Layer-by-layer accuracy analysis (corresponding to prefix depth hit@10 of eval.py)

---

## Stage 6: RL alignment + production-level optimization (future)

**Goal**: From "accurate prediction" to "good recommendation"

**Prerequisite**: At least 1-3 of Stage 1-5 are completed and the architecture is converged

**Source**: OneRec ECPO + IDEA-oxygen-0

| Components | Solutions |
|------|------|
| Reward Model | Multi-tower P-Score (ctr/lvtr/ltr/vtr towers + aggregation) |
| SFT | RSFT: filter bottom 50% sessions (by play duration), supervise fine-tuning |
| RL | ECPO (Early Clipped GRPO): group_size = 4× beam |
| Reasoning | Beam expanded to Pass@512 |
| Multiple scenarios | SA-GCPO (forward, current single scenario) |

**No rush for this stage**: OneRec paper and GenRank (IDEA-genrank-0) both prove **Architecture > Training Paradigm**. Get the architecture right first and then do RL.

---

## Overview

```
Stage 1 ✅       Stage 2          Stage 3a         Stage 3b          Stage 4          Stage 5          Stage 6
S-tier           Soft             Lazy             Full              Query-           Reasoning        RL
Decoder Prompt Dec-Only Enc-Dec Former / CoA Alignment
+ MoE                                                                                Prefix
+ Packed Train
+ Full-History
  Eval
+ Trie Beam
   │                │                │                │                │                │                │
   ▼                ▼                ▼                ▼                ▼                ▼                ▼
Strong baseline Verified users Inference acceleration Multi-scale 500+ sequences Decoding quality Online
39.5M represents value beam shared KV behavior modeling FLOP↓3-4x accuracy↑ indicator
 packed+trie
```

**Key decision points**:
- Stage 2 results determine Stage 3 priority
- Stage 3a vs 3b: If 3a is good enough, you can skip 3b
- Stage 5 A vs B: Depends on error analysis and data availability
- Stage 4 can be run in parallel with Stage 5

---

## IDEA not included in the current path but worthy of attention

| IDEA | Reasons | When to consider |
|------|------|----------|
| IDEA-llada-0 (Discrete Diffusion) | New decoding paradigm, high engineering complexity | If AR reaches a bottleneck after Stage 5 |
| IDEA-oxygen-0 (Fast-Slow Thinking) | Requires LLM reasoning, currently too complex | After Stage 6 |
| IDEA-gr2-0 (LLM Reranker) | Belongs to reranking, not retrieval | When there is a need for independent reranking |
| IDEA-higr-0 (Hierarchical Slate) | Belongs to reranking, 5x inference acceleration | slate recommended scenarios |
| IDEA-hpgr-0 (Session-MIM) | Requires session splitting + two PhaseTraining | When the sequence is long enough |
| IDEA-gti-0 (Grounded Token Init) | For LLM vocab extension | When taking the LLM CPT route |
