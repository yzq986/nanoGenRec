# NTP Side Information & Feature Engineering

[English](ntp-features.md) | [Chinese](ntp-features.zh.md)

The current NTP model input** only has SID token sequence + positional embedding**, without any side information. This document collects all optimization directions for "injecting additional features into NTP".

**Scope of influence**: `ntp/preprocess.py`, `ntp/train.py`, `ntp/model.py`

---

## Evolution path

```
Pure SID token + position embedding (current baseline, EXP-016)
├── IDEA-feat-0: Time Gap Embedding (time interval between adjacent items)
│ └── Distinguish "continuous brushing" vs "come back every other day", the interest in learning wanes
├── IDEA-feat-1: Action Type Embedding (behavior intensity signal)
│ └── Distinguish between click/like/fav/share, strong interaction > weak interaction
├── IDEA-feat-2: Segment Embedding decoupling (item_pos + layer_pos)
│ └── The current position encodes two dimensions at the same time, and they can be learned independently after decoupling.
├── IDEA-feat-3: Item Category Token (category side information)
│ └── Help cold-start + cross-category generalization
├── IDEA-feat-4: User Profile Prefix Token (global user portrait)
│ └── Short sequence cold start prior
└── IDEA-feat-5: Relative Time Encoding (continuous time → RoPE encoding)
└── More sophisticated time modeling than bucketed gap
```

---

## IDEA-feat-0: Time Gap Embedding (time interval feature)

**Priority**: ~~P0~~ → ✅ Verified and valid
**Source**: Original + SASRec-T (CIKM 2020), TiSASRec (WSDM 2020)
**Status**: ✅ Implemented and verified — EXP-036 clean ablation: +3.7pp R@500 (B vs A)

### Core Idea

Currently NTP only uses chronological order, losing time density information. The time interval between adjacent items `Δt = ts[i] - ts[i-1]` contains key signals:
- Δt < 1 min → Continuous brushing, the same interest continues within the session
- Δt ~ 30 min → session switching
- Δt > 1 day → interest may have waned/shifted

### Implementation plan

**Preprocess side** (`ntp/preprocess.py`):
- Calculate Δt (seconds) of adjacent items
- Bucketing: Logarithmic bucketing [0, 1min, 5min, 30min, 1h, 6h, 1d, 3d, 7d, 14d+] → ~10-16 buckets
- The 3 SID tokens of each item share the same time_gap bucket
- The gap of the first item in the sequence is set to a special token (BOS)

**Model side** (`ntp/model.py`):
```python
self.time_gap_emb = nn.Embedding(n_time_buckets, embed_dim)

# forward in:
x = token_emb + pos_emb + time_gap_emb[time_gaps]
```

### Change files

1. `ntp/preprocess.py` — `save_shard()` adds `time_gaps` array
2. `ntp/train.py` — `build_unified_sequences()` calculates Δt bucketing, `NTPDataset.__getitem__()` returns time_gaps
3. `ntp/model.py` — `NTPModel.__init__()` plus `time_gap_emb`, `forward()` plus

### Evaluate

- Baseline: EXP-016 14d-S (R@500=58.5%)
- Indicator: Recall@{10,50,100,500}, pay special attention to the prediction accuracy of long-interval items
- Ablation: different bucketing granularity (8 vs 16 vs 32 buckets)

### Key questions

1. Is the accuracy of `first_ts` in behavioral data second-level or day-level? Need to confirm
2. Design of bucket boundaries: logarithmic vs linear vs learned boundaries
3. Do the three SID tokens of the same item share the same time_gap, or are they only injected at the s0 position?

---

## IDEA-feat-1: Action Type Embedding (behavior type feature)

**Priority**: ~~P0~~ → ✅ Verified and valid
**Source**: Original + HSTU1B (KDD 2026), OneLoc Multi-behavior, GR4AD Value-Aware
**Status**: ✅ Implemented and verified — EXP-036 includes `--use_action_level`, Config B vs A: +3.7pp R@500

### Core Idea

Currently `action_bitmap > 0` only performs positive and negative filtering, discarding the behavior intensity information. Different bits in the bitmap represent different behaviors (click, like, fav, share, comment, purchase), and these behaviors express different intensity of interest.

### Implementation plan

**Option A — Action Level (Discrete Strength)**:
```python
# Map action_bitmap to action_level:
# click=1, like=2, fav=3, share=4, purchase=5
action_level = compute_action_level(action_bitmap)
self.action_emb = nn.Embedding(n_action_levels, embed_dim)
```
- Add to the s0 position of each item (or add all 3 tokens)

**Option B — Multi-hot Action Embedding**:
```python
# Retain the multi-hot nature of bitmap:
self.action_proj = nn.Linear(n_action_bits, embed_dim)
action_vec = bitmap_to_multihot(action_bitmap) # (n_bits,)
action_emb = self.action_proj(action_vec)
```

### Relation to IDEA-oneloc-5 (Multi-behavior)

- oneloc-5 is to **separate** sequences of different behaviors (click seq, buy seq, ...)
- This IDEA marks behavior types in a **unified sequence** and maintains chronological order
- The two are orthogonal: this IDEA is more lightweight, oneloc-5 is more thorough

### Change files

1. `ntp/preprocess.py` — `save_shard()` adds `action_levels` array
2. `ntp/train.py` — `_build_user_items()` retain action_bitmap, calculate action_level
3. `ntp/model.py` — `NTPModel` plus `action_emb`

### Evaluate

- Comparison: no action_emb vs action_level vs multi-hot
- Analysis: Whether the item prediction accuracy of high-value behaviors (purchase/fav) is improved

### Key questions

1. The meaning of each bit of action_bitmap in the current data needs to be confirmed (grep `_VIEW_EXIT_BIT`)
2. Whether the behavior distribution is seriously unbalanced (click >> purchase)
3. Will the model overfit the action type and ignore the sequence pattern?

---

## IDEA-feat-2: Segment Embedding decoupling (Item Position + Layer Position)

**Priority**: ~~P0~~ → ✅ Verified and valid
**Source**: Original; segment embedding design similar to BERT
**Status**: ✅ Implemented and verified — EXP-036 includes `--use_segment_emb`, Config B vs A: +3.7pp R@500

### Core Idea

Currently `pos_emb[i]` encodes two pieces of information at the same time:
- "This is the item number" (item_pos = i // n_sid_layers)
- "This is the layer SID in item" (layer_pos = i % n_sid_layers)

The two dimensions are coupled in the same embedding, and the model needs to learn disentangle by itself. After decoupling, each dimension learns independently:

```python
self.item_pos_emb = nn.Embedding(max_items, embed_dim)
self.layer_pos_emb = nn.Embedding(n_sid_layers, embed_dim) # Only 3

# forward:
item_pos = positions // n_sid_layers
layer_pos = positions % n_sid_layers
x = token_emb + item_pos_emb[item_pos] + layer_pos_emb[layer_pos]
```

### Revenue

- `item_pos_emb` can learn pure "sequence position → interest decay" signal
- `layer_pos_emb` can learn the structure of "L0 is coarse-grained categories, L1 is fine-grained, and L2 is disambiguation"
- Fewer parameters: `max_items + 3` vs `max_items * 3`

### Change files

1. `ntp/model.py` — replace `self.pos_emb` with `self.item_pos_emb` + `self.layer_pos_emb`

### Evaluate

- Direct replacement, compare loss and Recall@K
- Visualization: whether item_pos_emb shows a decay trend, whether layer_pos_emb L0/L1/L2 are obviously different

### Key questions

1. Current pos_emb parameter amount = max_seq_len * embed_dim; after decoupling = (max_items + 3) * embed_dim, less
2. Will decoupling lose position-layer interaction information? Available `item_pos_emb + layer_pos_emb + interaction_emb` reserved

---

## IDEA-feat-3: Item Category / Attribute Token

**Priority**: ~~P1~~ → P4 Suspended
**Source**: OneLoc §Category Prompt, UniRec Chain-of-Attribute, GeoGR
**Status**: On hold - Category information has been implicitly encoded through Qwen3 text embedding; EXP-036 full-features (time_gap+action_level+segment) has reached R@500=59.0%, and the marginal gain of category tokens is expected to be limited. Need to complete EXP-037→039 RL link before evaluation

### Core Idea

SID is semantic compression (embedding → discrete code), but it loses structured signals such as category/author. Supplementing category tokens can help the model understand cross-category generalization and cold-start items.

### Plan comparison

**Option A — Category Token Prefix** (change the sequence structure):
```
Sequence: [cat_1, s0_1, s1_1, s2_1, cat_2, s0_2, s1_2, s2_2, ...]
```
- Insert a category token before each item
- Sequence length +33%
- The model can predict s0 from cat token (coarse→fine)

**Option B — Category Embedding addition** (does not change the sequence length):
```python
self.cat_emb = nn.Embedding(n_categories, embed_dim)
# Add category embedding only at s0 position (or all 3 positions)
x[s0_positions] += cat_emb[category_ids]
```
- The sequence length remains unchanged
- But category information may be overwhelmed by token embedding

**Option C — Category as Soft Prompt** (refer to IDEA-glide-0):
```
Sequence: [soft_prompt(category), s0_1, s1_1, s2_1, ...]
```
- Use learned soft prompt to encode category, as sequence prefix

### Relationship with IDEA-unirec-0 (Chain-of-Attribute)

- UniRec lets the model **generate** category → brand → SID in sequence (chain-of-thought formula)
- This IDEA only injects category as **input feature** and does not change the generation target.
- UniRec is more thorough but has larger changes (needs to change the generation target and decoding strategy)

### Change files

1. New data: item → category mapping table
2. `ntp/preprocess.py` — Load category mapping and generate category sequence
3. `ntp/model.py` — add `cat_emb`, inject in forward

### Key questions

1. **Data Availability**: Is there category information in the current data? Need to get from original item metadata
2. Category granularity: first-level category (~20) vs second-level category (~200) vs third-level category (~2000)
3. Option A increases sequence length → increases training cost; Option B has zero cost but the effect may be weak

---

## IDEA-feat-4: User Profile Prefix Token (user portrait token)

**Priority**: ~~P1~~ → P4 Suspended
**Source**: MTGR Dynamic Masking, HPGR Preference Attention
**Status**: On hold — Align³GR ablation shows user characteristics +0.9pp, with limited benefits; the short sequence cold start scenario has limited a priori value, and the NTP sequence itself has implicitly modeled user preferences. RL aligned link (EXP-037→039) has higher priority

### Core Idea

Construct a "user summary" token using user history statistics and place it at the front of the sequence to provide a global prior to the model:
- Top-3 L0 frequency distribution (main interest categories)
- Activity bucket (daily activity/weekly activity/monthly activity)
- Average action intensity (highly interactive users vs browsing users)

### Implementation plan

```python
# Preprocess: Calculate user-level statistics
user_profile = {
    'top_l0': [most_frequent_l0_clusters],  # top-3
    'activity_bucket': activity_level,       # 0-7
    'avg_action_level': mean_action_level,   # 0-4
}

# Model:
self.profile_emb = nn.Sequential(
    nn.Linear(profile_dim, embed_dim),
    nn.GELU(),
    nn.Linear(embed_dim, embed_dim)
)

# Forward: as the first token in the sequence
profile_token = self.profile_emb(user_profile_features)
x = torch.cat([profile_token.unsqueeze(1), token_sequence], dim=1)
```

### Revenue

- Short sequence users: profile token provides a priori to make up for the lack of history
- Conditional generation: the model can adjust the prediction distribution according to user type
- Consistent with the instruction prefix direction of IDEA-sigma-0 (instruction multitasking)

### Key questions

1. Profile calculation needs to be based on data **before** the training window to avoid information leakage
2. If the sequence is long enough (>50 items), the marginal benefit of profile token may be small
3. There is some information overlap with IDEA-feat-0 (time gap) and IDEA-feat-1 (action type)

---

## IDEA-feat-5: Relative / Continuous Time Encoding

**Priority**: P2 → **Upgrade to P1**
**Source**: Time-LLM (ICLR 2024), TiSASRec (WSDM 2020), **TO-RoPE (arxiv 2510.20455, Roblox)**, **SynerGen (arxiv 2509.21777)**
**Status**: feat-0 has been verified to be valid (EXP-025 R@500=63.6%), ready to enter the TO-RoPE experiment

### Core Idea

IDEA-feat-0 encodes time intervals using discrete buckets, losing precise time information. More advanced options:

**Option A — Time-and-Order RoPE (TO-RoPE)** [New — Highly Recommended]:
TO-RoPE extends RoPE to encode both discrete index and wall-clock time:
- `θ_k(i) = (1-λ_k) · α_p · i · ω_p_k + λ_k · α_t · τ_i · ω_t_k`
- Three instantiations:
  - **Split-by-dim** (recommended): use index for part of the rotary plane and time for the other part. No intra-plane interference, explicit capacity allocation (e.g. 70% time / 30% index)
  - **Split-by-head**: Part of the head uses index-only, part of the head uses time-only. fusion between heads through output projection
  - **Early fusion**: index and time angle are added directly. Most flexible but risks disruptive interference
- Roblox large-scale industrial data: split-by-dim/head is consistently better than APE, HSTU-style relative bias, index-only RoPE
- Optimal split ratio: 0.3-0.5 (time accounts for 30-50% capacity)
- Naturally captures three temporal patterns: short-term burstiness, long-term rhythms, temporal periodicity (day-of-week/time-of-day)
- Compatible with FlashAttention, almost no additional parameters

**Option B — Continuous Time Kernel**:
- `time_emb = MLP(log(Δt))` → continuous mapping, no need for bucketing
- Higher accuracy but requires more parameters

**Option C — BOS Time Injection (OneLive)**:
- OneLive (Kuaishou, arxiv 2602.08612): Inject multi-granular temporal features into [BOS] token
- `x_BOS = x_BOS + MLP(Concat(x_Hour, x_Day, x_Week))`
- Only inject the global time at the beginning of the sequence without changing the position encoding
- Lightweight, but only a global signal of "current moment", does not encode the time relationship of each item in the sequence

### Relationship with IDEA-feat-0

- feat-0 (bucketing) is P0, verified to be effective — EXP-025 beam passes: R@500 61.2% → 63.6%
- feat-5 (TO-RoPE) is P1, **replace absolute pos_emb + time_gap_emb with unified TO-RoPE**
- Key advantage: TO-RoPE encodes order + time simultaneously, no need for separate time_gap bucketing feature

### Adaptation to us

The current model uses absolute learnable position embedding + time_gap bucket embedding (additive). Switch to TO-RoPE:
1. Remove `pos_emb` and `time_gap_emb`
2. Each item is associated with a normalized timestamp `τ_i = (ts_i - ts_0) / scale`
3. Three SID tokens of the same item share the same `τ_i`, but the index is `3*item_idx + layer`
4. Split-by-dim: e.g. 256d → 128d for index planes + 128d for time planes
5. When beam search is generated: index increases naturally, τ uses the real timestamp of target item

**NOTE**: This solves the core issue in EXP-023/024/025 - time_gap_emb as an additive feature in the beam search incremental path needs to be passed manually. TO-RoPE encodes time information into Q/K rotation, and KV cache naturally retains time information.

### Key questions

1. ~~Time-aware RoPE needs to replace the current absolute position embedding → Big changes~~ The TO-RoPE paper has verified the feasibility
2. Can the MLP of Continuous time kernel learn log-scale time mode well?
3. The numerical range of timestamp in the sequence: 14 days ≈ 1.2M seconds, needs normalization → TO-RoPE recommends using days or hours as the τ unit
4. The current model uses `nn.MultiheadAttention`, and you need to confirm whether you can easily inject custom RoPE (you may need to switch to handwritten attention or xformers)

---

## Priority summary

| Priority | ID | Feature | Benefit | Implementation Cost | Recommendation |
|--------|-----|------|------|---------|------|
| ~~P0~~ → ✅ | IDEA-feat-0 | Time Gap Embedding | ★★★ | Low | ✅ EXP-036 Verified valid, +3.7pp R@500 |
| ~~P0~~ → ✅ | IDEA-feat-1 | Action Type Embedding | ★★☆ | Low | ✅ EXP-036 included, three-in-one joint verification |
| ~~P0~~ → ✅ | IDEA-feat-2 | Segment Embedding decoupling | ★★ | Extremely Low | ✅ EXP-036 included, three-in-one joint verification |
| P4 on hold | IDEA-feat-3 | Item Category Token | ★★ | Medium | Category information has been implicit in text embedding, marginal revenue is limited |
| P4 on hold | IDEA-feat-4 | User Profile Prefix | ★☆ | Medium | Align³GR ablation only +0.9pp, do RL link first |
| **P1** | **IDEA-feat-5** | **TO-RoPE Time+Order Encoding** | **★★★** | **Medium** | **feat-0/1/2 Verified, next step to upgrade to TO-RoPE** |

**Current Status**: feat-0/1/2 3-in-1 has been verified as valid by EXP-036 (+3.7pp R@500). Next step: Complete RL alignment link (EXP-037→039), then consider feat-5 TO-RoPE for further improvement.

---

## Relations with other ideas/

| This File | Association | Relationship Description |
|--------|------|---------|
| IDEA-feat-1 | IDEA-oneloc-5 (training.md) | oneloc-5 separates behavior sequences, feat-1 labels behaviors in a unified sequence Medium |
| IDEA-feat-1 | IDEA-gr4ad-2 (training.md) | gr4ad-2 weights loss by behavior, feat-1 lets Model learn behavioral representation |
| IDEA-feat-3 | IDEA-unirec-0 (architecture.md) | unirec-0 generates category, feat-3 only injects category |
| IDEA-feat-3 | IDEA-glide-0 (architecture.md) | glide-0 soft prompt, feat-3 scheme C similar |
| IDEA-feat-4 | IDEA-mtgr-0 (training.md) | mtgr-0 dynamic masking supports profile bidirectional attn |
| IDEA-feat-5 | IDEA-gems-0 (architecture.md) | gems-0 multi-stream temporal, feat-5 is a lightweight alternative |
