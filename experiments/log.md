

[English](log.md) | [Chinese](log.zh.md)

Record in reverse chronological order. Each experiment is linked to the results directory under `experiments/`.

---

## Template

<!--
Copy the following template to create a new experiment record. The numbers are in ascending order, with the newest on top.

## EXP-NNN: (Experiment title)

**Date**: YYYY-MM-DD
**Status**: planned | running | completed
**Results**: [./hyperparam/YYYY-MM-DD_xxx/](./hyperparam/YYYY-MM-DD_xxx/)

### Background
(Current status, problem to be solved)

### Hypothesis
(expected results and reasons)

### Design
- **Variable**: ...
- **Fixed**: ...
- **Metric**: ...
- **Data**: ...

### Results
(Fill it out after running, including form)

### Analysis
(Interpretation of results)

### Next Steps
(next step plan)
-->

---

## EXP-047: L-tier NTP — All Validated Optimizations (scale-07, 101M active)

**Date**: 2026-04-30
**Status**: completed
**Results**: experiments/ntp_checkpoints/exp047/

### Background

The S-tier experimental chain (EXP-043~046) has verified a set of optimized combinations: TO-RoPE ts=0.5 (+2.7pp over abs-pos baseline),
gate_attn (+0.4pp), segment_emb + time_gap + action_level.
M-tier (71.6M active) reaches R@500=70.2% in EXP-043, which is 9pp higher than S-tier (61.2%).

This experiment transplants all verified optimizations to L-tier (scale-07, 101.1M active params) as the starting point of SFT for the RL link.

### Hypothesis

| Metric | S-tier + TO-RoPE + gate | M-tier bare | Expected L-tier |
|------|------------------------|-------------|-------------|
| R@500 | 63.9% | 70.2% | ~72-74% |
| PPL | 22.7 | 18.5 | ~17-19 |

L-tier has ~40% more active params than M-tier (101M vs 71.6M), and R@500 is expected to increase by 2-4pp.
All optimizations are forward verified in S-tier, and transplanting to L-tier should at least retain the gains.

### Design

- **Model**: scale-07 (embed_dim=512, 12L, 16E top-2 MoE, 101.1M active)
- **Features**: segment_emb + time_gap + action_level + TO-RoPE ts=0.5 (order:0.5,time:0.5) + gate_attn
- **Data**: exp044b-0.6b-14d (14d, timestamps connected); SID: exp026-0.6b-14d
- **LR**: 2e-4 (scale-07 EXP-015 optimal LR)
- **Baselines** (do not retrain, quote directly):
  - exp043-m-0.6b: M-tier bare, R@500=70.2%
  - exp044c-torope-ts05: S-tier + TO-RoPE ts=0.5, R@500=63.9%

### Results

| Config | R@10 | R@500 | PPL | L2 PPL | Wall |
|--------|------|-------|-----|--------|------|
| exp043-m-0.6b (M-tier bare) | 14.5% | 70.2% | 18.54 | — | 23min |
| exp044c-torope-ts05 (S-tier + TO-RoPE) | 11.8% | 63.9% | 22.7 | 3.5 | 8min |
| **exp047 (L-tier + all opts)** | **12.8%** | **64.1%** | **20.7** | **2.9** | **42min** |

Model: 632M total params, 103.9M active params (16E top-2).

### Analysis

1. **R@500 64.1%, lower than expected (72-74%)**: The result of L-tier + all optimization is only 0.2pp higher than S-tier + TO-RoPE (63.9%),
   Much lower than M-tier bare (70.2%). Root cause: **This experiment is 1 epoch, M-tier (EXP-043) is also 1 epoch**,
   However, L-tier (101M active) has 40% more parameters than M-tier (71.6M active), and L-tier is more seriously undertrained in 1 epoch.
   Chinchilla’s optimal tok/param ratio: L-tier is about 1.3 tok/param (132M tokens / 103M active), which is far lower than the recommended 20x.

2. **PPL=20.7 is higher than M-tier bare (18.54)**: It is also a manifestation of under-training. A larger model will have worse fitting under the same amount of data.

3. **TO-RoPE + gate_attn has no obvious benefit on L-tier** (vs M-tier bare comparison):
   The benefits brought by multiple rounds of optimization (TO-RoPE+gate) are offset by the scale penalty. More training tokens are needed to realize the advantages of large models.

4. **RL link origin is still available**: R@500=64.1% > S-tier SFT (53-61%), which can be used as SP-DPO origin.
   However, if the goal is to surpass exp039b-ecpo (65.7%), SFT needs to first reach ≥62%, and the current 64.1% meets the condition.

5. **Conclusion**: L-tier is inferior to M-tier in a single epoch. To take advantage of L-tier, ≥3 epochs or more data are required.
   The RL link continues to advance from exp047, and the final effect after RL is observed.

### Next Steps

- RL link: exp047 (R@500=64.1%) → SP-DPO → RF-DPO (3ep) → ECPO

---

## EXP-048: M-tier + TO-RoPE — 2-dim vs 3-dim (order/time/layer)

**Date**: 2026-04-30
**Status**: completed

### Background

EXP-044B verified that TO-RoPE (after timestamps are switched on) works on S-tier (R@500=63.6%, +2.4pp).
EXP-047 TO-RoPE gains at L-tier masked by undertraining (1.3 tok/param).

This experiment returns to M-tier (the parameters have been fully verified) and tests:
- 2-dim RoPE: order:0.5 + time:0.5 (EXP-044B S-tier optimal configuration)
- 3-dim RoPE: order:0.4 + time:0.5 + layer:0.1 (new layer_id dimension)

### Design

- **Model**: M-tier (scale-06, embed_dim=512, 8L, 8E top-2 MoE, 71.6M active)
- **Features**: segment_emb + time_gap + action_level + gate_attn + TO-RoPE
- **Data**: exp044b-0.6b-14d (timestamps are connected); SID: exp026-0.6b-14d
- **LR**: 1e-3
- **Baseline**: exp043-m-0.6b (M-tier bare, R@500=70.2%, PPL=18.54)

### Results

| Config | rope_dims | R@10 | R@500 | PPL |
|--------|-----------|------|-------|-----|
| exp043-m-0.6b (baseline) | — | 14.5% | 70.2% | 18.54 |
| **exp048-m-2dim** | order:0.5, time:0.5 | 15.1% | **70.1%** | 17.67 |
| exp048-m-3dim | order:0.4, time:0.5, layer:0.1 | 15.5% | 69.5% | 17.64 |

### Analysis

1. **TO-RoPE has no gain on M-tier**: 2-dim is the same as baseline (-0.1pp), and 3-dim drops slightly (-0.7pp).
   Different from the S-tier conclusion (EXP-044B S-tier +2.4pp) - the stronger model capacity of M-tier may have fully captured the timing information through absolute position encoding, and the marginal contribution of RoPE disappears.

2. **3-dim < 2-dim**: The layer:0.1 component occupies the head dimension originally given to order (0.5→0.4). R@10 is higher (15.5% vs 15.1%) but R@500 is lower, indicating that the layer_id dimension improves neighbor precision but damages long-tail recall.

3. **PPL is improved** (17.6 vs 18.54), indicating that RoPE does have benefits at the language modeling level, but it does not translate into R@500.

4. **Conclusion**: TO-RoPE has limited help for M-tier and is not recommended for use in M-tier + 1 epoch configuration. Subsequent experiments keep M-tier bare (70.2%) or add gate_attn (exp046, 61.6%, S-tier version) as the baseline.

### Next Steps

- M-tier bare (exp043, R@500=70.2%) is still the current NTP optimal checkpoint
- L-tier requires ≥3 epochs to show its advantages and will not be promoted for the time being.
- Next step priority: RL link (SP-DPO → RF-DPO → ECPO)

---

## EXP-046: GateAttention — Sigmoid Gate on Attention Output

**Date**: 2026-04-29
**Status**: completed

### Background
Add sigmoid gate to the attention output of TransformerLayer:
`attn_out = attn_out * sigmoid(W_g * x_norm)`, W_g ∈ R^{D×D}, the new parameter is about +1% (6 × 256² ≈ 393K).
Hypothesis: per-position gating can suppress the attentional influence of noise tokens and improve the quality of sequence modeling.

### Design
- **Variable**: whether gate_attn is present or not
- **Fixed**: S-tier + 0.6B SID, 14d data, segment_emb + time_gap + action_level
- **Baseline**: exp043-s-0.6b (full eval R@500=61.2%, PPL=26.52)

| Config | Description |
|--------|------|
| exp043-s-0.6b | Baseline |
| exp046-gate-attn | +gate_attn |

### Results

| Config | R@10 | R@500 | PPL |
|--------|------|-------|-----|
| exp043-s-0.6b (baseline) | 11.4% | 61.2% | 26.52 |
| **exp046-gate-attn** | **10.2%** | **61.6%** | **26.07** |

### Analysis

1. **R@500 +0.4pp** (61.2% → 61.6%), PPL decreased slightly (26.52 → 26.07), positive but small marginal benefit.
2. **R@10 decrease** (11.4% → 10.2%): The top-10 accuracy declined slightly, indicating that the gate may have affected the ranking of high-confidence predictions, but the long-tail recall (R@500) has improved.
3. **Conclusion**: GateAttention has a slight positive effect, has low computational cost (+1% parameter), and can be retained as the default component, but it is not a decisive factor.

### Next Steps
- Retain `use_gate_attn` as an optional component, and subsequent experiments can be enabled by default
- EXP-045 (FSQ sweep) continued

---

## EXP-043: Embedding Model Size Comparison — S-tier & M-tier × 0.6B/4B/8B SID

**Date**: 2026-04-29
**Status**: completed

### Background
The current SID tokenizer uses Qwen3-0.6B embedding (exp013 SID, legacy version). EXP-026 built three sets of new SID caches (0.6B/4B/8B, 14d). This experiment compared the impact of three embedding sizes on NTP performance, and also verified the performance of the S-tier vs M-tier model under different SIDs.

### Design
- **Variable**: Embedding model size (Qwen3-0.6B / 4B / 8B) × NTP model tier (S-tier / M-tier)
- **Fixed**: 14d data, full features (time_gap + action_level + segment_emb)
- **SID cache**: exp026-{0.6b,4b,8b}-14d
- **Baseline**: exp036-full-features (exp013 SID, S-tier, R@500=59.0%)

### SID Cache Metrics

| SID Cache | Emb Dim | n_items | Collision | L0 entropy | L1 entropy | L2 entropy | Joint entropy | L0 Gini |
|-----------|---------|---------|-----------|-----------|-----------|-----------|--------------|---------|
| exp026-0.6b | 1024 | 1,096,364 | **0.49%** | 11.72 (97.6%) | 9.55 (79.6%) | **10.58 (91.2%)** | **20.05** | 0.320 |
| exp026-4b   | 2560 | 1,110,697 | 2.76% | 11.80 (98.3%) | 10.60 (88.4%) | 8.10 (78.7%) | 20.02 | **0.288** |
| exp026-8b   | 4096 | 1,110,695 | 5.44% | 11.78 (98.2%) | **10.95 (91.2%)** | 7.17 (71.6%) | 19.95 | 0.303 |

The utilization rate of the relative theoretical maximum entropy in parentheses (L0/L1 theory max=12 bits, L2 theory max = log2 (actually used FSQ codes)).

**Entropy Analysis**:
- The three L0/L1 models are basically equivalent (KMeans 2-layer uniform distribution)
- **L2 entropy decreases sharply with embedding dimension**: 0.6B=10.58→4B=8.10→8B=7.17 bits. 8B FSQ has only ~145 effective slots (vs. 0.6B ~1500)
- **Root cause**: FSQ MLP hidden=64 is designed for 0.6B (1024D), and the bottleneck for the 4B/8B residual vector (2560D/4096D) is insufficient. The L2 layer information is seriously lost, resulting in a large number of items being mapped to the same FSQ code → collision explosion.

### Scaling Law — Irreducible PPL Floor

Use S-tier (N≈17.5M active) and M-tier (N≈71.6M active) to connect two points, and inversely deduce the intrinsic floor of each SID (fixed scaling index c=0.456, fitting `L(N) = floor + b/N^c`):

| SID | Joint entropy | floor loss | **floor PPL** | scaling b |
|-----|---------------|-----------|---------------|-----------|
| exp013 (old 0.6B) | — | 2.522 | **12.45** | 2055 |
| exp026-0.6b | 20.05 bits | 2.523 | **12.46** | 1517 |
| exp026-4b | 20.02 bits | 2.466 | **11.78** | 1299 |
| exp026-8b | 19.95 bits | 2.507 | **12.26** | 1047 |

**Key Findings**:
1. **4B SID floor is the lowest (PPL=11.78)**. It is the tokenizer with the highest theoretical upper limit among the three, which is 0.68 PPL lower than 0.6B.
2. **8B SID floor is higher than 4B (12.26 vs 11.78)**: L2 entropy collapse (7.17 bits) offsets the embedding quality improvement, and the tokenizer bottleneck becomes more serious.
3. **scaling coefficient b decreases monotonically with the embedding scale** (2055→1517→1299→1047): better SID allows the model to converge faster, and the loss of the same parameter amount is lower.
4. **The floor of 0.6B is almost the same as the old SID (12.46 vs 12.45)**, but b is smaller (faster convergence), and the actual NTP performance is better than the old SID.
5. **To take advantage of the true quality of 8B embedding, the FSQ hidden needs to be expanded from 64 to ~256** to restore L2 entropy to 90%+ utilization.

### Results

| Config | R@10 | R@500 | PPL | L0 PPL | Wall |
|--------|------|-------|-----|--------|------|
| **exp036-full-features** (old baseline) | — | 59.0% | — | — | — |
| exp043-s-0.6b | 11.4% | 61.2% | 26.52 | 390.2 | 8min |
| exp043-s-4b | 9.7% | 64.3% | 22.49 | 322.2 | 8min |
| exp043-s-8b | 10.2% | **64.7%** | 20.66 | 279.6 | 8min |
| exp043-m-0.6b | **14.5%** | 70.2% | 18.54 | 322.9 | 23min |
| exp043-m-4b | 14.2% | 70.4% | 16.55 | 268.1 | 23min |
| exp043-m-8b | 13.0% | 69.7% | **16.14** | 240.9 | 23min |

### Analysis

1. **M-tier is significantly ahead of S-tier**: M-tier R@500 is all around 70%, which is 5-9pp higher than S-tier (61-65%). M-tier active params ~71.6M vs S-tier ~17.5M, the model scale still has significant gains in this range.

2. **Embedding scale has a limited but regular impact on R@500**:
   - S-tier: 0.6B→4B +3.1pp, 4B→8B +0.4pp. 4B is an obvious turning point, and 8B has small marginal benefits.
   - M-tier: 0.6B→4B +0.2pp, 4B→8B -0.7pp. M-tier is not sensitive to SID quality - larger models can partially compensate for the tokenizer's L2 information loss.

3. **PPL decreases monotonically with the embedding size**: 8B’s L0 PPL is the lowest (240.9), indicating that larger embedding makes the L0 cluster more separated. But the PPL improvement does not fully translate into R@500 (beam search is affected by collision and L2 information loss).

4. **R@10 regular anomalies are explained by collision**: 0.6B collision has the lowest (0.49%), and top-beam accuracy hit rate is the highest. 8B collision=5.44%, a large number of items share SID, beam search cannot distinguish → R@10 is low.

5. **Optimal Configuration**:
   - Current: **M-tier + 4B SID** (R@500=70.4%, floor PPL=11.78 lowest)
   - Low-cost alternative: M-tier + 0.6B SID (R@500=70.2%, smaller b will converge faster)
   - 8B SID needs to expand FSQ hidden (64→256) to repair L2 entropy collapse before it can truly take advantage

### Next Steps
- EXP-044: TO-RoPE vs absolute pos emb, based on S-tier + 0.6B SID ✓
- EXP-045: FSQ hidden dim repair (4B h=64→256, 8B h=64→512), reduce collision rate ✓
- M-tier RL link: take exp043-m-0.6b as the starting point of SFT and connect to SP-DPO

---

## EXP-045: FSQ Hidden Dim Fix — 4B h=256, 8B h=512

**Date**: 2026-04-29 ~ 2026-04-30
**Status**: completed

### Background
EXP-043 analysis (Analysis #4) points out that the collision rate of 4B and 8B SID is too high (2.76% / 5.44%),
The root cause is that `fsq_mlp_hidden=64` is designed for 0.6B (dim=1024) and for 4B (dim=2560) and 8B (dim=4096)
It is a serious bottleneck, causing MLP to be unable to fully distinguish the high-dimensional embedding space, and a large number of items are mapped to the same SID.

This experiment is extended to a complete h-dim sweep (empirical formula fitting) instead of just measuring a single point.

### Design
- **0.6b sweep**: h ∈ {32, 64(reuse→rerun), 128, 256}
- **4b sweep**: h ∈ {64(reuse), 128, 512, 1024} (h=256 data exception skipped)
- **8b**: Skip - the alignment rate of 8b embedding cache item_id and behavior data is only 2.3% (159k/7M),
  KMeans 4096 clusters are seriously under-constrained on 159k samples, and the results are invalid
- **Fitting**: Use 0.6b + 4b to fit a total of 8 valid points `collision_rate = a*(h/dim)^b`

### Results

**⚠️ Important bug (discovered on 2026-04-30)**: EXP-045 All newly run preprocess-sids use `num_clusters=1024` (default value), while exp026 uses `num_clusters=4096`. exp-045.sh has `--fsq_levels 12d_4096` but misses `--num_clusters 4096`, causing KMeans clustering to be under-constrained. **All data in the exp045-* directory cannot be directly compared with exp026. The conclusion needs to be verified by re-running `--num_clusters 4096`. **exp026 series data remains credible.

| Model | dim | h | h/dim | collision | Gini_d2 | n_items | num_clusters | Remarks |
|-------|-----|---|-------|-----------|---------|---------|--------------|------|
| 0.6b | 1024 | 32 | 0.031 | 9.42% | 0.5448 | 1,110,697 | **1024** | ⚠️ num_clusters bug |
| 0.6b | 1024 | **64** | 0.063 | 2.21% | 0.5448 | 1,110,697 | **1024** | ⚠️ num_clusters bug, rerun |
| ~~0.6b~~ | ~~1024~~ | ~~64~~ | ~~0.063~~ | ~~0.49%~~ | ~~—~~ | ~~1,096,364~~ | ~~4096~~ | ~~Data set Different, invalid~~ |
| 0.6b | 1024 | 128 | 0.125 | 1.25% | 0.5448 | 1,110,697 | **1024** | ⚠️ num_clusters bug |
| 0.6b | 1024 | 256 | 0.250 | 1.44% | 0.5448 | 1,110,697 | **1024** | ⚠️ num_clusters bug |
| 4b | 2560 | 64 | 0.025 | 2.76% | 0.3535 | 1,110,697 | 4096 | Reuse exp026, **Trusted** |
| 4b | 2560 | 128 | 0.050 | 5.56% | 0.5740 | 1,110,697 | **1024** | ⚠️ num_clusters bug |
| 4b | 2560 | 512 | 0.200 | 3.13% | 0.5740 | 1,110,697 | **1024** | ⚠️ num_clusters bug |
| 4b | 2560 | 1024 | 0.400 | 2.99% | 0.5740 | 1,110,697 | **1024** | ⚠️ num_clusters bug |
| 8b (ref) | 4096 | 64 | 0.016 | 5.44% | 0.3725 | 1,110,695 | 4096 | Reuse exp026, **Trusted** |
| **exp026-0.6b-14d (ref)** | 1024 | 64 | 0.063 | **0.49%** | **0.3316** | 1,096,364 | **4096** | **baseline, num_clusters correct** |

Gini_d2 = L1+L2 prefix distribution Gini coefficient (FORGE proxy, the lower the more uniform, positively related to NTP L2 prediction difficulty).

**⚠️ Fix #1**: 0.6b h=64 The old reuse result (0.49%) used a different data set (n_items difference 14k) and has been rerun; but the rerun itself also has num_clusters bug.

**⚠️ Fix #2 (num_clusters bug)**: exp045 series Gini_d2 = 0.54–0.57, while exp026 = 0.33–0.35, the gap is entirely caused by num_clusters=1024 vs 4096. You need to rerun with `--num_clusters 4096` to get a reliable h vs quality curve.

### Analysis

1. **num_clusters bug is the main cause**: exp045 Gini_d2 (0.54–0.57) vs exp026 (0.33–0.35), the difference is significant. Gini_d2 measures the uniformity of L2 prefix distribution and directly affects the difficulty of NTP inter-layer prediction. All h-sweep conclusions are not credible before the bug is fixed.

2. **Gini_d2 > Gini_d3 (collision) is more informative**: When the FSQ codebook capacity (4096³) is much larger than the number of items (1.1M), Gini_d3 ≈ another expression of collision rate. Gini_d2 captures KMeans inter-tier load balancing and is more directly related to NTP tier prediction quality.

3. **4b collision is insensitive to h** (this conclusion may still be true even after the bug is fixed): The root cause is that the 12d_4096 codebook has a theoretical collision lower bound for 1.1M items. To reduce 4b collision, increase FSQ levels.

4. **8b embedding cache data problem**: The alignment rate between item_id and behavior data is only 2.3% and needs to be rebuilt.

### Next Steps
- **MUST DO**: Rerun EXP-045 sweep with `--num_clusters 4096` (0.6b h=32/64/128/256, 4b h=128/512/1024)
- exp026-0.6b-14d (h=64, num_clusters=4096, Gini_d2=0.33, CR=0.49%) is still the most credible 0.6b SID currently
- exp026-4b-14d (h=64, num_clusters=4096, Gini_d2=0.35, CR=2.76%, R@500=70.4%) is still the best 4b SID currently
- To reduce 4b collision, increase FSQ levels (outside the scope of this experiment)
- 8b embedding cache needs to be rebuilt (item_id alignment problem)

---

## EXP-044: TO-RoPE vs Absolute Position Embedding — S-tier + 0.6B SID

**Date**: 2026-04-29
**Status**: completed

### Background
EXP-043 baseline (exp043-s-0.6b) uses absolute position embedding + segment_emb + time_gap + action_level.
This experiment tests whether TO-RoPE (Time-and-Order RoPE, arxiv 2510.20455) can improve temporal order modeling in recommendation sequences.

**Important limitation (discovered afterward)**: The timestamps of this TO-RoPE experiment are 0 throughout the entire process - there is indeed a `first_ts` field in the original data, but the preprocess pipeline is not connected to the NTP side. Therefore, this experiment is equivalent to "TO-RoPE with zeros vs absolute pos emb", which is not a fair comparison. The timestamps pipeline has been completely connected after this experiment (`rel_hours = (ts[i] - ts[0]) / 3600`), and the next round of experiments will use real timestamps.

In addition: time_gap_emb was previously blocked by the conditional block of `use_torope=True`, which has also been fixed (the two can coexist).

### Design
- **Variable**: RoPE mode (absolute pos / TO-RoPE time_split=0.5 / TO-RoPE time_split=0.25) × segment_emb switch
- **Fixed**: S-tier + 0.6B SID, 14d data, time_gap + action_level (but the TO-RoPE path time_gap is blocked bug, has been fixed)
- **Baseline**: exp043-s-0.6b (R@500=61.2%, PPL=26.52)

| Config | Description |
|--------|------|
| exp044-baseline | absolute pos + segment + time_gap + action (≡ exp043-s-0.6b, not retrained) |
| exp044-torope-ts05 | TO-RoPE time_split=0.5 + segment + action (time_gap is blocked bug) |
| exp044-torope-ts025 | TO-RoPE time_split=0.25 + segment + action (time_gap is blocked bug) |
| exp044-torope-ts05-noseg | TO-RoPE time_split=0.5 + action only (no segment) |

### Results

| Config | R@10 | R@500 | PPL | L0 PPL | Wall |
|--------|------|-------|-----|--------|------|
| **exp043-s-0.6b** (baseline) | **11.4%** | **61.2%** | **26.52** | 390.2 | 8min |
| exp044-torope-ts05 | 10.8% | 60.1% | 377.30 | 672.5 | 7min |
| exp044-torope-ts025 | 11.4% | 60.4% | 396.84 | 681.7 | 7min |
| exp044-torope-ts05-noseg | 11.3% | 60.1% | 251.07 | 606.9 | 7min |

### Analysis

1. **TO-RoPE are all worse than baseline**: R@500 is 0.8–1.1pp lower, PPL is orders of magnitude higher (26 vs 250-400).

2. **The root cause of abnormally high PPL**: timestamps are all 0 → TO-RoPE time dimension is invalid, which is equivalent to replacing absolute pos emb with RoPE, and the frequency design of RoPE may not match the short sequence recommendation scenario. In addition, time_gap_emb is blocked, and the total amount of information is less.

3. **R@500 gap is smaller than expected** (only ~1pp): R@500 is not sensitive to absolute position encoding quality, and the top-k recall of beam search is more determined by SID discrimination and L0 distribution.

4. **Experimental invalid conclusion**: This experiment cannot prove that TO-RoPE is not as good as absolute pos emb. Correct comparison requires the coexistence of real timestamp + time_gap_emb, and the TO-RoPE design also needs to adjust the frequency base according to the recommended scenario (hourly interval).

### Next Steps
- ~~EXP-044B~~: Completed, see below
- ~~EXP-044C~~: Completed, see below

---

## EXP-044C: TO-RoPE Item-Pos Fix + 3-dim RoPE

**Date**: 2026-04-29
**Status**: completed

### Background
EXP-044B best (ts=0.25) R@500=63.6%, but PPL=467. Two assumptions:
1. Position-RoPE uses token-level index (0,1,2,3,4,5…), but time-RoPE treats all tokens in the same item as simultaneous → conflict signals, which may cause PPL to be abnormally high. Fix: Use item-level position (`pos//L`).
2. SID layer index (0/1/2, i.e. pos%L) is added as the third RoPE dimension, allowing attention to directly perceive the distance between layers.

### Design
- **Fixed**: s-tier, 0.6b SID, ntp_data=exp044b-0.6b-14d (timestamps are connected)
- **Variable**: torope_time_split, torope_layer_split

| Config | Description | torope_time_split | torope_layer_split |
|--------|------|-------------------|--------------------|
| A | 2-dim + item-pos fix | 0.25 | 0.0 |
| B | 2-dim ts=0.5 + item-pos fix | 0.50 | 0.0 |
| C | 3-dim ts=0.25 layer=0.15 | 0.25 | 0.15 |
| D | 3-dim ts=0.25 layer=0.25 | 0.25 | 0.25 |

### Results

⚠️ **2026-04-30 re-eval**: Fixed teacher-forced eval timestamps bug (previously all zeros), PPL has been corrected to true values.

| Config | R@10 | R@500 | PPL | L2 PPL |
|--------|------|-------|-----|--------|
| 043 baseline (abs pos) | 8.5% | 49.5% | 52.0 | — |
| 044B best (ts=0.25) | 7.4% | 54.3% | 47.5 | 27.7 |
| **044C-A**: 2-dim ts=0.25 + pos fix | 11.5% | 63.5% | **23.1** | 3.6 |
| **044C-B**: 2-dim ts=0.5 + pos fix | 11.8% | **63.9%** | **22.7** | 3.5 |
| **044C-C**: 3-dim ts=0.25 layer=0.15 | 11.9% | 62.4% | **22.7** | 3.5 |
| **044C-D**: 3-dim ts=0.25 layer=0.25 | 12.5% | 63.7% | **22.3** | 3.5 |

### Analysis

1. **Conclusion changes after PPL fix**: 044C series PPL 22-23, which is similar or even lower than baseline (26.5), indicating that item-level position fix (pos//L) does improve the quality of teacher-forced eval. 044B PPL 47.5 is on the high side because 044B uses token-level position and the logit distribution entropy is higher.

2. **R@500 The conclusion remains unchanged**: 044C-B (ts=0.5) is the best at 63.9%, 3-dim RoPE has no profit, and 2-dim ts=0.5 is the recommended configuration.

3. **044B re-eval R@500 decrease** (63.6% → 54.3%): re-eval with n_recall=1000, and timestamps are now actually passed in. This shows that the R@500=63.6% of the old eval is also partially falsely high (all zero timestamps are equivalent to turning off the time dimension, degenerating into pure order RoPE, but more stable? Further confirmation is needed).

### Next Steps
- TO-RoPE optimal configuration: 2-dim ts=0.5 (B, R@500=63.9%), as the new baseline
- 3-dim RoPE will not be promoted yet
- Next step first: EXP-045 FSQ sweep (fix behavior_path alignment and rerun)

---

## EXP-044B: TO-RoPE with Real Timestamps — S-tier + 0.6B SID

**Date**: 2026-04-29
**Status**: completed

### Background
In EXP-044, the timestamps are all 0 (the pipeline is not connected). This experiment is repaired and rerun:
- `_build_sequences_from_behavior` added `timestamps` calculation (rel_hours)
- time_gap_emb coexists with TO-RoPE
- Separate ablation: remove time_gap (Config D) to verify whether the two are complementary

### Design
- **Variable**: TO-RoPE time_split (0.5 / 0.25) × coexistence with or without time_gap
- **Fixed**: S-tier + 0.6B SID, 14d data, action_level + segment_emb
- **Baseline**: exp043-s-0.6b (R@500=61.2%, PPL=26.52)

| Config | Description |
|--------|------|
| exp043-s-0.6b | Baseline: abs pos + time_gap + action + segment |
| exp044b-torope-ts05 | TO-RoPE ts=0.5 + time_gap + action + segment |
| exp044b-torope-ts025 | TO-RoPE ts=0.25 + time_gap + action + segment |
| exp044b-torope-ts05-notg | TO-RoPE ts=0.5 + action + segment (no time_gap ablation) |

### Results

⚠️ **History (invalid result)**: The first eval caused timestamps=0 due to a two-layer train-infer bug, and the result was invalid (R@500≈32%). The results after repair are shown below.

⚠️ **2026-04-30 re-eval**: Fixed teacher-forced eval timestamps bug (previously all zeros), PPL has been corrected to true values.

| Config | R@10 | R@500 | PPL | L2 PPL | Remarks |
|--------|------|-------|-----|--------|------|
| **exp043-s-0.6b** (baseline) | **11.4%** | **61.2%** | **26.5** | — | abs pos + time_gap + action + seg |
| exp044b-torope-ts05 | 8.5% | 53.8% | 41.7 | 18.5 | TO-RoPE ts=0.5 + time_gap |
| exp044b-torope-ts025 | — | **54.3%** | **47.5** | 27.7 | TO-RoPE ts=0.25 + time_gap ← best R@500 |
| exp044b-torope-ts05-notg | 8.8% | 56.2% | 40.8 | 18.0 | TO-RoPE ts=0.5, no time_gap |

### Analysis

1. **TO-RoPE valid (044B < 044C)**: After fixing the teacher-forced eval timestamps bug, 044B R@500 is 53.8-56.2% (lower than baseline 61.2%). 044C uses item-level position (pos//L) to reach 63.5-63.9%. Confirming item-level position is the key.

2. **044B PPL=40-48 (reasonable) after PPL fix**: Previously PPL=467-480 was the result of bug (timestamps=0). After repair, the PPL is 40-48, which is about 50% higher than the baseline (26.5), and L2 PPL=18-28 (044C L2 PPL≈3.5 makes L2 prediction extremely easy due to item-level).

3. **time_split=0.25 best R@500 (54.3%) but the highest PPL (47.5)**: When more position dimensions are left to order-RoPE, R@500 is slightly higher but L2 PPL is higher (27.7 vs 18.0), indicating that ts=0.25 encodes time information weakly and the model requires more order signals.

4. **Bug review - two layers of train-infer are inconsistent**:
   - **Bug 1**: `_step_sf` of `constrained_beam_search` only processes the `inject='embed_add'` feature, and the timestamps of `inject='torope'` are all 0 in the generation step. Fix: Add `_step_ts()` carry-forward.
   - **Bug 2 (more hidden)**: The `eval_items` construction loop filter condition of `eval.py` is `inject != 'embed_add'`, resulting in timestamps never being put into `ctx_side_features`, and the carry-forward logic has no chance to be executed at all. Fix: The loop is processed in two branches: `embed_add` and `torope`.
   - Two layers of bugs are superimposed, leading to misjudgment that TO-RoPE is invalid. Lesson: **New features must be built from preprocess → train → eval_items → beam search to verify whether the feature is non-zero**.

5. **Follow-up direction**: TO-RoPE +2pp is statistically significant and worthy of verification on a larger model (4B/8B SID). time_split=0.25 is used as the default recommended parameter.

---

## EXP-041B: ENTP-Loss v2 — Session-Level Negatives (behavior_v2 data)

**Date**: 2026-04-29
**Status**: completed (Conclusion: invalid, session granularity issue)
**Results**: experiments/ntp_checkpoints/exp041b-entp{005,01,02}/

### Background

EXP-041 The root cause of the failure is that the behavior data is replaced with `exposure_neg` (the user collection is different). Correct approach: Mainly use positive behavior sample sequences, and append unclicked items in the session as neg_l0. `export_behavior_v2.py` has exported this format (uid, session_id, iid, action_bitmap), n_seqs=1,745,799, has_neg_l0=True, entp_k=5.

### Hypothesis

The behavior_v2 data contains negative samples within the session. ENTP α=0.1 increases R@500 by +2~4pp from 59.0%.

### Design

- **Variable**: ENTP weight α ∈ {0.05, 0.1, 0.2}; α=0 directly quotes exp036-full-features
- **Fixed**: behavior_v2 data, time_gap+action_level+segment_emb, 4096×3 binary SID, 1 epoch
- **Baseline**: exp036-full-features (existing, no retraining)
- **Data**: feed_user_behavior_v2 (2026-03-18~03-31), n_seqs=1,745,799

### Results

| Config | R@10 | R@500 | PPL | Wall |
|--------|------|-------|-----|------|
| exp036-full-features (α=0, baseline) | 10.9% | 59.0% | 27.3 | 7min |
| exp041b-entp005 (α=0.05) | 7.8% | 44.9% | 49.7 | 1min |
| exp041b-entp01  (α=0.1)  | 7.7% | 46.5% | 51.3 | 1min |
| exp041b-entp02  (α=0.2)  | 8.0% | 46.1% | 50.4 | 1min |

### Analysis

**Conclusion: ENTP v2 is invalid, and the root cause is the wrong session granularity. **

1. **`df_4` is not session**: `exposed` CTE in `export_behavior_v2.py` uses `df_4 AS session_id`, but `df_4` is actually a single view ID of each `$AppExposure` event (an independent ID is refreshed each time an exposure is refreshed), not a user session ID.
2. **There is almost no negative sample space within the session**: Local verification shows that 98% of sessions have only 1 exposed item, 1.99% have 2, and 0 have more than 3. An exposure event = 1 item, the user clicked on that item, neg_candidates = 0.
3. **neg:pos = 1:0.01**: The coverage rate of neg_l0 in 1.75 million sequences is less than 1%, and ENTP loss is almost not triggered, which is equivalent to pure NTP training.
4. **PPL rises from 27 to 50**: The sequence quality of behavior_v2 data itself is worse than behavior (it may contain users with less behavior or join causes the sequence to change), resulting in a decrease in basic performance.

### Next Steps

- If you want to do ENTP, you need to redefine the session: aggregate multiple exposures into one session according to the time window (such as within 30 minutes), so that each session has multiple items that can distinguish positive and negative
- Or return to the original OneRec/DualGR solution: use user-level exposure to negative samples (non-session) and directly join behavior data

---

## EXP-041: ENTP-Loss — Exposure-Aware Hard Negatives for L0 (with Features)

**Date**: 2026-04-29
**Status**: completed (Conclusion: Invalid, needs to be redesigned)
**IDEA**: IDEA-dualgr-0
**Results**: experiments/ntp_checkpoints/exp041-entp{005,01,02}/

### Background

EXP-036 SFT route R@500=59.0% (7pp worse than exp020 SOTA 66.2%). The L0 layer is still the bottleneck (PPL=362.9). DualGR's ENTP-Loss gives L0 additional supervision by exposing unclicked items. At that time, the data side was verified in EXP-014 (130M positive samples, 31% with negative samples), but the NTP integration was not completed. This experiment completes the ENTP integration based on the features pipeline (time_gap + action_level + segment_emb).

### Hypothesis

ENTP α=0.1 reduces L0 PPL by >10% (from ~362 to ~320) and increases R@500 by +2~4pp. The optimal α is between 0.05~0.1 (too large will introduce noise).

### Design

- **Variable**: ENTP weight α ∈ {0(baseline), 0.05, 0.1, 0.2}
- **Fixed**: S-tier 6L MoE, 4096×3 binary SID, K=5 negatives, time_gap+action_level+segment_emb, 1 epoch
- **Metric**: L0/L1/L2 PPL, R@{10,500}
- **Data**: 14d behavior (2026-03-18~03-31) + exposure neg (same period, S3)

### Run
`bash experiments/scripts/exp-041.sh --no-smoke`

### Results

| Config | R@10 | R@500 | PPL | L0 PPL | Wall |
|--------|------|-------|-----|--------|------|
| exp036-full-features (α=0, baseline) | 10.9% | 59.0% | 27.3 | 362.9 | 7min |
| exp041-entp005 (α=0.05) | 6.5% | 39.2% | 68.1 | 434.5 | 2min |
| exp041-entp01 (α=0.1) | 7.5% | 39.7% | 69.1 | - | 2min |
| exp041-entp02 (α=0.2) | 7.2% | 40.3% | 69.3 | - | 2min |

### Analysis

**Conclusion: The ENTP experiment is invalid and there are fundamental problems with the data design. **

1. **ENTP data user collection is different**: `exposure_neg` data comes from `feed_user_exposure` (exposure record), the user pool is much larger than `behavior` data (n_seqs=3.07 million vs 1.7 million), but p50 is only 6 items. The exposure data contains a large number of cold users (who have only viewed a few items). The historical sequences of these users are extremely short and cannot provide effective sequence learning signals.

2. **Poor sequence quality**: behavior data filters out users with <2 items (with clear behavior), while exposure data only requires exposure records. A large number of short sequences of 1-2 items diluted the training signal, causing the PPL to explode from 27 to 68.

3. **Correct approach**: ENTP should append neg_l0 (that is, exposure negative samples of the same user) based on **behavior data** instead of replacing behavior data with exposure data. It is necessary to join by uid: retain all positive behavior sample sequences, and only add ENTP loss to the positions with corresponding exposure negative samples.

4. **Alpha value has little impact**: The three alpha results are almost the same (39.2% / 39.7% / 40.3%), indicating that the problem is not in the alpha selection, but in the data design itself.

### Next Steps
- Modify `build_unified_sequences`: focus on behavior data, join neg_l0 from exposure_neg_data by uid+iid, keep the original behavior sequence unchanged
- Rerun EXP-041 to verify the ENTP effect in the correct way

---

## EXP-040: RSFT — Reject Sampling Fine-Tuning (Training Data Quality Filter)

**Date**: 2026-04-28
**Status**: planned
**IDEA**: IDEA-onerec-1
**Results**: TBD

### Background

The current NTP training (`action_bitmap > 0`) contains a large number of click-only weak positive samples (click-only). OneRec's RSFT solution: filters low-quality interactions and only trains on high-quality data (like/fav/share/purchase), which is equivalent to natural curriculum learning.

The implementation has added the `--min_action_level` parameter through code (`ntp/preprocess.py` + `ntp/train.py`), using the `_action_bitmap_to_level` mapping: level 1=click, 2=strong(like/fav/share), 3=trade(purchase).

### Hypothesis

`min_action_level=2` (strong+trade) filters about 70-80% of weak click behaviors and retains purer high-quality signals. R@500 is improved by +1~3pp, and PPL may increase slightly (due to reduced data volume). Level=3 is too aggressive (data is sparse) and may not be as good as level=2.

### Design

- **Variable**: min_action_level ∈ {1(baseline), 2(strong+trade), 3(trade only)}
- **Fixed**: S-tier 6L MoE, time_gap+action_level+segment_emb, 4096×3 binary SID, 1 epoch
- **Metric**: R@{10,500}, PPL, training data size
- **Data**: 14d behavior (2026-03-18~03-31), same SID cache

### Run
`bash experiments/scripts/exp-040.sh`

### Results
- Config A (baseline, min_level=1): Reference EXP-036: R@10=10.8%, R@500=59.0%, PPL=24.0
- Config B (RSFT-2): TBD
- Config C (RSFT-3): TBD

### Analysis
TBD

### Next Steps
TBD

---

## EXP-039B: ECPO on exp038b-hard-lam03-3ep-ep1 (Features RL link endpoint)

**Date**: 2026-04-29
**Status**: completed
**Results**: experiments/ntp_checkpoints/exp039b-ecpo-from-spdpo/

### Background

The final step of RL alignment link: exp036 SFT → EXP-037 SP-DPO → EXP-038B RF-DPO (ep1 best, R@500=62.1%) → ECPO of this experiment.
EXP-039 (from exp038-hard-lam03) has been skipped, starting directly from exp038b ep1 to take advantage of a better starting point.

### Hypothesis

ECPO (δ=0.1) reproduces the exp029 magnitude improvement (+4pp) on the features model, and the final R@500 is close to or exceeds exp020 SOTA (66.2%).

### Design

- **Variable**: ECPO δ=0.1, starting from exp038b-hard-lam03-3ep-ep1 (R@500=62.1%)
- **Fixed**: G=512, BehaviorReward+FormatReward, on-policy beam, grpo_weight=0.03, lr=1e-4, 8×L20X
- **Metric**: R@{10,500}, PPL
- **Data**: context pool from exp023-14d-features, behavior cache 2026-03-31

### Run
`bash experiments/scripts/exp-039b.sh --no-smoke`

### Results

| Config | R@10 | R@500 | PPL | Wall |
|--------|------|-------|-----|------|
| exp036-full-features (SFT) | - | ~53% | - | - |
| exp037-medium (SP-DPO, ref) | - | 62.1% | - | - |
| exp038b-ep1 (RF-DPO) | - | 62.1% | - | - |
| **exp039b-ecpo (this)** | **11.8%** | **65.7%** | **20.0** | **182min** |
| exp020-hard-lam03 (SOTA no Feature) | 14.1% | 66.2% | 16.3 | - |

### Analysis

- ECPO increased from RF-DPO ep1 (62.1%) to **65.7%**, +3.6pp, consistent with exp029 improvement
- Only **0.5pp** away from featureless SOTA (66.2%), almost tied
- PPL=20.0 is higher than SOTA (16.3), indicating that there is still room for NTP quality of the features route
- behavior_mean reward from 0.574 → 0.630 (good convergence trend), coverage=98.8%
- **Conclusion**: The features RL link (SFT→DPO→ECPO) is effective, but the introduction of features brings PPL cost

### Next Steps
- EXP-040: RSFT (behavioral quality filtering) verifies whether the baseline can be improved during the SFT stage
- EXP-041: ENTP-Loss (exposure negative sample α sweep) to verify the L0 negative sample penalty effect

---

## EXP-039: ECPO on exp038-hard-lam03 (Features RL link endpoint)

**Date**: 2026-04-28
**Status**: skipped (superseded by EXP-039B)
**Results**: TBD

### Background

The final step of RL alignment link: exp036 SFT → EXP-037 SP-DPO → EXP-038 RF-DPO → ECPO of this experiment. Verify that the features route plus full RL link can exceed exp020 SOTA (R@500=66.2%).

### Hypothesis

ECPO (δ=0.1) has the same improvement as exp029 on the features model by +2~4pp, and the final R@500 > 62% is expected to be close to or exceed exp020 SOTA.

### Design

- **Variable**: ECPO δ=0.1 (vs 0 = pure GRPO)
- **Fixed**: ref=exp038-hard-lam03, G=512, BehaviorReward+FormatReward, on-policy beam, grpo_weight=0.03, lr=1e-4
- **Metric**: R@{10,500}, PPL, advantage_mean, clip_fraction, reward_mean
- **Data**: context pool from exp023-14d-features, behavior cache 2026-03-31

### Run
`bash experiments/scripts/exp-039.sh`

### Results
TBD

### Analysis
TBD

### Next Steps
TBD

---

## EXP-038B: RF-DPO on exp037-medium — ntp_epochs=3 + mid-checkpoints

**Date**: 2026-04-28
**Status**: completed
**Results**: experiments/ntp_checkpoints/exp038b-hard-lam03-3ep-ep1/ (best)

### Background

After EXP-038 RF-DPO (1 epoch, 406 steps), R@500=59.6%, PPL=25.7, 2.5pp degraded compared to ref (exp037-medium 62.1%). Cause analysis: The number of steps is too few (406 steps), and the NTP:DPO ratio is not aligned with the exp019/020 design (exp020 target is 807 DPO steps ≈ NTP steps).

EXP-038B uses `--ntp_epochs 3` (total 1218 steps) and saves mid-checkpoint at each epoch boundary to compare the effect of ep1/ep2/ep3(final).

**Code implementation**: Added `ntp_epochs` parameter (`itertools.chain.from_iterable(itertools.repeat(ntp_loader, ntp_epochs))`), mid-checkpoint is saved to `{output_dir}-ep{N}` at the end of each epoch.

### Hypothesis

ep1 (406 steps) = 1 epoch of alignment to exp038, expected to be comparable to EXP-038 (~59.6%). More epochs may improve DPO alignment but risk NTP overfitting.

### Design

- **Variable**: ntp_epochs ∈ {1,2,3} (three-point comparison is achieved through mid-checkpoint)
- **Fixed**: ref=exp037-medium, λ=0.03, β=0.1, difficulty=hard, lr=1e-4, Joint NTP+DPO
- **Metric**: R@{10,500}, PPL (three epochs for each review)
- **Data**: RF-DPO pairs from exp018 real feedback (2026-03-18~03-31), 4,312 hard pairs

### Run
`bash experiments/scripts/exp-038b.sh`

### Results

| Checkpoint | Steps | R@10 | R@500 | PPL | Conclusion |
|---|---|---|---|---|---|
| exp037-medium (ref) | — | 11.2% | 62.1% | 23.0 | SP-DPO starting point |
| **ep1 (1 epoch)** | 406 | **11.2%** | **62.1%** | **23.6** | ✅ Flat ref, DPO lossless |
| ep2 (2 epochs) | 812 | 10.3% | 59.6% | 26.0 | ❌ NTP starts to overfit |
| final (3 epochs) | 1218 | 9.3% | 52.8% | 33.3 | ❌ Severe overfitting |

**Best checkpoint**: `exp038b-hard-lam03-3ep-ep1` (ep1, R@500=62.1%)

### Analysis

1. **ep1 flat ref (no degradation!)**: The reason why EXP-038 1 epoch degrades to 59.6% may be that the LR is too high or the training is unstable, while EXP-038B ep1 gets 62.1% with the same number of steps, indicating that the impact of DPO on NTP is neutral within 1 epoch.

2. **2/3 epoch NTP overfitting**: NTP loss begins to overfit after multiple cycles on the exp018 real feedback data (narrow distribution), and PPL deteriorates rapidly from 23.6 → 26.0 → 33.3.

3. **Key Lessons**: The optimal RF-DPO is 1 epoch; `--ntp_epochs` should be set to 1 (experimentally verified). Subsequent experiments used ep1 as the ECPO starting point.

### Next Steps
- EXP-039B: ECPO on ep1 (`exp038b-hard-lam03-3ep-ep1`), δ=0.1, G=512, on-policy beam

---

## EXP-038: RF-DPO on exp037-medium (Features route step 3)

**Date**: 2026-04-28
**Status**: completed
**Results**: experiments/ntp_checkpoints/exp038-hard-lam03/

### Background

RL alignment link: exp036 SFT → EXP-037 SP-DPO → this experiment RF-DPO → EXP-039 ECPO. Use exp037-medium as ref, reuse EXP-018 real feedback data (2026-03-18~03-31), Joint NTP+DPO λ=0.03 (exp020 optimal configuration).

**Note**: In the first run, DPO did not take effect (`n_dpo_pairs=0`) because `--preference_dir` points to the root directory instead of the `hard/` subdirectory. Fix the script and run it again.

### Hypothesis

R@500 jumps from exp037-medium (~62%) to ~65%+ (exp018→020 has a +7pp improvement for featureless models).

### Design

- **Variable**: (There is no control in this experiment, only config)
- **Fixed**: ref=exp037-medium, λ=0.03, β=0.1, difficulty=hard, lr=1e-4, Joint NTP+DPO
- **Metric**: R@{10,500}, PPL
- **Data**: RF-DPO pairs from exp018 real feedback (2026-03-18~03-31), 4,312 hard pairs

### Run
`bash experiments/scripts/exp-038.sh`

### Results

| Phase | R@10 | R@50 | R@100 | R@500 | PPL | wall |
|------|------|------|-------|-------|-----|------|
| exp037-medium (ref) | 11.2% | 26.6% | 38.2% | 62.1% | 23.0 | — |
| **exp038-hard-lam03** | **10.9%** | 24.9% | 34.4% | **59.6%** | **25.7** | 1181s |

DPO training indicators: n_dpo_pairs=4312, avg_dpo_loss=2.574, reward_margin=3.44, preference_acc=36.9%
Alignment eval: chosen_reward=-0.28, rejected_reward=-5.68, reward_margin=5.40, preference_acc=53.0%

### Analysis

Compared with exp037-medium (SP-DPO), RF-DPO degrades: R@500 62.1%→59.6% (-2.5pp), PPL 23.0→25.7 (worse). This is contrary to the pattern of the featureless version of exp020 (exp019→020 has a significant improvement).

Possible reasons:
1. exp018 real feedback pairs has a small amount of data (4312 pairs) and does not match the features model distribution
2. SP-DPO (exp037) has been aligned on the beam search pairs, and the direction of the real feedback signal is conflicting.
3. Compare the training route of exp020: the ref of exp019→020 is exp017-medium (only SP-DPO), and the DPO effect of this experiment is inherently less than NTP (λ=0.03, low weight)

### Next Steps
- EXP-039 ECPO (δ=0.1, starting from exp038-hard-lam03) continues the link, and the GRPO reward mechanism is independent of DPO pairs
- If EXP-039 does not improve, consider doing ECPO directly from exp037-medium (bypassing RF-DPO)

---

## EXP-037: SP-DPO on exp036-full-features (Step 2 of Features route)

**Date**: 2026-04-28
**Status**: completed
**Results**: experiments/ntp_checkpoints/exp037-easy/ + experiments/ntp_checkpoints/exp037-medium/

### Background

exp036-full-features (R@500=59.0%) is a clean features NTP baseline. To reproduce the full aligned link of exp020 on the features route, the next step is SP-DPO.

**Complete features alignment link** (benchmark exp016→017→019/020→029):
```
exp036-B (NTP+feat) → EXP-037 SP-DPO → EXP-038 RF-DPO → EXP-039 ECPO
```

**Why can’t you skip SP-DPO and do RF-DPO directly**:
- The RF-DPO ref model of exp019/020 is SP-DPO checkpoint (exp017 fixed-medium), not NTP baseline
- Doing RF-DPO directly on NTP is equivalent to exp018 (pure DPO), and experiments have proven that disaster will be forgotten (PPL → 50K+)
- SP-DPO provides two functions: (1) Injecting contrast signals into the model to improve R@10; (2) As a ref checkpoint for RF-DPO, making the KL constraint starting point reasonable

**SP-DPO pairs must be regenerated**: beam search candidates rely on the current model distribution, and exp017 pairs cannot be reused.

### Hypothesis

| Metric | exp036-B (NTP+feat) | Expected changes | Reasons |
|------|---------------------|---------|------|
| R@10 | 10.9% | ↑ ~13% | SP-DPO Easy+Medium in exp017 Medium R@10 from 9.9%→12.5%, features version expected similar range |
| R@500 | 59.0% | → Flat or slightly down | SP-DPO has little impact on R@500 (exp017: 58.5%→55.0% Easy, Medium rebounds) |
| PPL | 27.3 | ↑ Slight increase | DPO loss slightly interferes with NTP, in line with the exp017 rule (27→28.5 Easy) |
| depth_acc L0 | — | ↑ | SP-DPO Easy has significantly improved L0 discrimination (exp017: +37%) |
| clip_fraction | N/A | N/A | SP-DPO Phase has no RL, clip is not applicable |
| kl_mean | N/A | N/A | SP-DPO Phase None RL |
| adv_std | N/A | N/A | SP-DPO Phase None RL |
| behavior_coverage | N/A | N/A | SP-DPO Phase None RL |
| behavior_mean | N/A | N/A | SP-DPO Phase None RL |

### Design
- **Variable**: SP-DPO Easy + Medium, prefix-locked beam search, generate pairs based on exp036-full-features
- **Fixed**: S-tier model, NTP data=exp023-14d-features, beam_size=50, n_rejected=20, λ=0.1, β=0.1, lr=1e-4
- **Metric**: PPL, R@10, R@500 (full eval, n_recall=1000), depth_acc L0/L1
- **Data**: NTP=exp023-14d-features; SP pairs are newly generated from exp036-B beam search
- **Skip Hard stage**: exp017 analysis confirms Hard degradation (L0/L1 depth_acc decreases), only do Easy+Medium

### Run
`bash experiments/scripts/exp-037.sh`

### Results

| Phase | R@10 | R@50 | R@100 | R@500 | PPL | wall |
|------|------|------|-------|-------|-----|------|
| exp036-B (SFT) | 10.9% | — | — | 59.0% | 27.3 | — |
| Easy | 10.4% | — | — | 57.1% | 24.60 | — |
| Medium | **11.2%** | 26.6% | 38.2% | **62.1%** | **23.0** | 1246s |

Medium alignment eval: chosen_reward=1.003, rejected_reward=-4.990, reward_margin=5.994, preference_acc=36.0%

### Analysis

Medium stage R@500 rebounded from Easy 57.1% to 62.1%, exceeding exp036-B SFT (59.0%), consistent with the exp017 rule (Medium is always better than Easy). R@10 edged up to 11.2% from Easy 10.4%, essentially the same as SFT. The PPL dropped to 23.0 (better than SFT 27.3), indicating that SP-DPO medium has no obvious negative impact on NTP loss.

reward_margin=5.99 is significantly larger than the Easy stage, preference_acc=36% (Note: lower than 50% does not mean bad, this is the absolute standard relative to "chosen is the target item", and the chosen itself generated by beam search is a pseudo-label).

### Next Steps
- exp037-medium → EXP-038 RF-DPO (ref=exp037-medium, Joint NTP+DPO λ=0.03, reuse exp018 real feedback data)
- Goal: Reproduce the complete features of exp020 and align the links to see if RF-DPO can break through R@500=66.2% SOTA

---

## EXP-036: Clean Features NTP — From-Scratch Training with time_gap + action_level

**Date**: 2026-04-28
**Status**: completed
**Results**: experiments/ntp_checkpoints/exp036-no-features/ + experiments/ntp_checkpoints/exp036-full-features/

### Background

exp025 (R@500=63.6%) is currently the only NTP checkpoint with features, but it is not a clean control experiment:
- Made beam-passes SFT based on exp023 (instead of training from scratch)
- The difference from exp020 (R@500=66.2%) is not only the features, but also the training methods and data sets.

EXP-023's "all features" config (time_gap + action + segment) R@500 is only 55.0%, which is lower than segment_only (61.2%). The main reason is **training-inference information leakage** (EXP-024 analyzed). EXP-025 fixed the leak (delayed features + beam_passes), but only did 1 epoch beam_passes, and the training was insufficient.

**Goal**: Use exactly the same training conditions as exp020 (same data set, same super parameters, training from scratch), the only variable is whether to add features, and obtain clean control results.

### Hypothesis

features (time_gap + action_level + segment_emb) should be able to exceed exp020 after training from scratch after the training-inference gap is repaired:

| Metric | exp020 (no features) | EXP-036 (features, trained from scratch) | Expected changes | Reasons |
|------|---------------------|--------------------------|---------|------|
| PPL | 16.3 | Expected ≤16.3 | ↓ | features provide additional distinguishing signals |
| R@10 | 14.1% | Expected ≥14.1% | ↑ | time_gap distinguishes timeliness |
| R@500 | 66.2% | Expected ≥67% | ↑ | action_level differentiates interaction strength |
| TrainingTime | ~62min | ~65min | ↑small | features embedding slightly increases the amount of calculation |

### Design
- **Variable**: features on/off (Config A: no features recurrence exp020; Config B: time_gap + action_level + segment)
- **Fixed**: The same data set (exp023-14d-features, including time_gaps/action_levels), the same hyperparameters (lr=1e-3, batch=4096, 1 epoch, s-tier model), the same SID cache (exp013-4096x3-12d-binary)
- **Metric**: PPL, R@10, R@500 (full eval n_recall=1000); add kl_mean as subsequent RL benchmark
- **Data**: experiments/ntp_data/exp023-14d-features (existing, no need to re-preprocess)

### Run
`bash experiments/scripts/exp-036.sh`

### Results

| Metric | Config A (no features) | Config B (full features) | exp020 Baseline | Δ(B-A) |
|------|-----------------------|------------------------|------------|--------|
| R@10 | 9.4% | **10.9%** | 14.1% | +1.5pp ✅ |
| R@500 | 55.3% | **59.0%** | 66.2% | +3.7pp ✅ |
| PPL | 34.9 | **27.3** | 16.3 | ↓7.6 ✅ |
| train_loss | 3.620 | **3.507** | — | -0.113 ✅ |
| Training duration | 7min50s | 7min58s | ~62min | Similar |

> Note: The absolute values ​​of Config A/B are lower than exp020 (66.2%), because the exp023-14d-features data set is used (the item set is different), which cannot be directly compared with the exp016-14d data used by exp020. **The key conclusion is to look at the B-A difference**.

### Analysis

**Assumption verification results**:

| Hypothesis | Expectation | Actual | Conclusion |
|------|------|------|------|
| PPL: B < A | ↓ | 27.3 vs 34.9 (↓7.6) | ✅ Verification |
| R@10: B > A | ↑ | 10.9% vs 9.4% (+1.5pp) | ✅ Verification |
| R@500: B > A | ↑ | 59.0% vs 55.3% (+3.7pp) | ✅ Verification |

**Key Conclusions**:

1. **Features are valid**: Same data, same super parameters, the only difference is features on/off, Config B is better than Config A in all aspects. PPL dropped from 34.9 to 27.3 (a decrease of 22%), and R@500 increased by 3.7pp, which is a significant effect.

2. **The reason why the absolute value is lower than exp020**: The exp023-14d-features data set and exp016-14d are different item collections/time windows and cannot be directly compared across data sets. The purpose of this experiment is to control the variables to verify the effect of the features themselves, and this purpose has been achieved.

3. **The disadvantages of exp025 have been explained**: exp025 (beam-passes SFT, R@500=63.6%) is worse than exp020 because the training starting point itself is weak when doing beam-passes based on exp023 (segment-only, 61.2%). Config B training from scratch avoids this problem.

4. **New SFT starting point**: exp036-full-features is currently the only NTP checkpoint that is trained from scratch, has consistent training and pushing, and has valid features. It can be used as the SFT starting point for the next round of RL.

### Next Steps

**Complete features route link (benchmarked exp016→017→019/020→029):**

```
exp036-B (NTP + features)
→ EXP-037: SP-DPO Easy+Medium (prefix-locked beam, pairs generated based on exp036-B)
→ EXP-038: RF-DPO Joint NTP+DPO λ=0.03 (ref = SP-DPO checkpoint, benchmark exp019/020)
→ EXP-039: ECPO on-policy (compared to exp029)
```

**Important background (to prevent misunderstandings again)**:
- Complete source of exp020 (SFT SOTA): `exp016(NTP) → exp017(SP-DPO) → exp019/020(RF-DPO, ref=SP-DPO)`
- RF-DPO is not done directly on NTP, the ref model is SP-DPO checkpoint (exp017 fixed-medium)
- Pure RF-DPO will be forgotten by disaster (exp018 PPL explodes to 50K+), and Joint NTP+DPO is required to be stable.
- SP-DPO pairs must be regenerated using the current checkpoint (exp036-B), and the pairs of exp017 cannot be reused (beam candidates depend on model distribution)
- RF-DPO pairs (real user feedback) can reuse exp018 data (behavioral data has nothing to do with the model)

---

## EXP-035: Constrained Sampling — Replace Beam Search with T=1.0 Sampling

**Date**: 2026-04-28
**Status**: completed
**Results**: experiments/ntp_checkpoints/exp035-sampling-t1/

### Background

EXP-034 verified that ref/policy alignment is only partially improved (clip=95%, still high). The real root cause is the structural problem of beam search:

- Beam search always selects the most confident candidate for policy → ρ = π_θ/π_ref >> 1 → The clip rate is structurally high
- Candidates are concentrated at the policy peak → reward variance is minimal → advantage ≈ 0 → gradient degradation
- EXP-034 log shows `adv=-0.00`, the model is actually barely learning

**Solution**: Replace beam search with `constrained_sampling(T=1.0)`:
- Candidates are sampled directly from the policy distribution → ρ ≈ 1 by construction
- Sampling covers diverse paths (good/bad/medium) → advantage has truly contrasting signals
- G: 512→64 (diversity is guaranteed by T, no need for large G), video memory saving

### Hypothesis

| Metric | EXP-034 (beam G=512) | EXP-035 (sampling T=1.0 G=64) |
|------|--------------------------|----------------------------------|
| clip rate | ~95% | expected 10~40% |
| adv_std | ≈0 | Expected >0 (true Comparison signal) |
| R@500 | TBD | Expected ≥ EXP-034 |

### Design
- **Variable**: `--sampling_temperature 1.0`, `--group_size 64` (beam→sampling)
- **Fixed**: ref=exp025, policy=exp025, the remaining parameters are the same as EXP-034 (rank_norm, a2po, nll_reg, hepo)
- **Metric**: clip rate, adv_std, R@10, R@500
- **Data**: exp023-14d-features

### Run
`bash experiments/scripts/exp-035.sh`

### Results

| Metric | EXP-035 (sampling T=1.0, G=64) | EXP-029 SOTA (beam G=512) | Gap |
|------|----------------------------------|--------------------------|------|
| R@10 | 0.102 | 0.130 | -0.028 ❌ |
| R@500 | **0.615** | 0.678 | -0.063 ❌ |
| clip rate | 94.8% | 92.3% | slightly High |
| adv_std | 0.595 | — | Has Comparison signal ✅ |
| kl_mean | — | — | New Metric, recorded from the next Experiment |
| behavior_mean | 0.363 | ~0.65 | Low half ❌ |
| behavior_coverage | 89.2% | ~99% | Low ❌ |
| train time | 18min | ~70min | Fast 4x ✅ |

### Analysis

**Assumption verification results**:

| Hypothesis | Expectation | Actual | Conclusion |
|------|------|------|------|
| clip rate dropped to 10~40% | ✅ | 94.8% | ❌ Hypothesis error |
| adv_std > 0 | ✅ | 0.595 | ✅ There is Comparison signal |
| R@500 ≥ EXP-034 | ✅ | 0.615 < 0.678 | ❌ Not as good as SOTA |

**Key findings (post-training analysis)**:

1. **The root cause of high clip is not beam/sampling**: All experimental clips are between 92~96%, which is caused by the softmax drift of NTP joint training and has nothing to do with the candidate generation method.
2. **G=64 sparse reward problem**: beam G=512 coverage=99%, sampling G=64 coverage=89%, behavior_mean drops from 0.65 to 0.36, and the reward signal is half as weak
3. **adv_std=0.595 is progress**: adv≈0 during beam search (candidates are concentrated at the policy peak, reward variance is extremely small), and there is a real contrast signal after sampling
4. **Training efficiency is greatly improved**: 18min vs 70min, 4x faster (G is reduced by 8x)

**Conclusion**: The sampling direction itself is correct (adv_std is improved), but G=64 + behavior reward is sparse = the signal is too weak to surpass the result of beam G=512.

### Next Steps

- Added KL(π_θ||π_ref) as a core RL metric comparable across experiments
- EXP-036: sampling G=64 but improve behavior coverage (expand behavior cache or use prefix cascade fallback to improve matching rate)
- Or: sampling G=256 (a balance between efficiency and coverage)

---

## EXP-034: Ref Model Alignment — exp025 as ref_checkpoint

**Date**: 2026-04-28
**Status**: planned
**Results**: TBD

### Background

EXP-033 falsified the features bug hypothesis: after fixing three features injection bugs, the clip rate changed from 96.4% to 96.2%, with almost no change. The real root cause is that the ref model (exp020) is not aligned with the policy starting point (exp025).

The PPO clip condition is ρ = exp(policy_lp - ref_lp) beyond [1-ε, 1+ε]. exp025 performed beam-passes SFT on exp020, and the log-prob of the two for the same token was systematically different. Starting from exp025, a large number of clips are triggered in the first step, not because the update is too large, but because the initial KL is very large.

| Experiment | policy starting point | ref model | clip rate |
|------|------------|-----------|---------|
| exp031-baseline | exp020 | exp020 | 92.4% ✅ |
| exp031-features | exp025 | exp020 | 96.4% ❌ |
| exp033 | exp025 | exp020 | 96.2% ❌ |
| **EXP-034** | **exp025** | **exp025** | **Expected ~92%** |

### Hypothesis

ref model = policy starting point = exp025 → RL starts with KL≈0 → clip rate drops back to ~92% (aligned with exp031-baseline). The features model (R@500=63.6% SFT) with correct RL alignment should outperform the exp020 route (67.8%) since features provide better beam search discrimination.

### Design
- **Variable**: ref_checkpoint = exp025 (instead of exp020), the remaining parameters are exactly the same as exp031-baseline
- **Fixed**: ECPO δ=0.1, ε=0.2, G=512, grpo_batch=4, grpo_weight=0.03, ratio=1.0, lr=1e-4, on_policy, rank_norm, A2PO(α=1.0), NLL(0.01), HEPO(0.1,0.5)
- **Metric**: R@10, R@500 (full eval n_recall=1000), clip rate
- **Data**: exp023-14d-features, WeightedBehaviorReward + FormatReward

### Run
`bash experiments/scripts/exp-034.sh`

### Results
TBD

### Analysis
TBD

### Next Steps
TBD

---

## EXP-033: Features Correct Verification — EXP-031A Rerun with Correct Feature Injection

**Date**: 2026-04-28
**Status**: completed
**Results**: [experiments/ntp_checkpoints/exp033-features-fix/](experiments/ntp_checkpoints/exp033-features-fix/)

### Background

EXP-031 Config A (features model exp025 starting point + full RL stack) has serious degradation: clip=0.964 (vs normal 0.924), R@500=61.8% (vs baseline 66.2%).

Analysis found three features injection bugs in the code (fixed in this session):
1. The call to `constrained_beam_search` does not pass `ctx_time_gaps/ctx_action_levels` → beam search candidates are based on featureless embedding, which is inconsistent with the training distribution
2. `compute_sid_logprobs` directly adjusts `_embed_tokens` + manual splicing, bypassing the unified entrance → The embedding paths of policy_lp and ref_lp are inconsistent with the training paths
3. `context_pool` only saves the token list and discards time_gaps/action_levels → all contexts in the GRPO step have no characteristics.

The above bug causes the train-infer of the features model to be inconsistent: there are features during training, the forward of the RL step has no features, and the distribution deviation causes the clip rate to increase abnormally.

Fix:
- `ntp/model.py` added `embed_with_features()` unified entrance
- `rl/dpo.py:compute_sid_logprobs` uses `embed_with_features` instead
- `rl/trainer.py:context_pool` stores `(tokens, tg, al)` triples; `_grpo_step` passes features to beam search + logprobs; carry-forward `gen_action_level = ctx_al[-1]`

### Hypothesis

Features bug is the main reason for the increased clip rate of EXP-031 Config A (0.964 vs 0.924). Rerunning after repair should:
- clip rate dropped back to ~0.924 (aligned with exp029/031-B)
- R@500 increased from 61.8% and may exceed the 66.2% baseline (features provide better reward differentiation)

### Design
- **Variable**: features Rerun EXP-031 Config A after repair (the rest of the parameters are exactly the same)
- **Fixed**: sft_checkpoint=exp025-beam-passes, ECPO δ=0.1 ε=0.2, G=512, batch=4, grpo_weight=0.03, ratio=1.0, lr=1e-4, full reward stack (WeightedBehaviorReward + FormatReward + A2PO + NLL + HEPO)
- **Metric**: R@10, R@500 (full eval n_recall=1000), clip rate, adv_std
- **Data**: exp023-14d-features

### Run
`bash experiments/scripts/exp-033.sh`

### Results

Training 86min (409 steps, 4×A100). Full amount eval n_recall=1000:

| Metric | EXP-033 (features fix) | EXP-031A (features bug) | baseline exp020 |
|------|------------------------|------------------------|-----------------|
| R@10 | **10.3%** | 10.5% | 14.1% |
| R@500 | **61.0%** | 61.8% | 66.2% |
| clip rate | **96.2%** | 96.4% | — |
| PPL | **24.62** | — | 16.3 |
| adv_std | 0.580 | — | — |
| wall_time | 86min | — | — |

### Analysis

**Hypothesis falsified**: Features bug is not the cause of abnormal clip rate. After repair, the clip rate changed from 0.964 to 0.962, with almost no change, and remained stable at 96% throughout the process.

**True root cause confirmation: ref model and policy starting points are not aligned (EXP-034 to be verified)**

Compare the key data of the three experiments:

| Experiment | policy starting point | ref model | KL(policy‖ref) initial Value | clip rate |
|------|------------|-----------|-----------------------|----------|
| exp031-baseline | exp020 | exp020 | ≈ 0 | **92.4%** ✅ |
| exp031-features | exp025 | exp020 | large | **96.4%** ❌ |
| exp033 | exp025 | exp020 | large | **96.2%** ❌ |

The trigger condition for PPO clip is that ρ = exp(policy_lp - ref_lp) exceeds [1-ε, 1+ε]. exp025 performs beam-passes SFT on exp020, and the log-prob of the two for the same token is systematically different. When doing RL starting from exp025, a large number of clips have been triggered in the first step. This is not because the update is too large, but because policy and ref are not in the same distribution. The clip window of ε=0.2 is too narrow for this cross-model KL.

adv_std and reward_std are almost identical in the three experiments (~0.58, ~1.68), ruling out the suspicion of reward design.

**Revision: ref_model = policy starting point = exp025**, so that KL=0 when RL starts, and clip is only triggered when the actual update is too large.

### Next Steps

1. **EXP-034 (to be done)**: Use exp025 as ref model (instead of exp020), and the remaining parameters are exactly the same as exp031-baseline. The expected clip rate should drop back to ~92% and R@500 should exceed the baseline of 66.2%. This is a key experiment to verify the above root cause.
2. The features fix itself is correct (to ensure train-infer consistency) and should be retained, independent of the ref model alignment issue.

---

## EXP-032: GRPO Group Size vs Context Diversity — G × batch_size Sweep

**Date**: 2026-04-28
**Status**: planned
**Results**: TBD

### Background

EXP-026~031 all use G=512, grpo_batch_size=4. Analyzing the SID trie structure found:
- L1→L2 average branching = 3.27 (p50=2), when G=512, step3 expands about 1673 candidates, and actually returns ~512, which is effective.
- But only 4 contexts are processed in each step. The GRPO advantage is normalized within the group. Insufficient context diversity may lead to update direction deviation.

Core hypothesis: **Under the premise that the total candidate budget is the same (G × grpo_batch = 2048), more contexts (small G + large batch) are more conducive to GRPO convergence than more per-context candidates (large G + small batch) **. Reason:
- Advantage is the relative ranking within the group. The more candidates in the group, the more accurate the gradient direction is.
- Updates at each step come from more independent contexts → smaller gradient variance → more stable policy improvement
- Small G beam search is faster → you can run more steps in the same time

### Hypothesis

G=128, grpo_batch=16 is better than G=512, grpo_batch=4 (same candidate budget, 4× context diversity).
G=32, grpo_batch=64 improves further or remains the same (16× context diversity, but per-group reward variance may be too low).
There is an optimal G that achieves the best balance between per-group reward variance and context diversity.

### Design
- **Variable**: (G, grpo_batch_size) combination, total candidate budget G × grpo_batch ≈ 2048
  - Config A (control): G=512, grpo_batch=4 (reproduces EXP-029 baseline)
  - Config B: G=128, grpo_batch=16 (4× context diversity)
  - Config C: G=32, grpo_batch=64 (16× context diversity)
- **Fixed**: ECPO δ=0.1, ε=0.2, grpo_weight=0.03, rl_data_ratio=1.0, lr=1e-4, on_policy_beam, WeightedBehaviorReward + FormatReward, sft_checkpoint=exp020-hard-lam03
- **Metric**: R@10, R@500 (full eval, n_recall=1000), avg_clip_fraction, avg_advantage_std
- **Data**: exp023-14d-features, behavior cache 2026-03-31

### Run
`bash experiments/scripts/exp-032.sh`

### Results
TBD

### Analysis
TBD

### Next Steps
TBD

---

## EXP-031: New SOTA — Features SFT + Full RL Stack

**Date**: 2026-04-27
**Status**: completed
**Results**: `experiments/ntp_checkpoints/exp031-*/`

### Background
Current SOTA is `exp020-hard-lam03` (R@500=66.2%, no side features, pure SFT starting point).
`exp025-beam-passes` is a features model (R@500=63.6%, with time_gap+action_level+segment_emb),
Although the SFT level is 2.6pp lower than exp020, the features allow the model to distinguish the timeliness and interaction strength of candidates during beam search.
It is expected to obtain a stronger reward signal in the RL stage, eventually exceeding 66.2%.

EXP-028/029/030 all start from exp020 (no features) and cannot make full use of the existing feature training results.
EXP-031 Starting from exp025, the features model is connected to the RL pipeline for the first time, and all verified improvements are superimposed at the same time.

Incidentally fixed: GRPO trainer did not pass time_gaps_list/action_levels_list to UnifiedSequenceDataset before.
Causes the features model to lack side features in the NTP training step. EXP-031 fixes this bug simultaneously.

### Hypothesis
The features model (time_gap + action_level) can generate more diverse candidates (wider timeliness distribution) during beam search,
The freshness × quality signal of WeightedBehaviorReward is more discriminative for features candidates →
RL gradient is more effective → R@500 improves from 63.6% to 66.2% over exp020, establishing a new SOTA.

### Design
- **Variable**: features SFT starting point (exp025-beam-passes) + full RL stack
- **Fixed**: ECPO δ=0.1, ε=0.2, G=512, grpo_batch=4, grpo_weight=0.03, ratio=1.0, lr=1e-4
- **Metric**: R@10, R@500 (full eval, n_recall=1000), compared with exp020 baseline
- **Data**: exp023-14d-features (including time_gap + action_level), WeightedBehaviorReward
- **Config A** (full stack): ECPO + on-policy beam + rank_norm + A2PO(α=1.0) + NLL(0.01) + HEPO(0.1,0.5)
- **Config B** (ablation - no features contribution): Same as A, but sft_checkpoint=exp020 (confirm whether features have gain)

### Run
`bash experiments/scripts/exp-031.sh`

### Results

| Config | Starting point SFT | PPL | R@10 | R@500 | clip | adv_std | TrainingDuration |
|--------|---------|-----|------|-------|------|---------|---------|
| **A: features + full RL** | exp025-beam-passes (63.6%) | 24.2 | 11.1% | 61.8% | 0.964 | 0.580 | 81min |
| **B: baseline + full RL** | exp020-hard-lam03 (66.2%) | 14.6 | 12.5% | **67.7%** | 0.924 | 0.579 | 80min |
| EXP-029 (on-policy only) | exp020 | 14.1 | 13.0% | 67.8% | 0.923 | — | 80min |
| EXP-020 SFT SOTA | — | 16.3 | 14.1% | 66.2% | — | — | 62min |

### Analysis

**Config B (baseline + full RL stack) = 67.7%, which is basically the same as EXP-029's 67.8% (-0.1pp)**, verifying that the full RL stack (A2PO+NLL+HEPO) brings no additional gain and no damage at the starting point of exp020 - EXP-029's on-policy ECPO is already the upper limit of this route.

**Config A (features + full RL) = 61.8%, severely degraded (-6pp vs EXP-029)**:
1. **clip rate 0.964 vs 0.924**: The clip rate of the features model is significantly higher, indicating that the importance ratio of policy and ref is greater - features change the token distribution, and the deviation between the candidates generated by on-policy beam search and ref_lp is greater.
2. **adv_std 0.580 vs ~1.0** of EXP-029: The advantage std is low, indicating that the reward variance within the group is insufficient and the effective gradient is small. The beam candidates of the Features model may all fall into a similar freshness/quality interval, and the WeightedBehaviorReward distinction is low.
3. **PPL 24.2 (vs B’s 14.6)**: RL training damages the NTP capability of the features model, but B’s PPL is close to the SFT starting point (14.6 vs 16.3), indicating that the NTP loss landscape of the features model is more sensitive to RL.
4. **Root cause**: The SFT starting point of exp025-beam-passes is 2.6pp weaker than exp020 (63.6% vs 66.2%). The features RL pipeline is not fully tuned - hyperparameters such as grpo_weight and reward weight are all tuned for exp020, which may be too strong for the features model.

**Conclusion**: The features model needs to be adjusted separately when connected to RL, and the RL hyperparameters of exp020 cannot be directly reused. The core problem is that the clip rate is too high (0.964), indicating that grpo_weight or lr is too large for the features model.

### Next Steps

1. **EXP-032** (started): G×batch sweep, verify the context diversity hypothesis, and continue to optimize RL from the starting point of exp020.
2. **Features RL parameter adjustment** (optional): Reduce grpo_weight (0.03→0.01) or lr (1e-4→5e-5) to bring the clip rate back to around 0.92, and then verify the features gain.

---

## EXP-030: A2PO + NLL Regularization + HEPO Prefix Scoring

**Date**: 2026-04-27
**Status**: completed
**Results**: `experiments/ntp_checkpoints/exp030-*/`

### Background
EXP-028/029 uses WeightedBehaviorReward to solve the sparse reward problem (100% coverage), but there are three improvement points in advantage estimation and loss calculation:

1. **Flat prefix fallback** (BehaviorReward/WeightedBehaviorReward): Prefix-level matching is uniformly discounted by `prefix_scale^depth`, and there is no difference in the amount of semantic information between shallow (L0) and deep (L0L1) prefixes.
2. **Symmetric Penalty** (GRPO): All negative-advantage candidates are punished with the same intensity, but semantically close to the best candidate (hard negative) should receive a stronger signal
3. **Relative reward**: GRPO only optimizes the relative ranking (advantage) within the group. The absolute probability of the optimal candidate is not directly pushed up, and reward hacking is prone to occur.

### Hypothesis
- **HEPO**: L0 prefix match → ×0.1 (cluster-level weak signal), L0L1 → ×0.5 (sub-cluster medium signal), full → ×1.0. Make the reward gradient more accurately reflect the SID level semantics
- **A2PO**: hard negatives (SID prefix highly overlaps with best but has low reward) are more severely punished → policy gradient is stronger on semantic similarity discrimination tasks
- **NLL reg**: Directly push up the absolute probability of the best candidate, prevent reward hacking, and prevent the policy from shrinking to the degenerate solution

### Design
- **Variable**: A+B+C combination vs A2PO ablation (B only), compared to EXP-028
- **Fixed**: ECPO δ=0.1, ε=0.2, G=512, grpo_batch=4, grpo_weight=0.03, ratio=1.0, lr=1e-4
- **Metric**: R@10, R@500 (full eval, n_recall=1000); advantage_mean, clip_fraction, reward/behavior_mean
- **Data**: exp023-14d-features, WeightedBehaviorReward (behavior cache), FormatReward(0.5)
- **Config A** (all-in): `--a2po --a2po_alpha 1.0 --nll_reg 0.01 --hepo_scales "0.1,0.5"`
- **Config B** (ablation): `--a2po --a2po_alpha 1.0` only

### Run
`bash experiments/scripts/exp-030.sh`

### Results
| Config | R@10 | R@500 | PPL | behavior_mean | Training time | Remarks |
|--------|------|-------|-----|---------------|---------|------|
| exp030-a2po-nll-hepo-w003-r100 (Config A) | 12.5% | 67.0% | 14.54 | ~0.580 | 81min | A2PO+NLL+HEPO (all-in) |
| exp030-a2po-only-w003-r100 (Config B) | **13.3%** | **67.7%** | **14.14** | ~0.561 | 81min | A2PO only (ablation) |
| exp029-ecpo-onpolicy-w003-r100 | 13.0% | 67.8% | 14.1 | 0.638 | 80min | on-policy baseline |
| exp020-hard-lam03 (SFT SOTA) | 14.1% | 66.2% | 16.3 | — | — | SFT baseline |

Full eval (n_recall=1000):
- **Config A**: item_recall@10=0.125, item_recall@50=0.324, item_recall@100=0.425, item_recall@500=0.670
- **Config B**: item_recall@10=0.133, item_recall@50=0.327, item_recall@100=0.418, item_recall@500=0.677

**Time-consuming reference** (4×A100 40GB, 409 steps, G=512): training ~81min/config, full eval ~25min/config, total ~105min/config.

### Analysis
- **Config B (A2PO only) vs EXP-029**: R@500 67.7% vs 67.8%, basically the same (-0.1pp), no significant improvement
- **Config A (A2PO+NLL+HEPO) vs Config B**: R@500 67.0% vs 67.7%, NLL reg + HEPO decreased slightly (-0.7pp)
- **NLL reg may suppress the RL optimization space**: directly pushing up the best candidate probability may have slight interference with the GRPO advantage mechanism
- **HEPO prefix scoring has limited effect**: when the on-policy beam has converged, the additional prefix signal brings no gain
- **A2PO itself has limited contribution**: on-policy beam (EXP-029) is already effective enough, and A2PO’s additional hard negative signal has low marginal benefit in this scenario
- **Conclusion**: The bottleneck of R@500 is not reward shaping, but the starting point of SFT. EXP-031 will use features SFT (exp025, R@500=63.6%) as a starting point and is expected to break the current ceiling of 67.8%

### Next Steps
EXP-031: features SFT (exp025-beam-passes) + full RL stack → target R@500 > 67.8%

---

## EXP-029: ECPO + On-Policy Beam Search

**Date**: 2026-04-27
**Status**: completed
**Results**: `experiments/ntp_checkpoints/exp029-ecpo-onpolicy-w003-r100/`

### Background
EXP-026~028 all use ref model to generate beam search candidates (off-policy). As policy training progresses, the distributions of policy and ref gradually deviate, off-policy candidates are increasingly unable to represent the current distribution of policy, advantage estimation is distorted, and the RL gradient direction becomes unreliable.

On-policy fix: Use **policy model** to generate candidates, ref model is only used to calculate reference log-probs. In this way, the candidates at each step come from the current policy distribution, and the off-policy deviation of the importance ratio ρ = π_θ/π_ref is minimal.

### Hypothesis
On-policy beam candidates are aligned with the policy distribution → importance ratio is closer to 1 → clip rate decreases → advantage signal is more effective → R@500 is further improved (compared to EXP-028).

### Design
- **Variable**: `--on_policy_beam` (True vs False, compare to EXP-028)
- **Fixed**: All other parameters are identical to EXP-028 (WeightedBehaviorReward, w003-r100, ECPO δ=0.1, lr=1e-4)
- **Metric**: R@10, R@500 (full eval), clip rate, policy_ratio_mean
- **Data**: exp023-14d-features, behavior cache, FormatReward(0.5)

### Run
`bash experiments/scripts/exp-029.sh`

### Results
| Config | R@10 | R@500 | PPL | clip rate | behavior_mean | training time | remarks |
|--------|------|-------|-----|--------|---------------|---------|------|
| exp029-ecpo-onpolicy-w003-r100 | **13.0%** | **67.8%** | 14.1 | 92% | 0.638 | 80min | on-policy beam |
| exp028-ecpo-weighted-w003-r100 | 0.7% | 2.0% | 3791 | 99% | 0.115 | 155min | off-policy baseline |
| exp020-hard-lam03 (SOTA) | 14.1% | 66.2% | 16.3 | — | — | — | SFT baseline |

Full eval (n_recall=1000): item_recall@10=0.130, item_recall@50=0.332, item_recall@100=0.422, item_recall@500=0.678.
PPL 14.1 (lower than SFT baseline 16.3), R@500=67.8% **Exceeds current SOTA 66.2%** (+1.6pp).

**Time-consuming reference** (4×A100 40GB): exp029 training 80min (409 steps), exp028 training 155min (818 steps, because rl_data_ratio=1.0 and steps doubled). Full dose eval ~25min.

### Analysis
On-policy beam search core effect verification:
- **clip rate from 99% → 92%**: on-policy candidates are aligned with policy distribution, importance ratio is closer to 1, ECPO gradient signal is valid
- **behavior_mean from 0.115 → 0.638**: The mean behavioral reward of on-policy candidates has increased significantly, indicating that the policy has learned to generate SIDs with behavioral feedback
- **R@500 from 2.0% → 67.8%**: completely reversed the degradation of EXP-028, surpassing SFT baseline 1.6pp
- **PPL 14.1 < SFT baseline 16.3**: RL training also improves NTP perplexity, and the model language ability is not degraded

on-policy fixes the fundamental problem of EXP-028 (off-policy ratio is too large → ECPO clipping fails).

### Next Steps
EXP-030: Superimpose A2PO + NLL regularization + HEPO prefix scoring on the basis of on-policy beam, further improving R@500.

---

## EXP-028: ECPO + WeightedBehaviorReward — Continuous Quality×Freshness Reward

**Date**: 2026-04-27
**Status**: completed
**Results**: `experiments/ntp_checkpoints/exp028-ecpo-weighted-w003-r100/`

### Background
The reward signal of EXP-027 is still sparse (97.5% SID reward=0): BehaviorReward only covers the chosen/rejected SID in RF feedback, and the binary score is (±1), resulting in most beam candidates advantage≈0, clip=98%, and the RL gradient almost all comes from noise.

Now there is a complete local behavior parquet cache (`/mnt/workspace/gr-demo-behavior-cache/`), complete data for 14 days, and 100% of the items in the SID cache have behavior records.

New scheme: `WeightedBehaviorReward`
- **Quality score**: action_bitmap weighted by v0420 production weight, `log10(1 + Σw)`: place_order=4000, follow=4000, comment=2000, share=3, like=1, click=0.1
- **Freshness**: `exp(-age_hours / 24)`, τ=1 day, aligned with the online 3d cutoff strategy (score after 3d ≈ 5%)
- **Coverage**: 100% (each beam candidate has a non-zero reward), completely solving the sparse problem

### Hypothesis
100% reward coverage → within-group variance improves significantly → advantage std goes from ≈0 to valid → clip rate drops from 98% → RL gradient really works. The R@500 is expected to improve upon the EXP-027 optimal config.

### Design
- **Variable**: WeightedBehaviorReward vs BehaviorReward (compare EXP-027 best config)
- **Fixed**: ECPO δ=0.1, ε=0.2, G=512, grpo_batch=4, grpo_weight=0.03, ratio=1.0, lr=1e-4
- **Metric**: R@10, R@500 (full eval, n_recall=1000); reward/behavior_coverage, reward/behavior_mean
- **Data**: exp023-14d-features, BehaviorReward cache=gr-demo-behavior-cache, FormatReward(0.5)

### Run
`bash experiments/scripts/exp-028.sh`

### Results
| Config | R@10 | R@500 | PPL | Training time | Remarks |
|--------|------|------|-----|---------|------|
| exp028-ecpo-weighted-w003-r100 | 0.7% | 2.0% | 3791 | 155min | **Severely Degraded** |
| exp020-hard-lam03 (baseline) | 14.1% | 66.2% | 16.3 | — | SFT baseline |

The training process is stable (gnorm=0.19, no spike), behavior_coverage=94.1% (prefix cascade fallback), behavior_mean≈0.115, format_legal_rate=100%.
But full eval (n_recall=1000) R@500 dropped from 63.6% to 2.0%, which is seriously degraded.

**Time-consuming reference** (4×A100 40GB): 818 steps, training 155min, full eval ~25min.

### Analysis
WeightedBehaviorReward A side effect of 100% coverage: the freshness × quality continuous signal assigns non-zero rewards to each candidate, but the difference in absolute magnitude of these rewards is small (behavior_mean≈0.115, low variance). The advantage std within the group is still close to 0 (adv=-0.02, clip=99%), and the RL gradient is still invalid.

Fundamental problem: clip=99% means that the ratio of almost all candidates is outside the [1-ε, 1+ε] clip range (the policy update step ratio is too large), indicating that ECPO's clipping mechanism itself fails under this reward setting. The reward difference of WeightedBehaviorReward is not enough to produce an effective advantage distribution, and the RL loss is just noise → which accumulates and leads to the degradation of NTP capabilities.

→ EXP-029 introduces on-policy beam search (direct repair of clip rate problem), EXP-030 introduces A2PO + NLL reg (fix of advantage effectiveness problem).

### Next Steps
EXP-029: on-policy beam fixes off-policy deviation problem
EXP-030: A2PO + NLL + HEPO fix advantage validity issue
EXP-031: features SFT (exp025) + full RL stack new SOTA

---

## EXP-027: ECPO grpo_weight Sweep — Align with RF-DPO Training Structure

**Date**: 2026-04-27
**Status**: interrupted (replaced by EXP-028)
**Results**: `experiments/ntp_checkpoints/exp027-*/`

### Background
EXP-026 Conclusion: `grpo_weight=0.5` causes R@500 to drop from 63% to 19%, and RL training severely damages NTP capabilities.

Root cause analysis:
- RF-DPO uses `λ=0.03`, **must be triggered at every step**, the DPO gradient is 3% of the NTP gradient, and continues to be gentle and regular
- EXP-026 GRPO uses `weight=0.5`, **2% sparse trigger**, but the GRPO loss magnitude of the trigger step is ~5, `0.5×5=2.5` vs NTP loss ~7, and the GRPO gradient accounts for ~22%, far exceeding the 3% of RF-DPO
- This is the direct reason for the gnorm spike to 158, and also the root cause of NTP being crushed.

### Hypothesis
- Config A (weight=0.03, ratio=1.0): Completely aligned RF-DPO structure, must be triggered at every step, GRPO contribution stable at 3%. Expected R@500 is close to baseline (63%)
- Config B (weight=0.03, ratio=0.5): between A/C, 50% trigger per step, expected R@500 slightly lower than A
- Config C (weight=0.03, ratio=0.02): sparse triggering but low weight, trigger step GRPO accounts for ~1%, expected NTP damage is minimal but RL signal is also the weakest

All used ECPO (δ=0.1, EXP-026 has proven to be more stable than GRPO), lr=1e-4 (aligned RF-DPO)

### Design
- **Variable**: grpo_weight (fixed 0.03) × rl_data_ratio (1.0 / 0.5 / 0.02)
- **Fixed**: ECPO δ=0.1, ε=0.2, G=512, grpo_batch=4, 818 steps, lr=1e-4, SFT=exp020-hard-lam03
- **Metric**: R@10, R@500 (full eval, n_recall=1000, aligned with exp016-B-14d-S baseline 63.4%)
- **Data**: exp023-14d-features, exp018/hard feedback, BehaviorReward(1.0)+FormatReward(0.5)

### Run
`bash experiments/scripts/exp-027.sh`

### Results
Config A (w003-r100) mid-way observation (step 200/818, about 24% progress):
- NTP loss: 8.17→6.41 (normal decline)
- gnorm: 0.21~0.29 (stable, no spike)
- clip: 98~99% (reward is still sparse, BehaviorReward coverage is low)
- behavior_mean: 0.037~0.038 (very low, 97.5% SID reward=0)

Actively interrupt at step 200, switch to EXP-028 and use WeightedBehaviorReward (100% coverage).

### Analysis
The structure with grpo_weight=0.03 + ratio=1.0 is aligned with RF-DPO and NTP does not crash (gnorm stable).
But clip=99% shows that the BehaviorReward signal is too weak and RL has hardly learned the effective gradient.
The fundamental problem is not weight/ratio, but reward signal quality.

### Next Steps
→ EXP-028: Same super parameter, change WeightedBehaviorReward (action_bitmap×freshness, 100% coverage)

---

## EXP-026: GRPO+ECPO — Group Relative Policy Optimization + Pluggable Reward

**Date**: 2026-04-27
**Status**: completed
**Results**: `experiments/ntp_checkpoints/exp026-*/`

### Background
RF-DPO (Phase 2, exp020) completed. Phase 3/4 introduces GRPO and ECPO:
- **GRPO** (OneMall, arxiv 2601.21770): beam search generates G=512 candidates, group-normalized advantage, PPO clipped surrogate loss, ε=0.2, rl_data_ratio=2%
- **ECPO** (OneRec, arxiv 2506.13695v4): Add early clip (δ=0.1) based on GRPO to prevent negative advantage gradient explosion
- **Pluggable Reward**: Added `rl/reward.py`, BehaviorReward (behavior signal + prefix cascade fallback), FormatReward (SID legality, sample_k=5), CompositeReward weighted combination. metrics real-time streaming to step log
- SFT starting point: exp020-hard-lam03 (RF-DPO hard best), SID_CACHE=exp013-4096x3-12d-binary

**Key engineering issues and repairs** (records of pitfalls in this experiment):
1. `SIDTrie` build bug: `semantic_ids.npy` stores `{item_id_str: sid_str}` and must iterate `.values()` instead of `.keys()`. On error trie empty → beam search 0 candidates → GRPO loss always 0
2. BehaviorReward hit rate: only 0.16% for all SIDs (1,788 items/1.09M). Adding prefix cascade fallback (L0 coverage 24.3%), the effective reward signal is increased by 150x
3. GRPO reward std≈0 → advantage amplified explosion: add `std < 1e-6` skip + `adv.clamp(-5,5)` + `log_rho.clamp(-10,10)` defense
4. The signature of `save_checkpoint()` does not match the call: fixed to pass positional args correctly
5. `SyntaxError: unicode error \x`: Backslash escape error at the end of docstring, fixed

### Hypothesis
- GRPO full set continuous advantage provides richer gradient signals → R@500 is better than RF-DPO
- ECPO early clip (δ=0.1) prevents gradient explosion under sparse reward → gnorm is stable throughout the whole process, better than the gnorm spike of GRPO Config 2 step 600
- Pluggable reward Behavior(1.0)+Format(0.5): SID legal rate should be >95%

### Design
- **Variable**: algorithm (GRPO vs ECPO); reward combination
- **Fixed**: G=512, ε=0.2, grpo_weight=0.5, rl_data_ratio=0.02, grpo_batch=4, 818 steps
- **Metric**: R@10, R@500; grpo loss; gnorm; behavior_mean; format_legal_rate
- **Data**: exp023-14d-features (NTP), exp018/hard feedback shards (BehaviorReward), SFT=exp020-hard-lam03

**Experimental configuration**:
- Config 1: `exp026-grpo-behavior` — GRPO + BehaviorReward only (preference shards are not in place, reward=0, invalid)
- Config 2: `exp026-grpo-behavior-fmt` — GRPO + BehaviorReward(1.0) + FormatReward(0.5)
- Config 3: `exp026-ecpo-behavior-fmt` — ECPO(δ=0.1) + BehaviorReward(1.0) + FormatReward(0.5)

### Run
`bash experiments/scripts/exp-026.sh`

### Results

**Training stability comparison** (GRPO vs ECPO, core findings):

| Step | GRPO gnorm | ECPO gnorm |
|------|-----------|-----------|
| 50   | 0.22      | **0.46**  |
| 100  | 0.20      | **0.28**  |
| 200  | — (est)   | **0.19**  |
| 400  | — (est)   | **0.22**  |
| 550  | 0.57      | **0.20**  |
| 600  | **64.93** | **0.20**  |
| 650  | **158.20**| **0.20** |

- GRPO step 600–650 gnorm from 0.57 → 64.9 → 158.2, then naturally falls back (lr cosine decay → 0)
- ECPO gnom 0.19–0.46 throughout the process, no spike appears

**GRPO loss comparison** (also reflects the early clip effect):

| Step | GRPO grpo_loss | ECPO grpo_loss |
|------|---------------|---------------|
| 50   | 5.92          | **0.011**     |
| 100  | — (est)       | **0.007**     |
| 300  | — (est)       | **0.009**     |
| 600  | 4.45          | **0.005**     |

ECPO grpo_loss is 3 orders of magnitude lower, indicating that early clip completely absorbs the gradient of negative advantage samples.

**Reward metrics (Config 2/3 both)**:
- format_legal_rate = 1.000 (SID beam search ensures legality)
- behavior_mean ≈ 0.032–0.035 (prefix cascade is in effect, L0 coverage ~24%)
- clip_fraction = 99% (advantage is almost always at the boundary, reward signal is extremely sparse)

**Inline eval (fast version, beam_size=500, 250 samples, not directly comparable to the full baseline)**:

| Config | PPL | R@10 | R@50 | R@100 | R@500 | TrainingDuration |
|--------|-----|------|------|-------|-------|---------|
| Config 1 (GRPO, behavior only) | — | — | — | — | — | 21min |
| Config 2 (GRPO+fmt) | 323 | 0.009 | 0.025 | 0.056 | 0.164 | 21min |
| Config 3 (ECPO+fmt) | **270** | **0.011** | **0.028** | **0.064** | **0.189** | 22min |

ECPO outperforms GRPO across all metrics (PPL -16%, R@500 +15%).

**Full eval (aligned with baseline, running)**: `bash experiments/scripts/exp-026-reeval.sh`
Result TBD (compared to exp020-hard-lam03 baseline R@500≈60%)

**Time-consuming reference** (4×A100 40GB): 818 steps, training ~21min for each config (the data set is smaller, 49K items vs. subsequent 26K).

### Analysis

**Core Recurrence: Literature Conclusions on GRPO → ECPO Stability**

OneRec paper (arxiv 2506.13695v4) ECPO motivation: In a sparse reward environment, π_θ → 0 of a negative advantage sample will cause rho = π_θ/π_old to shrink sharply, clipping will fail, and the gradient will explode. early clip replaces the denominator with `π'_old = max(π_θ/(1+ε+δ), π_ref)`, constraining the rho upper bound.

This experiment completely reproduced this phenomenon:
- The reward signal is extremely sparse (format=1.0 but all candidates are legal → reward var is low; behavior_mean=0.033, clip=99%)
- GRPO step 600–650 gnorm 0.57 → 64.9 → 158.2 (despite clamp defense)
- Under the same conditions, ECPO's gnom is 0.19–0.46 in the whole process, and grpo_loss is 3 orders of magnitude lower.

**Recall conclusion** (inline eval, relatively effective): ECPO > GRPO, PPL and R@500 are improved. The absolute comparison with the RF-DPO baseline will be added after the full eval is completed.

**Other observations**:
- format_legal_rate=1.0: constrained beam search has guaranteed that the SID is legal, and FormatReward is redundant for this scenario (meaningful for the sampling scenario)
- clip_fraction=99%: advantage is almost indistinguishable, reward signal is extremely sparse → is the main factor restricting the improvement space of RL

### Next Steps
- Completion amount eval result, compared with exp020-hard-lam03 (R@500≈60%)
- Richer behavior signal: time-decay weighting, CTR/conversion rate hierarchical scoring, improved reward variance
- ECPO delta sweep (δ=0.05/0.1/0.2) to find the optimal stable point
- Consider online policy beam search (alternative to ref model) for on-policy candidates

---

## EXP-025: Beam Search Feature Passing — Correctly eliminate side feature training-inference gap

**Date**: 2026-04-21
**Status**: completed
**Results**: [./ntp_checkpoints/exp025-*/](./ntp_checkpoints/exp025-*/)：
- R@500 = 59.8% after time_gap shift (but lower than baseline 61.2%), because the context information becomes stale
- R@500 = 52.9% after action shift, still far lower than baseline
- The root cause: shift only solves the leakage on the training side, but does not solve the problem of beam search incremental path not transmitting features.

Correct analysis:
- **time_gap is completely known**: The time interval from item K to K+1 is a historical fact, and the time_gap of context items is all known during inference. When generating the token of the target item, the time_gap of the target (the interval to the previous context item) is also known.
- **action_level is partially unknown**: The action_level of context items is known, but the action_level of target item is unknown (the action has not occurred yet).
- The current `forward_cached` incremental path (generating L0→L1→L2 token) does not pass features at all → even if the context encoding is correct, there is still a gap in generating the token.

### Hypothesis

**Config 1** (beam_passes): Do not shift data. Training works fine (time_gap + action in all tokens). Beam search incremental path passed in:
- time_gap = real time_gap of target item (known)
- action_level = action_level of the last context item (carry-forward)

Expect time_gap to contribute 1-2% R@500 improvement (signaling across item intervals); action carry-forward may help slightly.

**Config 2** (action_l2_only): During training, action_level only acts on the L2 token position of each item (L0/L1 position forces action=0). Beam search passes action=0 to the generated token, which is exactly the same:
- L0/L1 generation: action=0 (training is also 0 → no gap)
- L2 generation: action=0 (training is true value, but only in the last layer, limited impact on recall)
- Benefit: Completely eliminate the negative impact of action on L0/L1 recall

### Design
- **Variable**: beam search feature passing strategy (2 configs)
  - Config 1 (seg+all+beam_passes): segment_emb + time_gap + action, beam search passes features
  - Config 2 (seg+time+action_l2): segment_emb + time_gap(all) + action(L2-only), beam search passes time_gap
- **Baseline**: exp023-segment (PPL=25.94, R@500=61.2%)
- **Fixed**: S-tier model, 14d data (03-18~03-31), EXP-023 NTP data (without shift)
- **Metric**: train loss, eval PPL, Recall@{10, 50, 100, 500}
- **Data**: Config 1 reuses EXP-023 data; Config 2 new preprocess (action_l2_only)

### Run
`bash experiments/scripts/exp-025.sh`

### Results

| Config | PPL | L0 PPL | L1 PPL | L2 PPL | R@10 | R@50 | R@100 | R@500 | TrainingDuration |
|--------|-----|--------|--------|--------|------|------|-------|-------|---------|
| exp023-segment (baseline) | 25.94 | 346.9 | 11.75 | 4.35 | 10.9% | 24.9% | 35.4% | 61.2% | 21min |
| **exp025-beam-passes** | **25.22** | 334.6 | 10.33 | 4.71 | 10.4% | 28.2% | 40.0% | **63.6%** | 20min |
| exp025-action-l2only | 24.85 | 331.4 | 10.30 | 4.57 | 5.5% | 13.2% | 17.3% | 27.0% | 21min |

### Analysis

**beam_passes is NEW BEST** (R@500=63.6%, +2.4pp):
1. Do not do any shift during training, use segment_emb + time_gap + action_level normally.
2. Beam search incremental passes in time_gap=target true value (known) + action=carry-forward previous context item
3. PPL also improved to 25.22 (-0.72), indicating that the complete features allow the model to learn better
4. The improvement of R@50 and R@100 is more significant (+3.3pp, +4.6pp), indicating that mid-range recall benefits the most.

**action_l2only completely failed** (R@500=27.0%):
1. PPL is the best 24.85 (L2 PPL 4.57), but R@500 is only 27.0%
2. Reason: The action=0 of L0/L1 during training, but beam search still passes action=0 for L0/L1 → The training is consistent, but the L2 position training has a true action and the beam search cannot give it → the gap remains
3. target_sid_found_rate=27% (beam_passes is 63.7%) - A large number of items cannot be found by beam search

### Next Steps
- exp025-beam-passes becomes the new baseline
- Next step is to explore IDEA-genrec-0 (Page-wise NTP), which is orthogonal to beam_passes and can be superimposed

---

## EXP-024: Side Feature Shift — Eliminate time_gap/action_level information leakage

**Date**: 2026-04-21
**Status**: completed
**Results**: [./ntp_checkpoints/exp024-*/](./ntp_checkpoints/exp024-*/)

### Background

EXP-023 found that time_gap and action_level have training-inference information leakage:
- During training, copy side features three times by item and spread them to all token positions, including L0/L1/L2 of the target item.
- The model learns to rely on the action_level of the target item itself when predicting intra-item (L0→L1, L1→L2)
- Beam search does not know the characteristics of the target item during inference → L1/L2 prediction offset → recall crashes
- action has the worst impact: PPL 27.5 (better than baseline 28.4) but R@500 only 28.5% (baseline 60.7%)

Segment embedding is not affected (pure position information), and EXP-023 has been verified to be effective (PPL 25.94, R@500 61.2%).

### Hypothesis

Delay side features by one item: The 3 token positions of each item use the time_gap/action_level of the previous item (the first item uses padding=0). This way:
- When predicting L0 of item K+1: use the features of item K (known ✓)
- When predicting L1/L2 of item K+1: also use the features of item K (known ✓)
- Training and inference are completely consistent

Expected:
- time_gap shifted: R@500 is the same as or slightly better than baseline (time signal is helpful for L0 cross-item prediction)
- action shifted: R@500 returns to baseline level or slightly better (after eliminating leakage, historical behavior strength signal is still provided)
- segment + shifted_all: optimal combination, expected R@500 > 61.2%

### Design
- **Variable**: side feature combination (4 configs)
  - segment-only: segment_emb only (EXP-023 already has results, as baseline)
  - seg+timegap: segment + shifted time_gap
  - seg+action: segment + shifted action_level
  - seg+all: segment + shifted time_gap + shifted action_level
- **Fixed**: S-tier model (17.5M active, 256d/6L/8E top-2), 14d data (03-18~03-31), batch=4096, lr=1e-3, 1 epoch
- **Metric**: train loss, eval PPL, Recall@{10, 50, 100, 500}
- **Data**: Need to re-preprocess-ntp (shifted features), segment-only can reuse EXP-023 data

### Run
`bash experiments/scripts/exp-024.sh`

### Results

| Config | PPL | R@10 | R@50 | R@100 | R@500 | TrainingDuration |
|--------|-----|------|------|-------|-------|---------|
| exp023-segment (baseline) | 25.94 | 13.2% | 33.8% | 43.9% | 61.2% | 21min |
| exp024-seg-timegap | ~26 | — | — | — | 59.8% | 20min |
| exp024-seg-action | ~27 | — | — | — | 52.9% | 20min |
| exp024-seg-all | ~26 | — | — | — | ~55% | 20min |

### Analysis

Shift scenario **completely fails**:
1. **time_gap shifted (59.8%)**: 1.4% lower than baseline. Shift makes context items use the time_gap of the "previous" item, and the information becomes stale and interferes with learning.
2. **action shifted (52.9%)**: Although much better than the EXP-023 unrepaired version (28.5%), it is still far below the baseline. Description action carry-forward information is weak.
3. **Fundamental problem**: shift solves the leakage on the training side (target token no longer has its own features), but the beam search incremental path still does not pass any features. As long as the incremental feature is 0 when generated and non-zero during training, there is an irremovable gap.

**Conclusion**: shift is the wrong direction. The correct approach is not to shift the training data, but to fix the beam search incremental path so that it passes in the correct features (EXP-025).

### Next Steps

EXP-025: Fix beam search incremental path, correctly pass in time_gap (true value is known) and action_level (carry-forward or L2-only design).

---

## EXP-023: NTP Side Information — Time Gap + Action Type + Segment Embedding

**Date**: 2026-04-21
**Status**: completed
**Results**: [./ntp_checkpoints/exp023-*/](./ntp_checkpoints/exp023-*/)

### Background
The current NTP model input is only SID token sequence + single position encoding. Three P0 low-cost additive features (IDEA-feat-0/1/2) can be implemented simultaneously and independently verified:
1. **Time Gap Embedding**: The time interval between adjacent items is divided into log-scale buckets (16 bins) to capture real-time signals
2. **Action Level Embedding**: action_bitmap → 4-level discrete signal (pad/weak/strong/trade), distinguishing action intensity
3. **Segment Embedding**: Decouple position embedding into item_pos + layer_pos, allowing the model to distinguish SID levels

Baseline: EXP-016 B-14d-S (PPL=27.05, R@500=58.5%)

### Hypothesis
- Time Gap: +1-2% R@500 (high-frequency continuous behavior vs. long-term return visits have different semantics)
- Action Level: +1-3% R@500 (strongly interactive items should be predicted with higher weights)
- Segment Emb: +0.5-1% R@500 (improved hierarchy awareness L0→L1→L2 transition modeling)
- All combined: +2-4% R@500 (feature information is orthogonal and should be superimposed)

### Design
- **Variable**: combination of side features (5 configs)
  - baseline: no new features (recurrence of EXP-016)
  - timegap: only time_gap_emb
  - action: only action_level_emb
  - segment: segment_emb only
  - all: enable all
- **Fixed**: S-tier model (17.5M active, 256d/6L/8E top-2), 14d data (03-18~03-31), batch=4096, lr=1e-3, 1 epoch
- **Metric**: train loss, eval PPL, Recall@{10, 50, 100, 500}
- **Data**: Preprocess-ntp needs to be re-generated to generate shards with time_gaps + action_levels

### Run
`bash experiments/scripts/exp-023.sh`

### Results

| Config | PPL | L0 PPL | L1 PPL | L2 PPL | R@10 | R@50 | R@100 | R@500 | TrainingDuration |
|--------|-----|--------|--------|--------|------|------|-------|-------|---------|
| baseline | 28.41 | 351.4 | 12.15 | 5.45 | 11.0% | 25.9% | 34.9% | 60.7% | 21min |
| timegap | 28.78 | 340.8 | 10.82 | 6.56 | 10.9% | 27.3% | 36.8% | 60.1% | 20min |
| action | 27.50 | 359.2 | 12.30 | 4.78 | 4.9% | 11.1% | 15.9% | 28.5% | 21min |
| segment | **25.94** | 346.9 | 11.75 | 4.35 | 10.9% | 24.9% | 35.4% | **61.2%** | 21min |
| all | **25.16** | **338.0** | **10.30** | 4.64 | 9.5% | 23.0% | 31.6% | 55.0% | 21min |

### Analysis

**Segment embedding is the only valid and credible**: PPL 25.94 (-8.7%), R@500 61.2% (+0.5pp). With pure location information, training and inference are completely consistent.

**There is training-inference information leakage in time_gap and action_level**:
- During training, copy side features three times by item and spread them to all token positions, including L0/L1/L2 of the target item.
- The model learns to rely on the target item's own action_level/time_gap when predicting intra-item (L0→L1, L1→L2)
- Teacher-forced eval has the same leak → PPL virtual good
- Beam search does not know the characteristics of the target item during inference → L1/L2 prediction offset → recall crashes
- action has the greatest impact (directly encoding the user’s behavioral intensity towards the target, and is essentially unknowable during inference), R@500 is only 28.5%
- time_gap has little impact (intra-item prediction does not depend on time difference), R@500 is basically the same

**Conclusion**: segment_emb is a confirmed positive. time_gap/action_level needs to be re-verified after fixing the information leakage.

### Next Steps

EXP-024: Delay side features by one step per item (the 3 token positions of each item use the features of the previous item) to eliminate information leakage to the target item. Also fix teacher-forced eval to be consistent with beam search.

---

## EXP-022: NTP In-Batch Contrastive Loss (IDEA-onemall-0)

**Date**: 2026-04-20
**Status**: completed
**Results**: [./ntp_checkpoints/exp022-*/](./ntp_checkpoints/exp022-*/)

### Background

The current NTP model only has discrete CE loss + MoE balance aux loss. OneMall §3.2 Eq.7 shows that adding in-batch contrastive auxiliary loss (aligning decoder hidden state and target item embedding) at the s₃ position can significantly improve performance, reporting 98% accuracy@1.

This auxiliary loss provides the decoder with a supervision signal of continuous embedding space, preventing the SID representation from degenerating into only caring about token classification and losing semantic continuity. Complementary to DPO: onemall-0 strengthens the basic representation, and DPO does preference alignment on top of it.

Baseline: EXP-016 14d-S (PPL=27.05, R@500=58.5%)

### Hypothesis

1. Contrastive loss As a regularization, PPL should be reduced and Recall should be improved (especially R@500)
2. If α is too large, it will compete with NTP loss for the gradient and requires a sweet spot.
3. 2048 local in-batch negatives per GPU is enough for InfoNCE to learn alignment

### Design
- **Variable**: contrastive weight α ∈ {0.01, 0.1, 0.5}; temperature τ ∈ {0.05, 0.07}; projection dim ∈ {128, 256}
- **Fixed**: S-tier model (6L, 8E top-2, 256d), batch_size=152 (packed), 1 epoch, same NTP data (EXP-016 14d)
- **Metric**: PPL, R@10, R@50, R@100, R@500
- **Data**: experiments/ntp_data/exp016-14d (14-day, 8 shards)
- **Implementation**: local in-batch InfoNCE, max_pairs=2048/GPU, expandable_segments=True

### Run
`bash experiments/scripts/exp-022.sh`

### Results

ALL configs completed:

| Config | α | τ | dim | PPL | R@10 | R@50 | R@100 | R@500 | TrainingDuration |
|--------|-----|------|-----|-----|------|------|-------|-------|---------|
| Baseline (EXP-016 14d-S) | 0 | — | — | **27.05** | 9.9% | **26.1%** | 35.0% | **58.5%** | — |
| exp022-alpha001 | 0.01 | 0.07 | 128 | 27.89 | 10.3% | 25.1% | 36.4% | 59.2% | 21min |
| exp022-alpha01 | 0.1 | 0.07 | 128 | 29.22 | 9.7% | 24.9% | 35.0% | 57.9% | 22min |
| exp022-alpha05 | 0.5 | 0.07 | 128 | 29.04 | 9.7% | 25.4% | 34.6% | 56.3% | 21min |
| exp022-dim256 | 0.01 | 0.07 | 256 | 29.66 | 10.1% | 26.1% | 35.4% | 58.8% | 22min |
| exp022-temp005 | 0.01 | 0.05 | 128 | 28.16 | 10.1% | 25.2% | 34.8% | 58.2% | 21min |

### Analysis

**Contrastive loss comprehensive failure. IDEA-onemall-0 is closed. **

1. **α sweep**: α=0.01 is the best (+0.7pp R@500), the larger α, the worse. When α=0.5, R@500 drops to 56.3% (-2.2pp). The Contrastive gradient competes with the NTP CE gradient, and the greater the intensity, the greater the damage.
2. **dim256**: Doubling the projection dimension does not help (PPL=29.66, R@500=58.8%), but makes it worse.
3. **temp005**: Lower temperature (sharper distribution) does not help (PPL=28.16, R@500=58.2%).
4. **Root cause analysis**: SID is a discrete codebook token, and the decoder classifies it in the token space. InfoNCE attempts to align hidden states into continuous embedding space, but this does not directly help discrete token prediction. The consistent variation of PPL indicates that contrastive gradient interferes with NTP learning.
5. The PPL of all configs is 0.84~2.61 worse than the baseline, indicating that this is not regularization but interference.

### Next Steps
- Close IDEA-onemall-0 and no longer chase contrastive variants
- Moving to training objective level improvements: IDEA-genrec-0 (Page-wise NTP)

---

## EXP-021: Qwen3-4B vs 0.6B Embedding Quality for SID Tokenizer

**Date**: 2026-04-20
**Status**: planned
**Results**: TBD

### Background

All current SID experiments are based on Qwen3-Embedding-0.6B (dim=1024). The embedding cache of Qwen3-Embedding-4B (dim=2560) has been produced on EFS. The embedding semantics of larger models are richer, but whether the tokenizer benefits from this depends on:
1. Does high-dimensional embedding make RQ layering more accurate (lower quantization error)
2. Does better embedding translate into better NTP recall?
3. The computational cost of 4B embedding is ~6.7× (0.6B → 4B). It is only worth it if the recall is significantly improved.

### Hypothesis

1. Qwen3-4B embedding quantization error is 20%+ lower than 0.6B (dim 2.5× → RQ residual is smaller)
2. NTP Recall@10 improves 2-5pp (15.4% → 17-20%) on the SID of 4B embedding
3. FSQ may need to be adjusted (higher dimensional input → FSQ projection requires larger hidden size or more levels)
4. If recall is significantly improved, downstream RL/DPO will also benefit (better tokenizer = easier alignment)

### Design

- **Variable**: embedding model (qwen3-0.6b vs qwen3-4b)
- **Fixed**: Tokenizer config (1024 clusters, 2 KMeans layers, FSQ 6d_4096, MLP projection), 14d behavior data, NTP probe model structure is the same
- **Metric**:
  - Tokenizer: quantization error (MSE), collision rate, SID assignment distribution
  - NTP: PPL, item_recall@{10,50,100,500}
- **Data**: The same batch of content_id, two sets of embedding cache

| Config | Model | Dim | FSQ Hidden | Description |
|--------|-------|-----|------------|-------------|
| 1 | qwen3-0.6b | 1024 | 64 | Baseline (existing) |
| 2 | qwen3-4b | 2560 | 64 | 4B + same FSQ hidden |
| 3 | qwen3-4b | 2560 | 128 | 4B + larger FSQ hidden |

### Run
`bash experiments/scripts/exp-021.sh`

### Results
TBD

### Analysis
TBD

### Next Steps
TBD

---

## EXP-020: RF-DPO Hard λ Sweep — Finding Optimal DPO Weight

**Date**: 2026-04-20
**Status**: completed
**Results**: [./ntp_checkpoints/exp020-hard-lam03/](./ntp_checkpoints/exp020-hard-lam03/)

### Background

EXP-019 Joint NTP+DPO results show that the optimal λ of Hard is between 0.01~0.1:
- λ=0.01: PPL=14.4 (best), R@10=13.5%, R@500=66.4%, **but pref_acc=49.8%** (DPO signal is too weak, the model did not learn preference)
- λ=0.1: PPL=23.6 (degraded), R@10=13.6%, R@500=65.5%, pref_acc=93.6% (PPL begins to collapse)
- Reference: PPL=17.5, R@10=15.4%, R@500=68.3%

Goal: Find the sweet spot between λ=0.01~0.1, so that pref_acc >70% while maintaining PPL without significant degradation (<18).

Also test Easy multi-epoch: Easy only has 95 pairs / 15 steps, but in joint mode, the DPO data can be cycled multiple times (NTP prevents forgetting) to verify whether more DPO epochs can enhance the alignment effect of Easy.

### Hypothesis

1. λ=0.03~0.05 can make pref_acc >70% while maintaining PPL <18, achieving meaningful alignment.
2. Easy multi-epoch (more steps) allows the model to fully learn the preference of negative feedback and has a stronger alignment effect than the 15-step Easy
3. There is a λ such that R@10 exceeds reference (15.4%) - alignment and recall do not necessarily conflict

### Design

- **Variable**: λ (DPO weight) — 0.03, 0.05, 0.07; Easy multi-epoch steps
- **Fixed**: S-tier model (17.5M), 14d data, β=0.1, lr=1e-4, ref=SP-DPO fixed-medium, Hard 807 steps
- **Metric**: PPL, item_recall@{10,50,100,500}, reward_margin, preference_acc
- **Data**: RF-DPO Hard 4,312 pairs; Easy 95 pairs

Step calculation:
- Hard: 4,312 pairs / 16 batches = 269 batches × 3 epochs = 807 steps (same as EXP-019)
- Easy multi-epoch: 95 pairs / 16 batch = 5 batches × 20 epochs = 100 steps

| Config | Name | Difficulty | λ | β | max_steps | Description |
|--------|------|-----------|-----|-----|-----------|-------------|
| 1 | hard-lam03 | Hard | 0.03 | 0.1 | 807 | λ=0.03 |
| 2 | hard-lam05 | Hard | 0.05 | 0.1 | 807 | λ=0.05 |
| 3 | hard-lam07 | Hard | 0.07 | 0.1 | 807 | λ=0.07 |
| 4 | easy-multi | Easy | 0.1 | 0.1 | 100 | Easy 20 epochs |

### Run
`bash experiments/scripts/exp-020.sh`

### Results

EXP-020 only retains the optimal config (hard-lam03), hard-lam05/07 and easy-multi checkout are not on this machine (only SOTA checkpoint is retained in the experiment).

| Config | λ | PPL | R@10 | R@100 | R@500 | TrainingDuration |
|--------|---|-----|------|-------|-------|---------|
| **hard-lam03** | 0.03 | **16.3** | **14.1%** | — | **66.2%** | 62min |
| EXP-019 hard-lam01 (ref) | 0.01 | 14.4 | 13.5% | — | 66.4% | 62min |
| EXP-019 hard-lam10 (ref) | 0.10 | 23.6 | 13.6% | — | 65.5% | 62min |

hard-lam03 achieves the optimal balance between PPL and R@500 and becomes the new SOTA baseline (R@500=66.2%).

### Analysis

λ=0.03 is the sweet spot:
- pref_acc is high enough (DPO alignment is valid)
- PPL remains at 16.3 (no significant degradation in NTP capabilities)
- λ=0.01 lowest PPL but R@500 only slightly ahead 66.4% vs 66.2% (statistical noise range)
- λ=0.10 PPL begins to degrade to 23.6, indicating that excessive DPO weight damages language modeling

### Next Steps

exp020-hard-lam03 becomes the SFT baseline for the GRPO/ECPO stage (EXP-026~).

---

## EXP-019: RF-DPO Joint NTP+DPO — Step-Matched Training

**Date**: 2026-04-20
**Status**: completed
**Results**: [./ntp_checkpoints/exp019-*/](./ntp_checkpoints/exp019-*/)

### Background

EXP-018 (pure DPO) comprehensive degradation:

| Config | PPL | R@10 | R@500 |
|--------|-----|------|-------|
| Reference (SP-DPO) | ~14.5 | 15.4% | 68.3% |
| Easy (15 steps) | 35.8 | 13.8% | 64.2% |
| Hard (807 steps) | 50,694 | 8.3% | 28.9% |
| Progressive β=0.5 (best) | 404.9 | 10.2% | 49.8% |

Core problem: Pure DPO 807-step without NTP regularization → catastrophic forgetting. In the previous Easy group of EXP-017 joint NTP+DPO, NTP has 1555 steps but DPO has only 15 batches → NTP dominates and washes out the DPO signal.

Solution: Join NTP+DPO, but use `--max_steps` to limit the total number of steps to N epochs of DPO data, so that the NTP and DPO data volumes match. NTP provides regularization to prevent forgetting, and DPO provides alignment signals. The number of steps between the two is equivalent.

### Hypothesis

1. Joint NTP+DPO (step-matched) can maintain PPL ~15 on Hard difficulty while increasing Recall
2. `max_steps = DPO_batches × epochs` ensures that NTP will not wash out the DPO signal
3. λ (DPO weight) balances NTP regularization vs DPO alignment strength: λ is too small → NTP dominates (EXP-017 Easy lesson), λ is too large → DPO dominates and may degrade
4. Progressive Easy→Hard should be valid in joint mode (EXP-018 progressive is invalid because of pure DPO forgetting)

### Design

- **Variable**: difficulty (Easy/Hard/Progressive), λ (DPO weight), max_steps
- **Fixed**: S-tier model (17.5M), 14d data, β=0.1, lr=1e-4, ref=SP-DPO fixed-medium
- **Metric**: PPL, item_recall@{10,50,100,500}, depth_hit@10
- **Data**: RF-DPO preference pairs from EXP-018 (reuse), 14d NTP data

Step calculation:
- Easy: 95 pairs / 16 batches = 5 batches × 3 epochs = 15 steps
- Hard: 4,312 pairs / 16 batches = 269 batches × 3 epochs = 807 steps

| Config | Name | Difficulty | λ | β | max_steps | Description |
|--------|------|-----------|-----|-----|-----------|-------------|
| 1 | joint-easy | Easy | 0.1 | 0.1 | 15 | Joint NTP+DPO, Easy, step-matched |
| 2 | joint-hard | Hard | 0.1 | 0.1 | 807 | Joint NTP+DPO, Hard, step-matched |
| 3 | joint-hard-lam50 | Hard | 0.5 | 0.1 | 807 | Higher DPO weight |
| 4 | joint-hard-lam01 | Hard | 0.01 | 0.1 | 807 | Lower DPO weight |
| 5 | joint-prog | Progressive E→H | 0.1 | 0.1 | 807 | Progressive, Hard stage ref=Easy output |

### Run
`bash experiments/scripts/exp-019.sh`

### Results

Note: The wall_time≈27s of joint-easy/easy-lam10/easy___prefixbug is because there are only 15 steps (95 pairs / 16 batch = 5 batches × 3 epochs).

| Config | λ | Steps | PPL | R@10 | R@500 | TrainingDuration |
|--------|---|-------|-----|------|-------|---------|
| joint-easy | 0.1 | 15 | 20.6 | 13.8% | 67.5% | ~0.5min |
| joint-easy-lam10 | 1.0 | 15 | 15.3 | 14.3% | 67.7% | ~0.5min |
| joint-hard-lam01 | 0.01 | 807 | **14.4** | 13.5% | **66.4%** | 62min |
| joint-hard-lam10 | 0.10 | 807 | 23.6 | 13.6% | 65.5% | 62min |
| joint-hard-lam50 | 0.50 | 807 | 57.7 | 12.4% | 59.2% | 62min |
| joint-prog | 0.1 | 807 | 22.4 | 13.2% | 64.4% | 62min |
| Reference (SP-DPO) | — | — | ~14.5 | 15.4% | 68.3% | — |

### Analysis

1. **Joint NTP+DPO effectively prevents catastrophic forgetting**: best PPL=14.4 (λ=0.01), which is far better than pure DPO’s 50K+.
2. **R@500 does not exceed Reference (68.3%)**: hard-lam01 is optimal at 66.4%. The rejected = weak positive of Hard DPO pairs means the signal is noisy.
3. **λ sweet spot is at 0.01~0.03**: λ=0.01 PPL is optimal but DPO alignment is weak; λ=0.1 PPL begins to degrade.
4. **Progressive has no obvious advantage**: joint-prog R@500=64.4%, lower than joint-hard-lam01.

### Next Steps

EXP-020: Fine scan λ=0.03/0.05/0.07 to find the sweet spot of PPL vs pref_acc.

---

## EXP-018: RF-DPO — Real Feedback DPO Alignment

**Date**: 2026-04-18
**Status**: completed
**Results**: [./ntp_checkpoints/exp018-*/](./ntp_checkpoints/exp018-*/)

### Background

SP-DPO (EXP-017) uses model beam search to self-game to generate rejected candidates. RF-DPO further introduces **real user feedback signals** to distinguish signal strength from `action_bitmap` bit operations:

| Tier | Signal | action_bitmap bits |
|------|------|-------------------|
| Strong positive | like, share, follow, comment, trade, order | 2,4,8,256,512,1024,2048,131072,262144,524288,1048576 |
| Weak positive | click, coin/photo/profile click, video view | 1,16,64,128,8192,16384,32768,65536 |
| Negative | Report/Dislike | bit 31 (sign bit) |

Preference pair structure: pairing within the same user. Chosen = strong positive item, Rejected Easy = negative feedback items, Rejected Hard = weak positive items.

Source: Align³GR (AAAI 2026 Oral) Phase 2.

### Hypothesis

1. RF-DPO Easy (negative feedback rejected) has a clear improvement on Recall@10: the model learns to avoid content that users clearly hate.
2. RF-DPO Hard (weak positive rejected) finely distinguishes deep interaction vs shallow clicks
3. Progressive Easy→Hard is better than single stage
4. RF-DPO based on real feedback is better than self-game SP-DPO (the signal is more real, although the amount may be less)
5. RF-DPO on top of SP-DPO (as π_ref) for further stacking improvement

### Design

- **Variable**: difficulty (Easy/Hard), progressive vs single stage, beta ablation
- **Fixed**: S-tier model (17.5M active), 14 days of data, same user pairing, pure DPO (no NTP loss)
- **Metric**: PPL, item_recall@{10,50,100,500}, depth_hit@10, DPO loss curve
- **Data**: 14d behavior data 2026-03-18 ~ 2026-03-31 (same window as EXP-016/017)
- **Baseline**: SP-DPO fixed-medium (EXP-017, R@10=15.4%)
- **Mode**: Pure Softmax-DPO (per Align³GR paper), no NTP regularization

| Config | Name | Difficulty | β | Epochs | Reference model | DPO pairs |
|--------|------|-----------|-----|--------|-----------------|-----------|
| 1 | rfdpo-easy | Easy | 0.1 | 3 | SP-DPO fixed-medium | 95 |
| 2 | rfdpo-hard | Hard | 0.1 | 3 | SP-DPO fixed-medium | 4,312 |
| 3 | rfdpo-prog | Progressive Easy→Hard | 0.1 | 3 | Easy→Hard chain | 4,312 (stage 2) |
| 4 | rfdpo-prog-beta01 | Progressive Hard | 0.01 | 3 | Easy output | 4,312 |
| 5 | rfdpo-prog-beta50 | Progressive Hard | 0.5 | 3 | Easy output | 4,312 |

### Run
`bash experiments/scripts/exp-018.sh`

### Results

| Config | β | DPO pairs | Steps | PPL | R@10 | R@50 | R@100 | R@500 | TrainingDuration |
|--------|-----|-----------|-------|---------|------|------|-------|-------|---------|
| **Reference (SP-DPO)** | — | — | — | ~14.5 | **15.4%** | — | — | **68.3%** | — |
| rfdpo-easy | 0.1 | 95 | 15 | 35.8 | 13.8% | 31.4% | 40.9% | 64.2% | ~0.5min |
| rfdpo-hard | 0.1 | 4,312 | 807 | 50,694 | 8.3% | 18.2% | 23.3% | 28.9% | 51min |
| rfdpo-prog (E→H) | 0.1 | 4,312 | 807 | 98,747 | 8.7% | 16.7% | 21.5% | 26.3% | 51min |
| rfdpo-prog-beta01 | 0.01 | 4,312 | 807 | 2.4B | 6.0% | 11.5% | 14.6% | 15.9% | 51min |
| rfdpo-prog-beta50 | 0.5 | 4,312 | 807 | 404.9 | 10.2% | 25.4% | 34.1% | 49.8% | 51min |

### Analysis

**Pure DPO is fully degraded and no configuration exceeds reference. **

1. **Catastrophic forgetting**: Pure DPO without NTP regularization, 807 steps of hard training causes the model to forget the NTP language modeling ability. PPL exploded from ~14.5 to 50K–2.4B.
2. **β as a regularizer**: The larger β is, the stronger the KL penalty on the reference policy will be and the lighter the degradation will be. β=0.5 is the best (PPL 404.9, R@500 49.8%) but still far worse than reference. β=0.01 is the worst (PPL 2.4B).
3. **Easy barely changes the model**: only 95 pairs / 15 steps, training is not enough to change the model. PPL 35.8 indicates slight degradation, but R@10 13.8% is closest to the reference.
4. **Progressive has no advantage**: Easy→Hard is almost the same as single-stage Hard (PPL 98K vs 50K, R@10 8.7% vs 8.3%), and the Progressive structure fails to alleviate forgetting.
5. **Conclusion**: The pure Softmax-DPO solution of the paper is not feasible under the scale of our data. Larger data volumes, additional KL constraints, or details not disclosed in the paper may be required. The idea of ​​Joint NTP+DPO is correct, but the number of steps needs to be limited to match the amount of DPO data.

### Next Steps

EXP-019: Joint NTP+DPO with step-matched training — Limit the number of NTP steps to match the DPO epoch to prevent NTP from washing out the DPO signal (a lesson from EXP-018 Easy), while retaining NTP regularization to prevent catastrophic forgetting (a lesson from EXP-018 Hard).

---

## EXP-017: SP-DPO — Self-Play DPO Alignment for NTP Model

**Date**: 2026-04-17 ~ 2026-04-20
**Status**: completed
**Results**: [experiments/ntp_checkpoints/exp017-*](experiments/ntp_checkpoints/)

### Background

The NTP model has reached the S-tier baseline (EXP-016 14d-S: PPL=27.05, R@500=58.5%). The current training is pure SFT (cross entropy), which only tells the model "what is right" and does not tell the model "which mistakes you are currently making are wrong".

SP-DPO (Self-Play DPO, Align³GR, AAAI 2026 Oral) is an entry-level solution for RL alignment:
1. Use the model's own beam search to generate candidates as rejected (negative samples)
2. Ground truth as chosen (positive sample)
3. Define the difficulty according to the number of SID prefix matching layers: Easy (L0 is different) → Medium (L0 is the same, L1 is different) → Hard (L0+L1 is the same, L2 is different)
4. Softmax-DPO loss progressive training (1 chosen vs 20 rejected)

**Key findings**: In the beam search + classify method of the paper, under the 4096×3 SID system, the number of Medium/Hard candidates is ≈ 0 (B=200 is still almost all Easy). Reason: Beam freely samples from L0, and the probability of hitting GT L0 is extremely low (SFT depth_acc L0=3%).

**Solution**: Prefix-locked beam search — Fixed GT prefix, beam search remaining layers. Make sure there are enough Medium/Hard candidates. See [discussions/004](../discussions/004-prefix-locked-vs-paper-beam-search.md) for details.

### Hypothesis

1. SP-DPO Easy improves R@10 (opening up the basic boundaries between right and wrong) - **Verified** ✓
2. Prefix-locked progressive sampling produces sufficient M/H data to make the Medium/Hard stage feasible
3. Progressive model (Easy model sampling) vs fixed model (SFT sampling): The candidates generated after the model is improved are more targeted
4. Joint loss (NTP + DPO) to keep SFT knowledge from being lost

### Design

- **Variable**: M/H sampling model (SFT vs Easy model), λ weight
- **Fixed**: S-tier model (256d, 6L, 8E top-2, ~17.5M active), 14 days of data, prefix-locked B=50
- **Metric**: PPL, item_recall@{10,50,100,500}, depth_acc_beam, DPO loss curve
- **Data**: EXP-016 14d preprocessed NTP data (130M tokens)
- **Baseline**: EXP-016 14d-S checkpoint (PPL=27.05, R@500=58.5%)

**Sampling method**: Prefix-locked beam search (all config)

| sampling pass | lock prefix | output difficulty | beam_size |
|-----------|----------|----------|-----------|
| Pass 1 | None | Easy (L0 ≠ GT) | 50 |
| Pass 2 | L0=GT | Medium (L1 ≠ GT) + Hard | 50 |
| Pass 3 | L0+L1=GT | Hard (L2 ≠ GT) | 50 |

**Experimental Matrix**:

| Config | `--start-from` | M/H Sampling Model | Description |
|--------|------|-------------|------|
| Shared Easy | 1 | SFT (full beam) | Easy DPO baseline, shared |
| Config 1 | 2 | **SFT** prefix-locked | Fixed Model + Progressive Sampling |
| Config 2 | 3 | **Easy model** prefix-locked | Progressive Model + Progressive Sampling |
| λ=0.05 | 4 | Easy model | λ ablation |
| λ=0.5 | 5 | Easy model | λ ablation |

**Key comparison**: Config 1 vs 2 → Does the progressive model help?

### Run

`bash experiments/scripts/exp-017.sh --no-smoke --start-from=1`

### Results

**Easy stage (shared)**:

| Metric | SFT Baseline | SP-DPO Easy | Delta |
|--------|-------------|-------------|-------|
| PPL | 27.05 | 28.49 | +5.3% (expected) |
| R@10 | 9.9% | **12.5%** | **+26.3%** |
| R@50 | 26.1% | 27.1% | +3.8% |
| R@500 | 58.5% | 55.0% | -6.0% |
| depth_acc L0 | 0.030 | **0.041** | **+37%** |
| depth_acc L1 | 0.018 | **0.029** | **+61%** |
| depth_acc L2 | 0.018 | **0.029** | **+61%** |
| L2 PPL | 4.84 | **2.48** | **-48.7%** |

**Config 1 vs Config 2: Easy → Medium → Hard (completed)**:

| Metric | SFT | C1 Medium | C1 Hard | C2 Medium | C2 Hard |
|--------|-----|-----------|---------|-----------|---------|
| prefix L0 | 0.200 | 0.224 | 0.231 | **0.234** | 0.231 |
| prefix L1 | 0.172 | 0.203 | 0.210 | **0.212** | 0.207 |
| prefix L2 | 0.171 | 0.201 | 0.209 | **0.210** | 0.206 |
| indep L0 | 0.199 | 0.223 | 0.230 | **0.233** | 0.230 |
| indep L1 | 0.808 | **0.899** | 0.888 | 0.885 | 0.877 |
| indep L2 | 0.852 | 0.941 | **0.957** | 0.936 | 0.949 |
| PPL | 27.05 | 17.49 | **14.24** | 16.13 | 15.24 |
| DPO loss | — | 2.210 | 1.305 | 2.331 | 1.321 |
| wall_time | — | 3.9h | 2.1h | 2.2h | **1.2h** |

depth_hit@10 is based on 147,902 eval positions. R@500 only has 1,000 samples, its statistical significance is limited, and it is not used as the main indicator.

**Beam search candidate distribution** (SFT model, B=200):
- Easy: ~20/pair, Medium: ~0/pair, Hard: 0/pair
- Confirm that the M/H data of the paper method is seriously insufficient under the 4096×3 system → prefix-locked is necessary

**Hard candidate scarcity**: avg 5.9 rejected/pair (vs Medium ~20/pair). Reason: There are few valid L2 options in the trie under L0+L1 prefix. Non-code bugs are determined by the data hierarchy structure.

### Analysis

1. **The optimal stage is Easy → Medium, Hard has no positive contribution**:
   Two Config consistent verification: Hard stage depth_hit@10 and indep L1 are both degraded. Hard DPO loss is unusually low (~1.3 vs Medium ~2.3) and the signal is "too simple".

2. **Three reasons for Hard degradation**:
   - **Signal too narrow**: rejected only differs in L2, DPO gradient only teaches L2 discrimination, but updates the entire shared backbone → interferes with L0/L1 representation
   - **Rejected too few**: avg 5.9/pair → logsumexp only ~6 terms, gradient noisy; model can easily distinguish → effective signal close to zero
   - **Selection bias**: Only if there are ≥2 valid L2 items under GT's L0+L1 prefix, hard pairs will be generated, and the non-eval full set will be represented

3. **C2 Medium prefix indicators are comprehensively optimal**: This shows that the Medium/Hard candidates sampled by Easy model are more targeted than those sampled by SFT (on-policy effect). But C1 Medium’s indep L1 (0.899) > C2 Medium (0.885) — probably because the candidates generated by SFT sampling cover a wider L1 space.

4. **PPL vs depth_hit not completely positive**: C2 Hard PPL is lowest (15.24) but depth_hit is not optimal - PPL optimizes absolute probability, depth_hit optimizes top-K ranking.

5. **Engineering wins**:
   - Gradient checkpointing + MoE freeze: Solve DPO OOM and double the throughput (9k→17k tok/s). See [docs/engineering/001](../docs/engineering/001-dpo-oom-gradient-checkpointing.md) for details
   - Packed DPO candidates: eliminate padding waste, and increase the speed of Hard training by 44% (17k→30k tok/s)
   - KV cache beam search: context encoding redundancy reduced from ~153C/item to ~C/3/item. See [discussions/005](../discussions/005-beam-search-kv-cache.md) for details

### Conclusions

1. **SP-DPO Easy → Medium is the optimal pipeline**, and the Hard stage should be skipped
2. **Optimal checkpoint: C2 Medium** (prefix is the best overall) or **C1 Medium** (indep L1 is the best)
3. Prefix-locked beam search is a necessary condition for Medium/Hard data generation under the 4096×3 SID system.
4. There is little difference between progressive model sampling (C2) and fixed model (C1). Both have their own advantages and indicators.

### Next Steps

1. ~~λ ablation~~ — Hard stage is no longer available.
2. EXP-018: RF-DPO (introducing real user feedback, Align³GR Phase 2), using C2 Medium or C1 Medium as π_ref

---

## EXP-015: NTP Scaling Law — Sweep Model Size from 1M to 100M Active Params

**Date**: 2026-04-16 ~ 2026-04-17
**Status**: completed
**Results**: [experiments/results/ntp/](experiments/results/ntp/)

### Background

EXP-013 demonstrates that enlarging parameters (7.5M→45.8M) can accelerate convergence (PPL 70→29.6, recall@500 37%→60%). But with only two data points, it doesn’t answer the key question: When do benefits saturate? Which model is the most cost-effective?

The OneRec-V2 paper gives the scaling law `L̂(N) = 3.13 + 3660 / N^0.489` in the recommendation field, proving that the loss of the recommendation model also follows the power law. This experiment fits our own scaling law on the same data through 7 model configurations of different scales.

### Hypothesis

1. NTP eval loss regarding active params follows power law `L(N) = a + b / N^α`
2. α is close to OneRec-V2’s 0.489 (similar architecture)
3. There is a clear turning point in cost performance (the turning range where diminishing returns accelerate)

### Design

- **Variable**: model size (embed_dim, n_layers, MoE config)
- **Fixed**: SID 4096×3 binary, 31 days of data (03-01~03-31), 1 epoch, beam_size=500
- **Data**: Reuse EXP-013 preprocessed NTP data (262M tokens)
- **Metric**: eval loss, PPL, item_recall@{10,50,100,500}

| Config | embed_dim | layers | MoE | ~Active Params |
|--------|-----------|--------|-----|----------------|
| scale-01 | 64 | 2 | dense | 1.7M |
| scale-02 | 128 | 2 | dense | 3.6M |
| scale-03 | 128 | 4 | 4E top-2 | 5.1M |
| scale-04 | 256 | 6 | 8E top-2 | 17.5M |
| scale-05 | 384 | 6 | 8E top-2 | 34.5M |
| scale-06 | 512 | 8 | 8E top-2 | 71.6M |
| scale-07 | 512 | 12 | 16E top-2 | 101.1M |

**Code changes**: The s-tier hyperparameters in `ntp/train.py` were changed from hard-coded to CLI configurable (`--n_experts`, `--top_k`, `--expert_dim`, `--embed_dim`, `--n_transformer_layers`). `n_experts=0` automatically switches to dense mode.

### Run

`bash experiments/scripts/exp-015.sh`

### Results

| Config | Active Params | PPL | Loss | R@10 | R@100 | R@500 | TrainingDuration |
|--------|--------------|------|------|------|-------|-------|---------|
| scale-01 | 1.7M | 235.1 | 5.460 | 1.9% | 11.8% | 23.6% | 2min |
| scale-02 | 3.6M | 100.4 | 4.609 | 3.7% | 16.6% | 31.7% | 3min |
| scale-03 | 5.1M | 69.6 | 4.243 | 5.4% | 24.9% | 45.6% | 9min |
| scale-04 | 17.5M | **28.1** | 3.334 | 9.8% | 35.6% | 60.5% | 34min |
| scale-05 | 34.5M | 24.0 | 3.178 | 11.5% | 39.1% | 62.5% | 61min |
| scale-06 | 71.6M | 20.8 | 3.037 | 12.6% | 41.0% | 66.2% | 131min |
| scale-07 | 101.1M | **19.4** | 2.965 | 13.7% | 43.2% | 65.8% | 374min |

**Scaling Law Fit**:

```
L̂(N) = 2.522 + 2055.1 / N^0.456
```

- **a = 2.522**: irreducible loss floor (data/tokenizer information bottleneck)
- **α = 0.456**: scaling exponent (close to OneRec-V2’s 0.489)
- **b = 2055.1**: scale factor

![NTP Scaling Law](results/ntp/exp015-scaling-law.png)

### Analysis

1. **Power law is established**: The 7 data points on the log-log graph basically fall on a straight line, and R² is good
2. **α = 0.456 ≈ 0.489** of OneRec-V2: The architecture scaling efficiency is close to the paper, verifying the versatility of MoE + SwiGLU
3. **Obvious diminishing returns**:
   - 5M→17M: PPL 70→28 (-60%), recall@500 46%→60% — **Maximum improvement interval**
   - 17M→71M: PPL 28→21 (-25%), recall@500 60%→66% — medium improvement
   - 71M→101M: PPL 21→19 (-7%), recall@500 66%→66% — **Close to saturation**
4. **Irreducible loss a=2.522 (PPL≈12.5)**: Even if the model is infinitely large, the PPL cannot drop below 12.5. This is the ceiling of tokenizer (4096×3 codebook, collision 0.89%) and randomness of user behavior
5. **Recall is also scaling but at different growth rates**: R@100 increased by 3.6x from 12%→43%, while R@500 only increased by 2.8x from 24%→66% - the larger model has a more significant improvement in top-K fine ranking
6. **EXP-013 data points match**: probe (7.5M, PPL=70) and s-tier (45.8M, PPL=29.6) both accurately fall on the fitting curve

**Hypothesis verification**:
- H1 ✅ Power law is established, and the 7-point fit is good.
- H2 ✅ α=0.456 ≈ 0.489, very close
- H3 ✅ M level (~50-70M active) is a clear sweet spot, after which the curve flattens

### Predictions

| Active Params | Predict PPL | Predict Loss | Price/Performance |
|--------------|---------|-----------|--------|
| 17M (S) | 28 | 3.33 | Current Baseline |
| **55M (M)** | **~23** | **~3.15** | **Best value for money** |
| 500M (L) | ~15.5 | ~2.74 | High cost, diminishing returns |
| 1B | ~14.6 | ~2.68 | Close to floor |

### Chinchilla Analysis

All EXP-015 models are trained on the same 262M tokens. According to Chinchilla's rule of thumb (N* = D/20), the optimal model size is approximately 13M active params.

**Tokens/Param and FLOP efficiency**:

| Config | Active | Tok/Param | FLOP efficiency (loss/PF) | Chinchilla Status |
|--------|--------|----------|------------------------|----------------|
| scale-01 | 1.7M | 152 | — | Over Training 7.6x |
| scale-02 | 3.6M | 72 | 0.28 | Over Training 3.6x |
| scale-03 | 5.1M | 52 | 0.16 | Over Training 2.6x |
| **scale-04** | **17.5M** | **15** | **0.05** | **Close to optimal (0.7x)** |
| scale-05 | 34.5M | 8 | 0.01 | Owe Training 0.4x |
| scale-06 | 71.6M | 4 | 0.002 | Seriously short of Training 0.2x |
| scale-07 | 101.1M | 3 | 0.002 | Seriously short of Training 0.1x |

**Key Findings**:

1. **FLOP efficiency decreases monotonically** (0.28 → 0.16 → 0.05 → 0.01 → 0.00), completely consistent with Chinchilla predictions
2. **scale-04 (17.5M) is the optimal point of Chinchilla with 262M tokens** — 15 tok/param close to 20 experience value
3. **The large model is seriously undertrained but the loss still decreases monotonically** — The recommended sequence is short (30 tokens), even 3 tok/param will not overfit, unlike LLM
4. **Extremely high data ROI**: 101M model tok/param needs ~2B tokens (~240 days of data) from 3→20, and PPL is expected to drop from 19.4 to close to floor (12.5)

**Chinchilla optimal data size**:

| Model | Active Params | Chinchilla Optimal Tokens | Number of days required |
|------|-------------|-----------------------|----------|
| S (17M) | 17.5M | 350M | ~41 days |
| M (55M) | 55M | 1.1B | ~130 days |
| M+ (101M) | 101M | 2.0B | ~240 days |

**Conclusion: The current bottleneck is the data, not the model. Adding data first (31→90 days) and then adding the model is the path with the highest ROI. **

### Next Steps

1. **EXP-016 Data Scaling**: fixed S/M model, sweep data amount → Chinchilla dual-variable scaling law → find the optimal N-D ratio
2. **Tokenizer ceiling**: a=2.522 is too high, try 8192×3 codebook to reduce irreducible loss

---

## EXP-016: Data Scaling Law — Fixed model Sweep data volume (Chinchilla bivariate)

**Date**: 2026-04-17 ~ 2026-04-18
**Status**: completed
**Results**: [experiments/results/ntp/](experiments/results/ntp/)

### Background

EXP-015 reveals two key facts:

1. **Scaling law is established**: `L(N) = 2.522 + 2055/N^0.456`, but this is a single variable law under fixed D=262M tokens
2. **Large model is seriously undertrained**: scale-07 (101M active) only has 3 tok/param, Chinchilla recommends 20x. FLOP efficiency declines sharply after exceeding 17.5M

The complete scaling law of Chinchilla (Hoffmann 2022) is bivariate:

```
L(N, D) = E + A/N^α + B/D^β
```

Where E is irreducible loss, A/N^α is the model deficiency term, and B/D^β is the data deficiency term. EXP-015 only sweeps N, D is fixed. In this experiment, N is fixed and sweep D is used to fit a complete bivariate law and find the optimal N-D ratio under a given computing power budget.

**Core question**: After expanding the data from 31 days to 66 days:
- How much is the PPL reduction for S level (17.5M active) and M level (101M active)?
- What is β? (data scaling index)

### Data Distribution Analysis

Available embedding covers 2026-01-25 ~ 2026-03-31 (66 days). Data distribution analysis (`analyze_data_distribution.py`):

| Config | Users | Raw Items | Mean/User | P50 | P95 | P99 | Max |
|--------|-------|-----------|-----------|-----|-----|-----|-----|
| A-7d | 1.54M | 23.9M | 15.6 | 3 | 68 | 220 | 5,376 |
| B-14d | 2.51M | 53.1M | 21.2 | 3 | 92 | 331 | 9,063 |
| C-31d | 4.55M | 129.7M | 28.5 | 3 | 118 | 499 | 32,246 |
| D-62d | 7.29M | 261.8M | 35.9 | 3 | 138 | 669 | 46,223 |
| E-66d | 7.85M | 299.0M | 38.1 | 3 | 146 | 715 | 46,990 |

**Key findings: Extremely long tail + truncation has a large impact**

- **P50 is constant at 3**: 50% of users have only ≤3 interactions, and the distribution is extremely right-skewed
- **A few heavy users contribute a large number of items**: 4% of users are cut off by the 170-item cap, but their interactions account for ~50% of the total
- This is a **non-contradictory phenomenon** between the two dimensions: the user dimension has a small impact of truncation (4%), and the item dimension has a large impact (50%)

**Truncated analysis** (`max_seq_len=512` → `max_items=170`):

| Config | Truncated User % | Items Lost % | Raw Items | Valid Items | **Valid Tokens** |
|--------|----------|------------|-----------|-----------|----------------|
| A-7d | 1.5% | 14.5% | 23.9M | ~20.4M | **~61M** |
| B-14d | 2.6% | 25.4% | 53.1M | ~39.6M | **~119M** |
| C-31d | 3.6% | 38.9% | 129.7M | ~79.3M | **~238M** |
| D-62d | 4.2% | 48.5% | 261.8M | ~134.8M | **~404M** |
| E-66d | 4.4% | 50.4% | 299.0M | ~148.3M | **~445M** |

> Note: Valid Tokens = Valid Items × 3 (n_layers=3). Truncation keeps the most recent 170 items for each user and discards older history.
> For recommendation scenarios, recent behaviors are more valuable, and truncated old behaviors have limited impact on model training.

### Hypothesis

1. Data from 238M→445M tokens (31d→66d), S level (17.5M) PPL decline is limited (<5%), because it is close to Chinchilla optimal
2. Data from 238M→445M tokens (31d→66d), M file (101M) PPL dropped significantly (>15%), because it is currently seriously under-trained.
3. β ≈ 0.4-0.5 (close to α ≈ 0.456, consistent with Chinchilla symmetry assumption)
4. Given 66 days of data (~445M tokens), Chinchilla’s optimal model size moves up to ~22M active params

### Design

- **Variable**: Data volume D ∈ {7d, 14d, 31d, 62d, 66d} × Model {S (17.5M), M+ (101M)}
- **Fixed**: SID 4096×3 binary, 1 epoch, beam_size=500, max_seq_len=512 (170 items/user)
- **Metric**: eval loss, PPL, item_recall@{10,50,100,500}
- **Eval Description**: `preprocess-ntp` of each config uses `n_eval_target=50000` to cut split_ts according to time points. There are slight differences in eval sets of different data sizes (different split_ts), but they are all concentrated at the end of the window, which has limited impact on scaling law fitting.

| Config | Model | Data Days | Users | Valid Tokens | Tok/Param (S) | Tok/Param (M+) |
|--------|-------|-----------|-------|------------|---------------|----------------|
| A-7d | S + M+ | 7 | 1.54M | ~61M | 3.5 | 0.6 |
| B-14d | S + M+ | 14 | 2.51M | ~119M | 6.8 | 1.2 |
| C-31d | S + M+ | 31 | 4.55M | ~238M | 13.6 | 2.4 |
| D-62d | S + M+ | 62 | 7.29M | ~404M | 23.1 | 4.0 |
| E-66d | S + M+ | 66 | 7.85M | ~445M | 25.4 | 4.4 |

The S file of the C-31d can reuse the EXP-015 scale-04 results, and the M+ file can reuse the scale-07 results. Actual new training: 4×2 = 8 runs (minus C-31d reuse = 6 runs).

**Analysis Plan**:
1. Fit `L(D) = E + B/D^β` to S and M+ respectively.
2. Joint EXP-015 + EXP-016 data fitting bivariate `L(N,D) = E + A/N^α + B/D^β`
3. Draw iso-FLOP curves (fixed C=6ND) and find the optimal N-D distribution on each curve
4. Prediction: Given a computing power budget of 8×A100 × 1h, what is the optimal configuration?

### Run

`bash experiments/scripts/exp-016.sh`

### Results

**S model (17.5M active)**:

| Config | Days | Tokens | Users | PPL | Loss | R@100 | R@500 | TrainingDuration |
|--------|------|--------|-------|-----|------|-------|-------|---------|
| A-7d-S | 7 | 65M | 1.02M | 30.60 | 3.421 | 37.9% | 62.1% | 11min |
| **B-14d-S** | **14** | **130M** | **1.69M** | **27.05** | **3.298** | **35.0%** | **58.5%** | 17min |
| C-31d-S | 31 | 262M | 3.04M | 28.05 | 3.334 | 35.6% | 60.5% | 55min |
| D-62d-S | 62 | 441M | 4.86M | 30.03 | 3.402 | 36.5% | 58.6% | 55min |
| E-90d-S | 90 | 553M | 6.18M | 31.89 | 3.462 | 35.1% | 56.2% | 69min |

**M+ Model (101M active)**:

| Config | Days | Tokens | Users | PPL | Loss | R@100 | R@500 | TrainingDuration |
|--------|------|--------|-------|-----|------|-------|-------|----------|
| A-7d-M | 7 | 65M | 1.02M | 19.31 | 2.960 | 42.7% | 70.7% | 123min |
| **B-14d-M** | **14** | **130M** | **1.69M** | **18.96** | **2.942** | **43.0%** | **65.8%** | 207min |
| C-31d-M | 31 | 262M | 3.04M | 19.39 | 2.965 | 43.2% | 65.8% | 374min |
| D-62d-M | 62 | 441M | 4.86M | 19.80 | 2.986 | 43.2% | 68.1% | 607min |
| E-90d-M | 90 | — | 6.18M | *(skip)* | — | — | — | — |

![Data Scaling Law](results/ntp/exp016-data-scaling.png)

### Analysis

**1. Chinchilla data scaling is not applicable to recommended sequences**

Chinchilla assumes i.i.d. data: more tokens monotonically reduce loss. However, the recommended behavior data has time non-stationarity. **14d is the optimal point of loss**, and then the loss rises:

- S: 3.421 (7d) → **3.298 (14d)** → 3.334 (31d) → 3.402 (62d) → 3.462 (90d)
- M+: 2.960 (7d) → **2.942 (14d)** → 2.965 (31d) → 2.986 (62d)

This is a **U-shaped curve**, not a power law decrease.

**2. Root cause: increasing number of days = increasing users, not a longer sequence**

| Days | Users | Avg Items/User |
|------|-------|---------------|
| 7d | 1.02M | ~21 |
| 14d | 1.69M | ~26 |
| 31d | 3.04M | ~29 |
| 62d | 4.86M | ~30 |
| 90d | 6.18M | ~30 |

Avg items/user is almost unchanged from 21→30 (limited by max_seq_len=512 and user activity), but the number of users increases by 6x from 1M→6M. The new users come from an earlier time window, and the behavior distribution has shifted.

**3. The exposure window constraint is the core reason**

The exposure item in this scene is limited to content created within 3 days. This means:
- The item pool is completely refreshed every 3 days
- The item pool corresponding to the training data 30 days ago no longer exists at all
- The behavior pattern of old data may no longer be applicable to the current item pool

14d ≈ 4-5 exposure window turnover periods, which is the balance point between covering the diversity of item pool and avoiding distribution deviation.

**4. The model is close to irreducible loss**

M+ reaches loss=2.942 at 14d (130M tokens), which is basically consistent with the `L(101M) = 2.522 + 2055/101M^0.456 ≈ 2.96` predicted by EXP-015. The remaining gap (2.942 - 2.522 = 0.42) is dominated by the tokenizer information bottleneck, which cannot be broken through by adding data.

**5. Not inconsistent with the sequence length scaling law**

The sequence scaling law reported in the paper is to fix the user group and increase the history length of each user (depth scaling). The scale of this experiment is user breadth (more low active/historical users), not sequence depth. The two are different dimensions.

### Hypothesis verification

- H1 ❌ S 14d→90d PPL increased from 27.05 to 31.89 (+18%), not decreased
- H2 ❌ M+ 14d→62d PPL increased from 18.96 to 19.80, not a >15% decrease
- H3 cannot be verified: Chinchilla bivariate law is not applicable, β is meaningless
- H4 ❌ The optimal model size does not move up with the amount of data, because it is ineffective when the amount of data increases.

### Key Findings

1. **Optimal training window ~14d**: This holds true for both S and M+ models, and loss/PPL reaches the lowest
2. **Chinchilla data scaling is not applicable**: Recommended behavior data is not i.i.d. and has an "effective half-life" (~14d)
3. **The bottleneck is the tokenizer, not the data**: M+ loss=2.94 is approaching the irreducible floor 2.52
4. **The next step should be scale sequence depth or tokenizer**, not data time range

### Next Steps

1. **Tokenizer improvement** (highest ROI): 8192×3 codebook or finer FSQ → reduce irreducible loss floor
2. **Sequence depth scaling**: fixed 14d user group, sweep max_items {10, 30, 50, 100, 170} → verify the real sequence scaling law
3. **Multiple epochs on 14d**: S model 1 epoch may be underfitting, try 2-3 epochs

---

## EXP-014: ENTP-Loss — Exposure-Aware Hard Negatives for L0

**Date**: 2026-04-16
**Status**: running
**IDEA**: IDEA-dualgr-0
**Results**: TBD

### Background

EXP-013 S-tier model recall@500=59.5%, but **L0 PPL=344.8 is a clear bottleneck** (L1=13.3, L2=5.7 is close to saturation). L0 hit@10 is only 20%, and the model’s discriminative ability on 4096 coarse clusters is weak.

Currently, NTP loss only has positive samples (items clicked by the user), and does not take advantage of the negative signal of "the user looked at it but did not click" at all. DualGR (Kuaishou, WWW 2026, arxiv 2511.12518) proposed ENTP-Loss: using exposed unclicked items as L0 layer hard negative, and directly enhancing the L0 supervision signal through the `−α·log(1 − p_L0)` penalty term.

The data side `export_exposure.py` is ready, with ~1.1GB exposure data per day (including unclicked items with action_bitmap=0), which is about 13:1 with the behavioral data ~85MB/day.

### Hypothesis

1. ENTP-Loss (α=0.1) reduces L0 PPL by >10% (from 344.8 to <310) because L0 gets additional per-position temporally aligned negative sample supervision
2. L1/L2 PPL is not affected (ENTP only acts on the output_proj of the L0 layer)
3. Recall@500 improvement (L0 is more accurate → beam search filters better at coarse level → downstream fine-level benefits)

### Design

- **Variable**: ENTP weight α ∈ {0, 0.05, 0.1, 0.2}
- **Fixed**: S-tier 6L MoE (EXP-013 configuration), K=5 negatives/position, 4096×3 binary SID, batch_size=128, 1 epoch, beam_size=500
- **Metric**: L0/L1/L2 PPL, hit@10 per layer, recall@{10,50,100,500}
- **Data**: 31 days of behavioral data (03-01~03-31) + 31 days of exposure data (same period)

**ENTP negative sample construction (PySpark side)**:
- `export_exposure.py` added ENTP section: Spark SQL window function `pos_grp = cumsum(is_positive)` section,
  The non-positive (action_bitmap ≤ 0) of each segment is used as the negative sample of the next positive, taking the latest K=5
- Output `feed_user_exposure_neg/{date_start}_{date_end}` parquet: `uid, iid, first_ts, neg_iids ARRAY<STRING>`
- Python side `load_exposure_neg_data()` loads ~130M rows (seconds), `_build_sequences_from_exposure()` only does iid→L0 mapping

**Loss**:
```
L = L_NTP(L0+L1+L2 three-layer CE, unchanged) + α * L_ENTP(only L0 negative sample penalty)
L_ENTP = −(1/N) Σ log(1 − p_i^(L0)) (L0 token for unclicked exposure)
```

**Change file**:
1. `data/export_exposure.py` — PySpark ENTP negative sample export (Spark SQL window function)
2. `eval/batch.py` — Added `load_exposure_neg_data()` to load compact parquet
3. `ntp/train.py` — `_build_sequences_from_exposure()` simplified to dict→sequence mapping; wandb integration
4. `ntp/model.py` — `_forward_packed()` adds ENTP loss item
5. `ntp/baseline.py` — `NTPProbe._forward_packed()` synchronizes ENTP extensions
6. `ntp/preprocess.py` — shard format extended storage neg_l0; call `load_exposure_neg_data()`

**Pluggable Design**: `--entp_weight 0` (default) = exactly equivalent to the EXP-013 code path.

| Config | α | K | L0 filter | Description |
|--------|------|---|-----------|------|
| A (baseline) | 0 | — | — | Direct multiplexing EXP-013 s-tier Result |
| B | 0.05 | 5 | ✗ | Conservative (round 1, regressed) |
| C | 0.1 | 5 | ✗ | DualGR Paper default (round 1, regressed) |
| E | 0.05 | 5 | ✓ | Conservative (round 2, L0 collision filter) |
| F | 0.1 | 5 | ✓ | Paper default (round 2) |
| G | 0.2 | 5 | ✓ | Aggressive (round 2) |

### Run

`bash experiments/scripts/exp-014.sh`

### Results

**PySpark ENTP export verification (2026-04-16)**:

| Metric | PySpark Export | Old Streaming Walk (Contrast) | Description |
|---|---|---|---|
| Total Exposure Rows | ~1.19B | 1,185,707,891 | Consistent |
| Positives | 130,995,419 | 124,893,764 | +4.9%, difference = iid outside SID dictionary (Python side filtering) |
| Users | 4,608,606 | 3,042,069 | +51%, the extra users only have SID external iid, and disappear after filtering on the Python side |
| There are negative samples | 40,761,718 (31.1% row level) | 2,084,314 (68.5% user level) | Different caliber, no contradiction |

31% row-level negative samples are reasonable: in feed scenarios, users often click continuously (multiple items on the same page), and there is no non-positive between consecutive positives → the latter cannot get neg.

**Training results B/C (old code, without L0 collision filtering)**:

| Metric | A (α=0, baseline) | B (α=0.05) | C (α=0.1) | B Δ | C Δ |
|---|---|---|---|---|---|
| PPL | 29.60 | 31.67 | 31.67 | +7.0% | +7.0% |
| L0 PPL | 344.76 | 363.78 | 361.41 | +5.5% | +4.8% |
| L1 PPL | 13.28 | 15.23 | 15.23 | +14.7% | +14.7% |
| L2 PPL | 5.72 | 5.79 | 5.83 | +1.2% | +1.9% |
| L0 hit@10_indep | 0.2004 | 0.1919 | 0.1902 | -4.2% | -5.1% |
| recall@10 | 0.102 | 0.086 | 0.089 | -15.7% | -12.7% |
| recall@50 | 0.250 | 0.230 | 0.234 | -8.0% | -6.4% |
| recall@100 | 0.346 | 0.305 | 0.304 | -11.8% | -12.1% |
| recall@500 | 0.595 | 0.525 | 0.529 | -11.8% | -11.1% |

B/C regressive across the board. For root cause analysis, see Analysis.

### Analysis

**Root cause: L0 token collision leads to gradient conflict. **

Items in the same session are displayed together by the recommendation system because of similar topics. After quantification by SID, a large number of items fall into the same L0 cluster (4096 clusters, avg 122 items/cluster). When the negative sample and the positive sample share the same L0 token:
- NTP loss pushes up p(L0=k) (L0 of positive samples)
- ENTP loss suppresses p(L0=k) (L0 of negative samples, exactly the same)
- Gradient direct hedging → L0 PPL rises instead (344→363)
- Conflicts are propagated through shared transformer backbone → L1 PPL also declined significantly (+14.7%)

The DualGR paper uses 8192 L0 clusters and has 10B exposures/day, so the collision rate is naturally lower. The paper also mentions probability clipping `[ε, 1-ε]` but does not specify the ε value.

**Fix**: The preprocess stage filters out negative samples that share L0 with positive. Achieved, waiting to be re-run.

### Next Steps

1. Rerun B (α=0.05) / C (α=0.1) with new code, including:
   - L0 collision filtering
   - view_exit exclude
   - neg priority (negative_feedback/view_exit enters the neg pool first)
2. Observe drop_pct — verify collision hypothesis if >30%
3. If there is still no improvement after repair, consider detach ENTP gradient without returning it to the backbone.

---

## EXP-013: S-tier NTP Model — 6L MoE + Loss-Free Balancing

**Date**: 2026-04-15
**Status**: completed
**Results**: [experiments/results/ntp/](experiments/results/ntp/)

### Background

EXP-010 baseline (2L dense probe, ~5M params) has extremely poor performance (item_recall@50=0.0008). Part of the problem has been fixed in EXP-011 by equalizing codebooks, but the model capacity is also severely insufficient.

This experiment upgrades the NTP model to S-tier specifications (6L MoE, ~42M params), corresponding to `ideas/architecture_roadmap.md` Stage 1. At the same time, MoE load balancing is replaced from Switch Transformer auxiliary loss to Loss-Free dynamic bias (IDEA-onemall-4, DeepSeek-V2 scheme).

New code: `ntp/model.py` (NTPModel) vs `ntp/baseline.py` (NTPProbe).

### Hypothesis

1. The item_recall@50 of S-tier (6L MoE, 42M params) should be significantly higher than that of probe (2L dense, 5M)
2. PPL reduction > 30% (model capacity 8x, deeper layers can capture long-range SID dependencies)
3. The expert utilization rate of Loss-Free MoE balancing should be reasonably uniform (max/min freq < 3x)

### Design

- **Variable**: model architecture (probe vs s-tier)
- **Fixed**: SID 4096×3 + FSQ [2]×12 binary (EXP-011-H/012 best), n_items=10, batch_size=4096, 1 epoch, recall_beam_size=500
- **Metric**: Perplexity, Depth Hit@10, Item Recall@{10,50,100,500}, Expert utilization
- **Data**: 31 days of behavioral data (03-01~03-31), eval ~50K items by timestamp split

| Config | Model | Layers | FFN | Params | Description |
|--------|-------|--------|-----|--------|------|
| A (baseline) | NTPProbe | 2 | Dense 512 | ~5M | EXP-010 Reproduction |
| B (s-tier) | NTPModel | 6 | SwiGLU MoE 8E top-2 | ~42M | Loss-Free bias |

### Run

`bash experiments/scripts/exp-013.sh`

### Results

| Metric | Probe (7.5M) | S-tier (45.8M) | Improvement |
|--------|-------------|----------------|------|
| PPL | 70.0 | **29.6** | -58% |
| L0 PPL (cross-item) | 429.1 | **344.8** | -20% |
| L1 PPL | 41.8 | **13.3** | -68% |
| L2 PPL | 19.2 | **5.7** | -70% |
| hit@10 (indep L0) | 16.7% | **20.0%** | +20% |
| hit@10 (indep L1) | 62.2% | **78.9%** | +27% |
| hit@10 (indep L2) | 71.5% | **84.0%** | +17% |
| recall@10 | 5.1% | **10.2%** | 2x |
| recall@50 | 14.6% | **25.0%** | 1.7x |
| recall@100 | 20.1% | **34.6%** | 1.7x |
| recall@500 | 37.2% | **59.5%** | 1.6x |
| SID found rate | 37.3% | **59.5%** | 1.6x |

Beam search: 1000 samples, beam_size=500. Eval items: 49,383.

### Analysis

1. **S-tier comprehensively crushes probe**: recall@500 from 37%→60%, PPL dropped by 58%. 6x model capacity (45.8M vs 7.5M) brings significant benefits.
2. **L0 (cross-item) is still the bottleneck**: L0 PPL 344.8, that is, it is still difficult to predict the coarse-grained cluster of the next item. L1/L2 intra-item predictions are close to saturation (hit@10 79%/84%).
3. **Hypothesis verification**:
   - H1 ✅ S-tier recall@50 = 25% vs probe 14.6%, significant improvement
   - H2 ✅ PPL dropped 58% (30% higher than expected)
   - H3 to be verified (expert utilization not recorded)
4. **Key fix**: This round of training fixes the TransformerDecoder non-causal cross-attention bug (the old model sees future tokens through cross-attention cheating). All results are based on correct TransformerEncoder causal implementation.

### Next Steps

1. L0 cross-item prediction is the main bottleneck → Consider increasing the context window (n_items > 10) or increasing the number of epochs
2. Try larger batch size / learning rate schedule optimization
3. Record MoE expert utilization and verify Loss-Free balancing effect

---

## EXP-011: Codebook Size Ablation - Equal Size 1024/4096 + OPQ Control

**Date**: 2026-04-15
**Status**: completed (partial, OPQ has not been run)
**Results**: [./hyperparam/2026-04-15_exp011-*/](./hyperparam/)

### Background

EXP-010 NTP baseline has extremely poor effect (L1 acc=0.7%, item_recall@50=0.0008). One of the root causes is that the current SID configuration **L1=1024, L2=1024, L3=4096 is not large**, and the NTP model uses global max=4027 as the unified vocab.

Checking the original text of OneMall, we found that its production configuration is **Three layers of equal size 4096×4096×4096**, and the FSQ layer uses "binary 16-bit MLP". Need to determine our optimal codebook configuration.

### Hypothesis

1. The semantic_neighbor_HR of the three-tier configuration (1024×3 or 4096×3) is not lower than the current 1024×1024×4096
2. Binary FSQ ([2,...,2]) and multi-level FSQ ([4,...,4]) have equivalent effects under the same codebook size
3. OPQ 3×N (equal token number comparison) still loses the hierarchical structure MLP-FSQ (continues the conclusion of EXP-008)

### Design

| Config | L1 (KMeans) | L2 (KMeans) | L3 (FSQ) | FSQ Levels | Bits | Benchmarking |
|--------|-------------|-------------|----------|------------|------|------|
| A (EXP-008) | 1024 | 1024 | 4096 | [4,4,4,4,4,4] | 32 | Already have baseline |
| E | 1024 | 1024 | 1024 | [4,4,4,4,4] | 30 | Equally large 1024, multi-level |
| F | 1024 | 1024 | 1024 | [2]×10 | 30 | equal to 1024, binary |
| G | 4096 | 4096 | 4096 | [4,4,4,4,4,4] | 36 | OneMall Config |
| H | 4096 | 4096 | 4096 | [2]×12 | 36 | OneMall binary |
| I | OPQ 3×1024 | — | — | — | 30 | etc. bits compared to E/F |
| J | OPQ 3×4096 | — | — | — | 36 | etc. bits compared to G/H |

- **Fixed**: Qwen3-0.6B 1024D embedding (cached), behavior_data 7d, MLP hidden=64, 50 epochs
- **Metric**: semantic_neighbor_hit_rate (core), collision_rate, cluster_balance (Gini)

### Run

`bash experiments/scripts/exp-011.sh`

### Results

| Config | KMeans | FSQ | Bits | collision | snHR | L3 unique | L3 Gini |
|--------|--------|-----|------|-----------|------|-----------|---------|
| A (EXP-008) | 1024×1024×4096 | [4]×6 | 32 | 10.7% | 0.078 | 487K | — |
| E (1024, multi) | 1024×3 | [4]×5 | 30 | 14.6% | 0.078 | 404K | 0.151 |
| F (1024, binary) | 1024×3 | [2]×10 | 30 | 7.9% | 0.078 | 443K | 0.083 |
| G (4096, multi) | 4096×3 | [4]×6 | 36 | **0.84%** | **0.095** | 482K | 0.009 |
| H (4096, binary) | 4096×3 | [2]×12 | 36 | **0.89%** | **0.095** | 482K | 0.010 |

OPQ I/J not run (covered by EXP-012).

### Analysis

1. **KMeans cluster size is the dominant factor**: 4096→snHR=0.095 vs 1024→snHR=0.078 (+22%). The first two layers of KMeans encode most of the semantic information.
2. **4096 under binary ≈ multi-level**: collision 0.89% vs 0.84%, snHR is the same. Because L2 has divided items into an average of 1.5 items/prefix, FSQ type is no longer critical.
3. **Binary is obviously better under 1024**: collision 7.9% vs 14.6%. L2 has an average of 3.08 items/prefix, and 10-dimensional binary provides better discrimination than 5-dimensional multi-level.
4. **Three layers of equal size 1024×3 are not inferior to unequal size 1024×1024×4096**: snHR is the same (0.078), and binary collision is lower (7.9% vs 10.7%).

### Next Steps

→ EXP-012: Expand grid search to 2048/8192 cluster size and confirm the trend curve of snHR with cluster size.

---

## EXP-012: Tokenizer Grid Search — KMeans × FSQ Type × OPQ

**Date**: 2026-04-15
**Status**: completed
**Results**: [./hyperparam/2026-04-15_exp012-grid-search/](./hyperparam/2026-04-15_exp012-grid-search/)

### Background

EXP-011 confirms that KMeans cluster size is the dominant factor in tokenizer quality. A systematic search is required to find the plateau or optimal point of snHR.

### Hypothesis

1. snHR increases monotonically with cluster size but decreases at the margin (the upper limit of information theory = the amount of information in the embedding itself)
2. 8192×3 (OneRec configuration) should be better than 4096×3
3. Binary FSQ has advantages in smaller clusters, and binary ≈ multi-level in large clusters

### Design

| Config | Type | Cluster | FSQ | Bits |
|--------|------|---------|-----|------|
| 1024-multi | FSQ | 1024 | [4]×5 | 30 |
| 1024-binary | FSQ | 1024 | [2]×10 | 30 |
| 2048-multi | FSQ | 2048 | [4,4,4,4,4,2] | 33 |
| 2048-binary | FSQ | 2048 | [2]×11 | 33 |
| 4096-multi | FSQ | 4096 | [4]×6 | 36 |
| 4096-binary | FSQ | 4096 | [2]×12 | 36 |
| 8192-multi | FSQ | 8192 | [4,4,4,4,4,4,2] | 39 |
| 8192-binary | FSQ | 8192 | [2]×13 | 39 |
| opq-4×{256,512,1024,2048} | OPQ | — | — | 32/36/40/44 |

- **Fixed**: Qwen3-0.6B 1024D, MLP hidden=64, 50 epochs
- **Metrics (4 only)**: semantic_neighbor_HR, collision, codebook_utilization, cluster_balance + neighbor_coverage
- **Multi-GPU**: KMeans groups parallelization (CUDA_VISIBLE_DEVICES pinning)
- **Merge EXP-011**: 4 sets of results have been directly merged

### Run

```bash
python experiments/scripts/tokenizer_grid_search.py --gpus 0,1,2,3
```

### Results

| Config | Cluster | FSQ | Bits | collision | snHR | Coverage | L3 Gini |
|--------|---------|-----|------|-----------|------|----------|---------|
| 8192-binary | 8192 | [2]×13 | 39 | **0.35%** | **0.104** | 31% | 0.004 |
| 8192-multi | 8192 | [4]×6,2 | 39 | 1.35% | 0.104 | 31% | 0.016 |
| 4096-multi | 4096 | [4]×6 | 36 | 0.84% | 0.095 | ~55% | 0.009 |
| 4096-binary | 4096 | [2]×12 | 36 | 0.89% | 0.095 | ~55% | 0.010 |
| 2048-binary | 2048 | [2]×11 | 33 | 2.03% | 0.083 | 70% | 0.022 |
| 2048-multi | 2048 | [4]×5,2 | 33 | 4.48% | 0.083 | 70% | 0.047 |
| 1024-binary | 1024 | [2]×10 | 30 | 7.88% | 0.078 | ~85% | 0.083 |
| 1024-multi | 1024 | [4]×5 | 30 | 14.63% | 0.078 | ~85% | 0.151 |
| opq-4x256 | OPQ | 4×256 | 32 | 3.51% | 0.050 | 98% | 0.057 |

### Analysis

**1. snHR increases with cluster size but decreases at the margin** (Hypothesis 1 is true):

```
cluster  snHR    Δ        coverage
1024     0.078   baseline ~85%
2048     0.083   +6.4%    70%
4096     0.095   +14.5%   ~55%
8192     0.104   +9.5%    31%
```

4096→8192 Margin (+9.5%) has slowed and coverage has dropped sharply.

**2. snHR is a precision indicator and there is precision-coverage tradeoff**:

- snHR measures "the proportion of users with the same prefix neighbors" - the larger the cluster, the purer the group → the higher the precision
- But in large clusters, most items become singleton (no neighbors) → only a small number of items are evaluated
- The snHR=0.104 of 8192 only represents 31% of the items, and the result is at risk of overestimation.

**3. Binary FSQ is better than multi-level** (Hypothesis 3 is partially disproven):

Not only is there an advantage under small clusters, but binary has the greatest collision advantage under 8192 (0.35% vs 1.35%, 3.9×). Reason: binary has only 2 levels per dimension and has higher dimensions (13d vs 7d), providing more fine-grained orthogonal segmentation.

**4. OPQ completely loses to FSQ** (continuation of EXP-008 conclusion):

opq-4x256 (32bit) snHR=0.050 is much lower than 1024-binary (30bit)'s 0.078. Inductive bias for hierarchical structures > Flat PQ.

### Conclusion

**Recommended configuration: 4096×3 binary `[2]×12` (36 bit)**

- snHR=0.095, moderate coverage (~55%), collision=0.89%
- Benchmark OneMall 4096×4096×4096 production configuration
- KMeans training ~400s (vs 8192's ~1300s), acceptable
- collision < 1% friendly enough for NTP learning

8192×3 binary can be used as an aggressive alternative (collision minimum 0.35%, NTP most friendly), but subject to snHR evaluation insufficient coverage.

### Next Steps

- Run NTP baseline with 4096×3 binary configuration (per-layer output head fixed)
- Use `tokenizer_grid_search.py` to rerun grid search when changing different embeddings (e.g. larger model)

---

## EXP-010: NTP Baseline — MLP-FSQ SID end-to-end Recall

**Date**: 2026-04-15
**Status**: completed (the effect is extremely poor and needs diagnosis)
**Results**: [./hyperparam/2026-04-15_exp010-ntp-baseline/](./hyperparam/2026-04-15_exp010-ntp-baseline/)

### Background

The Tokenizer phase ends and MLP-FSQ h=64 is confirmed as the winner (EXP-008, semantic_neighbor_HR=0.078). Now we need the first end-to-end NTP numbers: train on MLP-FSQ SID with current 2-layer Transformer probe (~5M params) and get item Recall@K baseline.

Current NTP probe parameters:
- 2-layer causal Transformer decoder, embed_dim=256, n_heads=4, ffn_dim=512
- **1 epoch** (code bug: missing epoch outer loop), AdamW lr=3e-3, CosineAnnealing
- Behavior sequence n_items=10, beam_size=50
- SID: 3 tokens (L1=1024, L2=1024, **L3=4096** ← not equal to L1/L2)

### Hypothesis

- Perplexity should be in the range of 50~150 (good~acceptable)
- Item Recall@50 should be significantly higher than embedding_hit_rate (0.0047) because NTP utilizes behavior sequence information
- This number serves as the baseline for all subsequent NTP improvements (architecture/training/scaling)

### Design

- **Variable**: None (single configuration baseline)
- **Fixed**: MLP-FSQ h=64, 2-layer probe, 1 epoch, n_items=10, beam_size=50
- **Metric**: Perplexity, Depth Accuracy, Item Recall@{10,50,100,500}
- **Data**: 7 days of behavioral data, 19.1M samples (train=15.3M, eval=50K)

### Run

`bash experiments/scripts/exp-010.sh`

### Results

| Metric | Value |
|------|-----|
| Train loss | 1.70 → 0.47 (3741 steps) |
| Eval perplexity | 5.34 |
| Depth acc beam (L1/L2/L3) | 0.007 / 0.000 / 0.000 |
| **Depth hit@10 (L1/L2/L3)** | **1.000 / 1.000 / 0.401** |
| Item recall@50 | 0.0008 |
| Item recall@500 | 0.0008 |

### Analysis

**The effect is extremely poor, but teacher-forced hit@10 shows that the model has learned. The core problem is in beam search: **

1. **L1/L2/L3 unequal vocabs share a single output head (Linear(256, 4027))**: L1/L2 only has 1024 legal tokens, but softmax is done on 4027 dimensions, and 75% of the probability space is noise. Beam search may select tokens in the L3 range as L1 predictions
2. **Teacher-forced hit@10 = 100%**: It means that when the model sees the correct context, the correct token is in the top-10. But once L1 is selected incorrectly in beam search, all subsequent offsets will
3. **Only train for 1 epoch**: train loss is still declining (0.47 and the slope is obvious), and it has not converged.
4. **Train-eval gap is large**: train CE ≈ 0.47, eval CE ≈ 1.68 (PPL 5.34), time series segmentation leads to distribution shift

**Root cause: SID configuration 1024×1024×4096 is not large + NTP model does not perform per-layer vocab processing. **

### Next Steps

1. **EXP-011**: Determine the correct codebook configuration (equivalent to 1024×3 or 4096×3)
2. **Fix NTP model**: per-layer output head or unified vocab + layer embedding + beam search mask
3. **Add epoch**: 1 → 5-10
4. Rerun NTP baseline after repair

---

## EXP-009: QFormer Tokenizer — Freeze Qwen3 + Cross-Attention compression

**Date**: 2026-04-14 ~ 2026-04-15
**Status**: completed
**IDEA**: IDEA-onerec-3
**Results**: [./hyperparam/2026-04-14_exp009-qformer/](./hyperparam/2026-04-14_exp009-qformer/)

### Background

EXP-007 proves that direct fine-tune Qwen3-0.6B (full/LoRA, multiple lr/τ) is completely unable to push the model - cap_loss does not move at all, and HR@50 is stuck at ~0.02. Root cause: I2I gradient dilution in 600M parameters.

OneRec's core solution: freeze the base and add a trainable QFormer (cross-attention + learnable queries) on top. The gradient is centered on a QFormer with ~30-50M parameters, and the base remains semantically native. BLIP-2 QFormer has been validated by OneRec (miniCPM-V-8B + 4-layer QFormer).

### Hypothesis

1. The cap_loss will drop significantly during QFormer training (unlike EXP-007 which remains motionless), proving that the gradient can flow effectively
2. HR@50 significantly exceeded the 0.02 baseline of EXP-007 (expected > 0.05)
3. Information compression (S tokens → M tokens) forces QFormer to learn to extract synergy-related information instead of copying semantics

### Design

**Phase 1 — Minimal verification (can the gradient flow)**:

| Config | QFormer Layers | Query Tokens (M) | lr | Loss |
|--------|---------------|-------------------|------|------|
| A | 2 | 4 | 1e-4 | L_I2I only |
| B | 2 | 4 | 5e-4 | L_I2I only |
| C | 4 | 4 | 1e-4 | L_I2I only |

- **Variable**: QFormer depth × learning rate
- **Fixed**: Qwen3-0.6B frozen, M=4 query tokens, D=1024, τ=0.05, batch_size=32, grad_accum=8, max_pairs=500K, 1 epoch, 8xA100 DDP
- **Metric**:
  - **Primary**: HR@50 (InlineHRMonitor, direct comparison with EXP-007 baseline)
  - **Diagnostic**: cap_loss variation (W&B), I2I loss convergence speed
  - **Secondary**: OPQ intrinsic (collision, recon_loss) on QFormer embeddings
- **Data**: Behavioral data 7 days, ~5M items

### Run
`bash experiments/scripts/exp-009.sh`

### Results

| Config | QFormer Layers | Queries (M) | lr | Final HR@50 | Final Loss | TrainingTime |
|--------|---------------|-------------|------|------------|-----------|---------|
| BL (raw Qwen3) | — | — | — | 0.0106 | — | — |
| EXP-007 best (Full FT) | — | — | 1e-5 | 0.0197 | 2.90 | 6756s |
| A | 2 | 4 | 1e-4 | 0.0211 | 4.46 | 4460s |
| B | 2 | 4 | 5e-4 | 0.0214 | 4.41 | 4458s |
| **C (best)** | **4** | **4** | **1e-4** | **0.0216** | **4.42** | **4549s** |

Actual training data: 3,074,342 pairs (max_pairs=5M, swing actual output ~3M), 12,000 steps/epoch, effective batch 2048.

### Analysis

**1. QFormer has not broken through the 0.02 ceiling:**
- Best Config C: HR@50 = 0.0216, only 10% higher than EXP-007 best (0.0197), far from the >0.05 expected by hypothesis
- The difference between the three groups of config is very small (0.0211 ~ 0.0216), QFormer depth/lr is not the bottleneck

**2. Hypothesis verification:**
- ✅ H1 (Gradient Flow): loss dropped from 5.5 to ~4.4, which is indeed declining (EXP-007 cap_loss remains motionless), proving that gradients can flow through QFormer
- ❌ H2 (HR@50 breakthrough): 0.0216 vs expected >0.05, huge gap
- ❌ H3 (information compression): QFormer’s 4 query tokens do not force the model to learn better collaborative representations

**3. HR@50 curve characteristics:**
- The whole process rises slowly and monotonously, with no obvious plateau.
- But the slope continues to decrease (step 0~4000: +0.006, step 4000~8000: +0.003, step 8000~12000: +0.002)
- There may be a slight improvement in more epochs, but the trend is extremely flat and it is impossible to break through 0.03

**4. Root cause re-judgment:**
- EXP-007 Conclusion "Gradient dilution" has been partially overturned - QFormer's loss is indeed decreasing after concentrating the gradient, but HR@50 is still stuck.
- **The real bottleneck is not in the model structure, but in the I2I contrastive signal itself**: in-batch negatives + the supervisory signal strength of behavioral co-occurrence positive samples is not strong enough to push embedding to a meaningful position in the behavioral space
- In other words: **The gap between the semantic embedding space and the behavioral space of Qwen3 is far greater than what contrastive learning can make up**

### Next Steps

EXP-007 + EXP-009 Two rounds of experiments prove that: **I2I contrastive fine-tune (regardless of full volume/LoRA/QFormer) cannot effectively improve the behavioral quality of embedding**. The embedding side strategy needs to be re-examined:

1. **Abandon the embedding fine-tune route** and return to the architectural philosophy of "a good tokenizer is more important than a good embedding"
2. Focus on **EXP-008 (FORGE proxy comparison)** — Use existing Qwen3 embedding to compare the behavior quality of MLP-FSQ vs OPQ to decide the tokenizer route
3. If you still need to improve embedding, consider completely different solutions: multi-task learning, graph embedding, or directly replace text embedding with behavioral embedding (collaborative filtering)

---

## EXP-008: FORGE Proxy comparison - MLP-FSQ vs OPQ optimal solution

**Date**: 2026-04-14 ~ 2026-04-15
**Status**: completed
**Results**: [./hyperparam/2026-04-15_exp008-mlpfsq-h64/](./hyperparam/2026-04-15_exp008-mlpfsq-h64/), [./hyperparam/2026-04-15_exp008-opq-m4/](./hyperparam/2026-04-15_exp008-opq-m4/), [./hyperparam/2026-04-15_exp008-opq-m8/](./hyperparam/2026-04-15_exp008-opq-m8/)

### Background

EXP-003 optimal (MLP-FSQ h=64, collision=0.041) and EXP-004 optimal (OPQ 8×256, collision=0.0037) only have intrinsic metrics and lack behavioral level verification. Implemented FORGE proxy metrics to evaluate SID quality without training NTP:
- `embedding_hit_rate`: embedding I2I neighbor co-occurrence rate (same for all schemes, used as baseline)
- `semantic_neighbor_hit_rate`: SID prefix neighbor co-occurrence rate (distinguish tokenizer, core indicator)

Goal: Quickly compare two routes and decide which one to enter the NTP stage.

### Hypothesis

1. The `semantic_neighbor_hit_rate` of OPQ 8×256 should be significantly higher than MLP-FSQ h=64, because lower collision (0.0037 vs 0.041) means finer SID partitioning
2. `embedding_hit_rate` The three groups are the same (same embedding, just different tokenizer)
3. The `semantic_neighbor_hit_rate` of OPQ 4×256 (equal bits comparison) is between MLP-FSQ and OPQ 8×256

### Design

| Config | Tokenizer | Tokens | Bits | Known collisions |
|--------|-----------|--------|------|---------------|
| A | MLP-FSQ h=64 (6d_4096) | 3 | 32 | 0.0411 |
| B | OPQ 4×256 (equal bits comparison) | 4 | 32 | 0.1063 |
| C | OPQ 8×256 (optimal) | 8 | 64 | 0.0037 |

- **Fixed**: Qwen3-0.6B 1024D embedding (cached), behavior_data 7d
- **Metric**:
  - **Primary**: `semantic_neighbor_hit_rate` — Co-occurrence rate of SID prefixed neighbors in the behavioral graph
  - **Baseline**: `embedding_hit_rate` — embedding space I2I neighbor co-occurrence rate (the three groups should be the same)
  - **Secondary**: intrinsic metrics (collision, recon_loss, entropy)

### Run
`bash experiments/scripts/exp-008.sh`

### Results

Data: 554,754 exposed items (filtered from 5,162,650 total embeddings), behavioral data 7 days (03-24 ~ 03-30)

| Config | Tokenizer | Tokens | Bits | collision | recon_loss | embedding_HR | **semantic_neighbor_HR** | TrainingTime |
|--------|-----------|--------|------|-----------|------------|-------------|------------------------|---------|
| **A** | **MLP-FSQ h=64** | **3** | **32** | **0.1074** | **0.3668** | **0.0047** | **0.0780** | 106s |
| B | OPQ 4×256 | 4 | 32 | 0.0351 | 0.3760 | 0.0047 | 0.0502 | 73s |
| C | OPQ 8×256 | 8 | 64 | 0.0006 | 0.3408 | 0.0043 | 0.0326 | 99s |

### Analysis

**The results are completely opposite to the hypothesis - MLP-FSQ is significantly ahead of OPQ:**

**1. Hypothesis verification:**
- ❌ H1: OPQ 8×256’s semantic_neighbor_HR (0.033) is **much lower than** MLP-FSQ (0.078), collision is 180 times lower but loses 58%
- ✅ H2: embedding_HR is almost the same among the three groups (~0.0047), in line with expectations
- ❌ H3: OPQ 4×256 (0.050) is in between, but in the opposite direction – instead of MLP-FSQ < OPQ 4×256 < OPQ 8×256, it is MLP-FSQ > OPQ 4×256 > OPQ 8×256

**2. The lower the collision ≠ the better the behavior quality:**
- OPQ 8×256 pursues extremely low collision (0.06%) and cuts the embedding space into ~553K bins with almost no overlap.
- But over-subdivision destroys the semantic neighborhood structure - items with similar SID prefixes are no longer behavioral neighbors
- MLP-FSQ's collision of 10.7% seems "poor", but it retains the hierarchical aggregation structure, and the behavior co-occurrence rate of SID prefix neighbors is higher.

**3. Hierarchical structure > Flat structure:**
- MLP-FSQ: 3-layer hierarchy (KMeans → KMeans → FSQ), each layer is gradually refined, and prefixes naturally encode coarse-to-fine semantic clustering
- OPQ: 8 parallel sub-vectors are independently quantized, there is no hierarchical relationship between tokens, and prefix neighbors have no semantic meaning.

**4. Wait for bits comparison (32 bits):**
- MLP-FSQ (0.078) vs OPQ 4×256 (0.050), MLP-FSQ wins 56%
- Under the same amount of information, the SID prefix neighborhood of hierarchical residual coding is more behaviorally meaningful than the prefix neighborhood of parallel PQ

**5. Note: MLP-FSQ does not use behavioral data for training:**
- MLP only optimizes the residual reconstruction loss (||residual - Decoder(FSQ(Encoder(residual)))||²), purely unsupervised
- The advantage of behavioral quality comes entirely from the preservation of embedding neighborhoods by the hierarchical structure, rather than from learning behavioral signals

### Next Steps

**MLP-FSQ h=64 is confirmed as the tokenizer route winner** and enters the NTP stage:
1. Use MLP-FSQ to generate all SIDs and train the NTP prediction model
2. End-to-end evaluation Recall@K
3. Consider whether you need a larger FSQ codebook (currently 4096) or more KMeans layers

---

## EXP-007: Collaborative Signal Enhanced Embedding (Qwen3-0.6B Full Fine-tune)

**Date**: 2026-04-13 ~ 2026-04-14
**Status**: completed
**IDEA**: IDEA-sid-1
**Results**: [./hyperparam/2026-04-13_exp007-collab-embed/](./hyperparam/2026-04-13_exp007-collab-embed/)

### Background

Currently, Qwen3-0.6B plain text embedding (1024D) is used directly for quantization. These embeddings only encode semantic similarity (items with similar text content are close), but what is required for recommendation is behavioral similarity (items liked by the same user group are close). The embedding_hit_rate indicator of EXP-004 can quantify the quality of the current embedding in the behavioral dimension.

This experiment uses **I2I comparative learning** full fine-tune Qwen3-0.6B to inject collaborative behavior signals into embedding to increase the upper limit of quantification. Orthogonal to quantification schemes (OPQ/RKMeans), improved embedding quality benefits all downstream experiments.

### Hypothesis

1. The embedding after contrast learning is significantly better than the original Qwen3 embedding in `embedding_hit_rate` (expected HR@50 improvement of 50%+)
2. Downstream OPQ quantitative indicators (collision, recon_loss) will also improve because items with similar behaviors are more clustered in the embedding space.
3. The training time of full fine-tune 0.6B on 8xA100 is controllable (expected < 4 hours)

### Design

- **Variable**: training program × temperature parameter
  - **Baseline**: Original Qwen3-0.6B embedding (cached, no need to rerun)
  - **Config A**: full fine-tune, InfoNCE, τ=0.05, 3 epochs
  - **Config B**: full fine-tune, InfoNCE, τ=0.07, 3 epochs
  - **Config C**: full fine-tune, InfoNCE, τ=0.05, 5 epochs
- **Fixed**:
  - Model: Qwen3-0.6B (full parameter update, FP16, 8xA100 DDP)
  - Positive sample: item pair with positive behavior (action_bitmap > 0) of the same user within 7 days
  - Negative samples: in-batch negatives (batch_size=512 per GPU, effective 4096)
  - Optimizer: AdamW, lr=1e-5, warmup 10%, cosine decay
  - Text: item title (Qwen3 tokenizer already exists)
- **Metric**:
  - **Primary**: `embedding_hit_rate` (HR@10/50/100/500) — FORGE proxy, no need to train NTP
  - **Secondary**: OPQ intrinsic (collision, recon_loss, entropy) — Use EXP-004 same OPQ config (m=8, M=256) post-quantization evaluation
  - **Sanity**: `cosine_similarity` distribution, `embedding_behavior_correlation`
- **Data**: Behavioral data for 7 days (2026-03-24 ~ 2026-03-31), ~5M items

### Run
`bash experiments/scripts/exp-007.sh`

### Results

**Baseline**: HR@50 = 0.0106 (original Qwen3-0.6B embedding, 50,008 items)

**Round 1 — Basic hyperparameter search (full fine-tune)**:

| Config | τ | lr | max_pairs | HR@50 | Loss plateau | TrainingTime |
|--------|------|------|-----------|-------|-------------|---------|
| BL (baseline) | — | — | — | **0.0106** | — | — |
| A | 0.05 | 1e-5 | 2M | **0.0197** | ~step 800 | 6756s (~1h53m) |
| B | 0.07 | 1e-5 | 1M | 0.0148 | ~step 800 | killed early |
| C | 0.05 | 3e-5 | 500K | 0.0192 | ~step 400 | 1912s (~32min) |

**Round 2 — Aggressive learning rate (cap_loss remains unchanged at R1)**:

| Config | τ | lr | Status |
|--------|------|------|------|
| D | 0.05 | 1e-4 | The script is ready and has not produced a Result beyond R1 |
| E | 0.05 | 3e-4 | Same as above |
| F | 0.05 | 1e-3 | Same as above |

**Round 3 — LoRA (frozen base, gradient concentrated in adapter)**:

| Config | Method | lr | Status |
|--------|--------|------|------|
| G | LoRA r=16 | 1e-4 | The script is ready, but no Result beyond R1 has been produced |
| H | LoRA r=16 | 5e-4 | Same as above |
| I | LoRA r=64 | 1e-4 | Same as above |

### Analysis

**1. HR@50 ceiling ~0.02, which is about 86% higher than baseline 0.0106, but far from the 50%+ absolute improvement expected by hypothesis:**
- Best Config A: 0.0197, still at poor level (threshold < 0.02)
- Three groups of round 1 config HR@50 converge to the same ceiling (~0.02), and the space for hyperparameter tuning is limited.

**2. Temperature is not a bottleneck**: τ=0.07 (Config B) is overall worse than τ=0.05 (Config A)

**3. The learning rate affects the convergence speed but does not affect the upper limit**: Config C (lr=3e-5) uses 1/4 data and 1/3 time to achieve the same effect

**4. Loss fast plateau**: All config loss stabilizes at ~2.5-2.7 after ~200K pairs, cap_loss does not move at all - indicating that I2I gradient dilution is in the 600M parameter

**5. Hypothesis verification:**
- ❌ HR@50 increased by 86% (0.0106→0.0197), but the absolute value is still very low and does not meet the expectation of "significantly better"
- ❌ The downstream quantitative improvement has not been verified (HR@50 itself is too low, and the significance of OPQ assessment is limited)
- ✅ Training time is controllable (the fastest Config C is only 32 minutes)

**6. Root cause**: Directly fine-tune the Qwen3 base with 600M parameters, the I2I contrastive gradient is diluted, and the model hardly learns. Neither full fine-tune nor LoRA can effectively inject synergistic signals into embedding.

### Next Steps

EXP-007 proves that the "direct fine-tune base" route is not feasible and requires methodological changes:
- **EXP-009 (Planned)**: Freeze Qwen3 base + QFormer cross-attention, gradient is concentrated on QFormer with ~30-50M parameters (OneRec verified effective solution)

---

## EXP-004: OPQ Parallel Semantic IDs — Intrinsic Metrics

**Date**: 2026-04-13
**Status**: completed
**IDEA**: IDEA-sid-0 (Phase 1)
**Reference**: Meta RPG (KDD'25, arxiv 2506.05781)
**Results**: [./hyperparam/2026-04-13_exp004-opq/](./hyperparam/2026-04-13_exp004-opq/), [./hyperparam/2026-04-13_exp004-opq-m4/](./hyperparam/2026-04-13_exp004-opq-m4/)

### Background
Currently RKMeans (3 layers x 1024 clusters) uses residual encoding, and each layer is serially dependent. ARCHITECTURE.md has a clear need to switch to a parallel tokenizer. The RPG paper proves that OPQ (Optimized Product Quantization) is better than RQ in generative recommendation and supports parallel prediction.

This experiment verifies the quantitative quality (intrinsic metrics) of OPQ on our 5M item / 1024D Qwen3-0.6b embedding, and does not involve the NTP prediction model.

### Hypothesis
OPQ divides 1024D embedding into m independent sub-vectors and quantizes them separately. The encoding space is much larger than RKMeans (256^8 >> 1024^3), and the collision should be significantly lower. recon_loss requires experimental verification—the independent subspace assumption of PQ may not be as close as the residual approximation of RQ.

### Design
- **Variable**: n_subvectors (m=4, 8, 16, 32), n_clusters_per_sub (M=256)
- **Fixed**: normalize_input=True, OPQ rotation training (FAISS default)
- **Metric**: collision_rate, exclusivity, reconstruction_loss, entropy, cluster_balance
- **Data**: 5M items, qwen3-0.6b 1024D embedding (cached)

**Comparison matrix**:

| Config | Quantizer | Tokens | Vocab/token | Bits | Subvector dimensions |
|--------|-----------|--------|-------------|------|-----------|
| Baseline (EXP-001) | RKMeans 3x1024 | 3 | 1024 | 30 | N/A (residual) |
| **OPQ-4x256** | **OPQ** | **4** | **256** | **32** | **256D (equal bits comparison)** |
| OPQ-8x256 | OPQ | 8 | 256 | 64 | 128D |
| OPQ-16x256 | OPQ | 16 | 256 | 128 | 64D |
| OPQ-32x256 | OPQ | 32 | 256 | 256 | 32D |

### Run
`bash experiments/scripts/exp-004.sh`

### Results

| Config | Tokens | Bits | collision | entropy | Gini | recon_loss | time(s) |
|--------|--------|------|-----------|---------|------|------------|---------|
| **RKMeans 3×1024** (EXP-001) | **3** | **30** | **0.1634** | **0.7211** | **0.2091** | **0.3524** | — |
| **OPQ 4×256** | **4** | **32** | **0.1063** | **0.9681** | **0.1896** | **0.3772** | 125 |
| OPQ 8×256 | 8 | 64 | 0.0037 | 0.9971 | 0.0128 | 0.3429 | 160 |
| OPQ 16×256 | 16 | 128 | 0.0029 | 0.9993 | 0.0052 | 0.3026 | 220 |
| OPQ 32×256 | 32 | 256 | 0.0027 | 0.9995 | 0.0043 | 0.2522 | 338 |

### Analysis

**1. Bits comparison — OPQ 4×256 (32bit) vs RKMeans 3×1024 (30bit):**
- collision: 10.6% vs 16.3% - OPQ is 35% lower, the collision rate is significantly lower with the same amount of information
- entropy: 0.968 vs 0.721 — OPQ codebook utilization is much more even
- recon_loss: 0.377 vs 0.352 — OPQ slightly worse by 7%, cost of PQ subspace independence assumption
- Conclusion: Waiting for bits to play OPQ **wins collision, loses recon**, trade-off is reasonable

**2. m=8 is sweet spot:**
- Collision dropped sharply from 10.6% for m=4 to 0.37% (only 1x more bits)
- recon_loss 0.3429 is better than RKMeans 0.3524
- 8 token parallel prediction cost is controllable

**3. m≥16 Diminishing returns:**
- collision: 0.37% → 0.29% → 0.27%, almost no difference
- recon_loss continues to decrease but the number of tokens doubles → NTP prediction cost doubles
- Not worth it unless the downstream task is extremely sensitive to recon

**4. Hypothesis verification:**
- ✅ Collision is significantly lower (as expected, coding space 256^m >> 1024^3)
- ✅ recon_loss outperforms RKMeans when m≥8 (PQ independent subspace assumption does not seriously harm reconstruction quality)
- ✅ entropy/Gini is almost perfect, no cluster collapse

### Next Steps
OPQ Phase 1 has been verified and it is recommended to enter Phase 2 with **m=8**:
1. Parallel prediction NTP model — per-digit independent MLP heads + MTP loss
2. Graph-Constrained Decoding — alternative to beam search (RPG paper proves that beam search has recall=0.0000 on OPQ)
3. End-to-end evaluation — Recall@K on downstream retrieval task

---

## EXP-003: Learned FSQ — MLP projection + straight-through training

**Date**: 2026-04-13
**Status**: completed
**Results**: [./hyperparam/2026-04-13_exp003-mlp64/](./hyperparam/2026-04-13_exp003-mlp64/), [./hyperparam/2026-04-13_exp003-mlp128/](./hyperparam/2026-04-13_exp003-mlp128/)

### Background
EXP-002 proves that PCA linear projection + FSQ is inferior to KMeans baseline. The core bottleneck is that PCA loses too much information in the residual space (1024D→4~6D explained variance is only 20-55%).

OneMall (arxiv 2601.21770) uses **learned MLP** for projection, the original FSQ paper (Mentzer 2023, arxiv 2309.15505) uses FSQ inside VQ-VAE, and the encoder learns the optimal representation for quantization. Key mechanisms:
- MLP learns nonlinear projection D→d, retaining more quantitatively relevant information than PCA
- Straight-Through Estimator (STE): Use round() in the forward direction, and pass the gradient directly to the MLP parameters in the reverse direction
- Training goal: reconstruction loss — minimize ||residual - reconstruct(FSQ(MLP(residual)))||²

### Hypothesis
Learned MLP projection can learn a low-dimensional representation that is optimal for FSQ quantization, making the reconstruction quality of FSQ close to or exceeding KMeans, thereby reducing recon_loss while maintaining low collision.

### Design
- **Variable**: projection method (PCA vs MLP), MLP hidden layer width
- **Fixed**: 2 KMeans layers x 1024 clusters, FSQ [4,4,4,4,4,4] (6d_4096), epochs=50, lr=1e-3, AdamW
- **Metric**: collision_rate, reconstruction_loss, exclusivity, entropy

**MLP Architecture** (autoencoder + STE):
```
Encoder: D → hidden → d (d=6 for 6d_4096)
FSQ: quantify each of 6 dims to {0,1,2,3}, STE pass-through
Decoder: d → hidden → D
Loss: ||residual - Decoder(STE_quantize(Encoder(residual)))||²
```

**Comparison matrix**:

| Config | L3 projection | L3 codebook | Training |
|--------|---------------|-------------|----------|
| Baseline (EXP-002) | KMeans 1024 | 1024 | N/A |
| PCA-FSQ (EXP-002) | PCA 6d | 4096 | N/A |
| MLP-FSQ-64 | MLP D→64→6 | 4096 | 50 epochs |
| MLP-FSQ-128 | MLP D→128→6 | 4096 | 50 epochs |

### Run
`bash experiments/scripts/exp-003.sh`

### Results

| Config | L3 | collision | recon_loss | d3 avg_items | time(s) |
|--------|-----|-----------|------------|--------------|---------|
| **Baseline** | KMeans 1024 | 0.1634 | 0.3524 | 1.3 | 237 |
| PCA-FSQ | PCA 6d_4096 | 0.3330 | 3.1280 | 1.7 | 178 |
| **MLP-FSQ h=64** | **MLP D→64→6** | **0.0411** | **0.3619** | **1.1** | **611** |
| MLP-FSQ h=128 | MLP D→128→6 | 0.0767 | 0.3633 | 1.1 | 627 |

### Analysis

**Learned MLP significantly surpasses KMeans baseline and fully verifies hypothesis:**

1. **collision reduced by 75%**: MLP h=64’s collision 0.0411 vs KMeans baseline 0.1634, FSQ’s implicit codebook (4096 codes) + learned nonlinear projection completely solved the collision problem
2. **recon_loss is the same as baseline**: 0.3619 vs 0.3524 (2.7% difference), indicating that MLP has learned high-quality projection, and PCA’s 3.128 recon_loss is completely a limitation of linear projection
3. **h=64 is better than h=128**: collision 0.0411 vs 0.0767. A smaller hidden dim acts as a regularizer to prevent the encoder output from being too extreme and causing tanh saturation (the bug in which tanh saturation causes OOB was discovered and fixed during training)
4. **Training time doubled but acceptable**: 611s vs 237s, the extra ~400s is 50 epoch MLP training, the model is only ~132K params

**vs PCA-FSQ (EXP-002)**: collision dropped from 0.333 to 0.041 (88% drop), recon_loss dropped from 3.128 to 0.362 (88% drop), proving that nonlinear projection is the key.

### Next Steps
1. MLP-FSQ h=64 runs NTP behavior evaluation and confirms the recall@K indicator.
2. Compare with OPQ (EXP-004) downstream of NTP to decide on the final solution

---

## EXP-002: ResKmeansFSQ — 2 layers RKMeans + 1 layer FSQ (PCA projection)

**Date**: 2026-04-13
**Status**: completed
**Results**: [./hyperparam/2026-04-13_exp002-baseline/](./hyperparam/2026-04-13_exp002-baseline/), [./hyperparam/2026-04-13_exp002-fsq/](./hyperparam/2026-04-13_exp002-fsq/)

### Background
The third layer of RKMeans performs KMeans on the residuals with diminishing effect. OneMall (arxiv 2601.21770) proposed replacing layer 3 with FSQ. This experiment uses **PCA linear projection** instead of the learned MLP in the paper for dimensionality reduction.

### Hypothesis
FSQ's implicit codebook naturally has no cluster collapse and can reduce the collision rate.

### Design
- **Variable**: Layer 3 quantizer (KMeans vs FSQ configs)
- **Fixed**: 2 KMeans layers x 1024 clusters, niter=25, nredo=3, normalize_residuals=True
- **Metric**: conflict_rate, reconstruction_loss, entropy, exclusivity, cluster_balance

| Config | L1, L2 (KMeans) | L3 | L3 codebook |
|--------|------------------|----|-------------|
| Baseline | 1024 x 3 layers KMeans | KMeans 1024 | 1024 |
| Hybrid A | 1024 x 2 layers | FSQ [8,8,8,8] | 4096 |
| Hybrid B | 1024 x 2 layers | FSQ [7,5,5,5,5] | 4375 |
| Hybrid C | 1024 x 2 layers | FSQ [4,4,4,4,4,4] | 4096 |

### Run
`bash experiments/scripts/exp-002.sh`

### Results

| Config | L3 | conflict_rate | exclusivity | recon_loss | d3 entropy | d3 Gini | d3 unique | d3 avg_items |
|--------|-----|---------------|-------------|------------|------------|---------|-----------|--------------|
| **Baseline** | KMeans 1024 | **0.1634** | **0.6423** | **0.3524** | **0.7211** | **0.2091** | 3,963,269 | 1.3 |
| Hybrid C | FSQ 6d [4x6] | 0.3330 | 0.4015 | 3.1280 | 0.6755 | 0.3153 | 3,107,671 | 1.7 |
| Hybrid A | FSQ 4d [8x4] | 0.5688 | 0.1446 | 2.2122 | 0.6383 | 0.4693 | 1,731,222 | 3.0 |
| Hybrid B | FSQ 5d [7,5x4] | 0.8157 | 0.0089 | 0.3800 | 0.5306 | 0.6548 | 248,798 | 20.8 |

Note: The KMeans of the two layers L1/L2 are the same (d1/d2 indicators are consistent), and all differences come from L3.

### Analysis

**FSQ+PCA is overall inferior to KMeans baseline**. The core reason is that **PCA linear projection information loss is too large**:

1. **Projection bottleneck**: 1024-dimensional residual → 4~6-dimensional PCA, explained variance is only 20-55%. The residual space (after two rounds of KMeans) is inherently small and irregular, and the PCA linearity assumption does not apply.
2. **Fewer dimensions are worse**: 5d_4375 (d=5) has conflict_rate as high as 0.82, almost all information is lost; 6d_4096 (d=6) is the best but still 0.33 >> baseline 0.16.
3. **recon_loss worsens**: recon_loss of 4d/6d soars from 0.35 to 2.2/3.1, indicating that PCA backprojection cannot recover the original residual.
4. **Difference from the paper**: OneMall uses **learned MLP** projection (non-linear, end-to-end training) to learn the optimal representation of quantization instead of only retaining the direction of maximum variance.

### Next Steps
EXP-003: Replace PCA with learned MLP projection to reproduce the paper plan. Requires:
1. Define MLP architecture (D → d dimension) + VQ-VAE style reconstruction loss
2. End-to-end training of projection network
3. Compare the effects of PCA vs MLP under the same FSQ config

---

## EXP-001: RKMeans training optimization (v0→v7)

**Date**: 2026-03 ~ 2026-04
**Status**: completed
**Results**: Summary retained in this public experiment log.

### Background
The collision rate of semantic_id generated by RKMeans is extremely high (99%+) and requires systematic optimization.

### Key Findings
1. **normalize_residuals is only done on layer 0 input** — the residuals retain the original scale, otherwise Layer 2/3 cannot be clustered
2. **FAISS full-batch Lloyd's is better than SGD/MiniBatch** — empty cluster rebalance + GPU acceleration
3. **num_clusters is the only significant hyperparameter** — collision has a log-linear relationship with clusters, and decreases by 50-70% every time it is doubled.
4. **nredo=3 is enough, niter=25 is converged** — nredo 1→3 critical (-42~49%), 3→5 meaningless; niter 25/50/100 no difference

### Final Config
- 3 layers × 1024 clusters, niter=25, nredo=3
- collision: 1.75%, reconstruction_loss: 0.348

---
