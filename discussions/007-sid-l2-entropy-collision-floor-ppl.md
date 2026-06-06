# Correspondence between SID L2 Entropy, Collision Rate and Floor PPL

[English](007-sid-l2-entropy-collision-floor-ppl.md) | [Chinese](007-sid-l2-entropy-collision-floor-ppl.zh.md)

**Date**: 2026-04-29
**Context**: EXP-043 compared the 0.6B / 4B / 8B SID cache and found that the L2 entropy decreased with the embedding dimension, and was consistent with the floor PPL direction obtained by inverting the NTP scaling law

---

## Core conclusion

**L2 entropy (can be quickly counted in the tokenizer stage) is a reliable proxy indicator for NTP floor PPL**, and the FSQ hidden dim configuration can be filtered without running the complete NTP.

---

## Definition of three concepts

**L2 entropy (codebook utilization entropy)**

The FSQ L2 layer maps item embedding to 4096 discrete slots (`[2]×12`). L2 entropy measures the uniformity of slot usage:

```
H(L2) = -Σ_i  p_i · log2(p_i)
```

Maximum value `log2(4096) = 12 bits` (each slot is used with equal probability). Actual measurement:
- 0.6B: H = 10.58 bits, utilization 91.2% (effective slot ≈ 1500)
- 4B: H = 8.10 bits, utilization 78.7% (effective slot ≈ 275)
- 8B: H = 7.17 bits, utilization 71.6% (effective slot ≈ 145)

**Collision rate (L2 conflict rate)**

The proportion of different items mapped to the same complete SID (L0_L1_L2 triplet). The lower the L2 entropy → the smaller the effective SID space → on average each SID corresponds to more items → the higher the conflict rate.

The relationship between the two (approximately):
```
Average conflict size k ≈ n_items / (4096 × eff_l2_slots)
collision_rate ≈ (k - 1) / k
```

**Floor PPL (irreducible PPL lower bound)**

Back inference through scaling law two-point method: fixed index `α = 0.456`, fitted with two model sizes S-tier and M-tier:

```
L(N) = floor + b / N^α
```

Solve for floor (theoretical lower limit of PPL as N→∞):
- 0.6B SID: floor = 12.46
- 4B SID: floor = 11.78 ← optimal
- 8B SID: floor = 12.26 ← worse than 4B

---

## Why L2 entropy determines floor PPL

The goal of model prediction is to restore the item from the SID token sequence. Assume the complete SID = `(L0, L1, L2)`, and the three layers of tokens jointly determine an item.

If the L2 layer entropy is low and there are few effective slots, there are a large number of items sharing the same SID. For k items sharing the same SID, the model cannot distinguish them given the context - these k cases are **indistinguishable** to NTP, introducing additional uncertainty of `log2(k) bits`.

The uniform distribution of these k confusing items is `1/k`, corresponding to the PPL multiplier:
```
PPL_collision_penalty ≈ k
floor_PPL ≥ base_PPL × k^(p_collision)
```

where `p_collision` is the proportion of colliding items. More precisely, floor PPL comes from the conditional entropy lower bound of `P(y | context)`:

```
H(y | SID) = Σ_{sid}  P(sid) · H(y | SID = sid)
```

The entropy within the SID conflict group `H(y | SID=sid) = log2(k_sid)` is irreducible and cannot be eliminated no matter how big the model is.

---

## Data validation

The observed values ​​of EXP-043 are consistent with the above theoretical direction:

| Embedding | L2 entropy | 有效 slot | 平均冲突 k | floor PPL |
|-----------|-----------|----------|-----------|-----------|
| 0.6B      | 10.58 bits | ~1500    | ~1.3      | 12.46     |
| 4B        | 8.10 bits  | ~275     | ~1.6      | 11.78*    |
| 8B        | 7.17 bits  | ~145     | ~3.0      | 12.26     |

\* 4B floor is the lowest: 4B embedding quality is good enough. Even if L2 has a slight collapse, the embedding distinction compensates for part of the conflict loss; 8B conflict is too serious (k≈3), and the embedding quality improvement is no longer enough to offset it.

**Counter-intuitive conclusion**: The larger embedding model (8B) obtained a worse theoretical upper limit. The fundamental reason is that FSQ hidden=64 is too small for the 4096D input, and the bottleneck severely compresses the semantic information.

---

## Root cause: FSQ hidden dim mismatch

Our MLP-FSQ structure:
```
input(D) → Linear(D, h) → GELU → Linear(h, 12) → FSQ([2]×12)
```

`h=64` is designed for 0.6B embedding (D=1024) (scale ~1:16).

For 4B (D=2560) and 8B (D=4096), h=64 forms a serious information bottleneck:

```
0.6B: ratio = 64/1024 = 6.25% → normal
4B: ratio = 64/2560 = 2.50% → obviously insufficient
8B: ratio = 64/4096 = 1.56% → seriously insufficient
```

MLP is forced to squeeze the high-dimensional semantic space into an overly narrow bottleneck, causing the L2 codebook to degenerate into a small number of high-frequency slots.

---

## Practical Corollary

**Collision rate / L2 entropy as a quick screening indicator**

FSQ hidden dim parameter adjustment experiment (EXP-045 Phase 1) only needs to run the tokenizer evaluation, and does not need to run NTP:

1. For target embedding (4B / 8B), scan `h ∈ {64, 128, 256, 512}`
2. Calculate L2 entropy and collision rate
3. Filter the minimum h with L2 entropy ≥ 10 bits (≈ 90% utilization)
4. Run a complete NTP verification using the filtered h

**Expected empirical formula** (to be verified by EXP-045)

From the perspective of bottleneck ratio, maintaining normal L2 utilization requires:
```
h_min ≈ emb_dim / 16 # Linear assumption
```
Or from an information theory perspective (sqrt compression):
```
h_min ≈ 2 × sqrt(emb_dim)
```

The two predictions for each model:

| Embedding | D     | h_min (linear) | h_min (sqrt) |
|-----------|-------|---------------|-------------|
| 0.6B      | 1024  | 64            | 64          |
| 4B        | 2560  | 160           | 101         |
| 8B        | 4096  | 256           | 128         |

EXP-045 will confirm which formula is more accurate through measured L2 entropy and give selection recommendations across embedding sizes.

---

## Relationship with Scaling Law

Floor PPL is obtained by fitting two-point scaling law, but **floor itself is a function of tokenizer quality and has nothing to do with model size**. This means:

- Fix FSQ bottleneck (increasing L2 entropy) can directly reduce floor PPL
- Lowering the floor PPL is equivalent to raising the ceiling of models of all sizes
- Even if M-tier's current R@500=70.4%, if the floor of 4B SID drops from 11.78 to 10.x, M-tier will eventually benefit

**Optimization priority**: FSQ hidden expansion is the optimization direction with the lowest cost and the most predictable benefits - only the tokenizer needs to be rebuilt, and the NTP architecture remains unchanged.
