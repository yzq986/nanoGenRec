# 004: Prefix-Locked vs Paper Beam Search — SP-DPO Candidate Generation

[English](004-prefix-locked-vs-paper-beam-search.md) | [Chinese](004-prefix-locked-vs-paper-beam-search.zh.md)

**Date**: 2026-04-19
**Context**: In the implementation of EXP-017 SP-DPO, it was found that the beam search method of the paper has an inherent bottleneck in the number of Medium/Hard candidates.

---

## question

Align3GR's SP-DPO uses beam search to generate rejected candidates, and then divides the difficulty according to prefix n-gram match:

| Difficulty | Definition | Meaning |
|-----------|------|------|
| Easy | L0 ≠ GT | Coarse-grained is Different |
| Medium | L0 = GT, L1 ≠ GT | The same coarse-grainedness, Medium and other difficulties |
| Hard | L0+L1 = GT, L2 ≠ GT | High degree of similarity, only fine-grained Different |

The problem is: **beam search starts from L0 and freely samples, and the L0 of most beam paths is not equal to GT**.

The SID system is 4096×4096×4096 (3 layers, 4096 clusters per layer). The `depth_acc_beam` L0 of the SFT baseline is only 0.030 (beam top-1 L0 hit rate 3%). Even if B=200:

- Easy candidates: ~190 (L0 is different, almost all beams)
- Medium candidates: ~5-10 (requires L0 to hit GT exactly)
- Hard candidates: ~0-2 (need L0+L1 both hit GT)

This means there is seriously insufficient training data for the Medium and Hard stages. The paper does not discuss this issue.

---

## Can progressive training provide relief?

The self-play progressive design of the paper (Easy→Medium→Hard) itself has a certain mitigating effect:

1. Easy DPO training improves L0 discrimination (EXP-017 Easy: L0 acc 0.030→0.041, +37%)
2. The improved model beam search will produce more L0 hit candidates → Medium will increase
3. Medium DPO training improves L1 discrimination → Hard increases

But this is an **indirect effect**, limited by beam size and model improvement. If the improvement at each stage is not large enough (for example, Easy DPO only improves L0 acc from 3% to 4.1%), the amount of Medium/Hard data is still very small.

---

## Prefix-Locked scheme

Directly lock the GT prefix and beam search the remaining layers:

| Sampling method | L0 | L1 | L2 | Guaranteed output |
|---------|----|----|-----|---------|
| Paper beam search | Sampling | Sampling | Sampling | Most Easy |
| Lock L0=GT | **Fixed** | Sampling | Sampling | All Medium+Hard |
| Lock L0+L1=GT | **Fixed** | **Fixed** | Sampling | All Hard |

Implementation: `constrained_beam_search` adds the `prefix` parameter, skips the beam search of the first P layer, and starts directly from layer P.

For each eval item, run beam search (progressive locking) up to 3 times:
1. Complete beam → Easy candidates
2. Lock L0 → Medium + Hard candidates
3. Lock L0+L1 → Hard candidates

Across pass dedups, each difficulty is capped at `n_rejected=20`.

---

## The essential difference between the two methods

### Paper method (beam search + classify)

```
P(rejected | context) ∝ model_score(context → rejected_sid)
```

Rejected is the **most likely wrong answer that the model thinks** - beam search is naturally sorted, and the high-scoring candidate is selected as rejected. These are the "most common mistakes" models make.

### Prefix-locked method

```
P(rejected | context, prefix=GT[:p]) ∝ model_score(context → prefix + remaining)
```

rejected is the answer that the model considers to be the most likely answer given the correct prefix constraints. These candidates share prefixes with GT and are semantically closer, but are not necessarily errors that the model will make without constraints.

### Key question: Which rejected is more effective?

**Hard candidates** for thesis method (if any):
- The model happens to go right on L0+L1 in free beam search, but goes wrong on L2
- This is the model's **real obfuscation mode** - it actually makes this mistake
- but in very small quantities

**Prefix-locked Hard candidates**:
- The model is forced to start from the correct L0+L1 and choose the most likely L2
- Sufficient quantity (B=200 Hard candidates)
- But these are not necessarily mistakes that the model would make when freely generated
- Similar to "Push students to the first two steps to the correct answer, and then see what mistakes they made in the last step"

**Possible results**:
1. Prefix-locked is better: sufficient Hard data > data authenticity, the model learns more refined L2 distinction
2. The method of the paper is better: Although there is less Hard data, each piece is real confusion and the signal is stronger.
3. Almost: DPO loss itself is not sensitive to the amount of data (the contrast signal of chosen vs rejected is more important than the absolute quantity)

---

## Experimental Design (EXP-017)

| Config | Sampling | Purpose |
|--------|------|------|
| Config 2 | Easy model beam B=200 (PaperMethod) | self-play baseline |
| Config 3 | Easy model prefix-locked B=200 | Progressive locked sampling |

The two groups share the Easy stage, only the Medium/Hard candidates are different. Compare eval indicators (PPL, Recall, depth_acc_beam).

**Expected Observations**:
- Config 3's preference statistics should show that there are far more Medium/Hard pairs than Config 2
- Training loss behavior may be different (Config 3 DPO loss may be harder to drop because locked prefix candidates are closer to GT)
- Recall improves uncertainty - the tradeoff depends on "data volume vs data authenticity"

---

## Extended thinking

### It would be better if prefix-locked

It shows that the beam search + classify scheme of the paper has too sparse signals in deep layers (L1, L2), and progressive locking is a better curriculum strategy. Consider:
- More aggressive locking: Easy uses full beam, Medium/Hard all use prefix-locked
- Dynamic beam size: Easy B=50 (enough), Medium locked B=100, Hard locked B=200

### If the paper method is better

It shows that "the real mistakes made by the model" are more effective than "artificially constructed difficult samples". This has implications for RL alignment - hard negative mining of contrastive learning is not necessarily as difficult as possible. The key is that the negative should be on the actual error distribution of the model.

### Hybrid solution

The two are not mutually exclusive. Can:
1. Complete beam search to obtain "real obfuscation" of all difficulties
2. Prefix-locked supplements the insufficient Medium/Hard
3. The weight of real confusion samples is higher (because the signal is stronger)
