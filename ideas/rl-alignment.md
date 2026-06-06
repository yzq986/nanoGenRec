# RL Alignment (reinforcement learning alignment)

[English](rl-alignment.md) | [Chinese](rl-alignment.zh.md)

Align generative recommendation models with business goals using RL/DPO/GRPO. It is an advanced optimization after the NTP model training is stable and has heavy pre-dependencies.

**Scope of influence**: Added `model/rl_trainer.py`, `metrics/sid_prediction.py`

---

## Evolution path

```
NTP purely supervised learning (current baseline)
├── IDEA-onemall-2: GRPO/DPO (OneMall solution)
│ └── GRPO > DPO, 768 candidates normalized advantage
├── IDEA-oneloc-2: DPO + dual target reward (OneLoc solution)
│ └── popularity + diversity dual goals
├── IDEA-align3-0: Progressive DPO (Align³GR, Kuaishou, AAAI 2026 Oral)
│ └── SP-DPO (self-game) → RF-DPO (real feedback), no external reward model required
│ └── EXP-017 SP-DPO ✅, EXP-018 RF-DPO pure DPO ❌ (forgetting), EXP-020 joint NTP+DPO ✅ SOTA 66.2%, EXP-037 SP-DPO features Ongoing
├── IDEA-rpo-0: RPO — SFT Loss as Adversarial Regularizer (NeurIPS 2024)
│ └── L = L_DPO + ηβ·L_SFT, theoretically proves that SFT regularity prevents overoptimization
│ └── Our joint NTP+DPO (EXP-019) is essentially RPO
├── IDEA-spot-0: Elastic Tether — Implicit regularization of DPO reward (HKU 2026)
│ └── β controls tether tightness: β large → tether tight → anti-forgetting
├── IDEA-rankgr-0: Listwise DPO + Rescore (RankGR, Taobao)
│ └── IAP (listwise DPO decoding) + RSP (light rescore), nearly 10,000 QPS
├── ~~IDEA-uni-0~~: Search Preference Optimization (UniSearch, Kuaishou) ❌ Close
│ └── reward model + user feedback, search scenario (no search scenario, technology is covered by align3-0)
├── IDEA-gr4ad-3: RSPO (GR4AD solution)
│   └── list-wise Lambda-weighted, NDCG-inspired
│ └── The strongest but the most pre-dependent
├── IDEA-sgrec-0: A2PO + Personalized Semantic Judge (S-GRec, Tencent)
│ └── Semantic-Business Asymmetric Gating, GMV +1.19%
└── IDEA-recast-0: Repair-then-Contrast Signal (Huawei, Apr 2026)
└── Sparse reward: rollout repair all-zero group + O(1) boundary update, Pass@1 +9~36%
```

---

## IDEA-onemall-2: GRPO/DPO reinforcement learning alignment

**Priority**: P1
**Source**: OneMall §3.3 Reinforcement Learning Policy
**Status**: Framework implemented — EXP-026 Implemented GRPO infrastructure (`rl/grpo.py`, `rl/reward.py`, `rl/trainer.py`); EXP-037 Running SP-DPO (features link) Validating DPO routes

### Core Idea

Aligning the retrieval model (generative NTP) with the ranking model using RL. Specific methods:
1. **Reward Model**: Online ranking model (using all user/item/cross features) as reward model, outputs CTR/CVR/EGPM predictions
2. **Reference Model**: Synchronize parameters from policy model regularly, and use beam search to sample candidate sets
3. **Policy Optimization**: GRPO or DPO optimization policy model

OneMall key findings:
- **GRPO > DPO**: GRPO is better than DPO in all candidate segments (Top10/100/500)
- GRPO calculates normalized advantage for all 768 sampling candidates, DPO only uses pairwise
- RL loss weight = 0.5, if it is too large, it will reduce SID accuracy
- Use only 2% training samples for RL

### Association with the current project

- Currently **there is no RL related code at all**, it is a new capability building
- The NTP model already has beam search infrastructure (`BeamSearchModule`), which can be reused for candidate sampling
- There is no online sorting model, and proxy reward needs to be constructed:
  - Option A: Offline CTR prediction model as reward
  - Option B: reward based on behavioral data (clicked=1, bought=5, exposed_not_clicked=0)
  - Solution C: embedding similarity as reward (simple but weak)
- **Rely on the NTP model to achieve reasonable baseline performance**, otherwise RL fine-tuning is meaningless

### Experimental Design Draft

**Phase 1: Offline Reward Model (simplified version)**

Construct reward function:
```
r(user, item) = α * is_clicked + β * is_bought + γ * embedding_sim
```

**Phase 2: GRPO Implementation**

Added `model/rl_trainer.py`:
1. Reference model: frozen NTP model checkpoint
2. Policy model: NTP model currently in training
3. Each user query: beam search samples N candidates (N=64~256, limited by GPU memory)
4. Calculate reward → normalize → advantage for each candidate
5. GRPO loss: clipped importance-weighted advantage (clip ratio 1±0.2)

**Joint loss**:
```
L = L_NTP + 0.5 * L_GRPO + α * L_contrastive
```

**Baseline**: NTP-only (no RL)

**Evaluation**: Offline Recall@K + reward score distribution changes

### Key questions

1. **Reward model quality is the fundamental bottleneck**: There is no online ranking model, and proxy reward may introduce bias.
2. **High sampling cost**: Each query does beam search and samples N candidates → the training speed may drop by 10x+
3. **Reference model synchronization frequency**: OneMall has not specified it in detail and needs to be determined experimentally.
4. **It is recommended to complete IDEA-onemall-0 (contrastive loss) first and establish a stronger NTP baseline before doing RL**

---

## IDEA-oneloc-2: DPO alignment + dual-objective reward function

**Priority**: ~~P1~~ → Overwritten by align3-0
**Source**: OneLoc §2.5 Reinforcement Learning
**Status**: Completely covered by IDEA-align3-0 - the SP-DPO + RF-DPO link of align3-0 is a superset of this IDEA. EXP-017/018/019/020 The DPO alignment framework has been fully verified; EXP-037 is being reproduced on the features link. The "dual-target reward" concept of oneloc-2 can be used as a reference in the reward design of the RF-DPO stage, and no separate experiment is required.

### Core Idea

The pre-trained NTP model only fits exposure data and cannot do fine-grained multi-objective balancing. OneLoc uses DPO for post-alignment:
1. Use the pre-trained model beam search to generate N candidates
2. Use reward function (geographic distance + GMV) to score candidates
3. Take the highest score as positive and the lowest score as negative to construct a preference pair
4. DPO loss combined with NTP loss training: `L = L_ntp + λ·L_dpo`

### Association with the current project

- **Zero RL/DPO code for current project**, this is a brand new module
- OneRec V1 paper in ARCHITECTURE.md also uses RL alignment, indicating that this is not unique to OneLoc
- What it means to us: **Use DPO to align NTP models to business goals**, for example:
  - Reward 1: item popularity / CTR estimated score (replacing OneLoc's GMV)
  - Bonus 2: diversity / category coverage (replacing OneLoc's geographical distance)
- DPO is much simpler than PPO, it does not require a critic network, only preference pairs

### Experimental Design Draft

**Step 1: Construct reward function**
- `R_popularity(v)`: item’s historical CTR or interaction count (already `data/export_behavior.py`)
- `R_diversity(v, S)`: category difference between recommended item and historical sequence

**Step 2: Generate preference pairs**
- Use trained NTP model beam search to generate top-N (N=50) candidates
- Calculate reward score for each candidate
- Select top-1 as positive and bottom-1 as negative

**Step 3: DPO training**
- Implement DPO loss in `model/train.py` or a new file
- Key hyperparameters: λ (DPO weight, OneLoc uses 0.05), β (DPO temperature)
- Training: NTP pre-training → freeze reference model → NTP + DPO joint training

**Evaluation**:
- Pre-training only vs DPO-aligned: recall, NDCG
- Changes in DPO-aligned reward distribution (whether the recommended item is more in line with the target)

### Key questions

1. **Pre-requisite**: The NTP model needs to be trained well enough first (the current `AutoregressiveNTPModel` may also need an architecture upgrade)
2. Reward function design: What to replace GMV and geographical distance? Need to be aligned with business goals
3. Negative sample quality: Is the bottom-1 of beam search really a "bad" recommendation? A more refined pair construction may be needed
4. Computational cost: Each training sample requires a beam search → N candidates → reward scoring, and the training speed may drop significantly.
5. **Priority judgment**: RL alignment is the "icing on the cake" and should be done after the basic NTP model and quantification scheme are stable.

---

## IDEA-gr4ad-3: RSPO ranking optimization (Ranking-Guided Softmax Preference Optimization)

**Priority**: P2
**Source**: GR4AD §RSPO, Table 1
**Status**: To be discussed

### Core Idea

GR4AD proposed the list-wise RL method RSPO: sort the candidate list produced by beam search by eCPM, and use NDCG-inspired Lambda weights for preference optimization. RSPO delivered a +1.06% delta compared to DPO (+0.70%) and GRPO (+0.65%). Core innovations: (1) Lambda weight ℳᵢⱼ focuses on the NDCG gain of sorting position exchange; (2) Reference gating Cᵢⱼ automatically turns off KL constraints when the reference model is unreliable.

### Association with the current project

- The current NTP model only does supervised learning without any RL/preference optimization
- **Pre-dependency heavy**: You need to have (1) a reasonable reward signal (the value token of IDEA-gr4ad-2); (2) a good enough beam search to produce multiple candidates (the current beam=5 has too few candidates)
- High implementation complexity: Requires reference model, reward model, Lambda NDCG calculation, and online learning pipeline
- More suitable for advanced optimization after the system matures

### Experimental Design Draft

**Simplified version — Getting started with Offline DPO**:
1. Use the current NTP model beam search to generate top-K candidates
2. Construct preference pairs based on behavioral data: clicked item > unclicked item
3. First implement the DPO loss verification framework, and then upgrade to RSPO

**Advanced Edition — RSPO**:
- Replace pairwise loss with list-wise Lambda-weighted softmax loss based on DPO
- Add reference gating mechanism

**Evaluation**: Hit@K, NDCG@K, compared with pure SL model

### Key questions

1. **High data requirements**: Real feedback from multiple candidates in the same context is required, which may not be available in the current demo data.
2. Training stability: It is difficult to adjust parameters for the RL method, and the reference model needs to be updated regularly.
3. Benefits depend on beam search quality — if beam search itself is not good enough (candidate homogeneity), ranking optimization has limited value
4. It is recommended that the priority be after IDEA-gr4ad-0/gr4ad-1/gr4ad-2

---

## IDEA-align3-0: Progressive DPO (SP-DPO → RF-DPO three-layer alignment)

**Priority**: P1
**Source**: Align³GR (Kuaishou, arxiv 2511.11255, Nov 2025, AAAI 2026 Oral)
**Status**: Experiment in progress — Full NTP baseline link:
- EXP-017 SP-DPO ✅ R@10 15.4%
- EXP-018 RF-DPO pure DPO ❌ forgetting (see IDEA-spot-0 for explanation of β ablation)
- EXP-019/020 joint NTP+DPO ✅ SOTA: R@500=66.2%, PPL=16.3 (exp020-hard-lam03)
- Features link (EXP-036→037→038→039): EXP-036 SFT starting point R@500=59.0%, EXP-037 SP-DPO in progress, EXP-038 RF-DPO ready to run, EXP-039 ECPO ready to run

### Core Idea

Align³GR proposes a unified three-layer alignment framework:

1. **Token-level Alignment**: Dual tokenization integrates semantics and collaborative signals (related to IDEA-pit-0)
2. **Behavior Modeling-level Alignment**: Bidirectional semantic alignment enhances behavior modeling
3. **Preference-level Alignment**: **Progressive DPO** — First SP-DPO (self-game) and then RF-DPO (real feedback):
   - **SP-DPO (Self-Play DPO)**: The model generates candidate sets by itself and constructs preference pairs sorted by reward → no external reward model is required
   - **RF-DPO (Real-Feedback DPO)**: Replace self-gaming reward with real user behavior feedback → more accurate alignment signal

**Results**: Recall@10 +17.8%, NDCG@10 +20.2% (offline). Kuaishou Industrial Deployment Online A/B has been significantly improved. AAAI 2026 Oral.

### Association with the current project

- Direct reinforcement IDEA-onemall-2 (GRPO/DPO): Progressive DPO is a more stable DPO training strategy
- **SP-DPO solves the pain point of "no external reward model"**: replace external reward with self-game
- The current project does not have an online ranking model for reward, and SP-DPO is the most practical RL entry-level solution.
- Behavioral data (clicked > not-clicked) can be used to replace online feedback in the RF-DPO stage

### Experimental Design Draft

**Phase 1 — SP-DPO**:
1. Train NTP baseline well
2. Use NTP model beam search to generate top-K candidates
3. Use simple reward (embedding similarity to ground truth) to rank candidates
4. Top-1 = positive, Bottom-1 = negative → DPO loss

**Phase 2 — RF-DPO**:
1. Use behavioral data: clicked item = positive, exposed-but-not-clicked item = negative
2. Replace preference pairs of SP-DPO

### Key questions

1. Relationship with IDEA-onemall-2 (GRPO): Which is better, Progressive DPO or GRPO? Comparative experiments can be done
2. SP-DPO’s reward design: What is used as the criterion for “self-game”?

---

## IDEA-rankgr-0: Listwise DPO + Two-Phase Decode-Rescore

**Priority**: P1
**Source**: RankGR (Alibaba/Taobao, arxiv 2602.08575, Feb 2026)
**Status**: To be discussed

### Core Idea

RankGR divides generative retrieval into two stages:

1. **Initial Assessment Phase (IAP)**: Inject **listwise DPO** into the autoregressive decoding to allow the model to understand the partial order relationship between candidates
2. **Refined Scoring Phase (RSP)**: Re-score the top-λ candidates of IAP using a lightweight scoring module (modeling the interaction between candidates and input sequences)

Two stages are jointly optimized in a unified GR model. Taobao's "Guess You Like" online verification + nearly 10,000 QPS real-time service.

### Association with the current project

- **Direct enhancement to IDEA-gr4ad-3 (RSPO)**: RankGR's listwise DPO is an industry-proven version of RSPO
- RSP (rescore stage) is a new technology: no external reranking model is required, a lightweight scorer is added inside the GR model
- Can be used with IDEA-gr4ad-4 (Dynamic Beam Search): IAP uses small beams to quickly screen, RSP accurately scores top candidates

### Experimental Design Draft

**Phase 1 — Listwise DPO in NTP**:
- When training NTP, beam search results for the same user are sorted by behavioral signals
- Construct listwise preference → DPO loss

**Phase 2 — RSP Module**:
- Add a cross-attention scorer to the NTP decoder output
- Input: user behavior sequence + candidate SID → Output: fine score
- Rerank top-K candidates using scores

### Key questions

1. Computational overhead of RSP: How much does the inference delay increase if cross-attention is performed on each top-λ candidate?
2. Cooperation with IDEA-gr4ad-4 (Dynamic Beam): beam output → RSP rescore → final top-K

---

## IDEA-uni-0: Search Preference Optimization (SPO)

**Priority**: ~~P2~~ → ❌ Close
**Source**: UniSearch (Kuaishou, arxiv 2509.06887, Sep 2025)
**Status**: ❌ Closed — There are currently no search scenarios, core RL technology has been fully covered by IDEA-align3-0 (Progressive DPO) and IDEA-onemall-2 (GRPO)

### Core Idea

UniSearch uses **Search Preference Optimization (SPO)** to integrate reward models and real user feedback into generative search:

1. Train reward model to score generated candidates
2. Use real user feedback (clicks, dwell time) as additional signals
3. Inject the reward signal into the generator through preference optimization

Kuaishou Live Search Deployment: **The largest single experimental improvement in recent years**.

### Association with the current project

- SPO is essentially a specialized version of GRPO/DPO for search scenarios
- Overlaps with IDEA-onemall-2 (GRPO) and IDEA-align3-0 (Progressive DPO)
- Unique value: The training method of reward model and the integration method of user feedback can be used for reference
- **Low Priority**: Currently there is no search scenario, and the core RL technology has been covered by other ideas.

### Key questions

1. Deduplication with IDEA-onemall-2 / IDEA-align3-0: What is the unique contribution of SPO?
2. There is currently no search scenario and limited value.

---

## IDEA-onerec-3: ECPO (Early Clipped GRPO) + Format Reward

**Priority**: P1
**Source**: OneRec (arxiv 2506.13695v4) §ECPO + §Format Reward
**Status**: To be discussed

### Core Idea

OneRec makes two key improvements to GRPO:

**1. ECPO (Early Clipped GRPO)**:
Standard GRPO's clipping is not aggressive enough for negative advantage samples—the policy may still assign higher probabilities to bad samples. ECPO introduces **early clipping**: when the sample advantage is negative, a tighter clip upper bound is used to suppress the policy ratio:

$$\pi_{\theta_{old}}'(o_i|u) = \max\left(\frac{\text{sg}(\pi_\theta(o_i|u))}{1+\epsilon+\delta}, \pi_{\theta_{old}}(o_i|u)\right)$$

$\delta=0.1$, allowing bad samples to be suppressed faster. At the same time, because RSFT and RL are trained in parallel, the KL divergence term is removed.

**2. Format Reward**:
The model may generate illegal SID tokens (not in the codebook) during RL training. Format Reward gives advantage=1 to legal output and advantage=0 to illegal output as independent reward signals:

$$A_i = \begin{cases} 1 & \text{if } o_i \in I_{\text{legal}} \\ 0 & \text{if } o_i \notin I_{\text{legal}} \end{cases}$$

Key findings: Use **random sampling** instead of top-k to select candidates for format reward, otherwise the legitimacy will increase first and then decrease.

### Association with the current project

- IDEA-onemall-2 already has GRPO basic design, ECPO is a direct upgrade
- Format Reward solves a practical problem: beam search generates invalid SIDs (IDEA-static-0's CSR constraint decoding also solves this problem, but on the training side rather than the inference side)
- OneRec experiment: group_size=512 is optimal (+1.82% App Stay Time), about 4 times that of inference Pass@K

### Key questions

1. It only makes sense to rely on the NTP model to be good enough - not doing it at this stage.
2. The engineering complexity of parallel training of RSFT and RL is high

---

## IDEA-gpr-0: HEPO — Hierarchy Enhanced Policy Optimization

**Priority**: P2
**Source**: GPR (Tencent/Weixin Channels, arxiv 2511.10138, Nov 2025)
**Status**: To be discussed

### Core Idea

GPR is a one-model generative recommendation framework for Tencent WeChat video account advertising. Its RL component **HEPO (Hierarchy Enhanced Policy Optimization)** uses the hierarchical structure of SID for alignment:

- Combined with MTP (Multi-Token Prediction), Value-Aware Fine-Tuning, and HEPO three-stage joint training
- HEPO uses the coarse-to-fine level information of SID to design rewards (different levels of prediction accuracy give different rewards)
- Unify interest modeling + value alignment + policy optimization

Full deployment of WeChat video account ads: **GMV and CTCVR significantly improved** (specific figures are in the text of the paper).

### Association with the current project

- HEPO uses the SID hierarchical structure to create hierarchical reward is a new idea - different from the flat reward of GRPO/DPO
- Example: Prediction of L1 is correct but L2/L3 is wrong → partial reward (better than completely wrong prediction)
- Complementary to IDEA-onerec-3 (ECPO + Format Reward): ECPO focuses on clipping, HEPO focuses on hierarchical reward structure

### Key questions

1. Depend on implementation after RL infrastructure matures
2. The details of the full text of the paper need to be supplemented with the specific algorithm of HEPO.
3. Suitable as an advanced improvement after verification of the GRPO/DPO basic solution

---

## IDEA-sgrec-0: A2PO + Personalized Semantic Judge (Asymmetric Advantage)

**Priority**: P1
**Source**: S-GRec (Tencent, arxiv 2025)
**Status**: To be discussed

### Core Idea

S-GRec found that standard GRPO/DPO has a **Semantic-Business Goal Asymmetry** problem in generative recommendations: Semantically similar candidates may have huge differences in business value (such as two similar videos, one with high GMV and one with low GMV), but standard RL gives them similar advantages. S-GRec proposes two mechanisms:

1. **A2PO (Asymmetric Advantage Policy Optimization)**: Use different advantage calculation methods for positive and negative candidates. Positive candidates use the standard normalized advantage, and negative candidates are additionally multiplied by a **semantic gating factor** — the closer the semantics is to the positive sample but the worse the business value of the negative sample, the heavier the penalty.
2. **Personalized Semantic Judge**: Train a lightweight discriminator, input user history + candidate SID, and output a joint score of semantic matching and business value. As a reward model for RL, it replaces pure business indicator reward

Core insight: **Hard negatives that are close in the semantic space but have poor business value are the most informative training signals**.

Tencent Online A/B: **GMV +1.19%, User Retention +0.8%**.

### Association with the current project

- Directly compatible with IDEA-onemall-2 (GRPO): A2PO is an advantage calculation improvement of GRPO without changing the overall framework
- Complementary to IDEA-align3-0 (Progressive DPO): align3-0 solves training stability (SP→RF progressive), sgrec-0 solves advantage quality
- Personalized Semantic Judge can reuse distance information in the SID embedding space, achieving moderate cost
- **Semantic-Business Asymmetric Gating** is a new idea: use the hierarchical structure of SID to determine the semantic distance (L1 identical = coarsely similar, L1/L2/L3 identical = highly similar)

### Experimental Design Draft

**Phase 1 — A2PO (improved on GRPO)**:
1. Add semantic gating to the advantage calculation of GRPO:
   - `gated_adv = adv * semantic_gate(candidate, positive)`
   - `semantic_gate = sigmoid(α * (1 - cosine_sim(sid_embed_candidate, sid_embed_positive)))`
2. Only apply gating to the negative advantage, leaving the positive unchanged (asymmetric)

**Phase 2 — Personalized Semantic Judge**:
1. Lightweight MLP: `[user_repr, candidate_sid_embed] → score`
2. Training data: positive (clicked) and negative (exposed-not-clicked) pairs in user behavior
3. Use judge score instead of simple reward signal

### Key questions

1. Pre-dependency on GRPO infrastructure (IDEA-onemall-2)
2. The α hyperparameter sensitivity of Semantic gating: if it is too large, far samples will be completely ignored; if it is too small, it will degenerate into standard GRPO.
3. The training data of Personalized Semantic Judge needs to expose unclicked samples

---

## Priority summary

| Priority | ID | Experiment | Reason |
|--------|-----|------|------|
| P1 (framework implemented) | IDEA-onemall-2 | GRPO/DPO reinforcement learning | EXP-026 GRPO framework implemented, EXP-037 DPO ExperimentMedium |
| P1 | IDEA-oneloc-2 | DPO + dual target reward | DPO is simpler than PPO and can be used as an entry into RL |
| **P1 (for Medium)** | **IDEA-align3-0** | **Progressive DPO (SP→RF)** | **EXP-020 ✅ SOTA 66.2%; EXP-037 SP-DPO features link for Medium** |
| P1 (to be completed by EXP-037) | IDEA-onerec-3 | ECPO + Format Reward | EXP-039 planned, need to complete EXP-037/038 first |
| P1 | IDEA-rankgr-0 | Listwise DPO + Rescore | Taobao verification, RSP module is a new technology |
| P1 | IDEA-sgrec-0 | A2PO + Semantic Judge | Tencent +1.19% GMV, Semantic-Business Asymmetric Gating |
| P1 | IDEA-genrec-2 | GRPO-SR + Hybrid Rewards | JD verification, NLL regular + relevance gating to prevent reward hacking |
| **P1 (verified)** | **IDEA-rpo-0** | **RPO: SFT Loss as DPO Regularizer** | **NeurIPS 2024 Theory Proof, EXP-019/020 joint NTP+DPO = RPO ✅** |
| **P1 (verified)** | **IDEA-spot-0** | **Elastic Tether: β Adaptive Regularization** | **HKU 2026, Explanation EXP-018 β ablation Result ✅** |
| P2 | IDEA-gr4ad-3 | RSPO sorting optimization | The biggest benefit but the heaviest pre-dependence |
| ❌ Close | ~~IDEA-uni-0~~ | ~~SPO search is optimized for Good~~ | No search scenario, the technology has been covered by align3-0/onemall-2 |
| P2 | IDEA-gpr-0 | HEPO Hierarchical Policy Opt | Tencent WeChat advertising deployment, Level reward new ideas |

---

## RF-DPO pitfall record: The real function of `--max_steps` is ratio control, not epoch control

**Importance: Extremely high - you will step on this pitfall every time you rerun RF-DPO**

### Core Insight

The function of `--max_steps` in Joint NTP+DPO mode is to **truncate NTP so that the NTP processing capacity and DPO processing capacity remain at 1:1**.
It does not control how many times the DPO looks at the data, nor does it control the "number of training rounds".

### The origin of step 807 (design of exp019/020)

```
exp018 hard preference pairs: 4,312 pairs
DPO batch_size: 16
DPO pass data: 4312 / 16 = 269 batches
Goal: 3 DPO runs = 269 × 3 = 807 DPO batches

Old dataset (exp016/017 era): NTP ~1,700 steps per epoch
--max_steps 807 Truncate NTP from 1,700 steps/epoch to 807 steps (about 0.47 epoch)

Result: NTP processed 807 batches ≈ DPO processed 807 batches → NTP:DPO ≈ 1:1
```

**This is where the 807 comes from: instead of "DPO runs 3 rounds", it's "let NTP steps equal DPO 3-round batches". **

### Why 1:1 ratio is important

- The weight of DPO signal in Joint training is `λ * dpo_loss` (λ=0.03)
- If the number of NTP steps is much more than the number of DPO steps → DPO is overwhelmed by NTP and the alignment signal is insufficient
- exp019/020 Key to success: ensure DPO accounts for ~100% of NTP training volume through `--max_steps 807`

### Issues with the current data set (exp023)

```
Current dataset (exp023-14d-features): NTP ~406 steps per epoch
--max_steps 807 > 406 (NTP loader itself is shorter) → truncation has no effect

Actual number of training steps = min(406, 807) = 406
DPO randomly draws batches within 406 steps → DPO actually runs ~1.5 rounds (instead of 3 rounds)
```

exp038/038b Root cause of failure/poor performance: NTP:DPO ratio imbalance, insufficient DPO signal.

### Correct approach (checklist when re-running RF-DPO)

1. Calculate the DPO target steps first: `target_steps = (n_pairs / dpo_batch_size) × target_dpo_epochs`
   - exp018 hard: `(4312 / 16) × 3 = 807`
2. Check NTP loader length: `n_train / (batch_size × world_size)`
   - exp023: `1,709,380 / (526 × 4) ≈ 812` — Wait, the actual measurement is 406, need to confirm
3. If NTP loader length < target_steps → `--max_steps` is invalid → trainer is required to support `ntp_epochs` (NTP multi-epoch)
4. If NTP loader length > target_steps → `--max_steps target_steps` correctly truncated

### trainer.py related code

```python
# rl/trainer.py, Joint mode
n_batches = min(len(ntp_loader), max_steps) # The upper limit of NTP steps
# DPO randomly inserts Bernoulli(p=rl_data_ratio) at each step
```

The `dpo_epochs` parameter is only valid in `pure_dpo=True` mode and is ignored in Joint mode.

### in conclusion

When redesigning RF-DPO, first confirm:
- `len(ntp_loader)` vs `(n_pairs / dpo_batch_size) × 3`
- If the NTP loader is shorter, NTP multi-epoch support must be added to the trainer (similar to the `ntp_epochs` parameter) to truly restore the 1:1 ratio of exp019/020

---

## IDEA-genrec-2: GRPO-SR + Hybrid Rewards (RL alignment to prevent Reward Hacking)

**Priority**: P1
**Source**: GenRec, JD.com (arxiv 2604.14878, SIGIR 2026)
**Status**: To be discussed

### Core Idea

JD GenRec proposes two key improvements based on GRPO to prevent reward hacking:

1. **Hybrid Reward + Relevance Gate**: Use dense reward model (SIM-based) to estimate the preference score r_pref, but add a relevance gate G = I (sim > τ) to filter out semantically irrelevant high reward candidates. Candidate rewards that do not satisfy the gate are set to zero. At the same time, the known positive samples (actual clicks/purchases by users) are forced to be assigned the highest reward in the group to correct the estimation bias of the reward model.

2. **NLL Supervised Regularization (SR)**: Add NLL regularization term (negative log likelihood for positive samples) to the GRPO objective to anchor the policy to the real user behavior distribution, replacing the standard KL divergence penalty.

Ablation experiment: After removing Gate G, HR@50 dropped from 0.74 to 0.70, and HaR increased from 2.68% to 1.96% - it seems that the illusion has been reduced, but the HR dropped significantly → the phenomenon of reward hacking is clear. Online: SFT + GRPO-SR is +1% click, +1.4% transaction better than pure SFT.

### Association with the current project

- Direct enhancement of IDEA-onemall-2 (GRPO/DPO): GenRec's GRPO-SR is an industrially proven improved version of GRPO
- Relevance Gate is particularly important for the SID system: there may be "valid but semantically irrelevant" combinations in the SID space
- NLL regularization replaces KL divergence → simpler to implement and does not require complete reasoning of the reference model
- Reward calibration (positive samples are forced to assign max reward) is a practical skill

### Experimental Design Draft

**Phase 1 — GRPO-SR on NTP baseline**:
- Connect GRPO-SR to the NTP baseline: rollout to generate G candidates, and use reward model to score
- Reward model can initially use SID embedding cosine similarity instead of SIM
- Add relevance gate: cosine_sim(generated_sid, positive_sid) > τ
- NLL regular: α * (-log P(positive_item | history))
- Evaluation: Recall@K, reward distribution, HaR

**Phase 2 — Dense Reward Model**:
- Train a specialized reward model (SIM-based or user preference model)
- Add positive calibration: reward of positive samples = max(group rewards)

### Key questions

1. Prerequisites: NTP SFT baseline + GRPO infrastructure (IDEA-onemall-2)
2. Choice of Reward model: SIM-based (needs training) vs simple embedding similarity (can be started quickly)
3. Parameter adjustment of Gate threshold τ: If it is too high, the filtering will be excessive, if it is too low, it will have no effect.
4. JD’s improvement (+1% over SFT) is marginal based on GRPO → Prioritize SFT

---

## IDEA-rpo-0: RPO — SFT Loss as Adversarial Regularizer for DPO

**Priority**: P1
**Source**: RPO (ByteDance + Northwestern + Stanford, arxiv 2405.16436, NeurIPS 2024)
**Status**: **Verified** — EXP-019 joint NTP+DPO is essentially RPO

### Core Idea

RPO theory proves that adding SFT loss as a regular term in DPO training can **provably mitigate reward overoptimization**:

```
L_RPO = L_DPO + ηβ · L_SFT(chosen)
```

Key theoretical findings:
1. DPO’s built-in β KL constraint **only controls gradient scale, not direction** — not enough to prevent overoptimization
2. SFT loss additionally corrects the gradient direction and anchors the policy to high-quality response distribution
3. Derivation from the perspective of adversarial reward model: SFT loss is equivalent to the penalty for the worst-case reward model

Experiment: RPO is consistently better than DPO on Zephyr-7b-beta and Zephyr-7b-gemma, effectively preventing the chosen response probability from declining during training (a typical symptom of overoptimization).

### Association with the current project

**Direct theoretical support for our joint NTP+DPO design**:
- Our `total_loss = ntp_loss + λ * dpo_loss` is RPO’s `L_RPO = L_SFT + (1/ηβ) * L_DPO`
- Our λ corresponds to 1/(ηβ) of RPO
- RPO papers use η=1, that is, the weights of SFT and DPO are equivalent → support our search range of λ=0.1~0.5

**Explanation of EXP-018 results**:
- Pure DPO (no SFT regularity) → catastrophic forgetting → PPL explosion
- β=0.5 is the least degraded but still not enough → RPO theory: β is not enough to control scale only, SFT loss is needed to control direction
- EXP-019 Add NTP (=SFT) regular → expected fix forgetting

### Experimental design

**Implemented in EXP-019**:
- Config 2 (λ=0.1), Config 3 (λ=0.5), Config 4 (λ=0.01) → Verify ηβ weight selection for RPO
- Expectation: λ is too small → return to the forgetting of EXP-018; λ is too large → NTP dominates and washes out the DPO signal
- Sweet spot is expected to be at λ=0.1~0.5

### Key questions

1. RPO uses chosen response to do SFT, we use NTP training data → the distribution may be different, but the regularization effect is similar
2. Multi-epoch NTP problem: RPO theory does not discuss SFT data duplication → our ~4.5 epoch may cause NTP overfitting
3. If EXP-019 verifies that joint NTP+DPO is valid, the next step can be to try the original form of RPO: only do SFT on the DPO chosen sample (not the entire NTP dataset)

---

## IDEA-spot-0: Elastic Tether — Implicit regularization of DPO Reward formula

**Priority**: P1
**Source**: SPoT (HKU, arxiv 2603.01683, Mar 2026)
**Status**: **EXP-018 beta ablation results explained**

### Core Idea

SPoT reveals the **Elastic Tether** regularization effect that comes with the DPO reward formula `r(x,y) = β log(π/π_ref)`:

Gradient scaling factor λ = 1 - σ(r_θ(x, y+)):
- **Acquisition Mode** (π approaches π_ref): r ≈ 0 → λ ≈ 0.5 → normal learning
- **Saturation Mode** (π away from π_ref): r → ∞ → λ → 0 → gradient disappears and updates automatically stop

When r_θ = 10, λ = 4.5×10⁻⁵ — 22,000 times smaller than SFT’s constant gradient.

Controlled experiment:
- SFT+ (proximal data) → forgetting (IFEval continues to decrease)
- DPO (same data) → not forgetting (IFEval stable)
- **Reward-SFT (only chosen, not rejected, but using reward formula)** → No forgetting either!

Conclusion: **Regularization comes from the reward formula itself (tethering effect of log π/π_ref), not from negative samples**. The larger β is, the tighter the tether is.

### Association with the current project

**Direct explanation of EXP-018’s β ablation**:

| Config | β | PPL | R@10 | Elastic Tether explained |
|--------|-----|-----|------|-----------------------------|
| hard | 0.1 | 50,694 | 8.3% | β small → tether loose → policy drift far → catastrophic forgetting |
| prog-beta01 | 0.01 | 2.4B | 6.0% | β very small → almost no tether → completely forgetting |
| prog-beta50 | 0.5 | 404.9 | 10.2% | β larger → tether tight → least degraded but still not enough |

In the 807-step hard DPO training, even if β=0.5 (the tightest tether), r_θ will gradually increase causing the tether to completely relax (λ → 0). At this point the model is in a "zero gradient" state but has strayed too far - tether can only slow down but not pull back. **External regularization (NTP/SFT loss) is required to provide a continuous gradient signal to pull the policy back**.

### Experimental design

**EXP-019 Core validation covered**.

Further experimental directions:
1. **Reward-SFT baseline**: Only use chosen preference pair to do reward-based SFT (no negative), compare with DPO → verify whether tethering is also true in recommendation scenarios
2. **β schedule**: Use small β in the early stage of training (fast learning), and gradually increase β in the later stage (tighten tether to prevent forgetting)
3. **Monitor r_θ during training**: record the growth trajectory of implicit reward and verify when tether relaxes

### Key questions

1. SPoT experiments were done on LLM (Qwen3-8B), and the behavior of the recommended model (45.8M) may be different
2. Elastic Tether is an analysis of single-step gradients, and the cumulative effect of 807 steps may be different.
3. Complementary with RPO (IDEA-rpo-0): RPO plus SFT loss correction direction, Elastic Tether explains the scale control effect of β

---

## IDEA-recast-0: ReCast — Repair-then-Contrast Signal for Sparse-Hit GRPO

**Priority**: P1
**Source**: ReCast (Huawei, arxiv 2604.22169, Apr 2026)
**Status**: To be discussed — Directly targeted at the "reward std≈0, 96% zero reward sample" issue that EXP-026 stepped on

### Core Idea

ReCast observed that the fundamental problem of general group-based RL (GRPO system) under sparse-hit generative recommendation is not reward assignment, but "many sampled groups never become learnable at all". In a representative RL setting from OpenOneRec:
- **85% of rollout groups are all-zero** (all candidates in the group did not hit GT)
- Another 13% are single-hit (one accidental hit dominates the update, noisy)
- Only 2% are multi-hit groups
- 96% of the sample level is zero reward

Standard GRPO normalize this group will produce: reward std ≈ 0 → advantage explosion / gradient instability. This is exactly the trap we stepped on with EXP-026 (relying on `std < 1e-6` group skip + `adv.clamp(-5, 5)` + `log_rho.clamp(-10, 10)` hard constraints to alleviate). ReCast fundamentally fixes it from the **signal construction** level:

**1. Rollout Repair (inject GT as positive anchor)**

For all-zero group, construct a legal positive example `R_anc` from ground-truth to replace the "lowest informative" response in the group (arg min sorted by structural score). This turns all wasted rollout into learnable units. Groups that already have positive values ​​will not be changed.

**2. Structural Score φ (level SID prefix matching)**

For 3-token SID `p = (pa, pb, pc)` and GT `t = (ta, tb, tc)`:
```
φ(p,t) = 1.0 if p == t # Exact match
         0.1 if (pa,pb) == (ta,tb) # First 2 levels of matching
         0.01 if pa == ta # Only the first level matches
         0 otherwise
```
Structural score is only used for **within-group ranking** (arg min when hardest negative and repair are selected), and the external reward is not changed. **This is the symmetrical expression of our BehaviorReward prefix cascade** — we use L0/L1/L2 as the reward itself, and ReCast uses it as the candidate sorting key.

**3. Boundary Contrastive Update (O(1) update support)**

For the group after repair:
- `i+ = argmax_i ri` (positive example of highest reward)
- `i- = argmax_{ri=0} ui` (the miss sample with the highest structural score, that is, "the closest but not matching")
- advantage = +w / -w / 0 (the rest are all 0)

Replace full-group normalization with **update only this pair**. The search width G remains, and the actor update width decreases from O(G) to O(1). The rollout budget continues to explore rare positive examples, but the actor no longer pays learning overhead for uninformative intermediate samples.

**4. System benefits of Search–Update Decoupling**

Wsearch = G, Wupdate = O(1) gives:
- actor-side update time **16.60× speedup**
- peak memory **-16.5%**
- actor MFU **+14.2%**
- Pass@1 **+9.1% ~ +36.6%** (5 tasks, Qwen3-1.7B baseline)
- matched-budget advantage: ReCast uses **4.1% rollout budget** to achieve baseline full performance
- Advantages scale with model size (Qwen3-14B benefits more)

### Association with the current project

**This is the most direct patch to the EXP-026 pitfall. ** Our current path to `rl/trainer.py::_grpo_step` is:
```python
# 768 candidates per context → BehaviorReward (SID exact + prefix cascade)
# → normalized advantage → clip → loss
```
Existing issues:
- 85% of all-zero groups are skipped by `std < 1e-6` - all the GPU time of these rollouts is wasted
- A few non-zero groups dominate the gradient and the training signal is unstable
- The `reward metric` in the step log is often all 0, making it difficult to diagnose.

ReCast’s mapping for us:

| ReCast mechanism | Corresponding to our implementation position | Transformation cost |
|-------------|-------------------|--------|
| Rollout repair (inject GT) | `rl/trainer.py::_grpo_step`, after the BehaviorReward is calculated and before the advantage is calculated | Low — directly constructs a response from the ground-truth SID of the context to replace the lowest Low structural score in the group |
| Structural score φ | Exists — prefix cascade L0/L1/L2 match function in `rl/reward.py::BehaviorReward` Medium | 0 — Direct reuse |
| Boundary contrastive update | Replace the normalized advantage calculation in `rl/grpo.py` | Medium — separate branch, retain the original GRPO as ablation |
| Search-update decoupling | Physically filter non-(i+, i-) samples, skip old_log_prob / ref_log_prob / update_actor | Medium — need to change log_prob calculation Path |

**Differences from existing RL ideas**:
- vs IDEA-onemall-2 (GRPO baseline): ReCast is GRPO's signal layer plug-in and does not change the external objective
- vs IDEA-onerec-3 (ECPO): ECPO changes clip boundaries, ReCast changes candidate selection — **orthogonal, combinable**
- vs IDEA-sgrec-0 (A2PO): A2PO adds semantic gating to negative examples; ReCast directly retains only one "hardest" negative example - more extreme, greater system benefits
- vs IDEA-gpr-0 (HEPO): HEPO uses hierarchical reward, ReCast uses hierarchical structural ranking - both use SID hierarchies but have different action points

### Experimental Design Draft

**Preliminary**: EXP-037/EXP-038 (SP-DPO, RF-DPO) after stabilization, choose one of EXP-039 (ECPO) or new EXP-040 (ReCast)

**EXP-040 — ReCast on top of EXP-026 GRPO pipeline** (local 8 GPU):

**Configuration**:
- Checkpoint: exp020-hard-lam03 (R@500=66.2% baseline)
- Rollout size G = 64 (keep EXP-026 setting)
- Structural score: directly use the prefix match of `BehaviorReward` (L0=0.01, L1=0.1, L2=1.0)
- Repair: Get its SID from the context's ground-truth next-item, use this SID to construct a generated output to replace the lowest `ui` in the group
- Boundary pair: `i+` = highest reward; `i-` = highest structural score among misses

**Comparison**:
| Config | Rollout utilization | Pass@1 / R@10 | actor update time | notes |
|--------|--------------|---------------|------------------|-------|
| A. EXP-026 GRPO (baseline) | ~15% (85% skip) | ? | T0 | Results available |
| B. + rollout repair only | 100% | ? | T0 | isolation repair contribution |
| C. + boundary contrast only | ~15% | ? | T0/60 | isolation contrast contribution |
| D. Full ReCast (B+C) | 100% | ? | T0/60 | Complete solution |
| E. ReCast + ECPO | 100% | ? | T0/60 | Overlay IDEA-onerec-3 |

**Key Experimental Questions**:
- Expected D relative to A: R@500 +0.5~2.0pp, and the training curve is more stable (`reward std` distribution becomes wider)
- Does C alone have negative returns? (Removing repair but only using boundary may degrade)
- The introduction of GT by Repair may produce **target leakage** — to check whether it is only used in RL advantage calculations, repaired rollout cannot be directly used as a positive sample in NTP joint loss

**Phase 1 (code changes)**:
1. Add `compute_boundary_advantages(rewards, structural_scores)` function in `rl/grpo.py`
2. Add `--rl_strategy {grpo, recast, recast_no_repair, recast_no_contrast}` switch in `rl/trainer.py::_grpo_step`
3. Reuse `BehaviorReward.prefix_match` as structural score
4. Physical filtering (skip old_log_prob / ref_log_prob calculation) — verify system-level acceleration (Huawei reports 16.60×)

### Key questions

1. Does **GT injection constitute cheating? ** Rollout repair uses GT to construct positive examples. Strictly speaking, ground-truth is not used as the label (the external reward is still an exact match), but the rollout distribution is artificially contaminated. It needs to be compared with the IDEA-align3-0 (SP-DPO/RF-DPO) route: SP-DPO’s self-play will also tend to be in the direction of high reward, which is similar in nature.
2. **Relationship with the existing BehaviorReward prefix cascade**: We have already done the prefix cascade fallback in the reward layer, and ReCast is done again in the advantage layer. Will superimposing the two overcompensate? Suggested ablation: `BehaviorReward exact only` + ReCast structural ranking vs `BehaviorReward prefix cascade` + ReCast structural ranking
3. **Contrast weight w selection**: The paper defaults to w=1, but in our sparse scenario it may need to be larger (e.g. w=2~5) to compensate for the gradient variance brought by only 2 samples.
4. **Is boundary contrast still needed for the already stable group? ** All groups in the paper use boundary, but intuitively it may be better to use full-group normalization for multi-hit groups (with rich signals). You can add a branch with `k_hit ≥ threshold`
5. **Reproducibility**: This experiment was run on Qwen3-1.7B/8B/14B + Ascend NPU, ours is ~45M NTP + L20X/H100. The model size difference is 2 orders of magnitude. The paper emphasizes that the benefits increase with scale, and the benefits of small models may be limited - small-scale ablation is required

### Related ideas

- IDEA-onemall-2 (GRPO baseline): ReCast insert signal layer on top of it
- IDEA-onerec-3 (ECPO): orthogonal improvement, combinable
- IDEA-sgrec-0 (A2PO): similar idea (weighting negative examples), ReCast is more radical (leaving only one negative example)
- IDEA-gpr-0 (HEPO): level reward, ReCast level ranking
- EXP-026 GRPO/ECPO pitfall record (recorded in CLAUDE.md): This idea is a systematic repair of this scene

---

## IDEA-raddpo-0: RAD-DPO — Robust Adaptive Denoising DPO for SID generative retrieval

**Priority**: P1
**Source**: RAD-DPO (JD.com, arxiv 2602.23964, Feb 2026)
**Status**: To be discussed — the three sub-retrofits can be independently integrated into the existing DPO pipeline

### Core Idea

RAD-DPO identifies **three structural flaws** of standard DPO in SID generative retrieval, each with corresponding corrections. JD.com Core Search A/B: **UCVR +0.34%**.

**Defect 1 - Gradient conflict of shared prefix**: SID level (L0→L1→L2), similar items share prefix. DPO simultaneously punishes preferred and rejected shared prefix → gradient conflict, destroying hierarchy.
**Correction**: **Token-Level Gradient Detachment** — Do stop-gradient on the shared prefix position, rejected path, and only let the preferred contribute gradient.

**Flaw 2 — Noisy pseudo-negatives**: The "no click=negative" of user click log may be a pseudo-negative example of exposure position suppression. DPO equal treatment → contaminated.
**Correction**: **Similarity-based Dynamic Reward Weighting** — Dynamically scale the loss weight according to the "embedding cosine of rejected vs ground-truth"; the similarity is high → reduce the weight.

**Flaw 3 — Multi-label squeezing**: In e-commerce with multiple positive examples (multiple valid items), vanilla DPO concentrates the probability into a single chosen → squeezing other co-valid.
**Correction**: **Multi-Label Global Contrastive + Global SFT Loss** — Global comparison of the entire Y_pos set vs. negative set, with global SFT loss covering all positive examples.

### Association with the current project

Directly target our DPO:

| We step on the trap | RAD-DPO corresponding correction |
|---------|---------------|
| EXP-018 hard DPO PPL→50K+ forgetting | Gradient detachment protection prefix |
| EXP-026 GRPO sparse reward, std≈0 | Similarity weighting scales weak signals |
| EXP-020 only focuses on top-1 reward | Multi-label contrastive covers multiple positive examples |

Can be orthogonally stacked with IDEA-align3-0 (Progressive DPO) and IDEA-onerec-3 (ECPO). Partially overlaps with IDEA-recast-0 (ReCast): ReCast only leaves the i+/i- poles, RAD-DPO covers all Y_pos.

### Experimental Design Draft

**Phase 1** — Gradient Detachment (easiest, 1 day): `rl/dpo.py::compute_sid_logprobs` Detach the rejected path that shares the prefix position, relax re-train on the EXP-020 baseline, and compare with PPL/R@K to see if forgetting is reduced.

**Phase 2** — Similarity-based Reward Weighting: Use Qwen3 embedding cosine to calculate loss weight `1 - max(0, cos - threshold)`, sweep threshold {0.5, 0.7, 0.9}.

**Phase 3** — Multi-Label Global Contrastive: Construct Y_pos multi-positive examples from session-level data, add global contrastive loss; weighted `L = L_DPO + α · L_contrastive`.

### Key questions

1. Gradient detach must accurately match the shared prefix, and cannot mistakenly detach the token after diverge.
2. Similarity threshold selection — too low will result in false positives, too high will result in ineffectiveness
3. Multi-label data availability — session-level multi-positive re-export required
4. Choose one or combine with ReCast: ReCast is more aggressive in sparse reward scenarios, and RAD-DPO is better in dense reward scenarios.

### Related ideas

- IDEA-align3-0 (Progressive DPO): curriculum, orthogonal
- IDEA-onerec-3 (ECPO): clipping, orthogonal
- IDEA-recast-0 (ReCast): multi-label vs boundary comparison
- IDEA-sgrec-0 (A2PO): Negative example gating has a similar idea
- IDEA-rpo-0 (RPO): global SFT loss consistent
