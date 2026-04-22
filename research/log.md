# Research Agent Log

Append-only 时间线。格式见 `schema.md`。

---

## [2026-04-22 00:00] STARTUP: Research agent system initialized

Prometheus branch created. Shared space (`research/`) initialized with:
- `program.md` — agent operating manual
- `schema.md` — file format conventions
- `status.md` — state dashboard
- `inbox/` / `outbox/` / `decisions/` / `paper-notes/` — communication channels
- `estimate_runtime.py` — runtime estimation tool

Current baseline: exp023-segment (PPL=25.94, R@500=61.2%).
92 ideas tracked across 8 dimensions. 20 papers available.

---

## [2026-04-22 18:05] STARTUP: Research agent activated (session 2)

Resumed autonomous loop. Tasks: check P0 ideas, analyze EXP-022~025 results, propose next experiments.

## [2026-04-22 18:06] INBOX_READ: 001-initial-directive

Bootstrap directive read. Instructions: read papers, analyze experiments, propose experiments.

## [2026-04-22 18:10] PAPER_READ: 2604.05113 (CRAB — Codebook Rebalancing)

Multi-resolution RQ codebook balancing. Low relevance to current NTP features work.

## [2026-04-22 18:12] PAPER_READ: 2604.05314 (NSGR — Next-Scale Generative Reranking, Meituan)

Scale-wise reranking for SID generation. Reranking idea could help but requires larger model.

## [2026-04-22 18:14] PAPER_READ: 2604.14878 (GenRec — JD.com, SIGIR 2026)

Page-wise NTP + Token Merger + GRPO-SR. **High impact**: PW-NTP directly applicable, +9.5% click online.

## [2026-04-22 18:20] EXPERIMENT_EVAL: EXP-022 contrastive loss — ALL results collected

All 5 configs evaluated. Conclusion: contrastive loss HURTS. All configs worse than baseline R@500=58.5%.
- α=0.01: PPL=27.89, R@500=59.2% (best contrastive, but still +0.7pp at cost of +0.84 PPL)
- α=0.1: PPL=29.22, R@500=57.9%
- α=0.5: PPL=29.04, R@500=56.3%
- dim256: PPL=29.66, R@500=58.8%
- temp005: PPL=28.16, R@500=58.2%

## [2026-04-22 18:22] EXPERIMENT_EVAL: EXP-025 beam search feature passing — results collected

Two configs. beam_passes is new best for NTP side features.
- beam_passes: PPL=25.22, R@500=63.6% ← **NEW BEST** (+2.4pp over exp023-segment)
- action_l2only: PPL=24.85, R@500=27.0% ← complete failure, L2-only action still leaks

## [2026-04-22 18:30] DECISION: EXP-022 contrastive loss — DISCARD

Contrastive loss (IDEA-onemall-0) tested exhaustively. No configuration improves over baseline.

## [2026-04-22 18:31] DECISION: EXP-025 beam_passes — MERGE

Beam search feature passing (time_gap + action carry-forward) confirmed positive: R@500 63.6% (+2.4pp).
New best config: segment_emb + time_gap + action + beam search feature passing.

## [2026-04-22 18:35] OUTBOX_WRITE: 002-experiment-analysis (EXP-022~025 trends + proposals)

---

## [2026-04-22 19:15] STARTUP: Research agent activated (session 3)

git pull company/prometheus failed (remote unreachable). No inbox changes since session 2. status.md current_task=null.

## [2026-04-22 19:20] PAPER_READ: 2603.28124 (RCLRec — Reverse Curriculum Learning for Conversions, Alibaba)

Sparse conversion problem in GR. RCPM selects k conversion-relevant history items as decoder prefix using pay-conditioned query. +10%+ on industrial and Tmall datasets. Medium relevance — encoder-decoder architecture, needs adaptation for decoder-only NTP. Connection: IDEA-dualgr-0 (conversion-aware sample weighting).

## [2026-04-22 19:22] PAPER_READ: 2603.11486 (Quantized Inference for OneRec-V2, Kuaishou)

FP8 PTQ for GR model: 49% latency reduction, 92% throughput gain, no online metric degradation. Low relevance now (deployment concern). Key finding: GR models with MoE (like ours) have LLM-like weight/activation distributions, making FP8 quantization feasible.

## [2026-04-22 19:25] PAPER_READ: 2603.01683 (SPOT — Surgical Post-Training, HKU)

HIGH RELEVANCE: reveals DPO structural flaw for rigid correctness tasks (optimizes margin via suppressing y- rather than boosting y+). SPoT-BCO: decoupled BCE with adaptive reward shift δ, avoids gradient vanishing and catastrophic forgetting. Direct application: replace SP-DPO loss with SPoT-BCO. Also: LCS quality filtering for beam-target pairs.

## [2026-04-22 19:30] OUTBOX_WRITE: 005-spot-dpo-finding (SPOT finding + IDEA-spot-0 proposal)

---

## [2026-04-22 19:45] STARTUP: Research agent activated (session 4, /loop)

git pull company/prometheus failed (remote unreachable). No inbox changes. Continuing P6 paper reading.

## [2026-04-22 19:47] PAPER_READ: 2602.08612 (OneLive — Dynamic Tokenizer, Sequential MTP, GRPO>DPO, Kuaishou)

MEDIUM RELEVANCE. Sequential MTP (+62% QPS by sharing KV cache across 3 decoders) directly applicable to our 3-token SID structure. QK Norm prevents logit explosion in deep bf16 training. GRPO > DPO confirmed independently (2nd confirmation). Dynamic tokenizer only needed for livestreaming (content changes every 30s).

## [2026-04-22 19:49] PAPER_READ: 2601.21770 (OneMall — ResKmeansFSQ, Sparse MoE, GRPO>DPO, Kuaishou)

HIGH RELEVANCE. ResKmeansFSQ (Res-Kmeans 2L + FSQ 1L) reduces SID collision rate 36%→11% (+4.1pp HR@500). In-batch contrastive loss +1.7% HR@500 but requires better item representations (explains why our EXP-022 was negative — FSQ-only tokenizer lacks the semantic richness). GRPO > DPO confirmed (3rd confirmation across OneLive + OneMall). Sparse MoE scaling validated (0.5B-A0.1B → +11% HR@50 over dense 0.1B).

## [2026-04-22 19:51] PAPER_READ: 2602.23671 (FuXi-Linear — Linear Attention GR, USTC+Huawei)

LOW RELEVANCE (short-term). O(n) training / O(1) inference linear attention for long user history sequences (3555 avg len). Temporal Retention Channel decouples time/semantic signals via complex-domain modeling r·e^(iθΔt) — aligned with IDEA-feat-5 (TO-RoPE). Not applicable at current 30-item window.

---

## [2026-04-22 20:05] STARTUP: Research agent activated (session 5, /loop cron)

No inbox changes. Continuing P6 paper reading: 10 remaining.

## [2026-04-22 20:07] PAPER_READ: 2601.18457 (TCA4Rec — Token-level CF Alignment for GR, USTC+Ant Group)

LOW-MEDIUM RELEVANCE. Projects CF logits (SASRec) into token-level distribution, uses soft NTP loss (label smoothing + CF bias). Requires independent CF tower — not directly applicable to our setup. Key insight: CF signal injection via soft labels is model-agnostic and low-overhead. SASRec still beats all LLM-based GR methods on small benchmarks, confirming CF signal quality is a bottleneck.

## [2026-04-22 20:09] PAPER_READ: 2511.20673 (FlexCode — Dual Codebook for GR, Roblox+Cambridge)

MEDIUM RELEVANCE. Dual codebook: semantic (RQ-VAE on text) + collaborative (RQ-VAE on SASRec embedding), popularity-aware MoE router allocates token budget. +8.0% NDCG@10 on KuaiRand, +13.2% on industrial dataset vs SASRec. Key insight: single codebook entangles semantic and collaborative signals — tail item improvements +11.3% NDCG@10. Requires extra CF tower.

## [2026-04-22 20:11] PAPER_READ: 2511.12113 (MetaGDPO — Group DPO for LLM Reasoning, IIE-CAS)

LOW RELEVANCE. Offline GRPO approximation (GDPO) for LLM reasoning distillation. Metacognitive data selection to prevent catastrophic forgetting. Not GR-specific. Tangentially relevant to RL alignment: offline group preference optimization as resource-efficient GRPO alternative.
