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
