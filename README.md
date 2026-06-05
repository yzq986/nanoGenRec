# gr_demo — Generative Recommendation Semantic ID Project

[English](README.md) | [中文](README.zh.md)

`gr_demo` is a research project for generative recommendation with Qwen3 embeddings and Semantic IDs. It develops the full path from item representation to sequence modeling and alignment: Semantic ID tokenization, next-token prediction over item sequences, RL-style preference optimization, full-recall evaluation, and experiment governance.

The project is intentionally organized as a reproducible research system rather than a collection of scripts. Experiments are specified through YAML configs, results are recorded by phase, and implementation notes are kept close to the model and data pipelines they affect.

References: [OneRec](https://arxiv.org/abs/2506.13695) / [OneRec-V2](https://arxiv.org/abs/2508.20900) / [GR4AD](https://arxiv.org/abs/2602.22732) / [OneMall](https://arxiv.org/abs/2601.21770)

## Project Status

```text
Semantic ID tokenizer -> NTP recommender -> RL alignment -> Deployment packaging
```

| Area | Current Position | Representative Run | Working Notes |
|------|------------------|--------------------|---------------|
| **Semantic ID tokenizer** | 4096x3 binary `[2]x12`, snHR=0.095, CR=0.89% | EXP-012 | [tokenizer logs](experiments/logs/tokenizer/README.md) |
| **Embedding scale** | 0.6B: snHR=0.092, CR=0.42%; **4B: snHR=0.131**, CR=1.28% at nc=8192 | EXP-049 | [tokenizer logs](experiments/logs/tokenizer/README.md) |
| **NTP recommender** | M-tier bare R@500=**70.2%**; L-tier SFT R@500=64.1% | EXP-043/047 | [NTP logs](experiments/logs/ntp/README.md) |
| **RL alignment** | ECPO R@500=**65.7%** on the S-tier pipeline; L-tier alignment is pending | EXP-039B | [RL logs](experiments/logs/rl/README.md) |

**Recent result**: EXP-049 confirmed nc=8192 as the stronger tokenizer setting (Gini_d2 0.35 -> 0.24). h=64 and h=128 are equivalent in the current sweep; the recommended configs are exp049-{0.6b,4b}-nc8192-h128.

**Next run**: EXP-050 evaluates M-tier NTP with 0.6B/4B Semantic IDs, output-gate/CADET variants, and bare+RoPE ablations across 6 queued variants.

## System Pipeline

```mermaid
graph LR
    subgraph S1 ["1. Behavioral data"]
        D1[Hive -> S3<br><code>data/export_*</code>]
    end
    subgraph S2 ["2. Representation"]
        D2[Qwen3 encode<br><code>data/encode_distributed</code><br>or train-time encoding]
    end
    subgraph S3 ["3. Semantic ID tokenizer"]
        direction TB
        T_TRAIN["Train<br>RKMeans · MLP-FSQ · OPQ<br><code>tokenizer/</code>"]
        T_EVAL["Evaluate<br>collision · codebook utilization<br>reconstruction · entropy<br>semantic neighbor HR"]
        T_TRAIN --> T_EVAL
    end
    subgraph S4 ["4. Generative recommender"]
        direction TB
        N_TRAIN["Train<br>Transformer + MoE<br>beam search · KV cache<br><code>ntp/</code>"]
        N_EVAL["Evaluate<br>Perplexity · depth accuracy<br>Item Recall@K"]
        N_TRAIN --> N_EVAL
    end
    subgraph S5 ["5. Preference alignment"]
        direction TB
        R_TRAIN["Train<br>SP-DPO -> RF-DPO -> GRPO -> ECPO<br><code>rl/</code>"]
        R_EVAL["Evaluate<br>Recall/NDCG deltas<br>reward distribution shifts"]
        R_TRAIN --> R_EVAL
    end
    subgraph S6 ["6. Packaging"]
        D5["Package model<br><code>model/pack.py</code>"]
    end
    S1 --> S2 --> S3 --> S4 --> S5 --> S6
```

The primary CLI entry point is `python run.py <command>`.

## Core Workflows

```bash
# Train a tokenizer and produce Semantic IDs
python run.py train --model qwen3-0.6b

# Skip embedding if a cache already exists
python run.py train --model qwen3-0.6b --skip_embedding

# Distributed embedding for large data
PYTHONPATH=. torchrun --nproc_per_node=8 data/encode_distributed.py --model qwen3-0.6b

# Evaluate a trained NTP checkpoint
PYTHONPATH=. torchrun --nproc_per_node=8 run.py eval-ntp \
    --checkpoint experiments/ntp_checkpoints/<name> \
    --n_recall 1000
```

## Experiment Discipline

New experiments are expected to go through `experiments/run_exp.py` and YAML configs. This keeps training commands reproducible, makes duplicate baselines easier to detect, and gives each result a stable place in the experiment logs.

```bash
# Inspect defaults before writing an experiment config
sed -n '1,220p' experiments/configs/_base.yaml

# Check for similar historical runs before training
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --check

# Run all variants
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --no-smoke --commit

# Resume or run one variant
python experiments/run_exp.py experiments/configs/exp-NNN.yaml --only expNNN-a --no-smoke
```

Queue a background experiment through the shared wrapper:

```bash
echo "run_config.sh experiments/configs/exp-NNN.yaml  /tmp/expNNN.log  exp-NNN complete!" >> experiments/queue.txt
```

## Repository Structure

| Path | Purpose |
|------|---------|
| `data/` | Export, loading, embedding synchronization, and distributed encoding. |
| `tokenizer/` | Semantic ID tokenizers and SID preprocessing. |
| `ntp/` | Generative recommendation model, feature pipeline, preprocessing, training, and evaluation. |
| `rl/` | Preference and RL alignment: SP-DPO, RF-DPO, GRPO, ECPO, rewards, and preference data. |
| `eval/` | Full-recall evaluation, behavior metrics, comparison reports, and wrappers. |
| `metrics/` | Intrinsic tokenizer and representation metrics. |
| `experiments/` | YAML configs, orchestration, queue state, logs, and result artifacts. |
| `ideas/` | Research backlog organized by improvement dimension. |
| `research/` | Autonomous research-agent protocol, status, logs, inbox, and outbox. |
| `docs/` | Bilingual architecture and engineering documentation. |
| `model/` | Legacy-compatible wrappers, embedders, packaging, and model utilities. |

## Core Documentation

| Topic | English | 中文 |
|------|---------|------|
| Documentation index | [docs/README.md](docs/README.md) | [docs/README.zh.md](docs/README.zh.md) |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | [docs/ARCHITECTURE.zh.md](docs/ARCHITECTURE.zh.md) |
| Engineering log | [docs/engineering/README.md](docs/engineering/README.md) | [docs/engineering/README.zh.md](docs/engineering/README.zh.md) |
| Engineering changelog | [docs/engineering/CHANGELOG.md](docs/engineering/CHANGELOG.md) | [docs/engineering/CHANGELOG.zh.md](docs/engineering/CHANGELOG.zh.md) |

## Experiment Lineage

The detailed history lives in [experiments/logs/](experiments/logs/). The condensed lineage is:

| Phase | Key Experiments | Takeaway |
|------|-----------------|----------|
| Tokenizer | EXP-001 to EXP-012 | Frozen Qwen3-0.6B + 4096x3 binary MLP-FSQ `[2]x12` is the tokenizer baseline. |
| NTP foundation | EXP-013 to EXP-016 | S-tier active params with a 14-day data window established the first strong NTP baseline. |
| SP-DPO | EXP-017 | Hard-split preference pairs improved R@500 from 58.5% to 68.3%. |
| RF-DPO | EXP-018 to EXP-020 | Pure DPO caused catastrophic forgetting; joint NTP+DPO with lambda=0.03 became the SFT baseline. |
| Side features | EXP-021 to EXP-025 | Beam-search feature passing fixed train/eval mismatch and reached R@500=63.6%. |
| GRPO/ECPO | EXP-026 to EXP-039 | On-policy beam search and ECPO produced the strongest RL-aligned checkpoints. |
| Embedding scaling | EXP-043 onward | Qwen3-4B improves semantic-neighbor quality and M-tier NTP recall. |

## Evaluation Notes

Inline eval during `train-ntp` is only a health check. It uses a limited beam-search candidate set and should not be compared directly with full baselines. Use full evaluation for reported numbers:

```bash
PYTHONPATH=. torchrun --nproc_per_node=N run.py eval-ntp \
    --checkpoint experiments/ntp_checkpoints/<name> \
    --n_recall 1000
```

After each completed experiment, update:

1. `experiments/logs/<phase>/exp-NNN.md`
2. `experiments/logs/<phase>/README.md`
3. `README.md`

## Environment

The standard training/evaluation environment is `/home/dev/.conda/envs/gr`.

| Package | Version |
|---------|---------|
| Python | 3.12.13 |
| torch | 2.7.1+cu128 |
| CUDA driver | 12.8 |
| faiss-gpu | 1.14.1 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| pyarrow | 24.0.0 |

For standalone scripts, set `PYTHONPATH` to the repository root itself:

```bash
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
```
