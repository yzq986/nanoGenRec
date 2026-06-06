# rl/

[English](README.md) | [Chinese](README.zh.md)

Preference learning and RL-style alignment for NTP recommenders.

This module starts from an SFT NTP checkpoint and applies preference or reward-based optimization. The validated sequence is SP-DPO -> RF-DPO -> ECPO, with GRPO-style grouped advantages used by the RL stages.

## Files

| File | Purpose |
|------|---------|
| `preference.py` | SP-DPO preference pair construction from beam-search outcomes. |
| `feedback.py` | RF-DPO preference pair construction from behavior feedback. |
| `dpo.py` | Softmax-DPO and RF-DPO losses, including SID log-prob computation. |
| `reward.py` | BehaviorReward, prefix fallback, A2PO/HEPO reward stack. |
| `grpo.py` | GRPO clipped surrogate and group-normalized advantage. |
| `trainer.py` | Unified SP-DPO, RF-DPO, GRPO, and ECPO training loops. |

## Alignment Path

```text
SFT checkpoint
  -> SP-DPO preference pairs
  -> RF-DPO with real feedback
  -> ECPO on-policy reward optimization
  -> full-recall evaluation
```

## Validated Settings

| Parameter | Current Setting | Source |
|-----------|-----------------|--------|
| RF-DPO lambda | 0.3 on hard pairs | EXP-020 |
| RF-DPO epochs | 3 epochs, best mid-checkpoint selected | EXP-038B |
| ECPO delta | 0.1 | EXP-028+ |
| ECPO epsilon | 0.2 | EXP-028+ |
| GRPO group size | 512, `grpo_batch=4` | EXP-029 |
| `grpo_weight` | 0.03 | EXP-029 |

Current phase results are summarized in [experiments/logs/rl/README.md](../experiments/logs/rl/README.md).

## Usage

```bash
# Build SP-DPO pairs
python run.py sp-dpo-prepare --checkpoint experiments/ntp_checkpoints/<name>

# Build RF-DPO pairs from feedback
python run.py rf-dpo-prepare --checkpoint experiments/ntp_checkpoints/<name>

# Train DPO alignment
PYTHONPATH=. torchrun --nproc_per_node=8 run.py sp-dpo-train \
    --config experiments/configs/exp-NNN.yaml

# Train GRPO/ECPO alignment
PYTHONPATH=. torchrun --nproc_per_node=8 run.py grpo-train \
    --config experiments/configs/exp-NNN.yaml

# Evaluate alignment metrics
python run.py alignment-eval --checkpoint experiments/ntp_checkpoints/<name>
```

## Implementation Notes

- ECPO is on-policy: candidates are regenerated from the current policy to avoid unstable off-policy ratios.
- Sparse reward groups with near-zero std are skipped or clamped to avoid advantage explosions.
- BehaviorReward uses prefix fallback because exact full-SID behavior coverage is sparse.
- Side features must stay aligned with `ntp/`; `dpo.py` and `trainer.py` pass them through dictionaries.

## Common Failure Modes

| Symptom | Likely Cause |
|---------|--------------|
| GRPO loss stays at zero | Empty SIDTrie or no generated candidates. |
| Advantage magnitude explodes | Reward std is near zero or ratios are unclamped. |
| Eval drops unexpectedly | Train/eval side feature mismatch or stale checkpoint selection. |
| Reward metrics disappear from logs | Metrics are emitted only on GRPO steps; aggregate over GRPO steps, not wall-clock steps. |
