# RL Alignment module for generative recommendation
# Phase 1: SP-DPO | Phase 2: RF-DPO | Phase 3: GRPO | Phase 4: ECPO

from rl.dpo import compute_sid_logprobs, softmax_dpo_loss

__all__ = ['compute_sid_logprobs', 'softmax_dpo_loss']
