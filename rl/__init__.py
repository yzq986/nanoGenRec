# RL Alignment module for generative recommendation
# Phase 1: SP-DPO | Phase 2: RF-DPO | Phase 3: GRPO | Phase 4: ECPO

from rl.dpo import compute_sid_logprobs, softmax_dpo_loss
from rl.grpo import grpo_loss, ecpo_loss
from rl.reward import (
    RewardFn, DiagnosticReward,
    BehaviorReward, WeightedBehaviorReward, FormatReward,
    NoveltyReward, ExternalReward, BusinessReward, CompositeReward,
    _bitmap_to_quality,
)

__all__ = [
    # Phase 1-2: DPO
    'compute_sid_logprobs', 'softmax_dpo_loss',
    # Phase 3-4: GRPO / ECPO
    'grpo_loss', 'ecpo_loss',
    # Reward plugins
    'RewardFn', 'DiagnosticReward',
    'BehaviorReward', 'WeightedBehaviorReward', 'FormatReward',
    'NoveltyReward', 'ExternalReward', 'BusinessReward', 'CompositeReward',
    '_bitmap_to_quality',
]
