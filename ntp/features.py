"""Side feature registry for NTP model.

Adding a new side feature requires only:
1. Add a FeatureDef entry to REGISTRY below.
2. Produce the feature in build_unified_sequences (ntp/train.py) and save_shard.
3. The model, trainer, eval, and DPO paths auto-detect it from shard keys.

FeatureDef fields:
    dtype       'long' | 'float'  — tensor dtype for embedding lookup / RoPE
    inject      'embed_add'       — embed via nn.Embedding, add to token embedding
                'rope'            — continuous float, passed to RoPE time planes
    emb_size    int               — codebook / bucket count (for 'embed_add' only)
    default_val int | float       — padding / generation default value
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FeatureDef:
    dtype: str          # 'long' or 'float'
    inject: str         # 'embed_add' or 'rope'
    emb_size: int = 0   # nn.Embedding codebook size (inject='embed_add' only)
    default_val: float = 0.0


# ── Single source of truth for all side features ──────────────────────────────
REGISTRY: Dict[str, FeatureDef] = {
    'time_gaps':     FeatureDef(dtype='long',  inject='embed_add', emb_size=16, default_val=0),
    'action_levels': FeatureDef(dtype='long',  inject='embed_add', emb_size=4,  default_val=0),
    'timestamps':    FeatureDef(dtype='float', inject='rope',      emb_size=0,  default_val=0.0),
}

# Keys whose shard tensors use float32 (all others use int64)
FLOAT_KEYS = frozenset(k for k, v in REGISTRY.items() if v.dtype == 'float')


def active_features(shard_keys) -> Dict[str, FeatureDef]:
    """Return {key: FeatureDef} for features present in shard_keys.

    Called by train_packed and eval to auto-detect what the data contains,
    without any --use_* CLI flags.
    """
    return {k: v for k, v in REGISTRY.items() if k in shard_keys}


def embed_add_features(shard_keys) -> Dict[str, FeatureDef]:
    """Subset of active_features with inject='embed_add'."""
    return {k: v for k, v in active_features(shard_keys).items()
            if v.inject == 'embed_add'}


def rope_features(shard_keys) -> Dict[str, FeatureDef]:
    """Subset of active_features with inject='rope'."""
    return {k: v for k, v in active_features(shard_keys).items()
            if v.inject == 'rope'}


# backward compat alias
torope_features = rope_features
