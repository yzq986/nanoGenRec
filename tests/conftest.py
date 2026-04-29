"""Shared fixtures for all tests.

Tiny-model helpers are intentionally CPU-only and fast (<1s each).
All random state is seeded for determinism.
"""

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

import torch

from ntp.model import NTPModel, SIDTrie
from ntp.features import REGISTRY, FeatureDef


# ── Model factories ──────────────────────────────────────────────────────────

N_LAYERS = 3
CLUSTERS = 32   # small enough for fast softmax
EMBED_DIM = 64
N_HEADS = 4
N_TRANSFORMER = 2
SEQ_LEN = 64    # max_seq_len

# Derived from registry — no hardcoded sizes here
_EMBED_ADD_FEATURES = [k for k, v in REGISTRY.items() if v.inject == 'embed_add']


def make_model(features: bool = False, seed: int = 42) -> NTPModel:
    """Tiny NTPModel, CPU, eval mode.

    Args:
        features: if True, activate all embed_add features from REGISTRY.
    """
    torch.manual_seed(seed)
    model = NTPModel(
        n_clusters_per_layer=[CLUSTERS] * N_LAYERS,
        n_sid_layers=N_LAYERS,
        n_items=10,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_transformer_layers=N_TRANSFORMER,
        dropout=0.0,
        use_moe=False,
        max_seq_len=SEQ_LEN,
        active_features=_EMBED_ADD_FEATURES if features else [],
    )
    model.eval()
    return model


def make_trie(n_items: int = 80, seed: int = 7) -> SIDTrie:
    """Small SIDTrie with random SIDs."""
    torch.manual_seed(seed)
    sid_to_items: dict = {}
    for i in range(n_items):
        sid = '_'.join(
            str(torch.randint(0, CLUSTERS, (1,)).item())
            for _ in range(N_LAYERS)
        )
        sid_to_items.setdefault(sid, set()).add(i)
    return SIDTrie(sid_to_items, N_LAYERS)


def make_ctx(batch: int = 2, length: int = 12, seed: int = 0) -> torch.Tensor:
    """Random context token tensor (B, T)."""
    torch.manual_seed(seed)
    return torch.randint(0, CLUSTERS, (batch, length))


def make_sid(batch: int = 2, seed: int = 1) -> torch.Tensor:
    """Random SID tensor (B, N_LAYERS)."""
    torch.manual_seed(seed)
    return torch.randint(0, CLUSTERS, (batch, N_LAYERS))


def make_features(batch: int, length: int, seed: int = 2) -> dict:
    """Random side features dict for all embed_add features in REGISTRY.

    Returns dict[str, Tensor] keyed by feature name — pass directly as
    side_features / ctx_side_features.
    """
    torch.manual_seed(seed)
    result = {}
    for key in _EMBED_ADD_FEATURES:
        fdef = REGISTRY[key]
        size = fdef.emb_size
        dtype = torch.long if fdef.dtype == 'long' else torch.float32
        result[key] = torch.randint(0, size, (batch, length)).to(dtype)
    return result
