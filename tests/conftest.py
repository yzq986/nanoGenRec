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


# ── Model factories ──────────────────────────────────────────────────────────

N_LAYERS = 3
CLUSTERS = 32   # small enough for fast softmax
EMBED_DIM = 64
N_HEADS = 4
N_TRANSFORMER = 2
N_TIME_BUCKETS = 8
N_ACTION_LEVELS = 4
SEQ_LEN = 64    # max_seq_len


def make_model(features: bool = False, seed: int = 42) -> NTPModel:
    """Tiny NTPModel, CPU, eval mode.

    Args:
        features: if True, add time_gap_emb + action_emb.
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
        n_time_buckets=N_TIME_BUCKETS if features else 0,
        n_action_levels=N_ACTION_LEVELS if features else 0,
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


def make_features(batch: int, length: int, seed: int = 2):
    """Random time_gaps (B, T) and action_levels (B, T)."""
    torch.manual_seed(seed)
    tg = torch.randint(0, N_TIME_BUCKETS, (batch, length))
    al = torch.randint(0, N_ACTION_LEVELS, (batch, length))
    return tg, al
