"""Numerical equivalence tests for KV cache beam search.

Verifies that the KV-cached code path produces identical results
to the legacy (no-cache) path at every level:
  1. TransformerLayer: full vs incremental forward
  2. NTPModel.forward_cached: cold start vs incremental
  3. constrained_beam_search: cached vs legacy
"""

import os
import sys

# Module resolution: add parent of repo root to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(repo_root))

import torch
import torch.nn.functional as F

from ntp.model import (
    NTPModel,
    SIDTrie,
    TransformerLayer,
    constrained_beam_search,
    _constrained_beam_search_legacy,
)


def _make_model(n_layers=3, clusters_per_layer=64, embed_dim=64, n_heads=4,
                n_transformer_layers=2, seed=42):
    """Create a small NTPModel for testing."""
    torch.manual_seed(seed)
    model = NTPModel(
        n_clusters_per_layer=[clusters_per_layer] * n_layers,
        n_sid_layers=n_layers,
        n_items=10,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
        dropout=0.0,
        use_moe=False,
        max_seq_len=128,
    )
    model.eval()
    return model


def _make_trie(n_layers=3, clusters_per_layer=64, n_items=100, seed=42):
    """Create a small SIDTrie for testing."""
    torch.manual_seed(seed)
    sid_to_items = {}
    for i in range(n_items):
        sid = '_'.join(
            str(torch.randint(0, clusters_per_layer, (1,)).item())
            for _ in range(n_layers))
        if sid not in sid_to_items:
            sid_to_items[sid] = set()
        sid_to_items[sid].add(i)
    return SIDTrie(sid_to_items, n_layers)


def test_transformer_layer_kv_cache():
    """TransformerLayer: full forward == incremental forward with cache."""
    torch.manual_seed(42)
    layer = TransformerLayer(embed_dim=64, n_heads=4, dropout=0.0, causal=True)
    layer.eval()

    x_full = torch.randn(1, 10, 64)

    # Full forward
    with torch.no_grad():
        out_full = layer(x_full)

    # Incremental: first 7 tokens, then 3 new tokens
    x_prefix = x_full[:, :7, :]
    x_new = x_full[:, 7:, :]

    with torch.no_grad():
        out_prefix, kv_cache = layer(x_prefix, use_cache=True)
        out_incr, _ = layer(x_new, kv_cache=kv_cache, use_cache=True)

    # The output for the last 3 positions should match
    diff = (out_full[:, 7:, :] - out_incr).abs().max().item()
    assert diff < 1e-5, f"TransformerLayer incremental mismatch: max diff = {diff}"

    # The output for the first 7 positions should also match
    diff_prefix = (out_full[:, :7, :] - out_prefix).abs().max().item()
    assert diff_prefix < 1e-5, f"TransformerLayer prefix mismatch: max diff = {diff_prefix}"

    print(f"  [PASS] TransformerLayer KV cache (max diff = {max(diff, diff_prefix):.2e})")


def test_transformer_layer_single_token():
    """TransformerLayer: single-token incremental (the beam search case)."""
    torch.manual_seed(42)
    layer = TransformerLayer(embed_dim=64, n_heads=4, dropout=0.0, causal=True)
    layer.eval()

    x_full = torch.randn(1, 8, 64)

    with torch.no_grad():
        out_full = layer(x_full)

    # Encode first 7, then 1 new token
    with torch.no_grad():
        _, kv_cache = layer(x_full[:, :7, :], use_cache=True)
        out_last, _ = layer(x_full[:, 7:8, :], kv_cache=kv_cache, use_cache=True)

    diff = (out_full[:, 7:8, :] - out_last).abs().max().item()
    assert diff < 1e-5, f"Single-token mismatch: max diff = {diff}"
    print(f"  [PASS] TransformerLayer single-token (max diff = {diff:.2e})")


def test_forward_cached_cold_start():
    """NTPModel.forward_cached cold start matches forward()."""
    model = _make_model()
    input_tokens = torch.randint(0, 64, (1, 15))

    with torch.no_grad():
        logits_old = model.forward(input_tokens)
        logits_new, _ = model.forward_cached(input_tokens)

    diff = (logits_old - logits_new).abs().max().item()
    assert diff < 1e-5, f"Cold start mismatch: max diff = {diff}"
    print(f"  [PASS] forward_cached cold start (max diff = {diff:.2e})")


def test_forward_cached_incremental():
    """NTPModel.forward_cached incremental matches full forward."""
    model = _make_model()
    ctx = torch.randint(0, 64, (1, 12))
    gen = torch.randint(0, 64, (1, 3))

    with torch.no_grad():
        logits_full = model.forward(ctx, gen)
        # Incremental: encode ctx first, then gen
        _, kv = model.forward_cached(ctx)
        logits_incr, _ = model.forward_cached(generated_tokens=gen, kv_caches=kv)

    diff = (logits_full - logits_incr).abs().max().item()
    assert diff < 1e-5, f"Incremental mismatch: max diff = {diff}"
    print(f"  [PASS] forward_cached incremental (max diff = {diff:.2e})")


def test_forward_cached_token_by_token():
    """NTPModel.forward_cached token-by-token matches full forward."""
    model = _make_model()
    ctx = torch.randint(0, 64, (1, 9))
    gen = torch.randint(0, 64, (1, 3))

    with torch.no_grad():
        logits_full = model.forward(ctx, gen)
        # Token-by-token
        _, kv = model.forward_cached(ctx)
        for i in range(3):
            logits_step, kv = model.forward_cached(
                generated_tokens=gen[:, i:i+1], kv_caches=kv)

    diff = (logits_full - logits_step).abs().max().item()
    assert diff < 1e-4, f"Token-by-token mismatch: max diff = {diff}"
    print(f"  [PASS] forward_cached token-by-token (max diff = {diff:.2e})")


def test_beam_search_equivalence():
    """constrained_beam_search: cached vs legacy produce identical beams."""
    model = _make_model(n_layers=3, clusters_per_layer=32, embed_dim=64)
    trie = _make_trie(n_layers=3, clusters_per_layer=32, n_items=200)
    ctx = torch.randint(0, 32, (1, 12))

    with torch.no_grad():
        beams_legacy, scores_legacy, _ = _constrained_beam_search_legacy(
            model, ctx, trie, beam_size=20)
        beams_cached, scores_cached, _ = constrained_beam_search(
            model, ctx, trie, beam_size=20)

    # Compare beams
    assert beams_legacy.shape == beams_cached.shape, \
        f"Shape mismatch: {beams_legacy.shape} vs {beams_cached.shape}"
    beam_match = (beams_legacy == beams_cached).all().item()
    assert beam_match, "Beam search beams differ!"

    score_diff = (scores_legacy - scores_cached).abs().max().item()
    assert score_diff < 1e-4, f"Beam search score mismatch: {score_diff}"
    print(f"  [PASS] Beam search equivalence (score diff = {score_diff:.2e})")


def test_beam_search_with_prefix():
    """constrained_beam_search with prefix: cached vs legacy."""
    model = _make_model(n_layers=3, clusters_per_layer=32, embed_dim=64)
    trie = _make_trie(n_layers=3, clusters_per_layer=32, n_items=200)
    ctx = torch.randint(0, 32, (1, 12))
    prefix = torch.randint(0, 32, (1, 1))  # lock L0

    with torch.no_grad():
        beams_legacy, scores_legacy, _ = _constrained_beam_search_legacy(
            model, ctx, trie, beam_size=20, prefix=prefix)
        beams_cached, scores_cached, _ = constrained_beam_search(
            model, ctx, trie, beam_size=20, prefix=prefix)

    assert beams_legacy.shape == beams_cached.shape, \
        f"Shape mismatch: {beams_legacy.shape} vs {beams_cached.shape}"
    beam_match = (beams_legacy == beams_cached).all().item()
    assert beam_match, "Prefix beam search beams differ!"

    score_diff = (scores_legacy - scores_cached).abs().max().item()
    assert score_diff < 1e-4, f"Prefix beam search score mismatch: {score_diff}"
    print(f"  [PASS] Beam search with prefix (score diff = {score_diff:.2e})")


def test_beam_search_cache_reuse():
    """Cross-pass KV cache reuse: same results with pre-computed cache."""
    model = _make_model(n_layers=3, clusters_per_layer=32, embed_dim=64)
    trie = _make_trie(n_layers=3, clusters_per_layer=32, n_items=200)
    ctx = torch.randint(0, 32, (1, 12))

    with torch.no_grad():
        # First call: produces ctx_kv_caches
        beams1, scores1, ctx_kv = constrained_beam_search(
            model, ctx, trie, beam_size=20)

        # Second call with prefix, reusing ctx_kv
        prefix = torch.randint(0, 32, (1, 1))
        beams2_reuse, scores2_reuse, _ = constrained_beam_search(
            model, ctx, trie, beam_size=20, prefix=prefix,
            ctx_kv_caches=ctx_kv)

        # Same call without cache reuse (fresh encode)
        beams2_fresh, scores2_fresh, _ = constrained_beam_search(
            model, ctx, trie, beam_size=20, prefix=prefix)

    assert beams2_reuse.shape == beams2_fresh.shape
    beam_match = (beams2_reuse == beams2_fresh).all().item()
    assert beam_match, "Cache reuse beams differ!"

    score_diff = (scores2_reuse - scores2_fresh).abs().max().item()
    assert score_diff < 1e-4, f"Cache reuse score mismatch: {score_diff}"
    print(f"  [PASS] Cross-pass cache reuse (score diff = {score_diff:.2e})")


if __name__ == '__main__':
    print("KV Cache Numerical Equivalence Tests")
    print("=" * 50)

    print("\n1. TransformerLayer")
    test_transformer_layer_kv_cache()
    test_transformer_layer_single_token()

    print("\n2. NTPModel.forward_cached")
    test_forward_cached_cold_start()
    test_forward_cached_incremental()
    test_forward_cached_token_by_token()

    print("\n3. constrained_beam_search")
    test_beam_search_equivalence()
    test_beam_search_with_prefix()
    test_beam_search_cache_reuse()

    print("\n" + "=" * 50)
    print("All tests passed!")
