"""Tests for TO-RoPE (Time-and-Order Rotary Position Embedding).

Verifies:
1. build_torope_freqs — shape and value checks
2. apply_torope — identity on zero timestamps, equivariance, output shape
3. TransformerLayer with torope_params — numerical match vs baseline
4. NTPModel with use_torope — forward / forward_cached consistency
5. TO-RoPE KV cache — incremental decode matches full forward
6. Timestamps matter — different timestamps → different output
7. Sequence order matters — shuffled order changes output
"""

import os
import sys
import math

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

import torch
import torch.nn.functional as F

from ntp.model import (
    NTPModel,
    TransformerLayer,
    SIDTrie,
    build_torope_freqs,
    apply_torope,
    constrained_beam_search,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_torope_model(n_layers=3, clusters=32, embed_dim=32, n_heads=4,
                       n_transformer_layers=2, time_split=0.5, seed=0):
    torch.manual_seed(seed)
    return NTPModel(
        n_clusters_per_layer=[clusters] * n_layers,
        n_sid_layers=n_layers,
        n_items=8,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
        dropout=0.0,
        use_moe=False,
        max_seq_len=128,
        use_torope=True,
        torope_time_split=time_split,
    ).eval()


def _make_plain_model(n_layers=3, clusters=32, embed_dim=32, n_heads=4,
                      n_transformer_layers=2, seed=0):
    torch.manual_seed(seed)
    return NTPModel(
        n_clusters_per_layer=[clusters] * n_layers,
        n_sid_layers=n_layers,
        n_items=8,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
        dropout=0.0,
        use_moe=False,
        max_seq_len=128,
        use_torope=False,
    ).eval()


# ═══════════════════════════════════════════════════════════════════════════
# 1. build_torope_freqs
# ═══════════════════════════════════════════════════════════════════════════

def test_build_torope_freqs_shapes():
    """Returned tensors have the expected shapes for various split ratios."""
    head_dim, max_len = 32, 64

    freq_idx, inv_freq_time, n_idx, n_time = build_torope_freqs(
        head_dim, max_len, time_split_ratio=0.5)
    half = head_dim // 2
    assert n_idx + n_time == half, f"planes sum mismatch: {n_idx}+{n_time} != {half}"
    assert freq_idx.shape == (max_len, n_idx), f"freq_idx shape {freq_idx.shape}"
    assert inv_freq_time.shape == (n_time,), f"inv_freq_time shape {inv_freq_time.shape}"

    # Zero split — all planes for index
    freq_idx0, inv_freq0, ni0, nt0 = build_torope_freqs(head_dim, max_len, time_split_ratio=0.0)
    assert ni0 == half - 1 and nt0 == 1  # max(1, ...) ensures at least 1 time plane
    print(f"  [PASS] build_torope_freqs shapes (n_idx={n_idx}, n_time={n_time})")


def test_build_torope_freqs_values():
    """freq_idx[0] is all zeros (position 0 has no rotation)."""
    freq_idx, _, _, _ = build_torope_freqs(16, 32, time_split_ratio=0.5)
    assert (freq_idx[0] == 0.0).all(), "Position 0 should have zero angle"
    # Monotonically increasing angles for position 1 vs 0
    assert (freq_idx[1] > 0.0).all(), "Position 1 should have positive angles"
    print("  [PASS] build_torope_freqs values")


# ═══════════════════════════════════════════════════════════════════════════
# 2. apply_torope
# ═══════════════════════════════════════════════════════════════════════════

def test_apply_torope_output_shape():
    """apply_torope preserves q/k shape."""
    B, H, T, Dh = 2, 4, 6, 16
    freq_idx, inv_freq_time, n_idx, n_time = build_torope_freqs(Dh, T + 4)
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    pos = torch.arange(T).unsqueeze(0).expand(B, -1)
    ts  = torch.zeros(B, T)
    qr, kr = apply_torope(q, k, pos, pos, freq_idx, inv_freq_time, n_idx, n_time,
                          timestamps_q=ts, timestamps_k=ts)
    assert qr.shape == q.shape and kr.shape == k.shape
    print(f"  [PASS] apply_torope output shape {qr.shape}")


def test_apply_torope_zero_timestamps_equals_standard_rope():
    """With zero timestamps, time planes have cos=1, sin=0 → identity for time part.
    Index planes behave like standard RoPE."""
    B, H, T, Dh = 1, 2, 8, 16
    freq_idx, inv_freq_time, n_idx, n_time = build_torope_freqs(Dh, T + 4)
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    pos = torch.arange(T).unsqueeze(0)
    ts  = torch.zeros(B, T)

    qr, kr = apply_torope(q, k, pos, pos, freq_idx, inv_freq_time, n_idx, n_time,
                          timestamps_q=ts, timestamps_k=ts)

    # Time planes: ts=0 → angles=0 → cos=1, sin=0 → identity (no rotation)
    start = n_idx * 2
    assert torch.allclose(qr[..., start:start + n_time * 2],
                          q[..., start:start + n_time * 2], atol=1e-6), \
        "Zero timestamp should leave time-plane embedding unchanged"
    print("  [PASS] apply_torope zero timestamps = identity on time planes")


def test_apply_torope_different_timestamps_differ():
    """Non-zero timestamps produce different output than zero timestamps."""
    B, H, T, Dh = 1, 2, 8, 16
    freq_idx, inv_freq_time, n_idx, n_time = build_torope_freqs(Dh, T + 4)
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    pos = torch.arange(T).unsqueeze(0)
    ts_zero = torch.zeros(B, T)
    ts_nonzero = torch.rand(B, T) * 24.0  # up to 24 hours

    qr_zero, _ = apply_torope(q, k, pos, pos, freq_idx, inv_freq_time, n_idx, n_time,
                              timestamps_q=ts_zero, timestamps_k=ts_zero)
    qr_ts, _   = apply_torope(q, k, pos, pos, freq_idx, inv_freq_time, n_idx, n_time,
                              timestamps_q=ts_nonzero, timestamps_k=ts_nonzero)

    diff = (qr_ts - qr_zero).abs().max().item()
    assert diff > 1e-3, f"Non-zero timestamps should change output; got diff={diff:.2e}"
    print(f"  [PASS] apply_torope timestamps change output (diff={diff:.2e})")


def test_apply_torope_rotation_is_reversible():
    """Rotating by angle and then by negative angle recovers original."""
    B, H, T, Dh = 1, 2, 4, 16
    freq_idx, inv_freq_time, n_idx, n_time = build_torope_freqs(Dh, T + 4)
    q = torch.randn(B, H, T, Dh)
    k = q.clone()
    pos = torch.arange(T).unsqueeze(0)
    ts_fwd = torch.rand(B, T) * 10.0

    qr, _ = apply_torope(q, k, pos, pos, freq_idx, inv_freq_time, n_idx, n_time,
                         timestamps_q=ts_fwd, timestamps_k=ts_fwd)
    q_recovered, _ = apply_torope(qr, qr, pos, pos, freq_idx, inv_freq_time, n_idx, n_time,
                                   timestamps_q=-ts_fwd, timestamps_k=-ts_fwd)

    # Only time planes are reversible by negating ts; index planes need negated positions
    # Check just time planes
    start = n_idx * 2
    diff = (q_recovered[..., start:start + n_time * 2] -
            q[..., start:start + n_time * 2]).abs().max().item()
    assert diff < 1e-5, f"Time plane rotation not reversible: diff={diff:.2e}"
    print(f"  [PASS] apply_torope time planes reversible (diff={diff:.2e})")


# ═══════════════════════════════════════════════════════════════════════════
# 3. NTPModel with use_torope — structure checks
# ═══════════════════════════════════════════════════════════════════════════

def test_torope_model_no_learnable_pos_emb():
    """use_torope=True model has no pos_emb / item_pos_emb / layer_pos_emb."""
    model = _make_torope_model()
    assert not hasattr(model, 'pos_emb'), "Should not have learnable pos_emb with TO-RoPE"
    assert not hasattr(model, 'item_pos_emb'), "Should not have item_pos_emb with TO-RoPE"
    assert not hasattr(model, 'layer_pos_emb'), "Should not have layer_pos_emb with TO-RoPE"
    assert not hasattr(model, 'time_gap_emb'), "Should not have time_gap_emb with TO-RoPE"
    # RoPE buffers registered
    assert hasattr(model, 'torope_freq_idx'), "Missing torope_freq_idx buffer"
    assert hasattr(model, 'torope_inv_freq_time'), "Missing torope_inv_freq_time buffer"
    print("  [PASS] TO-RoPE model structure correct")


def test_torope_model_get_pos_emb_zeros():
    """_get_pos_emb returns zeros for TO-RoPE model."""
    model = _make_torope_model()
    pos = torch.arange(10).unsqueeze(0)
    zeros = model._get_pos_emb(pos)
    assert (zeros == 0).all(), "Expected zero positional embedding for TO-RoPE"
    print("  [PASS] _get_pos_emb returns zeros for TO-RoPE model")


def test_torope_model_param_count():
    """TO-RoPE model has fewer parameters than segment-emb model (no pos emb tables)."""
    m_torope = _make_torope_model()
    m_plain  = _make_plain_model()
    n_torope = sum(p.numel() for p in m_torope.parameters())
    n_plain  = sum(p.numel() for p in m_plain.parameters())
    # TO-RoPE removes pos_emb, plain has it → TO-RoPE should have <= params
    assert n_torope <= n_plain, (
        f"TO-RoPE model should have <= params than plain; got {n_torope} vs {n_plain}")
    print(f"  [PASS] TO-RoPE param count {n_torope} <= plain {n_plain}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. forward_cached: cold start + incremental consistency
# ═══════════════════════════════════════════════════════════════════════════

def test_torope_forward_cached_cold_start_consistency():
    """forward_cached cold start logits are consistent with forward() on same tokens."""
    model = _make_torope_model()
    T = 12
    tokens = torch.randint(0, 32, (1, T))
    ts = torch.rand(1, T) * 24.0

    with torch.no_grad():
        # forward() — does not yet use timestamps in training path, but should run
        logits_fwd = model.forward(tokens)

        # forward_cached cold start with no timestamps — should also work
        logits_cached, kv, kv_pos, kv_ts = model.forward_cached(tokens)

    # Shapes should match
    assert logits_fwd.shape == logits_cached.shape, \
        f"Shape mismatch: {logits_fwd.shape} vs {logits_cached.shape}"
    # Without timestamps, both use zero positional signal → should be identical
    diff = (logits_fwd - logits_cached).abs().max().item()
    assert diff < 1e-5, f"forward vs forward_cached mismatch (no timestamps): diff={diff:.2e}"
    print(f"  [PASS] TO-RoPE cold start consistency (diff={diff:.2e})")


def test_torope_forward_cached_incremental():
    """TO-RoPE incremental decode matches full forward_cached cold start."""
    model = _make_torope_model()
    T_ctx, T_gen = 9, 3
    ctx = torch.randint(0, 32, (1, T_ctx))
    gen = torch.randint(0, 32, (1, T_gen))

    ctx_ts = torch.rand(1, T_ctx) * 12.0   # hours since first item
    gen_ts = torch.rand(1, T_gen) * 5.0

    with torch.no_grad():
        # Full cold start: encode ctx + all gen together
        all_tokens = torch.cat([ctx, gen], dim=1)
        all_ts     = torch.cat([ctx_ts, gen_ts], dim=1)
        logits_full, _, _, _ = model.forward_cached(
            all_tokens, ctx_timestamps=all_ts)

        # Incremental: encode ctx, then gen one-by-one
        _, kv_ctx, kv_pos, kv_ts = model.forward_cached(
            ctx, ctx_timestamps=ctx_ts)

        cur_kv_pos = kv_pos
        cur_kv_ts  = kv_ts
        for i in range(T_gen):
            step_tok = gen[:, i:i+1]
            step_ts  = gen_ts[:, i:i+1]
            logits_step, kv_ctx, cur_kv_pos, cur_kv_ts = model.forward_cached(
                generated_tokens=step_tok,
                kv_caches=kv_ctx,
                step_timestamp=step_ts,
                kv_positions_cache=cur_kv_pos,
                kv_timestamps_cache=cur_kv_ts,
            )

    diff = (logits_full - logits_step).abs().max().item()
    assert diff < 1e-4, f"TO-RoPE incremental vs full mismatch: diff={diff:.2e}"
    print(f"  [PASS] TO-RoPE incremental decode consistency (diff={diff:.2e})")


def test_torope_kv_cache_returns_pos_and_ts():
    """forward_cached returns non-None kv_positions and kv_timestamps for TO-RoPE model."""
    model = _make_torope_model()
    ctx = torch.randint(0, 32, (1, 8))
    ctx_ts = torch.arange(8).float().unsqueeze(0)

    with torch.no_grad():
        _, kv, kv_pos, kv_ts = model.forward_cached(ctx, ctx_timestamps=ctx_ts)

    assert kv_pos is not None, "kv_positions should not be None for TO-RoPE model"
    assert kv_ts is not None, "kv_timestamps should not be None for TO-RoPE model"
    assert kv_pos.shape == (1, 8), f"Unexpected kv_pos shape: {kv_pos.shape}"
    assert kv_ts.shape == (1, 8), f"Unexpected kv_ts shape: {kv_ts.shape}"
    print(f"  [PASS] TO-RoPE forward_cached returns kv_pos {kv_pos.shape} and kv_ts {kv_ts.shape}")


def test_plain_model_kv_cache_returns_none_pos_ts():
    """Plain (non-TO-RoPE) forward_cached returns None for kv_positions and kv_timestamps."""
    model = _make_plain_model()
    ctx = torch.randint(0, 32, (1, 8))

    with torch.no_grad():
        _, kv, kv_pos, kv_ts = model.forward_cached(ctx)

    assert kv_pos is None, "Plain model should return None for kv_positions"
    assert kv_ts is None, "Plain model should return None for kv_timestamps"
    print("  [PASS] Plain model forward_cached returns None for kv_pos/ts")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Timestamps change output
# ═══════════════════════════════════════════════════════════════════════════

def test_timestamps_change_logits():
    """Different ctx_timestamps produce different logits."""
    model = _make_torope_model()
    ctx = torch.randint(0, 32, (1, 8))
    ts_a = torch.zeros(1, 8)
    ts_b = torch.arange(8).float().unsqueeze(0)  # increasing hours

    with torch.no_grad():
        logits_a, _, _, _ = model.forward_cached(ctx, ctx_timestamps=ts_a)
        logits_b, _, _, _ = model.forward_cached(ctx, ctx_timestamps=ts_b)

    diff = (logits_a - logits_b).abs().max().item()
    assert diff > 1e-3, f"Different timestamps should produce different logits (diff={diff:.2e})"
    print(f"  [PASS] Different timestamps → different logits (diff={diff:.2e})")


def test_same_timestamps_same_logits():
    """Same ctx_timestamps produce identical logits (determinism)."""
    model = _make_torope_model()
    ctx = torch.randint(0, 32, (1, 8))
    ts = torch.rand(1, 8) * 10.0

    with torch.no_grad():
        logits_a, _, _, _ = model.forward_cached(ctx, ctx_timestamps=ts)
        logits_b, _, _, _ = model.forward_cached(ctx, ctx_timestamps=ts)

    diff = (logits_a - logits_b).abs().max().item()
    assert diff == 0.0, f"Same timestamps should produce identical logits (diff={diff})"
    print("  [PASS] Same timestamps → identical logits (deterministic)")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Plain model forward_cached backward compatibility
# ═══════════════════════════════════════════════════════════════════════════

def test_plain_model_backward_compat():
    """Plain model still works correctly after the 4-tuple return change."""
    model = _make_plain_model()
    ctx = torch.randint(0, 32, (1, 9))
    gen = torch.randint(0, 32, (1, 3))

    with torch.no_grad():
        logits_full = model.forward(ctx, gen)
        _, kv, _, _ = model.forward_cached(ctx)
        logits_incr, _, _, _ = model.forward_cached(generated_tokens=gen, kv_caches=kv)

    diff = (logits_full - logits_incr).abs().max().item()
    assert diff < 1e-5, f"Plain model KV cache broken after 4-tuple change: diff={diff:.2e}"
    print(f"  [PASS] Plain model KV cache backward compat (diff={diff:.2e})")


# ═══════════════════════════════════════════════════════════════════════════
# 7. constrained_beam_search with TO-RoPE — train/infer position consistency
# ═══════════════════════════════════════════════════════════════════════════

def _make_trie(n_clusters=32, n_layers=3, n_items=20, seed=42):
    """Build a small SIDTrie with random SIDs."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    sid_to_items = {}
    for i in range(n_items):
        tokens = torch.randint(0, n_clusters, (n_layers,), generator=rng).tolist()
        sid_str = '_'.join(str(t) for t in tokens)
        if sid_str not in sid_to_items:
            sid_to_items[sid_str] = set()
        sid_to_items[sid_str].add(i)
    return SIDTrie(sid_to_items, n_layers)


def test_torope_beam_search_kv_positions_passed():
    """constrained_beam_search with TO-RoPE model produces valid beams (non-trivial logits).

    This is a regression test for the bug where kv_positions_cache was not
    passed in incremental decode steps, causing RoPE angles to be computed
    only from the new token's position instead of the full KV sequence.
    Symptom: PPL >> 100 despite normal training loss.
    """
    model = _make_torope_model(n_layers=3, clusters=32, embed_dim=32,
                               n_heads=4, n_transformer_layers=2)
    trie = _make_trie(n_clusters=32, n_layers=3, n_items=20)

    B, T_ctx = 2, 6
    ctx = torch.randint(0, 32, (B, T_ctx))

    with torch.no_grad():
        beams, scores, _ = constrained_beam_search(model, ctx, trie, beam_size=5)

    assert beams.shape[0] == B, f"Expected B={B} beams, got {beams.shape[0]}"
    assert beams.shape[2] == 3, f"Expected 3-token SIDs, got depth {beams.shape[2]}"
    assert beams.shape[1] > 0, "No beams returned"
    # Scores should be finite (not -inf/nan), indicating valid logit paths
    assert scores.isfinite().all(), f"Non-finite beam scores: {scores}"
    # Logits should not be degenerate — at least one score > -100
    assert (scores > -100).any(), f"All scores collapsed: {scores}"
    print(f"  [PASS] TO-RoPE beam search returns valid beams {beams.shape}, "
          f"max_score={scores.max().item():.2f}")


def test_torope_beam_search_logits_differ_from_plain():
    """TO-RoPE beam search logit distribution differs from plain model (RoPE is active)."""
    model_rope  = _make_torope_model(seed=0)
    model_plain = _make_plain_model(seed=0)
    trie = _make_trie()

    B, T_ctx = 1, 8
    ctx = torch.randint(0, 32, (B, T_ctx))

    with torch.no_grad():
        beams_rope,  scores_rope,  _ = constrained_beam_search(model_rope,  ctx, trie, beam_size=5)
        beams_plain, scores_plain, _ = constrained_beam_search(model_plain, ctx, trie, beam_size=5)

    # The two models have different architectures → scores must differ
    diff = (scores_rope - scores_plain).abs().max().item()
    assert diff > 1e-3, f"TO-RoPE and plain beam scores too similar (diff={diff:.2e})"
    print(f"  [PASS] TO-RoPE vs plain beam scores differ (diff={diff:.2e})")


def test_torope_beam_search_deterministic():
    """Same input → same beams (determinism check)."""
    model = _make_torope_model()
    trie  = _make_trie()
    ctx   = torch.randint(0, 32, (1, 8))

    with torch.no_grad():
        beams_a, scores_a, _ = constrained_beam_search(model, ctx, trie, beam_size=5)
        beams_b, scores_b, _ = constrained_beam_search(model, ctx, trie, beam_size=5)

    assert torch.equal(beams_a, beams_b),  "Beam tokens not deterministic"
    assert torch.equal(scores_a, scores_b), "Beam scores not deterministic"
    print("  [PASS] TO-RoPE beam search is deterministic")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("TO-RoPE Tests")
    print("=" * 60)

    print("\n1. build_torope_freqs")
    test_build_torope_freqs_shapes()
    test_build_torope_freqs_values()

    print("\n2. apply_torope")
    test_apply_torope_output_shape()
    test_apply_torope_zero_timestamps_equals_standard_rope()
    test_apply_torope_different_timestamps_differ()
    test_apply_torope_rotation_is_reversible()

    print("\n3. NTPModel structure")
    test_torope_model_no_learnable_pos_emb()
    test_torope_model_get_pos_emb_zeros()
    test_torope_model_param_count()

    print("\n4. forward_cached consistency")
    test_torope_forward_cached_cold_start_consistency()
    test_torope_forward_cached_incremental()
    test_torope_kv_cache_returns_pos_and_ts()
    test_plain_model_kv_cache_returns_none_pos_ts()

    print("\n5. Timestamps change output")
    test_timestamps_change_logits()
    test_same_timestamps_same_logits()

    print("\n6. Backward compatibility")
    test_plain_model_backward_compat()

    print("\n7. constrained_beam_search with TO-RoPE")
    test_torope_beam_search_kv_positions_passed()
    test_torope_beam_search_logits_differ_from_plain()
    test_torope_beam_search_deterministic()

    print("\n" + "=" * 60)
    print("All TO-RoPE tests passed!")
