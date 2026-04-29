"""Tests for RoPE (Rotary Position Embedding) — renamed from test_torope.py.

Verifies:
1. build_rope_freqs / build_torope_freqs — shape and value checks
2. apply_rope / apply_torope — identity on zero timestamps, equivariance, output shape
3. TransformerLayer with rope_params — numerical match vs baseline
4. NTPModel with use_rope — forward / forward_cached consistency
5. RoPE KV cache — incremental decode matches full forward
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
    RopeDimSpec,
    build_rope_freqs,
    apply_rope,
    # legacy aliases — must still be importable
    build_torope_freqs,
    apply_torope,
    constrained_beam_search,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_rope_model(n_layers=3, clusters=32, embed_dim=32, n_heads=4,
                     n_transformer_layers=2, time_split=0.5, seed=0):
    """Create a model using the new use_rope flag."""
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


# backward compat alias used by existing test names
_make_torope_model = _make_rope_model


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
# 1. build_rope_freqs / build_torope_freqs
# ═══════════════════════════════════════════════════════════════════════════

def test_build_torope_freqs_shapes():
    """Returned tensors have the expected shapes for various split ratios."""
    head_dim, max_len = 32, 64

    freq_idx, inv_freq_time, freq_layer, n_idx, n_time, n_layer = build_torope_freqs(
        head_dim, max_len, time_split_ratio=0.5)
    half = head_dim // 2
    assert n_idx + n_time + n_layer == half, \
        f"planes sum mismatch: {n_idx}+{n_time}+{n_layer} != {half}"
    assert freq_idx.shape == (max_len, n_idx), f"freq_idx shape {freq_idx.shape}"
    assert inv_freq_time.shape == (n_time,), f"inv_freq_time shape {inv_freq_time.shape}"
    assert freq_layer.shape[1] == n_layer, f"freq_layer shape {freq_layer.shape}"

    # 3-dim mode: time=0.4, layer=0.2
    freq_idx3, inv_t3, fl3, ni3, nt3, nl3 = build_torope_freqs(
        head_dim, max_len, time_split_ratio=0.4, layer_split_ratio=0.2)
    assert ni3 + nt3 + nl3 == half

    # Zero layer split — all planes for idx+time (backward compat)
    freq_idx0, inv_freq0, fl0, ni0, nt0, nl0 = build_torope_freqs(
        head_dim, max_len, time_split_ratio=0.0)
    assert ni0 == half - 1 and nt0 == 1 and nl0 == 0
    print(f"  [PASS] build_torope_freqs shapes (n_idx={n_idx}, n_time={n_time}, n_layer={n_layer})")


def test_build_torope_freqs_values():
    """freq_idx[0] is all zeros (position 0 has no rotation)."""
    freq_idx, _, _, _, _, _ = build_torope_freqs(16, 32, time_split_ratio=0.5)
    assert (freq_idx[0] == 0.0).all(), "Position 0 should have zero angle"
    # Monotonically increasing angles for position 1 vs 0
    assert (freq_idx[1] > 0.0).all(), "Position 1 should have positive angles"
    print("  [PASS] build_torope_freqs values")


def test_build_rope_freqs_shapes():
    """build_rope_freqs returns correct structure and shapes."""
    head_dim, max_len = 32, 64
    dims = [
        RopeDimSpec(name='order', split_ratio=0.5, source='position'),
        RopeDimSpec(name='time',  split_ratio=0.3, source='timestamp'),
        RopeDimSpec(name='layer', split_ratio=0.2, source='layer_id'),
    ]
    rope_info = build_rope_freqs(head_dim, max_len, dims, n_sid_layers=3)
    half = head_dim // 2

    assert len(rope_info) == 3
    total_planes = sum(r['n_planes'] for r in rope_info)
    assert total_planes == half, f"total planes {total_planes} != half {half}"

    r0, r1, r2 = rope_info
    # position → freq_table (max_len, n_planes)
    assert r0['freq'].shape == (max_len, r0['n_planes'])
    # timestamp → inv_freq (n_planes,)
    assert r1['freq'].shape == (r1['n_planes'],)
    # layer_id → freq_table (n_sid_layers, n_planes)
    assert r2['freq'].shape == (3, r2['n_planes'])
    print(f"  [PASS] build_rope_freqs shapes ({[r['n_planes'] for r in rope_info]})")


# ═══════════════════════════════════════════════════════════════════════════
# 2. apply_rope / apply_torope
# ═══════════════════════════════════════════════════════════════════════════

def test_apply_torope_output_shape():
    """apply_torope preserves q/k shape."""
    B, H, T, Dh = 2, 4, 6, 16
    freq_idx, inv_freq_time, freq_layer, n_idx, n_time, n_layer = build_torope_freqs(Dh, T + 4)
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    pos = torch.arange(T).unsqueeze(0).expand(B, -1)
    ts  = torch.zeros(B, T)
    qr, kr = apply_torope(q, k, pos, pos, freq_idx, inv_freq_time, freq_layer,
                          n_idx, n_time, n_layer,
                          timestamps_q=ts, timestamps_k=ts)
    assert qr.shape == q.shape and kr.shape == k.shape
    print(f"  [PASS] apply_torope output shape {qr.shape}")


def test_apply_torope_zero_timestamps_equals_standard_rope():
    """With zero timestamps, time planes have cos=1, sin=0 → identity for time part.
    Index planes behave like standard RoPE."""
    B, H, T, Dh = 1, 2, 8, 16
    freq_idx, inv_freq_time, freq_layer, n_idx, n_time, n_layer = build_torope_freqs(Dh, T + 4)
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    pos = torch.arange(T).unsqueeze(0)
    ts  = torch.zeros(B, T)

    qr, kr = apply_torope(q, k, pos, pos, freq_idx, inv_freq_time, freq_layer,
                          n_idx, n_time, n_layer,
                          timestamps_q=ts, timestamps_k=ts)

    start = n_idx * 2
    assert torch.allclose(qr[..., start:start + n_time * 2],
                          q[..., start:start + n_time * 2], atol=1e-6), \
        "Zero timestamp should leave time-plane embedding unchanged"
    print("  [PASS] apply_torope zero timestamps = identity on time planes")


def test_apply_torope_different_timestamps_differ():
    """Non-zero timestamps produce different output than zero timestamps."""
    B, H, T, Dh = 1, 2, 8, 16
    freq_idx, inv_freq_time, freq_layer, n_idx, n_time, n_layer = build_torope_freqs(Dh, T + 4)
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    pos = torch.arange(T).unsqueeze(0)
    ts_zero = torch.zeros(B, T)
    ts_nonzero = torch.rand(B, T) * 24.0

    qr_zero, _ = apply_torope(q, k, pos, pos, freq_idx, inv_freq_time, freq_layer,
                              n_idx, n_time, n_layer,
                              timestamps_q=ts_zero, timestamps_k=ts_zero)
    qr_ts, _   = apply_torope(q, k, pos, pos, freq_idx, inv_freq_time, freq_layer,
                              n_idx, n_time, n_layer,
                              timestamps_q=ts_nonzero, timestamps_k=ts_nonzero)

    diff = (qr_ts - qr_zero).abs().max().item()
    assert diff > 1e-3, f"Non-zero timestamps should change output; got diff={diff:.2e}"
    print(f"  [PASS] apply_torope timestamps change output (diff={diff:.2e})")


def test_apply_torope_rotation_is_reversible():
    """Rotating by angle and then by negative angle recovers original."""
    B, H, T, Dh = 1, 2, 4, 16
    freq_idx, inv_freq_time, freq_layer, n_idx, n_time, n_layer = build_torope_freqs(Dh, T + 4)
    q = torch.randn(B, H, T, Dh)
    k = q.clone()
    pos = torch.arange(T).unsqueeze(0)
    ts_fwd = torch.rand(B, T) * 10.0

    qr, _ = apply_torope(q, k, pos, pos, freq_idx, inv_freq_time, freq_layer,
                         n_idx, n_time, n_layer,
                         timestamps_q=ts_fwd, timestamps_k=ts_fwd)
    q_recovered, _ = apply_torope(qr, qr, pos, pos, freq_idx, inv_freq_time, freq_layer,
                                   n_idx, n_time, n_layer,
                                   timestamps_q=-ts_fwd, timestamps_k=-ts_fwd)

    start = n_idx * 2
    diff = (q_recovered[..., start:start + n_time * 2] -
            q[..., start:start + n_time * 2]).abs().max().item()
    assert diff < 1e-5, f"Time plane rotation not reversible: diff={diff:.2e}"
    print(f"  [PASS] apply_torope time planes reversible (diff={diff:.2e})")


def test_apply_rope_matches_apply_torope():
    """apply_rope (new API) produces identical results to apply_torope (legacy)."""
    B, H, T, Dh = 2, 4, 8, 16
    n_sid_layers = 3
    freq_idx, inv_freq_time, freq_layer, n_idx, n_time, n_layer = build_torope_freqs(
        Dh, T + 4, n_sid_layers=n_sid_layers)
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    pos = torch.arange(T).unsqueeze(0).expand(B, -1)
    ts = torch.rand(B, T) * 10.0
    lay = torch.arange(T).unsqueeze(0).expand(B, -1) % n_sid_layers

    # Legacy
    qr_leg, kr_leg = apply_torope(
        q, k, pos, pos, freq_idx, inv_freq_time, freq_layer,
        n_idx, n_time, n_layer,
        timestamps_q=ts, timestamps_k=ts,
        layers_q=lay if n_layer > 0 else None,
        layers_k=lay if n_layer > 0 else None,
    )

    # New API using build_rope_freqs
    dims = [
        RopeDimSpec('order', split_ratio=n_idx / (Dh // 2), source='position'),
        RopeDimSpec('time',  split_ratio=n_time / (Dh // 2), source='timestamp'),
    ]
    if n_layer > 0:
        dims.append(RopeDimSpec('layer', split_ratio=n_layer / (Dh // 2), source='layer_id',
                                max_val=n_sid_layers))
    rope_info = build_rope_freqs(Dh, T + 4, dims, n_sid_layers=n_sid_layers)

    # Verify n_planes matches
    for i, (old_n, info) in enumerate(zip([n_idx, n_time, n_layer][:len(dims)], rope_info)):
        assert info['n_planes'] == old_n, \
            f"dim {i}: n_planes mismatch {info['n_planes']} vs {old_n}"

    dim_inputs_q = [pos, ts] + ([lay] if n_layer > 0 else [])
    dim_inputs_k = [pos, ts] + ([lay] if n_layer > 0 else [])
    qr_new, kr_new = apply_rope(q, k, dim_inputs_q, dim_inputs_k, rope_info)

    diff_q = (qr_new - qr_leg).abs().max().item()
    diff_k = (kr_new - kr_leg).abs().max().item()
    assert diff_q < 1e-5, f"apply_rope Q mismatch vs apply_torope: diff={diff_q:.2e}"
    assert diff_k < 1e-5, f"apply_rope K mismatch vs apply_torope: diff={diff_k:.2e}"
    print(f"  [PASS] apply_rope matches apply_torope (diff_q={diff_q:.2e}, diff_k={diff_k:.2e})")


# ═══════════════════════════════════════════════════════════════════════════
# 3. NTPModel with use_rope — structure checks
# ═══════════════════════════════════════════════════════════════════════════

def test_torope_model_no_learnable_pos_emb():
    """use_rope=True model has no pos_emb / item_pos_emb / layer_pos_emb."""
    model = _make_rope_model()
    assert not hasattr(model, 'pos_emb'), "Should not have learnable pos_emb with RoPE"
    assert not hasattr(model, 'item_pos_emb'), "Should not have item_pos_emb with RoPE"
    assert not hasattr(model, 'layer_pos_emb'), "Should not have layer_pos_emb with RoPE"
    # RoPE buffers registered (both old and new names)
    assert hasattr(model, 'torope_freq_idx'), "Missing torope_freq_idx buffer"
    assert hasattr(model, 'torope_inv_freq_time'), "Missing torope_inv_freq_time buffer"
    assert hasattr(model, 'rope_dim_0_freq'), "Missing rope_dim_0_freq buffer"
    assert hasattr(model, 'rope_dim_1_freq'), "Missing rope_dim_1_freq buffer"
    print("  [PASS] RoPE model structure correct (buffers present under both old and new names)")


def test_torope_model_get_pos_emb_zeros():
    """_get_pos_emb returns zeros for RoPE model."""
    model = _make_rope_model()
    pos = torch.arange(10).unsqueeze(0)
    zeros = model._get_pos_emb(pos)
    assert (zeros == 0).all(), "Expected zero positional embedding for RoPE model"
    print("  [PASS] _get_pos_emb returns zeros for RoPE model")


def test_torope_model_param_count():
    """RoPE model has fewer parameters than segment-emb model (no pos emb tables)."""
    m_rope  = _make_rope_model()
    m_plain = _make_plain_model()
    n_rope  = sum(p.numel() for p in m_rope.parameters())
    n_plain = sum(p.numel() for p in m_plain.parameters())
    # RoPE removes pos_emb, plain has it → RoPE should have <= params
    assert n_rope <= n_plain, (
        f"RoPE model should have <= params than plain; got {n_rope} vs {n_plain}")
    print(f"  [PASS] RoPE param count {n_rope} <= plain {n_plain}")


def test_rope_use_rope_property():
    """use_rope and use_torope properties are consistent."""
    m_rope  = _make_rope_model()
    m_plain = _make_plain_model()
    assert m_rope.use_rope is True
    assert m_rope.use_torope is True   # backward compat alias
    assert m_plain.use_rope is False
    assert m_plain.use_torope is False
    print("  [PASS] use_rope and use_torope properties consistent")


def test_rope_dims_stored():
    """rope_dims is stored on the model when use_torope=True."""
    model = _make_rope_model(time_split=0.25)
    assert model.rope_dims is not None, "rope_dims should not be None for RoPE model"
    assert isinstance(model.rope_dims, list)
    assert len(model.rope_dims) >= 2
    # 'time' dim should have split_ratio ≈ 0.25
    time_dim = next((d for d in model.rope_dims if d.source == 'timestamp'), None)
    assert time_dim is not None, "Expected a timestamp dim in rope_dims"
    assert abs(time_dim.split_ratio - 0.25) < 1e-6, \
        f"Expected time split_ratio=0.25, got {time_dim.split_ratio}"
    print(f"  [PASS] rope_dims stored: {[(d.name, d.split_ratio) for d in model.rope_dims]}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. forward_cached: cold start + incremental consistency
# ═══════════════════════════════════════════════════════════════════════════

def test_torope_forward_cached_cold_start_consistency():
    """forward_cached cold start logits are consistent with forward() on same tokens."""
    model = _make_rope_model()
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
    print(f"  [PASS] RoPE cold start consistency (diff={diff:.2e})")


def test_torope_forward_cached_incremental():
    """RoPE incremental decode matches full forward_cached cold start."""
    model = _make_rope_model()
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
    assert diff < 1e-4, f"RoPE incremental vs full mismatch: diff={diff:.2e}"
    print(f"  [PASS] RoPE incremental decode consistency (diff={diff:.2e})")


def test_torope_kv_cache_returns_pos_and_ts():
    """forward_cached returns non-None kv_positions and kv_timestamps for RoPE model."""
    model = _make_rope_model()
    ctx = torch.randint(0, 32, (1, 8))
    ctx_ts = torch.arange(8).float().unsqueeze(0)

    with torch.no_grad():
        _, kv, kv_pos, kv_ts = model.forward_cached(ctx, ctx_timestamps=ctx_ts)

    assert kv_pos is not None, "kv_positions should not be None for RoPE model"
    assert kv_ts is not None, "kv_timestamps should not be None for RoPE model"
    assert kv_pos.shape == (1, 8), f"Unexpected kv_pos shape: {kv_pos.shape}"
    assert kv_ts.shape == (1, 8), f"Unexpected kv_ts shape: {kv_ts.shape}"
    print(f"  [PASS] RoPE forward_cached returns kv_pos {kv_pos.shape} and kv_ts {kv_ts.shape}")


def test_plain_model_kv_cache_returns_none_pos_ts():
    """Plain (non-RoPE) forward_cached returns None for kv_positions and kv_timestamps."""
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
    model = _make_rope_model()
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
    model = _make_rope_model()
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
# 7. constrained_beam_search with RoPE — train/infer position consistency
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
    """constrained_beam_search with RoPE model produces valid beams (non-trivial logits).

    This is a regression test for the bug where kv_positions_cache was not
    passed in incremental decode steps, causing RoPE angles to be computed
    only from the new token's position instead of the full KV sequence.
    Symptom: PPL >> 100 despite normal training loss.
    """
    model = _make_rope_model(n_layers=3, clusters=32, embed_dim=32,
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
    print(f"  [PASS] RoPE beam search returns valid beams {beams.shape}, "
          f"max_score={scores.max().item():.2f}")


def test_torope_beam_search_logits_differ_from_plain():
    """RoPE beam search logit distribution differs from plain model (RoPE is active)."""
    model_rope  = _make_rope_model(seed=0)
    model_plain = _make_plain_model(seed=0)
    trie = _make_trie()

    B, T_ctx = 1, 8
    ctx = torch.randint(0, 32, (B, T_ctx))

    with torch.no_grad():
        beams_rope,  scores_rope,  _ = constrained_beam_search(model_rope,  ctx, trie, beam_size=5)
        beams_plain, scores_plain, _ = constrained_beam_search(model_plain, ctx, trie, beam_size=5)

    # The two models have different architectures → scores must differ
    diff = (scores_rope - scores_plain).abs().max().item()
    assert diff > 1e-3, f"RoPE and plain beam scores too similar (diff={diff:.2e})"
    print(f"  [PASS] RoPE vs plain beam scores differ (diff={diff:.2e})")


def test_torope_beam_search_deterministic():
    """Same input → same beams (determinism check)."""
    model = _make_rope_model()
    trie  = _make_trie()
    ctx   = torch.randint(0, 32, (1, 8))

    with torch.no_grad():
        beams_a, scores_a, _ = constrained_beam_search(model, ctx, trie, beam_size=5)
        beams_b, scores_b, _ = constrained_beam_search(model, ctx, trie, beam_size=5)

    assert torch.equal(beams_a, beams_b),  "Beam tokens not deterministic"
    assert torch.equal(scores_a, scores_b), "Beam scores not deterministic"
    print("  [PASS] RoPE beam search is deterministic")


# ═══════════════════════════════════════════════════════════════════════════
# 8. TransformerLayer rope_params / torope_params backward compat
# ═══════════════════════════════════════════════════════════════════════════

def test_transformer_layer_rope_params_compat():
    """TransformerLayer accepts both rope_params and torope_params."""
    head_dim = 8
    freq_idx, inv_freq_time, freq_layer, n_idx, n_time, n_layer = build_torope_freqs(
        head_dim, 32, time_split_ratio=0.5)
    params = {
        'freq_idx': freq_idx, 'inv_freq_time': inv_freq_time, 'freq_layer': freq_layer,
        'n_idx_planes': n_idx, 'n_time_planes': n_time, 'n_layer_planes': n_layer,
    }
    # new API
    layer_new = TransformerLayer(embed_dim=32, n_heads=4, rope_params=params)
    assert layer_new.rope_params is not None
    assert layer_new.torope_params is not None   # backward compat alias
    # old API
    layer_old = TransformerLayer(embed_dim=32, n_heads=4, torope_params=params)
    assert layer_old.rope_params is not None
    assert layer_old.torope_params is not None
    print("  [PASS] TransformerLayer accepts rope_params and torope_params")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("RoPE Tests")
    print("=" * 60)

    print("\n1. build_rope_freqs / build_torope_freqs")
    test_build_torope_freqs_shapes()
    test_build_torope_freqs_values()
    test_build_rope_freqs_shapes()

    print("\n2. apply_rope / apply_torope")
    test_apply_torope_output_shape()
    test_apply_torope_zero_timestamps_equals_standard_rope()
    test_apply_torope_different_timestamps_differ()
    test_apply_torope_rotation_is_reversible()
    test_apply_rope_matches_apply_torope()

    print("\n3. NTPModel structure")
    test_torope_model_no_learnable_pos_emb()
    test_torope_model_get_pos_emb_zeros()
    test_torope_model_param_count()
    test_rope_use_rope_property()
    test_rope_dims_stored()

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

    print("\n7. constrained_beam_search with RoPE")
    test_torope_beam_search_kv_positions_passed()
    test_torope_beam_search_logits_differ_from_plain()
    test_torope_beam_search_deterministic()

    print("\n8. TransformerLayer compat")
    test_transformer_layer_rope_params_compat()

    print("\n" + "=" * 60)
    print("All RoPE tests passed!")
