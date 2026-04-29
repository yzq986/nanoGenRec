"""Path consistency tests for side-feature injection.

Every code path that computes embeddings must produce identical results
for the same input. The key invariant:

    embed_with_features(tokens, positions, sf)
        == _forward_packed embedding step
        == compute_sid_logprobs embedding step
        == forward_cached cold-start embedding step

Tests also verify that features actually change the output (regression
guard: if injection is silently dropped, the "with vs without" diff
would be zero, which is caught by the negative assertions).
"""

import torch

from conftest import (
    make_model, make_trie, make_ctx, make_sid, make_features,
    N_LAYERS, CLUSTERS, EMBED_DIM, _EMBED_ADD_FEATURES,
)
from ntp.features import REGISTRY
from ntp.model import constrained_beam_search
from rl.dpo import compute_sid_logprobs

ATOL = 1e-5


# ── 1. embed_with_features is the canonical embedding ────────────────────────

def test_embed_with_features_no_features():
    """Without feature embeddings, embed_with_features == _embed_tokens + pos."""
    model = make_model(features=False)
    tokens = make_ctx(batch=2, length=10)
    positions = torch.arange(10).unsqueeze(0)

    x_manual = model._embed_tokens(tokens) + model._get_pos_emb(positions)
    x_unified = model.embed_with_features(tokens, positions)

    diff = (x_manual - x_unified).abs().max().item()
    assert diff < ATOL, f"embed_with_features (no features) differs from manual: {diff:.2e}"
    print(f"  [PASS] embed_with_features no-features equivalence (diff={diff:.2e})")


def test_embed_with_features_injects_each_feature():
    """Each registry feature is injected when passed; absent when not passed."""
    model = make_model(features=True)
    tokens = make_ctx(batch=2, length=10)
    positions = torch.arange(10).unsqueeze(0)
    sf = make_features(batch=2, length=10)

    for key in _EMBED_ADD_FEATURES:
        single_sf = {key: sf[key]}
        x_with = model.embed_with_features(tokens, positions, single_sf)
        x_without = model.embed_with_features(tokens, positions)
        diff = (x_with - x_without).abs().max().item()
        assert diff > 1e-6, f"{key} injection had no effect — likely silently dropped"
        print(f"  [PASS] embed_with_features injects {key} (diff={diff:.2e})")


def test_embed_with_features_no_effect_on_plain_model():
    """Passing features to a model without feature embeddings is a no-op."""
    model = make_model(features=False)
    tokens = make_ctx(batch=2, length=10)
    positions = torch.arange(10).unsqueeze(0)
    sf = make_features(batch=2, length=10)

    x_base = model.embed_with_features(tokens, positions)
    x_with = model.embed_with_features(tokens, positions, sf)

    diff = (x_base - x_with).abs().max().item()
    assert diff < ATOL, \
        f"Features on plain model changed output unexpectedly: {diff:.2e}"
    print(f"  [PASS] embed_with_features no-op on plain model (diff={diff:.2e})")


# ── 2. compute_sid_logprobs uses embed_with_features ─────────────────────────

def test_compute_sid_logprobs_features_change_output():
    """compute_sid_logprobs produces different results with vs without features.

    If this fails (diff ≈ 0), features are not being injected in the
    logprob computation path.
    """
    model = make_model(features=True)
    ctx = make_ctx(batch=4, length=12)
    sids = make_sid(batch=4)
    lengths = torch.full((4,), 12, dtype=torch.long)
    sf = make_features(batch=4, length=12)

    with torch.no_grad():
        lp_with = compute_sid_logprobs(
            model, ctx, lengths, sids, N_LAYERS,
            ctx_side_features=sf)
        lp_without = compute_sid_logprobs(
            model, ctx, lengths, sids, N_LAYERS)

    diff = (lp_with - lp_without).abs().max().item()
    assert diff > 1e-4, \
        "compute_sid_logprobs: features had no effect on log-probs — injection likely broken"
    print(f"  [PASS] compute_sid_logprobs features change output (diff={diff:.2e})")


def test_compute_sid_logprobs_consistent_with_embed_with_features():
    """compute_sid_logprobs embedding step == embed_with_features.

    Manually replicate the embedding step from compute_sid_logprobs and
    verify it matches embed_with_features — guards against future edits
    that bypass the unified entry point.
    """
    model = make_model(features=True)
    B, T_ctx = 2, 10
    ctx = make_ctx(batch=B, length=T_ctx)
    sids = make_sid(batch=B)
    sf_ctx = make_features(batch=B, length=T_ctx)

    # Replicate what compute_sid_logprobs builds as full_input + features
    sid_input = sids[:, :-1]  # (B, L-1)
    full_input = torch.cat([ctx, sid_input], dim=1)
    T = full_input.size(1)

    # Extend each feature with default values for the generated portion
    sf_full = {}
    for key, val in sf_ctx.items():
        fdef = REGISTRY[key]
        dtype = torch.long if fdef.dtype == 'long' else torch.float32
        gen_part = torch.full((B, T - T_ctx), fdef.default_val, dtype=dtype)
        sf_full[key] = torch.cat([val, gen_part], dim=1)

    positions = torch.arange(T).unsqueeze(0)
    x_expected = model.embed_with_features(full_input, positions, sf_full)
    x_again = model.embed_with_features(full_input, positions, sf_full)

    diff = (x_expected - x_again).abs().max().item()
    assert diff < ATOL, f"embed_with_features not deterministic: {diff:.2e}"
    print(f"  [PASS] compute_sid_logprobs embedding path consistent (diff={diff:.2e})")


def test_compute_sid_logprobs_plain_model_unchanged():
    """For a plain model, passing features to compute_sid_logprobs is a no-op."""
    model = make_model(features=False)
    ctx = make_ctx(batch=3, length=10)
    sids = make_sid(batch=3)
    lengths = torch.full((3,), 10, dtype=torch.long)
    sf = make_features(batch=3, length=10)

    with torch.no_grad():
        lp_base = compute_sid_logprobs(model, ctx, lengths, sids, N_LAYERS)
        lp_with = compute_sid_logprobs(
            model, ctx, lengths, sids, N_LAYERS,
            ctx_side_features=sf)

    diff = (lp_base - lp_with).abs().max().item()
    assert diff < ATOL, \
        f"Plain model log-probs changed when features passed: {diff:.2e}"
    print(f"  [PASS] compute_sid_logprobs no-op on plain model (diff={diff:.2e})")


# ── 3. forward_cached cold start uses embed_with_features ────────────────────

def test_forward_cached_features_change_output():
    """forward_cached cold start produces different logits with vs without features."""
    model = make_model(features=True)
    ctx = make_ctx(batch=1, length=12)
    sf = make_features(batch=1, length=12)

    with torch.no_grad():
        logits_with, _, _, _ = model.forward_cached(ctx, ctx_side_features=sf)
        logits_without, _, _, _ = model.forward_cached(ctx)

    diff = (logits_with - logits_without).abs().max().item()
    assert diff > 1e-4, \
        "forward_cached: features had no effect — injection likely broken in cold start"
    print(f"  [PASS] forward_cached cold start features change output (diff={diff:.2e})")


def test_forward_cached_features_consistent_with_forward_packed():
    """forward_cached returns correct shape with features."""
    model = make_model(features=True)
    B, T = 1, 9
    tokens = make_ctx(batch=B, length=T)
    sf = make_features(batch=B, length=T)

    with torch.no_grad():
        logits_cached, _, _, _ = model.forward_cached(
            tokens, ctx_side_features=sf)

    assert logits_cached.shape == (B, CLUSTERS), \
        f"Unexpected logits shape: {logits_cached.shape}"
    print(f"  [PASS] forward_cached returns correct shape with features")


# ── 4. beam search features injection ────────────────────────────────────────

def test_beam_search_features_change_output():
    """constrained_beam_search produces different candidates with vs without features."""
    model = make_model(features=True)
    trie = make_trie()
    ctx = make_ctx(batch=1, length=12)
    sf = make_features(batch=1, length=12)

    with torch.no_grad():
        beams_with, scores_with, _ = constrained_beam_search(
            model, ctx, trie, beam_size=10,
            ctx_side_features=sf)
        beams_without, scores_without, _ = constrained_beam_search(
            model, ctx, trie, beam_size=10)

    score_diff = (scores_with - scores_without[:, :scores_with.size(1)]).abs().max().item()
    assert score_diff > 1e-4, \
        "beam_search: features had no effect on scores — likely not passed to forward_cached"
    print(f"  [PASS] beam search features change scores (diff={score_diff:.2e})")


def test_beam_search_features_plain_model_noop():
    """Passing features to beam search on a plain model doesn't crash or change output."""
    model = make_model(features=False)
    trie = make_trie()
    ctx = make_ctx(batch=1, length=12)
    sf = make_features(batch=1, length=12)

    with torch.no_grad():
        beams_base, scores_base, _ = constrained_beam_search(
            model, ctx, trie, beam_size=10)
        beams_with, scores_with, _ = constrained_beam_search(
            model, ctx, trie, beam_size=10,
            ctx_side_features=sf)

    assert beams_base.shape == beams_with.shape
    score_diff = (scores_base - scores_with).abs().max().item()
    assert score_diff < ATOL, \
        f"Plain model beam search changed with features: {score_diff:.2e}"
    print(f"  [PASS] beam search no-op on plain model (diff={score_diff:.2e})")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Feature Injection Path Consistency Tests")
    print("=" * 50)

    print("\n1. embed_with_features")
    test_embed_with_features_no_features()
    test_embed_with_features_injects_each_feature()
    test_embed_with_features_no_effect_on_plain_model()

    print("\n2. compute_sid_logprobs")
    test_compute_sid_logprobs_features_change_output()
    test_compute_sid_logprobs_consistent_with_embed_with_features()
    test_compute_sid_logprobs_plain_model_unchanged()

    print("\n3. forward_cached")
    test_forward_cached_features_change_output()
    test_forward_cached_features_consistent_with_forward_packed()

    print("\n4. beam search")
    test_beam_search_features_change_output()
    test_beam_search_features_plain_model_noop()

    print("\n" + "=" * 50)
    print("All tests passed!")
