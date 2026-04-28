"""Tests for model/fsq.py — Finite Scalar Quantization.

Covers:
  - _codebook_size: mixed-radix product
  - FSQLayer: encode/decode round-trip, PCA projection, reconstruction
  - FSQLayer: codebook coverage after quantization
  - FSQLayer: save/load state round-trip
  - LearnedFSQLayer: basic interface (shape contracts)
  - fsq_layer_from_state: deserializer
"""

import torch

from model.fsq import FSQLayer, LearnedFSQLayer, _codebook_size, fsq_layer_from_state


def _assert_raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"Expected {exc.__name__} but no exception was raised")


# ── _codebook_size ────────────────────────────────────────────────────────────

def test_codebook_size_single():
    assert _codebook_size([8]) == 8
    print(f"  [PASS] _codebook_size single level")


def test_codebook_size_product():
    assert _codebook_size([4, 8, 4]) == 128
    print(f"  [PASS] _codebook_size product 4×8×4=128")


def test_codebook_size_uniform():
    assert _codebook_size([5, 5, 5]) == 125
    print(f"  [PASS] _codebook_size 5³=125")


# ── FSQLayer helpers ──────────────────────────────────────────────────────────

def _make_fsq(levels=None, n_features=32, n_train=500, seed=0):
    if levels is None:
        levels = [4, 4, 4]
    torch.manual_seed(seed)
    layer = FSQLayer(levels=levels, n_features=n_features)
    data = torch.randn(n_train, n_features)
    layer.train(data)
    return layer


# ── FSQLayer encode/decode ────────────────────────────────────────────────────

def test_fsq_codes_in_range():
    """Codes returned by predict() are in [0, codebook_size)."""
    layer = _make_fsq(levels=[4, 4, 4])
    data = torch.randn(100, 32)
    codes = layer.predict(data)
    assert codes.shape == (100,)
    assert (codes >= 0).all()
    assert (codes < layer.codebook_size).all()
    print(f"  [PASS] FSQLayer codes in [0, {layer.codebook_size})")


def test_fsq_decode_encode_round_trip():
    """Mixed-radix encode followed by decode recovers per-dim codes."""
    layer = _make_fsq(levels=[3, 5, 4])
    torch.manual_seed(7)
    # Construct known per-dim codes
    per_dim = torch.tensor([[0, 2, 3], [2, 4, 0], [1, 1, 1]], dtype=torch.long)
    encoded = layer._encode(per_dim)
    decoded = layer._decode_index(encoded)
    assert (decoded == per_dim).all(), f"Round-trip failed:\n{per_dim}\n{decoded}"
    print(f"  [PASS] FSQLayer encode/decode round-trip")


def test_fsq_reconstruction_shape():
    """get_centroids_for_codes returns (N, D) tensor."""
    layer = _make_fsq(levels=[4, 4, 4], n_features=16)
    data = torch.randn(20, 16)
    codes = layer.predict(data)
    recon = layer.get_centroids_for_codes(codes)
    assert recon.shape == (20, 16)
    print(f"  [PASS] FSQLayer reconstruction shape")


def test_fsq_reconstruction_finite():
    """Reconstruction contains no NaN or Inf."""
    layer = _make_fsq()
    data = torch.randn(50, 32)
    codes = layer.predict(data)
    recon = layer.get_centroids_for_codes(codes)
    assert torch.isfinite(recon).all(), "Reconstruction contains NaN/Inf"
    print(f"  [PASS] FSQLayer reconstruction finite")


def test_fsq_codebook_coverage():
    """With enough data, predict() uses multiple distinct codes."""
    layer = _make_fsq(levels=[4, 4, 4])  # codebook_size=64
    data = torch.randn(500, 32)
    codes = layer.predict(data)
    n_unique = codes.unique().numel()
    assert n_unique > 10, f"Only {n_unique} distinct codes used out of 64"
    print(f"  [PASS] FSQLayer coverage: {n_unique}/64 codes used")


# ── FSQLayer: odd vs even levels ──────────────────────────────────────────────

def test_fsq_odd_levels():
    """Odd levels (e.g. [5, 5]) produce valid codes without error."""
    layer = _make_fsq(levels=[5, 5], n_features=16)
    data = torch.randn(30, 16)
    codes = layer.predict(data)
    assert (codes >= 0).all() and (codes < 25).all()
    print(f"  [PASS] FSQLayer odd levels [5,5]")


def test_fsq_even_levels():
    """Even levels (e.g. [4, 8]) produce valid codes."""
    layer = _make_fsq(levels=[4, 8], n_features=16)
    data = torch.randn(30, 16)
    codes = layer.predict(data)
    assert (codes >= 0).all() and (codes < 32).all()
    print(f"  [PASS] FSQLayer even levels [4,8]")


# ── FSQLayer serialization ────────────────────────────────────────────────────

def test_fsq_save_load_state():
    """FSQLayer.save_state() / from_state() produces identical predictions."""
    layer = _make_fsq(levels=[4, 4, 4])
    data = torch.randn(20, 32)
    codes_before = layer.predict(data)

    state = layer.save_state()
    layer2 = FSQLayer.from_state(state)
    codes_after = layer2.predict(data)

    assert (codes_before == codes_after).all(), "Codes differ after save/load"
    print(f"  [PASS] FSQLayer save/load state round-trip")


def test_fsq_layer_from_state():
    """fsq_layer_from_state() function works identically to FSQLayer.from_state()."""
    layer = _make_fsq(levels=[3, 4, 5], n_features=24)
    state = layer.save_state()
    layer2 = fsq_layer_from_state(state)
    data = torch.randn(10, 24)
    assert (layer.predict(data) == layer2.predict(data)).all()
    print(f"  [PASS] fsq_layer_from_state()")


# ── FSQLayer: requires fitted PCA ────────────────────────────────────────────

def test_fsq_predict_without_fit_raises():
    """predict() before train() raises an error (pca_components is None)."""
    layer = FSQLayer(levels=[4, 4], n_features=16)
    data = torch.randn(5, 16)
    _assert_raises(lambda: layer.predict(data))
    print(f"  [PASS] FSQLayer predict without fit raises")


# ── LearnedFSQLayer shape contracts ──────────────────────────────────────────

def test_learned_fsq_predict_shape():
    """LearnedFSQLayer.predict() returns (N,) codes after training."""
    torch.manual_seed(0)
    levels = [4, 4]
    layer = LearnedFSQLayer(
        levels=levels, n_features=16, hidden_dim=32,
        epochs=2, batch_size=64, lr=1e-3, device='cpu'
    )
    data = torch.randn(200, 16)
    layer.train(data)
    codes = layer.predict(data[:20])
    assert codes.shape == (20,)
    assert (codes >= 0).all() and (codes < 16).all()
    print(f"  [PASS] LearnedFSQLayer predict shape")


def test_learned_fsq_reconstruct_shape():
    """LearnedFSQLayer.get_centroids_for_codes() returns (N, D)."""
    torch.manual_seed(1)
    layer = LearnedFSQLayer(
        levels=[4, 4], n_features=16, hidden_dim=32,
        epochs=2, batch_size=64, device='cpu'
    )
    data = torch.randn(100, 16)
    layer.train(data)
    codes = layer.predict(data[:10])
    recon = layer.get_centroids_for_codes(codes)
    assert recon.shape == (10, 16)
    print(f"  [PASS] LearnedFSQLayer reconstruct shape")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("FSQ Quantization Tests")
    print("=" * 50)

    print("\n1. _codebook_size")
    test_codebook_size_single()
    test_codebook_size_product()
    test_codebook_size_uniform()

    print("\n2. FSQLayer encode/decode")
    test_fsq_codes_in_range()
    test_fsq_decode_encode_round_trip()
    test_fsq_reconstruction_shape()
    test_fsq_reconstruction_finite()
    test_fsq_codebook_coverage()

    print("\n3. FSQLayer odd/even levels")
    test_fsq_odd_levels()
    test_fsq_even_levels()

    print("\n4. FSQLayer serialization")
    test_fsq_save_load_state()
    test_fsq_layer_from_state()
    test_fsq_predict_without_fit_raises()

    print("\n5. LearnedFSQLayer")
    test_learned_fsq_predict_shape()
    test_learned_fsq_reconstruct_shape()

    print("\n" + "=" * 50)
    print("All tests passed!")
