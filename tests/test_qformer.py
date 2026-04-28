"""Tests for model/qformer.py — QFormer cross-attention architecture.

Covers:
  - QFormerLayer: output shape, cross-attention masking
  - QFormer: forward shape, encoder mask, determinism, param_count
  - QFormer: learnable query tokens are updated by gradients
"""

import torch
import torch.nn as nn

from model.qformer import QFormer, QFormerLayer


# ── QFormerLayer ──────────────────────────────────────────────────────────────

def test_qformer_layer_output_shape():
    """QFormerLayer output has the same shape as input queries."""
    torch.manual_seed(0)
    B, M, S = 2, 4, 16
    d_model, n_heads, d_enc = 64, 4, 128

    layer = QFormerLayer(d_model=d_model, n_heads=n_heads, d_enc=d_enc, dropout=0.0)
    queries = torch.randn(B, M, d_model)
    enc_hidden = torch.randn(B, S, d_enc)

    out = layer(queries, enc_hidden)
    assert out.shape == (B, M, d_model), f"Expected {(B, M, d_model)}, got {out.shape}"
    print(f"  [PASS] QFormerLayer output shape {out.shape}")


def test_qformer_layer_with_mask():
    """Encoder mask is applied without error; masked positions are ignored."""
    torch.manual_seed(1)
    B, M, S = 2, 3, 10
    d_model, n_heads, d_enc = 32, 4, 64

    layer = QFormerLayer(d_model=d_model, n_heads=n_heads, d_enc=d_enc, dropout=0.0)
    layer.eval()

    queries = torch.randn(B, M, d_model)
    enc_hidden = torch.randn(B, S, d_enc)
    # Mask: first 5 tokens valid, last 5 masked
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[:, :5] = True

    out_masked = layer(queries, enc_hidden, encoder_mask=mask)
    assert out_masked.shape == (B, M, d_model)
    assert torch.isfinite(out_masked).all()
    print(f"  [PASS] QFormerLayer with encoder mask (finite output)")


def test_qformer_layer_finite_output():
    """Output is finite for various inputs."""
    torch.manual_seed(2)
    layer = QFormerLayer(d_model=64, n_heads=4, d_enc=128, dropout=0.0)
    layer.eval()
    queries = torch.randn(3, 4, 64)
    enc = torch.randn(3, 20, 128)
    out = layer(queries, enc)
    assert torch.isfinite(out).all(), f"Non-finite QFormerLayer output"
    print(f"  [PASS] QFormerLayer output all finite")


# ── QFormer ───────────────────────────────────────────────────────────────────

def test_qformer_forward_shape():
    """QFormer.forward() returns (B, d_model) after mean-pooling queries."""
    torch.manual_seed(0)
    B, S = 4, 32
    d_enc, d_model = 128, 64
    num_queries, num_layers = 6, 2

    model = QFormer(
        num_queries=num_queries, num_layers=num_layers,
        d_model=d_model, d_enc=d_enc, n_heads=4, dropout=0.0
    )
    model.eval()

    enc_hidden = torch.randn(B, S, d_enc)
    out = model(enc_hidden)
    assert out.shape == (B, d_model), f"Expected {(B, d_model)}, got {out.shape}"
    print(f"  [PASS] QFormer output shape (B={B}, d_model={d_model})")


def test_qformer_forward_finite():
    """QFormer output is finite for random input."""
    torch.manual_seed(1)
    model = QFormer(num_queries=4, num_layers=2, d_model=64, d_enc=128, n_heads=4, dropout=0.0)
    model.eval()
    enc = torch.randn(2, 16, 128)
    out = model(enc)
    assert torch.isfinite(out).all()
    print(f"  [PASS] QFormer output all finite")


def test_qformer_with_mask():
    """QFormer accepts encoder_mask and produces valid output."""
    torch.manual_seed(2)
    B, S = 3, 20
    model = QFormer(num_queries=4, num_layers=2, d_model=32, d_enc=64, n_heads=4, dropout=0.0)
    model.eval()

    enc = torch.randn(B, S, 64)
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[:, -5:] = False  # last 5 tokens masked

    out = model(enc, encoder_mask=mask)
    assert out.shape == (B, 32)
    assert torch.isfinite(out).all()
    print(f"  [PASS] QFormer with encoder mask produces finite output")


def test_qformer_deterministic():
    """Same input produces same output (eval mode, no dropout)."""
    torch.manual_seed(3)
    model = QFormer(num_queries=4, num_layers=2, d_model=64, d_enc=128, n_heads=4, dropout=0.0)
    model.eval()

    enc = torch.randn(2, 16, 128)
    out_a = model(enc)
    out_b = model(enc)
    assert (out_a == out_b).all(), "QFormer not deterministic in eval mode"
    print(f"  [PASS] QFormer deterministic in eval mode")


def test_qformer_query_tokens_trainable():
    """Gradient flows back to the learnable query_tokens parameter."""
    torch.manual_seed(4)
    model = QFormer(num_queries=4, num_layers=1, d_model=32, d_enc=64, n_heads=4, dropout=0.0)

    enc = torch.randn(2, 8, 64)
    out = model(enc)
    loss = out.mean()
    loss.backward()

    assert model.query_tokens.grad is not None
    assert model.query_tokens.grad.abs().sum() > 0, "query_tokens gradient is all zero"
    print(f"  [PASS] QFormer query_tokens receive gradients")


def test_qformer_param_count_format():
    """param_count() returns a non-empty string."""
    model = QFormer(num_queries=4, num_layers=2, d_model=128, d_enc=256, n_heads=8)
    s = model.param_count()
    assert isinstance(s, str) and 'M' in s
    print(f"  [PASS] QFormer param_count: '{s}'")


def test_qformer_different_query_counts():
    """Different num_queries values produce correct output dimensions."""
    torch.manual_seed(5)
    for num_q in [1, 4, 16]:
        model = QFormer(num_queries=num_q, num_layers=1, d_model=32, d_enc=64, n_heads=4, dropout=0.0)
        model.eval()
        enc = torch.randn(2, 10, 64)
        out = model(enc)
        assert out.shape == (2, 32), f"Wrong shape for num_queries={num_q}: {out.shape}"
    print(f"  [PASS] QFormer num_queries variability (1, 4, 16)")


def test_qformer_different_batch_sizes():
    """Model handles batch size 1 and batch size > 1."""
    torch.manual_seed(6)
    model = QFormer(num_queries=4, num_layers=2, d_model=32, d_enc=64, n_heads=4, dropout=0.0)
    model.eval()
    for B in [1, 8]:
        enc = torch.randn(B, 12, 64)
        out = model(enc)
        assert out.shape == (B, 32)
    print(f"  [PASS] QFormer handles batch size 1 and 8")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("QFormer Architecture Tests")
    print("=" * 50)

    print("\n1. QFormerLayer")
    test_qformer_layer_output_shape()
    test_qformer_layer_with_mask()
    test_qformer_layer_finite_output()

    print("\n2. QFormer")
    test_qformer_forward_shape()
    test_qformer_forward_finite()
    test_qformer_with_mask()
    test_qformer_deterministic()
    test_qformer_query_tokens_trainable()
    test_qformer_param_count_format()
    test_qformer_different_query_counts()
    test_qformer_different_batch_sizes()

    print("\n" + "=" * 50)
    print("All tests passed!")
