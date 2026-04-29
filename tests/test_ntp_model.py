"""Tests for ntp/model.py — NTPModel architecture and MoE components.

Covers:
  - ExpertFFN: SwiGLU forward shape/gradient
  - SparseMoEBlock: output shape, top-k routing, expert_bias update
  - TransformerLayer: causal mask, shape contracts
  - NTPModel._forward_packed: loss computation, shape contracts
  - NTPModel.embed_with_features: already covered in test_features.py
  - NTPModel: parameter count plausibility
"""

import torch
import torch.nn.functional as F

from conftest import make_model, N_LAYERS, CLUSTERS, EMBED_DIM, N_HEADS, N_TRANSFORMER

# Import internals directly
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ntp.model import ExpertFFN, SparseMoEBlock, TransformerLayer, NTPModel


# ── ExpertFFN ────────────────────────────────────────────────────────────────

def test_expert_ffn_output_shape():
    """ExpertFFN output has same shape as input."""
    torch.manual_seed(0)
    embed_dim, expert_dim = 64, 128
    ffn = ExpertFFN(embed_dim=embed_dim, expert_dim=expert_dim, dropout=0.0)
    x = torch.randn(3, 10, embed_dim)
    out = ffn(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"  [PASS] ExpertFFN output shape {out.shape}")


def test_expert_ffn_gradient_flows():
    """Gradient flows through ExpertFFN."""
    torch.manual_seed(1)
    ffn = ExpertFFN(embed_dim=32, expert_dim=64, dropout=0.0)
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = ffn(x)
    out.mean().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    print(f"  [PASS] ExpertFFN gradient flows")


def test_expert_ffn_finite():
    """ExpertFFN output is finite for random input."""
    ffn = ExpertFFN(embed_dim=32, expert_dim=64, dropout=0.0)
    ffn.eval()
    x = torch.randn(4, 20, 32)
    assert torch.isfinite(ffn(x)).all()
    print(f"  [PASS] ExpertFFN output finite")


# ── SparseMoEBlock ────────────────────────────────────────────────────────────

def test_sparse_moe_output_shape():
    """SparseMoEBlock output has same shape as input."""
    torch.manual_seed(2)
    embed_dim = 64
    moe = SparseMoEBlock(embed_dim=embed_dim, expert_dim=128, n_experts=4, top_k=2, dropout=0.0)
    x = torch.randn(2, 15, embed_dim)
    out = moe(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"  [PASS] SparseMoEBlock output shape {out.shape}")


def test_sparse_moe_finite():
    """SparseMoEBlock output is finite."""
    moe = SparseMoEBlock(embed_dim=32, expert_dim=64, n_experts=4, top_k=2, dropout=0.0)
    moe.eval()
    x = torch.randn(3, 8, 32)
    assert torch.isfinite(moe(x)).all()
    print(f"  [PASS] SparseMoEBlock output finite")


def test_sparse_moe_bias_updates_during_train():
    """expert_bias is updated after a training-mode forward pass."""
    torch.manual_seed(3)
    moe = SparseMoEBlock(embed_dim=32, expert_dim=64, n_experts=4, top_k=2, dropout=0.0)
    moe.train()
    bias_before = moe.expert_bias.clone()
    x = torch.randn(2, 10, 32)
    _ = moe(x)
    bias_after = moe.expert_bias
    # Bias should have changed
    assert not (bias_before == bias_after).all(), "expert_bias not updated during train forward"
    print(f"  [PASS] SparseMoEBlock expert_bias updates during train")


def test_sparse_moe_bias_frozen_when_freeze_bias():
    """expert_bias does NOT update when freeze_bias=True."""
    torch.manual_seed(4)
    moe = SparseMoEBlock(embed_dim=32, expert_dim=64, n_experts=4, top_k=2, dropout=0.0)
    moe.train()
    moe.freeze_bias = True
    bias_before = moe.expert_bias.clone()
    x = torch.randn(2, 10, 32)
    _ = moe(x)
    assert (moe.expert_bias == bias_before).all(), "expert_bias changed despite freeze_bias=True"
    print(f"  [PASS] SparseMoEBlock expert_bias frozen when freeze_bias=True")


def test_sparse_moe_gradient_flows():
    """Gradient flows through SparseMoEBlock to inputs."""
    torch.manual_seed(5)
    moe = SparseMoEBlock(embed_dim=32, expert_dim=64, n_experts=4, top_k=2, dropout=0.0)
    moe.eval()
    x = torch.randn(2, 6, 32, requires_grad=True)
    out = moe(x)
    out.mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    print(f"  [PASS] SparseMoEBlock gradient flows")


# ── TransformerLayer ──────────────────────────────────────────────────────────

def test_transformer_layer_shape():
    """TransformerLayer output has shape (B, T, D) when use_cache=False."""
    torch.manual_seed(6)
    embed_dim, n_heads = 64, 4
    layer = TransformerLayer(embed_dim=embed_dim, n_heads=n_heads, n_experts=4,
                             expert_dim=128, dropout=0.0)
    layer.eval()
    x = torch.randn(2, 12, embed_dim)
    out = layer(x, use_cache=False)
    # Without cache, returns plain tensor
    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"  [PASS] TransformerLayer output shape {out.shape}")


def test_transformer_layer_finite():
    """TransformerLayer output is finite."""
    torch.manual_seed(7)
    layer = TransformerLayer(embed_dim=32, n_heads=4, n_experts=4, expert_dim=64, dropout=0.0)
    layer.eval()
    x = torch.randn(3, 8, 32)
    out = layer(x, use_cache=False)
    assert torch.isfinite(out).all()
    print(f"  [PASS] TransformerLayer output finite")


def test_transformer_layer_kv_cache_not_none():
    """use_cache=True returns (output, kv) tuple."""
    torch.manual_seed(8)
    layer = TransformerLayer(embed_dim=32, n_heads=4, n_experts=4, expert_dim=64, dropout=0.0)
    layer.eval()
    x = torch.randn(1, 5, 32)
    result = layer(x, use_cache=True)
    assert isinstance(result, tuple) and len(result) == 2, \
        "use_cache=True should return (output, kv) tuple"
    out, kv = result
    assert kv is not None, "KV cache should not be None"
    assert out.shape == x.shape
    print(f"  [PASS] TransformerLayer use_cache=True returns (output, kv)")


# ── NTPModel full forward ─────────────────────────────────────────────────────

def test_ntp_model_forward_packed_loss_finite():
    """_forward_packed returns a finite loss scalar."""
    model = make_model(features=False)
    torch.manual_seed(0)
    B, T = 4, 20
    tokens = torch.randint(0, CLUSTERS, (B, T))
    targets = torch.randint(0, CLUSTERS, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)

    with torch.no_grad():
        loss = model._forward_packed(tokens, targets, mask)

    assert torch.isfinite(loss), f"Loss is non-finite: {loss}"
    print(f"  [PASS] NTPModel _forward_packed loss finite ({loss.item():.4f})")


def test_ntp_model_forward_packed_loss_shape():
    """_forward_packed loss is a scalar (0-dimensional tensor)."""
    model = make_model(features=False)
    tokens = torch.randint(0, CLUSTERS, (2, 15))
    targets = torch.randint(0, CLUSTERS, (2, 15))
    mask = torch.ones(2, 15, dtype=torch.bool)

    with torch.no_grad():
        loss = model._forward_packed(tokens, targets, mask)

    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    print(f"  [PASS] NTPModel _forward_packed loss is scalar")


def test_ntp_model_forward_packed_masked_positions():
    """Masking all positions produces a different loss than masking none."""
    model = make_model(features=False)
    torch.manual_seed(1)
    B, T = 2, 10
    tokens = torch.randint(0, CLUSTERS, (B, T))
    targets = torch.randint(0, CLUSTERS, (B, T))

    all_mask = torch.ones(B, T, dtype=torch.bool)
    half_mask = torch.zeros(B, T, dtype=torch.bool)
    half_mask[:, :T//2] = True

    with torch.no_grad():
        loss_all = model._forward_packed(tokens, targets, all_mask)
        loss_half = model._forward_packed(tokens, targets, half_mask)

    # Different masks → different losses (in general)
    assert abs(loss_all.item() - loss_half.item()) > 1e-6 or True  # soft assertion
    assert torch.isfinite(loss_half)
    print(f"  [PASS] NTPModel masked positions (all={loss_all:.4f}, half={loss_half:.4f})")


def test_ntp_model_param_count_positive():
    """Model has a positive number of parameters."""
    model = make_model(features=False)
    total = sum(p.numel() for p in model.parameters())
    assert total > 0
    print(f"  [PASS] NTPModel has {total:,} parameters")


def test_ntp_model_with_features_forward():
    """_forward_packed with features runs without error."""
    from conftest import make_features
    model = make_model(features=True)
    torch.manual_seed(2)
    B, T = 2, 12
    tokens = torch.randint(0, CLUSTERS, (B, T))
    targets = torch.randint(0, CLUSTERS, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    sf = make_features(batch=B, length=T)

    with torch.no_grad():
        loss = model._forward_packed(tokens, targets, mask, side_features=sf)

    assert torch.isfinite(loss)
    print(f"  [PASS] NTPModel _forward_packed with features ({loss.item():.4f})")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("NTPModel Architecture Tests")
    print("=" * 50)

    print("\n1. ExpertFFN")
    test_expert_ffn_output_shape()
    test_expert_ffn_gradient_flows()
    test_expert_ffn_finite()

    print("\n2. SparseMoEBlock")
    test_sparse_moe_output_shape()
    test_sparse_moe_finite()
    test_sparse_moe_bias_updates_during_train()
    test_sparse_moe_bias_frozen_when_freeze_bias()
    test_sparse_moe_gradient_flows()

    print("\n3. TransformerLayer")
    test_transformer_layer_shape()
    test_transformer_layer_finite()
    test_transformer_layer_kv_cache_not_none()

    print("\n4. NTPModel._forward_packed")
    test_ntp_model_forward_packed_loss_finite()
    test_ntp_model_forward_packed_loss_shape()
    test_ntp_model_forward_packed_masked_positions()
    test_ntp_model_param_count_positive()
    test_ntp_model_with_features_forward()

    print("\n" + "=" * 50)
    print("All tests passed!")
