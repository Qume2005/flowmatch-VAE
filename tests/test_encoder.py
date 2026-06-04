"""Tests for MultiScaleConvEncoder (conv_encoder.py).

Covers:
- Output shapes (mu, logvar, per_scale_tokens)
- Gradient flow
- Per-scale token shape correctness
- FusionAttention module
- AttnPool2x2 module (on CUDA)
"""
import pytest
import torch
from flowmatch_vae.config import Config, MultiScaleEncoderConfig
from flowmatch_vae.models.conv_encoder import (
    AttnPool2x2,
    FusionAttention,
    MultiScaleConvEncoder,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Encoder tests require CUDA"
)


def _small_encoder_cfg():
    """Minimal encoder config for fast tests."""
    return MultiScaleEncoderConfig(
        num_conv_blocks=1,
        dilations=(1,),
    )


# ---------------------------------------------------------------------------
# test_encoder_output_shape
# ---------------------------------------------------------------------------


def test_encoder_output_shape():
    """mu and logvar are (B, 8, 8, 256); per_scale_tokens is a dict with keys 0-5."""
    cfg = _small_encoder_cfg()
    device = torch.device("cuda")
    encoder = MultiScaleConvEncoder(cfg).to(device)
    x = torch.randn(2, 3, 64, 64, device=device)

    mu, logvar, per_scale_tokens = encoder(x)

    assert mu.shape == (2, 8, 8, 256), f"Expected mu (2,8,8,256), got {mu.shape}"
    assert logvar.shape == (2, 8, 8, 256), f"Expected logvar (2,8,8,256), got {logvar.shape}"
    assert isinstance(per_scale_tokens, dict), "per_scale_tokens should be a dict"
    assert set(per_scale_tokens.keys()) == {0, 1, 2, 3, 4, 5}, (
        f"Expected keys {{0..5}}, got {set(per_scale_tokens.keys())}"
    )


# ---------------------------------------------------------------------------
# test_encoder_gradient_flows
# ---------------------------------------------------------------------------


def test_encoder_gradient_flows():
    """Gradients flow through all encoder parameters."""
    cfg = _small_encoder_cfg()
    device = torch.device("cuda")
    encoder = MultiScaleConvEncoder(cfg).to(device)
    x = torch.randn(2, 3, 64, 64, device=device)

    mu, logvar, per_scale_tokens = encoder(x)
    loss = mu.sum() + logvar.sum()
    loss.backward()

    for name, p in encoder.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# test_encoder_scale_tokens_shapes
# ---------------------------------------------------------------------------


def test_encoder_scale_tokens_shapes():
    """Each scale token tensor has the correct number of spatial elements.

    6 stages of AttnPool2x2 produce spatial sizes:
        stage 0: 32x32 = 1024
        stage 1: 16x16 = 256
        stage 2:  8x8  = 64   <-- latent_scale_idx
        stage 3:  4x4  = 16
        stage 4:  2x2  = 4
        stage 5:  1x1  = 1
    """
    cfg = _small_encoder_cfg()
    device = torch.device("cuda")
    encoder = MultiScaleConvEncoder(cfg).to(device)
    x = torch.randn(2, 3, 64, 64, device=device)

    _, _, per_scale_tokens = encoder(x)

    expected = {
        0: 1024,  # 32x32
        1: 256,   # 16x16
        2: 64,    # 8x8
        3: 16,    # 4x4
        4: 4,     # 2x2
        5: 1,     # 1x1
    }
    for scale_idx, n_tokens in expected.items():
        tok = per_scale_tokens[scale_idx]
        assert tok.shape == (2, n_tokens, cfg.embed_dim), (
            f"Scale {scale_idx}: expected (2,{n_tokens},{cfg.embed_dim}), got {tok.shape}"
        )


# ---------------------------------------------------------------------------
# test_fusion_attention
# ---------------------------------------------------------------------------


def test_fusion_attention():
    """FusionAttention produces correct output shape with 2D PoPE."""
    device = torch.device("cuda")
    dim = 64
    num_heads = 4
    B, N = 2, 32

    fusion = FusionAttention(dim, num_heads=num_heads, max_h=8, max_w=8).to(device)
    x = torch.randn(B, N, dim, device=device)
    # Use a spatial grid that sums to N=32 tokens: 4x8
    spatial_sizes = [(4, 8)]

    out = fusion(x, spatial_sizes)

    assert out.shape == (B, N, dim), f"Expected ({B},{N},{dim}), got {out.shape}"


def test_fusion_attention_gradient():
    """Gradients flow through FusionAttention."""
    device = torch.device("cuda")
    dim = 64
    B, N = 1, 16

    fusion = FusionAttention(dim, num_heads=4, max_h=4, max_w=4).to(device)
    x = torch.randn(B, N, dim, device=device, requires_grad=True)
    spatial_sizes = [(4, 4)]

    out = fusion(x, spatial_sizes)
    out.sum().backward()

    assert x.grad is not None, "Input gradient should not be None"
    for name, p in fusion.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# test_attn_pool_2x2  (on CUDA — complements CPU version in test_vae.py)
# ---------------------------------------------------------------------------


def test_attn_pool_2x2_cuda():
    """AttnPool2x2 halves spatial dims on CUDA and gradients flow."""
    device = torch.device("cuda")
    pool = AttnPool2x2(64).to(device)
    x = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)

    out = pool(x)

    assert out.shape == (2, 64, 8, 8), f"Expected (2,64,8,8), got {out.shape}"
    out.sum().backward()
    assert x.grad is not None, "Gradient should flow through AttnPool2x2"


# ---------------------------------------------------------------------------
# test_encoder_adaptive_sizing — encoder adapts to different input sizes
# ---------------------------------------------------------------------------


def test_encoder_adaptive_sizing():
    """Encoder adapts to 32x32 input with latent_spatial_size=4."""
    cfg = MultiScaleEncoderConfig(
        num_conv_blocks=1,
        dilations=(1,),
        latent_spatial_size=4,
        max_input_size=32,
    )
    device = torch.device("cuda")
    encoder = MultiScaleConvEncoder(cfg).to(device)
    x = torch.randn(2, 3, 32, 32, device=device)

    mu, logvar, per_scale_tokens = encoder(x)

    # 32x32 -> 5 stages: 16x16, 8x8, 4x4, 2x2, 1x1
    assert mu.shape == (2, 4, 4, 256), f"Expected mu (2,4,4,256), got {mu.shape}"
    assert logvar.shape == (2, 4, 4, 256), f"Expected logvar (2,4,4,256), got {logvar.shape}"
    assert set(per_scale_tokens.keys()) == {0, 1, 2, 3, 4}, (
        f"Expected 5 scales for 32x32 input, got {set(per_scale_tokens.keys())}"
    )

    # Verify spatial sizes
    expected_tokens = {
        0: 256,  # 16x16
        1: 64,   # 8x8
        2: 16,   # 4x4  <- latent
        3: 4,    # 2x2
        4: 1,    # 1x1
    }
    for scale_idx, n_tokens in expected_tokens.items():
        tok = per_scale_tokens[scale_idx]
        assert tok.shape == (2, n_tokens, cfg.embed_dim), (
            f"Scale {scale_idx}: expected (2,{n_tokens},{cfg.embed_dim}), got {tok.shape}"
        )
