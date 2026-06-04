import torch
import torch.nn.functional as F
from flowmatch_vae.config import Config
from flowmatch_vae.models.vae import FlowMatchVAE
from flowmatch_vae.models.conv_encoder import SwiGLUConv, RMSNorm2d, AttnPool2x2


def test_rmsnorm2d():
    norm = RMSNorm2d(64)
    x = torch.randn(2, 64, 16, 16)
    out = norm(x)
    assert out.shape == x.shape
    # Check that output is normalized (RMS ≈ 1 per channel)
    rms = out.pow(2).mean(dim=1, keepdim=True).sqrt()
    assert (rms - 1.0).abs().max() < 0.1


def test_swiglu_conv_same_size():
    """SwiGLUConv preserves spatial dimensions with padding."""
    conv = SwiGLUConv(64, 64, kernel_size=3, dilation=2)
    x = torch.randn(2, 64, 16, 16)
    out = conv(x)
    assert out.shape == (2, 64, 16, 16)


def test_swiglu_conv_channel_change():
    """SwiGLUConv can change channel count."""
    conv = SwiGLUConv(3, 128, kernel_size=3, dilation=1)
    x = torch.randn(2, 3, 64, 64)
    out = conv(x)
    assert out.shape == (2, 128, 64, 64)


def test_swiglu_conv_pointwise():
    """SwiGLUConv with kernel_size=1 works as pointwise."""
    conv = SwiGLUConv(64, 64, kernel_size=1, dilation=1)
    x = torch.randn(2, 64, 4, 4)
    out = conv(x)
    assert out.shape == (2, 64, 4, 4)


def test_attn_pool_halves_spatial():
    """AttnPool2x2 halves H and W."""
    pool = AttnPool2x2(64)
    x = torch.randn(2, 64, 16, 16)
    out = pool(x)
    assert out.shape == (2, 64, 8, 8)


def test_attn_pool_weights_sum_to_one():
    """Attention weights in AttnPool2x2 sum to 1."""
    pool = AttnPool2x2(64)
    x = torch.randn(2, 64, 4, 4)
    B, C, H, W = x.shape
    x_bhwc = x.permute(0, 2, 3, 1)
    x_blocks = x_bhwc.view(B, H // 2, 2, W // 2, 2, C)
    x_blocks = x_blocks.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H // 2, W // 2, 4, C)
    x_norm = x_blocks / (x_blocks.norm(dim=-1, keepdim=True) + 1e-6)
    P = B * (H // 2) * (W // 2) * 4
    logits = pool.alpha * pool.phi(x_norm.reshape(P, C)) + pool.phi.bias
    weights = F.softmax(logits, dim=-1)
    assert weights.shape == (P, 4)
    assert (weights.sum(dim=-1) - 1.0).abs().max() < 1e-5


def test_attn_pool_grad_flows():
    """Gradients flow through AttnPool2x2."""
    pool = AttnPool2x2(32)
    x = torch.randn(1, 32, 8, 8, requires_grad=True)
    out = pool(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
