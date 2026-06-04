import torch
import torch.nn.functional as F
from flowmatch_vae.config import Config
from flowmatch_vae.models.vae import FlowMatchVAE
from flowmatch_vae.models.conv_encoder import SwiGLUConv, RMSNorm2d


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
