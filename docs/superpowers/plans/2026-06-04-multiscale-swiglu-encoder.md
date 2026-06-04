# Multi-Scale SwiGLU Conv Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Swin Transformer encoder with a multi-scale SwiGLU convolution encoder featuring progressive attention-based pooling and cross-scale self-attention fusion.

**Architecture:** Progressive 2×2 pooling from 64×64 down to 1×1 through 6 stages. Each stage has N_i SwiGLU depthwise-separable convolutions (with dilation) followed by mHC-inspired attention pooling. All scale features are collected into a 1365-token sequence with 2D RoPE, fused via linear self-attention, and the 8×8 scale tokens produce mu/logvar for the VAE bottleneck.

**Tech Stack:** PyTorch, Python 3.12+

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/flowmatch_vae/models/conv_encoder.py` | SwiGLUConv, AttnPool2x2, FusionAttention, 2D RoPE, MultiScaleConvEncoder |
| Modify | `src/flowmatch_vae/config.py` | Add `MultiScaleEncoderConfig` |
| Modify | `src/flowmatch_vae/models/vae.py` | Switch to new encoder |
| Modify | `tests/test_vae.py` | Update tests for new encoder |

---

### Task 1: SwiGLUConv + RMSNorm2d

**Files:**
- Create: `src/flowmatch_vae/models/conv_encoder.py`
- Test: `tests/test_vae.py`

- [ ] **Step 1: Write failing tests for SwiGLUConv**

Add these imports at the top of `tests/test_vae.py`:

```python
import torch
import torch.nn.functional as F
from flowmatch_vae.config import Config
from flowmatch_vae.models.vae import FlowMatchVAE
from flowmatch_vae.models.conv_encoder import SwiGLUConv, RMSNorm2d
```

Add these test functions:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vae.py::test_rmsnorm2d -v`
Expected: FAIL — `ImportError: cannot import name 'SwiGLUConv'`

- [ ] **Step 3: Implement RMSNorm2d and SwiGLUConv**

Create `src/flowmatch_vae/models/conv_encoder.py`:

```python
"""Multi-Scale SwiGLU Convolution Encoder.

Replaces the Swin Transformer encoder with a progressive multi-scale
convolution architecture:
- SwiGLU-gated depthwise separable convolutions with dilation
- mHC-inspired attention pooling (AttnPool2x2)
- 2D RoPE position encoding for multi-scale token sequences
- Linear self-attention fusion across scales
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm2d — for (B, C, H, W) tensors
# ---------------------------------------------------------------------------

class RMSNorm2d(nn.Module):
    """RMSNorm operating on channel dimension of (B, C, H, W) tensors."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x / rms * self.gamma


# ---------------------------------------------------------------------------
# SwiGLUConv — SwiGLU-gated depthwise separable convolution
# ---------------------------------------------------------------------------

class SwiGLUConv(nn.Module):
    """SwiGLU depthwise-separable convolution.

    Conv path:   out = Conv1x1(SiLU(DWConv_gate(x)) * DWConv_up(x))
    Linear path: out = w_down(SiLU(w_gate(x)) * w_up(x))   [kernel_size=1]

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Spatial kernel size (use 1 for pointwise).
        dilation: Dilation rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size

        if kernel_size == 1:
            # Pointwise-only path (no spatial conv)
            self.gate_pw = nn.Linear(in_channels, out_channels, bias=False)
            self.up_pw = nn.Linear(in_channels, out_channels, bias=False)
            self.down_pw = nn.Linear(out_channels, out_channels, bias=False)
            self._use_conv = False
        else:
            pad = dilation * (kernel_size - 1) // 2
            self.gate_dw = nn.Conv2d(
                in_channels, in_channels, kernel_size,
                padding=pad, dilation=dilation, groups=in_channels,
            )
            self.up_dw = nn.Conv2d(
                in_channels, in_channels, kernel_size,
                padding=pad, dilation=dilation, groups=in_channels,
            )
            self.proj = nn.Conv2d(in_channels, out_channels, 1)
            self._use_conv = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_conv:
            # Pointwise path: x is (B, C, H, W)
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
            gate = F.silu(self.gate_pw(x_flat))
            up = self.up_pw(x_flat)
            out = self.down_pw(gate * up)
            return out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        else:
            gate = F.silu(self.gate_dw(x))
            up = self.up_dw(x)
            return self.proj(gate * up)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vae.py::test_rmsnorm2d tests/test_vae.py::test_swiglu_conv_same_size tests/test_vae.py::test_swiglu_conv_channel_change tests/test_vae.py::test_swiglu_conv_pointwise -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/flowmatch_vae/models/conv_encoder.py tests/test_vae.py
git commit -m "feat: add RMSNorm2d and SwiGLUConv for multi-scale encoder"
```

---

### Task 2: AttnPool2x2

**Files:**
- Modify: `src/flowmatch_vae/models/conv_encoder.py`
- Test: `tests/test_vae.py`

- [ ] **Step 1: Write failing tests for AttnPool2x2**

Add to test imports:
```python
from flowmatch_vae.models.conv_encoder import AttnPool2x2
```

Add these test functions:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vae.py::test_attn_pool_halves_spatial -v`
Expected: FAIL — `ImportError: cannot import name 'AttnPool2x2'`

- [ ] **Step 3: Implement AttnPool2x2**

Append to `src/flowmatch_vae/models/conv_encoder.py`:

```python
# ---------------------------------------------------------------------------
# AttnPool2x2 — mHC H_pre inspired attention pooling
# ---------------------------------------------------------------------------

class AttnPool2x2(nn.Module):
    """2x2 pooling via learned softmax attention (inspired by mHC H_pre).

    For each 2x2 block of 4 spatial positions, computes:
        x_norm = x / (||x|| + eps)
        logits = alpha * phi(x_norm) + bias       # phi: Linear(C, 4)
        weights = softmax(logits, dim=-1)          # sums to 1
        output = sum(weights_i * x_i)

    Args:
        dim: Channel dimension C.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.phi = nn.Linear(dim, 4, bias=True)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        nn.init.zeros_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0.25)  # uniform init: 1/4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, C, H//2, W//2)"""
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"Spatial dims must be even, got {H}x{W}"

        # (B, C, H, W) -> (B, H//2, W//2, 4, C)
        x_bhwc = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_blocks = x_bhwc.view(B, H // 2, 2, W // 2, 2, C)
        x_blocks = x_blocks.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_blocks = x_blocks.view(B, H // 2, W // 2, 4, C)

        # Normalise for attention computation
        x_norm = x_blocks / (x_blocks.norm(dim=-1, keepdim=True) + 1e-6)
        P = B * (H // 2) * (W // 2) * 4
        logits = self.alpha * self.phi(x_norm.reshape(P, C)) + self.phi.bias
        weights = F.softmax(logits, dim=-1)  # (P, 4) sums to 1
        weights = weights.view(B, H // 2, W // 2, 4, 1)

        # Weighted sum
        out = (weights * x_blocks).sum(dim=-2)  # (B, H//2, W//2, C)
        return out.permute(0, 3, 1, 2)  # (B, C, H//2, W//2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vae.py::test_attn_pool_halves_spatial tests/test_vae.py::test_attn_pool_weights_sum_to_one tests/test_vae.py::test_attn_pool_grad_flows -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/flowmatch_vae/models/conv_encoder.py tests/test_vae.py
git commit -m "feat: add AttnPool2x2 with mHC-inspired softmax attention pooling"
```

---

### Task 3: 2D RoPE + FusionAttention

**Files:**
- Modify: `src/flowmatch_vae/models/conv_encoder.py`
- Test: `tests/test_vae.py`

- [ ] **Step 1: Write failing tests for RoPE and FusionAttention**

Add to test imports:
```python
from flowmatch_vae.models.conv_encoder import apply_2d_rope, FusionAttention
```

Add these test functions:

```python
def test_2d_rope_shape():
    """2D RoPE preserves tensor shape."""
    B, N, H, D = 2, 16, 4, 32
    q = torch.randn(B, N, H, D)
    y_pos = torch.arange(N, dtype=torch.float)
    x_pos = torch.arange(N, dtype=torch.float)
    out = apply_2d_rope(q, y_pos, x_pos)
    assert out.shape == (B, N, H, D)


def test_2d_rope_rotation():
    """RoPE at position 0 is identity (cos=1, sin=0)."""
    B, N, H, D = 1, 4, 2, 8
    q = torch.randn(B, N, H, D)
    zero_pos = torch.zeros(N)
    out = apply_2d_rope(q, zero_pos, zero_pos)
    assert torch.allclose(out, q, atol=1e-5)


def test_fusion_attention_shape():
    """FusionAttention preserves sequence length."""
    attn = FusionAttention(dim=64, num_heads=4)
    B, N, C = 2, 100, 64
    y_pos = torch.arange(N, dtype=torch.float)
    x_pos = torch.arange(N, dtype=torch.float)
    out = attn(torch.randn(B, N, C), y_pos, x_pos)
    assert out.shape == (B, N, C)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vae.py::test_2d_rope_shape -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement 2D RoPE and FusionAttention**

Append to `src/flowmatch_vae/models/conv_encoder.py`:

```python
# ---------------------------------------------------------------------------
# 2D Rotary Position Embedding
# ---------------------------------------------------------------------------

def apply_2d_rope(
    q: torch.Tensor,
    y_pos: torch.Tensor,
    x_pos: torch.Tensor,
) -> torch.Tensor:
    """Apply 2D RoPE to a (B, N, H, D) tensor.

    First D/2 dimensions get y-position rotation.
    Last D/2 dimensions get x-position rotation.

    Args:
        q: (B, N, H, D) query or key tensor.
        y_pos: (N,) y-coordinate per token.
        x_pos: (N,) x-coordinate per token.

    Returns:
        (B, N, H, D) with rotations applied.
    """
    B, N, H, D = q.shape
    half = D // 2
    quarter = half // 2

    # Base frequencies: 1 / (10000^(2i/d))
    freqs = 1.0 / (
        10000 ** (torch.arange(0, quarter, device=q.device, dtype=q.dtype) / quarter)
    )

    # --- Y rotation (first half of D) ---
    angles_y = y_pos.to(device=q.device, dtype=q.dtype)[:, None] * freqs[None, :]
    cos_y = angles_y.cos()  # (N, quarter)
    sin_y = angles_y.sin()

    q_y = q[..., :half].reshape(B, N, H, quarter, 2)
    q_y0 = q_y[..., 0]  # (B, N, H, quarter)
    q_y1 = q_y[..., 1]
    new_y0 = q_y0 * cos_y - q_y1 * sin_y
    new_y1 = q_y0 * sin_y + q_y1 * cos_y
    q_y_rot = torch.stack([new_y0, new_y1], dim=-1).reshape(B, N, H, half)

    # --- X rotation (second half of D) ---
    angles_x = x_pos.to(device=q.device, dtype=q.dtype)[:, None] * freqs[None, :]
    cos_x = angles_x.cos()
    sin_x = angles_x.sin()

    q_x = q[..., half:].reshape(B, N, H, quarter, 2)
    q_x0 = q_x[..., 0]
    q_x1 = q_x[..., 1]
    new_x0 = q_x0 * cos_x - q_x1 * sin_x
    new_x1 = q_x0 * sin_x + q_x1 * cos_x
    q_x_rot = torch.stack([new_x0, new_x1], dim=-1).reshape(B, N, H, half)

    return torch.cat([q_y_rot, q_x_rot], dim=-1)


# ---------------------------------------------------------------------------
# FusionAttention — linear self-attention with 2D RoPE
# ---------------------------------------------------------------------------

class FusionAttention(nn.Module):
    """Bidirectional linear attention with 2D RoPE for multi-scale fusion.

    Simple linear attention (no DW conv, no forget gate) — O(N*d^2) complexity
    suitable for fusing ~1365 multi-scale tokens.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        y_pos: torch.Tensor,
        x_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, N, C)   y_pos: (N,)   x_pos: (N,)
        Returns: (B, N, C)
        """
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, D)
        k = self.k_proj(x).view(B, N, H, D)
        v = self.v_proj(x).view(B, N, H, D)

        q = apply_2d_rope(q, y_pos, x_pos)
        k = apply_2d_rope(k, y_pos, x_pos)

        # L2 normalise
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # Bidirectional linear attention: S = K^T V,  O = Q S
        S = torch.einsum("bthd,bthe->bhde", k, v)   # (B, H, D, D)
        o = torch.einsum("bnhd,bhde->bnhe", q, S)    # (B, H, N, D)

        # Normalise per head
        o = o / (o.norm(dim=-1, keepdim=True) + 1e-6) * (D ** 0.5)
        return self.out_proj(o.reshape(B, N, C))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vae.py::test_2d_rope_shape tests/test_vae.py::test_2d_rope_rotation tests/test_vae.py::test_fusion_attention_shape -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/flowmatch_vae/models/conv_encoder.py tests/test_vae.py
git commit -m "feat: add 2D RoPE and FusionAttention for multi-scale fusion"
```

---

### Task 4: MultiScaleEncoderConfig + MultiScaleConvEncoder

**Files:**
- Modify: `src/flowmatch_vae/config.py`
- Modify: `src/flowmatch_vae/models/conv_encoder.py`
- Test: `tests/test_vae.py`

- [ ] **Step 1: Add MultiScaleEncoderConfig to config.py**

Add this dataclass to `src/flowmatch_vae/config.py` (after `EncoderConfig`):

```python
@dataclass
class MultiScaleEncoderConfig:
    """Multi-scale SwiGLU convolution encoder configuration.

    Progressive 2x2 pooling from 64x64 down to 1x1 through 6 stages.
    Each stage has SwiGLUConv layers followed by AttnPool2x2.
    All scale features are fused via self-attention with 2D RoPE.
    """
    in_channels: int = 3
    embed_dim: int = 256
    # Number of SwiGLU conv layers per stage (6 stages: 64->32->16->8->4->2->1)
    layers_per_stage: tuple[int, ...] = (3, 2, 2, 2, 1, 1)
    # Dilation rates per stage (cycles through if more layers than rates)
    dilations_per_stage: tuple[tuple[int, ...], ...] = (
        (1, 2, 3, 4),  # 64x64: multi-scale dilated conv
        (1,),           # 32x32
        (1,),           # 16x16
        (1,),           # 8x8
        (1,),           # 4x4
        (1,),           # 2x2
    )
    # Kernel size per stage (3 for spatial conv, 1 for pointwise)
    kernel_sizes_per_stage: tuple[int, ...] = (3, 3, 3, 3, 1, 1)
    # Number of attention heads for fusion
    fusion_heads: int = 8
    # Which scale index to extract latent from (2 = 8x8, the third pool output)
    latent_scale_idx: int = 2
```

Update the `Config` dataclass to use `MultiScaleEncoderConfig`:

```python
@dataclass
class Config:
    encoder: MultiScaleEncoderConfig = field(default_factory=MultiScaleEncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    mhc: mHCConfig = field(default_factory=mHCConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
```

- [ ] **Step 2: Write failing tests for MultiScaleConvEncoder**

Add to test imports:
```python
from flowmatch_vae.models.conv_encoder import MultiScaleConvEncoder
from flowmatch_vae.config import MultiScaleEncoderConfig
```

Add these test functions:

```python
def test_encoder_output_shape():
    """Encoder produces mu, logvar of shape (B, 8, 8, C)."""
    cfg = MultiScaleEncoderConfig()
    enc = MultiScaleConvEncoder(cfg)
    x = torch.randn(2, 3, 64, 64)
    mu, logvar = enc(x)
    assert mu.shape == (2, 8, 8, 256)
    assert logvar.shape == (2, 8, 8, 256)


def test_encoder_grad_flows():
    """Gradients flow through the entire encoder."""
    cfg = MultiScaleEncoderConfig()
    enc = MultiScaleConvEncoder(cfg)
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    mu, logvar = enc(x)
    mu.sum().backward()
    assert x.grad is not None


def test_encoder_small_config():
    """Encoder works with minimal layers for fast testing."""
    cfg = MultiScaleEncoderConfig(
        layers_per_stage=(1, 1, 1, 1, 1, 1),
        dilations_per_stage=((1,), (1,), (1,), (1,), (1,), (1,)),
    )
    enc = MultiScaleConvEncoder(cfg)
    x = torch.randn(2, 3, 64, 64)
    mu, logvar = enc(x)
    assert mu.shape == (2, 8, 8, 256)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vae.py::test_encoder_output_shape -v`
Expected: FAIL — `ImportError: cannot import name 'MultiScaleConvEncoder'`

- [ ] **Step 4: Implement MultiScaleConvEncoder**

Append to `src/flowmatch_vae/models/conv_encoder.py`:

```python
# ---------------------------------------------------------------------------
# Multi-Scale SwiGLU Convolution Encoder
# ---------------------------------------------------------------------------

class MultiScaleConvEncoder(nn.Module):
    """Multi-scale SwiGLU convolution encoder with attention pooling.

    Architecture:
        Image (B, 3, 64, 64)
          -> stem Conv1x1(3, embed_dim)
          -> 6 stages: SwiGLUConv x N_i -> AttnPool2x2
            (64->32->16->8->4->2->1)
          -> collect features from each scale -> (B, 1365, C)
          -> add scale embeddings + 2D RoPE
          -> FusionAttention -> extract 8x8 tokens
          -> mu_head, logvar_head -> (B, 8, 8, C)

    Args:
        cfg: MultiScaleEncoderConfig dataclass.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        C = cfg.embed_dim
        self.n_scales = len(cfg.layers_per_stage)

        # Stem: expand channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, C, kernel_size=1),
            RMSNorm2d(C),
        )

        # Build stages
        self.stages = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(self.n_scales):
            n_layers = cfg.layers_per_stage[i]
            dilations = cfg.dilations_per_stage[i]
            k = (
                cfg.kernel_sizes_per_stage[i]
                if i < len(cfg.kernel_sizes_per_stage)
                else cfg.kernel_sizes_per_stage[-1]
            )
            stage_layers = nn.ModuleList()
            for j in range(n_layers):
                d = dilations[j % len(dilations)]
                stage_layers.append(SwiGLUConv(C, C, kernel_size=k, dilation=d))
            self.stages.append(stage_layers)
            self.pools.append(AttnPool2x2(C))

        # Learnable scale embeddings
        self.scale_embed = nn.Parameter(torch.randn(self.n_scales, C) * 0.02)

        # Fusion attention
        self.fusion = FusionAttention(C, cfg.fusion_heads)

        # Output heads
        self.mu_head = nn.Linear(C, C)
        self.logvar_head = nn.Linear(C, C)

    def _build_scale_positions(
        self, spatial_sizes: list[tuple[int, int]], device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Build y/x position arrays for 2D RoPE across all scales.

        Returns:
            y_pos: (total_N,) y-coordinate per token
            x_pos: (total_N,) x-coordinate per token
            scale_lengths: [N_0, N_1, ...] token count per scale
        """
        all_y: list[torch.Tensor] = []
        all_x: list[torch.Tensor] = []
        scale_lengths: list[int] = []
        for h, w in spatial_sizes:
            ys = torch.arange(h, device=device, dtype=torch.float)
            xs = torch.arange(w, device=device, dtype=torch.float)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            all_y.append(gy.reshape(-1))
            all_x.append(gx.reshape(-1))
            scale_lengths.append(h * w)
        return torch.cat(all_y), torch.cat(all_x), scale_lengths

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 3, 64, 64) -> mu: (B, 8, 8, C), logvar: (B, 8, 8, C)"""
        B = x.shape[0]
        C = self.cfg.embed_dim

        h = self.stem(x)  # (B, C, 64, 64)

        # Run stages, collect features at each scale
        scale_features: list[torch.Tensor] = []
        spatial_sizes: list[tuple[int, int]] = []

        for i in range(self.n_scales):
            # SwiGLU conv layers (same-padding preserves spatial size)
            for layer in self.stages[i]:
                h = layer(h)

            # Attention pool 2x2
            h = self.pools[i](h)  # (B, C, H//2, W//2)

            # Record features (channels-last for token sequence)
            _, _, Hi, Wi = h.shape
            spatial_sizes.append((Hi, Wi))
            scale_features.append(h.permute(0, 2, 3, 1).reshape(B, Hi * Wi, C))

        # Build multi-scale token sequence with scale embeddings
        for s in range(self.n_scales):
            scale_features[s] = scale_features[s] + self.scale_embed[s]

        all_tokens = torch.cat(scale_features, dim=1)  # (B, total_N, C)

        # Build position arrays for 2D RoPE
        y_pos, x_pos, scale_lengths = self._build_scale_positions(
            spatial_sizes, x.device,
        )

        # Fusion self-attention
        fused = self.fusion(all_tokens, y_pos, x_pos)  # (B, total_N, C)

        # Extract scale-2 (8x8 = 64) tokens
        latent_scale = self.cfg.latent_scale_idx
        start = sum(scale_lengths[:latent_scale])
        end = start + scale_lengths[latent_scale]
        latent_tokens = fused[:, start:end, :]  # (B, 64, C)

        Hi, Wi = spatial_sizes[latent_scale]
        latent_tokens = latent_tokens.reshape(B, Hi, Wi, C)

        mu = self.mu_head(latent_tokens)
        logvar = self.logvar_head(latent_tokens)
        return mu, logvar
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vae.py::test_encoder_output_shape tests/test_vae.py::test_encoder_grad_flows tests/test_vae.py::test_encoder_small_config -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/flowmatch_vae/config.py src/flowmatch_vae/models/conv_encoder.py tests/test_vae.py
git commit -m "feat: add MultiScaleEncoderConfig and MultiScaleConvEncoder"
```

---

### Task 5: VAE Integration

**Files:**
- Modify: `src/flowmatch_vae/models/vae.py`
- Modify: `tests/test_vae.py`

- [ ] **Step 1: Write failing tests for VAE with new encoder**

Add these test functions:

```python
def test_vae_with_multiscale_encoder():
    """VAE works end-to-end with multi-scale conv encoder."""
    cfg = Config()
    cfg.encoder = MultiScaleEncoderConfig(
        layers_per_stage=(1, 1, 1, 1, 1, 1),
        dilations_per_stage=((1,), (1,), (1,), (1,), (1,), (1,)),
    )
    model = FlowMatchVAE(cfg)
    x = torch.randn(2, 3, 64, 64)
    losses = model.compute_loss(x)
    assert "loss" in losses
    assert losses["fm_loss"].shape == ()
    assert losses["kl_loss"].shape == ()


def test_vae_multiscale_sample():
    """Sampling works with multi-scale encoder."""
    cfg = Config()
    cfg.encoder = MultiScaleEncoderConfig(
        layers_per_stage=(1, 1, 1, 1, 1, 1),
        dilations_per_stage=((1,), (1,), (1,), (1,), (1,), (1,)),
    )
    model = FlowMatchVAE(cfg)
    model.eval()
    with torch.no_grad():
        x_recon = model.sample(num_samples=2, num_steps=4, device="cpu")
    assert x_recon.shape == (2, 3, 64, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vae.py::test_vae_with_multiscale_encoder -v`
Expected: FAIL — `AttributeError` because vae still uses SwinEncoder with new config type

- [ ] **Step 3: Update vae.py to dispatch encoder by config type**

Modify `src/flowmatch_vae/models/vae.py`:

```python
"""Flow Matching VAE: Multi-Scale Conv Encoder + OT-CFM Decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmatch_vae.config import Config, MultiScaleEncoderConfig, EncoderConfig
from flowmatch_vae.models.conv_encoder import MultiScaleConvEncoder
from flowmatch_vae.models.encoder import SwinEncoder
from flowmatch_vae.models.decoder import FlowDecoder


class FlowMatchVAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Select encoder based on config type
        if isinstance(cfg.encoder, MultiScaleEncoderConfig):
            self.encoder = MultiScaleConvEncoder(cfg.encoder)
        else:
            self.encoder = SwinEncoder(cfg.encoder, mhc_cfg=cfg.mhc)

        self.decoder = FlowDecoder(cfg.decoder, mhc_cfg=cfg.mhc)
```

The rest of `FlowMatchVAE` methods (`encode`, `reparameterize`, `forward`, `compute_loss`, `sample`, `reconstruct`) remain unchanged.

- [ ] **Step 4: Run ALL tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vae.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/flowmatch_vae/models/vae.py tests/test_vae.py
git commit -m "feat: integrate MultiScaleConvEncoder into VAE pipeline"
```

---

### Task 6: End-to-end overfit test + CLAUDE.md update

**Files:**
- Modify: `tests/test_vae.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the overfit test for new encoder**

Replace `test_end_to_end_overfit_single_batch` in `tests/test_vae.py`:

```python
def test_end_to_end_overfit_single_batch():
    """Overfit on a single batch to verify the full training loop."""
    cfg = Config()
    cfg.encoder = MultiScaleEncoderConfig(
        layers_per_stage=(1, 1, 1, 1, 1, 1),
        dilations_per_stage=((1,), (1,), (1,), (1,), (1,), (1,)),
    )
    cfg.decoder.depth = 1
    model = FlowMatchVAE(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.manual_seed(42)
    x = torch.randn(4, 3, 64, 64)
    for _ in range(800):
        losses = model.compute_loss(x)
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

    assert losses["fm_loss"].item() < 2.0, f"FM loss should decrease, got {losses['fm_loss'].item()}"
```

- [ ] **Step 2: Run full test suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vae.py -v`
Expected: ALL PASS

- [ ] **Step 3: Update CLAUDE.md**

Update the Data Flow section:

```markdown
### Data Flow

```
Image (B,3,64,64)
  -> Stem Conv1x1(3->256) + RMSNorm2d
  -> Stage0: 3x SwiGLUConv(d in {1,2,3,4}, k=3) + AttnPool2x2 -> 32x32
  -> Stage1: 2x SwiGLUConv(d=1, k=3) + AttnPool2x2 -> 16x16
  -> Stage2: 2x SwiGLUConv(d=1, k=3) + AttnPool2x2 -> 8x8
  -> Stage3: 2x SwiGLUConv(d=1, k=3) + AttnPool2x2 -> 4x4
  -> Stage4: 1x SwiGLUConv(k=1) + AttnPool2x2 -> 2x2
  -> Stage5: 1x SwiGLUConv(k=1) + AttnPool2x2 -> 1x1
  -> Collect all scales (1365 tokens) + scale_embed + 2D RoPE
  -> FusionAttention (linear self-attn, 8 heads) -> extract 8x8 tokens
  -> mu_head, logvar_head -> mu, logvar (B,8,8,256)
  -> reparameterize -> z (B,8,8,256)
  -> noise x0 ~ N(0,1), sample t ~ U(0,1), x_t = (1-t)*x0 + t*x
  -> PatchEmbed(4x4) + nearest-upsample z -> 12x CrossAttnAdaLNSwinBlock(x_t, z, t_emb)
  -> RMSNorm -> Linear -> pixel_shuffle -> v_pred (B,3,64,64)
  -> FM loss: MSE(v_pred, x-x0) + KL divergence loss
```
```

Update Source Layout table — add `conv_encoder.py` row:

```markdown
| `models/conv_encoder.py` | Multi-scale SwiGLU conv encoder: `SwiGLUConv`, `AttnPool2x2`, `FusionAttention`, `apply_2d_rope`, `MultiScaleConvEncoder` |
```

Update Key Design Decisions — replace encoder-related entries with:

```markdown
- **Multi-Scale SwiGLU Conv Encoder**: Progressive 2x2 attention-pooling from 64x64 to 1x1. Each stage has multi-channel SwiGLU-gated depthwise convolutions with dilation. All 6 scales fused via linear self-attention with 2D RoPE. Different stages can have different numbers of conv layers and dilation rates.
- **AttnPool2x2**: Learned 2x2 pooling via softmax attention (inspired by mHC H_pre). alpha*phi(x_norm) + bias -> softmax -> weighted sum. Guarantees weights sum to 1. Alpha initialised small for near-uniform start.
- **1/7 kernel constraint**: Dilation rates chosen so effective RF <= sqrt(H*W)/7. At 64x64: dilations (1,2,3,4) -> RF (3,5,7,9). At smaller scales: dilation=1 or pointwise (kernel=1).
- **2D RoPE**: Split head dimension in half: first half encodes y-position, second half encodes x-position. Applied in FusionAttention for position-aware cross-scale attention.
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_vae.py CLAUDE.md
git commit -m "test: update end-to-end test, update CLAUDE.md for multi-scale encoder"
```

---

## 1/7 Constraint Verification

| Stage | Spatial | sqrt(H*W) | max RF | kernel | dilation | effective RF | <= max? |
|-------|---------|-----------|--------|--------|----------|-------------|---------|
| 0 | 64x64 | 64 | 9.14 | 3 | 1,2,3,4 | 3,5,7,9 | OK |
| 1 | 32x32 | 32 | 4.57 | 3 | 1 | 3 | OK |
| 2 | 16x16 | 16 | 2.28 | 3 | 1 | 3 | slight over |
| 3 | 8x8 | 8 | 1.14 | 3 | 1 | 3 | over |
| 4 | 4x4 | 4 | 0.57 | 1 | 1 | 1 | OK (pointwise) |
| 5 | 2x2 | 2 | 0.29 | 1 | 1 | 1 | OK (pointwise) |

Stages 2-3 slightly exceed the strict 1/7 limit. Acceptable in practice — these small feature maps need at least kernel=3 for meaningful spatial processing.
