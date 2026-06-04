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

        # Compute attention logits: phi maps each position (C) -> 4 scores.
        # For each block of 4 positions, average the 4 sets of scores and softmax.
        N = B * (H // 2) * (W // 2)
        phi_out = self.phi(x_norm.reshape(N * 4, C))  # (N*4, 4)
        logits = self.alpha * phi_out + self.phi.bias   # (N*4, 4)
        logits = logits.view(N, 4, 4).mean(dim=1)       # (N, 4) — average votes
        weights = F.softmax(logits, dim=-1)              # (N, 4) sums to 1
        weights = weights.view(B, H // 2, W // 2, 4, 1)

        # Weighted sum
        out = (weights * x_blocks).sum(dim=-2)  # (B, H//2, W//2, C)
        return out.permute(0, 3, 1, 2)  # (B, C, H//2, W//2)


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
    angles_y = y_pos.to(device=q.device, dtype=q.dtype)[:, None] * freqs[None, :]  # (N, quarter)
    cos_y = angles_y.cos()[None, :, None, :]  # (1, N, 1, quarter)
    sin_y = angles_y.sin()[None, :, None, :]

    q_y = q[..., :half].reshape(B, N, H, quarter, 2)
    q_y0 = q_y[..., 0]  # (B, N, H, quarter)
    q_y1 = q_y[..., 1]
    new_y0 = q_y0 * cos_y - q_y1 * sin_y
    new_y1 = q_y0 * sin_y + q_y1 * cos_y
    q_y_rot = torch.stack([new_y0, new_y1], dim=-1).reshape(B, N, H, half)

    # --- X rotation (second half of D) ---
    angles_x = x_pos.to(device=q.device, dtype=q.dtype)[:, None] * freqs[None, :]  # (N, quarter)
    cos_x = angles_x.cos()[None, :, None, :]  # (1, N, 1, quarter)
    sin_x = angles_x.sin()[None, :, None, :]

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
