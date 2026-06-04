"""Swin Transformer building blocks.

Implements core Swin Transformer components including:
- window_partition / window_reverse
- WindowAttention with relative position bias
- SwinBlock (standard W-MSA / SW-MSA block)
- AdaLNSwinBlock (adaptive LayerNorm modulated block for decoder)
- PatchEmbed / PatchMerge
- DropPath (stochastic depth)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DropPath (stochastic depth)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, expand the first dim
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


# ---------------------------------------------------------------------------
# Window partition / reverse
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition (B, H, W, C) into (B * nH * nW, ws, ws, C).

    Args:
        x: Input tensor of shape (B, H, W, C).
        window_size: Window size (ws).

    Returns:
        Windows tensor of shape (B * nH * nW, ws, ws, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """Reverse window partition back to (B, H, W, C).

    Args:
        windows: (B * nH * nW, ws, ws, C)
        window_size: Window size.
        H: Original height.
        W: Original width.

    Returns:
        Restored tensor of shape (B, H, W, C).
    """
    nH = H // window_size
    nW = W // window_size
    B = windows.shape[0] // (nH * nW)
    x = windows.view(B, nH, nW, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
# Shifted window attention mask
# ---------------------------------------------------------------------------

def _compute_shift_mask(
    window_size: int, shift_size: int, H: int, W: int
) -> torch.Tensor:
    """Compute the attention mask for shifted-window MSA.

    Returns:
        Attention mask of shape (nW, ws*ws, ws*ws).
    """
    img_mask = torch.zeros((1, H, W, 1))
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # (nW, ws, ws, 1)
    nW = mask_windows.shape[0]
    mask_windows = mask_windows.view(nW, -1)  # (nW, ws*ws)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask  # (nW, ws*ws, ws*ws)


# ---------------------------------------------------------------------------
# WindowAttention
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        qkv_bias: Whether to add bias to qkv projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table: (2*ws-1, 2*ws-1, num_heads)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Pre-compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, ws, ws)
        coords_flatten = torch.flatten(coords, 1)  # (2, ws*ws)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, ws*ws, ws*ws)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (ws*ws, ws*ws, 2)
        # Shift to start from 0
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (ws*ws, ws*ws)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (num_windows, ws*ws, dim)
            mask: Optional attention mask (nW, ws*ws, ws*ws).

        Returns:
            (num_windows, ws*ws, dim)
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B_, num_heads, N, head_dim)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, num_heads, N, N)

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# SwinBlock
# ---------------------------------------------------------------------------

class SwinBlock(nn.Module):
    """Swin Transformer block with W-MSA / SW-MSA.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size (0 for W-MSA, window_size//2 for SW-MSA).
        mlp_ratio: MLP hidden dim multiplier.
        drop_path: DropPath rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, H, W, C)

        Returns:
            (B, H, W, C)
        """
        B, H, W, C = x.shape
        shortcut = x

        # Cyclic shift
        shifted_x = x
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Attention mask for shifted windows
        attn_mask = None
        if self.shift_size > 0:
            attn_mask = _compute_shift_mask(self.window_size, self.shift_size, H, W)
            attn_mask = attn_mask.to(x.device)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_out = shifted_x

        # Residual
        x_out = shortcut + self.drop_path(self.norm1(x_out))
        x_out = x_out + self.drop_path(self.mlp(self.norm2(x_out)))
        return x_out


# ---------------------------------------------------------------------------
# AdaLNSwinBlock
# ---------------------------------------------------------------------------

class AdaLNSwinBlock(nn.Module):
    """Swin Transformer block with adaptive LayerNorm modulation.

    Accepts a conditioning vector ``cond`` and uses it to predict per-sample
    scale / shift parameters for both LayerNorm layers.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size.
        mlp_ratio: MLP hidden dim multiplier.
        drop_path: DropPath rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

        # AdaLN modulation: cond -> 6 * dim (s1, sh1, s2, sh2, g1, g2)
        self.adaLN_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        # Zero-init the last layer
        nn.init.constant_(self.adaLN_mlp[-1].weight, 0.0)
        nn.init.constant_(self.adaLN_mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, H, W, C)
            cond: (B, C)

        Returns:
            (B, H, W, C)
        """
        B, H, W, C = x.shape

        # AdaLN parameters from conditioning
        params = self.adaLN_mlp(cond)  # (B, 6*C)
        s1, sh1, s2, sh2, g1, g2 = params.chunk(6, dim=-1)  # each (B, C)

        # Reshape for broadcasting over spatial dims
        s1 = s1.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, C)
        sh1 = sh1.unsqueeze(1).unsqueeze(2)
        s2 = s2.unsqueeze(1).unsqueeze(2)
        sh2 = sh2.unsqueeze(1).unsqueeze(2)
        g1 = g1.unsqueeze(1).unsqueeze(2)
        g2 = g2.unsqueeze(1).unsqueeze(2)

        shortcut = x

        # Cyclic shift
        shifted_x = x
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Attention mask for shifted windows
        attn_mask = None
        if self.shift_size > 0:
            attn_mask = _compute_shift_mask(self.window_size, self.shift_size, H, W)
            attn_mask = attn_mask.to(x.device)

        # Apply adaptive modulation to norm1 output inside windows
        # We need to apply norm1 before partitioning so the spatial modulation works
        # Actually, apply norm before window partition
        # Re-do: norm1 with adaLN modulation, then partition
        normed1 = self.norm1(shifted_x)  # (B, H, W, C)
        normed1 = normed1 * (1 + s1) + sh1

        x_windows = window_partition(normed1, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # Residual with gate g1
        x_out = shortcut + g1 * self.drop_path(shifted_x)

        # FFN with adaLN modulation on norm2
        normed2 = self.norm2(x_out)
        normed2 = normed2 * (1 + s2) + sh2
        x_out = x_out + g2 * self.drop_path(self.mlp(normed2))

        return x_out


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to patch embedding using a Conv2d projection.

    Args:
        in_channels: Number of input image channels.
        patch_size: Patch size.
        embed_dim: Embedding dimension.
    """

    def __init__(self, in_channels: int = 3, patch_size: int = 4, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W) image tensor.

        Returns:
            (B, H/patch_size, W/patch_size, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.permute(0, 2, 3, 1)  # (B, H/ps, W/ps, embed_dim)
        return x


# ---------------------------------------------------------------------------
# PatchMerge
# ---------------------------------------------------------------------------

class PatchMerge(nn.Module):
    """Merge 2x2 neighbouring patches, halving spatial and doubling channels.

    Args:
        dim: Input channel dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, H, W, C)

        Returns:
            (B, H/2, W/2, 2*C)
        """
        B, H, W, C = x.shape
        # View + permute to gather 2x2 neighbourhood
        x = x.view(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H/2, W/2, 2, 2, C)
        x = x.view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ---------------------------------------------------------------------------
# CrossAttnAdaLNSwinBlock
# ---------------------------------------------------------------------------

class CrossAttnAdaLNSwinBlock(nn.Module):
    """Swin Transformer block with adaptive LayerNorm modulation and cross-attention for z injection.

    Structure:
        AdaLN(x) -> WindowSelfAttn -> residual (gate g1)
        -> LN -> CrossAttn(q=x, kv=z) -> residual (gate g2)
        -> AdaLN -> FFN -> residual (gate g3)

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size for self-attention.
        shift_size: Shift size for self-attention.
        mlp_ratio: MLP hidden dim multiplier.
        z_dim: Dimension of z tokens (default: equal to dim).
        drop_path: DropPath rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        z_dim: int | None = None,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.head_dim = dim // num_heads

        if z_dim is None:
            z_dim = dim
        self.z_dim = z_dim

        # Self-attention (windowed)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.drop_path = DropPath(drop_path)

        # Cross-attention (standard, non-windowed)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_q = nn.Linear(dim, dim)
        self.cross_kv = nn.Linear(z_dim, dim * 2)
        self.cross_proj = nn.Linear(dim, dim)

        # FFN
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

        # AdaLN modulation: cond -> 9 * dim
        # (s1, sh1, g1) for self-attn, (s2, sh2, g2) for cross-attn, (s3, sh3, g3) for FFN
        self.adaLN_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim),
        )
        # Zero-init the last layer
        nn.init.constant_(self.adaLN_mlp[-1].weight, 0.0)
        nn.init.constant_(self.adaLN_mlp[-1].bias, 0.0)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, z_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, H, W, C)
            cond: (B, C) time embedding
            z_tokens: (B, 16, 16, C) upsampled z tokens

        Returns:
            (B, H, W, C)
        """
        B, H, W, C = x.shape
        N = H * W

        # AdaLN parameters from conditioning
        params = self.adaLN_mlp(cond)  # (B, 9*C)
        s1, sh1, g1, s2, sh2, g2, s3, sh3, g3 = params.chunk(9, dim=-1)

        # Reshape for broadcasting over spatial dims
        s1 = s1.unsqueeze(1).unsqueeze(2)
        sh1 = sh1.unsqueeze(1).unsqueeze(2)
        g1 = g1.unsqueeze(1).unsqueeze(2)
        s2 = s2.unsqueeze(1).unsqueeze(2)
        sh2 = sh2.unsqueeze(1).unsqueeze(2)
        g2 = g2.unsqueeze(1).unsqueeze(2)
        s3 = s3.unsqueeze(1).unsqueeze(2)
        sh3 = sh3.unsqueeze(1).unsqueeze(2)
        g3 = g3.unsqueeze(1).unsqueeze(2)

        # ------------------------------------------------------------------
        # 1. Self-attention (windowed, same as AdaLNSwinBlock)
        # ------------------------------------------------------------------
        shortcut = x

        # Cyclic shift
        shifted_x = x
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # AdaLN modulation on norm1
        normed1 = self.norm1(shifted_x)
        normed1 = normed1 * (1 + s1) + sh1

        # Window partition
        x_windows = window_partition(normed1, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Attention mask for shifted windows
        attn_mask = None
        if self.shift_size > 0:
            attn_mask = _compute_shift_mask(self.window_size, self.shift_size, H, W)
            attn_mask = attn_mask.to(x.device)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # Residual with gate g1
        x_out = shortcut + g1 * self.drop_path(shifted_x)

        # ------------------------------------------------------------------
        # 2. Cross-attention (standard, non-windowed)
        # ------------------------------------------------------------------
        x_flat = x_out.reshape(B, N, C)  # (B, N, C)
        z_flat = z_tokens.reshape(B, -1, C)  # (B, N_z, C)

        x_norm = self.cross_norm(x_flat)
        q = self.cross_q(x_norm)  # (B, N, C)
        kv = self.cross_kv(z_flat)  # (B, N_z, 2*C)
        k, v = kv.chunk(2, dim=-1)

        # Multi-head attention
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        cross_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        cross_out = self.cross_proj(cross_out)
        cross_out = cross_out.reshape(B, H, W, C)

        x_out = x_out + g2 * cross_out

        # ------------------------------------------------------------------
        # 3. FFN (same as AdaLNSwinBlock)
        # ------------------------------------------------------------------
        normed2 = self.norm2(x_out)
        normed2 = normed2 * (1 + s3) + sh3
        x_out = x_out + g3 * self.drop_path(self.mlp(normed2))

        return x_out
