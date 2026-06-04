"""Swin Transformer building blocks — upgraded architecture.

Key changes from the baseline:
- **Kimi Linear Attention** (KDA) replaces standard softmax window attention.
  Channel-wise gated delta attention with L2-normalised Q/K, output gating,
  and short depthwise convolutions (arXiv:2510.26692).
- **mHC** (Manifold-Constrained Hyper-Connections) replaces standard residual
  connections.  Doubly-stochastic Sinkhorn-Knopp projection on H_res,
  sigmoid-constrained H_pre / H_post, n-stream expansion (arXiv:2512.24880).
- **SwiGLU FFN** replaces standard GELU MLP.
- **RMSNorm** everywhere — no LayerNorm, no BatchNorm.
- **Swish (SiLU)** everywhere — no GELU, no ReLU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation.

    ``y = x / RMS(x) * gamma``  where  ``RMS(x) = sqrt(mean(x^2) + eps)``.

    Args:
        dim: Normalised dimension.
        eps: Small constant for numerical stability.
        elementwise_affine: If *True*, learn a per-element ``gamma`` parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("gamma", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        if self.elementwise_affine:
            x_normed = x_normed * self.gamma
        return x_normed


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
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


# ---------------------------------------------------------------------------
# Window partition / reverse
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) → (B·nH·nW, ws, ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """(B·nH·nW, ws, ws, C) → (B, H, W, C)."""
    nH, nW = H // window_size, W // window_size
    B = windows.shape[0] // (nH * nW)
    x = windows.view(B, nH, nW, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def _compute_shift_mask(window_size: int, shift_size: int, H: int, W: int) -> torch.Tensor:
    """Attention mask for shifted-window MSA.  Shape: (nW, ws², ws²)."""
    img_mask = torch.zeros((1, H, W, 1))
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


# ---------------------------------------------------------------------------
# Kimi Linear Attention (KDA) — bidirectional variant for vision
# ---------------------------------------------------------------------------

class KimiLinearAttention(nn.Module):
    """Kimi Delta Attention — bidirectional, windowed.

    Replaces softmax attention with linear-attention plus:
    - L2 normalisation on Q and K
    - Learnable channel-wise decay gate (alpha)
    - Delta-rule corrective term (beta)
    - Output gating (sigmoid)
    - Short depthwise conv for Q, K, V

    Args:
        dim: Token dimension.
        num_heads: Number of attention heads.
        window_size: Window size (used for depthwise conv kernel size hint).
    """

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        # Separate Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # Short depthwise convolutions (kernel=3, padding=1)
        self.dw_conv_q = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dw_conv_k = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dw_conv_v = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # Channel-wise forget gate: low-rank projection
        self.alpha_up = nn.Linear(dim, dim, bias=False)
        self.alpha_down = nn.Linear(dim, dim, bias=True)

        # Scalar beta (delta-rule learning rate)
        self.beta_proj = nn.Linear(dim, num_heads, bias=True)

        # Output projection + gating
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate_up = nn.Linear(dim, dim, bias=False)
        self.gate_down = nn.Linear(dim, dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
        for m in (self.dw_conv_q, self.dw_conv_k, self.dw_conv_v):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (num_windows, ws², dim)
        Returns: (num_windows, ws², dim)
        """
        B_, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Short depthwise conv + Swish
        q = F.silu(self.dw_conv_q(q.transpose(1, 2))).transpose(1, 2)
        k = F.silu(self.dw_conv_k(k.transpose(1, 2))).transpose(1, 2)
        v = F.silu(self.dw_conv_v(v.transpose(1, 2))).transpose(1, 2)

        # L2 normalise Q and K
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # Channel-wise forget gate alpha
        alpha = torch.sigmoid(self.alpha_down(F.silu(self.alpha_up(x))))  # (B_, N, C)

        # Scalar beta
        beta = torch.sigmoid(self.beta_proj(x))  # (B_, N, H)

        # Reshape into heads
        q = q.view(B_, N, H, D)
        k = k.view(B_, N, H, D)
        v = v.view(B_, N, H, D)
        alpha = alpha.view(B_, N, H, D)

        # Bidirectional linear attention:
        #   S = Σ_t (beta_t * alpha_t ⊙ k_t) ⊗ v_t   →  (B_, H, D, D)
        beta_exp = beta.unsqueeze(-1)            # (B_, N, H, 1)
        k_w = beta_exp * alpha * k               # (B_, N, H, D)

        S = torch.einsum("bthd,bthe->bhde", k_w, v)  # (B_, H, D, D)

        # o = q @ S  →  (B_, H, N, D)
        o = torch.einsum("bnhd,bhde->bnhe", q, S)

        # RMSNorm per head
        o = o / (o.norm(dim=-1, keepdim=True) + 1e-6) * (D ** 0.5)

        o = o.reshape(B_, N, C)

        # Output gating
        gate = torch.sigmoid(self.gate_down(F.silu(self.gate_up(x))))
        o = gate * o

        return self.out_proj(o)


# ---------------------------------------------------------------------------
# Kimi Linear Cross-Attention
# ---------------------------------------------------------------------------

class KimiLinearCrossAttention(nn.Module):
    """Cross-attention using Kimi Linear (bidirectional).

    Q comes from x, K/V come from z_tokens.
    """

    def __init__(self, dim: int, z_dim: int | None = None, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if z_dim is None:
            z_dim = dim
        self.z_dim = z_dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(z_dim, dim, bias=False)
        self.v_proj = nn.Linear(z_dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate_up = nn.Linear(dim, dim, bias=False)
        self.gate_down = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, dim)   z: (B, N_z, z_dim)
        Returns: (B, N, dim)
        """
        B, N, C = x.shape
        N_z = z.shape[1]
        H = self.num_heads
        D = self.head_dim

        q = F.silu(self.q_proj(x))
        k = F.silu(self.k_proj(z))
        v = F.silu(self.v_proj(z))

        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        q = q.view(B, N, H, D)
        k = k.view(B, N_z, H, D)
        v = v.view(B, N_z, H, D)

        S = torch.einsum("bthd,bthe->bhde", k, v)      # (B, H, D, D)
        o = torch.einsum("bnhd,bhde->bnhe", q, S)      # (B, H, N, D)

        o = o / (o.norm(dim=-1, keepdim=True) + 1e-6) * (D ** 0.5)
        o = o.reshape(B, N, C)

        gate = torch.sigmoid(self.gate_down(F.silu(self.gate_up(x))))
        o = gate * o
        return self.out_proj(o)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network.

    hidden_mult = 8/3 ≈ 2.67 matches the parameter count of a standard 4× MLP
    when rounded to 256-multiples.
    """

    def __init__(self, dim: int, hidden_mult: float = 8.0 / 3.0, out_features: int | None = None):
        super().__init__()
        hidden = int(dim * hidden_mult)
        hidden = ((hidden + 255) // 256) * 256
        out_features = out_features or dim

        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# mHC — Manifold-Constrained Hyper-Connections
# ---------------------------------------------------------------------------

def sinkhorn_knopp(M: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Project M (..., n, n) to doubly-stochastic via Sinkhorn-Knopp."""
    M = torch.exp(M)
    for _ in range(iters):
        M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
        M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
    return M


class mHCConnection(nn.Module):
    """Manifold-Constrained Hyper-Connection for a single sub-layer.

    Two-phase usage:
        1. ``layer_input, ctx = mhc.read(x_stream)``
        2. Run sublayer on ``layer_input``
        3. ``x_stream = mhc.write(x_stream, sublayer_out, ctx)``

    This ensures the mappings (H_pre, H_res, H_post) are computed once from
    the current stream state and reused consistently.

    Args:
        dim: Token dimension C.
        expansion_rate: Number of parallel residual streams n.
        sinkhorn_iters: Sinkhorn-Knopp iterations.
        gating_init: Initial value for learnable gating scalars.
    """

    def __init__(
        self,
        dim: int,
        expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
        gating_init: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.n = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        nC = self.n * dim

        # Input-dependent projections (operate on flattened stream nC)
        self.phi_pre = nn.Linear(nC, self.n, bias=True)
        self.phi_post = nn.Linear(nC, self.n, bias=True)
        self.phi_res = nn.Linear(nC, self.n * self.n, bias=True)

        # Learnable gating scalars (small init → near-identity at start)
        self.alpha_pre = nn.Parameter(torch.tensor(gating_init))
        self.alpha_post = nn.Parameter(torch.tensor(gating_init))
        self.alpha_res = nn.Parameter(torch.tensor(gating_init))

    def _compute_mappings(self, x_flat: torch.Tensor):
        """Compute H_pre, H_res, H_post from flat stream state.

        Args:
            x_flat: (P, nC) where P = B·H·W or B·N

        Returns:
            H_pre:   (P, n)       sigmoid-constrained
            H_post:  (P, n)       2·sigmoid-constrained
            H_res:   (P, n, n)    doubly-stochastic via Sinkhorn-Knopp
        """
        # RMSNorm on stream
        x_norm = x_flat / (x_flat.norm(dim=-1, keepdim=True) + 1e-6)

        H_raw_pre = self.alpha_pre * self.phi_pre(x_norm) + self.phi_pre.bias
        H_raw_post = self.alpha_post * self.phi_post(x_norm) + self.phi_post.bias
        H_raw_res = self.alpha_res * self.phi_res(x_norm) + self.phi_res.bias

        H_pre = torch.sigmoid(H_raw_pre)                                   # (P, n) ∈ (0,1)
        H_post = 2.0 * torch.sigmoid(H_raw_post)                           # (P, n) ∈ (0,2)
        H_res = sinkhorn_knopp(H_raw_res.view(-1, self.n, self.n), self.sinkhorn_iters)

        return H_pre, H_post, H_res

    def read(self, x_stream: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Read from the expanded stream: extract sub-layer input.

        Args:
            x_stream: (B, ..., n·C) expanded residual stream.

        Returns:
            layer_input: (B, ..., C) aggregated input for the sub-layer.
            ctx: dict with H_pre, H_res, H_post for the write phase.
        """
        lead_shape = x_stream.shape[:-1]
        C = self.dim
        n = self.n

        x_flat = x_stream.reshape(-1, n * C)
        H_pre, H_post, H_res = self._compute_mappings(x_flat)

        # layer_input = H_pre @ x_stream_mat  →  (P, C)
        x_mat = x_flat.view(-1, n, C)          # (P, n, C)
        layer_input = torch.einsum("pn,pnc->pc", H_pre, x_mat)  # (P, C)
        layer_input = layer_input.view(*lead_shape, C)

        ctx = {"H_pre": H_pre, "H_post": H_post, "H_res": H_res}
        return layer_input, ctx

    def write(
        self, x_stream: torch.Tensor, sublayer_output: torch.Tensor, ctx: dict
    ) -> torch.Tensor:
        """Write sub-layer output back to the stream.

        Args:
            x_stream: (B, ..., n·C) current stream (same as passed to read).
            sublayer_output: (B, ..., C) output from the sub-layer.
            ctx: dict from ``read()``.

        Returns:
            new_stream: (B, ..., n·C) updated residual stream.
        """
        lead_shape = x_stream.shape[:-1]
        C = self.dim
        n = self.n

        x_flat = x_stream.reshape(-1, n * C)
        x_mat = x_flat.view(-1, n, C)
        sub_flat = sublayer_output.reshape(-1, C)

        H_res = ctx["H_res"]    # (P, n, n)
        H_post = ctx["H_post"]  # (P, n)

        # x_next = H_res @ x_stream + H_post^T ⊗ sublayer_output
        x_res = torch.bmm(H_res, x_mat)                                  # (P, n, C)
        x_write = H_post.unsqueeze(-1) * sub_flat.unsqueeze(1)           # (P, n, C)
        x_next = x_res + x_write

        return x_next.view(*lead_shape, n * C)


# ---------------------------------------------------------------------------
# Helper: expand / contract n-stream
# ---------------------------------------------------------------------------

def _expand_stream(x: torch.Tensor, n: int) -> torch.Tensor:
    """(B, ..., C) → (B, ..., n·C) by repeating along new dim."""
    C = x.shape[-1]
    return x.unsqueeze(-2).expand(*x.shape[:-1], n, C).reshape(*x.shape[:-1], n * C)


def _contract_stream(x_stream: torch.Tensor, n: int) -> torch.Tensor:
    """(B, ..., n·C) → (B, ..., C) by averaging over streams."""
    C = x_stream.shape[-1] // n
    return x_stream.reshape(*x_stream.shape[:-1], n, C).mean(dim=-2)


# ---------------------------------------------------------------------------
# SwinBlock (encoder)
# ---------------------------------------------------------------------------

class SwinBlock(nn.Module):
    """Encoder block: KimiLinear attn + SwiGLU FFN + mHC.

    Args:
        dim: Token dimension C.
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size (0 or ws//2).
        mlp_ratio: Kept for API compat (SwiGLU manages its own hidden size).
        drop_path: DropPath rate.
        mhc_expansion: mHC stream count.
        mhc_sinkhorn_iters: Sinkhorn iterations.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        mhc_expansion: int = 4,
        mhc_sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.dim = dim
        self.n = mhc_expansion
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = RMSNorm(dim)
        self.attn = KimiLinearAttention(dim, num_heads, window_size)
        self.drop_path = DropPath(drop_path)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUFFN(dim)

        self.mhc_attn = mHCConnection(dim, mhc_expansion, mhc_sinkhorn_iters)
        self.mhc_ffn = mHCConnection(dim, mhc_expansion, mhc_sinkhorn_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C) → (B, H, W, C)"""
        B, H, W, C = x.shape

        # Expand to n-stream
        xs = _expand_stream(x, self.n)  # (B, H, W, nC)

        # --- Attention sub-layer ---
        layer_in, ctx = self.mhc_attn.read(xs)       # (B, H, W, C)
        normed = self.norm1(layer_in)

        # Cyclic shift
        shifted = normed
        if self.shift_size > 0:
            shifted = torch.roll(normed, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Window partition → attention → merge
        ws = self.window_size
        x_win = window_partition(shifted, ws).view(-1, ws * ws, C)
        attn_out = self.attn(x_win)
        attn_out = window_reverse(
            attn_out.view(-1, ws, ws, C), ws, H, W,
        )

        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        attn_out = self.drop_path(attn_out)
        xs = self.mhc_attn.write(xs, attn_out, ctx)

        # --- FFN sub-layer ---
        layer_in, ctx = self.mhc_ffn.read(xs)
        ffn_out = self.drop_path(self.mlp(self.norm2(layer_in)))
        xs = self.mhc_ffn.write(xs, ffn_out, ctx)

        return _contract_stream(xs, self.n)


# ---------------------------------------------------------------------------
# AdaLNSwinBlock (decoder)
# ---------------------------------------------------------------------------

class AdaLNSwinBlock(nn.Module):
    """Decoder block with AdaLN: KimiLinear attn + SwiGLU + mHC.

    AdaLN modulation: cond → 6·C  (s1, sh1, g1, s2, sh2, g2).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        mhc_expansion: int = 4,
        mhc_sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.dim = dim
        self.n = mhc_expansion
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.attn = KimiLinearAttention(dim, num_heads, window_size)
        self.drop_path = DropPath(drop_path)
        self.norm2 = RMSNorm(dim, elementwise_affine=False)
        self.mlp = SwiGLUFFN(dim)

        # AdaLN: cond → 6·C
        self.adaLN_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.constant_(self.adaLN_mlp[-1].weight, 0.0)
        nn.init.constant_(self.adaLN_mlp[-1].bias, 0.0)

        self.mhc_attn = mHCConnection(dim, mhc_expansion, mhc_sinkhorn_iters)
        self.mhc_ffn = mHCConnection(dim, mhc_expansion, mhc_sinkhorn_iters)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        # AdaLN params
        p = self.adaLN_mlp(cond)  # (B, 6C)
        s1, sh1, g1, s2, sh2, g2 = p.chunk(6, dim=-1)
        for t in (s1, sh1, g1, s2, sh2, g2):
            # make (B,1,1,C) for broadcasting
            pass
        s1  = s1.unsqueeze(1).unsqueeze(2)
        sh1 = sh1.unsqueeze(1).unsqueeze(2)
        g1  = g1.unsqueeze(1).unsqueeze(2)
        s2  = s2.unsqueeze(1).unsqueeze(2)
        sh2 = sh2.unsqueeze(1).unsqueeze(2)
        g2  = g2.unsqueeze(1).unsqueeze(2)

        xs = _expand_stream(x, self.n)

        # --- Attention ---
        layer_in, ctx = self.mhc_attn.read(xs)
        normed = self.norm1(layer_in) * (1 + s1) + sh1

        shifted = normed
        if self.shift_size > 0:
            shifted = torch.roll(normed, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        ws = self.window_size
        x_win = window_partition(shifted, ws).view(-1, ws * ws, C)
        attn_out = self.attn(x_win)
        attn_out = window_reverse(attn_out.view(-1, ws, ws, C), ws, H, W)

        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        attn_out = g1 * self.drop_path(attn_out)
        xs = self.mhc_attn.write(xs, attn_out, ctx)

        # --- FFN ---
        layer_in, ctx = self.mhc_ffn.read(xs)
        normed2 = self.norm2(layer_in) * (1 + s2) + sh2
        ffn_out = g2 * self.drop_path(self.mlp(normed2))
        xs = self.mhc_ffn.write(xs, ffn_out, ctx)

        return _contract_stream(xs, self.n)


# ---------------------------------------------------------------------------
# PatchEmbed / PatchMerge
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image → patch embedding via Conv2d."""

    def __init__(self, in_channels: int = 3, patch_size: int = 4, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)


class PatchMerge(nn.Module):
    """Merge 2×2 patches → halve spatial, double channels."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H // 2, W // 2, 4 * C)
        return self.reduction(self.norm(x))


# ---------------------------------------------------------------------------
# CrossAttnAdaLNSwinBlock — full decoder block
# ---------------------------------------------------------------------------

class CrossAttnAdaLNSwinBlock(nn.Module):
    """Decoder block: Kimi Linear self-attn + cross-attn + SwiGLU FFN.

    AdaLN: cond → 9·C  (3 groups of s/sh/g for self-attn, cross-attn, FFN).
    mHC: three independent mHC connections.
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
        mhc_expansion: int = 4,
        mhc_sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.head_dim = dim // num_heads
        self.n = mhc_expansion

        if z_dim is None:
            z_dim = dim
        self.z_dim = z_dim

        # Self-attention (windowed)
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.attn = KimiLinearAttention(dim, num_heads, window_size)
        self.drop_path = DropPath(drop_path)

        # Cross-attention (non-windowed)
        self.cross_norm = RMSNorm(dim)
        self.cross_attn = KimiLinearCrossAttention(dim, z_dim, num_heads)

        # FFN
        self.norm2 = RMSNorm(dim, elementwise_affine=False)
        self.mlp = SwiGLUFFN(dim)

        # AdaLN: cond → 9·C
        self.adaLN_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.constant_(self.adaLN_mlp[-1].weight, 0.0)
        nn.init.constant_(self.adaLN_mlp[-1].bias, 0.0)

        # mHC
        self.mhc_self = mHCConnection(dim, mhc_expansion, mhc_sinkhorn_iters)
        self.mhc_cross = mHCConnection(dim, mhc_expansion, mhc_sinkhorn_iters)
        self.mhc_ffn = mHCConnection(dim, mhc_expansion, mhc_sinkhorn_iters)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, z_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (B, H, W, C)   cond: (B, C)   z_tokens: (B, Hz, Wz, C)
        """
        B, H, W, C = x.shape
        N = H * W

        # AdaLN params
        p = self.adaLN_mlp(cond)  # (B, 9C)
        s1, sh1, g1, s2, sh2, g2, s3, sh3, g3 = p.chunk(9, dim=-1)
        s1  = s1.unsqueeze(1).unsqueeze(2)
        sh1 = sh1.unsqueeze(1).unsqueeze(2)
        g1  = g1.unsqueeze(1).unsqueeze(2)
        s2  = s2.unsqueeze(1).unsqueeze(2)
        sh2 = sh2.unsqueeze(1).unsqueeze(2)
        g2  = g2.unsqueeze(1).unsqueeze(2)
        s3  = s3.unsqueeze(1).unsqueeze(2)
        sh3 = sh3.unsqueeze(1).unsqueeze(2)
        g3  = g3.unsqueeze(1).unsqueeze(2)

        xs = _expand_stream(x, self.n)

        # --- 1. Self-attention (windowed) ---
        layer_in, ctx = self.mhc_self.read(xs)
        normed = self.norm1(layer_in) * (1 + s1) + sh1

        shifted = normed
        if self.shift_size > 0:
            shifted = torch.roll(normed, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        ws = self.window_size
        x_win = window_partition(shifted, ws).view(-1, ws * ws, C)
        attn_out = self.attn(x_win)
        attn_out = window_reverse(attn_out.view(-1, ws, ws, C), ws, H, W)

        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        self_out = g1 * self.drop_path(attn_out)
        xs = self.mhc_self.write(xs, self_out, ctx)

        # --- 2. Cross-attention (non-windowed) ---
        layer_in, ctx = self.mhc_cross.read(xs)
        x_flat = self.cross_norm(layer_in.reshape(B, N, C))
        z_flat = z_tokens.reshape(B, -1, C)
        cross_out = self.cross_attn(x_flat, z_flat).reshape(B, H, W, C)
        cross_out = g2 * cross_out
        xs = self.mhc_cross.write(xs, cross_out, ctx)

        # --- 3. FFN ---
        layer_in, ctx = self.mhc_ffn.read(xs)
        normed2 = self.norm2(layer_in) * (1 + s3) + sh3
        ffn_out = g3 * self.drop_path(self.mlp(normed2))
        xs = self.mhc_ffn.write(xs, ffn_out, ctx)

        return _contract_stream(xs, self.n)
