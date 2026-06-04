"""Multi-Scale SwiGLU Convolution Encoder.

Replaces the Swin Transformer encoder with a progressive multi-scale
convolution architecture:
- SwiGLU-gated depthwise separable convolutions with dilation
- mHC-inspired attention pooling (AttnPool2x2)
- 2D PoPE (Legendre Orthogonal Polynomial) positional encoding for multi-scale tokens
- Linear self-attention fusion across scales
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmatch_vae.models.swin import PoPE2D


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
            ).to(memory_format=torch.channels_last)
            self.up_dw = nn.Conv2d(
                in_channels, in_channels, kernel_size,
                padding=pad, dilation=dilation, groups=in_channels,
            ).to(memory_format=torch.channels_last)
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
# FusionAttention — linear self-attention with 2D PoPE
# ---------------------------------------------------------------------------

class FusionAttention(nn.Module):
    """Bidirectional linear attention with 2D PoPE for multi-scale fusion.

    Simple linear attention (no DW conv, no forget gate) — O(N*d^2) complexity
    suitable for fusing ~1365 multi-scale tokens.

    PoPE (Legendre Orthogonal Polynomial Positional Encoding) is applied
    additively to the input before QKV projection, not multiplicatively into
    Q/K after projection like RoPE.
    """

    def __init__(self, dim: int, num_heads: int = 8, max_h: int = 32, max_w: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # 2D PoPE: additive positional encoding applied before QKV projection
        self.pope = PoPE2D(dim, max_h, max_w)

    def forward(
        self,
        x: torch.Tensor,
        spatial_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        x: (B, N, C)  — concatenated multi-scale tokens
        spatial_sizes: [(h0, w0), (h1, w1), ...] per scale

        Returns: (B, N, C)
        """
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        # Add PoPE positional encoding (additive, before QKV projection)
        # We apply it to the full token sequence in one shot by building the
        # combined spatial PE for all scales.
        x = self._add_multi_scale_pope(x, spatial_sizes)

        q = self.q_proj(x).view(B, N, H, D)
        k = self.k_proj(x).view(B, N, H, D)
        v = self.v_proj(x).view(B, N, H, D)

        # L2 normalise
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # Bidirectional linear attention: S = K^T V,  O = Q S
        S = torch.einsum("bthd,bthe->bhde", k, v)   # (B, H, D, D)
        o = torch.einsum("bnhd,bhde->bnhe", q, S)    # (B, H, N, D)

        # Normalise per head
        o = o / (o.norm(dim=-1, keepdim=True) + 1e-6) * (D ** 0.5)
        return self.out_proj(o.reshape(B, N, C))

    def _add_multi_scale_pope(
        self,
        tokens: torch.Tensor,
        spatial_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Add 2D PoPE to a concatenated multi-scale token sequence.

        For each scale, compute the PoPE encoding for its (h, w) spatial
        grid and apply it to the corresponding slice of the token sequence.

        Args:
            tokens: (B, N_total, C) concatenated tokens.
            spatial_sizes: [(h0, w0), ...] per scale.

        Returns:
            (B, N_total, C) with PoPE added.
        """
        B, N_total, C = tokens.shape
        half = C // 2

        out = tokens
        offset = 0
        for h, w in spatial_sizes:
            n = h * w
            # Get scale slice
            scale_slice = out[:, offset:offset + n, :]

            # Build PE for this scale
            rows = torch.arange(h, device=tokens.device)
            cols = torch.arange(w, device=tokens.device)

            pe_y = self.pope.pe_h[rows]  # (h, half)
            pe_x = self.pope.pe_w[cols]  # (w, half)

            pe_y_exp = pe_y.unsqueeze(1).expand(h, w, half)  # (h, w, half)
            pe_x_exp = pe_x.unsqueeze(0).expand(h, w, half)  # (h, w, half)
            pe_2d = torch.cat([pe_y_exp, pe_x_exp], dim=-1)  # (h, w, C)
            pe_flat = pe_2d.reshape(n, C)  # (n, C)

            # Add PE to this scale's tokens
            out = torch.cat([
                out[:, :offset, :],
                scale_slice + pe_flat.unsqueeze(0),
                out[:, offset + n:, :],
            ], dim=1)

            offset += n

        return out


# ---------------------------------------------------------------------------
# Multi-Scale SwiGLU Convolution Encoder
# ---------------------------------------------------------------------------

class MultiScaleConvEncoder(nn.Module):
    """Multi-scale SwiGLU convolution encoder with attention pooling.

    Adaptive parameter-sharing architecture: a fixed set of SwiGLUConv blocks
    and a single AttnPool2x2 are reused at every stage.  The number of stages
    is computed dynamically from the input spatial size (halving until 1x1).

    Architecture:
        Image (B, 3, H, W)
          -> stem Conv1x1(3, embed_dim)
          -> n_stages stages: shared SwiGLUConv blocks -> shared AttnPool2x2
            (H -> H/2 -> ... -> 1)
          -> collect features from each scale
          -> add scale embeddings + 2D PoPE
          -> FusionAttention -> extract latent-scale tokens
          -> mu_head, logvar_head -> (B, latent_spatial, latent_spatial, C)

    Args:
        cfg: MultiScaleEncoderConfig dataclass.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        C = cfg.embed_dim

        # Maximum number of scales for scale_embed and PoPE buffers
        max_scales = int(math.log2(cfg.max_input_size))
        self.max_scales = max_scales

        # Stem: expand channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, C, kernel_size=1),
            RMSNorm2d(C),
        )

        # Shared SwiGLU conv blocks (parameter sharing across stages)
        self.conv_blocks = nn.ModuleList([
            SwiGLUConv(C, C, kernel_size=cfg.kernel_size, dilation=cfg.dilations[j % len(cfg.dilations)])
            for j in range(cfg.num_conv_blocks)
        ])

        # Shared attention pool (single instance, reused at every stage)
        self.pool = AttnPool2x2(C)

        # Learnable scale embeddings (sized for max possible scales)
        self.scale_embed = nn.Parameter(torch.randn(max_scales, C) * 0.02)

        # Fusion attention (max_h/max_w cover largest scale after first pool)
        self.fusion = FusionAttention(
            C, cfg.fusion_heads,
            max_h=cfg.max_input_size // 2,
            max_w=cfg.max_input_size // 2,
        )

        # Output heads
        self.mu_head = nn.Linear(C, C)
        self.logvar_head = nn.Linear(C, C)

    @staticmethod
    def _compute_num_stages(spatial_size: int) -> int:
        """Compute the number of 2x2-pooling stages to go from spatial_size to 1."""
        n = 0
        s = spatial_size
        while s > 1:
            s //= 2
            n += 1
        return n

    @staticmethod
    def _find_latent_scale(spatial_sizes: list[tuple[int, int]], latent_spatial_size: int) -> int:
        """Find which scale index has spatial size == latent_spatial_size.

        Args:
            spatial_sizes: list of (H, W) tuples for each scale.
            latent_spatial_size: target spatial size (e.g. 8).

        Returns:
            Scale index whose spatial dimensions match latent_spatial_size.

        Raises:
            ValueError: if no scale matches the target size.
        """
        for idx, (h, w) in enumerate(spatial_sizes):
            if h == latent_spatial_size and w == latent_spatial_size:
                return idx
        raise ValueError(
            f"No scale with spatial size {latent_spatial_size} found in "
            f"{[s for s in spatial_sizes]}"
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        """x: (B, 3, H, W) -> mu: (B, latent_s, latent_s, C), logvar, per_scale_tokens"""
        B = x.shape[0]
        C = self.cfg.embed_dim
        H_in, W_in = x.shape[2], x.shape[3]

        h = self.stem(x)  # (B, C, H_in, W_in)

        # Compute number of stages dynamically from input spatial size
        n_stages = self._compute_num_stages(H_in)
        assert n_stages <= self.max_scales, (
            f"Input spatial size {H_in} requires {n_stages} stages, "
            f"but max_scales={self.max_scales} (max_input_size={self.cfg.max_input_size})"
        )

        # Run stages, collect features at each scale
        scale_features: list[torch.Tensor] = []
        spatial_sizes: list[tuple[int, int]] = []

        for i in range(n_stages):
            # Apply shared conv blocks sequentially
            for block in self.conv_blocks:
                h = block(h)

            # Apply shared attention pool
            h = self.pool(h)  # (B, C, H//2, W//2)

            # Record features (channels-last for token sequence)
            _, _, Hi, Wi = h.shape
            spatial_sizes.append((Hi, Wi))
            scale_features.append(h.permute(0, 2, 3, 1).reshape(B, Hi * Wi, C))

        # Build multi-scale token sequence with scale embeddings
        for s in range(n_stages):
            scale_features[s] = scale_features[s] + self.scale_embed[s]

        all_tokens = torch.cat(scale_features, dim=1)  # (B, total_N, C)

        # Build scale lengths for extracting per-scale tokens later
        scale_lengths = [hh * ww for hh, ww in spatial_sizes]

        # Fusion self-attention (with 2D PoPE)
        fused = self.fusion(all_tokens, spatial_sizes)  # (B, total_N, C)

        # Extract per-scale tokens from fused sequence
        per_scale_tokens: dict[int, torch.Tensor] = {}
        offset = 0
        for s_idx, slen in enumerate(scale_lengths):
            per_scale_tokens[s_idx] = fused[:, offset:offset + slen, :]
            offset += slen

        # Find latent scale dynamically and extract mu/logvar
        latent_scale = self._find_latent_scale(spatial_sizes, self.cfg.latent_spatial_size)
        latent_tokens = per_scale_tokens[latent_scale]
        Hi, Wi = spatial_sizes[latent_scale]
        latent_tokens = latent_tokens.reshape(B, Hi, Wi, C)

        mu = self.mu_head(latent_tokens)
        logvar = self.logvar_head(latent_tokens)
        return mu, logvar, per_scale_tokens


# ---------------------------------------------------------------------------
# Upsample2x — nearest upsample + SwiGLUConv refinement
# ---------------------------------------------------------------------------

class Upsample2x(nn.Module):
    """Nearest-neighbor 2x upsample followed by SwiGLUConv refinement."""

    def __init__(self, dim: int, shared_conv: SwiGLUConv | None = None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        if shared_conv is not None:
            self.refine = shared_conv
        else:
            self.refine = SwiGLUConv(dim, dim, kernel_size=3, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, C, H*2, W*2)"""
        return self.refine(self.up(x))


# ---------------------------------------------------------------------------
# MultiScalePrior — predict multi-scale features from z
# ---------------------------------------------------------------------------

class MultiScalePrior(nn.Module):
    """Predict multi-scale VAE encoder features from z for generation.

    Takes z at ``latent_spatial_size`` resolution and predicts features at all
    *other* scales in the encoder's hierarchy.  Scales larger than the latent
    are reached via ConvTranspose2d (2x upsample), scales smaller via Conv2d
    stride=2 (2x pool).  Each prediction path includes RMSNorm + SwiGLUConv.

    Args:
        dim: Channel dimension.
        latent_spatial_size: Spatial size of the latent z (e.g. 8 for 8x8).
        spatial_sizes: Full list of (H, W) spatial sizes produced by the
            encoder, in order (largest to smallest).  If *None*, defaults to
            the 64x64-input layout: [(32,32), (16,16), (8,8), (4,4), (2,2), (1,1)].
    """

    def __init__(
        self,
        dim: int = 256,
        latent_spatial_size: int = 8,
        spatial_sizes: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.latent_spatial_size = latent_spatial_size

        # Default to 64x64 -> 6 scales layout
        if spatial_sizes is None:
            spatial_sizes = [(32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
        self.spatial_sizes = spatial_sizes

        # Identify the latent scale index and build per-scale prediction paths
        self.latent_scale_idx = None
        self._paths = nn.ModuleDict()

        for idx, (h, w) in enumerate(spatial_sizes):
            if h == latent_spatial_size and w == latent_spatial_size:
                self.latent_scale_idx = idx
                continue  # z itself, no prediction needed

            name = f"scale_{idx}"
            if h > latent_spatial_size:
                # UP path: one or more ConvTranspose2d 2x upsamples
                n_ups = int(math.log2(h // latent_spatial_size))
                layers = []
                for _ in range(n_ups):
                    layers.append(nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1))
                    layers.append(RMSNorm2d(dim))
                    layers.append(SwiGLUConv(dim, dim, kernel_size=3, dilation=1))
                self._paths[name] = nn.Sequential(*layers)
            else:
                # DOWN path: one or more Conv2d stride=2 pools
                n_downs = int(math.log2(latent_spatial_size // h))
                layers = []
                for d in range(n_downs):
                    layers.append(nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1))
                    layers.append(RMSNorm2d(dim))
                self._paths[name] = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> dict[int, torch.Tensor]:
        """z: (B, latent_s, latent_s, C) in channels-last format.

        Returns dict mapping scale index -> (B, N_s, C) tokens for all
        scales except the latent scale.
        """
        z_bchw = z.permute(0, 3, 1, 2)  # (B, C, latent_s, latent_s)

        result: dict[int, torch.Tensor] = {}

        def to_tokens(feat: torch.Tensor) -> torch.Tensor:
            B, C, H, W = feat.shape
            return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        for idx, (h, w) in enumerate(self.spatial_sizes):
            if h == self.latent_spatial_size and w == self.latent_spatial_size:
                continue
            name = f"scale_{idx}"
            feat = self._paths[name](z_bchw)
            result[idx] = to_tokens(feat)

        return result
