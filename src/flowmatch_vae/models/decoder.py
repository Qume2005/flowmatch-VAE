"""U-Net Multi-Scale OT-CFM Velocity Network.

Decoder operates at 5 spatial levels (16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1)
with DiT blocks at each level. Down path uses AttnPool2x2, up path uses
Upsample2x. Skip connections via addition. Each level cross-attends to
the VAE encoder's corresponding scale features.

Architecture:
    x_t -> PatchEmbed -> (B, 16, 16, C)
    Down:  Level 0 (16x16) -> Level 1 (8x8) -> ... -> Level 4 (1x1)
    Up:    Level 4 (1x1) -> Level 3 (2x2) -> ... -> Level 0 (16x16)
    Output: RMSNorm -> Linear -> pixel_shuffle -> v_pred
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmatch_vae.models.swin import PatchEmbed, CrossAttnAdaLNSwinBlock, RMSNorm
from flowmatch_vae.models.conv_encoder import AttnPool2x2, Upsample2x, SwiGLUConv
from flowmatch_vae.models.moe import MoEConvLayer, MoEDiTLayer


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding: scalar t -> vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


class FlowDecoder(nn.Module):
    """U-Net multi-scale OT-CFM velocity network.

    Args:
        cfg: MultiScaleDecoderConfig dataclass.
        mhc_cfg: Optional mHC config for DiT blocks.
        vae_scale_map: List mapping each decoder level to a VAE encoder scale
            index.  Defaults to [1, 2, 3, 4, 5] for the standard 64x64 layout.
        encoder_dilations: Dilation rates from the encoder's shared convs.
            Required when cfg.moe is set with moe_conv=True.
    """

    def __init__(self, cfg, mhc_cfg=None, vae_scale_map: list[int] | None = None, latent_vae_scale: int = 2,
                 shared_convs: nn.ModuleList | None = None, shared_pool: AttnPool2x2 | None = None,
                 encoder_dilations: tuple[int, ...] | None = None):
        super().__init__()
        # Normalize legacy DecoderConfig -> MultiScaleDecoderConfig
        from flowmatch_vae.config import DecoderConfig, MultiScaleDecoderConfig
        if isinstance(cfg, DecoderConfig) and not isinstance(cfg, MultiScaleDecoderConfig):
            cfg = MultiScaleDecoderConfig(
                out_channels=cfg.out_channels,
                patch_size=cfg.patch_size,
                embed_dim=cfg.embed_dim,
                blocks_down=(cfg.depth // 2, cfg.depth // 2),
                blocks_up=(cfg.depth // 4, cfg.depth // 4),
                num_heads=cfg.num_heads,
                window_size=cfg.window_size,
                latent_spatial_size=cfg.latent_spatial_size,
                latent_dim=cfg.latent_dim,
            )
        self.cfg = cfg
        C = cfg.embed_dim
        ps = cfg.patch_size

        # mHC config
        mhc_expansion = getattr(mhc_cfg, "expansion_rate", 4) if mhc_cfg else 4
        mhc_sinkhorn_iters = getattr(mhc_cfg, "sinkhorn_iters", 20) if mhc_cfg else 20

        # Patch embedding for x_t
        self.patch_embed = PatchEmbed(
            in_channels=cfg.out_channels,
            patch_size=ps,
            embed_dim=C,
        )

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(C)

        # Shared modules from encoder
        self.shared_convs = shared_convs  # ModuleList of SwiGLUConv from encoder
        self.shared_pool = shared_pool    # AttnPool2x2 from encoder

        # Decoder levels (from fine to coarse): 16x16, 8x8, 4x4, 2x2, 1x1
        # Map to VAE encoder scale indices
        self.vae_scale_map = vae_scale_map or [1, 2, 3, 4, 5]
        # Which VAE encoder scale corresponds to the latent z
        self._latent_vae_scale = latent_vae_scale
        n_levels = len(cfg.blocks_down)

        # Fallback: create separate pools if shared_pool is not provided
        if self.shared_pool is None:
            self.down_pools = nn.ModuleList()
            for level in range(n_levels - 1):
                self.down_pools.append(AttnPool2x2(C))

        # MoE config
        moe_cfg = getattr(cfg, "moe", None)
        self._use_moe = moe_cfg is not None

        # --- Down path: standard blocks (always created for backward compat) ---
        self.down_blocks = nn.ModuleList()
        for level in range(n_levels):
            n_blocks = cfg.blocks_down[level]
            ws = cfg.window_size
            level_blocks = nn.ModuleList()
            for j in range(n_blocks):
                shift = 0 if j % 2 == 0 else ws // 2
                level_blocks.append(CrossAttnAdaLNSwinBlock(
                    dim=C,
                    num_heads=cfg.num_heads,
                    window_size=ws,
                    shift_size=shift,
                    mhc_expansion=mhc_expansion,
                    mhc_sinkhorn_iters=mhc_sinkhorn_iters,
                ))
            self.down_blocks.append(level_blocks)

        # --- Down path: MoE conv layers ---
        if self._use_moe and moe_cfg.moe_conv:
            self.moe_conv_layers = nn.ModuleList()
            dilations = encoder_dilations or (1,)
            for i in range(len(self.shared_convs) if self.shared_convs else 0):
                d = dilations[i % len(dilations)]
                # Read kernel_size from the shared conv
                ks = self.shared_convs[i].kernel_size
                self.moe_conv_layers.append(MoEConvLayer(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=ks,
                    dilation=d,
                    num_experts=moe_cfg.num_experts,
                    include_zero_expert=moe_cfg.include_zero_expert,
                    gate_hidden_dim=moe_cfg.gate_hidden_dim,
                    routing_mode=moe_cfg.routing_mode,
                    prob_threshold=moe_cfg.prob_threshold,
                    top_k=moe_cfg.top_k,
                ))
        else:
            self.moe_conv_layers = None

        # --- Down path: MoE DiT blocks ---
        if self._use_moe and moe_cfg.moe_dit:
            self.moe_down_blocks = nn.ModuleList()
            for level in range(n_levels):
                n_blocks = cfg.blocks_down[level]
                ws = cfg.window_size
                level_blocks = nn.ModuleList()
                for j in range(n_blocks):
                    shift = 0 if j % 2 == 0 else ws // 2
                    level_blocks.append(MoEDiTLayer(
                        dim=C,
                        num_heads=cfg.num_heads,
                        window_size=ws,
                        shift_size=shift,
                        num_experts=moe_cfg.num_experts,
                        include_zero_expert=moe_cfg.include_zero_expert,
                        gate_hidden_dim=moe_cfg.gate_hidden_dim,
                        routing_mode=moe_cfg.routing_mode,
                        prob_threshold=moe_cfg.prob_threshold,
                        top_k=moe_cfg.top_k,
                        z_dim=C,
                        mhc_expansion=mhc_expansion,
                        mhc_sinkhorn_iters=mhc_sinkhorn_iters,
                    ))
                self.moe_down_blocks.append(level_blocks)
        else:
            self.moe_down_blocks = None

        # --- Up path ---
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level in range(len(cfg.blocks_up)):
            # Use first shared conv (dilation=1) for Upsample2x refinement
            up_conv = self.shared_convs[0] if self.shared_convs is not None else None
            self.up_samples.append(Upsample2x(C, shared_conv=up_conv))
            n_blocks = cfg.blocks_up[level]
            ws = cfg.window_size
            level_blocks = nn.ModuleList()
            for j in range(n_blocks):
                shift = 0 if j % 2 == 0 else ws // 2
                level_blocks.append(CrossAttnAdaLNSwinBlock(
                    dim=C,
                    num_heads=cfg.num_heads,
                    window_size=ws,
                    shift_size=shift,
                    mhc_expansion=mhc_expansion,
                    mhc_sinkhorn_iters=mhc_sinkhorn_iters,
                ))
            self.up_blocks.append(level_blocks)

        # Output
        self.out_norm = RMSNorm(C)
        self.out_proj = nn.Linear(C, ps * ps * cfg.out_channels)

    def _get_vae_tokens(
        self,
        scale_tokens: dict[int, torch.Tensor] | None,
        vae_scale: int,
        z: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        """Get VAE tokens for cross-attention at the given scale.

        For the latent scale (typically 8x8), returns z flattened.
        For other scales, returns from scale_tokens dict.
        Falls back to z flattened if scale_tokens is None or scale missing.
        """
        C = z.shape[-1]
        if vae_scale == self._latent_vae_scale:
            return z.reshape(B, -1, C)
        if scale_tokens is not None and vae_scale in scale_tokens:
            return scale_tokens[vae_scale]
        # Fallback: use z
        return z.reshape(B, -1, C)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        scale_tokens: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass through U-Net decoder.

        Args:
            x_t: (B, 3, 64, 64) noisy image.
            t: (B,) timestep.
            z: (B, 8, 8, latent_dim) latent variable.
            scale_tokens: dict mapping VAE scale index -> (B, N_s, C) tokens.

        Returns:
            (B, 3, 64, 64) predicted velocity field.
        """
        B = x_t.shape[0]

        # Patch embed x_t -> (B, 16, 16, C)
        h = self.patch_embed(x_t)

        # Time embedding
        t_emb = self.time_embed(t)

        # === Down path ===
        skips: list[torch.Tensor] = []
        use_moe_conv = self.moe_conv_layers is not None
        use_moe_dit = self.moe_down_blocks is not None

        for level in range(len(self.down_blocks)):
            vae_scale = self.vae_scale_map[level]
            vae_toks = self._get_vae_tokens(scale_tokens, vae_scale, z, B)

            # Apply convs: MoE or shared
            h_bchw = h.permute(0, 3, 1, 2)              # (B, C, H, W)
            if use_moe_conv:
                for moe_conv in self.moe_conv_layers:
                    h_bchw = moe_conv(h_bchw, t_emb)
            elif self.shared_convs is not None:
                for conv in self.shared_convs:
                    h_bchw = conv(h_bchw)
            h = h_bchw.permute(0, 2, 3, 1)               # (B, H, W, C)

            # Apply DiT blocks: MoE or standard
            blocks = self.moe_down_blocks[level] if use_moe_dit else self.down_blocks[level]
            for block in blocks:
                h = block(h, t_emb, vae_toks)

            skips.append(h)

            if level < len(self.down_blocks) - 1:
                h = h.permute(0, 3, 1, 2)              # (B, C, H, W)
                if self.shared_pool is not None:
                    h = self.shared_pool(h)              # (B, C, H/2, W/2)
                else:
                    h = self.down_pools[level](h)        # (B, C, H/2, W/2)
                h = h.permute(0, 2, 3, 1)               # (B, H/2, W/2, C)

        # === Up path ===
        for up_level, blocks in enumerate(self.up_blocks):
            down_level = len(self.down_blocks) - 2 - up_level
            vae_scale = self.vae_scale_map[down_level]

            h = h.permute(0, 3, 1, 2)              # (B, C, H, W)
            h = self.up_samples[up_level](h)        # (B, C, H*2, W*2)
            h = h.permute(0, 2, 3, 1)              # (B, H*2, W*2, C)

            # Skip connection (addition)
            h = h + skips[down_level]

            # DiT blocks with cross-attention
            vae_toks = self._get_vae_tokens(scale_tokens, vae_scale, z, B)
            for block in blocks:
                h = block(h, t_emb, vae_toks)

        # === Output ===
        h = self.out_norm(h)
        h = self.out_proj(h)

        # Reshape to image via pixel_shuffle
        ps = self.cfg.patch_size
        h = h.permute(0, 3, 1, 2)
        h = F.pixel_shuffle(h, ps)

        return h
