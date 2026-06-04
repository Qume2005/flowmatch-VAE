"""OT-CFM 速度场网络：条件 Swin 架构，预测 velocity field。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from flowmatch_vae.config import DecoderConfig
from flowmatch_vae.models.swin import PatchEmbed, CrossAttnAdaLNSwinBlock, RMSNorm


class SinusoidalTimeEmbedding(nn.Module):
    """将标量时间 t 编码为向量。"""

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
    """OT-CFM velocity network: v_θ(x_t, t, z) -> velocity field。

    架构:
    - x_t 通过 PatchEmbed 编码为 tokens
    - z 上采样后通过 cross-attention 注入
    - t 通过 sinusoidal embedding + MLP -> adaLN conditioning
    - N 个 CrossAttnAdaLNSwinBlock 处理
    - Linear 输出头 -> velocity
    """

    def __init__(self, cfg: DecoderConfig, mhc_cfg=None):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size

        # mHC config
        mhc_expansion = getattr(mhc_cfg, "expansion_rate", 4) if mhc_cfg else 4
        mhc_sinkhorn_iters = getattr(mhc_cfg, "sinkhorn_iters", 20) if mhc_cfg else 20

        self.patch_embed = PatchEmbed(
            in_channels=cfg.out_channels,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
        )

        self.time_embed = SinusoidalTimeEmbedding(cfg.embed_dim)

        self.z_proj = nn.Linear(cfg.latent_dim, cfg.embed_dim)
        self.z_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.blocks = nn.ModuleList([
            CrossAttnAdaLNSwinBlock(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                window_size=cfg.window_size,
                shift_size=0 if (i % 2 == 0) else cfg.window_size // 2,
                mlp_ratio=cfg.mlp_ratio,
                mhc_expansion=mhc_expansion,
                mhc_sinkhorn_iters=mhc_sinkhorn_iters,
            )
            for i in range(cfg.depth)
        ])

        self.out_norm = RMSNorm(cfg.embed_dim)
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.patch_size * cfg.patch_size * cfg.out_channels)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, 3, 64, 64) 噪声图
        t:   (B,) 时间
        z:   (B, 8, 8, latent_dim) 潜在向量 (空间式)
        返回: (B, 3, 64, 64) 速度场
        """
        B = x_t.shape[0]

        # Patch embed x_t -> (B, 16, 16, embed_dim)
        h = self.patch_embed(x_t)

        # z 投影 + 上采样 -> (B, 16, 16, embed_dim)
        z_proj = self.z_proj(z)
        z_proj = z_proj.permute(0, 3, 1, 2)
        z_proj = self.z_upsample(z_proj)
        z_proj = z_proj.permute(0, 2, 3, 1)

        t_emb = self.time_embed(t)

        for block in self.blocks:
            h = block(h, t_emb, z_proj)

        h = self.out_norm(h)
        h = self.out_proj(h)

        # Reshape to image via pixel_shuffle
        ps = self.patch_size
        h = h.permute(0, 3, 1, 2)
        h = nn.functional.pixel_shuffle(h, ps)

        return h
