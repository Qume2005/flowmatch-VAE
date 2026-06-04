"""Swin Transformer Encoder: 图片 -> μ, logvar 潜在空间。"""

from __future__ import annotations

import torch
import torch.nn as nn

from flowmatch_vae.config import EncoderConfig
from flowmatch_vae.models.swin import PatchEmbed, PatchMerge, SwinBlock


class SwinEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbed(
            in_channels=cfg.in_channels,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
        )

        depths = cfg.depths
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        dim = cfg.embed_dim

        for i, depth in enumerate(depths):
            stage = nn.Sequential(*[
                SwinBlock(
                    dim=dim,
                    num_heads=cfg.num_heads[i],
                    window_size=cfg.window_size,
                    shift_size=0 if (j % 2 == 0) else cfg.window_size // 2,
                    mlp_ratio=cfg.mlp_ratio,
                    drop_path=dpr[sum(depths[:i]) + j],
                )
                for j in range(depth)
            ])
            self.stages.append(stage)

            if i < len(depths) - 1:
                self.downsample.append(PatchMerge(dim))
                dim *= 2
            else:
                self.downsample.append(None)

        self.out_dim = dim
        self.mu_head = nn.Linear(dim, dim)
        self.logvar_head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 3, 64, 64) -> μ: (B, 8, 8, C), logvar: (B, 8, 8, C)"""
        x = self.patch_embed(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.downsample[i] is not None:
                x = self.downsample[i](x)

        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar
