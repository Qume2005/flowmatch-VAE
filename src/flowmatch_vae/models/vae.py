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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 3, 64, 64) -> mu, logvar"""
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.compute_loss(x)

    def compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B = x.shape[0]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        x1 = x
        x0 = torch.randn_like(x1)
        t = torch.rand(B, 1, 1, 1, device=x.device)
        x_t = (1 - t) * x0 + t * x1
        v_target = x1 - x0

        v_pred = self.decoder(x_t, t.squeeze().view(B), z)
        fm_loss = F.mse_loss(v_pred, v_target)

        loss = fm_loss + self.cfg.train.kl_weight * kl_loss
        return {"loss": loss, "fm_loss": fm_loss, "kl_loss": kl_loss}

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        num_steps: int = 8,
        z: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        if z is None:
            z = torch.randn(num_samples, 8, 8, 256, device=device)
        else:
            num_samples = z.shape[0]

        x = torch.randn(num_samples, 3, 64, 64, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((num_samples,), t_val, device=device)
            v = self.decoder(x, t, z)
            x = x + v * dt

        return x

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, num_steps: int = 8) -> torch.Tensor:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.sample(num_steps=num_steps, z=z, device=x.device)
