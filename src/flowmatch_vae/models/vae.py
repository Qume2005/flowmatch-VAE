"""Flow Matching VAE: Multi-Scale Conv Encoder + U-Net Decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmatch_vae.config import (
    Config,
    MultiScaleEncoderConfig,
    MultiScaleDecoderConfig,
    EncoderConfig,
    DecoderConfig,
)
from flowmatch_vae.models.conv_encoder import MultiScaleConvEncoder, MultiScalePrior
from flowmatch_vae.models.encoder import SwinEncoder
from flowmatch_vae.models.decoder import FlowDecoder


class FlowMatchVAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Encoder
        if isinstance(cfg.encoder, MultiScaleEncoderConfig):
            self.encoder = MultiScaleConvEncoder(cfg.encoder)
        else:
            self.encoder = SwinEncoder(cfg.encoder, mhc_cfg=cfg.mhc)

        # Decoder
        if isinstance(cfg.decoder, MultiScaleDecoderConfig):
            self.decoder = FlowDecoder(cfg.decoder, mhc_cfg=cfg.mhc)
        else:
            self.decoder = FlowDecoder(cfg.decoder, mhc_cfg=cfg.mhc)

        # Prior network (only used with multi-scale encoder)
        if isinstance(cfg.encoder, MultiScaleEncoderConfig):
            self.prior = MultiScalePrior(cfg.encoder.embed_dim)
        else:
            self.prior = None

    def encode(self, x: torch.Tensor):
        """Encode image to latent space.

        Returns:
            mu: (B, 8, 8, C)
            logvar: (B, 8, 8, C)
            scale_tokens: dict[int, (B, N_s, C)] or empty dict
        """
        result = self.encoder(x)
        if len(result) == 3:
            mu, logvar, scale_tokens = result
            return mu, logvar, scale_tokens
        return result[0], result[1], {}

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.compute_loss(x)

    def _build_cond_tokens(
        self,
        z: torch.Tensor,
        encoder_tokens: dict[int, torch.Tensor],
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor | None]:
        """Build conditioning tokens for the decoder.

        During training: randomly choose encoder features or prior predictions (50/50).
        During eval: use encoder features (reconstruction) -- caller overrides for generation.

        Returns:
            cond_tokens: dict with scale 2 set to z_flat
            prior_loss: MSE loss if prior is used, else None
        """
        B = z.shape[0]
        C = z.shape[-1]
        z_flat = z.reshape(B, -1, C)

        prior_loss = None

        if self.prior is not None and len(encoder_tokens) > 0:
            prior_tokens = self.prior(z)

            # Prior loss: predict encoder features
            prior_loss = torch.tensor(0.0, device=z.device)
            for s in prior_tokens:
                if s in encoder_tokens and s != 2:
                    prior_loss = prior_loss + F.mse_loss(prior_tokens[s], encoder_tokens[s])

            # Random choice during training
            if self.training and torch.rand(1).item() < 0.5:
                cond_tokens = dict(prior_tokens)
            else:
                cond_tokens = dict(encoder_tokens)
        else:
            cond_tokens = dict(encoder_tokens)

        # Scale 2 is always z
        cond_tokens[2] = z_flat

        return cond_tokens, prior_loss

    def compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute OT-CFM + KL + prior loss."""
        B = x.shape[0]

        # Encode
        mu, logvar, scale_tokens = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Build conditioning tokens
        cond_tokens, prior_loss = self._build_cond_tokens(z, scale_tokens)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # OT-CFM loss
        x1 = x
        x0 = torch.randn_like(x1)
        t = torch.rand(B, 1, 1, 1, device=x.device)
        x_t = (1 - t) * x0 + t * x1
        v_target = x1 - x0

        v_pred = self.decoder(x_t, t.squeeze().view(B), z, scale_tokens=cond_tokens)
        fm_loss = F.mse_loss(v_pred, v_target)

        # Total loss
        loss = fm_loss + self.cfg.train.kl_weight * kl_loss
        if prior_loss is not None:
            loss = loss + self.cfg.train.prior_weight * prior_loss

        result = {"loss": loss, "fm_loss": fm_loss, "kl_loss": kl_loss}
        if prior_loss is not None:
            result["prior_loss"] = prior_loss
        return result

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        num_steps: int = 8,
        z: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Sample from prior using predicted multi-scale features."""
        if z is None:
            z = torch.randn(num_samples, 8, 8, 256, device=device)
        else:
            num_samples = z.shape[0]

        # Predict multi-scale features from z
        B, Hz, Wz, C = z.shape
        z_flat = z.reshape(B, -1, C)
        cond_tokens: dict[int, torch.Tensor] = {}
        cond_tokens[2] = z_flat
        if self.prior is not None:
            cond_tokens.update(self.prior(z))

        x = torch.randn(num_samples, 3, 64, 64, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((num_samples,), t_val, device=device)
            v = self.decoder(x, t, z, scale_tokens=cond_tokens)
            x = x + v * dt

        return x

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, num_steps: int = 8) -> torch.Tensor:
        """Encode then reconstruct image."""
        mu, logvar, scale_tokens = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.sample(num_steps=num_steps, z=z, device=x.device)
