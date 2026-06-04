"""训练 Flow Matching VAE。"""

from __future__ import annotations

import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from flowmatch_vae.config import Config
from flowmatch_vae.data.celeba import get_dataloader
from flowmatch_vae.models.vae import FlowMatchVAE


def train(cfg: Config | None = None):
    cfg = cfg or Config()
    tc = cfg.train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    os.makedirs(tc.save_dir, exist_ok=True)
    os.makedirs(tc.log_dir, exist_ok=True)

    loader = get_dataloader(
        data_path=tc.data_path,
        batch_size=tc.batch_size,
        image_size=tc.image_size,
    )
    print(f"Dataset size: {len(loader.dataset)}")

    model = FlowMatchVAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=tc.epochs)

    for epoch in range(1, tc.epochs + 1):
        model.train()
        total_loss = 0.0
        total_fm = 0.0
        total_kl = 0.0
        n_batches = 0

        kl_weight = min(1.0, epoch / max(tc.kl_warmup_epochs, 1)) * tc.kl_weight

        epoch_start = time.time()
        for batch_idx, images in enumerate(loader):
            images = images.to(device)

            mu, logvar = model.encode(images)
            z = model.reparameterize(mu, logvar)

            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            B = images.shape[0]
            x1 = images
            x0 = torch.randn_like(x1)
            t = torch.rand(B, 1, 1, 1, device=device)
            x_t = (1 - t) * x0 + t * x1
            v_target = x1 - x0

            v_pred = model.decoder(x_t, t.view(B), z)
            fm_loss = torch.nn.functional.mse_loss(v_pred, v_target)

            loss = fm_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_fm += fm_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                    f"loss={loss.item():.4f} fm={fm_loss.item():.4f} "
                    f"kl={kl_loss.item():.4f} kl_w={kl_weight:.4f}"
                )

        scheduler.step()
        elapsed = time.time() - epoch_start
        avg_loss = total_loss / n_batches
        avg_fm = total_fm / n_batches
        avg_kl = total_kl / n_batches

        print(
            f"Epoch {epoch}/{tc.epochs} | "
            f"loss={avg_loss:.4f} fm={avg_fm:.4f} kl={avg_kl:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f} | "
            f"time={elapsed:.1f}s"
        )

        if epoch % tc.save_interval == 0 or epoch == tc.epochs:
            path = os.path.join(tc.save_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            }, path)
            print(f"Saved checkpoint: {path}")

        if epoch % tc.sample_interval == 0:
            _save_samples(model, cfg, epoch, device)

    print("Training complete!")


@torch.no_grad()
def _save_samples(model: FlowMatchVAE, cfg: Config, epoch: int, device: str):
    from torchvision.utils import save_image

    model.eval()
    samples = model.sample(num_samples=16, num_steps=cfg.train.num_sample_steps, device=device)
    samples = (samples.clamp(-1, 1) + 1) / 2

    path = os.path.join(cfg.train.log_dir, f"samples_epoch{epoch}.png")
    save_image(samples, path, nrow=4)
    print(f"Saved samples: {path}")
    model.train()


if __name__ == "__main__":
    train()
