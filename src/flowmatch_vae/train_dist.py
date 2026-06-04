"""分布式训练 Flow Matching VAE (Ray Train + PyTorch DDP, 单机 8 卡)。"""

from __future__ import annotations

import os
import time

import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CelebA

from flowmatch_vae.config import Config
from flowmatch_vae.data.celeba import get_transforms
from flowmatch_vae.models.vae import FlowMatchVAE


def train_func(config: dict):
    """每个 Ray worker 上运行的训练函数。"""
    cfg = Config()
    tc = cfg.train

    # 从 Ray config 覆盖参数
    for k, v in config.items():
        if hasattr(tc, k):
            setattr(tc, k, v)

    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    device = train.torch.get_device()

    if rank == 0:
        print(f"Distributed training: {world_size} GPUs, device={device}")

    # ---- Data ----
    transform = get_transforms(tc.image_size)

    # rank 0 先下载，其他 worker 等下载完再建 dataset
    if rank == 0:
        CelebA(root=tc.data_path, split="train", target_type=["attr"],
                transform=transform, download=True)
    torch.distributed.barrier()

    dataset = CelebA(root=tc.data_path, split="train", target_type=["attr"],
                     transform=transform, download=False)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=True,
    )
    per_gpu_bs = max(1, tc.batch_size // world_size)
    loader = DataLoader(
        dataset, batch_size=per_gpu_bs, sampler=sampler,
        num_workers=4, pin_memory=True,
    )

    # ---- Model (DDP) ----
    model = FlowMatchVAE(cfg).to(device)
    model = DDP(model, device_ids=[device], output_device=device)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params / 1e6:.2f}M params, per-GPU batch: {per_gpu_bs}, "
              f"total batch: {per_gpu_bs * world_size}")

    optimizer = AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=tc.epochs)

    # ---- Training Loop ----
    for epoch in range(1, tc.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)

        # KL warmup: 更新 model 内部的 kl_weight
        kl_weight = min(1.0, epoch / max(tc.kl_warmup_epochs, 1)) * tc.kl_weight
        model.module.cfg.train.kl_weight = kl_weight

        total_loss, total_fm, total_kl, n_batches = 0.0, 0.0, 0.0, 0
        epoch_start = time.time()

        for batch_idx, images in enumerate(loader):
            images = images.to(device)

            # forward 通过 DDP，梯度自动 all-reduce
            losses = model(images)

            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += losses["loss"].item()
            total_fm += losses["fm_loss"].item()
            total_kl += losses["kl_loss"].item()
            n_batches += 1

            if rank == 0 and (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                      f"loss={losses['loss'].item():.4f} "
                      f"fm={losses['fm_loss'].item():.4f} "
                      f"kl={losses['kl_loss'].item():.4f} kl_w={kl_weight:.4f}")

        scheduler.step()
        elapsed = time.time() - epoch_start

        # Checkpoint + 日志 (rank 0 only)
        if rank == 0:
            os.makedirs(tc.save_dir, exist_ok=True)
            if epoch % tc.save_interval == 0 or epoch == tc.epochs:
                path = os.path.join(tc.save_dir, f"checkpoint_epoch{epoch}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                }, path)
                print(f"Saved checkpoint: {path}")

            print(f"Epoch {epoch}/{tc.epochs} | "
                  f"loss={total_loss / n_batches:.4f} "
                  f"fm={total_fm / n_batches:.4f} "
                  f"kl={total_kl / n_batches:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f} | "
                  f"time={elapsed:.1f}s")

            # 采样可视化
            if epoch % tc.sample_interval == 0:
                _save_samples(model.module, cfg, epoch, device)

        # Ray Train 指标上报 (所有 worker 都要调用)
        train.report({
            "loss": total_loss / n_batches,
            "fm_loss": total_fm / n_batches,
            "kl_loss": total_kl / n_batches,
            "epoch": epoch,
        })

    if rank == 0:
        print("Training complete!")


@torch.no_grad()
def _save_samples(model: FlowMatchVAE, cfg: Config, epoch: int, device):
    from torchvision.utils import save_image

    model.eval()
    samples = model.sample(num_samples=16, num_steps=cfg.train.num_sample_steps, device=device)
    samples = (samples.clamp(-1, 1) + 1) / 2
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    path = os.path.join(cfg.train.log_dir, f"samples_epoch{epoch}.png")
    save_image(samples, path, nrow=4)
    print(f"Saved samples: {path}")
    model.train()


def main():
    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={},
        scaling_config=ScalingConfig(
            num_workers=8,
            use_gpu=True,
            resources_per_worker={"GPU": 1},
        ),
    )

    result = trainer.fit()
    print(f"Training result: {result}")


if __name__ == "__main__":
    main()
