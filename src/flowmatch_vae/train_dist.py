"""分布式训练 Flow Matching VAE (Ray Actor + PyTorch NCCL, 单机 8 卡)。

Ray 只负责 worker 放置和生命周期管理，梯度同步走原生 PyTorch DDP (NCCL)。
没有 Ray Train 的内部守护进程噪音。
"""

from __future__ import annotations

import os
import socket
import time

import ray
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from flowmatch_vae.config import Config
from flowmatch_vae.data.celeba import cache_dataset, get_transforms
from flowmatch_vae.models.vae import FlowMatchVAE


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@ray.remote(num_gpus=1, num_cpus=2)
class TrainingWorker:
    """每个 GPU 一个 worker，Ray 管理放置，PyTorch NCCL 管理通信。"""

    def setup(self, rank: int, world_size: int, master_addr: str, master_port: int):
        self.rank = rank
        self.world_size = world_size

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
        )
        self.device = torch.device("cuda")

    def train(self, cfg_dict: dict) -> dict:
        cfg = Config()
        tc = cfg.train
        for k, v in cfg_dict.items():
            if hasattr(tc, k):
                setattr(tc, k, v)

        # ---- Data (内存缓存) ----
        dataset = cache_dataset(tc.data_path, tc.image_size)
        sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank,
            shuffle=True, drop_last=True,
        )
        per_gpu_bs = max(1, tc.batch_size // self.world_size)
        loader = DataLoader(
            dataset, batch_size=per_gpu_bs, sampler=sampler,
            num_workers=8, pin_memory=True,
        )

        # ---- Model (DDP) ----
        model = FlowMatchVAE(cfg).to(self.device)
        model = DDP(model, device_ids=[self.device], output_device=self.device)

        if self.rank == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {n_params / 1e6:.2f}M params, "
                  f"per-GPU batch: {per_gpu_bs}, total: {per_gpu_bs * self.world_size}")

        optimizer = AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=tc.epochs)

        # ---- Training Loop ----
        for epoch in range(1, tc.epochs + 1):
            model.train()
            sampler.set_epoch(epoch)

            kl_weight = min(1.0, epoch / max(tc.kl_warmup_epochs, 1)) * tc.kl_weight
            model.module.cfg.train.kl_weight = kl_weight

            total_loss, total_fm, total_kl, n_batches = 0.0, 0.0, 0.0, 0
            epoch_start = time.time()

            for batch_idx, (images,) in enumerate(loader):
                images = images.to(self.device, non_blocking=True)

                losses = model(images)

                optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += losses["loss"].item()
                total_fm += losses["fm_loss"].item()
                total_kl += losses["kl_loss"].item()
                n_batches += 1

                if self.rank == 0 and (batch_idx + 1) % 50 == 0:
                    print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                          f"loss={losses['loss'].item():.4f} "
                          f"fm={losses['fm_loss'].item():.4f} "
                          f"kl={losses['kl_loss'].item():.4f}")

            scheduler.step()
            elapsed = time.time() - epoch_start

            if self.rank == 0:
                os.makedirs(tc.save_dir, exist_ok=True)
                if epoch % tc.save_interval == 0 or epoch == tc.epochs:
                    path = os.path.join(tc.save_dir, f"checkpoint_epoch{epoch}.pt")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": cfg,
                    }, path)
                    print(f"Saved: {path}")

                print(f"Epoch {epoch}/{tc.epochs} | "
                      f"loss={total_loss / n_batches:.4f} "
                      f"fm={total_fm / n_batches:.4f} "
                      f"kl={total_kl / n_batches:.4f} | "
                      f"lr={scheduler.get_last_lr()[0]:.6f} | "
                      f"time={elapsed:.1f}s")

                if epoch % tc.sample_interval == 0:
                    _save_samples(model.module, cfg, epoch, self.device)

            dist.barrier()

        if self.rank == 0:
            print("Training complete!")

        dist.destroy_process_group()
        return {"rank": self.rank, "status": "done"}


@torch.no_grad()
def _save_samples(model, cfg, epoch, device):
    from torchvision.utils import save_image

    model.eval()
    samples = model.sample(num_samples=16, num_steps=cfg.train.num_sample_steps, device=device)
    samples = (samples.clamp(-1, 1) + 1) / 2
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    path = os.path.join(cfg.train.log_dir, f"samples_epoch{epoch}.png")
    save_image(samples, path, nrow=4)
    print(f"Saved samples: {path}")
    model.train()


def prepare_data(cfg: Config):
    from torchvision.datasets import CelebA

    tc = cfg.train
    print(f"Checking dataset at {tc.data_path}...")
    CelebA(root=tc.data_path, split="train", target_type=["attr"],
           transform=get_transforms(tc.image_size), download=True)
    print("Dataset ready.")


def main():
    ray.init()

    cfg = Config()
    cfg.train.data_path = os.path.abspath(cfg.train.data_path)
    cfg.train.save_dir = os.path.abspath(cfg.train.save_dir)
    cfg.train.log_dir = os.path.abspath(cfg.train.log_dir)

    prepare_data(cfg)

    world_size = 8
    master_addr = ray.util.get_node_ip_address()
    master_port = _find_free_port()

    print(f"Launching {world_size} workers, master={master_addr}:{master_port}")

    # 创建 8 个 GPU worker
    workers = [TrainingWorker.remote() for _ in range(world_size)]

    # 初始化 torch.distributed（所有 worker 同时连上 TCP master）
    ray.get([
        w.setup.remote(i, world_size, master_addr, master_port)
        for i, w in enumerate(workers)
    ])
    print("All workers initialized.")

    # 开始训练
    cfg_dict = {
        "data_path": cfg.train.data_path,
        "save_dir": cfg.train.save_dir,
        "log_dir": cfg.train.log_dir,
    }
    results = ray.get([w.train.remote(cfg_dict) for w in workers])
    print(f"All workers finished: {results}")


if __name__ == "__main__":
    main()
