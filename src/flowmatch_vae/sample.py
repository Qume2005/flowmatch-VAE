"""从训练好的 FlowMatchVAE 采样或重建图片。"""

from __future__ import annotations

import argparse
import os

import torch
from torchvision.utils import save_image

from flowmatch_vae.config import Config
from flowmatch_vae.models.vae import FlowMatchVAE


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple[FlowMatchVAE, Config]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Config 存为 dict 以支持 weights_only=True
    cfg_dict = ckpt["config"]
    if isinstance(cfg_dict, dict):
        from dataclasses import fields
        cfg = Config()
        for section_name in ("encoder", "decoder", "train"):
            section_cfg = getattr(cfg, section_name)
            section_dict = cfg_dict.get(section_name, {})
            for f in fields(section_cfg):
                if f.name in section_dict:
                    setattr(section_cfg, f.name, section_dict[f.name])
    else:
        cfg = cfg_dict
    model = FlowMatchVAE(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def sample(
    checkpoint_path: str,
    num_samples: int = 16,
    num_steps: int = 8,
    output_path: str = "samples.png",
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = load_model(checkpoint_path, device)
    images = model.sample(num_samples=num_samples, num_steps=num_steps, device=device)
    images = (images.clamp(-1, 1) + 1) / 2
    save_image(images, output_path, nrow=4)
    print(f"Saved {num_samples} samples to {output_path}")


@torch.no_grad()
def reconstruct(
    checkpoint_path: str,
    image_path: str,
    num_steps: int = 8,
    output_path: str = "reconstruction.png",
    device: str = "auto",
):
    from torchvision import transforms
    from PIL import Image

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = load_model(checkpoint_path, device)
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    recon = model.reconstruct(x, num_steps=num_steps)
    comparison = torch.cat([x, recon], dim=0)
    comparison = (comparison.clamp(-1, 1) + 1) / 2
    save_image(comparison, output_path, nrow=2)
    print(f"Saved reconstruction to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowMatch VAE Sampling")
    parser.add_argument("checkpoint", help="Path to checkpoint")
    parser.add_argument("--mode", choices=["sample", "reconstruct"], default="sample")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--image", type=str, help="Input image for reconstruction")
    parser.add_argument("--output", type=str, default="output.png")
    args = parser.parse_args()

    if args.mode == "sample":
        sample(args.checkpoint, args.num_samples, args.num_steps, args.output)
    else:
        reconstruct(args.checkpoint, args.image, args.num_steps, args.output)
