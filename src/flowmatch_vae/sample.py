"""从训练好的 FlowMatchVAE 采样或重建图片。"""

from __future__ import annotations

import argparse
import os

import torch
from torchvision.utils import save_image

from flowmatch_vae.config import Config
from flowmatch_vae.models.vae import FlowMatchVAE


def _restore_nested_config(section_cfg, section_dict):
    """Recursively restore nested dataclass fields from plain dicts.

    When ``dataclasses.asdict()`` serialises a config, nested dataclasses
    (e.g. ``MoEConfig`` inside ``MultiScaleDecoderConfig``) become plain
    dicts.  This helper walks the fields of *section_cfg* and converts
    any dict value whose field type is (or contains) a dataclass back into
    an instance.
    """
    import types
    from dataclasses import fields, is_dataclass

    for f in fields(section_cfg):
        if f.name not in section_dict:
            continue
        value = section_dict[f.name]
        if not isinstance(value, dict):
            setattr(section_cfg, f.name, value)
            continue

        # Collect candidate types from the annotation (may be a Union like
        # ``MoEConfig | None``).  We try each non-NoneType candidate.
        candidates = []
        ftype = f.type
        if isinstance(ftype, types.UnionType):
            candidates = [a for a in ftype.__args__ if a is not type(None)]
        elif is_dataclass(ftype):
            candidates = [ftype]

        restored = False
        for cls in candidates:
            if is_dataclass(cls):
                try:
                    setattr(section_cfg, f.name, cls(**value))
                    restored = True
                    break
                except TypeError:
                    pass

        if not restored:
            # Not a known dataclass — store the dict as-is (or scalar).
            setattr(section_cfg, f.name, value)


def _detect_config_type(section_dict: dict, new_cls, old_cls):
    """Detect whether a section dict corresponds to *new_cls* or *old_cls*.

    Compares the set of dict keys against the fields of each dataclass.
    Returns the class whose field names best match the dict keys.
    """
    from dataclasses import fields as dc_fields
    dict_keys = set(section_dict.keys())
    new_keys = {f.name for f in dc_fields(new_cls)}
    old_keys = {f.name for f in dc_fields(old_cls)}
    # Prefer the class whose keys are a superset of (or equal to) the dict keys
    if dict_keys <= new_keys:
        return new_cls
    if dict_keys <= old_keys:
        return old_cls
    # Fallback: whichever has more overlap
    new_overlap = len(dict_keys & new_keys)
    old_overlap = len(dict_keys & old_keys)
    return new_cls if new_overlap >= old_overlap else old_cls


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple[FlowMatchVAE, Config]:
    from flowmatch_vae.config import EncoderConfig, DecoderConfig

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Config stored as dict to support weights_only=True
    cfg_dict = ckpt["config"]
    if isinstance(cfg_dict, dict):
        cfg = Config()

        # Detect and instantiate the correct config type for each section.
        # _restore_nested_config handles converting nested dicts (e.g. MoEConfig)
        # back into proper dataclass instances.
        section_map = {
            "encoder": (type(cfg.encoder), EncoderConfig),
            "decoder": (type(cfg.decoder), DecoderConfig),
        }
        for section_name, (new_cls, old_cls) in section_map.items():
            section_dict = cfg_dict.get(section_name, {})
            detected_cls = _detect_config_type(section_dict, new_cls, old_cls)
            section_cfg = detected_cls()
            _restore_nested_config(section_cfg, section_dict)
            setattr(cfg, section_name, section_cfg)

        # Restore remaining sections (mhc, train) — no ambiguity in type
        for section_name in ("mhc", "train"):
            section_cfg = getattr(cfg, section_name)
            section_dict = cfg_dict.get(section_name, {})
            _restore_nested_config(section_cfg, section_dict)
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
