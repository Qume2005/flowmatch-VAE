"""CelebA 数据集加载, 预处理为 64×64, 归一化到 [-1, 1]。"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.datasets import CelebA


class _ImageOnly(Dataset):
    """包装 CelebA，只返回图片张量。"""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, _ = self.dataset[idx]
        return img


def get_transforms(image_size: int = 64) -> transforms.Compose:
    """CelebA 标准预处理: center crop 178×178 → resize → normalize [-1, 1]。"""
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def cache_dataset(data_path: str, image_size: int = 64, split: str = "train") -> TensorDataset:
    """把整个数据集预处理后存进内存，后续加载零 IO。"""
    transform = get_transforms(image_size)
    dataset = _ImageOnly(CelebA(
        root=data_path, split=split, target_type=["attr"],
        transform=transform, download=False,
    ))
    print(f"Caching {len(dataset)} images into memory...")
    images = torch.stack([dataset[i] for i in range(len(dataset))])
    print(f"Cached: {images.shape}, {images.numel() * 4 / 1e9:.1f} GB")
    return TensorDataset(images)


def get_dataloader(
    data_path: str = "./data",
    batch_size: int = 512,
    image_size: int = 64,
    num_workers: int = 8,
    split: str = "train",
    use_cache: bool = True,
) -> DataLoader:
    transform = get_transforms(image_size)

    if use_cache:
        dataset = cache_dataset(data_path, image_size, split)
    else:
        dataset = _ImageOnly(CelebA(
            root=data_path, split=split, target_type=["attr"],
            transform=transform, download=True,
        ))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
