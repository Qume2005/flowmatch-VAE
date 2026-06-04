"""CelebA 数据集加载, 预处理为 64×64, 归一化到 [-1, 1]。"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset
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


def get_dataloader(
    data_path: str = "./data",
    batch_size: int = 128,
    image_size: int = 64,
    num_workers: int = 4,
    split: str = "train",
) -> DataLoader:
    transform = get_transforms(image_size)
    dataset = CelebA(
        root=data_path,
        split=split,
        target_type=["attr"],
        transform=transform,
        download=True,
    )
    dataset = _ImageOnly(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
