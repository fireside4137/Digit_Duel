"""
Dataset and DataLoader utilities for MNIST.
"""

from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.config import DATA_DIR, BATCH_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def get_transforms() -> transforms.Compose:
    """Returns standard MNIST transforms: ToTensor + Normalization [-1, 1]."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    data_dir: str = str(DATA_DIR),
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns training and test DataLoaders.

    Args:
        batch_size: Batch size for both loaders.
        data_dir: Root directory of MNIST dataset.
        download: Whether to download if not present.
    """
    transform = get_transforms()

    train_set = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )

    test_set = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
