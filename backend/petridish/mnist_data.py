"""Canonical MNIST datasets and deterministic data loaders."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def default_data_root() -> Path:
    """Keep downloaded scientific data in the repository's ignored data folder."""

    return Path(__file__).resolve().parents[2] / "data"


def load_mnist_datasets(root: Path | None = None) -> tuple[Dataset, Dataset]:
    """Download if necessary and return the canonical 60k/10k MNIST splits."""

    data_root = root or default_data_root()
    transform = ToTensor()
    return (
        MNIST(data_root, train=True, download=True, transform=transform),
        MNIST(data_root, train=False, download=True, transform=transform),
    )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Create a cross-platform deterministic loader appropriate for a live loop."""

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=shuffle,
    )
