from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import TrainConfig


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def create_dataloaders(cfg: TrainConfig, device: torch.device):
    transform = build_transform()
    train_ds = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=transform)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1000,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader

