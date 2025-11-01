from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 20
    target_acc: float = 0.995
    num_workers: int = 2
    deterministic: bool = False
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    gpu: int = 0
    seed: int | None = 42

