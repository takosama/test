"""MNIST CNN training package.

Exports:
- SimpleCNN: CNN model
- evaluate: accuracy evaluation helper
- build_argparser: CLI arg builder
- run_training: entry point to launch training
- build_transform / create_dataloaders: data utilities
- TrainConfig: typed configuration container
"""

from .models import SimpleCNN
from .eval import evaluate
from .cli import build_argparser, main as run_training
from .data import build_transform, create_dataloaders
from .config import TrainConfig

__all__ = [
    "SimpleCNN",
    "evaluate",
    "build_argparser",
    "run_training",
    "build_transform",
    "create_dataloaders",
    "TrainConfig",
]
