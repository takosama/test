#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from multiprocessing import freeze_support
import torch
import torch.nn as nn
import torch.optim as optim

# Backwards-compatible constants for tests
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEVICE.type == "cuda"
NON_BLOCKING = IS_CUDA

# Re-export API from the refactored package
from mnist_cnn.models import SimpleCNN  # noqa: E402
from mnist_cnn.eval import evaluate as _evaluate_core  # noqa: E402
from mnist_cnn.cli import main as _cli_main  # noqa: E402


# Keep a criterion object for existing tests
criterion = nn.CrossEntropyLoss()


def evaluate(model: nn.Module, loader) -> float:
    """Compatibility wrapper that uses module-level device flags."""
    return _evaluate_core(model, loader, device=DEVICE, is_cuda=IS_CUDA, non_blocking=NON_BLOCKING)



def main() -> None:
    freeze_support()
    _cli_main()


if __name__ == "__main__":
    main()
