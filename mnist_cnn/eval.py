from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import torch.nn as nn


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    is_cuda: bool,
    non_blocking: bool,
) -> float:
    """Compute accuracy in [0,1] on loader while preserving training mode.

    Parameters are explicit to decouple from global state.
    """
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for data, target in loader:
            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    model.train(was_training)
    return correct / max(1, total)


__all__ = ["evaluate"]
