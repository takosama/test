from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Minimal CNN for MNIST (28x28, 1 channel).

    Conv(1→32,3x3)->ReLU->Conv(32→64,3x3)->ReLU->MaxPool(2)
    -> Flatten -> Linear(9216→128) -> ReLU -> Linear(128→10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


__all__ = ["SimpleCNN"]
