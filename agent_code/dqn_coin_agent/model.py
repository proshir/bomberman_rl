"""Small action-value network used by the coin-navigation DQN pilot.

The network is deliberately compact.  Training may use CUDA, but the same
weights are loaded on CPU for the official game.
"""

from __future__ import annotations

import torch
from torch import nn


HIDDEN_SIZE = 128


class QNetwork(nn.Module):
    """Two-layer MLP mapping the engineered state vector to action values."""

    def __init__(self, input_size: int, action_count: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)
