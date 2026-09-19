"""Two-hidden-layer, 256-wide dueling Q-network for Agent 028."""

from torch import nn


HIDDEN_SIZE = 256


class DuelingQNetwork(nn.Module):
    """Return Q-values as value plus mean-centered action advantages."""

    def __init__(self, input_dim, n_actions=6):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_actions = int(n_actions)
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN_SIZE), nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(),
        )
        self.value = nn.Linear(HIDDEN_SIZE, 1)
        self.advantage = nn.Linear(HIDDEN_SIZE, self.n_actions)

    def forward(self, states):
        hidden = self.trunk(states)
        advantage = self.advantage(hidden)
        return self.value(hidden) + advantage - advantage.mean(
            dim=-1, keepdim=True
        )
