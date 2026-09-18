"""Small vanilla-DQN network and checkpoint helpers."""

# Sahand was here.

from pathlib import Path

import torch
from torch import nn

from .config import HIDDEN_SIZE, N_ACTIONS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions=N_ACTIONS):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_actions = int(n_actions)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, self.n_actions),
        )

    def forward(self, states):
        return self.network(states)


def vanilla_targets(target_net, next_states, rewards, dones, gamma,
                    next_action_masks):
    """Compute masked vanilla-DQN targets with the target network.

    The mask is the same safe/useful candidate set used by action selection.
    A row with no candidate is treated as having zero bootstrap value; this
    also keeps terminal rows numerically well-defined.
    """
    next_action_masks = next_action_masks.to(dtype=torch.bool)
    with torch.no_grad():
        next_q_values = target_net(next_states)
        masked_values = next_q_values.masked_fill(~next_action_masks, -torch.inf)
        next_values, _ = masked_values.max(dim=1)
        has_candidates = next_action_masks.any(dim=1)
        bootstrap = torch.where(
            (dones > 0) | ~has_candidates,
            torch.zeros_like(next_values),
            next_values,
        )
        return rewards + float(gamma) * (1.0 - dones) * bootstrap


def save_checkpoint(learner, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "combat_vanilla_dqn_v1",
        "input_dim": learner.policy_net.input_dim,
        "n_actions": learner.policy_net.n_actions,
        "policy_state_dict": learner.policy_net.state_dict(),
        "target_state_dict": learner.target_net.state_dict(),
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "epsilon": float(learner.epsilon),
        "env_steps": int(learner.env_steps),
        "optimizer_steps": int(learner.optimizer_steps),
    }, path)


def load_checkpoint(path):
    return torch.load(Path(path), map_location=DEVICE, weights_only=False)
