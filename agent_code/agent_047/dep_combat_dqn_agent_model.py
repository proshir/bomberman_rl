"""Small DQN network, target, and checkpoint helpers."""

# Sahand was here.

from pathlib import Path

import torch
from torch import nn

from .dep_combat_dqn_agent_config import HIDDEN_SIZE, N_ACTIONS


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


def dqn_targets(policy_net, target_net, next_states, rewards, dones, gamma,
                next_action_masks, algorithm="dqn"):
    """Compute masked vanilla- or Double-DQN targets.

    The mask is the same safe/useful candidate set used by action selection.
    A row with no candidate is treated as having zero bootstrap value; this
    also keeps terminal rows numerically well-defined.
    """
    algorithm = algorithm.strip().lower()
    if algorithm not in {"dqn", "ddqn", "double_dqn"}:
        raise ValueError(
            "algorithm must be 'dqn', 'ddqn', or 'double_dqn', "
            f"not {algorithm!r}"
        )
    next_action_masks = next_action_masks.to(dtype=torch.bool)
    with torch.no_grad():
        target_next_q = target_net(next_states)
        selection_q = (
            target_next_q
            if algorithm == "dqn"
            else policy_net(next_states)
        )
        masked_selection_q = selection_q.masked_fill(
            ~next_action_masks, -torch.inf
        )
        best_actions = masked_selection_q.argmax(dim=1, keepdim=True)
        next_values = target_next_q.gather(1, best_actions).squeeze(1)
        has_candidates = next_action_masks.any(dim=1)
        bootstrap = torch.where(
            (dones > 0) | ~has_candidates,
            torch.zeros_like(next_values),
            next_values,
        )
        return rewards + float(gamma) * (1.0 - dones) * bootstrap


def vanilla_targets(target_net, next_states, rewards, dones, gamma,
                    next_action_masks):
    """Keep the original target helper available for existing callers."""
    return dqn_targets(
        target_net, target_net, next_states, rewards, dones, gamma,
        next_action_masks, algorithm="dqn",
    )


def save_checkpoint(learner, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "combat_vanilla_dqn_v1",
        "input_dim": learner.policy_net.input_dim,
        "n_actions": learner.policy_net.n_actions,
        "policy_state_dict": learner.policy_net.state_dict(),
        "target_state_dict": learner.target_net.state_dict(),
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "epsilon": float(learner.epsilon),
        "env_steps": int(learner.env_steps),
        "optimizer_steps": int(learner.optimizer_steps),
    }
    if hasattr(learner, "combat_env_steps"):
        checkpoint["combat_env_steps"] = int(learner.combat_env_steps)
    torch.save(checkpoint, path)


def load_checkpoint(path):
    return torch.load(Path(path), map_location=DEVICE, weights_only=False)
