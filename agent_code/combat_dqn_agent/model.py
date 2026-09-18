"""Network, masking, and Double-DQN target helpers."""

# Sahand was here.

from pathlib import Path

import torch
from torch import nn

from .config import GAMMA, HIDDEN_SIZE, N_ACTIONS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Two-hidden-layer MLP for the existing one-dimensional feature vector."""

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


def masked_argmax(values, action_mask):
    """Return an argmax that can never select an unmasked action."""
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=values.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("Every state needs at least one allowed action.")
    masked = values.masked_fill(~mask, torch.finfo(values.dtype).min)
    return masked.argmax(dim=1)


def double_dqn_targets(policy_net, target_net, next_states, rewards, dones,
                       next_action_masks, gamma=GAMMA):
    """Compute detached Double-DQN targets with terminal zero bootstrapping."""
    with torch.no_grad():
        policy_values = policy_net(next_states)
        target_values = target_net(next_states)
        selected = torch.zeros_like(rewards)
        nonterminal = ~dones
        if nonterminal.any():
            next_actions = masked_argmax(
                policy_values[nonterminal], next_action_masks[nonterminal]
            )
            selected[nonterminal] = target_values[nonterminal].gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
        bootstrap = selected
        return rewards + float(gamma) * bootstrap


def checkpoint_payload(learner):
    """Serialize model, optimizer, epsilon, and training counters."""
    return {
        "format": "combat_double_dqn_v1",
        "input_dim": learner.policy_net.input_dim,
        "n_actions": learner.policy_net.n_actions,
        "policy_state_dict": learner.policy_net.state_dict(),
        "target_state_dict": learner.target_net.state_dict(),
        "optimizer_state_dict": (
            learner.optimizer.state_dict() if learner.optimizer is not None else None
        ),
        "epsilon": float(getattr(learner, "epsilon", 1.0)),
        "env_steps": int(getattr(learner, "env_steps", 0)),
        "optimizer_steps": int(getattr(learner, "optimizer_steps", 0)),
        "last_loss": getattr(learner, "last_loss", None),
    }


def save_checkpoint(learner, path):
    """Save a complete DQN checkpoint at ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(learner), path)


def load_checkpoint(path, map_location=DEVICE):
    """Load a checkpoint without assuming a CUDA device is available."""
    return torch.load(Path(path), map_location=map_location, weights_only=False)
