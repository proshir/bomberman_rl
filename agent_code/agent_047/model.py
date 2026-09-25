"""Small DQN network, target, and checkpoint helpers."""

# Sahand was here.

import copy
from pathlib import Path

import torch
from torch import nn

import os

ALGORITHM = os.environ.get("BOMBERMAN_DQN_ALGORITHM", "dqn")
GAMMA = 0.99
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
REPLAY_CAPACITY = 100_000
WARMUP_TRANSITIONS = 5_000
TRAIN_EVERY = 4
TARGET_UPDATE_EVERY = 1_000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 100_000
GRADIENT_CLIP_NORM = 10.0
HIDDEN_SIZE = 128
N_ACTIONS = 6


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


def _pad_checkpoint_inputs(checkpoint, new_size, label):
    old_size = int(checkpoint["input_dim"])
    result = copy.deepcopy(checkpoint)
    added = new_size - old_size
    first_weight = "network.0.weight"
    old_shape = result["policy_state_dict"][first_weight].shape

    def pad(tensor):
        if tensor.ndim != 2 or tensor.shape[1] != old_size:
            raise ValueError(
                f"Unexpected first-layer checkpoint shape: {tuple(tensor.shape)}"
            )
        return torch.cat(
            (tensor, tensor.new_zeros((tensor.shape[0], added))), dim=1
        )

    for name in ("policy_state_dict", "target_state_dict"):
        result[name][first_weight] = pad(result[name][first_weight])

    optimizer = result.get("optimizer_state_dict", {})
    groups = optimizer.get("param_groups", [])
    if groups and groups[0].get("params"):
        first_parameter = groups[0]["params"][0]
        state = optimizer.get("state", {}).get(first_parameter, {})
        for name, value in list(state.items()):
            if torch.is_tensor(value) and value.shape == old_shape:
                state[name] = pad(value)

    result["input_dim"] = new_size
    result["migration"] = f"{old_size}-to-{new_size} zero-padded {label} inputs"
    return result


def expand_checkpoint(checkpoint):
    """Widen supported earlier policy inputs while preserving old Q values."""
    from .features import (
        AGENT_040_FEATURE_SIZE, AGENT_041_FEATURE_SIZE,
        AGENT_042_FEATURE_SIZE, FEATURE_SIZE,
    )

    old_size = int(checkpoint["input_dim"])
    if old_size == FEATURE_SIZE:
        return checkpoint
    if old_size not in (
        AGENT_040_FEATURE_SIZE, AGENT_041_FEATURE_SIZE,
        AGENT_042_FEATURE_SIZE,
    ):
        raise ValueError(
            f"Agent 047 accepts {AGENT_040_FEATURE_SIZE}, "
            f"{AGENT_041_FEATURE_SIZE}, {AGENT_042_FEATURE_SIZE}, or "
            f"{FEATURE_SIZE}-input checkpoints; got {old_size}"
        )
    if old_size != AGENT_042_FEATURE_SIZE:
        checkpoint = _pad_checkpoint_inputs(
            checkpoint, AGENT_042_FEATURE_SIZE, "Agent 042"
        )
    return _pad_checkpoint_inputs(
        checkpoint, FEATURE_SIZE, "Agent 043 novelty"
    )
