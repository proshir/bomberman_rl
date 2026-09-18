"""Inference callbacks for the combat Double-DQN agent."""

# Sahand was here.

from collections import deque
from pathlib import Path

import numpy as np
import settings as s
import torch
from torch import optim

from .config import (EPSILON_END, LEARNING_RATE, N_ACTIONS)
from .features import ACTIONS, state_to_features
from .model import DEVICE, QNetwork, load_checkpoint, masked_argmax, save_checkpoint
from .safety import best_survival_action_indices, safe_action_indices


MODEL_PATH = Path(__file__).resolve().parent / "dqn_checkpoint.pt"


def state_key(game_state):
    """Identify one observation within a match."""
    return int(game_state["round"]), int(game_state["step"])


def progress_signature(game_state):
    """Summarize meaningful progress for the existing history features."""
    return (
        tuple(sorted(tuple(coin) for coin in game_state["coins"])),
        int(np.count_nonzero(game_state["field"] == 1)),
        len(game_state["others"]),
        int(game_state["self"][1]),
    )


def _probe_state():
    """Build a legal observation solely to infer the extractor dimension."""
    field = np.zeros((s.COLS, s.ROWS), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "bombs": [],
        "explosion_map": np.zeros_like(field),
        "coins": [(1, 1)],
        "self": ("probe", 0, True, (1, 1)),
        "others": [],
        "user_input": None,
    }


def _initialize_model(self, input_dim=None):
    """Create the policy/target networks and optionally load a checkpoint."""
    if self.policy_net is not None:
        return
    if input_dim is None:
        input_dim = len(state_to_features(_probe_state()))
    self.policy_net = QNetwork(input_dim, N_ACTIONS).to(DEVICE)
    self.target_net = QNetwork(input_dim, N_ACTIONS).to(DEVICE)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()
    self.optimizer = optim.Adam(
        self.policy_net.parameters(), lr=LEARNING_RATE
    ) if self.train else None

    if not self.train:
        checkpoint = load_checkpoint(self.model_path)
        if int(checkpoint["input_dim"]) != int(input_dim):
            raise ValueError("DQN checkpoint feature dimension does not match.")
        if int(checkpoint["n_actions"]) != N_ACTIONS:
            raise ValueError("DQN checkpoint action dimension does not match.")
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.epsilon = float(checkpoint.get("epsilon", EPSILON_END))
        self.env_steps = int(checkpoint.get("env_steps", 0))
        self.optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
        self.last_loss = checkpoint.get("last_loss")
        self.policy_net.eval()


def setup(self):
    """Initialize persistent DQN state and infer the feature dimension."""
    self.rng = np.random.default_rng(getattr(self, "seed", None))
    self.model_path = Path(getattr(self, "model_path", MODEL_PATH))
    if self.train and self.model_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {self.model_path}")
    if self.train:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    self.policy_net = None
    self.target_net = None
    self.optimizer = None
    self.epsilon = 1.0
    self.env_steps = 0
    self.optimizer_steps = 0
    self.last_loss = None
    _initialize_model(self)

    self.round_id = None
    self.last_progress = None
    self.last_progress_step = 0
    self.previous_action = ACTIONS.index("WAIT")
    self.positions = deque(maxlen=8)
    self.feature_cache = {}


def features_for_act(self, game_state):
    """Build features and cache exactly what the action callback observed."""
    if game_state["round"] != self.round_id:
        self.round_id = game_state["round"]
        self.positions.clear()
        self.feature_cache.clear()
        self.last_progress = None
        self.last_progress_step = int(game_state["step"])
        self.previous_action = ACTIONS.index("WAIT")

    signature = progress_signature(game_state)
    if signature != self.last_progress:
        self.positions.clear()
        self.last_progress = signature
        self.last_progress_step = int(game_state["step"])

    position = tuple(game_state["self"][3])
    feature = state_to_features(
        game_state,
        self.previous_action,
        self.positions.count(position),
        int(game_state["step"]) - self.last_progress_step,
    )
    self.feature_cache[state_key(game_state)] = (
        feature,
        tuple(self.positions),
        signature,
        self.last_progress_step,
    )
    self.positions.append(position)
    return feature


def next_features(self, old_state, action, new_state):
    """Reconstruct the history context at the next decision."""
    _, history, old_progress, last_progress_step = self.feature_cache[
        state_key(old_state)
    ]
    history = deque(history, maxlen=8)
    history.append(tuple(old_state["self"][3]))
    new_progress = progress_signature(new_state)
    if new_progress != old_progress:
        history.clear()
        last_progress_step = int(new_state["step"])
    position = tuple(new_state["self"][3])
    return state_to_features(
        new_state,
        ACTIONS.index(action),
        history.count(position),
        int(new_state["step"]) - last_progress_step,
    )


def act(self, game_state):
    """Select a legal/safe action with epsilon-greedy exploration in training."""
    if game_state is None:
        return "WAIT"
    candidates = safe_action_indices(game_state)
    if not candidates:
        candidates = best_survival_action_indices(game_state)
    features = features_for_act(self, game_state)
    state_tensor = torch.as_tensor(features, dtype=torch.float32,
                                   device=DEVICE).unsqueeze(0)
    if self.train and self.rng.random() < self.epsilon:
        choice = int(self.rng.choice(candidates))
    else:
        self.policy_net.eval()
        with torch.no_grad():
            values = self.policy_net(state_tensor)
            mask = np.zeros(N_ACTIONS, dtype=bool)
            mask[candidates] = True
            choice = int(masked_argmax(values, mask)[0].item())
    self.previous_action = choice
    return ACTIONS[choice]


__all__ = [
    "MODEL_PATH", "act", "features_for_act", "next_features", "progress_signature",
    "save_checkpoint", "setup", "state_key",
]
