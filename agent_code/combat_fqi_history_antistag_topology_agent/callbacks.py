"""Callbacks for the topology-augmented combat FQI agent."""

# Sahand was here.

import pickle
from collections import deque
from pathlib import Path

import numpy as np

from .features import ACTIONS, state_to_features
from .safety import best_survival_action_indices, safe_action_indices


MODEL_PATH = Path(__file__).resolve().parent / "trees.pkl"


def state_key(game_state):
    """Identify one observation within a match."""
    return int(game_state["round"]), int(game_state["step"])


def progress_signature(game_state):
    """Summarize changes that reset movement stagnation context."""
    return (
        tuple(sorted(tuple(coin) for coin in game_state["coins"])),
        int(np.count_nonzero(game_state["field"] == 1)),
        len(game_state["others"]),
        int(game_state["self"][1]),
    )


def setup(self):
    """Initialize the model and episode-local movement memory."""
    self.rng = np.random.default_rng(getattr(self, "seed", None))
    self.model_path = Path(getattr(self, "model_path", MODEL_PATH))
    if self.train:
        if self.model_path.exists():
            raise FileExistsError(f"Checkpoint already exists: {self.model_path}")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.trees = [None] * len(ACTIONS)
    else:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Trained model not found: {self.model_path}")
        with open(self.model_path, "rb") as file:
            self.trees = pickle.load(file)
        if len(self.trees) != len(ACTIONS):
            raise ValueError("Checkpoint has the wrong number of action trees.")

    self.round_id = None
    self.last_progress = None
    self.last_progress_step = 0
    self.previous_action = ACTIONS.index("WAIT")
    self.positions = deque(maxlen=8)
    self.feature_cache = {}


def predict_values(trees, states):
    """Predict one value per action; actions without observations stay zero."""
    states = np.asarray(states, dtype=float)
    values = np.zeros((len(states), len(ACTIONS)))
    for action, tree in enumerate(trees):
        if tree is not None:
            values[:, action] = tree.predict(states)
    return values


def features_for_act(self, game_state):
    """Build features and cache the exact action-time context."""
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
    """Reconstruct the history and topology context after one action."""
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
    """Choose a learned action among actions with a predicted survival route."""
    if game_state is None:
        return "WAIT"
    candidates = safe_action_indices(game_state)
    if not candidates:
        candidates = best_survival_action_indices(game_state)

    features = features_for_act(self, game_state)
    if self.train and self.rng.random() < self.epsilon:
        choice = int(self.rng.choice(candidates))
    else:
        values = predict_values(self.trees, [features])[0]
        best_value = max(values[index] for index in candidates)
        best = [index for index in candidates if values[index] == best_value]
        choice = int(self.rng.choice(best))
    self.previous_action = choice
    return ACTIONS[choice]
