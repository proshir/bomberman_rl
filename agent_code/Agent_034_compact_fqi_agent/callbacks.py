"""Decision-time snapshots and frozen six-tree inference for Agent 034."""

import pickle
from collections import deque
from pathlib import Path

import numpy as np

from .features import FEATURE_SCHEMA, FEATURE_SIZE, HISTORY_LENGTH, state_to_features
from .safety import ACTIONS, allowed_action_indices


MODEL_PATH = Path(__file__).resolve().parent / "trees.pkl"


def state_key(game_state):
    return int(game_state["round"]), int(game_state["step"])


def progress_signature(game_state):
    """Observable progress shared by training and frozen inference."""
    return (int(game_state["self"][1]),
            int(np.count_nonzero(game_state["field"] == 1)))


def predict_values(trees, states):
    states = np.asarray(states, dtype=np.float32).reshape(-1, FEATURE_SIZE)
    values = np.zeros((len(states), len(ACTIONS)), dtype=float)
    for index, tree in enumerate(trees):
        if tree is not None:
            values[:, index] = tree.predict(states)
    return values


def setup(self):
    self.rng = np.random.default_rng(getattr(self, "seed", None))
    self.model_path = Path(getattr(self, "model_path", MODEL_PATH))
    if self.train:
        if self.model_path.exists():
            raise FileExistsError(f"Checkpoint already exists: {self.model_path}")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.trees = [None] * len(ACTIONS)
    else:
        with self.model_path.open("rb") as stream:
            checkpoint = pickle.load(stream)
        if (checkpoint.get("feature_schema") != FEATURE_SCHEMA or
                len(checkpoint.get("trees", [])) != len(ACTIONS)):
            raise ValueError("Incompatible Agent 034 checkpoint")
        self.trees = checkpoint["trees"]
    self.round_id = None
    self.positions = deque(maxlen=HISTORY_LENGTH)
    self.progress_step = 0
    self.last_progress = None
    self.previous_action = None
    self.observations = {}


def save_checkpoint(self, path):
    with Path(path).open("wb") as stream:
        pickle.dump({"feature_schema": FEATURE_SCHEMA, "trees": self.trees}, stream)


def observe(self, game_state):
    """Materialize exactly one feature/mask snapshot for a decision."""
    key = state_key(game_state)
    if game_state["round"] != self.round_id:
        self.round_id = game_state["round"]
        self.positions.clear()
        self.progress_step = int(game_state["step"])
        self.last_progress = None
        self.previous_action = None
        self.observations.clear()
    progress = progress_signature(game_state)
    if progress != self.last_progress:
        self.positions.clear()
        self.progress_step = int(game_state["step"])
        self.last_progress = progress
    if key not in self.observations:
        position = tuple(game_state["self"][3])
        features = state_to_features(
            game_state, self.previous_action, self.positions.count(position),
            int(game_state["step"]) - self.progress_step,
        )
        allowed = np.zeros(len(ACTIONS), dtype=bool)
        allowed[allowed_action_indices(game_state)] = True
        self.observations[key] = (features, allowed)
        self.positions.append(position)
    return self.observations[key]


def act(self, game_state):
    if game_state is None:
        return "WAIT"
    features, allowed = observe(self, game_state)
    candidates = np.flatnonzero(allowed)
    if self.train and self.rng.random() < self.epsilon:
        choice = int(self.rng.choice(candidates))
    else:
        values = predict_values(self.trees, [features])[0]
        choice = int(candidates[np.argmax(values[candidates])])
    self.previous_action = choice
    return ACTIONS[choice]
