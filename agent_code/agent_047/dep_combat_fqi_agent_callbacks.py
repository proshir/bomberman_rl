"""Tournament callbacks for the bomb-aware fitted-Q agent."""

# Sahand was here.

import pickle
from pathlib import Path

import numpy as np

from .dep_combat_fqi_agent_features import ACTIONS, state_to_features
from .dep_combat_fqi_agent_safety import best_survival_action_indices, safe_action_indices


MODEL_PATH = Path(__file__).resolve().parent / "trees.pkl"


def setup(self):
    """Create an empty model for training or load a frozen tournament model."""
    self.rng = np.random.default_rng(getattr(self, "seed", None))
    self.model_path = Path(getattr(self, "model_path", MODEL_PATH))
    if self.train:
        if self.model_path.exists():
            raise FileExistsError(f"Checkpoint already exists: {self.model_path}")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.trees = [None] * len(ACTIONS)
        return

    if not self.model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {self.model_path}")
    with open(self.model_path, "rb") as file:
        self.trees = pickle.load(file)
    if len(self.trees) != len(ACTIONS):
        raise ValueError("Checkpoint has the wrong number of action trees.")


def predict_values(trees, states):
    """Predict all action values; actions without data start at zero."""
    states = np.asarray(states, dtype=float)
    values = np.zeros((len(states), len(ACTIONS)))
    for action, tree in enumerate(trees):
        if tree is not None:
            values[:, action] = tree.predict(states)
    return values


def act(self, game_state):
    """Choose among actions for which the safety search finds an escape route."""
    if game_state is None:
        return "WAIT"

    candidates = safe_action_indices(game_state)
    if not candidates:
        candidates = best_survival_action_indices(game_state)

    if self.train and self.rng.random() < self.epsilon:
        return ACTIONS[int(self.rng.choice(candidates))]

    values = predict_values(self.trees, [state_to_features(game_state)])[0]
    best_value = max(values[index] for index in candidates)
    best = [index for index in candidates if values[index] == best_value]
    return ACTIONS[int(self.rng.choice(best))]
