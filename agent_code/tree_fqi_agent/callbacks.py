"""Coin navigation with one fitted regression tree per action."""

import pickle
from pathlib import Path

import numpy as np

from .features import ACTIONS, legal_action_indices, state_to_features

MODEL_PATH = Path(__file__).resolve().parent / 'trees.pkl'
FEATURE_MODE = 'distance'


def setup(self):
    if FEATURE_MODE != 'distance':
        raise ValueError('Tree FQI uses distance features only.')
    self.rng = np.random.default_rng(getattr(self, 'seed', None))
    self.model_path = Path(getattr(self, 'model_path', MODEL_PATH))
    if self.train:
        if self.model_path.exists():
            raise FileExistsError(f'Checkpoint already exists: {self.model_path}')
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.trees = [None] * len(ACTIONS)
    else:
        with open(self.model_path, 'rb') as file:
            self.trees = pickle.load(file)
        if len(self.trees) != len(ACTIONS):
            raise ValueError('Checkpoint has the wrong number of trees.')


def predict_values(trees, states):
    """Unobserved actions start with value zero."""
    values = np.zeros((len(states), len(ACTIONS)))
    for action, tree in enumerate(trees):
        if tree is not None:
            values[:, action] = tree.predict(states)
    return values


def act(self, game_state):
    if game_state is None:
        return 'WAIT'
    legal = legal_action_indices(game_state)
    if self.train and self.rng.random() < self.epsilon:
        action = int(self.rng.choice(legal))
    else:
        values = predict_values(self.trees, [state_to_features(game_state)])[0]
        best_value = max(values[index] for index in legal)
        best = [index for index in legal if values[index] == best_value]
        action = int(self.rng.choice(best))
    return ACTIONS[action]
