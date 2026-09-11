"""Linear action values for the same coin features as the masked Q-table."""

import pickle
from pathlib import Path

import numpy as np

from .features import ACTIONS, legal_action_indices, state_to_features

MODEL_PATH = Path(__file__).resolve().parent / 'weights.pkl'
FEATURE_MODE = 'distance'
FEATURE_SIZES = (2, 2, 2, 2, 3, 3, 6, 5)
N_FEATURES = 1 + sum(FEATURE_SIZES)


def encode_state(game_state):
    """One category per input plus a bias; no new game information."""
    state = state_to_features(game_state)
    categories = list(state)
    categories[4] += 1
    categories[5] += 1
    features = np.zeros(N_FEATURES)
    features[0] = 1.0
    offset = 1
    for category, size in zip(categories, FEATURE_SIZES):
        features[offset + category] = 1.0
        offset += size
    # Nine active entries: unit norm keeps the learning-rate scale manageable.
    return features / 3.0


def setup(self):
    if FEATURE_MODE != 'distance':
        raise ValueError('Linear SARSA uses distance features only.')
    self.rng = np.random.default_rng(getattr(self, 'seed', None))
    self.model_path = Path(getattr(self, 'model_path', MODEL_PATH))
    if self.train:
        if self.model_path.exists():
            raise FileExistsError(f'Checkpoint already exists: {self.model_path}')
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.weights = np.zeros((len(ACTIONS), N_FEATURES))
    else:
        with open(self.model_path, 'rb') as file:
            self.weights = pickle.load(file)
        if self.weights.shape != (len(ACTIONS), N_FEATURES):
            raise ValueError('Checkpoint has the wrong weight shape.')
        if not np.isfinite(self.weights).all():
            raise ValueError('Checkpoint contains non-finite weights.')


def act(self, game_state):
    if game_state is None:
        return 'WAIT'
    features = encode_state(game_state)
    values = self.weights @ features
    legal = legal_action_indices(game_state)
    if self.train and self.rng.random() < self.epsilon:
        action = int(self.rng.choice(legal))
    else:
        best_value = max(values[index] for index in legal)
        best = [index for index in legal if values[index] == best_value]
        action = int(self.rng.choice(best))

    if self.train and self.pending is not None:
        from .train import update
        old_features, old_action, reward = self.pending
        # Use the action we will actually take, selected before the update.
        update(self, old_features, old_action, reward, values[action])
        self.pending = None
    return ACTIONS[action]
