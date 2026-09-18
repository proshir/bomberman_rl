"""Tree fitted-Q navigation with learned recent-movement inputs."""

import pickle
from collections import deque
from pathlib import Path

import numpy as np

from .features import ACTIONS, legal_action_indices, state_to_features

MODEL_PATH = Path(__file__).resolve().parent / 'trees.pkl'
FEATURE_MODE = 'distance'
HISTORY_LENGTH = 8


def state_key(game_state):
    return game_state['round'], game_state['step']


def setup(self):
    if FEATURE_MODE != 'distance':
        raise ValueError('Tree FQI history agent uses distance mode only.')
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
    self.round_id = None
    self.last_coins = None
    self.previous_action = ACTIONS.index('WAIT')
    self.positions = deque(maxlen=HISTORY_LENGTH)
    self.feature_cache = {}


def predict_values(trees, states):
    values = np.zeros((len(states), len(ACTIONS)))
    for action, tree in enumerate(trees):
        if tree is not None:
            values[:, action] = tree.predict(states)
    return values


def features_for_act(self, game_state):
    if game_state['round'] != self.round_id:
        self.round_id = game_state['round']
        self.positions.clear()
        self.last_coins = None
        self.previous_action = ACTIONS.index('WAIT')
    coins = tuple(sorted(game_state['coins']))
    if coins != self.last_coins:
        self.positions.clear()
        self.last_coins = coins
    position = game_state['self'][3]
    feature = state_to_features(game_state, self.previous_action,
                                self.positions.count(position))
    self.feature_cache[state_key(game_state)] = (
        feature, tuple(self.positions), coins, self.previous_action)
    self.positions.append(position)
    return feature


def next_features(self, old_state, action, new_state):
    """Construct the history inputs that the next decision will observe."""
    feature, history, old_coins, _ = self.feature_cache[state_key(old_state)]
    del feature
    history = deque(history, maxlen=HISTORY_LENGTH)
    history.append(old_state['self'][3])
    new_coins = tuple(sorted(new_state['coins']))
    if new_coins != old_coins:
        history.clear()
    return state_to_features(new_state, ACTIONS.index(action),
                             history.count(new_state['self'][3]))


def act(self, game_state):
    if game_state is None:
        return 'WAIT'
    state = features_for_act(self, game_state)
    legal = legal_action_indices(game_state)
    if self.train and self.rng.random() < self.epsilon:
        action = int(self.rng.choice(legal))
    else:
        values = predict_values(self.trees, [state])[0]
        best_value = max(values[index] for index in legal)
        best = [index for index in legal if values[index] == best_value]
        action = int(self.rng.choice(best))
    self.previous_action = action
    return ACTIONS[action]
