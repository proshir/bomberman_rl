"""Frozen tree FQI with an evaluation-time escape from repeated positions."""

from collections import deque
import pickle
from pathlib import Path

import numpy as np

from .features import ACTIONS, legal_action_indices, state_to_features

MODEL_PATH = Path(__file__).resolve().parent / 'trees.pkl'
HISTORY_LENGTH = 8
REPEAT_THRESHOLD = 3


def setup(self):
    if self.train:
        raise ValueError('The loop variant evaluates frozen checkpoints only.')
    self.rng = np.random.default_rng(getattr(self, 'seed', None))
    self.loop_rng = np.random.default_rng(getattr(self, 'seed', None))
    self.model_path = Path(getattr(self, 'model_path', MODEL_PATH))
    with open(self.model_path, 'rb') as file:
        self.trees = pickle.load(file)
    if len(self.trees) != len(ACTIONS):
        raise ValueError('Checkpoint has the wrong number of trees.')
    self.history = deque(maxlen=HISTORY_LENGTH)
    self.last_coins = None
    self.last_round = None
    self.loop_interventions = 0


def predict_values(trees, states):
    values = np.zeros((len(states), len(ACTIONS)))
    for action, tree in enumerate(trees):
        if tree is not None:
            values[:, action] = tree.predict(states)
    return values


def learned_action(self, game_state):
    legal = legal_action_indices(game_state)
    values = predict_values(self.trees, [state_to_features(game_state)])[0]
    best_value = max(values[index] for index in legal)
    best = [index for index in legal if values[index] == best_value]
    return ACTIONS[int(self.rng.choice(best))]


def act(self, game_state):
    if game_state is None:
        return 'WAIT'
    action = learned_action(self, game_state)
    if game_state['round'] != self.last_round:
        self.history.clear()
        self.last_coins = None
        self.loop_interventions = 0
        self.last_round = game_state['round']
    coins = tuple(sorted(game_state['coins']))
    if coins != self.last_coins:
        self.history.clear()
        self.last_coins = coins
    position = game_state['self'][3]
    self.history.append(position)
    if self.history.count(position) < REPEAT_THRESHOLD:
        return action
    alternatives = [ACTIONS[index] for index in legal_action_indices(game_state)
                    if ACTIONS[index] not in ('WAIT', action)]
    if not alternatives:
        return action
    self.loop_interventions += 1
    self.history.clear()
    return str(self.loop_rng.choice(alternatives))
