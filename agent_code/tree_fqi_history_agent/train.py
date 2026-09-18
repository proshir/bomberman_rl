"""Fitted-Q training for the tree agent with learned history inputs."""

import pickle
import os
from collections import deque

import numpy as np
from sklearn.tree import DecisionTreeRegressor

import events as e
from .callbacks import ACTIONS, legal_action_indices, next_features, predict_values, state_key

DISCOUNT = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
COIN_REWARD = 1.0
STEP_COST = 0.01
REVISIT_PENALTY = float(os.environ.get('TREE_REVISIT_PENALTY', '0'))
BUFFER_SIZE = 30000
FIT_ITERATIONS = 5
MAX_DEPTH = 8
MIN_SAMPLES_LEAF = 5


def setup_training(self):
    self.epsilon = EPSILON_START
    self.transitions = deque(maxlen=BUFFER_SIZE)
    self.pending = None
    self.round_reward = 0.0
    self.round_steps = 0


def reward_from_events(events):
    return events.count(e.COIN_COLLECTED) * COIN_REWARD - STEP_COST


def remember(self, state, action, next_state, events):
    cached = self.feature_cache[state_key(state)]
    features = cached[0]
    recent_positions = cached[1]
    revisit = (next_state is not None and
               next_state['self'][3] in recent_positions and
               e.COIN_COLLECTED not in events)
    reward = reward_from_events(events) - (REVISIT_PENALTY if revisit else 0.0)
    legal = np.zeros(len(ACTIONS), dtype=bool)
    future = None
    if next_state is not None:
        future = next_features(self, state, action, next_state)
        legal[legal_action_indices(next_state)] = True
    self.transitions.append((features, ACTIONS.index(action), reward, future, legal))
    self.round_reward += reward
    self.round_steps += 1


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action is None:
        return
    if self.pending is not None:
        remember(self, *self.pending)
    self.pending = (old_game_state, self_action, new_game_state, list(events))


def fit_trees(self):
    if not self.transitions:
        return
    states = np.asarray([row[0] for row in self.transitions])
    actions = np.asarray([row[1] for row in self.transitions])
    rewards = np.asarray([row[2] for row in self.transitions])
    continuing = np.asarray([row[3] is not None for row in self.transitions])
    next_states = np.asarray([row[3] for row in self.transitions if row[3] is not None])
    next_legal = np.asarray([row[4] for row in self.transitions if row[3] is not None])
    for _ in range(FIT_ITERATIONS):
        targets = rewards.copy()
        if continuing.any():
            values = predict_values(self.trees, next_states)
            values[~next_legal] = -np.inf
            targets[continuing] += DISCOUNT * values.max(axis=1)
        for action in range(len(ACTIONS)):
            selected = actions == action
            if selected.any():
                tree = DecisionTreeRegressor(max_depth=MAX_DEPTH,
                                             min_samples_leaf=MIN_SAMPLES_LEAF,
                                             random_state=getattr(self, 'seed', 0))
                tree.fit(states[selected], targets[selected])
                self.trees[action] = tree


def end_of_round(self, last_game_state, last_action, events):
    if self.pending is not None:
        state = self.pending[0]
        if (last_game_state is None or (state['round'], state['step']) !=
                (last_game_state['round'], last_game_state['step'])):
            remember(self, *self.pending)
    if last_game_state is not None and last_action is not None:
        remember(self, last_game_state, last_action, None, events)
    self.pending = None
    fit_trees(self)
    self.last_round_reward = self.round_reward
    self.last_round_steps = self.round_steps
    with open(self.model_path, 'wb') as file:
        pickle.dump(self.trees, file)
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.round_reward = 0.0
    self.round_steps = 0
