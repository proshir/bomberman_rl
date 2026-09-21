"""Fitted-Q training for the topology-augmented combat agent."""

# Sahand was here.

import pickle
from collections import defaultdict, deque

import numpy as np
from sklearn.tree import DecisionTreeRegressor

import events as e
from .callbacks import next_features, predict_values, state_key
from .features import ACTIONS
from .safety import (best_survival_action_indices, bomb_is_useful,
                     earliest_danger, safe_action_indices)


DISCOUNT = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
BUFFER_SIZE = 30000
FIT_ITERATIONS = 3
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 5

COIN_REWARD = 1.0
KILL_REWARD = 5.0
CRATE_REWARD = 0.2
COIN_FOUND_REWARD = 0.2
SURVIVAL_REWARD = 0.5
DEATH_PENALTY = 5.0
INVALID_ACTION_PENALTY = 1.0
USELESS_BOMB_PENALTY = 0.1
ESCAPE_REWARD = 0.1
STEP_COST = 0.01


def setup_training(self):
    self.epsilon = EPSILON_START
    self.transitions = deque(maxlen=BUFFER_SIZE)
    self.pending = None
    self.round_reward = 0.0
    self.round_steps = 0
    self.round_events = defaultdict(int)


def reward_from_transition(old_state, action, new_state, events):
    """Use the controlled combat reward without topology-specific shaping."""
    reward = -STEP_COST
    reward += events.count(e.COIN_COLLECTED) * COIN_REWARD
    reward += events.count(e.KILLED_OPPONENT) * KILL_REWARD
    reward += events.count(e.CRATE_DESTROYED) * CRATE_REWARD
    reward += events.count(e.COIN_FOUND) * COIN_FOUND_REWARD
    reward += events.count(e.SURVIVED_ROUND) * SURVIVAL_REWARD
    reward -= events.count(e.INVALID_ACTION) * INVALID_ACTION_PENALTY
    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        reward -= DEATH_PENALTY
    if action == "BOMB" and not bomb_is_useful(old_state):
        reward -= USELESS_BOMB_PENALTY
    if new_state is not None:
        old_time = earliest_danger(old_state, old_state["self"][3])
        new_time = earliest_danger(new_state, new_state["self"][3])
        if old_time and (not new_time or new_time > old_time):
            reward += ESCAPE_REWARD
    return reward


def remember(self, old_state, action, new_state, events):
    reward = reward_from_transition(old_state, action, new_state, events)
    legal = np.zeros(len(ACTIONS), dtype=bool)
    future = None
    if new_state is not None:
        future = next_features(self, old_state, action, new_state)
        candidates = safe_action_indices(new_state)
        if not candidates:
            candidates = best_survival_action_indices(new_state)
        legal[candidates] = True
    features = self.feature_cache[state_key(old_state)][0]
    self.transitions.append(
        (features, ACTIONS.index(action), reward, future, legal)
    )
    self.round_reward += reward
    self.round_steps += 1
    for event in events:
        self.round_events[event] += 1


def game_events_occurred(self, old_game_state, self_action,
                         new_game_state, events):
    """Delay one transition so the final callback can mark it terminal."""
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
    next_states = np.asarray([
        row[3] for row in self.transitions if row[3] is not None
    ])
    next_legal = np.asarray([
        row[4] for row in self.transitions if row[3] is not None
    ])
    for _ in range(FIT_ITERATIONS):
        targets = rewards.copy()
        if continuing.any():
            values = predict_values(self.trees, next_states)
            values[~next_legal] = -np.inf
            targets[continuing] += DISCOUNT * values.max(axis=1)
        replacements = list(self.trees)
        for action in range(len(ACTIONS)):
            selected = actions == action
            if selected.any():
                tree = DecisionTreeRegressor(
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    random_state=getattr(self, "seed", 0),
                )
                tree.fit(states[selected], targets[selected])
                replacements[action] = tree
        self.trees = replacements


def end_of_round(self, last_game_state, last_action, events):
    """Store the terminal transition, fit, and save the trees."""
    if self.pending is not None:
        pending_state = self.pending[0]
        pending_is_final = (
            last_game_state is not None and
            state_key(pending_state) == state_key(last_game_state)
        )
        if not pending_is_final:
            remember(self, *self.pending)
    if last_game_state is not None and last_action is not None:
        remember(self, last_game_state, last_action, None, list(events))
    self.pending = None
    fit_trees(self)
    self.last_round_reward = self.round_reward
    self.last_round_steps = self.round_steps
    self.last_round_events = dict(self.round_events)
    with open(self.model_path, "wb") as file:
        pickle.dump(self.trees, file)
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.round_reward = 0.0
    self.round_steps = 0
    self.round_events.clear()
