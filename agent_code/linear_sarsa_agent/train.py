"""Semi-gradient SARSA(lambda) with accumulating eligibility traces."""

import pickle

import numpy as np

import events as e
from .callbacks import ACTIONS, encode_state

LEARNING_RATE = 0.1
DISCOUNT = 0.95
TRACE_DECAY = 0.8
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
COIN_REWARD = 1.0
STEP_COST = 0.01


def setup_training(self):
    self.epsilon = EPSILON_START
    self.traces = np.zeros_like(self.weights)
    self.pending = None
    self.round_reward = 0.0
    self.round_steps = 0


def reward_from_events(events):
    return events.count(e.COIN_COLLECTED) * COIN_REWARD - STEP_COST


def update(self, features, action, reward, next_value):
    error = reward + DISCOUNT * next_value - self.weights[action] @ features
    self.traces *= DISCOUNT * TRACE_DECAY
    self.traces[action] += features
    self.weights += LEARNING_RATE * error * self.traces
    self.round_reward += reward
    self.round_steps += 1


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action is None:
        return
    # Wait for the next chosen action. The final step also reaches end_of_round.
    self.pending = (encode_state(old_game_state), ACTIONS.index(self_action),
                    reward_from_events(events))


def end_of_round(self, last_game_state, last_action, events):
    if last_game_state is not None and last_action is not None:
        update(self, encode_state(last_game_state), ACTIONS.index(last_action),
               reward_from_events(events), 0.0)
    self.last_round_reward = self.round_reward
    self.last_round_steps = self.round_steps
    with open(self.model_path, 'wb') as file:
        pickle.dump(self.weights, file)
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.traces.fill(0.0)
    self.pending = None
    self.round_reward = 0.0
    self.round_steps = 0
