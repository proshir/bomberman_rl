"""One-step Q-learning with legal-action bootstrap targets."""

import pickle

import numpy as np

import events as e
from .callbacks import ACTIONS, get_q_values, legal_action_indices, state_to_features

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
COIN_REWARD = 1.0
STEP_COST = 0.01


def setup_training(self):
    """Initialize Q-learning settings and per-round counters."""
    self.learning_rate = LEARNING_RATE
    self.discount = DISCOUNT
    self.epsilon = EPSILON_START
    self.round_reward = 0.0
    self.round_steps = 0
    self.last_update = None


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: list):
    """Update one nonterminal transition and record its final-step identity."""
    if old_game_state is None or self_action is None:
        return
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    reward = reward_from_events(events)
    previous_q = float(get_q_values(self, old_state)[ACTIONS.index(self_action)])
    self.last_update = (old_game_state['round'], old_game_state['step'], previous_q, reward)
    update_q_value(self, old_state, self_action, new_state, reward, new_game_state)
    self.round_reward += reward
    self.round_steps += 1


def update_q_value(self, state, action, next_state, reward, next_game_state=None):
    """Update a selected action and bootstrap only over legal next actions."""
    action_index = ACTIONS.index(action)
    q_values = get_q_values(self, state)
    target = reward
    if next_state is not None:
        if next_game_state is None:
            next_values = get_q_values(self, next_state)
            target += self.discount * np.max(next_values)
        else:
            legal = legal_action_indices(next_game_state)
            next_values = get_q_values(self, next_state)
            target += self.discount * max(next_values[index] for index in legal)
    q_values[action_index] += self.learning_rate * (target - q_values[action_index])


def reward_from_events(events: list) -> float:
    """Reward collected coins and charge one small cost for the action."""
    return events.count(e.COIN_COLLECTED) * COIN_REWARD - STEP_COST


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """Apply the terminal transition once, save the table, and decay exploration."""
    if last_game_state is not None and last_action is not None:
        last_state = state_to_features(last_game_state)
        reward = reward_from_events(events)
        if (self.last_update is not None and
                self.last_update[:2] == (last_game_state['round'], last_game_state['step'])):
            get_q_values(self, last_state)[ACTIONS.index(last_action)] = self.last_update[2]
            self.round_reward -= self.last_update[3]
            self.round_steps -= 1
        update_q_value(self, last_state, last_action, None, reward)
        self.round_reward += reward
        self.round_steps += 1
    self.last_round_reward = self.round_reward
    with open(self.model_path, 'wb') as file:
        pickle.dump(self.q_table, file)
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.last_update = None
    self.round_reward = 0.0
    self.round_steps = 0
