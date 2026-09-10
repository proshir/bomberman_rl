"""One-step Q-learning with a coin reward and a small cost per action."""

import pickle

import numpy as np

import events as e
from .callbacks import ACTIONS, state_to_features, get_q_values

LEARNING_RATE = 0.1       # Move 10% of the way toward each new target.
DISCOUNT = 0.95          # Give nearby future rewards more weight.
EPSILON_START = 1.0      # Begin with uniform exploration.
EPSILON_MIN = 0.05       # Keep some exploration throughout training.
EPSILON_DECAY = 0.995    # Multiply epsilon once per completed round.
COIN_REWARD = 1.0        # Match the game's reward for collecting a coin.
STEP_COST = 0.01         # Charge every action, including WAIT and blocked moves.


def setup_training(self):
    """Initialize learning settings and counters for the first training round."""
    self.learning_rate = LEARNING_RATE
    self.discount = DISCOUNT
    self.epsilon = EPSILON_START
    self.round_reward = 0.0
    self.round_steps = 0
    self.last_update = None


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: list):
    """Learn one intermediate transition and accumulate its reward and step."""
    if old_game_state is None or self_action is None:
        return
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    reward = reward_from_events(events)
    previous_q = float(get_q_values(self, old_state)[ACTIONS.index(self_action)])
    self.last_update = (old_game_state['round'], old_game_state['step'], previous_q, reward)

    update_q_value(self, old_state, self_action, new_state, reward)
    self.round_reward += reward
    self.round_steps += 1


def update_q_value(self, state, action, next_state, reward):
    """Update only the chosen Q entry; terminal targets contain no future value."""
    action_index = ACTIONS.index(action)
    q_values = get_q_values(self, state)
    target = reward
    if next_state is not None:
        target += self.discount * np.max(get_q_values(self, next_state))
    q_values[action_index] += self.learning_rate * (target - q_values[action_index])


def reward_from_events(events: list) -> float:
    """Reward each collected coin and subtract the step cost exactly once."""
    return events.count(e.COIN_COLLECTED) * COIN_REWARD - STEP_COST


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """Learn the final transition, save this run's table, and decay exploration."""
    if last_game_state is not None and last_action is not None:
        last_state = state_to_features(last_game_state)
        reward = reward_from_events(events)
        # The framework also sends this action through game_events_occurred.
        # Replace that update with the terminal update and count the action once.
        if (self.last_update is not None and
                self.last_update[:2] == (last_game_state['round'], last_game_state['step'])):
            get_q_values(self, last_state)[ACTIONS.index(last_action)] = self.last_update[2]
            self.round_reward -= self.last_update[3]
            self.round_steps -= 1

        update_q_value(self, last_state, last_action, None, reward)
        self.round_reward += reward
        self.round_steps += 1
    round_number = last_game_state['round'] if last_game_state is not None else None
    self.logger.info(
        f'Round {round_number}: reward={self.round_reward:.2f}, '
        f'steps={self.round_steps}, epsilon={self.epsilon:.4f}, '
        f'table_size={len(self.q_table)}')
    with open(self.model_path, 'wb') as file:
        pickle.dump(self.q_table, file)
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.last_round_reward = self.round_reward
    self.last_update = None
    self.round_reward = 0.0
    self.round_steps = 0
