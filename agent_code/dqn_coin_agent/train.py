"""Replay-buffer training callback for the compact coin-navigation DQN.

Sahand was here.

This is a deliberately conventional DQN: replay memory, a separate target
network, Huber loss, gradient clipping, and legal-action masking.  The callback
does not assume CUDA; the runner records whether the training process used it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

import events as e
from .callbacks import ACTIONS, legal_action_indices
from .features import state_to_features
from .model import QNetwork


DISCOUNT = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
REPLAY_CAPACITY = 100_000
WARMUP_TRANSITIONS = 2_000
TARGET_UPDATE_STEPS = 1_000
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 50_000
COIN_REWARD = 1.0
STEP_COST = 0.01
INVALID_PENALTY = 0.1
SURVIVAL_REWARD = 0.2


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray | None
    next_legal: tuple[int, ...]
    done: bool


class ReplayBuffer:
    """Bounded transition memory with deterministic seeded sampling."""

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.data = deque(maxlen=capacity)
        self.rng = rng

    def append(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, count: int) -> list[Transition]:
        indices = self.rng.choice(len(self.data), size=count, replace=False)
        return [self.data[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self.data)


def setup_training(self):
    """Initialize the target network, optimizer, replay memory, and counters."""
    self.target_model = QNetwork(self.model.network[0].in_features, len(ACTIONS)).to(self.device)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    self.replay = ReplayBuffer(REPLAY_CAPACITY, self.rng)
    self.total_steps = 0
    self.update_count = 0
    self.last_loss = None
    self.last_round_reward = 0.0
    self.round_reward = 0.0
    self.pending_transition = None
    self.last_transition_id = None
    self.epsilon = EPSILON_START


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: list):
    """Queue one transition and flush the previous one as nonterminal."""
    if old_game_state is None or self_action not in ACTIONS:
        return
    _flush_pending(self, terminal=False)
    self.pending_transition = _make_transition(
        old_game_state, self_action, new_game_state, events, done=False)
    self.last_transition_id = (old_game_state['round'], old_game_state['step'])


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """Flush the final transition exactly once and reset round counters."""
    transition_id = None
    if last_game_state is not None:
        transition_id = (last_game_state['round'], last_game_state['step'])
    if (self.pending_transition is not None and transition_id == self.last_transition_id):
        # The callback for the final alive step saw a nonterminal next state;
        # the round callback supplies the terminal reward and closes it here.
        self.pending_transition = _make_transition(
            last_game_state, last_action, None, events, done=True)
    elif last_game_state is not None and last_action in ACTIONS:
        # A dead agent does not receive game_events_occurred for its final move.
        _flush_pending(self, terminal=False)
        self.pending_transition = _make_transition(
            last_game_state, last_action, None, events, done=True)
    _flush_pending(self, terminal=True)
    self.last_round_reward = self.round_reward
    self.round_reward = 0.0
    self.pending_transition = None
    self.last_transition_id = None


def _make_transition(old_state, action, new_state, events, done):
    next_features = state_to_features(new_state) if new_state is not None else None
    next_legal = tuple(legal_action_indices(new_state)) if new_state is not None else ()
    return Transition(
        state=state_to_features(old_state),
        action=ACTIONS.index(action),
        reward=reward_from_events(events),
        next_state=next_features,
        next_legal=next_legal,
        done=done,
    )


def _flush_pending(self, terminal):
    if self.pending_transition is None:
        return
    transition = self.pending_transition
    self.replay.append(transition)
    self.round_reward += transition.reward
    self.total_steps += 1
    self.epsilon = max(
        EPSILON_MIN,
        EPSILON_START - (EPSILON_START - EPSILON_MIN)
        * min(self.total_steps, EPSILON_DECAY_STEPS) / EPSILON_DECAY_STEPS,
    )
    if len(self.replay) >= WARMUP_TRANSITIONS:
        self.last_loss = _learn(self)
    self.pending_transition = None


def _learn(self):
    batch = self.replay.sample(BATCH_SIZE)
    states = torch.as_tensor(np.stack([item.state for item in batch]),
                             dtype=torch.float32, device=self.device)
    actions = torch.as_tensor([item.action for item in batch], dtype=torch.long,
                              device=self.device)
    rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32,
                              device=self.device)
    done = torch.as_tensor([item.done for item in batch], dtype=torch.float32,
                           device=self.device)

    predictions = self.model(states).gather(1, actions[:, None]).squeeze(1)
    targets = rewards.clone()
    nonterminal = [index for index, item in enumerate(batch)
                   if not item.done and item.next_state is not None and item.next_legal]
    if nonterminal:
        next_states = torch.as_tensor(
            np.stack([batch[index].next_state for index in nonterminal]),
            dtype=torch.float32, device=self.device)
        online_next = self.model(next_states)
        legal_mask = torch.full_like(online_next, float('-inf'))
        for row, index in enumerate(nonterminal):
            legal_mask[row, list(batch[index].next_legal)] = 0.0
        chosen = (online_next + legal_mask).argmax(dim=1)
        target_next = self.target_model(next_states).gather(1, chosen[:, None]).squeeze(1)
        targets[nonterminal] += DISCOUNT * target_next.detach()

    loss = F.smooth_l1_loss(predictions, targets.detach())
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
    self.optimizer.step()
    self.update_count += 1
    if self.update_count % TARGET_UPDATE_STEPS == 0:
        self.target_model.load_state_dict(self.model.state_dict())
    return float(loss.detach().cpu())


def reward_from_events(events: list) -> float:
    """Use only game outcomes plus a small time cost for coin navigation."""
    reward = events.count(e.COIN_COLLECTED) * COIN_REWARD
    reward -= STEP_COST
    reward -= events.count(e.INVALID_ACTION) * INVALID_PENALTY
    reward += events.count(e.SURVIVED_ROUND) * SURVIVAL_REWARD
    return float(reward)
