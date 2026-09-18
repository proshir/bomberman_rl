"""Bounded replay buffer for combat Double-DQN."""

# Sahand was here.

from collections import namedtuple

import numpy as np


TransitionBatch = namedtuple(
    "TransitionBatch",
    "states actions rewards next_states dones next_action_masks",
)


class ReplayBuffer:
    """Fixed-capacity replay storage with deterministic seeded sampling."""

    def __init__(self, capacity, seed=None):
        if int(capacity) < 1:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self._storage = []
        self._position = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done, next_action_mask):
        state = np.asarray(state, dtype=np.float32).copy()
        next_state = None if next_state is None else np.asarray(
            next_state, dtype=np.float32).copy()
        mask = np.asarray(next_action_mask, dtype=bool).copy()
        transition = (state, int(action), float(reward), next_state,
                      bool(done), mask)
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size):
        if int(batch_size) > len(self._storage):
            raise ValueError("Cannot sample more transitions than stored.")
        indices = self._rng.choice(len(self._storage), int(batch_size), replace=False)
        rows = [self._storage[int(index)] for index in indices]
        states, actions, rewards, next_states, dones, masks = zip(*rows)
        reference = states[0]
        next_states = [np.zeros_like(reference) if state is None else state
                       for state in next_states]
        return TransitionBatch(
            states=np.stack(states).astype(np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_states=np.stack(next_states).astype(np.float32),
            dones=np.asarray(dones, dtype=bool),
            next_action_masks=np.stack(masks).astype(bool),
        )
