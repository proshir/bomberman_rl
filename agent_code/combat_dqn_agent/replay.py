"""Simple bounded replay buffer."""

# Sahand was here.

from collections import namedtuple

import numpy as np


TransitionBatch = namedtuple(
    "TransitionBatch", "states actions rewards next_states dones"
)


class ReplayBuffer:
    def __init__(self, capacity, seed=None):
        self.capacity = int(capacity)
        self.storage = []
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.storage)

    def add(self, state, action, reward, next_state, done):
        item = (
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            None if next_state is None else np.asarray(next_state, dtype=np.float32).copy(),
            bool(done),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size > len(self.storage):
            raise ValueError("Not enough transitions in replay buffer.")
        indices = self.rng.choice(len(self.storage), batch_size, replace=False)
        rows = [self.storage[int(i)] for i in indices]
        states, actions, rewards, next_states, dones = zip(*rows)
        reference = states[0]
        next_states = [reference * 0 if state is None else state
                       for state in next_states]
        return TransitionBatch(
            np.stack(states).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.asarray(dones, dtype=np.float32),
        )
