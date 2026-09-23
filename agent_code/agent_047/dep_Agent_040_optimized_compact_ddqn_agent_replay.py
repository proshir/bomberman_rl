"""Preallocated replay with allocation-light scenario sampling."""

from collections import defaultdict

import numpy as np
import torch

from .dep_Agent_030_combat_ddqn_escape_replay_agent_replay import (
    ESCAPE_FRACTION,
    ESCAPE_TAG,
)
from .dep_Agent_032_combat_ddqn_optimized_features_agent_replay import (
    ScenarioReplayBuffer as _BaseScenarioReplayBuffer,
)
from .dep_combat_dqn_agent_replay import TransitionBatch

from .dep_Agent_040_optimized_compact_ddqn_agent_features import FEATURE_SIZE


class _IndexPool:
    """Dense index set supporting O(1) removal and allocation-free views."""

    def __init__(self, capacity):
        self.indices = np.empty(int(capacity), dtype=np.int64)
        self.positions = np.full(int(capacity), -1, dtype=np.int64)
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, index):
        index = int(index)
        if self.positions[index] >= 0:
            return
        self.positions[index] = self.size
        self.indices[self.size] = index
        self.size += 1

    def discard(self, index):
        index = int(index)
        position = int(self.positions[index])
        if position < 0:
            return
        last_position = self.size - 1
        last_index = int(self.indices[last_position])
        self.indices[position] = last_index
        self.positions[last_index] = position
        self.positions[index] = -1
        self.size = last_position

    def view(self):
        return self.indices[:self.size]


class ScenarioReplayBuffer(_BaseScenarioReplayBuffer):
    """Agent 032's fixed arrays with dense tag membership indexes."""

    def __init__(self, capacity, seed=None, n_actions=6,
                 state_dim=FEATURE_SIZE):
        # Reuse the array and tensor setup, then replace only the hot tag
        # membership representation.  The parent keeps all public APIs and
        # batch staging behavior unchanged.
        super().__init__(
            capacity, seed=seed, n_actions=n_actions, state_dim=state_dim,
        )
        self.indices_by_tag = defaultdict(lambda: _IndexPool(self.capacity))

    def add(self, state, action, reward, next_state, done,
            next_action_mask=None):
        # The parent implementation already performs the membership update;
        # _IndexPool supplies the same add/discard interface without the
        # per-sample set-to-array conversion.
        super().add(state, action, reward, next_state, done, next_action_mask)

    def _sample_indices(self, batch_size):
        batch_size = int(batch_size)
        if batch_size > self._size:
            raise ValueError("Not enough transitions in replay buffer.")

        available = {
            tag: pool for tag, pool in self.indices_by_tag.items()
            if len(pool)
        }
        if len(available) <= 1:
            return self.rng.choice(self._size, batch_size, replace=False)

        active_weights = {
            tag: self.weights.get(tag, 0.0) for tag in available
        }
        total_weight = sum(active_weights.values())
        if total_weight <= 0:
            active_weights = {tag: 1.0 for tag in available}
            total_weight = float(len(active_weights))

        counts = {
            tag: min(len(pool), int(batch_size * active_weights[tag] /
                                   total_weight))
            for tag, pool in available.items()
        }
        remaining = batch_size - sum(counts.values())
        order = sorted(available, key=lambda tag: (-active_weights[tag], tag))
        while remaining:
            eligible = [
                tag for tag in order if counts[tag] < len(available[tag])
            ]
            if not eligible:
                break
            for tag in eligible:
                if remaining == 0:
                    break
                counts[tag] += 1
                remaining -= 1

        chosen = np.empty(batch_size, dtype=np.int64)
        cursor = 0
        for tag, count in counts.items():
            if count:
                picked = self.rng.choice(
                    available[tag].view(), count, replace=False
                )
                chosen[cursor:cursor + count] = picked
                cursor += count

        if cursor < batch_size:
            used = np.zeros(self._size, dtype=bool)
            used[chosen[:cursor]] = True
            leftovers = np.flatnonzero(~used)
            count = batch_size - cursor
            chosen[cursor:] = self.rng.choice(
                leftovers, count, replace=False
            )
        self.rng.shuffle(chosen)
        return chosen


class CombatEscapeReplayBuffer(ScenarioReplayBuffer):
    """Reserve replay capacity for combat bomb-escape transitions."""

    def __init__(self, capacity, seed=None, n_actions=6, state_dim=FEATURE_SIZE):
        super().__init__(
            capacity, seed=seed, n_actions=n_actions, state_dim=state_dim,
        )
        self.transition_tag = None

    def set_context(self, tag, weights):
        super().set_context(
            tag,
            {
                name: (1.0 - ESCAPE_FRACTION) * weight
                for name, weight in weights.items()
            } | {ESCAPE_TAG: ESCAPE_FRACTION},
        )

    def add(self, *args, **kwargs):
        original_tag = self.current_tag
        if self.transition_tag is not None:
            self.current_tag = self.transition_tag
        try:
            super().add(*args, **kwargs)
        finally:
            self.current_tag = original_tag
            self.transition_tag = None


__all__ = [
    "CombatEscapeReplayBuffer", "ESCAPE_FRACTION", "ESCAPE_TAG",
    "ScenarioReplayBuffer",
]
