"""Preallocated, device-aware replay for Agent 032.

The older agents keep their Python-list replay implementation.  This variant
uses fixed NumPy arrays for the circular store, reusable host staging arrays,
and reusable tensors for each training device/batch size.  The sampled values
and scenario quotas remain the same as the Agent 030 replay contract.
"""

from collections import defaultdict

import numpy as np
import torch

from .dep_Agent_030_combat_ddqn_escape_replay_agent_replay import (
    ESCAPE_FRACTION,
    ESCAPE_TAG,
)
from .dep_Agent_032_combat_ddqn_optimized_features_agent_features import (
    FEATURE_SIZE,
)
from .dep_combat_dqn_agent_replay import TransitionBatch


class ScenarioReplayBuffer:
    """Bounded, scenario-tagged replay backed by preallocated arrays."""

    def __init__(self, capacity, seed=None, n_actions=6,
                 state_dim=FEATURE_SIZE):
        self.capacity = int(capacity)
        self.n_actions = int(n_actions)
        self.state_dim = int(state_dim)
        self.states = np.empty(
            (self.capacity, self.state_dim), dtype=np.float32
        )
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=np.bool_)
        self.next_action_masks = np.empty(
            (self.capacity, self.n_actions), dtype=np.bool_
        )
        self.position = 0
        self._size = 0
        self._tags = [None] * self.capacity
        self.indices_by_tag = defaultdict(set)
        self.current_tag = "coin-heaven"
        self.weights = {"coin-heaven": 1.0}
        self.rng = np.random.default_rng(seed)
        self._host_batches = {}
        self._device_batches = {}

    def __len__(self):
        return self._size

    @property
    def tags(self):
        """Compatibility view of tags for the populated part of the buffer."""
        return self._tags[:self._size]

    @property
    def storage(self):
        """Compatibility view without keeping a duplicate tuple store."""
        return [
            (
                self.states[index],
                int(self.actions[index]),
                float(self.rewards[index]),
                None if self.dones[index] else self.next_states[index],
                bool(self.dones[index]),
                self.next_action_masks[index],
            )
            for index in range(self._size)
        ]

    def set_context(self, tag, weights):
        self.current_tag = str(tag)
        self.weights = {
            str(name): max(0.0, float(weight))
            for name, weight in weights.items()
        }

    def add(self, state, action, reward, next_state, done,
            next_action_mask=None):
        """Write one transition into the fixed circular arrays."""
        if next_action_mask is None:
            next_action_mask = (
                np.zeros(self.n_actions, dtype=bool)
                if next_state is None or done
                else np.ones(self.n_actions, dtype=bool)
            )
        next_action_mask = np.asarray(next_action_mask, dtype=bool)
        if next_action_mask.shape != (self.n_actions,):
            raise ValueError(
                f"next_action_mask must have shape ({self.n_actions},), "
                f"got {next_action_mask.shape}"
            )

        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.state_dim,):
            raise ValueError(
                f"state must have shape ({self.state_dim},), "
                f"got {state_array.shape}"
            )
        if next_state is None:
            next_state_array = np.zeros(self.state_dim, dtype=np.float32)
        else:
            next_state_array = np.asarray(next_state, dtype=np.float32)
            if next_state_array.shape != (self.state_dim,):
                raise ValueError(
                    f"next_state must have shape ({self.state_dim},), "
                    f"got {next_state_array.shape}"
                )

        index = self.position
        replacing = self._size >= self.capacity
        if replacing:
            old_tag = self._tags[index]
            self.indices_by_tag[old_tag].discard(index)

        np.copyto(self.states[index], state_array, casting="unsafe")
        np.copyto(self.next_states[index], next_state_array, casting="unsafe")
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.dones[index] = bool(done)
        np.copyto(self.next_action_masks[index], next_action_mask)

        if not replacing:
            self._size += 1
        self._tags[index] = self.current_tag
        self.indices_by_tag[self.current_tag].add(index)
        self.position = (self.position + 1) % self.capacity

    def _sample_indices(self, batch_size):
        if batch_size > self._size:
            raise ValueError("Not enough transitions in replay buffer.")

        available = {
            tag: np.fromiter(indices, dtype=np.int64, count=len(indices))
            for tag, indices in self.indices_by_tag.items() if indices
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
            tag: min(len(indices), int(batch_size * weight / total_weight))
            for tag, indices in available.items()
            for weight in [active_weights[tag]]
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

        chosen = []
        for tag, count in counts.items():
            if count:
                chosen.extend(
                    self.rng.choice(available[tag], count, replace=False)
                )
        if len(chosen) < batch_size:
            used = set(int(index) for index in chosen)
            leftovers = np.asarray(
                [index for index in range(self._size) if index not in used],
                dtype=np.int64,
            )
            chosen.extend(
                self.rng.choice(leftovers, batch_size - len(chosen),
                                replace=False)
            )
        chosen = np.asarray(chosen, dtype=np.int64)
        self.rng.shuffle(chosen)
        return chosen

    def _new_host_batch(self, batch_size, pin_memory=False):
        if pin_memory:
            # A pinned NumPy view allows a genuine non-blocking H2D copy.
            def pinned(shape, dtype):
                tensor_dtype = {
                    np.dtype(np.float32): torch.float32,
                    np.dtype(np.int64): torch.int64,
                    np.dtype(np.bool_): torch.bool,
                }[np.dtype(dtype)]
                return torch.empty(
                    shape, dtype=tensor_dtype, pin_memory=True
                ).numpy()
        else:
            def pinned(shape, dtype):
                return np.empty(shape, dtype=dtype)

        return TransitionBatch(
            pinned((batch_size, self.state_dim), np.float32),
            pinned((batch_size,), np.int64),
            pinned((batch_size,), np.float32),
            pinned((batch_size, self.state_dim), np.float32),
            pinned((batch_size,), np.float32),
            pinned((batch_size, self.n_actions), np.bool_),
        )

    def _sample_into(self, indices, batch_size, pin_memory=False):
        key = (batch_size, bool(pin_memory))
        host = self._host_batches.get(key)
        if host is None:
            host = self._new_host_batch(batch_size, pin_memory=pin_memory)
            self._host_batches[key] = host
        np.take(self.states, indices, axis=0, out=host.states)
        np.take(self.actions, indices, axis=0, out=host.actions)
        np.take(self.rewards, indices, axis=0, out=host.rewards)
        np.take(self.next_states, indices, axis=0, out=host.next_states)
        np.copyto(host.dones, self.dones[indices], casting="unsafe")
        np.take(self.next_action_masks, indices, axis=0,
                out=host.next_action_masks)
        return host

    def sample(self, batch_size):
        """Return a NumPy batch, retaining the old public API."""
        indices = self._sample_indices(batch_size)
        host = self._sample_into(indices, int(batch_size))
        return TransitionBatch(*(array.copy() for array in host))

    def sample_torch(self, batch_size, device):
        """Sample into reusable tensors and perform one transfer per field."""
        batch_size = int(batch_size)
        indices = self._sample_indices(batch_size)
        host = self._sample_into(
            indices, batch_size, pin_memory=device.type == "cuda"
        )
        device = torch.device(device)
        key = (str(device), batch_size)
        if device.type == "cpu":
            return TransitionBatch(*(torch.from_numpy(array) for array in host))

        device_batch = self._device_batches.get(key)
        if device_batch is None:
            device_batch = TransitionBatch(
                torch.empty((batch_size, self.state_dim),
                            dtype=torch.float32, device=device),
                torch.empty(batch_size, dtype=torch.int64, device=device),
                torch.empty(batch_size, dtype=torch.float32, device=device),
                torch.empty((batch_size, self.state_dim),
                            dtype=torch.float32, device=device),
                torch.empty(batch_size, dtype=torch.float32, device=device),
                torch.empty((batch_size, self.n_actions),
                            dtype=torch.bool, device=device),
            )
            self._device_batches[key] = device_batch
        for target, source in zip(device_batch, host):
            target.copy_(torch.from_numpy(source), non_blocking=True)
        return device_batch


class CombatEscapeReplayBuffer(ScenarioReplayBuffer):
    """Reserve replay capacity for combat bomb-escape transitions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transition_tag = None

    def set_context(self, tag, weights):
        super().set_context(
            tag,
            {name: (1.0 - ESCAPE_FRACTION) * weight
             for name, weight in weights.items()} | {ESCAPE_TAG: ESCAPE_FRACTION},
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
