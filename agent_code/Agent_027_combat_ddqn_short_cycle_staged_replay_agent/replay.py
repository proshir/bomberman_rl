"""Scenario-tagged replay that keeps curriculum experience available."""

from collections import defaultdict

import numpy as np

from agent_code.combat_dqn_agent.replay import ReplayBuffer


class ScenarioReplayBuffer(ReplayBuffer):
    """Bounded replay with weighted sampling across observed scenarios.

    The storage remains one ordinary circular buffer.  Tags and index sets only
    control sampling, so transitions retain the exact existing DDQN format.
    Before all scenarios have appeared, unavailable quota is redistributed to
    the available ones rather than delaying optimization.
    """

    def __init__(self, capacity, seed=None, n_actions=6):
        super().__init__(capacity, seed=seed, n_actions=n_actions)
        self.tags = []
        self.indices_by_tag = defaultdict(set)
        self.current_tag = "coin-heaven"
        self.weights = {"coin-heaven": 1.0}

    def set_context(self, tag, weights):
        self.current_tag = str(tag)
        self.weights = {
            str(name): max(0.0, float(weight))
            for name, weight in weights.items()
        }

    def add(self, state, action, reward, next_state, done,
            next_action_mask=None):
        index = self.position
        replacing = len(self.storage) >= self.capacity
        if replacing:
            old_tag = self.tags[index]
            self.indices_by_tag[old_tag].discard(index)
        super().add(state, action, reward, next_state, done, next_action_mask)
        if replacing:
            self.tags[index] = self.current_tag
        else:
            self.tags.append(self.current_tag)
        self.indices_by_tag[self.current_tag].add(index)

    def sample(self, batch_size):
        if batch_size > len(self.storage):
            raise ValueError("Not enough transitions in replay buffer.")
        available = {
            tag: sorted(indices)
            for tag, indices in self.indices_by_tag.items() if indices
        }
        if len(available) <= 1:
            return super().sample(batch_size)

        active_weights = {
            tag: self.weights.get(tag, 0.0) for tag in available
        }
        total_weight = sum(active_weights.values())
        if total_weight <= 0:
            active_weights = {tag: 1.0 for tag in available}
            total_weight = float(len(active_weights))

        counts = {
            tag: min(len(indices), int(batch_size * weight / total_weight))
            for tag, (indices, weight) in
            ((tag, (available[tag], active_weights[tag])) for tag in available)
        }
        remaining = batch_size - sum(counts.values())
        order = sorted(available, key=lambda tag: (-active_weights[tag], tag))
        while remaining:
            eligible = [tag for tag in order if counts[tag] < len(available[tag])]
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
                chosen.extend(self.rng.choice(available[tag], count, replace=False))
        if len(chosen) < batch_size:
            used = {int(index) for index in chosen}
            leftovers = [index for index in range(len(self.storage)) if index not in used]
            chosen.extend(self.rng.choice(leftovers, batch_size - len(chosen), replace=False))
        self.rng.shuffle(chosen)
        rows = [self.storage[int(index)] for index in chosen]
        states, actions, rewards, next_states, dones, next_action_masks = zip(*rows)
        reference = states[0]
        next_states = [reference * 0 if state is None else state
                       for state in next_states]
        from agent_code.combat_dqn_agent.replay import TransitionBatch
        return TransitionBatch(
            np.stack(states).astype(np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.asarray(dones, dtype=np.float32),
            np.stack(next_action_masks).astype(bool),
        )
