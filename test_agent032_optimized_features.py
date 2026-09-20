"""Agent 032 feature-contract and equivalence checks."""

import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import optim

from test_combat_adversarial_window import open_state
from agent_code.Agent_031_combat_ddqn_offensive_escape_agent import features as old
from agent_code.Agent_032_combat_ddqn_optimized_features_agent import features as optimized
from agent_code.Agent_032_combat_ddqn_optimized_features_agent.replay import (
    CombatEscapeReplayBuffer,
)
from agent_code.Agent_032_combat_ddqn_optimized_features_agent.train import (
    optimize_model,
)
from agent_code.combat_dqn_agent.model import DEVICE, QNetwork


class OptimizedFeatureTests(unittest.TestCase):
    def _states(self):
        states = []
        opponents = [
            (),
            (("enemy-a", 0, True, (9, 7)),),
            (("enemy-a", 0, True, (9, 7)),
             ("enemy-b", 0, False, (7, 11)),
             ("enemy-c", 0, True, (4, 7))),
        ]
        for index, others in enumerate(opponents):
            state = open_state(others)
            state["field"][5, 5] = 1
            state["field"][6, 5] = 1
            state["field"][8, 8] = 1
            state["bombs"] = [((7, 9), 2)] if index else []
            state["explosion_map"][7, 8] = 2 if index == 2 else 0
            state["self"] = ("learner", 0, index != 1, (7, 7))
            states.append(state)
        return states

    def test_optimized_features_preserve_agent031_values(self):
        for state in self._states():
            old_features = old.state_to_features(
                state, 4, 2, 3, (0, 1, 4), (1, 1, 0),
                ((7, 7), (7, 6), (7, 7)),
            )
            new_features = optimized.state_to_features(
                copy.deepcopy(state), 4, 2, 3, (0, 1, 4), (1, 1, 0),
                ((7, 7), (7, 6), (7, 7)),
            )
            self.assertEqual(new_features.shape, (113,))
            self.assertEqual(new_features.dtype, np.float32)
            np.testing.assert_array_equal(new_features, old_features)

    def test_preallocated_replay_and_torch_batch_contract(self):
        buffer = CombatEscapeReplayBuffer(4, seed=7, state_dim=113)
        buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
        for index in range(6):
            state = np.full(113, index, dtype=np.float32)
            buffer.transition_tag = "combat_escape" if index % 2 else None
            buffer.add(
                state, index % 6, float(index), state + 1.0, False,
                np.ones(6, dtype=bool),
            )

        self.assertEqual(len(buffer), 4)
        self.assertEqual(buffer.states.shape, (4, 113))
        self.assertEqual(buffer.next_action_masks.dtype, np.bool_)
        self.assertEqual(len(buffer.indices_by_tag["combat_escape"]), 2)

        numpy_batch = buffer.sample(3)
        self.assertEqual(numpy_batch.states.shape, (3, 113))
        self.assertEqual(numpy_batch.actions.dtype, np.int64)

        torch_batch = buffer.sample_torch(3, torch.device("cpu"))
        self.assertTrue(all(isinstance(item, torch.Tensor)
                            for item in torch_batch))
        self.assertEqual(tuple(torch_batch.states.shape), (3, 113))
        self.assertEqual(torch_batch.actions.dtype, torch.int64)

    def test_agent032_optimizer_consumes_reusable_batch(self):
        buffer = CombatEscapeReplayBuffer(5_100, seed=3, state_dim=113)
        buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
        state = np.zeros(113, dtype=np.float32)
        next_state = np.ones(113, dtype=np.float32)
        mask = np.ones(6, dtype=bool)
        for index in range(5_000):
            buffer.add(state, index % 6, 0.1, next_state, False, mask)

        policy = QNetwork(113, 6).to(DEVICE)
        target = QNetwork(113, 6).to(DEVICE)
        target.load_state_dict(policy.state_dict())
        learner = SimpleNamespace(
            replay_buffer=buffer,
            policy_net=policy,
            target_net=target,
            optimizer=optim.Adam(policy.parameters(), lr=1e-3),
            optimizer_steps=0,
            dqn_algorithm="ddqn",
            round_loss_total=0.0,
            round_loss_count=0,
        )
        loss = optimize_model(learner)
        self.assertIsInstance(loss, float)
        self.assertEqual(learner.optimizer_steps, 1)
        self.assertEqual(learner.round_loss_count, 1)


if __name__ == "__main__":
    unittest.main()
