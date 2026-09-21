"""Contract checks for the Agent 037 tournament continuation implementation."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import optim

from test_agent032_optimized_features import OptimizedFeatureTests
from agent_code.Agent_029_combat_ddqn_adversarial_window_agent import (
    features as agent029_features,
)
from agent_code.Agent_037_tournament_fast_ddqn_agent import features
from agent_code.Agent_037_tournament_fast_ddqn_agent.replay import (
    CombatEscapeReplayBuffer,
)
from agent_code.Agent_037_tournament_fast_ddqn_agent.train import optimize_model
from agent_code.combat_dqn_agent.model import DEVICE, QNetwork


class Agent037Tests(unittest.TestCase):
    def test_exact_agent029_feature_contract(self):
        for state in OptimizedFeatureTests()._states():
            arguments = (4, 2, 3, (0, 1, 4), (1, 1, 0),
                         ((7, 7), (7, 6), (7, 7)))
            expected = agent029_features.state_to_features(state, *arguments)
            actual = features.state_to_features(copy.deepcopy(state), *arguments)
            self.assertEqual(actual.shape, (101,))
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, expected)

    def test_preallocated_101_replay_contract(self):
        buffer = CombatEscapeReplayBuffer(4, seed=7)
        buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
        for index in range(6):
            state = np.full(101, index, dtype=np.float32)
            buffer.transition_tag = "combat_escape" if index % 2 else None
            buffer.add(
                state, index % 6, float(index), state + 1.0, False,
                np.ones(6, dtype=bool),
            )
        self.assertEqual(len(buffer), 4)
        self.assertEqual(buffer.states.shape, (4, 101))
        self.assertEqual(len(buffer.indices_by_tag["combat_escape"]), 2)
        batch = buffer.sample_torch(3, torch.device("cpu"))
        self.assertEqual(tuple(batch.states.shape), (3, 101))
        self.assertEqual(batch.actions.dtype, torch.int64)

    def test_optimizer_consumes_101_input_replay(self):
        buffer = CombatEscapeReplayBuffer(5_100, seed=3)
        buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
        state = np.zeros(101, dtype=np.float32)
        next_state = np.ones(101, dtype=np.float32)
        mask = np.ones(6, dtype=bool)
        for index in range(5_000):
            buffer.add(state, index % 6, 0.1, next_state, False, mask)
        policy = QNetwork(101, 6).to(DEVICE)
        target = QNetwork(101, 6).to(DEVICE)
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
