"""Basic tests for the minimal vanilla DQN."""

# Sahand was here.

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import optim

from agent_code.combat_dqn_agent import train as dqn_train
from agent_code.combat_dqn_agent.config import N_ACTIONS
from agent_code.combat_dqn_agent.model import QNetwork, save_checkpoint, vanilla_targets
from agent_code.combat_dqn_agent.replay import ReplayBuffer


class VanillaDQNTest(unittest.TestCase):
    def test_replay_is_bounded(self):
        replay = ReplayBuffer(2, seed=0)
        for index in range(3):
            replay.add(np.ones(4) * index, 0, 1.0, np.ones(4), False)
        self.assertEqual(len(replay), 2)
        self.assertEqual(replay.sample(2).states.shape, (2, 4))

    def test_vanilla_target_bootstraps_max_target_value_and_not_terminal(self):
        target = QNetwork(2, 2)
        with torch.no_grad():
            for parameter in target.parameters():
                parameter.zero_()
            target.network[-1].bias[:] = torch.tensor([2.0, 5.0])
        states = torch.zeros((2, 2))
        rewards = torch.tensor([1.0, 3.0])
        dones = torch.tensor([0.0, 1.0])
        values = vanilla_targets(target, states, rewards, dones, 0.99)
        self.assertAlmostEqual(values[0].item(), 5.95, places=5)
        self.assertAlmostEqual(values[1].item(), 3.0, places=5)
        self.assertFalse(values.requires_grad)

    def test_network_shape(self):
        self.assertEqual(tuple(QNetwork(32, N_ACTIONS)(torch.zeros(3, 32)).shape), (3, 6))

    def test_checkpoint_round_trip_contains_optimizer_and_counters(self):
        policy = QNetwork(4, N_ACTIONS)
        learner = SimpleNamespace(
            policy_net=policy, target_net=QNetwork(4, N_ACTIONS),
            optimizer=optim.Adam(policy.parameters()), epsilon=0.5,
            env_steps=4, optimizer_steps=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(learner, path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertIn("optimizer_state_dict", payload)
        self.assertEqual(payload["env_steps"], 4)

    def test_optimizer_update(self):
        learner = SimpleNamespace(
            policy_net=QNetwork(4, N_ACTIONS), target_net=QNetwork(4, N_ACTIONS),
            optimizer=None, optimizer_steps=0, round_loss_total=0.0,
            round_loss_count=0,
        )
        learner.optimizer = optim.Adam(learner.policy_net.parameters(), lr=1e-3)
        learner.replay_buffer = ReplayBuffer(8, seed=0)
        for index in range(4):
            learner.replay_buffer.add(np.ones(4) * index, 0, 1.0, np.ones(4), False)
        before = [p.detach().clone() for p in learner.policy_net.parameters()]
        with mock.patch.object(dqn_train, "WARMUP_TRANSITIONS", 4), \
             mock.patch.object(dqn_train, "BATCH_SIZE", 4):
            dqn_train.optimize_model(learner)
        self.assertEqual(learner.optimizer_steps, 1)
        self.assertTrue(any(not torch.equal(old, new)
                            for old, new in zip(before, learner.policy_net.parameters())))
