"""Correctness checks for the combat Double-DQN implementation."""

# Sahand was here.

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn, optim

from agent_code.combat_dqn_agent import train as dqn_train
from agent_code.combat_dqn_agent.config import N_ACTIONS
from agent_code.combat_dqn_agent.model import (
    QNetwork,
    checkpoint_payload,
    double_dqn_targets,
    masked_argmax,
    save_checkpoint,
)
from agent_code.combat_dqn_agent.replay import ReplayBuffer


class FixedNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.as_tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values[:len(states)]


class CombatDQNTest(unittest.TestCase):
    def test_replay_buffer_is_bounded_and_samples_shapes(self):
        buffer = ReplayBuffer(2, seed=0)
        for index in range(3):
            buffer.add(np.full(4, index), index % N_ACTIONS, index,
                       np.full(4, index + 1), False,
                       np.ones(N_ACTIONS, dtype=bool))
        self.assertEqual(len(buffer), 2)
        batch = buffer.sample(2)
        self.assertEqual(batch.states.shape, (2, 4))
        self.assertEqual(batch.next_states.shape, (2, 4))
        self.assertEqual(batch.actions.shape, (2,))
        self.assertEqual(batch.next_action_masks.shape, (2, N_ACTIONS))

    def test_masked_argmax_never_selects_illegal_action(self):
        values = torch.tensor([[10.0, 100.0, 20.0]])
        action = masked_argmax(values, [[True, False, True]])
        self.assertEqual(action.item(), 2)

    def test_terminal_double_dqn_target_does_not_bootstrap(self):
        policy = FixedNetwork([[1.0, 5.0, 3.0], [99.0, 98.0, 97.0]])
        target = FixedNetwork([[10.0, 20.0, 30.0], [90.0, 80.0, 70.0]])
        next_states = torch.zeros((2, 4))
        rewards = torch.tensor([1.0, 2.0])
        dones = torch.tensor([False, True])
        masks = torch.tensor([[True, False, True], [False, False, False]])
        result = double_dqn_targets(
            policy, target, next_states, rewards, dones, masks, gamma=0.99
        )
        self.assertAlmostEqual(result[0].item(), 30.7, places=5)
        self.assertAlmostEqual(result[1].item(), 2.0, places=5)
        self.assertFalse(result.requires_grad)

    def test_network_shapes(self):
        network = QNetwork(32, N_ACTIONS)
        output = network(torch.zeros((7, 32)))
        self.assertEqual(tuple(output.shape), (7, N_ACTIONS))

    def test_checkpoint_round_trip(self):
        policy = QNetwork(4, N_ACTIONS)
        target = QNetwork(4, N_ACTIONS)
        optimizer = optim.Adam(policy.parameters(), lr=1e-4)
        learner = SimpleNamespace(
            policy_net=policy,
            target_net=target,
            optimizer=optimizer,
            epsilon=0.42,
            env_steps=17,
            optimizer_steps=3,
            last_loss=0.125,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(learner, path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["input_dim"], 4)
        self.assertEqual(payload["n_actions"], N_ACTIONS)
        self.assertEqual(payload["env_steps"], 17)
        self.assertAlmostEqual(payload["epsilon"], 0.42)
        self.assertEqual(set(payload["policy_state_dict"]),
                         set(policy.state_dict()))

    def test_one_optimizer_update_changes_parameters(self):
        learner = SimpleNamespace(
            policy_net=QNetwork(4, N_ACTIONS),
            target_net=QNetwork(4, N_ACTIONS),
            optimizer=None,
            optimizer_steps=0,
            last_loss=None,
        )
        learner.optimizer = optim.Adam(learner.policy_net.parameters(), lr=1e-3)
        learner.replay_buffer = ReplayBuffer(16, seed=0)
        for index in range(4):
            learner.replay_buffer.add(
                np.full(4, index, dtype=np.float32), index % N_ACTIONS, 1.0,
                np.full(4, index + 1, dtype=np.float32), False,
                np.ones(N_ACTIONS, dtype=bool),
            )
        before = [parameter.detach().clone()
                  for parameter in learner.policy_net.parameters()]
        with mock.patch.object(dqn_train, "WARMUP_TRANSITIONS", 4), \
             mock.patch.object(dqn_train, "BATCH_SIZE", 4):
            loss = dqn_train.optimize_model(learner)
        after = list(learner.policy_net.parameters())
        self.assertIsInstance(loss, float)
        self.assertEqual(learner.optimizer_steps, 1)
        self.assertTrue(any(not torch.equal(old, new)
                            for old, new in zip(before, after)))


if __name__ == "__main__":
    unittest.main()
