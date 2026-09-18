"""Basic tests for the minimal DQN implementations."""

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
from agent_code.combat_dqn_agent import callbacks as dqn_callbacks
from agent_code.combat_dqn_agent.config import N_ACTIONS
from agent_code.combat_dqn_agent.features import ACTIONS
from agent_code.combat_dqn_agent.model import (
    QNetwork, dqn_targets, save_checkpoint, vanilla_targets,
)
from agent_code.combat_dqn_agent.replay import ReplayBuffer


class VanillaDQNTest(unittest.TestCase):
    @staticmethod
    def _constant_network(values):
        network = QNetwork(2, len(values))
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.network[-1].bias[:] = torch.tensor(values)
        return network

    def test_replay_is_bounded(self):
        replay = ReplayBuffer(2, seed=0)
        for index in range(3):
            replay.add(np.ones(4) * index, 0, 1.0, np.ones(4), False,
                       [True, False, True, False, False, False])
        self.assertEqual(len(replay), 2)
        self.assertEqual(replay.sample(2).states.shape, (2, 4))
        np.testing.assert_array_equal(
            replay.sample(2).next_action_masks.shape, (2, N_ACTIONS)
        )

    def test_vanilla_target_masks_excluded_max_and_not_terminal(self):
        target = QNetwork(2, 2)
        with torch.no_grad():
            for parameter in target.parameters():
                parameter.zero_()
            target.network[-1].bias[:] = torch.tensor([2.0, 5.0])
        states = torch.zeros((2, 2))
        rewards = torch.tensor([1.0, 3.0])
        dones = torch.tensor([0.0, 1.0])
        masks = torch.tensor([[True, False], [False, False]])
        values = vanilla_targets(target, states, rewards, dones, 0.99, masks)
        self.assertAlmostEqual(values[0].item(), 2.98, places=5)
        self.assertAlmostEqual(values[1].item(), 3.0, places=5)
        self.assertFalse(values.requires_grad)

    def test_vanilla_and_double_dqn_use_their_respective_selectors(self):
        policy = self._constant_network([9.0, 1.0, 8.0])
        target = self._constant_network([2.0, 5.0, 4.0])
        states = torch.zeros((2, 2))
        rewards = torch.zeros(2)
        dones = torch.zeros(2)
        masks = torch.tensor([
            [True, True, True],
            [False, True, True],
        ])

        vanilla = dqn_targets(
            policy, target, states, rewards, dones, 1.0, masks,
            algorithm="dqn",
        )
        double = dqn_targets(
            policy, target, states, rewards, dones, 1.0, masks,
            algorithm="double_dqn",
        )

        # Vanilla selects target-network maxima: action 1 in both rows.
        torch.testing.assert_close(vanilla, torch.tensor([5.0, 5.0]))
        # DDQN selects with policy (actions 0 and 2), then evaluates via target.
        torch.testing.assert_close(double, torch.tensor([2.0, 4.0]))
        self.assertEqual(vanilla.shape, rewards.shape)
        self.assertEqual(double.shape, rewards.shape)
        self.assertFalse(vanilla.requires_grad)
        self.assertFalse(double.requires_grad)

    def test_terminal_transitions_never_bootstrap_in_either_mode(self):
        policy = self._constant_network([9.0, 1.0])
        target = self._constant_network([2.0, 5.0])
        states = torch.zeros((2, 2))
        rewards = torch.tensor([3.0, -2.0])
        dones = torch.ones(2)
        masks = torch.tensor([[True, True], [False, False]])

        for algorithm in ("dqn", "ddqn"):
            values = dqn_targets(
                policy, target, states, rewards, dones, 0.99, masks,
                algorithm=algorithm,
            )
            torch.testing.assert_close(values, rewards)
            self.assertEqual(values.shape, (2,))

    def test_network_shape(self):
        self.assertEqual(tuple(QNetwork(32, N_ACTIONS)(torch.zeros(3, 32)).shape), (3, 6))

    def test_action_selection_stays_within_legal_candidates(self):
        learner = SimpleNamespace(
            train=True, seed=0, model_path=Path(tempfile.mkdtemp()) / "unused.pt",
        )
        dqn_callbacks.setup(learner)
        state = {
            "round": 1, "step": 1,
            "field": np.zeros((17, 17), dtype=int),
            "bombs": [], "explosion_map": np.zeros((17, 17), dtype=int),
            "coins": [], "self": ("dqn", 0, True, (8, 8)), "others": [],
        }
        with mock.patch.object(dqn_callbacks, "safe_action_indices", return_value=[2]), \
             mock.patch.object(dqn_callbacks, "best_survival_action_indices", return_value=[2]):
            self.assertEqual(dqn_callbacks.act(learner, state), ACTIONS[2])

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

    def test_training_setup_resumes_checkpoint_state(self):
        policy = QNetwork(32, N_ACTIONS)
        source = SimpleNamespace(
            policy_net=policy, target_net=QNetwork(32, N_ACTIONS),
            optimizer=optim.Adam(policy.parameters()), epsilon=0.25,
            env_steps=123, optimizer_steps=17,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "episode_0007.pkl"
            save_checkpoint(source, checkpoint)
            learner = SimpleNamespace(
                train=True, seed=0,
                model_path=Path(directory) / "training.pkl",
            )
            previous_resume_path = dqn_callbacks.RESUME_PATH
            dqn_callbacks.RESUME_PATH = checkpoint
            try:
                dqn_callbacks.setup(learner)
            finally:
                dqn_callbacks.RESUME_PATH = previous_resume_path
        self.assertEqual(learner.env_steps, 123)
        self.assertEqual(learner.optimizer_steps, 17)
        self.assertAlmostEqual(learner.epsilon, 0.25)

    def test_optimizer_update(self):
        learner = SimpleNamespace(
            policy_net=QNetwork(4, N_ACTIONS), target_net=QNetwork(4, N_ACTIONS),
            optimizer=None, optimizer_steps=0, round_loss_total=0.0,
            round_loss_count=0,
        )
        learner.optimizer = optim.Adam(learner.policy_net.parameters(), lr=1e-3)
        learner.replay_buffer = ReplayBuffer(8, seed=0)
        for index in range(4):
            learner.replay_buffer.add(
                np.ones(4) * index, 0, 1.0, np.ones(4), False,
                [True, False, True, False, False, False]
            )
        before = [p.detach().clone() for p in learner.policy_net.parameters()]
        with mock.patch.object(dqn_train, "WARMUP_TRANSITIONS", 4), \
             mock.patch.object(dqn_train, "BATCH_SIZE", 4):
            dqn_train.optimize_model(learner)
        self.assertEqual(learner.optimizer_steps, 1)
        self.assertTrue(any(not torch.equal(old, new)
                            for old, new in zip(before, learner.policy_net.parameters())))

    def test_replay_preserves_next_action_mask_for_terminal_rows(self):
        replay = ReplayBuffer(4, seed=0)
        replay.add(np.zeros(3), 0, 1.0, None, True)
        batch = replay.sample(1)
        self.assertFalse(batch.next_action_masks.any())

    def test_remember_uses_next_action_time_features_and_mask(self):
        learner = SimpleNamespace(
            replay_buffer=ReplayBuffer(4, seed=0),
            feature_cache={}, env_steps=0, epsilon=1.0,
            round_reward=0.0, round_steps=0, round_events={},
        )
        old = {
            "round": 1, "step": 1, "field": np.zeros((17, 17), dtype=int),
            "bombs": [], "explosion_map": np.zeros((17, 17), dtype=int),
            "coins": [], "self": ("dqn", 0, True, (8, 8)), "others": [],
        }
        new = dict(old, step=1, **{"self": ("dqn", 0, True, (9, 8))})
        next_decision = dict(new, step=2)
        learner.feature_cache[(1, 1)] = (np.ones(32), (), (), 0)
        expected = np.full(32, 7.0)
        learner.feature_cache[(1, 2)] = (expected, (), (), 0)
        with mock.patch.object(dqn_train, "_candidate_mask", return_value=np.array(
                [True, False, True, False, False, False])):
            dqn_train.remember(
                learner, old, "WAIT", new, [],
                next_decision_state=next_decision,
            )
        stored = learner.replay_buffer.storage[0]
        np.testing.assert_array_equal(stored[3], expected)
        np.testing.assert_array_equal(
            stored[5], [True, False, True, False, False, False]
        )
