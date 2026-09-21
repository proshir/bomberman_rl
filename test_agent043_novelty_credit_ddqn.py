"""Contract tests for Agent 043 novelty and credit-assignment changes."""

import tempfile
import unittest
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import events as e
import numpy as np
import torch
from torch import optim

from test_combat_adversarial_window import open_state
from agent_code.Agent_043_novelty_credit_ddqn_agent import callbacks, features, symmetry, train
from agent_code.Agent_043_novelty_credit_ddqn_agent.checkpoint import expand_checkpoint
from agent_code.Agent_043_novelty_credit_ddqn_agent.replay import CombatEscapeReplayBuffer
from agent_code.combat_dqn_agent.model import QNetwork


class _CaptureReplay:
    def __init__(self):
        self.rows = []
        self.transition_tag = None

    def add(self, *row):
        self.rows.append(row)


class Agent043Tests(unittest.TestCase):
    def test_feature_contract_and_action_aligned_novelty(self):
        state = open_state()
        position = tuple(state["self"][3])
        value = features.state_to_features(
            state, position_history=(position, position),
        )
        self.assertEqual(value.shape, (146,))
        self.assertEqual(value.dtype, np.float32)
        wait = features.ACTIONS.index("WAIT")
        self.assertAlmostEqual(value[features.NOVELTY_START + wait], 1.0 / 3.0)
        self.assertTrue(np.all(value[features.NOVELTY_START:] >= 0.0))

    def test_novelty_block_transforms_with_action_labels(self):
        vector = np.zeros(features.FEATURE_SIZE, dtype=np.float32)
        vector[features.NOVELTY_START:] = np.arange(6, dtype=np.float32)
        transform = (1, False)
        permutation = symmetry.direction_permutation(transform)
        result = symmetry.transform_features(vector, transform)
        for old_action, new_action in enumerate(permutation):
            self.assertEqual(
                result[features.NOVELTY_START + new_action],
                vector[features.NOVELTY_START + old_action],
            )

    def test_replay_and_setup_use_146_inputs(self):
        buffer = CombatEscapeReplayBuffer(4, seed=7)
        buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
        state = np.arange(features.FEATURE_SIZE, dtype=np.float32)
        buffer.add(state, 1, 0.5, state + 1, False,
                   np.ones(6, dtype=bool))
        self.assertEqual(buffer.states.shape, (4, features.FEATURE_SIZE))
        with tempfile.TemporaryDirectory() as directory:
            learner = SimpleNamespace(
                train=True, seed=0,
                model_path=Path(directory) / "training.pkl",
            )
            callbacks.setup(learner)
        self.assertEqual(learner.policy_net.input_dim, features.FEATURE_SIZE)

    def test_checkpoint_migration_preserves_agent042_q_values(self):
        torch.manual_seed(5)
        old = QNetwork(features.AGENT_042_FEATURE_SIZE, 6)
        target = QNetwork(features.AGENT_042_FEATURE_SIZE, 6)
        target.load_state_dict(old.state_dict())
        optimizer = optim.Adam(old.parameters(), lr=1e-4)
        checkpoint = {
            "input_dim": features.AGENT_042_FEATURE_SIZE, "n_actions": 6,
            "policy_state_dict": old.state_dict(),
            "target_state_dict": target.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        migrated = expand_checkpoint(checkpoint)
        wide = QNetwork(features.FEATURE_SIZE, 6)
        wide.load_state_dict(migrated["policy_state_dict"])
        sample = torch.randn(3, features.AGENT_042_FEATURE_SIZE)
        suffix = torch.zeros(3, features.FEATURE_SIZE - features.AGENT_042_FEATURE_SIZE)
        with torch.no_grad():
            torch.testing.assert_close(
                wide(torch.cat((sample, suffix), dim=1)), old(sample),
                rtol=1e-6, atol=1e-7,
            )

    def test_progress_events_advance_only_attributed_clock(self):
        learner = SimpleNamespace(
            last_progress_step=3,
            attributed_progress_events=defaultdict(int),
        )
        state = open_state()
        state["step"] = 11
        callbacks.record_progress_events(learner, state, [e.COIN_COLLECTED])
        self.assertEqual(learner.last_progress_step, 11)
        self.assertEqual(learner.attributed_progress_events[e.COIN_COLLECTED], 1)
        callbacks.record_progress_events(learner, state, ["OPPONENT_ELIMINATED"])
        self.assertEqual(learner.last_progress_step, 11)

    def test_three_step_return_uses_gamma_powers(self):
        state = open_state()
        zero = np.zeros(features.FEATURE_SIZE, dtype=np.float32)
        mask = np.ones(6, dtype=bool)
        learner = SimpleNamespace(
            n_step_queue=deque(), rng=np.random.default_rng(2),
            replay_buffer=_CaptureReplay(),
        )
        for reward in (1.0, 2.0, 3.0):
            learner.n_step_queue.append((
                zero, 0, reward, zero, False, mask, state, state, False,
            ))
        train._flush_n_step(learner)
        self.assertEqual(len(learner.replay_buffer.rows), 1)
        expected = 1.0 + train.GAMMA * 2.0 + train.GAMMA ** 2 * 3.0
        self.assertAlmostEqual(learner.replay_buffer.rows[0][2], expected)


if __name__ == "__main__":
    unittest.main()
