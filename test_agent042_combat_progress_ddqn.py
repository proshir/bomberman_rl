"""Contract tests for Agent 042's combat-progress DDQN representation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import optim

from test_combat_adversarial_window import open_state
from agent_code.Agent_042_combat_progress_ddqn_agent import callbacks, features, symmetry
from agent_code.Agent_042_combat_progress_ddqn_agent.checkpoint import expand_checkpoint
from agent_code.Agent_042_combat_progress_ddqn_agent.replay import CombatEscapeReplayBuffer
from agent_code.combat_dqn_agent.model import QNetwork


class Agent042Tests(unittest.TestCase):
    def test_feature_contract_and_solo_neutrality(self):
        state = open_state()
        value = features.state_to_features(state)
        self.assertEqual(value.shape, (140,))
        self.assertEqual(value.dtype, np.float32)
        self.assertTrue(np.all(value[features.COMBAT_PROGRESS_START:] == 0.0))

    def test_combat_progress_is_action_aligned(self):
        state = open_state((("enemy", 0, True, (7, 5)),))
        value = features.state_to_features(state)
        up = features.ACTIONS.index("UP")
        down = features.ACTIONS.index("DOWN")
        self.assertGreater(value[features.COMBAT_PROGRESS_START + up], 0.0)
        self.assertLess(value[features.COMBAT_PROGRESS_START + down], 0.0)
        self.assertEqual(value[features.COMBAT_REACHABLE_START + up], 1.0)
        self.assertGreater(value[features.COMBAT_PRESSURE_START + up], 0.0)

    def test_action_blocks_transform_with_labels(self):
        vector = np.zeros(features.FEATURE_SIZE, dtype=np.float32)
        starts = (features.COMBAT_PROGRESS_START,
                  features.COMBAT_REACHABLE_START,
                  features.COMBAT_PRESSURE_START)
        for offset, start in enumerate(starts):
            vector[start:start + 6] = np.arange(6, dtype=np.float32) + offset
        transform = (1, False)
        permutation = symmetry.direction_permutation(transform)
        result = symmetry.transform_features(vector, transform)
        for start in starts:
            for old_action, new_action in enumerate(permutation):
                self.assertEqual(result[start + new_action],
                                 vector[start + old_action])

    def test_replay_defaults_to_140_inputs(self):
        buffer = CombatEscapeReplayBuffer(4, seed=7)
        buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
        state = np.arange(features.FEATURE_SIZE, dtype=np.float32)
        buffer.add(state, 1, 0.5, state + 1, False,
                   np.ones(6, dtype=bool))
        self.assertEqual(buffer.states.shape, (4, features.FEATURE_SIZE))
        batch = buffer.sample_torch(1, torch.device("cpu"))
        self.assertEqual(tuple(batch.states.shape), (1, features.FEATURE_SIZE))

    def test_setup_builds_140_input_ddqn(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = SimpleNamespace(
                train=True, seed=0,
                model_path=Path(directory) / "training.pkl",
            )
            callbacks.setup(learner)
        self.assertEqual(learner.policy_net.input_dim, features.FEATURE_SIZE)
        self.assertEqual(learner.dqn_algorithm, "ddqn")

    def test_agent040_and_agent041_checkpoints_can_be_warm_started(self):
        torch.manual_seed(4)
        for old_dim in (features.AGENT_040_FEATURE_SIZE,
                        features.AGENT_041_FEATURE_SIZE):
            old = QNetwork(old_dim, 6)
            target = QNetwork(old_dim, 6)
            target.load_state_dict(old.state_dict())
            optimizer = optim.Adam(old.parameters(), lr=1e-4)
            checkpoint = {
                "input_dim": old_dim, "n_actions": 6,
                "policy_state_dict": old.state_dict(),
                "target_state_dict": target.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            migrated = expand_checkpoint(checkpoint)
            wide = QNetwork(features.FEATURE_SIZE, 6)
            wide.load_state_dict(migrated["policy_state_dict"])
            sample = torch.randn(3, old_dim)
            suffix = torch.zeros(3, features.FEATURE_SIZE - old_dim)
            with torch.no_grad():
                torch.testing.assert_close(
                    wide(torch.cat((sample, suffix), dim=1)), old(sample),
                    rtol=1e-6, atol=1e-7,
                )

    def test_progress_signature_ignores_global_field_and_opponent_changes(self):
        state = open_state()
        changed = open_state((("enemy", 0, True, (10, 7)),))
        changed["field"][5, 5] = 1
        self.assertEqual(
            callbacks.progress_signature(state),
            callbacks.progress_signature(changed),
        )


if __name__ == "__main__":
    unittest.main()
