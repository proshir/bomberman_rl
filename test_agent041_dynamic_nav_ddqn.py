"""Contract tests for Agent041's DDQN-controlled navigation inputs."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import optim

from test_combat_adversarial_window import open_state
from agent_code.Agent_041_dynamic_nav_ddqn_agent import features, symmetry
from agent_code.Agent_041_dynamic_nav_ddqn_agent import callbacks
from agent_code.Agent_041_dynamic_nav_ddqn_agent.checkpoint import (
    expand_agent040_checkpoint,
)
from agent_code.combat_dqn_agent.model import QNetwork
from agent_code.Agent_041_dynamic_nav_ddqn_agent.replay import (
    CombatEscapeReplayBuffer,
)


class Agent041Tests(unittest.TestCase):
    def test_navigation_features_are_action_aligned(self):
        state = open_state()
        state["coins"] = [(2, 2)]
        state["self"] = ("learner", 0, True, (7, 7))
        value = features.state_to_features(
            state, features.ACTIONS.index("WAIT"), 0, 0,
            position_history=((7, 7), (7, 7)),
        )
        self.assertEqual(value.shape, (122,))
        self.assertEqual(value.dtype, np.float32)
        np.testing.assert_array_equal(
            value[features.NAV_PROGRESS_START:features.NAV_PROGRESS_START + 6],
            np.asarray([1, -1, -1, 1, 0, 0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            value[features.NAV_REACHABLE_START:features.NAV_REACHABLE_START + 6],
            np.asarray([1, 1, 1, 1, 1, 0], dtype=np.float32),
        )
        self.assertAlmostEqual(
            value[features.NAV_VISITS_START + features.ACTIONS.index("WAIT")],
            2.0 / 3.0,
        )

    def test_all_action_blocks_transform_with_action_labels(self):
        vector = np.zeros(features.FEATURE_SIZE, dtype=np.float32)
        for offset, start in enumerate((
                features.NAV_PROGRESS_START,
                features.NAV_REACHABLE_START,
                features.NAV_VISITS_START)):
            vector[start:start + 6] = np.arange(6, dtype=np.float32) + offset
        transform = (1, False)
        permutation = symmetry.direction_permutation(transform)
        result = symmetry.transform_features(vector, transform)
        for start in (
                features.NAV_PROGRESS_START,
                features.NAV_REACHABLE_START,
                features.NAV_VISITS_START):
            for old_action, new_action in enumerate(permutation):
                self.assertEqual(result[start + new_action],
                                 vector[start + old_action])

    def test_symmetry_is_invertible_for_122_inputs(self):
        vector = np.arange(features.FEATURE_SIZE, dtype=np.float32)
        for transform in symmetry.TRANSFORMS:
            transformed = symmetry.transform_features(vector, transform)
            inverse = next(candidate for candidate in symmetry.TRANSFORMS
                           if all(
                               symmetry.transform_action(
                                   symmetry.transform_action(action, transform),
                                   candidate) == action
                               for action in range(6)))
            np.testing.assert_array_equal(
                symmetry.transform_features(transformed, inverse), vector
            )

    def test_replay_defaults_to_122_inputs(self):
        buffer = CombatEscapeReplayBuffer(4, seed=7)
        buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
        state = np.arange(features.FEATURE_SIZE, dtype=np.float32)
        buffer.add(state, 1, 0.5, state + 1, False,
                   np.ones(6, dtype=bool))
        self.assertEqual(buffer.states.shape, (4, features.FEATURE_SIZE))
        batch = buffer.sample_torch(1, torch.device("cpu"))
        self.assertEqual(tuple(batch.states.shape), (1, features.FEATURE_SIZE))

    def test_setup_builds_a_122_input_ddqn(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = SimpleNamespace(
                train=True, seed=0,
                model_path=Path(directory) / "training.pkl",
            )
            callbacks.setup(learner)
        self.assertEqual(learner.policy_net.input_dim, features.FEATURE_SIZE)
        self.assertEqual(learner.dqn_algorithm, "ddqn")

    def test_agent040_checkpoint_can_be_warm_started(self):
        torch.manual_seed(4)
        old = QNetwork(features.AGENT_040_FEATURE_SIZE, 6)
        target = QNetwork(features.AGENT_040_FEATURE_SIZE, 6)
        target.load_state_dict(old.state_dict())
        optimizer = optim.Adam(old.parameters(), lr=1e-4)
        checkpoint = {
            "input_dim": features.AGENT_040_FEATURE_SIZE,
            "n_actions": 6,
            "policy_state_dict": old.state_dict(),
            "target_state_dict": target.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        migrated = expand_agent040_checkpoint(checkpoint)
        wide = QNetwork(features.FEATURE_SIZE, 6)
        wide.load_state_dict(migrated["policy_state_dict"])
        sample = torch.randn(3, features.AGENT_040_FEATURE_SIZE)
        suffix = torch.zeros(3, features.FEATURE_SIZE - features.AGENT_040_FEATURE_SIZE)
        with torch.no_grad():
            torch.testing.assert_close(
                wide(torch.cat((sample, suffix), dim=1)), old(sample),
                rtol=1e-6, atol=1e-7,
            )


if __name__ == "__main__":
    unittest.main()
