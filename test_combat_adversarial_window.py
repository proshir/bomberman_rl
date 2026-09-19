"""Focused regression checks for Agent 029's opponent-response inputs."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_code.Agent_029_combat_ddqn_adversarial_window_agent import callbacks
from agent_code.Agent_029_combat_ddqn_adversarial_window_agent.features import (
    ACTIONS,
    FEATURE_SIZE,
    OPPONENT_FEATURES_PER_ACTION,
    opponent_window_features,
    state_to_features,
)
from agent_code.Agent_027_combat_ddqn_short_cycle_staged_replay_agent.features import (
    state_to_features as agent027_state_to_features,
)


def open_state(others=()):
    field = np.zeros((17, 17), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "bombs": [],
        "explosion_map": np.zeros_like(field),
        "coins": [(2, 2)],
        "self": ("learner", 0, True, (7, 7)),
        "others": list(others),
        "user_input": None,
    }


class AdversarialWindowFeaturesTest(unittest.TestCase):
    def test_feature_contract_preserves_agent027_prefix(self):
        state = open_state()
        feature = state_to_features(
            state, ACTIONS.index("WAIT"), 0, 0, (0, 1), (1, 1),
            ((7, 7), (7, 6), (7, 7)),
        )
        self.assertEqual(feature.shape, (FEATURE_SIZE,))
        self.assertEqual(feature.dtype, np.float32)
        agent027_feature = agent027_state_to_features(
            state, ACTIONS.index("WAIT"), 0, 0, (0, 1), (1, 1),
            ((7, 7), (7, 6), (7, 7)),
        )
        np.testing.assert_array_equal(feature[:len(agent027_feature)], agent027_feature)

    def test_armed_opponent_changes_only_opponent_window_group(self):
        without_enemy = opponent_window_features(open_state())
        with_enemy = opponent_window_features(
            open_state((("enemy", 0, True, (10, 7)),))
        )
        armed_offsets = range(1, len(with_enemy), OPPONENT_FEATURES_PER_ACTION)
        self.assertTrue(np.all(without_enemy[list(armed_offsets)] == 0.0))
        self.assertTrue(np.all(with_enemy[list(armed_offsets)] == 1.0))

    def test_moving_toward_enemy_is_marked_contested(self):
        state = open_state((("enemy", 0, True, (9, 7)),))
        features = opponent_window_features(state)
        right_offset = ACTIONS.index("RIGHT") * OPPONENT_FEATURES_PER_ACTION
        self.assertEqual(features[right_offset + 2], 1.0)

    def test_setup_builds_a_101_input_network(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = SimpleNamespace(
                train=True, seed=0, model_path=Path(directory) / "model.pt",
            )
            callbacks.setup(learner)
        self.assertEqual(learner.policy_net.input_dim, FEATURE_SIZE)


if __name__ == "__main__":
    unittest.main()
