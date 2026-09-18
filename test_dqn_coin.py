"""Focused tests for the coin-heaven DQN representation and action mask."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agent_code.dqn_coin_agent.callbacks import ACTIONS, legal_action_indices
from agent_code.dqn_coin_agent.features import FEATURE_SIZE, state_to_features
from agent_code.dqn_coin_agent.model import QNetwork


def example_state():
    field = np.zeros((7, 7), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    field[3, 2] = -1
    return {
        'round': 1,
        'step': 4,
        'field': field,
        'bombs': [],
        'explosion_map': np.zeros_like(field),
        'coins': [(5, 5), (1, 5)],
        'self': ('dqn', 0, True, (3, 3)),
        'others': [],
    }


class DqnCoinTests(unittest.TestCase):
    def test_features_have_fixed_finite_shape(self):
        features = state_to_features(example_state())
        self.assertEqual(features.shape, (FEATURE_SIZE,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_legal_actions_mask_walls_and_wait(self):
        state = example_state()
        legal = legal_action_indices(state)
        self.assertNotIn(ACTIONS.index('UP'), legal)  # the custom wall at (3, 2)
        self.assertIn(ACTIONS.index('RIGHT'), legal)
        self.assertIn(ACTIONS.index('DOWN'), legal)
        self.assertIn(ACTIONS.index('LEFT'), legal)
        self.assertIn(ACTIONS.index('WAIT'), legal)

    def test_legal_actions_mask_other_agent(self):
        state = example_state()
        state['others'] = [('other', 0, True, (4, 3))]
        self.assertNotIn(ACTIONS.index('RIGHT'), legal_action_indices(state))

    def test_network_has_expected_action_values(self):
        network = QNetwork(FEATURE_SIZE, len(ACTIONS))
        output = network(torch.zeros((3, FEATURE_SIZE), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3, len(ACTIONS)))

    def test_cpu_checkpoint_round_trip(self):
        network = QNetwork(FEATURE_SIZE, len(ACTIONS))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save({'model_state': network.state_dict()}, path)
            restored = QNetwork(FEATURE_SIZE, len(ACTIONS))
            restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state'])
            for first, second in zip(network.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(first, second))


if __name__ == '__main__':
    unittest.main()
