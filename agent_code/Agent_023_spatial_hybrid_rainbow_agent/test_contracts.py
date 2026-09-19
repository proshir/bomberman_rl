"""Fast deterministic Phase-0 checks (run with ``python -m unittest``)."""

import unittest
import numpy as np
from .config import FEATURE_SIZE
from .features import split_features, state_to_features
from .safety import safe_action_indices

def state():
    field = np.zeros((17, 17), dtype=int); field[[0, -1], :] = -1; field[:, [0, -1]] = -1
    return {"round": 1, "step": 1, "field": field, "bombs": [], "explosion_map": np.zeros_like(field), "coins": [(3, 3)], "self": ("me", 0, True, (1, 1)), "others": [], "user_input": None}

class ContractTests(unittest.TestCase):
    def test_fixed_feature_layout(self):
        vector = state_to_features(state())
        self.assertEqual(vector.size, FEATURE_SIZE)
        spatial, global_values, action_values = split_features(vector)
        self.assertEqual(spatial.shape, (1, 20, 17, 17)); self.assertEqual(global_values.shape, (1, 36)); self.assertEqual(action_values.shape, (1, 6, 16))
    def test_illegal_action_has_zero_action_bank(self):
        value = state_to_features(state()); _, _, actions = split_features(value)
        self.assertTrue(np.all(actions[0, 0] == 0))  # UP enters the border wall.
    def test_safety_never_returns_illegal_up(self):
        self.assertNotIn(0, safe_action_indices(state()))

if __name__ == "__main__": unittest.main()
