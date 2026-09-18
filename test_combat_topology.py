"""Focused checks for the topology-augmented combat representation."""

# Sahand was here.

import unittest

import numpy as np

from agent_code.combat_fqi_history_antistag_topology_agent.features import (
    local_topology_features,
    state_to_features,
)


def open_field(size=7):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    return field


def make_state(field, position=(3, 3)):
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "bombs": [],
        "explosion_map": np.zeros_like(field),
        "coins": [(1, 1)],
        "self": ("learner", 0, True, position),
        "others": [],
        "user_input": None,
    }


class CombatTopologyFeaturesTest(unittest.TestCase):
    def test_topology_appends_fourteen_features(self):
        state = make_state(open_field())
        self.assertEqual(len(local_topology_features(state)), 14)
        self.assertEqual(len(state_to_features(state)), 46)

    def test_patch_and_aggregates_change_with_local_crates(self):
        open_state = make_state(open_field())
        field = open_field()
        field[2, 3] = 1
        field[3, 2] = 1
        crate_state = make_state(field)
        open_topology = local_topology_features(open_state)
        crate_topology = local_topology_features(crate_state)
        self.assertFalse(np.array_equal(open_topology, crate_topology))
        self.assertEqual(crate_topology[-4], 2)  # two adjacent crates

    def test_outside_board_is_treated_as_wall(self):
        state = make_state(open_field(), position=(1, 1))
        topology = local_topology_features(state)
        self.assertEqual(topology[0], -1)
        self.assertEqual(topology[1], -1)


if __name__ == "__main__":
    unittest.main()
