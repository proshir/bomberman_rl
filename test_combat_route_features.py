"""Regression tests for the route-aware Double-DQN representation.

Sahand was here.
"""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_code.Agent_022_combat_ddqn_route_agent.features import (
    FEATURE_SIZE,
    action_route_features,
    pruned_topology_features,
    state_to_features,
)
from agent_code.Agent_022_combat_ddqn_route_agent import callbacks
from agent_code.combat_dqn_r_topology_agent.features import (
    state_to_features as topology_features,
)
from agent_code.combat_fqi_history_antistag_agent.features import ACTIONS


def normal_field():
    """Build the normal 17x17 fixed-wall layout used in the alias witness."""
    field = np.zeros((17, 17), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    for x in range(17):
        for y in range(17):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = -1
    return field


def alias_state(remote_crate):
    """Return one side of the documented route-aliasing board pair."""
    field = normal_field()
    field[8, 7] = 1
    field[remote_crate] = 1
    return {
        "round": 1,
        "step": 100,
        "field": field,
        "bombs": [],
        "explosion_map": np.zeros_like(field),
        "coins": [(9, 7)],
        "self": ("route", 0, False, (7, 7)),
        "others": [],
        "user_input": None,
    }


class CombatRouteFeaturesTest(unittest.TestCase):
    def test_route_features_resolve_the_existing_alias_witness(self):
        north_crate = alias_state((7, 5))
        south_crate = alias_state((7, 9))

        # The old 46-vector cannot see either remote crate and is identical.
        old_north = topology_features(north_crate, ACTIONS.index("WAIT"), 0, 0)
        old_south = topology_features(south_crate, ACTIONS.index("WAIT"), 0, 0)
        np.testing.assert_array_equal(old_north, old_south)

        north_routes = action_route_features(north_crate, north_crate["coins"])
        south_routes = action_route_features(south_crate, south_crate["coins"])
        # Feature pairs are (normalized route cost, reachable) for
        # UP, RIGHT, DOWN, LEFT.  The uniquely route-improving direction flips.
        self.assertGreater(north_routes[0], north_routes[4])
        self.assertLess(south_routes[0], south_routes[4])
        self.assertEqual(north_routes[1], 1.0)
        self.assertEqual(south_routes[5], 1.0)

        route_north = state_to_features(north_crate, ACTIONS.index("WAIT"), 0, 0)
        route_south = state_to_features(south_crate, ACTIONS.index("WAIT"), 0, 0)
        self.assertFalse(np.array_equal(route_north, route_south))

    def test_feature_size_pruned_topology_and_remaining_time(self):
        state = alias_state((7, 5))
        features = state_to_features(state, ACTIONS.index("LEFT"), 2, 10)
        self.assertEqual(len(features), FEATURE_SIZE)
        self.assertEqual(len(pruned_topology_features(state)), 9)
        self.assertAlmostEqual(features[-1], 0.75)

    def test_unreachable_routes_are_explicit(self):
        state = alias_state((7, 5))
        state["coins"] = []
        routes = action_route_features(state, state["coins"])
        np.testing.assert_array_equal(routes, np.zeros(8))

    def test_callbacks_build_a_route_network_and_request_ddqn(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = SimpleNamespace(
                train=True,
                seed=0,
                model_path=Path(directory) / "training.pkl",
            )
            callbacks.setup(learner)
        self.assertEqual(learner.policy_net.input_dim, FEATURE_SIZE)
        self.assertEqual(learner.dqn_algorithm, "ddqn")


if __name__ == "__main__":
    unittest.main()
