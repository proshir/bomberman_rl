"""Regression tests for the history and progress context of the new agent."""

# Sahand was here.

import unittest
from types import SimpleNamespace

import numpy as np

from agent_code.combat_fqi_history_antistag_agent import callbacks
from agent_code.combat_fqi_history_antistag_agent.features import (
    ACTIONS,
    state_to_features,
)


def open_field(size=7):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    return field


def make_state(step=1, position=(3, 3), coins=((1, 1),), crates=(), score=0):
    field = open_field()
    for crate in crates:
        field[crate] = 1
    return {
        "round": 1,
        "step": step,
        "field": field,
        "bombs": [],
        "explosion_map": np.zeros_like(field),
        "coins": list(coins),
        "self": ("learner", score, True, position),
        "others": [],
        "user_input": None,
    }


class CombatHistoryFeaturesTest(unittest.TestCase):
    def test_history_inputs_change_only_the_three_appended_features(self):
        state = make_state()
        baseline = state_to_features(state, ACTIONS.index("WAIT"), 0, 0)
        contextual = state_to_features(state, ACTIONS.index("LEFT"), 3, 20)
        np.testing.assert_array_equal(baseline[:-3], contextual[:-3])
        np.testing.assert_array_equal(contextual[-3:], [3, 3, 4])

    def test_crate_destruction_resets_stagnation_and_recent_visits(self):
        learner = SimpleNamespace(
            round_id=None,
            last_progress=None,
            last_progress_step=0,
            previous_action=ACTIONS.index("WAIT"),
            positions=None,
            feature_cache={},
        )
        # setup normally creates the deque; this keeps the test independent of
        # model files while exercising the same episode-local state.
        from collections import deque
        learner.positions = deque(maxlen=8)
        old = make_state(step=10, crates=((4, 3),))
        callbacks.features_for_act(learner, old)
        learner.last_progress_step = 2
        cached = learner.feature_cache[callbacks.state_key(old)]
        learner.feature_cache[callbacks.state_key(old)] = (
            cached[0], ((3, 3), (3, 3)), cached[2], 2
        )
        new = make_state(step=11, position=(3, 3), crates=())
        future = callbacks.next_features(learner, old, "WAIT", new)
        self.assertEqual(future[-2], 0)
        self.assertEqual(future[-1], 0)

    def test_waiting_without_progress_increases_stagnation_bucket(self):
        state = make_state(step=40)
        early = state_to_features(state, ACTIONS.index("WAIT"), 1, 2)
        late = state_to_features(state, ACTIONS.index("WAIT"), 1, 33)
        self.assertLess(early[-1], late[-1])


if __name__ == "__main__":
    unittest.main()
