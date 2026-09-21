"""Focused contract tests for the compact FQI candidate."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_code.Agent_034_compact_fqi_agent import callbacks, train
from agent_code.Agent_034_compact_fqi_agent.features import (
    FEATURE_SIZE, movement_routes, state_to_features,
)
from test_combat_route_features import alias_state


class CompactFQITest(unittest.TestCase):
    def test_opposite_route_witness_and_numeric_contract(self):
        north, south = alias_state((7, 5)), alias_state((7, 9))
        first = state_to_features(north)
        second = state_to_features(south)
        self.assertEqual(first.shape, (FEATURE_SIZE,))
        self.assertFalse(np.array_equal(first, second))
        # Four coin costs occupy indices 0:4; UP and DOWN reverse order.
        self.assertGreater(first[0], first[2])
        self.assertLess(second[0], second[2])
        self.assertEqual(first[19:26].tolist(), [0, 0, 0, 0, 0, 0, 1])
        no_coins = dict(north, coins=[])
        self.assertEqual(movement_routes(no_coins, []), [0.0] * 8)

    def test_cached_observation_and_terminal_masked_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = SimpleNamespace(train=True, seed=0,
                                    model_path=Path(directory) / "trees.pkl")
            callbacks.setup(agent)
            train.setup_training(agent)
            state = alias_state((7, 5))
            features, mask = callbacks.observe(agent, state)
            repeated, repeated_mask = callbacks.observe(agent, state)
            self.assertIs(features, repeated)
            self.assertIs(mask, repeated_mask)
            self.assertEqual(len(agent.positions), 1)
            action = callbacks.act(agent, state)
            train.remember(agent, state, action, None, [], True)
            record = agent.transitions[-1]
            self.assertTrue(record.terminal)
            self.assertIsNone(record.next_features)
            self.assertEqual(train.fitted_targets([record], [None] * 6)[0], record.reward)
            self.assertTrue(record.allowed[record.action])
            self.assertFalse(record.next_allowed.any())
            train.fit_trees(agent)
            callbacks.save_checkpoint(agent, agent.model_path)
            frozen = SimpleNamespace(train=False, model_path=agent.model_path, seed=0)
            callbacks.setup(frozen)
            self.assertEqual(len(frozen.trees), 6)

    def test_nonterminal_backup_excludes_masked_high_value(self):
        class Constant:
            def __init__(self, value):
                self.value = value

            def predict(self, states):
                return np.full(len(states), self.value)

        z = np.zeros(FEATURE_SIZE, dtype=np.float32)
        current = np.zeros(6, dtype=bool)
        current[0] = True
        next_allowed = np.zeros(6, dtype=bool)
        next_allowed[1] = True
        record = train.Transition(z, 0, 2.0, z, False, current,
                                  next_allowed, (), "coin-heaven", (), 1,
                                  0, 1, ())
        trees = [Constant(100), Constant(3)] + [None] * 4
        self.assertAlmostEqual(train.fitted_targets([record], trees)[0],
                               2.0 + train.DISCOUNT * 3)

    def test_post_action_state_uses_following_decision_key(self):
        """The engine gives old and new states the same step number."""
        with tempfile.TemporaryDirectory() as directory:
            agent = SimpleNamespace(train=True, seed=0,
                                    model_path=Path(directory) / "trees.pkl")
            callbacks.setup(agent)
            train.setup_training(agent)
            old = alias_state((7, 5))
            old_features, _ = callbacks.observe(agent, old)
            moved = dict(old, self=("route", 0, False, (7, 6)))
            self.assertEqual(callbacks.state_key(old), callbacks.state_key(moved))

            train.game_events_occurred(agent, old, "UP", moved, [])
            following = dict(moved, step=old["step"] + 1)
            actual_features, actual_mask = callbacks.observe(agent, following)
            next_action = callbacks.act(agent, following)
            train.game_events_occurred(agent, following, next_action,
                                       dict(following), [])
            self.assertEqual(len(agent.transitions), 1)
            record = agent.transitions[-1]
            np.testing.assert_array_equal(record.features, old_features)
            np.testing.assert_array_equal(record.next_features, actual_features)
            np.testing.assert_array_equal(record.next_allowed, actual_mask)
            self.assertFalse(np.array_equal(record.features, record.next_features))
            self.assertEqual(actual_features[19:26].tolist(), [1, 0, 0, 0, 0, 0, 0])

    def test_progress_history_matches_frozen_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = SimpleNamespace(train=True, seed=0,
                                    model_path=Path(directory) / "trees.pkl")
            callbacks.setup(agent)
            train.setup_training(agent)
            old = alias_state((7, 5))
            callbacks.observe(agent, old)
            collected = dict(old, self=("route", 1, False, (7, 6)),
                             coins=[])
            train.game_events_occurred(agent, old, "UP", collected, [])
            following = dict(collected, step=old["step"] + 1)
            feature, _ = callbacks.observe(agent, following)
            self.assertEqual(feature[26], 0)
            self.assertEqual(feature[27], 0)
            self.assertEqual(len(agent.positions), 1)


if __name__ == "__main__":
    unittest.main()
