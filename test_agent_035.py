"""Contract tests for corrected compact FQI and its D4 augmentation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import events as e
from agent_code.Agent_035_compact_fqi_symmetry_agent import callbacks, train
from agent_code.Agent_035_compact_fqi_symmetry_agent.features import (
    FEATURE_SIZE, candidate_crate_tiles, feasible_crate_tiles, state_to_features,
)
from agent_code.Agent_035_compact_fqi_symmetry_agent.symmetry import (
    TRANSFORMS, direction_permutation, transform_action, transform_features,
    transform_mask, transform_transition,
)
from test_combat_route_features import alias_state


def open_state():
    field = np.zeros((9, 9), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    field[3, 3] = 1
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("learner", 0, True, (1, 3)), "others": [],
        "bombs": [], "coins": [], "explosion_map": np.zeros_like(field),
    }


class Agent035Test(unittest.TestCase):
    def test_route_witness_and_full_blast_range_crate_targets(self):
        state = open_state()
        self.assertIn((1, 3), candidate_crate_tiles(state["field"]))
        self.assertIn((1, 3), feasible_crate_tiles(state))
        north, south = alias_state((7, 5)), alias_state((7, 9))
        first, second = state_to_features(north), state_to_features(south)
        self.assertEqual(first.shape, (FEATURE_SIZE,))
        self.assertFalse(np.array_equal(first, second))
        self.assertGreater(first[0], first[2])
        self.assertLess(second[0], second[2])

    def test_terminal_reward_counts_final_events_once(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = SimpleNamespace(train=True, seed=0,
                                    model_path=Path(directory) / "training.pkl")
            callbacks.setup(agent)
            train.setup_training(agent)
            state = open_state()
            action = callbacks.act(agent, state)
            train.game_events_occurred(agent, state, action,
                                       dict(state), [e.COIN_COLLECTED])
            train.end_of_round(agent, state, action,
                               [e.COIN_COLLECTED, e.SURVIVED_ROUND])
            self.assertEqual(len(agent.transitions), 1)
            record = agent.transitions[-1]
            self.assertEqual(record.events, (e.COIN_COLLECTED, e.SURVIVED_ROUND))
            self.assertTrue(record.terminal)
            self.assertAlmostEqual(record.reward, 1.49)
            self.assertEqual(train.fitted_targets([record], [None] * 6)[0],
                             record.reward)

    def test_symmetry_transforms_actions_features_and_both_masks(self):
        self.assertEqual(len(TRANSFORMS), 8)
        self.assertEqual(len({direction_permutation(t) for t in TRANSFORMS}), 8)
        features = np.arange(28, dtype=np.float32)
        mask = np.array([True, False, False, True, True, False])
        record = train.Transition(features, 0, 2.0, features, False,
                                  mask, mask, (), "coin-heaven", (), 1, 0, 1,
                                  train.CHECKPOINT_SCHEMAS)
        for transform in TRANSFORMS:
            mapped = transform_transition(record, transform)
            self.assertEqual(mapped.action, transform_action(0, transform))
            self.assertTrue(mapped.allowed[mapped.action])
            np.testing.assert_array_equal(mapped.allowed,
                                          transform_mask(mask, transform))
            np.testing.assert_array_equal(mapped.next_allowed,
                                          transform_mask(mask, transform))
            np.testing.assert_array_equal(mapped.features,
                                          transform_features(features, transform))
            self.assertEqual(mapped.features[19 + mapped.action], features[19])
            self.assertEqual(mapped.features[26], features[26])
            self.assertEqual(mapped.features[27], features[27])

    def test_scenario_weights_do_not_multiply_with_lineup_count(self):
        records = []
        z = np.zeros(28, dtype=np.float32)
        mask = np.ones(6, dtype=bool)
        for scenario, lineups in (("coin-heaven", [()]),
                                  ("loot-crate", [()]),
                                  ("classic", [("a",), ("b",), ("c",)])):
            for lineup in lineups:
                for step in range(20):
                    records.append(train.Transition(
                        z, 0, 0.0, None, True, mask, mask, (), scenario,
                        lineup, 1, 0, step, train.CHECKPOINT_SCHEMAS))
        selected = train.balanced_records(
            records, {"coin-heaven": .2, "loot-crate": .2, "classic": .6},
            max_records=100)
        counts = {scenario: sum(row.scenario == scenario for row in selected)
                  for scenario in ("coin-heaven", "loot-crate", "classic")}
        self.assertEqual(counts, {"coin-heaven": 20, "loot-crate": 20,
                                  "classic": 60})

    def test_checkpoint_round_trip_restores_replay_epsilon_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.pkl"
            agent = SimpleNamespace(train=True, seed=7, model_path=path)
            callbacks.setup(agent)
            train.setup_training(agent)
            state = open_state()
            action = callbacks.act(agent, state)
            train.remember(agent, state, action, None, [], True)
            agent.completed_rounds = 4
            agent.epsilon = .37
            agent.env_steps = 123
            agent.action_count = 17
            agent.fit_seconds = 2.5
            callbacks.save_checkpoint(agent, path)
            expected_next = agent.rng.random()
            resumed = SimpleNamespace(train=True, seed=7, model_path=path,
                                      resume_path=path)
            callbacks.setup(resumed)
            train.setup_training(resumed)
            self.assertEqual(resumed.completed_rounds, 4)
            self.assertEqual(resumed.epsilon, .37)
            self.assertEqual(resumed.env_steps, 123)
            self.assertEqual(resumed.action_count, 17)
            self.assertEqual(resumed.fit_seconds, 2.5)
            self.assertEqual(len(resumed.transitions), 1)
            self.assertEqual(resumed.rng.random(), expected_next)
            frozen = SimpleNamespace(train=False, seed=0, model_path=path)
            callbacks.setup(frozen)
            self.assertEqual(len(frozen.trees), 6)


if __name__ == "__main__":
    unittest.main()
