"""Contract tests for Agent 036's timing, replay, and policy fixes."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import events as e
from agent_code.Agent_036_compact_fqi_robust_agent import callbacks, train
from agent_code.Agent_036_compact_fqi_robust_agent.features import (
    FEATURE_SIZE, bomb_escape_count, candidate_crate_tiles,
    feasible_crate_tiles, state_to_features,
)
from agent_code.Agent_036_compact_fqi_robust_agent.safety import (
    ACTIONS, safe_action_indices, surviving_followup_actions,
)
from agent_code.Agent_036_compact_fqi_robust_agent.symmetry import (
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


class Agent036Test(unittest.TestCase):
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

    def test_bomb_escape_count_uses_original_search_timeline(self):
        state = open_state()
        state["bombs"] = [((5, 3), 2)]
        state["explosion_map"][4, 3] = 1
        state["others"] = [("enemy", 0, True, (7, 7))]
        followups = surviving_followup_actions(state, "BOMB")
        self.assertEqual(bomb_escape_count(state), len(followups))
        self.assertGreater(len(followups), 0)
        self.assertTrue(set(followups).issubset(set(ACTIONS[:5])))

    def test_useful_bomb_restriction_is_configurable(self):
        state = open_state()
        state["field"][3, 3] = 0
        self.assertNotIn(ACTIONS.index("BOMB"), safe_action_indices(state))
        self.assertIn(
            ACTIONS.index("BOMB"),
            safe_action_indices(state, require_useful_bomb=False),
        )

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

    def test_rare_group_does_not_shrink_entire_fit(self):
        z = np.zeros(28, dtype=np.float32)
        mask = np.ones(6, dtype=bool)
        records = [
            train.Transition(
                z, 0, 0.0, None, True, mask, mask, (), "coin-heaven",
                (), 1, 0, step, train.CHECKPOINT_SCHEMAS,
            )
            for step in range(40000)
        ]
        records.extend(
            train.Transition(
                z, 0, 0.0, None, True, mask, mask, (), "loot-crate",
                (), 1, 0, step, train.CHECKPOINT_SCHEMAS,
            )
            for step in range(100)
        )
        selected, sample_weights, metadata = train.balanced_records(
            records, {"coin-heaven": .5, "loot-crate": .5},
            max_records=24000, return_metadata=True)
        self.assertEqual(len(selected), 24000)
        self.assertEqual(metadata["selected"][
            ("loot-crate", ())], 100)
        self.assertGreater(metadata["selected"][("coin-heaven", ())], 200)
        self.assertAlmostEqual(
            metadata["effective"]["coin-heaven"], .5, places=6)
        self.assertAlmostEqual(
            metadata["effective"]["loot-crate"], .5, places=6)
        self.assertEqual(len(sample_weights), len(selected))

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
            agent.replay_weights = {"coin-heaven": .8, "loot-crate": .2}
            agent.training_config = {"curriculum": "mixed"}
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
            self.assertEqual(
                resumed.replay_weights,
                {"coin-heaven": .8, "loot-crate": .2},
            )
            self.assertEqual(resumed.training_config, {"curriculum": "mixed"})
            self.assertEqual(len(resumed.transitions), 1)
            self.assertEqual(resumed.rng.random(), expected_next)
            frozen = SimpleNamespace(train=False, seed=0, model_path=path)
            callbacks.setup(frozen)
            self.assertEqual(len(frozen.trees), 6)

    def test_callback_step_convention_matches_next_actual_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = SimpleNamespace(train=True, seed=0,
                                    model_path=Path(directory) / "training.pkl")
            callbacks.setup(agent)
            train.setup_training(agent)
            old = open_state()
            old_features, _ = callbacks.observe(agent, old)
            new = dict(old, self=("learner", 0, True, (2, 3)),
                       step=old["step"])
            train.game_events_occurred(agent, old, "RIGHT", new, [])
            following = callbacks.next_decision_state(new)
            actual_features, actual_mask = callbacks.observe(agent, following)
            next_action = callbacks.act(agent, following)
            train.game_events_occurred(agent, following, next_action,
                                       dict(following), [])
            self.assertEqual(callbacks.next_decision_key(new),
                             callbacks.state_key(following))
            record = agent.transitions[-1]
            np.testing.assert_array_equal(record.features, old_features)
            np.testing.assert_array_equal(record.next_features,
                                          actual_features)
            np.testing.assert_array_equal(record.next_allowed, actual_mask)

    def test_global_crate_change_does_not_reset_personal_history(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = SimpleNamespace(train=True, seed=0,
                                    model_path=Path(directory) / "training.pkl")
            callbacks.setup(agent)
            train.setup_training(agent)
            state = open_state()
            callbacks.observe(agent, state)
            changed = dict(state, step=2, field=state["field"].copy())
            changed["field"][3, 3] = 0
            feature, _ = callbacks.observe(agent, changed)
            self.assertEqual(len(agent.positions), 2)
            self.assertEqual(feature[26], 1.0)

    def test_attributable_progress_resets_timer_but_keeps_history(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = SimpleNamespace(train=True, seed=0,
                                    model_path=Path(directory) / "training.pkl")
            callbacks.setup(agent)
            train.setup_training(agent)
            state = open_state()
            callbacks.observe(agent, state)
            callbacks.note_progress_events(agent, [e.CRATE_DESTROYED], 2)
            next_state = dict(state, step=3)
            feature, _ = callbacks.observe(agent, next_state)
            self.assertEqual(feature[27], 0.0)
            self.assertEqual(feature[26], 1.0)


if __name__ == "__main__":
    unittest.main()
