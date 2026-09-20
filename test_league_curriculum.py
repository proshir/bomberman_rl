"""Deterministic checks for the one-learner league schedule."""

import unittest

from run_combat_training import episode_plan


class LeagueCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "curriculum": "league-combat", "scenario": "loot-crate",
            "seed": 7,
            "classic_lineups": [["op_a"], ["op_b", "op_c"],
                                ["op_a", "op_b", "op_c"]],
        }

    def test_warmup_and_retained_solo(self):
        self.assertEqual(episode_plan(self.config, 1)[0], "coin-heaven")
        self.assertEqual(episode_plan(self.config, 101)[0], "coin-heaven")
        self.assertEqual(
            [episode_plan(self.config, episode)[0]
             for episode in range(301, 306)],
            ["classic", "classic", "classic", "coin-heaven", "loot-crate"],
        )

    def test_balanced_reproducible_lineup_shuffle(self):
        chosen = [tuple(episode_plan(self.config, episode)[1])
                  for episode in range(301, 306) if
                  episode_plan(self.config, episode)[0] == "classic"]
        self.assertCountEqual(chosen, map(tuple, self.config["classic_lineups"]))
        self.assertEqual(chosen, [tuple(episode_plan(self.config, episode)[1])
                                  for episode in range(301, 304)])

    def test_replay_weights_aggregate_duplicate_slots(self):
        self.config["classic_lineups"].append(["op_a"])
        _, _, weights = episode_plan(self.config, 301)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["classic|op_a"], 0.30)


if __name__ == "__main__":
    unittest.main()
