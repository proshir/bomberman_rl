"""Contract tests for Agent 043's strict staged league curriculum."""

import tempfile
import unittest
from pathlib import Path

from run_combat_training import episode_plan, parse_args


class Agent043StagedLeagueTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "curriculum": "agent043-staged-league",
            "scenario": "loot-crate",
            "seed": 3,
            "classic_lineups": [
                ["rule_based_agent"] * 3,
                ["imp_alii_arbiter", "imp_li_deep_killer",
                 "rule_based_agent"],
            ],
        }

    def test_exact_three_stage_schedule(self):
        first = [episode_plan(self.config, episode)
                 for episode in range(1, 101)]
        self.assertEqual({plan[0] for plan in first}, {"coin-heaven"})
        self.assertEqual(first[0][2], {"coin-heaven": 1.0})

        middle = [episode_plan(self.config, episode)
                  for episode in range(101, 301)]
        self.assertEqual(
            [sum(plan[0] == scenario for plan in middle)
             for scenario in ("coin-heaven", "loot-crate")],
            [100, 100],
        )
        self.assertEqual(
            middle[0][2], {"coin-heaven": 0.5, "loot-crate": 0.5},
        )

        final = [episode_plan(self.config, episode)
                 for episode in range(301, 901)]
        self.assertEqual({plan[0] for plan in final}, {"classic"})
        self.assertTrue(all(len(plan[1]) == 3 for plan in final))
        counts = {
            tuple(lineup): sum(tuple(plan[1]) == tuple(lineup)
                               for plan in final)
            for lineup in self.config["classic_lineups"]
        }
        self.assertEqual(set(counts.values()), {300})
        self.assertTrue(all(sum(plan[2].values()) == 1.0 for plan in final))

    def test_cli_requires_complete_strong_lineups(self):
        with tempfile.TemporaryDirectory() as directory:
            common = [
                "--agent", "Agent_043_novelty_credit_ddqn_agent",
                "--curriculum", "agent043-staged-league",
                "--rounds", "900", "--output", str(Path(directory) / "run"),
            ]
            parsed = parse_args(common + [
                "--classic-lineup", "rule_based_agent", "rule_based_agent",
                "rule_based_agent",
            ])
            self.assertEqual(parsed.curriculum, "agent043-staged-league")
            with self.assertRaises(SystemExit):
                parse_args(common + [
                    "--classic-lineup", "rule_based_agent",
                ])
            with self.assertRaises(SystemExit):
                parse_args(common + [
                    "--classic-lineup", "rule_based_agent",
                    "coin_collector_agent", "rule_based_agent",
                ])

    def test_memory_retention_is_randomized_exact_eighty_ten_ten(self):
        config = dict(self.config)
        config["curriculum"] = "agent043-staged-league-memory-retention"
        final = [episode_plan(config, episode)
                 for episode in range(301, 901)]
        counts = {
            scenario: sum(plan[0] == scenario for plan in final)
            for scenario in ("classic", "coin-heaven", "loot-crate")
        }
        self.assertEqual(counts, {
            "classic": 480, "coin-heaven": 60, "loot-crate": 60,
        })
        for start in range(0, len(final), 10):
            block = final[start:start + 10]
            self.assertEqual(sum(plan[0] == "classic" for plan in block), 8)
            self.assertEqual(sum(plan[0] == "coin-heaven" for plan in block), 1)
            self.assertEqual(sum(plan[0] == "loot-crate" for plan in block), 1)
        lineup_counts = {
            tuple(lineup): sum(
                plan[0] == "classic" and tuple(plan[1]) == tuple(lineup)
                for plan in final
            )
            for lineup in config["classic_lineups"]
        }
        self.assertEqual(set(lineup_counts.values()), {240})
        self.assertEqual(final, [episode_plan(config, episode)
                                 for episode in range(301, 901)])


if __name__ == "__main__":
    unittest.main()
