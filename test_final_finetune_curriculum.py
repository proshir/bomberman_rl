"""Contract tests for the final four-player fine-tuning schedule."""

import unittest

from run_combat_training import episode_plan, parse_args


class FinalFinetuneCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "curriculum": "final-finetune",
            "scenario": "loot-crate",
            "seed": 7,
            "classic_lineups": [
                ["rule_based_agent"] * 3,
                ["imp_li_deep_killer", "imp_alii_arbiter", "rule_based_agent"],
            ],
        }

    def test_solo_warmup_then_three_complete_classic_games(self):
        self.assertEqual(episode_plan(self.config, 1)[0], "coin-heaven")
        self.assertEqual(episode_plan(self.config, 101)[0], "coin-heaven")
        plans = [episode_plan(self.config, episode)
                 for episode in range(301, 306)]
        self.assertEqual(
            [plan[0] for plan in plans],
            ["classic", "classic", "classic", "coin-heaven", "loot-crate"],
        )
        for scenario, lineup, _ in plans[:3]:
            self.assertEqual(scenario, "classic")
            self.assertEqual(len(lineup), 3)

    def test_replay_weights_are_sixty_twenty_twenty(self):
        _, _, weights = episode_plan(self.config, 301)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["coin-heaven"], 0.20)
        self.assertAlmostEqual(weights["loot-crate"], 0.20)
        self.assertAlmostEqual(
            sum(value for name, value in weights.items()
                if name.startswith("classic|")),
            0.60,
        )

    def test_cli_rejects_incomplete_or_weak_lineups(self):
        common = [
            "--curriculum", "Final_finetune", "--rounds", "1",
            "--max-steps", "1", "--output", "/tmp/final-finetune-test-output",
        ]
        with self.assertRaises(SystemExit):
            parse_args(common + ["--classic-lineup", "rule_based_agent"])
        with self.assertRaises(SystemExit):
            parse_args(common + [
                "--classic-lineup", "rule_based_agent", "peaceful_agent",
                "rule_based_agent",
            ])


if __name__ == "__main__":
    unittest.main()
