"""Contract tests for the Agent 038 117-input wide teacher."""

import copy
import unittest

import numpy as np
import torch
from torch import optim

from test_agent032_optimized_features import OptimizedFeatureTests
from agent_code.Agent_036_compact_fqi_robust_agent import features as agent036
from agent_code.Agent_037_tournament_fast_ddqn_agent import features as agent037
from agent_code.Agent_038_symmetric_population_ddqn_agent import features
from agent_code.Agent_038_symmetric_population_ddqn_agent.checkpoint import (
    expand_agent037_checkpoint,
    policy_only_warm_start,
)
from agent_code.Agent_038_symmetric_population_ddqn_agent.replay import (
    CombatEscapeReplayBuffer,
)
from agent_code.Agent_038_symmetric_population_ddqn_agent.symmetry import (
    TRANSFORMS,
    direction_permutation,
    transform_action,
    transform_features,
    transform_mask,
    transform_vector,
)
from agent_code.combat_dqn_agent.model import QNetwork
from run_combat_training import episode_plan


class Agent038Tests(unittest.TestCase):
    def test_101_prefix_and_16_route_suffix_are_exact(self):
        arguments = (
            4, 2, 3, (0, 1, 4), (1, 1, 0),
            ((7, 7), (7, 6), (7, 7)),
        )
        for state in OptimizedFeatureTests()._states():
            state = copy.deepcopy(state)
            expected_prefix = agent037.state_to_features(state, *arguments)
            expected_suffix = np.asarray(
                agent036.movement_routes(state, state["coins"])
                + agent036.movement_routes(
                    state, agent036.feasible_crate_tiles(state)
                ),
                dtype=np.float32,
            )
            actual = features.state_to_features(state, *arguments)
            self.assertEqual(actual.shape, (117,))
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_array_equal(actual[:101], expected_prefix)
            np.testing.assert_array_equal(actual[101:], expected_suffix)

    def test_preallocated_replay_defaults_to_117_inputs(self):
        buffer = CombatEscapeReplayBuffer(4, seed=7)
        buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
        self.assertAlmostEqual(buffer.weights["combat_escape"], 0.10)
        self.assertAlmostEqual(buffer.weights["coin-heaven"], 0.90)
        state = np.arange(117, dtype=np.float32)
        buffer.add(
            state, 1, 0.5, state + 1, False, np.ones(6, dtype=bool)
        )
        self.assertEqual(buffer.states.shape, (4, 117))
        batch = buffer.sample_torch(1, torch.device("cpu"))
        self.assertEqual(tuple(batch.states.shape), (1, 117))

    def test_checkpoint_expansion_preserves_q_and_adam_moments(self):
        torch.manual_seed(11)
        old = QNetwork(101, 6)
        old_target = QNetwork(101, 6)
        old_target.load_state_dict(old.state_dict())
        old_optimizer = optim.Adam(old.parameters(), lr=1e-4)
        sample = torch.randn(5, 101)
        old(sample).square().mean().backward()
        old_optimizer.step()
        checkpoint = {
            "input_dim": 101,
            "n_actions": 6,
            "policy_state_dict": old.state_dict(),
            "target_state_dict": old_target.state_dict(),
            "optimizer_state_dict": old_optimizer.state_dict(),
        }

        migrated = expand_agent037_checkpoint(checkpoint)
        wide = QNetwork(117, 6)
        wide_target = QNetwork(117, 6)
        wide_optimizer = optim.Adam(wide.parameters(), lr=1e-4)
        wide.load_state_dict(migrated["policy_state_dict"])
        wide_target.load_state_dict(migrated["target_state_dict"])
        wide_optimizer.load_state_dict(migrated["optimizer_state_dict"])

        suffix = torch.randn(5, 16)
        with torch.no_grad():
            torch.testing.assert_close(
                wide(torch.cat((sample, suffix), dim=1)), old(sample),
                rtol=0, atol=0,
            )
            torch.testing.assert_close(
                wide_target(torch.cat((sample, suffix), dim=1)),
                old_target(sample), rtol=0, atol=0,
            )
        old_first = next(iter(old_optimizer.state.values()))
        wide_first = next(iter(wide_optimizer.state.values()))
        torch.testing.assert_close(
            wide_first["exp_avg"][:, :101], old_first["exp_avg"],
            rtol=0, atol=0,
        )
        self.assertEqual(tuple(wide_first["exp_avg"].shape), (128, 117))
        self.assertEqual(
            torch.count_nonzero(wide_first["exp_avg"][:, 101:]).item(), 0
        )

    def test_unrelated_checkpoint_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "accepts 101- or 117-input"):
            expand_agent037_checkpoint({"input_dim": 28})

    def test_policy_only_warm_start_resets_training_state(self):
        network = QNetwork(101, 6)
        target = QNetwork(101, 6)
        optimizer = optim.Adam(network.parameters(), lr=3e-4)
        network(torch.randn(2, 101)).sum().backward()
        optimizer.step()
        checkpoint = {
            "input_dim": 101,
            "n_actions": 6,
            "policy_state_dict": network.state_dict(),
            "target_state_dict": target.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epsilon": 0.05,
            "env_steps": 99,
            "combat_env_steps": 88,
            "optimizer_steps": 77,
        }
        result = policy_only_warm_start(checkpoint)
        self.assertEqual(result["input_dim"], 117)
        self.assertEqual(result["optimizer_state_dict"]["state"], {})
        self.assertEqual(result["epsilon"], 0.30)
        self.assertEqual(result["env_steps"], 0)
        self.assertEqual(result["combat_env_steps"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        for name, value in result["policy_state_dict"].items():
            torch.testing.assert_close(
                value, result["target_state_dict"][name], rtol=0, atol=0
            )

    def test_agent038_population_curriculum_uses_registered_shares(self):
        lineups = [
            ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
            ["imp_li_deep_killer", "imp_alii_arbiter", "rule_based_agent"],
        ]
        config = {
            "curriculum": "agent038-population",
            "classic_lineups": lineups,
            "seed": 3,
        }
        plans = [episode_plan(config, episode) for episode in range(301, 321)]
        scenarios = [plan[0] for plan in plans]
        self.assertEqual(scenarios.count("classic"), 14)
        self.assertEqual(scenarios.count("coin-heaven"), 3)
        self.assertEqual(scenarios.count("loot-crate"), 3)
        for _, _, weights in plans:
            self.assertAlmostEqual(weights["coin-heaven"], 0.15)
            self.assertAlmostEqual(weights["loot-crate"], 0.15)
            classic_weight = sum(
                value for name, value in weights.items()
                if name.startswith("classic|")
            )
            self.assertAlmostEqual(classic_weight, 0.70)

    def test_all_eight_symmetries_are_unique_and_invertible(self):
        self.assertEqual(len(set(direction_permutation(t) for t in TRANSFORMS)), 8)
        vector = np.arange(117, dtype=np.float32)
        mask = np.asarray([True, False, True, False, True, False])
        for transform in TRANSFORMS:
            transformed = transform_features(vector, transform)
            transformed_mask = transform_mask(mask, transform)
            inverse = next(
                candidate for candidate in TRANSFORMS
                if all(
                    transform_action(transform_action(action, transform), candidate)
                    == action
                    for action in range(6)
                )
            )
            np.testing.assert_array_equal(
                transform_features(transformed, inverse), vector
            )
            np.testing.assert_array_equal(
                transform_mask(transformed_mask, inverse), mask
            )

    def test_symmetry_keeps_action_conditioned_blocks_aligned(self):
        vector = np.zeros(117, dtype=np.float32)
        vector[1] = 11
        vector[10] = 12
        vector[29] = 1
        vector[46 + 1] = 13
        vector[65 + 1 * 6:65 + 2 * 6] = np.arange(20, 26)
        vector[101 + 1] = 14
        transform = (1, False)
        destination = direction_permutation(transform)[1]
        result = transform_features(vector, transform)
        self.assertEqual(result[destination], 11)
        self.assertEqual(result[9 + destination], 12)
        self.assertEqual(result[29], destination)
        self.assertEqual(result[46 + destination], 13)
        np.testing.assert_array_equal(
            result[65 + destination * 6:65 + (destination + 1) * 6],
            np.arange(20, 26),
        )
        self.assertEqual(result[101 + destination], 14)

    def test_symmetry_matches_features_recomputed_on_transformed_board(self):
        def point(position, transform, shape):
            centre = ((shape[0] - 1) // 2, (shape[1] - 1) // 2)
            relative = (
                int(position[0]) - centre[0],
                int(position[1]) - centre[1],
            )
            transformed = transform_vector(relative, transform)
            return transformed[0] + centre[0], transformed[1] + centre[1]

        def board(array, transform):
            result = np.empty_like(array)
            for x in range(array.shape[0]):
                for y in range(array.shape[1]):
                    result[point((x, y), transform, array.shape)] = array[x, y]
            return result

        def game_state(state, transform):
            result = copy.deepcopy(state)
            shape = state["field"].shape
            result["field"] = board(state["field"], transform)
            result["explosion_map"] = board(
                state["explosion_map"], transform
            )
            name, score, armed, position = state["self"]
            result["self"] = (
                name, score, armed, point(position, transform, shape)
            )
            result["others"] = [
                (name, score, armed, point(position, transform, shape))
                for name, score, armed, position in state["others"]
            ]
            result["bombs"] = [
                (point(position, transform, shape), timer)
                for position, timer in state["bombs"]
            ]
            result["coins"] = [
                point(position, transform, shape)
                for position in state["coins"]
            ]
            return result

        for source in OptimizedFeatureTests()._states():
            shape = source["field"].shape
            history = ((7, 7), (7, 6), (8, 6), (8, 7))
            arguments = (1, 2, 3, (0, 1, 4, 3), (1, 1, 0, 1), history)
            original = features.state_to_features(source, *arguments)
            for transform in TRANSFORMS:
                transformed_arguments = (
                    transform_action(arguments[0], transform),
                    arguments[1],
                    arguments[2],
                    tuple(
                        transform_action(action, transform)
                        for action in arguments[3]
                    ),
                    arguments[4],
                    tuple(
                        point(position, transform, shape)
                        for position in arguments[5]
                    ),
                )
                recomputed = features.state_to_features(
                    game_state(source, transform), *transformed_arguments
                )
                np.testing.assert_allclose(
                    transform_features(original, transform, source),
                    recomputed,
                    rtol=0,
                    atol=1e-7,
                )


if __name__ == "__main__":
    unittest.main()
