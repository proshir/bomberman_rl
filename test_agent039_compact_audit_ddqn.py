"""Contract tests for Agent 039's 104-input compact audit schema."""

import copy
import unittest

import numpy as np
import torch
from torch import optim

from test_agent032_optimized_features import OptimizedFeatureTests
from agent_code.Agent_038_symmetric_population_ddqn_agent import features as wide
from agent_code.Agent_039_compact_audit_ddqn_agent import features
from agent_code.Agent_039_compact_audit_ddqn_agent.checkpoint import (
    compact_agent038_checkpoint,
    policy_only_warm_start,
)
from agent_code.Agent_039_compact_audit_ddqn_agent.replay import (
    CombatEscapeReplayBuffer,
)
from agent_code.Agent_039_compact_audit_ddqn_agent.symmetry import (
    TRANSFORMS,
    direction_permutation,
    transform_action,
    transform_features,
    transform_mask,
    transform_vector,
)
from agent_code.combat_dqn_agent.model import QNetwork


class Agent039Tests(unittest.TestCase):
    def test_compact_mapping_is_104_and_finite(self):
        arguments = (
            4, 2, 3, (0, 1, 4), (1, 1, 0),
            ((7, 7), (7, 6), (7, 7)),
        )
        for state in OptimizedFeatureTests()._states():
            state = copy.deepcopy(state)
            source = wide.state_to_features(state, *arguments)
            actual = features.state_to_features(state, *arguments)
            self.assertEqual(source.shape, (117,))
            self.assertEqual(actual.shape, (104,))
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.isfinite(actual).all())
            expected = np.concatenate((
                source[0:29], source[30:32], source[32:36], source[37:41],
                source[41:46], source[46:65],
                np.asarray([features.any_armed_opponent(state)], dtype=np.float32),
                *[source[65 + 6 * action + 2:65 + 6 * action + 6]
                  for action in range(6)],
                source[101:117],
            ))
            np.testing.assert_array_equal(actual, expected)

    def test_removed_semantics_are_verified_on_representative_states(self):
        for state in OptimizedFeatureTests()._states():
            source = wide.state_to_features(
                copy.deepcopy(state), 4, 2, 3, (0, 1, 4), (1, 1, 0),
                ((7, 7), (7, 6), (7, 7)),
            )
            self.assertEqual(source[36], 0.0)
            self.assertEqual(source[65], 1.0 - source[0])
            self.assertEqual(source[71], 1.0 - source[1])
            self.assertEqual(source[77], 1.0 - source[2])
            self.assertEqual(source[83], 1.0 - source[3])
            self.assertEqual(source[89], 1.0)
            self.assertEqual(source[95], source[13])

    def test_replay_defaults_to_104_inputs(self):
        buffer = CombatEscapeReplayBuffer(4, seed=7)
        buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
        state = np.arange(104, dtype=np.float32)
        buffer.add(state, 1, 0.5, state + 1, False, np.ones(6, dtype=bool))
        self.assertEqual(buffer.states.shape, (4, 104))
        batch = buffer.sample_torch(1, torch.device("cpu"))
        self.assertEqual(tuple(batch.states.shape), (1, 104))

    def test_checkpoint_migration_maps_117_to_104(self):
        torch.manual_seed(11)
        old = QNetwork(117, 6)
        target = QNetwork(117, 6)
        target.load_state_dict(old.state_dict())
        optimizer = optim.Adam(old.parameters(), lr=1e-4)
        old(torch.randn(2, 117)).sum().backward()
        optimizer.step()
        checkpoint = {
            "input_dim": 117,
            "n_actions": 6,
            "policy_state_dict": old.state_dict(),
            "target_state_dict": target.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        migrated = compact_agent038_checkpoint(checkpoint)
        self.assertEqual(migrated["input_dim"], 104)
        self.assertEqual(
            tuple(migrated["policy_state_dict"]["network.0.weight"].shape),
            (128, 104),
        )
        compact = QNetwork(104, 6)
        compact.load_state_dict(migrated["policy_state_dict"])
        self.assertEqual(compact(torch.randn(2, 104)).shape, (2, 6))

    def test_policy_only_warm_start_resets_state(self):
        network = QNetwork(117, 6)
        target = QNetwork(117, 6)
        optimizer = optim.Adam(network.parameters(), lr=3e-4)
        network(torch.randn(2, 117)).sum().backward()
        optimizer.step()
        checkpoint = {
            "input_dim": 117,
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
        self.assertEqual(result["input_dim"], 104)
        self.assertEqual(result["optimizer_state_dict"]["state"], {})
        self.assertEqual(result["epsilon"], 0.30)
        self.assertEqual(result["env_steps"], 0)
        self.assertEqual(result["combat_env_steps"], 0)
        self.assertEqual(result["optimizer_steps"], 0)

    def test_all_eight_symmetries_are_invertible(self):
        self.assertEqual(len(set(direction_permutation(t) for t in TRANSFORMS)), 8)
        vector = np.arange(104, dtype=np.float32)
        mask = np.asarray([True, False, True, False, True, False])
        for transform in TRANSFORMS:
            transformed = transform_features(vector, transform)
            transformed_mask = transform_mask(mask, transform)
            inverse = next(
                candidate for candidate in TRANSFORMS
                if all(
                    transform_action(transform_action(action, transform), candidate)
                    == action for action in range(6)
                )
            )
            np.testing.assert_array_equal(
                transform_features(transformed, inverse), vector
            )
            np.testing.assert_array_equal(
                transform_mask(transformed_mask, inverse), mask
            )

    def test_symmetry_permutates_compact_action_blocks_and_routes(self):
        vector = np.zeros(104, dtype=np.float32)
        vector[1] = 11
        vector[10] = 12
        vector[44 + 1] = 13
        vector[64 + 1 * 4:64 + 2 * 4] = np.arange(20, 24)
        vector[88 + 1] = 14
        transform = (1, False)
        destination = direction_permutation(transform)[1]
        result = transform_features(vector, transform)
        self.assertEqual(result[destination], 11)
        self.assertEqual(result[9 + destination], 12)
        self.assertEqual(result[44 + destination], 13)
        np.testing.assert_array_equal(
            result[64 + destination * 4:64 + (destination + 1) * 4],
            np.arange(20, 24),
        )
        self.assertEqual(result[88 + destination], 14)


if __name__ == "__main__":
    unittest.main()
