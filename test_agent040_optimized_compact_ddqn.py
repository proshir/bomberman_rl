"""Contract and equivalence tests for Agent 040."""

import copy
import unittest

import numpy as np
import torch

from test_agent032_optimized_features import OptimizedFeatureTests
from agent_code.Agent_039_compact_audit_ddqn_agent import features as agent039_features
from agent_code.Agent_039_compact_audit_ddqn_agent import symmetry as agent039_symmetry
from agent_code.Agent_040_optimized_compact_ddqn_agent import features
from agent_code.Agent_040_optimized_compact_ddqn_agent import symmetry
from agent_code.Agent_040_optimized_compact_ddqn_agent.replay import (
    CombatEscapeReplayBuffer,
)


class Agent040Tests(unittest.TestCase):
    def test_direct_features_match_agent039_compact_contract(self):
        arguments = (
            4, 2, 3, (0, 1, 4), (1, 1, 0),
            ((7, 7), (7, 6), (7, 7)),
        )
        for source_state in OptimizedFeatureTests()._states():
            state = copy.deepcopy(source_state)
            expected = agent039_features.state_to_features(state, *arguments)
            context = features.StateContext(state)
            actual = features.state_to_features(
                state, *arguments, context=context
            )
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.shape, (104,))
            self.assertEqual(actual.dtype, np.float32)

    def test_fast_symmetry_matches_agent039(self):
        vector = np.arange(104, dtype=np.float32)
        mask = np.asarray([True, False, True, False, True, False])
        for transform in symmetry.TRANSFORMS:
            np.testing.assert_array_equal(
                symmetry.transform_features(vector, transform),
                agent039_symmetry.transform_features(vector, transform),
            )
            np.testing.assert_array_equal(
                symmetry.transform_mask(mask, transform),
                agent039_symmetry.transform_mask(mask, transform),
            )

    def test_dense_replay_tag_indexes_remain_consistent(self):
        buffer = CombatEscapeReplayBuffer(5, seed=4)
        buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
        for index in range(12):
            buffer.transition_tag = (
                "combat_escape" if index % 3 == 0 else None
            )
            state = np.full(104, index, dtype=np.float32)
            buffer.add(state, index % 6, 0.0, state, False,
                       np.ones(6, dtype=bool))
        self.assertEqual(sum(map(len, buffer.indices_by_tag.values())), 5)
        batch = buffer.sample_torch(4, torch.device("cpu"))
        self.assertEqual(tuple(batch.states.shape), (4, 104))


if __name__ == "__main__":
    unittest.main()
