"""Agent 038's preallocated replay with the Agent 039 state width."""

from .dep_Agent_032_combat_ddqn_optimized_features_agent_replay import (
    CombatEscapeReplayBuffer as _OptimizedCombatEscapeReplayBuffer,
    ScenarioReplayBuffer,
)

from .dep_Agent_039_compact_audit_ddqn_agent_features import FEATURE_SIZE


ESCAPE_TAG = "combat_escape"
ESCAPE_FRACTION = 0.10


class CombatEscapeReplayBuffer(_OptimizedCombatEscapeReplayBuffer):
    def __init__(self, capacity, seed=None, n_actions=6, state_dim=FEATURE_SIZE):
        super().__init__(
            capacity, seed=seed, n_actions=n_actions, state_dim=state_dim,
        )

    def set_context(self, tag, weights):
        ScenarioReplayBuffer.set_context(
            self,
            tag,
            {
                name: (1.0 - ESCAPE_FRACTION) * weight
                for name, weight in weights.items()
            } | {ESCAPE_TAG: ESCAPE_FRACTION},
        )


__all__ = [
    "CombatEscapeReplayBuffer", "ESCAPE_FRACTION", "ESCAPE_TAG",
    "ScenarioReplayBuffer",
]
