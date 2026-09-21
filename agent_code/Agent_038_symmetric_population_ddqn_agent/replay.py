"""Agent 032's preallocated replay specialized to 117-value transitions."""

from agent_code.Agent_032_combat_ddqn_optimized_features_agent.replay import (
    CombatEscapeReplayBuffer as _OptimizedCombatEscapeReplayBuffer,
    ScenarioReplayBuffer,
)

from .features import FEATURE_SIZE


ESCAPE_TAG = "combat_escape"
ESCAPE_FRACTION = 0.10


class CombatEscapeReplayBuffer(_OptimizedCombatEscapeReplayBuffer):
    """Preallocated tagged replay whose default state width is exactly 117."""

    def __init__(self, capacity, seed=None, n_actions=6, state_dim=FEATURE_SIZE):
        super().__init__(
            capacity, seed=seed, n_actions=n_actions, state_dim=state_dim,
        )

    def set_context(self, tag, weights):
        """Cap escape-sequence sampling at Agent 038's registered 10%."""
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
