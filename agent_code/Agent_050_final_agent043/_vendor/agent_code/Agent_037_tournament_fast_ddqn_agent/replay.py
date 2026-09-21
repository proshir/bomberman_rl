"""Agent 032's preallocated replay specialized to 101-value transitions."""

from agent_code.Agent_032_combat_ddqn_optimized_features_agent.replay import (
    CombatEscapeReplayBuffer as _OptimizedCombatEscapeReplayBuffer,
    ScenarioReplayBuffer,
)

from .features import FEATURE_SIZE


ESCAPE_TAG = "combat_escape"
ESCAPE_FRACTION = 0.20


class CombatEscapeReplayBuffer(_OptimizedCombatEscapeReplayBuffer):
    """Preallocated tagged replay whose default state width is exactly 101."""

    def __init__(self, capacity, seed=None, n_actions=6, state_dim=FEATURE_SIZE):
        super().__init__(
            capacity, seed=seed, n_actions=n_actions, state_dim=state_dim,
        )


__all__ = [
    "CombatEscapeReplayBuffer", "ESCAPE_FRACTION", "ESCAPE_TAG",
    "ScenarioReplayBuffer",
]
