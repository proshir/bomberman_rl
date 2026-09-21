"""Agent 040 replay implementation with Agent 042 state width."""

from agent_code.Agent_040_optimized_compact_ddqn_agent.replay import (
    CombatEscapeReplayBuffer as _CombatEscapeReplayBuffer,
    ESCAPE_FRACTION,
    ESCAPE_TAG,
    ScenarioReplayBuffer as _ScenarioReplayBuffer,
)

from .features import FEATURE_SIZE


class ScenarioReplayBuffer(_ScenarioReplayBuffer):
    def __init__(self, capacity, seed=None, n_actions=6,
                 state_dim=FEATURE_SIZE):
        super().__init__(
            capacity, seed=seed, n_actions=n_actions, state_dim=state_dim,
        )


class CombatEscapeReplayBuffer(_CombatEscapeReplayBuffer):
    def __init__(self, capacity, seed=None, n_actions=6,
                 state_dim=FEATURE_SIZE):
        super().__init__(
            capacity, seed=seed, n_actions=n_actions, state_dim=state_dim,
        )


__all__ = [
    "CombatEscapeReplayBuffer", "ESCAPE_FRACTION", "ESCAPE_TAG",
    "ScenarioReplayBuffer",
]
