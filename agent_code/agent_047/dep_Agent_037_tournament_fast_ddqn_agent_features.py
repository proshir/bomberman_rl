"""Agent 029's exact 101 inputs with Agent 032's faster opponent window.

The 12 Agent 031 offensive inputs are intentionally not computed.  This keeps
the policy compatible with corrected Agent 030 checkpoints while reusing the
audited allocation-light opponent-envelope calculation from Agent 032.
"""

import numpy as np

from .dep_Agent_025_combat_ddqn_short_cycle_agent_features import (
    ACTIONS,
    FEATURE_SIZE as AGENT_027_FEATURE_SIZE,
    SHORT_CYCLE_FEATURES,
    short_cycle_features,
)
from .dep_Agent_032_combat_ddqn_optimized_features_agent_features import (
    OPPONENT_FEATURES_PER_ACTION,
    OPPONENT_WINDOW,
    opponent_window_features,
)
from .dep_combat_dqn_r_topology_agent_features import (
    state_to_features as _base_features,
)


FEATURE_SIZE = AGENT_027_FEATURE_SIZE + len(ACTIONS) * OPPONENT_FEATURES_PER_ACTION


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0, action_history=(),
                      action_successes=(), position_history=()):
    """Return Agent 029's feature contract without the Agent 031 suffix."""
    base = _base_features(
        game_state, previous_action, recent_visits, steps_since_progress,
    )
    if base is None:
        return None
    history = short_cycle_features(
        action_history, action_successes, position_history,
    )
    values = np.concatenate((base, history, opponent_window_features(game_state)))
    if values.shape != (FEATURE_SIZE,):
        raise AssertionError(
            f"Agent 037 feature size changed: expected {FEATURE_SIZE}, "
            f"got {values.shape}"
        )
    return values.astype(np.float32, copy=False)


__all__ = [
    "ACTIONS", "AGENT_027_FEATURE_SIZE", "FEATURE_SIZE",
    "OPPONENT_FEATURES_PER_ACTION", "OPPONENT_WINDOW", "SHORT_CYCLE_FEATURES",
    "opponent_window_features", "short_cycle_features", "state_to_features",
]
