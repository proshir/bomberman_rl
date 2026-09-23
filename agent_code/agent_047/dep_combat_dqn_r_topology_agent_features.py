"""The repaired DQN's 32 features plus the audited 14-value topology block."""

# Sahand was here.

import numpy as np

from .dep_combat_fqi_history_antistag_agent_features import (
    ACTIONS,
    state_to_features as history_features,
)
from .dep_combat_fqi_history_antistag_topology_agent_features import (
    local_topology_features,
)


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0):
    """Append local geometry without changing the repaired DQN algorithm."""
    base = history_features(
        game_state, previous_action, recent_visits, steps_since_progress
    )
    if base is None:
        return None
    return np.concatenate((base, local_topology_features(game_state)))


__all__ = ["ACTIONS", "state_to_features"]
