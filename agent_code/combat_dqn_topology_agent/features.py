"""The failed FQI local-topology block, reused for a neural ablation."""

# Sahand was here.

import numpy as np

from agent_code.combat_fqi_history_antistag_topology_agent.features import (
    local_topology_features,
)
from agent_code.combat_fqi_history_antistag_agent.features import (
    ACTIONS,
    state_to_features as history_features,
)


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0):
    """Append exactly the 14 topology values to the 32-feature control."""
    base = history_features(
        game_state, previous_action, recent_visits, steps_since_progress
    )
    if base is None:
        return None
    return np.concatenate((base, local_topology_features(game_state)))


__all__ = ["ACTIONS", "state_to_features"]
