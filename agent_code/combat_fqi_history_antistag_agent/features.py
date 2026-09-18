"""Combat features augmented with short movement and progress history."""

# Sahand was here.

import numpy as np

from agent_code.combat_fqi_agent.features import state_to_features as combat_features
from .safety import ACTIONS


HISTORY_LENGTH = 8


def stagnation_bucket(steps):
    """Compress time without progress into ranges useful to small trees."""
    if steps <= 2:
        return 0
    if steps <= 4:
        return 1
    if steps <= 8:
        return 2
    if steps <= 16:
        return 3
    if steps <= 32:
        return 4
    return 5


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0):
    """Add learned anti-loop context to the original combat description.

    No action is prescribed here.  The tree can learn whether the last move,
    repeated visits, or a long period without progress should change a Q-value.
    """
    base = combat_features(game_state)
    if base is None:
        return None
    if previous_action is None:
        previous_action = ACTIONS.index("WAIT")
    history = np.asarray((int(previous_action), min(int(recent_visits), 3),
                          stagnation_bucket(int(steps_since_progress))),
                         dtype=float)
    return np.concatenate((base, history))
