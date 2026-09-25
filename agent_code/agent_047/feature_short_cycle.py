"""Add explicit short-cycle history without changing target features."""

# Sahand was here.

import numpy as np

from .feature_topology import (
    state_to_features as topology_features,
)
from .feature_history import ACTIONS


SHORT_CYCLE_FEATURES = 19
FEATURE_SIZE = 46 + SHORT_CYCLE_FEATURES
OPPOSITE = {0: 2, 2: 0, 1: 3, 3: 1}


def _one_hot(action):
    result = np.zeros(len(ACTIONS), dtype=float)
    if action is not None and 0 <= int(action) < len(ACTIONS):
        result[int(action)] = 1.0
    return result


def short_cycle_features(action_history=(), action_successes=(),
                         position_history=()):
    """Encode recent actions and repeated movement patterns.

    The history is informational only.  It never vetoes an action.  Missing
    actions at round start remain all-zero one-hot vectors.
    """
    actions = list(action_history)[-8:]
    successes = list(action_successes)[-8:]
    positions = [tuple(position) for position in position_history][-8:]

    previous_two = actions[-2:]
    action_values = []
    for action in previous_two:
        action_values.extend(_one_hot(action))
    while len(action_values) < 2 * len(ACTIONS):
        action_values.extend(np.zeros(len(ACTIONS), dtype=float))

    success_values = [
        float(value) for value in successes[-2:]
    ]
    success_values = ([0.0] * (2 - len(success_values)) + success_values)

    reversal = 0.0
    if len(actions) >= 2:
        first, second = actions[-2:]
        reversal = float(first in OPPOSITE and OPPOSITE[first] == second)

    waits = 0
    for action in reversed(actions):
        if action != ACTIONS.index("WAIT"):
            break
        waits += 1

    two_cycle = 0.0
    if len(positions) >= 4:
        two_cycle = float(
            positions[-4] == positions[-2] and
            positions[-3] == positions[-1] and
            positions[-4] != positions[-3]
        )

    four_cycle = 0.0
    if len(positions) >= 8:
        four_cycle = float(positions[-8:-4] == positions[-4:])

    displacement = 0.0
    if len(positions) >= 2:
        start, end = positions[0], positions[-1]
        displacement = min(
            1.0, (abs(end[0] - start[0]) + abs(end[1] - start[1])) / 8.0
        )

    values = np.asarray(
        action_values + success_values + [
            reversal,
            min(1.0, waits / 8.0),
            two_cycle,
            four_cycle,
            displacement,
        ],
        dtype=float,
    )
    if len(values) != SHORT_CYCLE_FEATURES:
        raise AssertionError(
            f"Agent 025 short-cycle size changed: expected "
            f"{SHORT_CYCLE_FEATURES}, got {len(values)}"
        )
    return values


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0, action_history=(),
                      action_successes=(), position_history=()):
    base = topology_features(
        game_state, previous_action, recent_visits, steps_since_progress
    )
    if base is None:
        return None
    features = np.concatenate((
        base,
        short_cycle_features(
            action_history, action_successes, position_history
        ),
    ))
    if len(features) != FEATURE_SIZE:
        raise AssertionError(
            f"Agent 025 feature size changed: expected {FEATURE_SIZE}, "
            f"got {len(features)}"
        )
    return features


__all__ = [
    "ACTIONS", "FEATURE_SIZE", "SHORT_CYCLE_FEATURES",
    "short_cycle_features", "state_to_features",
]
