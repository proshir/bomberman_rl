"""Agent 041's Agent 040 representation plus dynamic navigation inputs.

The policy still produces one Q value for every action.  Navigation is only
represented in the observation; it never filters or directly selects an
action.  The added blocks are action-aligned so the DDQN can learn when route
progress matters and when combat value should override it.
"""

import numpy as np

from . import dep_Agent_040_optimized_compact_ddqn_agent_features as _base
from .dep_Agent_040_optimized_compact_ddqn_agent_features import (
    ACTIONS,
    MOVE_DELTAS,
    StateContext,
)


BASE_FEATURE_SIZE = _base.FEATURE_SIZE
NAV_PROGRESS_START = BASE_FEATURE_SIZE
NAV_REACHABLE_START = NAV_PROGRESS_START + len(ACTIONS)
NAV_VISITS_START = NAV_REACHABLE_START + len(ACTIONS)
NAV_FEATURE_SIZE = 3 * len(ACTIONS)
FEATURE_SIZE = BASE_FEATURE_SIZE + NAV_FEATURE_SIZE
FEATURE_SCHEMA = "agent041-agent040-dynamic-navigation-v1"
AGENT_040_FEATURE_SIZE = BASE_FEATURE_SIZE


def _navigation_features(game_state, context, position_history):
    """Return signed route progress, route validity, and dynamic visits.

    ``StateContext.distance_map`` is a multi-source BFS from all currently
    collectable coins.  This makes the values exact for the current board,
    while the network remains responsible for choosing among all actions.
    """
    coins = tuple(tuple(coin) for coin in game_state["coins"])
    distances = context.distance_map(coins)
    position = tuple(game_state["self"][3])
    current_distance = distances.get(position)
    legal = set(context.legal_indices())

    history = [tuple(item) for item in position_history]
    if not history or history[-1] != position:
        history.append(position)

    progress = np.zeros(len(ACTIONS), dtype=np.float32)
    reachable = np.zeros(len(ACTIONS), dtype=np.float32)
    visits = np.zeros(len(ACTIONS), dtype=np.float32)

    for index, action in enumerate(ACTIONS):
        # Bombing has no movement destination.  Its combat value is already
        # represented by the retained bomb features and is left to the DDQN.
        if action == "BOMB":
            continue

        if action == "WAIT":
            destination = position
        else:
            dx, dy = MOVE_DELTAS[action]
            destination = (position[0] + dx, position[1] + dy)

        if index in legal:
            visits[index] = min(3, history.count(destination)) / 3.0

        if (index not in legal or current_distance is None or
                destination not in distances):
            # Keep no-target states neutral.  If a target exists but this
            # action cannot reach it, expose a negative signal and let the
            # reachability block disambiguate it from a neutral WAIT/BOMB.
            if coins and current_distance is not None and action != "WAIT":
                progress[index] = -1.0
            continue

        reachable[index] = 1.0
        progress[index] = float(current_distance - distances[destination])

    return np.concatenate((progress, reachable, visits)).astype(
        np.float32, copy=False
    )


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0, action_history=(),
                      action_successes=(), position_history=(), context=None):
    if game_state is None:
        return None
    if context is None:
        context = StateContext(game_state)
    base = _base.state_to_features(
        game_state, previous_action, recent_visits, steps_since_progress,
        action_history, action_successes, position_history, context=context,
    )
    if base.shape != (BASE_FEATURE_SIZE,) or not np.isfinite(base).all():
        raise AssertionError("Agent 040 feature contract changed")

    navigation = _navigation_features(game_state, context, position_history)
    result = np.concatenate((base, navigation)).astype(np.float32, copy=False)
    if result.shape != (FEATURE_SIZE,) or not np.isfinite(result).all():
        raise AssertionError(
            f"Agent 041 feature size changed: expected {FEATURE_SIZE}, "
            f"got {result.shape}"
        )
    return result


__all__ = [
    "ACTIONS", "AGENT_040_FEATURE_SIZE", "BASE_FEATURE_SIZE",
    "FEATURE_SCHEMA", "FEATURE_SIZE", "MOVE_DELTAS", "NAV_FEATURE_SIZE",
    "NAV_PROGRESS_START", "NAV_REACHABLE_START", "NAV_VISITS_START",
    "StateContext", "state_to_features",
]
