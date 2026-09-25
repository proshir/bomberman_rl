"""Navigation, combat-progress, and novelty inputs for Agent 047."""

import numpy as np

from .feature_compact import (
    ACTIONS,
    FEATURE_SIZE as COMPACT_FEATURE_SIZE,
    MOVE_DELTAS,
    StateContext,
    bomb_value,
    state_to_features as compact_state_to_features,
)


NAV_PROGRESS_START = COMPACT_FEATURE_SIZE
NAV_REACHABLE_START = NAV_PROGRESS_START + len(ACTIONS)
NAV_VISITS_START = NAV_REACHABLE_START + len(ACTIONS)
NAV_FEATURE_SIZE = 3 * len(ACTIONS)
COMBAT_PROGRESS_START = COMPACT_FEATURE_SIZE + NAV_FEATURE_SIZE
COMBAT_REACHABLE_START = COMBAT_PROGRESS_START + len(ACTIONS)
COMBAT_PRESSURE_START = COMBAT_REACHABLE_START + len(ACTIONS)
COMBAT_FEATURE_SIZE = 3 * len(ACTIONS)
COMBAT_PROGRESS_FEATURE_SIZE = (
    COMPACT_FEATURE_SIZE + NAV_FEATURE_SIZE + COMBAT_FEATURE_SIZE
)
BASE_FEATURE_SIZE = COMBAT_PROGRESS_FEATURE_SIZE
NOVELTY_START = COMBAT_PROGRESS_FEATURE_SIZE
NOVELTY_FEATURE_SIZE = len(ACTIONS)
FEATURE_SIZE = NOVELTY_START + NOVELTY_FEATURE_SIZE
FEATURE_SCHEMA = "agent043-agent042-novelty-v1"
AGENT_040_FEATURE_SIZE = COMPACT_FEATURE_SIZE
AGENT_041_FEATURE_SIZE = COMPACT_FEATURE_SIZE + NAV_FEATURE_SIZE
AGENT_042_FEATURE_SIZE = COMBAT_PROGRESS_FEATURE_SIZE


def _destination(position, action):
    if action == "WAIT":
        return position
    dx, dy = MOVE_DELTAS[action]
    return position[0] + dx, position[1] + dy


def _navigation_features(game_state, context, position_history):
    """Return signed route progress, route validity, and dynamic visits."""
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
        if action == "BOMB":
            continue
        destination = _destination(position, action)
        if index in legal:
            visits[index] = min(3, history.count(destination)) / 3.0

        if (index not in legal or current_distance is None or
                destination not in distances):
            if coins and current_distance is not None and action != "WAIT":
                progress[index] = -1.0
            continue

        reachable[index] = 1.0
        progress[index] = float(current_distance - distances[destination])

    return np.concatenate((progress, reachable, visits)).astype(
        np.float32, copy=False
    )


def _combat_features(game_state, context):
    """Create bounded, action-aligned opponent approach signals."""
    opponents = tuple(tuple(other[3]) for other in game_state["others"])
    result = np.zeros(COMBAT_FEATURE_SIZE, dtype=np.float32)
    if not opponents:
        return result

    distances = context.distance_map(opponents)
    position = tuple(game_state["self"][3])
    current_distance = distances.get(position)
    legal = set(context.legal_indices())
    _, opponents_hit = bomb_value(game_state)

    for index, action in enumerate(ACTIONS):
        if index not in legal:
            continue
        if action == "BOMB":
            result[2 * len(ACTIONS) + index] = min(
                1.0, float(opponents_hit) / 3.0
            )
            continue

        destination = _destination(position, action)
        destination_distance = distances.get(destination)
        if destination_distance is None:
            continue
        result[len(ACTIONS) + index] = 1.0
        if current_distance is not None:
            result[index] = np.clip(
                (current_distance - destination_distance) / 4.0,
                -1.0, 1.0,
            )
        result[2 * len(ACTIONS) + index] = 1.0 / (
            1.0 + float(destination_distance)
        )
    return result


def _combat_progress_features(game_state, previous_action=None,
                              recent_visits=0, steps_since_progress=0,
                              action_history=(), action_successes=(),
                              position_history=(), context=None):
    if game_state is None:
        return None
    if context is None:
        context = StateContext(game_state)
    base = compact_state_to_features(
        game_state, previous_action, recent_visits, steps_since_progress,
        action_history, action_successes, position_history, context=context,
    )
    if (base.shape != (COMPACT_FEATURE_SIZE,) or
            not np.isfinite(base).all()):
        raise AssertionError("Agent 040 feature contract changed")
    navigation = _navigation_features(game_state, context, position_history)
    combat = _combat_features(game_state, context)
    result = np.concatenate((base, navigation, combat)).astype(
        np.float32, copy=False
    )
    if (result.shape != (COMBAT_PROGRESS_FEATURE_SIZE,) or
            not np.isfinite(result).all()):
        raise AssertionError(
            f"Agent 042 feature size changed: expected "
            f"{COMBAT_PROGRESS_FEATURE_SIZE}, got {result.shape}"
        )
    return result


def action_novelty(game_state, position_history=(), context=None):
    """Return [0,1] novelty for each safe candidate destination."""
    if game_state is None:
        return np.zeros(len(ACTIONS), dtype=np.float32)
    if context is None:
        context = StateContext(game_state)
    history = tuple(tuple(position) for position in position_history)
    legal = set(context.candidate_indices())
    position = tuple(game_state["self"][3])
    result = np.zeros(len(ACTIONS), dtype=np.float32)
    for index, action in enumerate(ACTIONS):
        if index not in legal or action == "BOMB":
            continue
        destination = _destination(position, action)
        result[index] = 1.0 / (1.0 + history.count(destination))
    return result


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0, action_history=(),
                      action_successes=(), position_history=(), context=None):
    if game_state is None:
        return None
    if context is None:
        context = StateContext(game_state)
    base = _combat_progress_features(
        game_state, previous_action, recent_visits, steps_since_progress,
        action_history, action_successes, position_history, context=context,
    )
    novelty = action_novelty(game_state, position_history, context=context)
    result = np.concatenate((base, novelty)).astype(np.float32, copy=False)
    if result.shape != (FEATURE_SIZE,) or not np.isfinite(result).all():
        raise AssertionError(
            f"Agent 043 feature size changed: expected {FEATURE_SIZE}, "
            f"got {result.shape}"
        )
    return result


__all__ = [
    "ACTIONS", "AGENT_040_FEATURE_SIZE", "AGENT_041_FEATURE_SIZE",
    "AGENT_042_FEATURE_SIZE", "BASE_FEATURE_SIZE", "COMBAT_FEATURE_SIZE",
    "COMBAT_PRESSURE_START", "COMBAT_PROGRESS_START",
    "COMBAT_REACHABLE_START", "COMPACT_FEATURE_SIZE", "FEATURE_SCHEMA",
    "MOVE_DELTAS", "NAV_FEATURE_SIZE", "NAV_PROGRESS_START",
    "NAV_REACHABLE_START", "NAV_VISITS_START", "NOVELTY_FEATURE_SIZE",
    "NOVELTY_START", "StateContext", "action_novelty", "state_to_features",
]
