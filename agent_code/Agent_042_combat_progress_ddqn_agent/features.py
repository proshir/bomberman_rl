"""Agent 040 features plus action-aligned combat progress inputs.

Agent 040 already exposed a coarse nearest-opponent direction and distance.
Agent 042 adds three six-action blocks: signed progress toward the nearest
opponent, whether the action's destination is connected to an opponent, and
local combat pressure (or bomb hit value for BOMB).  These are observations,
not an action override; the DDQN still scores every action and the existing
safety candidate mask remains authoritative.
"""

import numpy as np

from agent_code.Agent_040_optimized_compact_ddqn_agent import features as _base
from agent_code.Agent_040_optimized_compact_ddqn_agent.features import (
    ACTIONS,
    MOVE_DELTAS,
    StateContext,
)
from agent_code.Agent_041_dynamic_nav_ddqn_agent.features import (
    _navigation_features,
)
from agent_code.combat_fqi_agent.features import bomb_value


BASE_FEATURE_SIZE = _base.FEATURE_SIZE
NAV_PROGRESS_START = BASE_FEATURE_SIZE
NAV_REACHABLE_START = NAV_PROGRESS_START + len(ACTIONS)
NAV_VISITS_START = NAV_REACHABLE_START + len(ACTIONS)
NAV_FEATURE_SIZE = 3 * len(ACTIONS)
COMBAT_PROGRESS_START = BASE_FEATURE_SIZE + NAV_FEATURE_SIZE
COMBAT_REACHABLE_START = COMBAT_PROGRESS_START + len(ACTIONS)
COMBAT_PRESSURE_START = COMBAT_REACHABLE_START + len(ACTIONS)
COMBAT_FEATURE_SIZE = 3 * len(ACTIONS)
FEATURE_SIZE = BASE_FEATURE_SIZE + NAV_FEATURE_SIZE + COMBAT_FEATURE_SIZE
FEATURE_SCHEMA = "agent042-agent040-combat-progress-v1"
AGENT_040_FEATURE_SIZE = BASE_FEATURE_SIZE
AGENT_041_FEATURE_SIZE = BASE_FEATURE_SIZE + 3 * len(ACTIONS)


def _destination(position, action):
    if action == "WAIT":
        return position
    dx, dy = MOVE_DELTAS[action]
    return position[0] + dx, position[1] + dy


def _combat_features(game_state, context):
    """Create bounded, action-aligned opponent approach signals.

    A zero vector is deliberately returned when there are no opponents.  The
    solo representation therefore stays semantically neutral and gains no
    hidden target or reward signal.
    """
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
            # BOMB has no destination; its action-aligned combat value is the
            # number of opponents currently in the blast, normalized to [0,1].
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
            # Four cells is the largest meaningful local change in the compact
            # board representation; clipping keeps this block well scaled.
            result[index] = np.clip(
                (current_distance - destination_distance) / 4.0,
                -1.0, 1.0,
            )
        result[2 * len(ACTIONS) + index] = 1.0 / (
            1.0 + float(destination_distance)
        )
    return result


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
    combat = _combat_features(game_state, context)
    result = np.concatenate((base, navigation, combat)).astype(
        np.float32, copy=False
    )
    if result.shape != (FEATURE_SIZE,) or not np.isfinite(result).all():
        raise AssertionError(
            f"Agent 042 feature size changed: expected {FEATURE_SIZE}, "
            f"got {result.shape}"
        )
    return result


__all__ = [
    "ACTIONS", "AGENT_040_FEATURE_SIZE", "AGENT_041_FEATURE_SIZE",
    "BASE_FEATURE_SIZE", "COMBAT_FEATURE_SIZE", "COMBAT_PRESSURE_START",
    "COMBAT_PROGRESS_START", "COMBAT_REACHABLE_START", "FEATURE_SCHEMA",
    "FEATURE_SIZE", "MOVE_DELTAS", "NAV_FEATURE_SIZE", "NAV_PROGRESS_START",
    "NAV_REACHABLE_START", "NAV_VISITS_START", "StateContext",
    "state_to_features",
]
