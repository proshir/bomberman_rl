"""Agent 042 representation plus action-aligned novelty observations."""

import numpy as np

from agent_code.Agent_042_combat_progress_ddqn_agent import features as _base
from agent_code.Agent_042_combat_progress_ddqn_agent.features import (
    ACTIONS,
    MOVE_DELTAS,
    StateContext,
)


BASE_FEATURE_SIZE = _base.FEATURE_SIZE
NOVELTY_START = BASE_FEATURE_SIZE
NOVELTY_FEATURE_SIZE = len(ACTIONS)
FEATURE_SIZE = BASE_FEATURE_SIZE + NOVELTY_FEATURE_SIZE
FEATURE_SCHEMA = "agent043-agent042-novelty-v1"
AGENT_040_FEATURE_SIZE = _base.AGENT_040_FEATURE_SIZE
AGENT_041_FEATURE_SIZE = _base.AGENT_041_FEATURE_SIZE
AGENT_042_FEATURE_SIZE = BASE_FEATURE_SIZE


def _destination(position, action):
    if action == "WAIT":
        return position
    dx, dy = MOVE_DELTAS[action]
    return position[0] + dx, position[1] + dy


def action_novelty(game_state, position_history=(), context=None):
    """Return [0,1] novelty for each safe candidate destination.

    A destination absent from the rolling history is maximally novel. Bombing
    has no destination and is intentionally neutral; the safety mask remains
    the only hard action constraint.
    """
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
    base = _base.state_to_features(
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
    "AGENT_042_FEATURE_SIZE", "BASE_FEATURE_SIZE", "FEATURE_SCHEMA",
    "FEATURE_SIZE", "MOVE_DELTAS", "NOVELTY_FEATURE_SIZE", "NOVELTY_START",
    "StateContext", "action_novelty", "state_to_features",
]
