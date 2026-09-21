"""Agent 037's 101 inputs plus Agent 036's 16 route inputs.

The prefix is deliberately bit-for-bit compatible with Agent 037. The suffix
contains four coin-route costs, four coin-route reachability flags, four
feasible-crate-route costs, and four crate-route reachability flags. This wide
contract supports a later measured pruning pass without guessing which
features matter before population training.
"""

import numpy as np

from agent_code.Agent_036_compact_fqi_robust_agent.features import (
    feasible_crate_tiles,
    movement_routes,
)
from agent_code.Agent_037_tournament_fast_ddqn_agent import features as _agent037


ACTIONS = _agent037.ACTIONS
AGENT_037_FEATURE_SIZE = _agent037.FEATURE_SIZE
ROUTE_FEATURE_SIZE = 16
FEATURE_SIZE = AGENT_037_FEATURE_SIZE + ROUTE_FEATURE_SIZE
FEATURE_SCHEMA = "agent037-plus-agent036-routes-v1"


def route_features(game_state):
    """Return Agent 036's coin and feasible-crate route blocks."""
    values = (
        movement_routes(game_state, game_state["coins"])
        + movement_routes(game_state, feasible_crate_tiles(game_state))
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (ROUTE_FEATURE_SIZE,) or not np.isfinite(result).all():
        raise AssertionError("Invalid Agent 038 route feature suffix")
    return result


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0, action_history=(),
                      action_successes=(), position_history=()):
    if game_state is None:
        return None
    prefix = _agent037.state_to_features(
        game_state, previous_action, recent_visits, steps_since_progress,
        action_history, action_successes, position_history,
    )
    values = np.concatenate((prefix, route_features(game_state)))
    if values.shape != (FEATURE_SIZE,) or not np.isfinite(values).all():
        raise AssertionError(
            f"Agent 038 feature size changed: expected {FEATURE_SIZE}, "
            f"got {values.shape}"
        )
    return values.astype(np.float32, copy=False)


__all__ = [
    "ACTIONS", "AGENT_037_FEATURE_SIZE", "FEATURE_SCHEMA", "FEATURE_SIZE",
    "ROUTE_FEATURE_SIZE", "feasible_crate_tiles", "movement_routes",
    "route_features", "state_to_features",
]
