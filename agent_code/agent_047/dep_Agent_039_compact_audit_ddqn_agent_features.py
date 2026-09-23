"""Agent 039's compact audit representation.

This is Agent 038's 117-value representation with only the approved,
semantically verified compression:

* remove the scalar previous-action index;
* remove the constant centre cell of the 3x3 topology patch;
* replace six conditionally repeated armed-opponent values with one global
  armed-opponent flag;
* remove the six immediate-legality copies while retaining the authoritative
  action mask and the four response values in every opponent block.

The underlying feature calculations remain Agent 038-compatible so that the
candidate changes can be evaluated without changing unrelated behavior.
"""

import numpy as np

from . import dep_Agent_038_symmetric_population_ddqn_agent_features as _wide


ACTIONS = _wide.ACTIONS
AGENT_038_FEATURE_SIZE = _wide.FEATURE_SIZE
FEATURE_SIZE = 104
FEATURE_SCHEMA = "agent039-compact-audit-v1"

# New compact layout constants.  These are deliberately named rather than
# relying on scattered literal indices in the symmetry implementation.
PATCH_START = 31
PATCH_OFFSETS = tuple(
    (dx, dy)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if (dx, dy) != (0, 0)
)
SHORT_CYCLE_STARTS = (44, 50)
GLOBAL_ARMED_OPPONENT_INDEX = 63
OPPONENT_BLOCK_START = 64
OPPONENT_BLOCK_SIZE = 4
ROUTE_STARTS = (88, 92, 96, 100)


def any_armed_opponent(game_state):
    """Return the state-wide armed-opponent flag used by Agent 037/038."""
    return float(any(bool(other[2]) for other in game_state["others"]))


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0, action_history=(),
                      action_successes=(), position_history=()):
    if game_state is None:
        return None
    source = _wide.state_to_features(
        game_state, previous_action, recent_visits, steps_since_progress,
        action_history, action_successes, position_history,
    )
    # Keep the source contract explicit.  This catches accidental upstream
    # schema drift before it silently changes Agent 039's column mapping.
    if source.shape != (AGENT_038_FEATURE_SIZE,) or not np.isfinite(source).all():
        raise AssertionError("Agent 038 source feature contract changed")

    parts = [
        source[0:29],       # combat values; omit old index 29
        source[30:32],      # recent visits and stagnation bucket
        source[32:36],      # first four non-centre patch cells
        source[37:41],      # final four non-centre patch cells
        source[41:46],      # topology summaries
        source[46:65],      # short-cycle history
        np.asarray([any_armed_opponent(game_state)], dtype=np.float32),
    ]

    # Agent 038 blocks are:
    # legal, armed-opponent, contested, worst-time, worst-frontier, new-trap.
    # Keep the four response values; legality is represented by the existing
    # action mask and the old block's zero gating remains in these values.
    for action_index in range(len(ACTIONS)):
        start = 65 + 6 * action_index
        parts.append(source[start + 2:start + 6])

    parts.append(source[101:117])  # route suffix
    result = np.concatenate(parts).astype(np.float32, copy=False)
    if result.shape != (FEATURE_SIZE,) or not np.isfinite(result).all():
        raise AssertionError(
            f"Agent 039 feature size changed: expected {FEATURE_SIZE}, "
            f"got {result.shape}"
        )
    return result


__all__ = [
    "ACTIONS", "AGENT_038_FEATURE_SIZE", "FEATURE_SCHEMA", "FEATURE_SIZE",
    "GLOBAL_ARMED_OPPONENT_INDEX", "OPPONENT_BLOCK_SIZE",
    "OPPONENT_BLOCK_START", "PATCH_OFFSETS", "PATCH_START", "ROUTE_STARTS",
    "SHORT_CYCLE_STARTS", "any_armed_opponent", "state_to_features",
]
