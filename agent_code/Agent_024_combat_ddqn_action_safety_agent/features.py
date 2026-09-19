"""Add action-conditioned safety and bomb-consequence features to Agent 021."""

# Sahand was here.

from collections import deque

import numpy as np
import settings as s

from agent_code.combat_dqn_r_topology_agent.features import (
    state_to_features as topology_features,
)
from agent_code.combat_fqi_agent.safety import (
    ACTIONS,
    MOVE_DELTAS,
    _search_after_action,
    action_is_legal,
    blast_tiles,
    bomb_value,
    build_danger_schedule,
)


ACTION_CONSEQUENCE_FEATURES = 10
FEATURE_SIZE = 46 + len(ACTIONS) * ACTION_CONSEQUENCE_FEATURES


def _destination(game_state, action):
    x, y = game_state["self"][3]
    dx, dy = MOVE_DELTAS[action]
    return int(x + dx), int(y + dy)


def _safe_region_size(game_state, action, danger):
    """Count free tiles outside the predicted danger schedule.

    This is deliberately a conservative static summary.  The exact
    time-expanded survival result remains in the safety mask and the survival
    time feature; this value gives the network a graded region-size signal.
    """
    start = _destination(game_state, action)
    field = game_state["field"]
    if not (0 <= start[0] < field.shape[0] and
            0 <= start[1] < field.shape[1] and field[start] == 0):
        return 0
    if danger.get(start):
        return 0
    queue = deque([start])
    visited = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            tile = x + dx, y + dy
            if not (0 <= tile[0] < field.shape[0] and
                    0 <= tile[1] < field.shape[1]):
                continue
            if field[tile] != 0 or tile in visited or danger.get(tile):
                continue
            visited.add(tile)
            queue.append(tile)
    return len(visited)


def _escape_exit_count(game_state, action, danger):
    """Count immediate free exits from the post-action destination."""
    origin = _destination(game_state, action)
    field = game_state["field"]
    exits = 0
    for move in ACTIONS[:4]:
        dx, dy = MOVE_DELTAS[move]
        tile = origin[0] + dx, origin[1] + dy
        if (0 <= tile[0] < field.shape[0] and
                0 <= tile[1] < field.shape[1] and field[tile] == 0 and
                not danger.get(tile)):
            exits += 1
    return exits


def action_consequence_features(game_state):
    """Return ten normalized, action-specific safety/value values per action.

    The feature order for every action is:
    legal, safe, earliest danger, survived horizon, safe region size,
    escape margin, immediate exits, crate yield, opponent yield, useful bomb.
    """
    field = game_state["field"]
    board_area = float(field.shape[0] * field.shape[1])
    current_position = tuple(game_state["self"][3])
    values = []
    for action in ACTIONS:
        legal = action_is_legal(game_state, action)
        extra_bomb = action == "BOMB"
        danger = build_danger_schedule(game_state, extra_bomb=extra_bomb)
        destination = _destination(game_state, action)
        earliest = min(danger.get(destination, ()), default=0)
        survived, max_time, escaped_distance = _search_after_action(
            game_state, action
        ) if legal else (False, 0, None)
        safe_region = _safe_region_size(game_state, action, danger)
        exits = _escape_exit_count(game_state, action, danger)
        crates = opponents = useful = 0
        escape_margin = 0.0
        if action == "BOMB":
            crates, opponents = bomb_value(game_state)
            useful = int(bool(crates or opponents))
            if escaped_distance is not None:
                escape_margin = max(
                    0.0, float(s.BOMB_TIMER - escaped_distance)
                ) / max(1.0, float(s.BOMB_TIMER))
        values.extend((
            float(legal),
            float(survived),
            min(1.0, earliest / 5.0),
            min(1.0, max_time / max(1.0, float(s.MAX_STEPS))),
            min(1.0, safe_region / board_area),
            escape_margin,
            min(1.0, exits / 4.0),
            min(1.0, crates / 4.0),
            min(1.0, opponents / 3.0),
            float(useful),
        ))
    return np.asarray(values, dtype=float)


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0):
    base = topology_features(
        game_state, previous_action, recent_visits, steps_since_progress
    )
    if base is None:
        return None
    features = np.concatenate((base, action_consequence_features(game_state)))
    if len(features) != FEATURE_SIZE:
        raise AssertionError(
            f"Agent 024 feature size changed: expected {FEATURE_SIZE}, "
            f"got {len(features)}"
        )
    return features


__all__ = [
    "ACTIONS", "ACTION_CONSEQUENCE_FEATURES", "FEATURE_SIZE",
    "action_consequence_features", "state_to_features",
]
