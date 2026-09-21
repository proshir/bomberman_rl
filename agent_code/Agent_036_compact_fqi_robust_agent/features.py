"""The 22-logical-entry, 28-numeric-input Agent 036 representation.

Routes use the current free-tile graph. Bombs and opponents block the first
move through action_is_legal; later route cells treat them as transient.
Crate targets use an immediate-at-target hypothetical bomb feasibility proxy.
Actual bombing is independently checked by the policy safety mask on arrival.
"""

from collections import deque

import numpy as np
import settings as s

from .safety import (ACTIONS, MOVE_DELTAS, action_is_legal, bomb_value,
                     can_survive_action, surviving_followup_actions)


FEATURE_SIZE = 28
HISTORY_LENGTH = 8
FEATURE_SCHEMA = "compact-routes-history-blast-range-v4-agent036"


def _distance_map(field, targets):
    distances = {tuple(tile): 0 for tile in targets if field[tuple(tile)] == 0}
    queue = deque(distances)
    while queue:
        x, y = queue.popleft()
        for dx, dy in (MOVE_DELTAS[action] for action in ACTIONS[:4]):
            tile = (x + dx, y + dy)
            if (0 <= tile[0] < field.shape[0] and
                    0 <= tile[1] < field.shape[1] and
                    field[tile] == 0 and tile not in distances):
                distances[tile] = distances[(x, y)] + 1
                queue.append(tile)
    return distances


def movement_routes(game_state, targets):
    """Return four costs followed by four explicit reachability indicators."""
    distances = _distance_map(game_state["field"], targets)
    x, y = game_state["self"][3]
    costs, reached = [], []
    for action in ACTIONS[:4]:
        dx, dy = MOVE_DELTAS[action]
        distance = distances.get((x + dx, y + dy)) if action_is_legal(game_state, action) else None
        costs.append(float(distance + 1) if distance is not None else 0.0)
        reached.append(float(distance is not None))
    return costs + reached


def candidate_crate_tiles(field):
    """Every free bomb tile whose ray can hit at least one current crate."""
    candidates = set()
    for x, y in zip(*(field == 1).nonzero()):
        for dx, dy in (MOVE_DELTAS[action] for action in ("UP", "RIGHT", "DOWN", "LEFT")):
            for distance in range(1, s.BOMB_POWER + 1):
                tile = (int(x + dx * distance), int(y + dy * distance))
                if not (0 <= tile[0] < field.shape[0] and
                        0 <= tile[1] < field.shape[1]) or field[tile] == -1:
                    break
                if field[tile] == 0:
                    candidates.add(tile)
    return candidates


def feasible_crate_tiles(game_state):
    """Use present-world, immediate-at-tile bomb feasibility as a route proxy."""
    field = game_state["field"]
    current = tuple(game_state["self"][3])
    occupied = {tuple(other[3]) for other in game_state["others"]}
    occupied.update(tuple(position) for position, _ in game_state["bombs"])
    # No route can reach a tile outside the current static free-tile component.
    reachable = _distance_map(field, (current,))
    result = set()
    for tile in sorted(candidate_crate_tiles(field).intersection(reachable)):
        if tile != current and tile in occupied:
            continue
        candidate = dict(game_state)
        name, score, _, _ = game_state["self"]
        candidate["self"] = (name, score, True, tile)
        if bomb_value(candidate)[0] > 0 and can_survive_action(candidate, "BOMB"):
            result.add(tile)
    return result


def progress_bucket(steps):
    if steps < 4:
        return 0
    if steps < 8:
        return 1
    if steps < 16:
        return 2
    if steps < 32:
        return 3
    return 4


def bomb_escape_count(game_state):
    """Count safe actions immediately after bomb placement.

    The count is derived from the original BOMB search.  This keeps existing
    bomb timers, active explosions, and opponent reachability on one timeline
    instead of constructing a partially advanced state with inconsistent
    timing.
    """
    if not action_is_legal(game_state, "BOMB"):
        return 0
    return len(surviving_followup_actions(game_state, "BOMB"))


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0):
    if game_state is None:
        return None
    previous = 6 if previous_action is None else int(previous_action)
    if not 0 <= previous <= 6:
        raise ValueError("previous_action must be an action index or START=6")
    available = bool(game_state["self"][2])
    escape_count = bomb_escape_count(game_state) if available else 0
    crates_hit = bomb_value(game_state)[0] if available and escape_count else 0
    one_hot = [float(index == previous) for index in range(7)]
    values = (
        movement_routes(game_state, game_state["coins"]) +
        movement_routes(game_state, feasible_crate_tiles(game_state)) +
        [float(available), float(crates_hit), float(escape_count)] +
        one_hot +
        [float(recent_visits), float(progress_bucket(steps_since_progress))]
    )
    features = np.asarray(values, dtype=np.float32)
    if features.shape != (FEATURE_SIZE,) or not np.isfinite(features).all():
        raise AssertionError("Invalid compact FQI feature vector")
    return features
