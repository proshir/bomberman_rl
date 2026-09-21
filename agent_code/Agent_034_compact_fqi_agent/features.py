"""The 22-logical-entry, 28-numeric-input compact FQI representation.

Routes use the current free-tile graph. Bombs and opponents block the first
move through action_is_legal; later route cells treat them as transient.
Crate targets use an immediate-at-target hypothetical bomb feasibility proxy.
Actual bombing is independently checked by the policy safety mask on arrival.
"""

from collections import deque

import numpy as np
import settings as s

from agent_code.combat_fqi_agent.features import crate_approach_tiles
from .safety import (ACTIONS, MOVE_DELTAS, action_is_legal, bomb_value,
                     can_survive_action)


FEATURE_SIZE = 28
HISTORY_LENGTH = 8
FEATURE_SCHEMA = "compact-routes-history-v2"


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


def feasible_crate_tiles(game_state):
    """Assess bombing immediately at a candidate tile, before future travel.

    This is a planning proxy. The current bombs/danger and opponent positions
    are retained; only our location and future bomb availability are changed.
    """
    field = game_state["field"]
    current = tuple(game_state["self"][3])
    result = set()
    for tile in sorted(crate_approach_tiles(field)):
        tile = tuple(tile)
        if tile != current and any(tile == tuple(other[3]) for other in game_state["others"]):
            continue
        if tile != current and any(tile == tuple(bomb[0]) for bomb in game_state["bombs"]):
            continue
        candidate = dict(game_state)
        name, score, _, _ = game_state["self"]
        candidate["self"] = (name, score, True, tile)
        # crate_approach_tiles can include cells whose blast is blocked by
        # stone walls; verify positive actual yield before the safety search.
        if bomb_value(candidate)[0] and can_survive_action(candidate, "BOMB"):
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
    """Count distinct safe first moves following a hypothetical bomb."""
    if not action_is_legal(game_state, "BOMB") or not can_survive_action(game_state, "BOMB"):
        return 0
    origin = tuple(game_state["self"][3])
    bomb_state = dict(game_state)
    bomb_state["bombs"] = list(game_state["bombs"]) + [(origin, s.BOMB_TIMER - 1)]
    bomb_state["self"] = (game_state["self"][0], game_state["self"][1], False, origin)
    return sum(can_survive_action(bomb_state, action) for action in ACTIONS[:5])


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0):
    if game_state is None:
        return None
    previous = 6 if previous_action is None else int(previous_action)
    if not 0 <= previous <= 6:
        raise ValueError("previous_action must be an action index or START=6")
    available = bool(game_state["self"][2])
    crates_hit = bomb_value(game_state)[0] if available else 0
    escape_count = bomb_escape_count(game_state) if available else 0
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
