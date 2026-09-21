"""Compact combat features for the fitted-Q agent."""

# Sahand was here.

from collections import deque

import numpy as np

from .safety import (ACTIONS, MOVE_DELTAS, action_is_legal, bomb_value,
                     build_danger_schedule, escape_distance_after_bomb,
                     safe_action_indices)


def _direction(source, target):
    if target is None:
        return 0, 0
    return (int(np.sign(target[0] - source[0])),
            int(np.sign(target[1] - source[1])))


def distance_bucket(distance):
    """Compress path lengths without losing the important short distances."""
    if distance is None or distance <= 0:
        return 0
    if distance == 1:
        return 1
    if distance == 2:
        return 2
    if distance <= 4:
        return 3
    if distance <= 7:
        return 4
    return 5


def count_bucket(count):
    if count <= 0:
        return 0
    if count <= 2:
        return 1
    if count <= 5:
        return 2
    if count <= 15:
        return 3
    return 4


def danger_bucket(time_step):
    if not time_step:
        return 0
    if time_step == 1:
        return 1
    if time_step == 2:
        return 2
    return 3


def _free_neighbours(field, position):
    x, y = position
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        neighbour = x + dx, y + dy
        if (0 <= neighbour[0] < field.shape[0] and
                0 <= neighbour[1] < field.shape[1] and
                field[neighbour] == 0):
            yield neighbour


def nearest_target(field, start, targets):
    """Find the nearest target reachable through currently free board tiles."""
    targets = set(targets)
    if not targets:
        return None, None
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        position, distance = queue.popleft()
        if position in targets:
            return position, distance
        for neighbour in _free_neighbours(field, position):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))
    return None, None


def crate_approach_tiles(field):
    """Return free tiles from which a bomb could hit at least one crate."""
    targets = set()
    crate_positions = zip(*((field == 1).nonzero()))
    for x, y in crate_positions:
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            tile = int(x + dx), int(y + dy)
            if (0 <= tile[0] < field.shape[0] and
                    0 <= tile[1] < field.shape[1] and field[tile] == 0):
                targets.add(tile)
    return targets


def _nearest_opponent(position, others):
    if not others:
        return None, None
    locations = [tuple(other[3]) for other in others]
    target = min(locations,
                 key=lambda point: abs(point[0] - position[0]) +
                                   abs(point[1] - position[1]))
    return target, abs(target[0] - position[0]) + abs(target[1] - position[1])


def state_to_features(game_state):
    """Describe navigation, danger, bombing opportunities, and opponents."""
    if game_state is None:
        return None

    field = game_state["field"]
    position = tuple(game_state["self"][3])
    safe = set(safe_action_indices(game_state))
    blocked = tuple(int(not action_is_legal(game_state, action))
                    for action in ACTIONS[:4])
    safe_moves = tuple(int(index in safe) for index in range(4))

    danger = build_danger_schedule(game_state)
    local_positions = [position]
    for action in ACTIONS[:4]:
        dx, dy = MOVE_DELTAS[action]
        local_positions.append((position[0] + dx, position[1] + dy))
    local_danger = []
    for tile in local_positions:
        times = danger.get(tile, ())
        local_danger.append(danger_bucket(min(times) if times else 0))

    coin, coin_distance = nearest_target(field, position, game_state["coins"])
    crate, crate_distance = nearest_target(field, position,
                                           crate_approach_tiles(field))
    opponent, opponent_distance = _nearest_opponent(position, game_state["others"])
    coin_direction = _direction(position, coin)
    crate_direction = _direction(position, crate)
    opponent_direction = _direction(position, opponent)
    crates_hit, opponents_hit = bomb_value(game_state)

    return np.asarray(
        blocked + safe_moves + tuple(local_danger) +
        (int(game_state["self"][2]),) +
        coin_direction + (distance_bucket(coin_distance),) +
        crate_direction + (distance_bucket(crate_distance),) +
        opponent_direction + (distance_bucket(opponent_distance),) +
        (min(crates_hit, 4), min(opponents_hit, 3),
         distance_bucket(escape_distance_after_bomb(game_state)),
         count_bucket(int(np.count_nonzero(field == 1))),
         count_bucket(len(game_state["coins"])),
         min(len(game_state["others"]), 3)),
        dtype=float,
    )
