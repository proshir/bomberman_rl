"""Numerical coin-navigation features for the DQN.

The representation describes the board; it does not prescribe an action.  In
particular, the nearest-coin direction is only one input among local walls,
coin density, position, and time.  The network still has to learn how to use
the information and choose among legal actions.
"""

from __future__ import annotations

from collections import deque

import numpy as np


FEATURE_SIZE = 21


def state_to_features(game_state: dict | None) -> np.ndarray | None:
    """Encode one game state as a fixed-size float32 vector."""
    if game_state is None:
        return None

    field = np.asarray(game_state['field'])
    x, y = game_state['self'][3]
    width, height = field.shape
    coins = {tuple(coin) for coin in game_state['coins']}

    blocked = [
        int(y - 1 < 0 or field[x, y - 1] != 0),
        int(x + 1 >= width or field[x + 1, y] != 0),
        int(y + 1 >= height or field[x, y + 1] != 0),
        int(x - 1 < 0 or field[x - 1, y] != 0),
    ]

    nearest = nearest_coin(field, (x, y), coins)
    if nearest is None:
        direction_x, direction_y, distance = 0.0, 0.0, 0.0
    else:
        (target_x, target_y), distance = nearest
        direction_x = float(np.sign(target_x - x))
        direction_y = float(np.sign(target_y - y))

    local_coins = [
        int((x, y - 1) in coins),
        int((x + 1, y) in coins),
        int((x, y + 1) in coins),
        int((x - 1, y) in coins),
        int((x, y) in coins),
    ]

    # Count coins in four broad regions.  These values help distinguish two
    # equally short paths without exposing a hand-written route.
    quadrant_counts = [0, 0, 0, 0]
    for coin_x, coin_y in coins:
        horizontal = int(coin_x >= x)
        vertical = int(coin_y >= y)
        quadrant_counts[vertical * 2 + horizontal] += 1

    step = float(game_state.get('step', 0))
    max_steps = 400.0
    features = blocked + [
        direction_x,
        direction_y,
        min(float(distance), 64.0) / 64.0,
        len(coins) / 50.0,
        x / max(width - 1, 1),
        y / max(height - 1, 1),
        min(step, max_steps) / max_steps,
    ]
    features.extend(local_coins)
    features.extend(min(count, 50) / 50.0 for count in quadrant_counts)
    # Remaining free-neighbour count is useful at corners and dead ends.
    features.append(float(4 - sum(blocked)) / 4.0)
    return np.asarray(features, dtype=np.float32)


def nearest_coin(field: np.ndarray, start: tuple[int, int], coins: set[tuple[int, int]]):
    """Return the closest reachable coin and its maze distance."""
    if not coins:
        return None
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        position, distance = queue.popleft()
        if position in coins:
            return position, distance
        x, y = position
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            neighbour = (x + dx, y + dy)
            nx, ny = neighbour
            if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                    and field[nx, ny] == 0 and neighbour not in visited):
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))
    return None
