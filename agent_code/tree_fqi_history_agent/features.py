"""Base coin-navigation features plus small movement-history inputs."""

from collections import deque

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']


def legal_action_indices(game_state):
    field = game_state['field']
    x, y = game_state['self'][3]
    neighbours = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]
    legal = []
    for index, (nx, ny) in enumerate(neighbours):
        if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and
                field[nx, ny] == 0):
            legal.append(index)
    legal.append(ACTIONS.index('WAIT'))
    return legal


def state_to_features(game_state, previous_action, recent_visits):
    """Encode the current state and two facts about the preceding movement."""
    if game_state is None:
        return None
    field = game_state['field']
    x, y = game_state['self'][3]
    blocked = (
        int(y - 1 < 0 or field[x, y - 1] != 0),
        int(x + 1 >= field.shape[0] or field[x + 1, y] != 0),
        int(y + 1 >= field.shape[1] or field[x, y + 1] != 0),
        int(x - 1 < 0 or field[x - 1, y] != 0),
    )
    nearest = nearest_coin(field, (x, y), game_state['coins'])
    dx, dy, distance = 0, 0, 0
    if nearest is not None:
        target, distance = nearest
        dx = int(np.sign(target[0] - x))
        dy = int(np.sign(target[1] - y))
    return blocked + (dx, dy, distance_bucket(distance),
                      remaining_coins_bucket(len(game_state['coins'])),
                      previous_action, min(recent_visits, 3))


def nearest_coin(field, position, coins):
    coins = set(coins)
    if not coins:
        return None
    queue = deque([(position, 0)])
    visited = {position}
    while queue:
        current, distance = queue.popleft()
        if current in coins:
            return current, distance
        x, y = current
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            neighbour = (x + dx, y + dy)
            if (0 <= neighbour[0] < field.shape[0] and
                    0 <= neighbour[1] < field.shape[1] and
                    field[neighbour] == 0 and neighbour not in visited):
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))
    return None


def distance_bucket(distance):
    if distance == 0:
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


def remaining_coins_bucket(count):
    if count == 0:
        return 0
    if count <= 5:
        return 1
    if count <= 15:
        return 2
    if count <= 30:
        return 3
    return 4
