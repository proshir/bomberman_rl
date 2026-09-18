"""Base coin-navigation features plus small movement-history inputs."""

from collections import deque
import os

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
USE_STAGNATION = os.environ.get('TREE_USE_STAGNATION', '0') == '1'
USE_TIME = os.environ.get('TREE_USE_TIME', '0') == '1'
MAX_STEPS = int(os.environ.get('TREE_MAX_STEPS', '400'))


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


def state_to_features(game_state, previous_action, recent_visits,
                      steps_since_coin=0, remaining_steps=None):
    """Encode navigation state plus optional stagnation and time signals."""
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
    result = blocked + (dx, dy, distance_bucket(distance),
                        remaining_coins_bucket(len(game_state['coins'])),
                        previous_action, min(recent_visits, 3))
    if USE_STAGNATION:
        result += (stagnation_bucket(steps_since_coin),)
    if USE_TIME:
        if remaining_steps is None:
            remaining_steps = max(0, MAX_STEPS - int(game_state['step']))
        result += (remaining_steps_bucket(remaining_steps),)
    return result


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


def stagnation_bucket(steps):
    """Bucket time since the last coin so trees can learn to break cycles."""
    if steps <= 2:
        return 0
    if steps <= 4:
        return 1
    if steps <= 8:
        return 2
    if steps <= 16:
        return 3
    if steps <= 32:
        return 4
    return 5


def remaining_steps_bucket(steps):
    """Bucket the remaining horizon without exposing an exact clock."""
    if steps <= 25:
        return 0
    if steps <= 50:
        return 1
    if steps <= 100:
        return 2
    if steps <= 200:
        return 3
    return 4
