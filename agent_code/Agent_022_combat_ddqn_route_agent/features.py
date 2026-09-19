"""Route-aware combat features that preserve candidate movement information.

Sahand was here.

The old 46-input topology representation compressed target routes to a sign and
distance bucket.  Here each cardinal movement receives its own exact shortest
path cost and reachability flag for coins and crate-approach tiles.  This makes
the documented UP-versus-DOWN route witness observable to the Q-network.
"""

from collections import deque

import numpy as np

from agent_code.combat_fqi_agent.features import crate_approach_tiles
from agent_code.combat_fqi_agent.safety import ACTIONS, MOVE_DELTAS, action_is_legal
from agent_code.combat_fqi_history_antistag_agent.features import (
    state_to_features as history_features,
)


MOVEMENT_ACTIONS = tuple(ACTIONS[:4])
ROUTE_FEATURES_PER_ACTION = 2
ROUTE_FEATURES_PER_TARGET = len(MOVEMENT_ACTIONS) * ROUTE_FEATURES_PER_ACTION
BASE_PREFIX_SIZE = 14
BASE_TARGET_SIZE = 3
BASE_SUFFIX_START = BASE_PREFIX_SIZE + 2 * BASE_TARGET_SIZE
HISTORY_SIZE = 3
PRUNED_TOPOLOGY_SIZE = 9
FEATURE_SIZE = (
    BASE_PREFIX_SIZE + 2 * ROUTE_FEATURES_PER_TARGET +
    (29 - BASE_SUFFIX_START) + HISTORY_SIZE + PRUNED_TOPOLOGY_SIZE + 1
)


def _free_neighbours(field, position):
    """Yield cardinal free neighbours for the static route graph."""
    x, y = position
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        neighbour = x + dx, y + dy
        if (0 <= neighbour[0] < field.shape[0] and
                0 <= neighbour[1] < field.shape[1] and
                field[neighbour] == 0):
            yield neighbour


def _distance_map(field, targets):
    """Return exact shortest-path distances to the closest reachable target."""
    starts = {
        tuple(target) for target in targets
        if (0 <= target[0] < field.shape[0] and
            0 <= target[1] < field.shape[1] and field[tuple(target)] == 0)
    }
    distances = {target: 0 for target in starts}
    queue = deque(starts)
    while queue:
        position = queue.popleft()
        for neighbour in _free_neighbours(field, position):
            if neighbour not in distances:
                distances[neighbour] = distances[position] + 1
                queue.append(neighbour)
    return distances


def action_route_features(game_state, targets):
    """Describe route cost and reachability after each cardinal movement.

    The scalar cost is normalized by the number of board cells but is otherwise
    exact: distinct route lengths remain distinct.  An accompanying flag keeps
    an unreachable target separate from a zero-length route after a move.
    """
    field = game_state["field"]
    distances = _distance_map(field, targets)
    normalizer = float(field.shape[0] * field.shape[1])
    x, y = game_state["self"][3]
    values = []
    for action in MOVEMENT_ACTIONS:
        if not action_is_legal(game_state, action):
            values.extend((0.0, 0.0))
            continue
        dx, dy = MOVE_DELTAS[action]
        distance = distances.get((x + dx, y + dy))
        if distance is None:
            values.extend((0.0, 0.0))
        else:
            values.extend((float(distance) / normalizer, 1.0))
    return np.asarray(values, dtype=float)


def pruned_topology_features(game_state):
    """Keep raw local geometry while omitting five exact derived duplicates."""
    field = game_state["field"]
    x, y = map(int, game_state["self"][3])

    def tile(dx, dy):
        xx, yy = x + dx, y + dy
        if not (0 <= xx < field.shape[0] and 0 <= yy < field.shape[1]):
            return -1
        return int(field[xx, yy])

    # Keep all eight neighbouring cells.  The removed centre is always the
    # agent's free tile, while the removed counts/flags are exact functions of
    # the cardinal neighbours already present here.
    patch = [
        tile(dx, dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if not (dx == 0 and dy == 0)
    ]
    radius_two = [
        tile(dx, dy)
        for dx in range(-2, 3)
        for dy in range(-2, 3)
        if abs(dx) + abs(dy) <= 2
    ]
    return np.asarray(patch + [sum(value == 0 for value in radius_two)],
                      dtype=float)


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0):
    """Build the 52-input route-aware replacement for the 46-input vector."""
    history = history_features(
        game_state, previous_action, recent_visits, steps_since_progress
    )
    if history is None:
        return None
    # The shared history extractor returns 29 combat values followed by three
    # temporal values.  Remove only its coin/crate direction-and-bucket blocks.
    base = history[:29]
    temporal = history[29:]
    features = np.concatenate((
        base[:BASE_PREFIX_SIZE],
        action_route_features(game_state, game_state["coins"]),
        action_route_features(game_state, crate_approach_tiles(game_state["field"])),
        base[BASE_SUFFIX_START:],
        temporal,
        pruned_topology_features(game_state),
        np.asarray([
            max(0, 400 - int(game_state["step"])) / 400.0,
        ], dtype=float),
    ))
    if len(features) != FEATURE_SIZE:
        raise AssertionError(
            f"Route feature size changed: expected {FEATURE_SIZE}, got {len(features)}"
        )
    return features


__all__ = [
    "ACTIONS", "FEATURE_SIZE", "MOVEMENT_ACTIONS", "action_route_features",
    "pruned_topology_features", "state_to_features",
]
