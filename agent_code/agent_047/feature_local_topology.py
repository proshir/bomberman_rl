"""History-aware combat features with compact local board topology."""

# Sahand was here.

import numpy as np

from .feature_history import (
    ACTIONS,
    state_to_features as history_features,
)


def _tile_value(field, x, y):
    """Return a local board value, treating outside the board as a wall."""
    if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
        return -1
    return int(field[x, y])


def local_topology_features(game_state):
    """Describe immediate geometry around the agent.

    The 3x3 patch preserves nearby wall/crate arrangement.  The aggregate
    values make common structural cases easy for shallow trees to separate:
    local openness, dead ends, and straight corridors.  Dynamic bomb and
    explosion information remains handled by the shared safety module and
    combat features.
    """
    field = game_state["field"]
    x, y = map(int, game_state["self"][3])

    patch = [
        _tile_value(field, x + dx, y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    ]
    neighbours = [
        _tile_value(field, x - 1, y),
        _tile_value(field, x + 1, y),
        _tile_value(field, x, y - 1),
        _tile_value(field, x, y + 1),
    ]
    free_neighbours = sum(tile == 0 for tile in neighbours)
    crate_neighbours = sum(tile == 1 for tile in neighbours)

    radius_two = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if abs(dx) + abs(dy) <= 2:
                radius_two.append(_tile_value(field, x + dx, y + dy))
    free_radius_two = sum(tile == 0 for tile in radius_two)

    dead_end = int(free_neighbours <= 1)
    straight_corridor = int(
        free_neighbours == 2 and
        ((_tile_value(field, x - 1, y) == 0 and
          _tile_value(field, x + 1, y) == 0) or
         (_tile_value(field, x, y - 1) == 0 and
          _tile_value(field, x, y + 1) == 0))
    )
    return np.asarray(
        patch + [free_neighbours, crate_neighbours, free_radius_two,
                 dead_end, straight_corridor],
        dtype=float,
    )


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0):
    """Append topology context without changing the audited base features."""
    base = history_features(
        game_state, previous_action, recent_visits, steps_since_progress
    )
    if base is None:
        return None
    return np.concatenate((base, local_topology_features(game_state)))
