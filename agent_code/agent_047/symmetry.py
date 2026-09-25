"""D4 board and action transforms for Agent 047's 146 inputs."""

from collections import deque

import numpy as np

from .feature_compact import crate_approach_tiles, distance_bucket
from .features import (
    COMBAT_PRESSURE_START,
    COMBAT_PROGRESS_START,
    COMBAT_REACHABLE_START,
    NAV_PROGRESS_START,
    NAV_REACHABLE_START,
    NAV_VISITS_START,
)
from .feature_compact import (
    FEATURE_SIZE as COMPACT_FEATURE_SIZE,
    OPPONENT_BLOCK_START,
    PATCH_OFFSETS,
    PATCH_START,
    ROUTE_STARTS,
    SHORT_CYCLE_STARTS,
)
from .features import FEATURE_SIZE, NOVELTY_START


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
TRANSFORMS = tuple(
    (rotation, reflected)
    for reflected in (False, True)
    for rotation in range(4)
)
SYMMETRY_SCHEMA = "d4-eight-agent047-146-v1"
COMBAT_BLOCK_STARTS = (
    COMBAT_PROGRESS_START, COMBAT_REACHABLE_START, COMBAT_PRESSURE_START,
)
NAV_BLOCK_STARTS = (NAV_PROGRESS_START, NAV_REACHABLE_START, NAV_VISITS_START)


def transform_vector(vector, transform):
    x, y = vector
    rotation, reflected = transform
    for _ in range(rotation):
        x, y = -y, x
    if reflected:
        x = -x
    return x, y


def _direction_permutation(transform):
    return tuple(
        DIRECTIONS.index(transform_vector(direction, transform))
        for direction in DIRECTIONS
    )


_DIRECTION_PERMUTATIONS = {
    transform: _direction_permutation(transform) for transform in TRANSFORMS
}
_SEARCH_ORDERS = {
    transform: tuple(
        _DIRECTION_PERMUTATIONS[transform].index(new) for new in range(4)
    )
    for transform in TRANSFORMS
}


def direction_permutation(transform):
    return _DIRECTION_PERMUTATIONS[transform]


def transform_action(action, transform):
    action = int(action)
    return direction_permutation(transform)[action] if action < 4 else action


def transform_mask(mask, transform):
    source = np.asarray(mask, dtype=bool)
    if source.shape != (6,):
        raise ValueError("Expected one six-action mask")
    result = source.copy()
    permutation = direction_permutation(transform)
    result[list(permutation)] = source[:4]
    result[4:] = source[4:]
    return result


def _build_index_maps():
    maps = {}
    for transform in TRANSFORMS:
        permutation = direction_permutation(transform)
        indices = np.arange(COMPACT_FEATURE_SIZE, dtype=np.int64)
        for start in (0, 4, 9, *SHORT_CYCLE_STARTS, *ROUTE_STARTS):
            indices[start:start + 4] = [
                start + old for old in np.argsort(permutation)
            ]
        indices[OPPONENT_BLOCK_START:OPPONENT_BLOCK_START + 24] = [
            OPPONENT_BLOCK_START + old * 4 + offset
            for new in range(6)
            for old in [permutation.index(new) if new < 4 else new]
            for offset in range(4)
        ]
        for old, offset in enumerate(PATCH_OFFSETS):
            new_offset = transform_vector(offset, transform)
            new = PATCH_OFFSETS.index(new_offset)
            indices[PATCH_START + new] = PATCH_START + old
        maps[transform] = indices
    return maps


_INDEX_MAPS = _build_index_maps()


def _transformed_crate_triplet(game_state, transform):
    """Recompute the tie-broken crate route in transformed board order."""
    field = game_state["field"]
    start = tuple(game_state["self"][3])
    targets = set(crate_approach_tiles(field))
    queue = deque([(start, 0)])
    visited = {start}
    target = None
    target_distance = None
    while queue:
        position, distance = queue.popleft()
        if position in targets:
            target = position
            target_distance = distance
            break
        for old_direction in _SEARCH_ORDERS[transform]:
            dx, dy = DIRECTIONS[old_direction]
            neighbour = position[0] + dx, position[1] + dy
            if (
                0 <= neighbour[0] < field.shape[0]
                and 0 <= neighbour[1] < field.shape[1]
                and field[neighbour] == 0
                and neighbour not in visited
            ):
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))
    if target is None:
        direction = (0, 0)
    else:
        direction = (
            int(np.sign(target[0] - start[0])),
            int(np.sign(target[1] - start[1])),
        )
        direction = transform_vector(direction, transform)
    return direction[0], direction[1], distance_bucket(target_distance)


def _transform_compact_features(features, transform, game_state=None):
    source = np.asarray(features, dtype=np.float32)
    if source.shape != (COMPACT_FEATURE_SIZE,):
        raise ValueError(
            f"Expected one {COMPACT_FEATURE_SIZE}-input compact vector"
        )
    result = source[_INDEX_MAPS[transform]].copy()
    for start in (14, 17, 20):
        result[start:start + 2] = transform_vector(
            source[start:start + 2], transform
        )
    if game_state is not None:
        result[17:20] = _transformed_crate_triplet(game_state, transform)
    return result


def _transform_action_block(source, transform):
    permutation = direction_permutation(transform)
    indices = [permutation.index(new) if new < 4 else new
               for new in range(6)]
    return source[np.asarray(indices, dtype=np.int64)]


def _transform_agent042_features(source, transform, game_state=None):
    result = np.empty(COMPACT_FEATURE_SIZE + 36, dtype=np.float32)
    result[:COMPACT_FEATURE_SIZE] = _transform_compact_features(
        source[:COMPACT_FEATURE_SIZE], transform, game_state
    )
    for start in NAV_BLOCK_STARTS + COMBAT_BLOCK_STARTS:
        result[start:start + 6] = _transform_action_block(
            source[start:start + 6], transform
        )
    return result


def transform_features(features, transform, game_state=None):
    source = np.asarray(features, dtype=np.float32)
    if source.shape != (FEATURE_SIZE,):
        raise ValueError(f"Expected one {FEATURE_SIZE}-input state vector")
    if transform not in TRANSFORMS:
        raise ValueError(f"Unknown transform: {transform!r}")
    result = np.empty(FEATURE_SIZE, dtype=np.float32)
    result[:NOVELTY_START] = _transform_agent042_features(
        source[:NOVELTY_START], transform, game_state
    )
    result[NOVELTY_START:] = _transform_action_block(
        source[NOVELTY_START:], transform
    )
    return result


def transform_transition(state, action, next_state, next_mask, transform,
                         game_state=None, next_game_state=None):
    return (
        transform_features(state, transform, game_state),
        transform_action(action, transform),
        None if next_state is None else transform_features(
            next_state, transform, next_game_state
        ),
        transform_mask(next_mask, transform),
    )


__all__ = [
    "COMBAT_BLOCK_STARTS", "FEATURE_SIZE", "NAV_BLOCK_STARTS",
    "SYMMETRY_SCHEMA", "TRANSFORMS", "direction_permutation",
    "transform_action", "transform_features", "transform_mask",
    "transform_transition", "transform_vector",
]
