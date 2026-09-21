"""Allocation-light D4 transforms for Agent 040's 104-input contract."""

from collections import deque

import numpy as np

from agent_code.combat_fqi_agent.features import crate_approach_tiles, distance_bucket

from .features import (
    FEATURE_SIZE, OPPONENT_BLOCK_START, PATCH_OFFSETS, PATCH_START,
    ROUTE_STARTS, SHORT_CYCLE_STARTS,
)


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
TRANSFORMS = tuple(
    (rotation, reflected)
    for reflected in (False, True)
    for rotation in range(4)
)
SYMMETRY_SCHEMA = "d4-eight-agent040-104-v1"


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
        _DIRECTION_PERMUTATIONS[transform].index(new)
        for new in range(4)
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
        indices = np.arange(FEATURE_SIZE, dtype=np.int64)
        for start in (0, 4, 9):
            indices[start:start + 4] = [
                start + old for old in np.argsort(permutation)
            ]
        for start in SHORT_CYCLE_STARTS:
            indices[start:start + 4] = [
                start + old for old in np.argsort(permutation)
            ]
        for start in (64,):
            indices[start:start + 24] = [
                start + old * 4 + offset
                for new in range(6) for old in [
                    permutation.index(new) if new < 4 else new
                ] for offset in range(4)
            ]
        for start in ROUTE_STARTS:
            indices[start:start + 4] = [
                start + old for old in np.argsort(permutation)
            ]
        for old, offset in enumerate(PATCH_OFFSETS):
            new_offset = transform_vector(offset, transform)
            new = PATCH_OFFSETS.index(new_offset)
            indices[PATCH_START + new] = PATCH_START + old
        maps[transform] = indices
    return maps


_INDEX_MAPS = _build_index_maps()


def _transform_xy(result, source, start, transform):
    result[start:start + 2] = transform_vector(
        source[start:start + 2], transform
    )


def _transformed_crate_triplet(game_state, transform):
    """Retain Agent 039's tie-breaking correction with precomputed order."""
    field = game_state["field"]
    start = tuple(game_state["self"][3])
    targets = set(crate_approach_tiles(field))
    search_order = _SEARCH_ORDERS[transform]
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
        for old_direction in search_order:
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


def transform_features(features, transform, game_state=None):
    source = np.asarray(features, dtype=np.float32)
    if source.shape != (FEATURE_SIZE,):
        raise ValueError(f"Expected one {FEATURE_SIZE}-input state vector")
    if transform not in TRANSFORMS:
        raise ValueError(f"Unknown transform: {transform!r}")

    result = source[_INDEX_MAPS[transform]].copy()
    for start in (14, 17, 20):
        _transform_xy(result, source, start, transform)
    if game_state is not None:
        result[17:20] = _transformed_crate_triplet(game_state, transform)
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
    "DIRECTIONS", "FEATURE_SIZE", "SYMMETRY_SCHEMA", "TRANSFORMS",
    "direction_permutation", "transform_action", "transform_features",
    "transform_mask", "transform_transition", "transform_vector",
]
