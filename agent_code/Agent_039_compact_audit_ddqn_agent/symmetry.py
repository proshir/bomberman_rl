"""D4 transforms for Agent 039's 104-input replay contract."""

from collections import deque

import numpy as np

from agent_code.combat_fqi_agent.features import crate_approach_tiles, distance_bucket

from .features import (
    FEATURE_SIZE, OPPONENT_BLOCK_SIZE, OPPONENT_BLOCK_START, PATCH_OFFSETS,
    PATCH_START, ROUTE_STARTS, SHORT_CYCLE_STARTS,
)


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
TRANSFORMS = tuple(
    (rotation, reflected)
    for reflected in (False, True)
    for rotation in range(4)
)
SYMMETRY_SCHEMA = "d4-eight-agent039-104-v1"


def transform_vector(vector, transform):
    x, y = vector
    rotation, reflected = transform
    for _ in range(rotation):
        x, y = -y, x
    if reflected:
        x = -x
    return x, y


def direction_permutation(transform):
    return tuple(
        DIRECTIONS.index(transform_vector(direction, transform))
        for direction in DIRECTIONS
    )


def transform_action(action, transform):
    action = int(action)
    return direction_permutation(transform)[action] if action < 4 else action


def transform_mask(mask, transform):
    source = np.asarray(mask, dtype=bool)
    if source.shape != (6,):
        raise ValueError("Expected one six-action mask")
    result = source.copy()
    for old, new in enumerate(direction_permutation(transform)):
        result[new] = source[old]
    return result


def _permute_directions(result, source, start, transform, block_size=1):
    permutation = direction_permutation(transform)
    for old, new in enumerate(permutation):
        old_slice = slice(start + old * block_size,
                          start + (old + 1) * block_size)
        new_slice = slice(start + new * block_size,
                          start + (new + 1) * block_size)
        result[new_slice] = source[old_slice]


def _transform_xy(result, source, start, transform):
    result[start:start + 2] = transform_vector(
        source[start:start + 2], transform
    )


def _transform_patch(result, source, transform):
    for old, offset in enumerate(PATCH_OFFSETS):
        new_offset = transform_vector(offset, transform)
        result[PATCH_START + PATCH_OFFSETS.index(new_offset)] = (
            source[PATCH_START + old]
        )


def _transformed_crate_triplet(game_state, transform):
    """Match Agent 038's directional nearest-crate tie-break handling."""
    field = game_state["field"]
    start = tuple(game_state["self"][3])
    targets = set(crate_approach_tiles(field))
    permutation = direction_permutation(transform)
    search_order = tuple(permutation.index(new) for new in range(4))
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
    result = source.copy()

    for start in (0, 4, 9):
        _permute_directions(result, source, start, transform)
    for start in (14, 17, 20):
        _transform_xy(result, source, start, transform)
    _transform_patch(result, source, transform)
    for start in SHORT_CYCLE_STARTS:
        _permute_directions(result, source, start, transform)
    _permute_directions(
        result, source, OPPONENT_BLOCK_START, transform,
        block_size=OPPONENT_BLOCK_SIZE,
    )
    for start in ROUTE_STARTS:
        _permute_directions(result, source, start, transform)

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
