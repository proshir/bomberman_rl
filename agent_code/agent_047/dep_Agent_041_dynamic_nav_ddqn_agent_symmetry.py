"""D4 transforms for Agent041's action-aligned navigation features."""

import numpy as np

from . import dep_Agent_040_optimized_compact_ddqn_agent_symmetry as _base

from .dep_Agent_041_dynamic_nav_ddqn_agent_features import (
    FEATURE_SIZE, NAV_PROGRESS_START, NAV_REACHABLE_START, NAV_VISITS_START,
)


TRANSFORMS = _base.TRANSFORMS
SYMMETRY_SCHEMA = "d4-eight-agent041-122-v1"
NAV_BLOCK_STARTS = (
    NAV_PROGRESS_START, NAV_REACHABLE_START, NAV_VISITS_START,
)


def direction_permutation(transform):
    return _base.direction_permutation(transform)


def transform_vector(vector, transform):
    return _base.transform_vector(vector, transform)


def transform_action(action, transform):
    return _base.transform_action(action, transform)


def transform_mask(mask, transform):
    return _base.transform_mask(mask, transform)


def _transform_action_block(source, transform):
    permutation = direction_permutation(transform)
    indices = [
        permutation.index(new) if new < 4 else new
        for new in range(6)
    ]
    return source[np.asarray(indices, dtype=np.int64)]


def transform_features(features, transform, game_state=None):
    source = np.asarray(features, dtype=np.float32)
    if source.shape != (FEATURE_SIZE,):
        raise ValueError(f"Expected one {FEATURE_SIZE}-input state vector")
    if transform not in TRANSFORMS:
        raise ValueError(f"Unknown transform: {transform!r}")

    result = np.empty(FEATURE_SIZE, dtype=np.float32)
    result[:NAV_PROGRESS_START] = _base.transform_features(
        source[:NAV_PROGRESS_START], transform, game_state
    )
    for start in NAV_BLOCK_STARTS:
        result[start:start + 6] = _transform_action_block(
            source[start:start + 6], transform
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
    "FEATURE_SIZE", "NAV_BLOCK_STARTS", "SYMMETRY_SCHEMA", "TRANSFORMS",
    "direction_permutation", "transform_action", "transform_features",
    "transform_mask", "transform_transition", "transform_vector",
]
