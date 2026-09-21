"""D4 transforms for Agent 043's action-aligned novelty block."""

import numpy as np

from agent_code.Agent_042_combat_progress_ddqn_agent import symmetry as _base

from .features import FEATURE_SIZE, NOVELTY_START


TRANSFORMS = _base.TRANSFORMS
SYMMETRY_SCHEMA = "d4-eight-agent043-146-v1"


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
    indices = [permutation.index(new) if new < 4 else new
               for new in range(6)]
    return source[np.asarray(indices, dtype=np.int64)]


def transform_features(features, transform, game_state=None):
    source = np.asarray(features, dtype=np.float32)
    if source.shape != (FEATURE_SIZE,):
        raise ValueError(f"Expected one {FEATURE_SIZE}-input state vector")
    if transform not in TRANSFORMS:
        raise ValueError(f"Unknown transform: {transform!r}")
    result = np.empty(FEATURE_SIZE, dtype=np.float32)
    result[:NOVELTY_START] = _base.transform_features(
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
    "FEATURE_SIZE", "SYMMETRY_SCHEMA", "TRANSFORMS", "direction_permutation",
    "transform_action", "transform_features", "transform_mask",
    "transform_transition", "transform_vector",
]
