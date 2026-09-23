"""Eight unique square symmetries for the 28-input/action-mask contract.

Feature directions use UP, RIGHT, DOWN, LEFT. A transform rotates clockwise
zero to three times, then optionally reflects the x coordinate.
"""

from dataclasses import replace

import numpy as np

from .dep_Agent_036_compact_fqi_robust_agent_features import FEATURE_SIZE


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
TRANSFORMS = tuple((rotation, reflected)
                   for reflected in (False, True) for rotation in range(4))
SYMMETRY_SCHEMA = "d4-eight-actions-history-masks-v1"


def direction_permutation(transform):
    rotation, reflected = transform
    result = []
    for x, y in DIRECTIONS:
        for _ in range(rotation):
            x, y = -y, x
        if reflected:
            x = -x
        result.append(DIRECTIONS.index((x, y)))
    return tuple(result)


def transform_action(action, transform):
    return direction_permutation(transform)[action] if action < 4 else action


def transform_mask(mask, transform):
    result = np.asarray(mask, dtype=bool).copy()
    permutation = direction_permutation(transform)
    for old, new in enumerate(permutation):
        result[new] = mask[old]
    return result


def transform_features(features, transform):
    result = np.asarray(features, dtype=np.float32).copy()
    if result.shape != (FEATURE_SIZE,):
        raise ValueError("Expected one 28-input state vector")
    permutation = direction_permutation(transform)
    for start in (0, 4, 8, 12):
        for old, new in enumerate(permutation):
            result[start + new] = features[start + old]
    for old, new in enumerate(permutation):
        result[19 + new] = features[19 + old]
    # Bomb availability/yield/escape, WAIT/BOMB/START, visits and progress
    # are invariant under board orientation.
    return result


def transform_transition(record, transform):
    return replace(
        record,
        features=transform_features(record.features, transform),
        action=transform_action(record.action, transform),
        next_features=(None if record.next_features is None else
                       transform_features(record.next_features, transform)),
        allowed=transform_mask(record.allowed, transform),
        next_allowed=transform_mask(record.next_allowed, transform),
    )
