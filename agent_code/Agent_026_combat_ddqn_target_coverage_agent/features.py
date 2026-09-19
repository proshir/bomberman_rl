"""Add global target-coverage summaries without replacing local target cues."""

# Sahand was here.

from collections import deque

import numpy as np

from agent_code.combat_dqn_r_topology_agent.features import (
    state_to_features as topology_features,
)
from agent_code.combat_fqi_agent.features import crate_approach_tiles
from agent_code.combat_fqi_history_antistag_agent.features import ACTIONS


COVERAGE_FEATURES = 12
FEATURE_SIZE = 46 + COVERAGE_FEATURES
BANDS = (4, 8, 12)


def _distance_map(field, targets):
    starts = {
        tuple(target) for target in targets
        if (0 <= target[0] < field.shape[0] and
            0 <= target[1] < field.shape[1] and field[tuple(target)] == 0)
    }
    distances = {target: 0 for target in starts}
    queue = deque(starts)
    while queue:
        position = queue.popleft()
        x, y = position
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            neighbour = x + dx, y + dy
            if not (0 <= neighbour[0] < field.shape[0] and
                    0 <= neighbour[1] < field.shape[1]):
                continue
            if field[neighbour] != 0 or neighbour in distances:
                continue
            distances[neighbour] = distances[position] + 1
            queue.append(neighbour)
    return distances


def _component(field, start):
    distances = _distance_map(field, [start])
    return distances


def target_coverage_features(game_state):
    """Summarize how much useful target structure remains reachable.

    These are visible-state summaries only.  The existing nearest-target
    direction and distance features remain in the 46-feature base vector.
    """
    field = game_state["field"]
    position = tuple(game_state["self"][3])
    component = _component(field, position)
    coin_distances = _distance_map(field, game_state["coins"])
    crate_targets = crate_approach_tiles(field)
    crate_distances = _distance_map(field, crate_targets)

    values = []
    for distances, normalizer in (
        (coin_distances, max(1.0, float(len(game_state["coins"]))),),
        (crate_distances, max(1.0, float(len(crate_targets))),),
    ):
        for band in BANDS:
            count = sum(
                distance <= band
                for tile, distance in distances.items()
                if tile in component
            )
            values.append(min(1.0, count / normalizer))

    reachable_coins = sum(
        tile in component for tile in game_state["coins"]
    )
    reachable_crate_targets = sum(
        tile in component for tile in crate_targets
    )
    free_tiles = max(1, int(np.count_nonzero(field == 0)))
    values.extend((
        min(1.0, len(component) / float(field.size)),
        min(1.0, reachable_coins / max(1, len(game_state["coins"]))),
        min(1.0, reachable_crate_targets / max(1, len(crate_targets))),
        min(1.0, reachable_coins / 10.0),
        min(1.0, reachable_crate_targets / 20.0),
        min(1.0, len(component) / float(free_tiles)),
    ))
    result = np.asarray(values, dtype=float)
    if len(result) != COVERAGE_FEATURES:
        raise AssertionError(
            f"Agent 026 coverage size changed: expected {COVERAGE_FEATURES}, "
            f"got {len(result)}"
        )
    return result


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0):
    base = topology_features(
        game_state, previous_action, recent_visits, steps_since_progress
    )
    if base is None:
        return None
    features = np.concatenate((base, target_coverage_features(game_state)))
    if len(features) != FEATURE_SIZE:
        raise AssertionError(
            f"Agent 026 feature size changed: expected {FEATURE_SIZE}, "
            f"got {len(features)}"
        )
    return features


__all__ = [
    "ACTIONS", "COVERAGE_FEATURES", "FEATURE_SIZE",
    "state_to_features", "target_coverage_features",
]
