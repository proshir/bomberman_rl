"""Twelve offensive summaries appended to Agent 029's unchanged 101 inputs.

Searches use visible static crates, known bomb timers and lingering flames.
They assume opponents may move or wait, not plant future bombs. Other players'
future occupancy is unknown. A predicted trap is therefore conditional, not a
guaranteed kill. No additional hard action mask or reward is introduced.
"""

from collections import deque

import numpy as np
import settings as s

from agent_code.Agent_029_combat_ddqn_adversarial_window_agent.features import (
    ACTIONS, state_to_features as base_features,
)
from agent_code.combat_fqi_agent.safety import blast_tiles, build_danger_schedule

FEATURE_SIZE = 113
MOVES = ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0))


def _open(field, point):
    x, y = point
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[point] == 0


def _survivor_tiles(field, origin, danger, bombs, horizon):
    """Distinct endpoints reachable after the entire danger window.

    An opponent already standing on a bomb may leave it, but cannot return
    until it has exploded. All paths share the same comparison horizon.
    """
    frontier = {(origin, False)}
    for step in range(1, horizon + 1):
        following = set()
        for (x, y), left_origin in frontier:
            for dx, dy in MOVES:
                point = x + dx, y + dy
                left = left_origin or point != origin
                if not _open(field, point) or step in danger.get(point, ()):
                    continue
                if any(point == pos and step < deadline and
                       not (point == origin and not left)
                       for pos, deadline in bombs):
                    continue
                following.add((point, left))
        frontier = following
        if not frontier:
            break
    return {point for point, _ in frontier}


def _approach_distances(game_state):
    field = game_state["field"]
    origin = tuple(game_state["self"][3])
    blocked = {tuple(pos) for pos, _ in game_state["bombs"]}
    blocked.update(tuple(other[3]) for other in game_state["others"])
    distances = {origin: 0}
    queue = deque([origin])
    while queue:
        x, y = queue.popleft()
        for dx, dy in MOVES[1:]:
            point = x + dx, y + dy
            if _open(field, point) and point not in blocked and point not in distances:
                distances[point] = distances[(x, y)] + 1
                queue.append(point)
    return distances


def offensive_features(game_state):
    values = np.zeros((3, 4), dtype=np.float32)
    if game_state is None or not game_state["others"]:
        return values.ravel()
    field = game_state["field"]
    origin = tuple(game_state["self"][3])
    armed = bool(game_state["self"][2])
    baseline = build_danger_schedule(game_state)
    planted = build_danger_schedule(game_state, extra_bomb=True) if armed else baseline
    horizon = max((max(times) for times in planted.values()), default=1) + 1
    bombs = [(tuple(pos), int(timer) + 1) for pos, timer in game_state["bombs"]]
    added_bombs = bombs + ([(origin, s.BOMB_TIMER + 1)] if armed else [])
    affected = set(blast_tiles(field, origin))
    distances = _approach_distances(game_state)
    # Stable spatial ordering, independent of engine list order or agent names.
    opponents = sorted(game_state["others"], key=lambda other: (
        abs(other[3][0] - origin[0]) + abs(other[3][1] - origin[1]),
        tuple(other[3]),
    ))[:3]
    for index, opponent in enumerate(opponents):
        point = tuple(opponent[3])
        before = _survivor_tiles(field, point, baseline, bombs, horizon)
        after = _survivor_tiles(field, point, planted, added_bombs, horizon)
        # Blast geometry is symmetric: reverse rays identify attack tiles.
        approaches = [distances[tile] for tile in blast_tiles(field, point)
                      if tile in distances]
        values[index] = (
            float(armed and point in affected),
            max(0.0, 1.0 - len(after) / len(before)) if before else 0.0,
            float(armed and bool(before) and not after),
            1.0 / (1.0 + min(approaches)) if approaches else 0.0,
        )
    return values.ravel()


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0, action_history=(),
                      action_successes=(), position_history=()):
    prefix = base_features(game_state, previous_action, recent_visits,
                           steps_since_progress, action_history,
                           action_successes, position_history)
    if prefix is None:
        return None
    return np.concatenate((prefix, offensive_features(game_state))).astype(np.float32)
