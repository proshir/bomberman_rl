"""Agent 031 features with shared and allocation-light board searches.

The representation remains exactly 113 inputs: Agent 029's 101-input vector
followed by Agent 031's 12 offensive summaries.  This module changes the
implementation of the searches only.  In particular, the opponent envelope
and legality results are shared across the six action-conditioned summaries,
and the survivor search keeps its hot loop local.
"""

from collections import deque

import numpy as np
import settings as s

from agent_code.Agent_025_combat_ddqn_short_cycle_agent.features import (
    ACTIONS,
    FEATURE_SIZE as AGENT_027_FEATURE_SIZE,
    SHORT_CYCLE_FEATURES,
    short_cycle_features,
)
from agent_code.combat_dqn_r_topology_agent.features import state_to_features as _base_features
from agent_code.combat_fqi_agent.safety import (
    MOVE_DELTAS,
    action_is_legal,
    blast_tiles,
    build_danger_schedule,
)


OPPONENT_WINDOW = s.BOMB_TIMER + 1
OPPONENT_FEATURES_PER_ACTION = 6
ADVERSARIAL_FEATURE_SIZE = AGENT_027_FEATURE_SIZE + len(ACTIONS) * OPPONENT_FEATURES_PER_ACTION
FEATURE_SIZE = 113
MOVES = ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0))


def _inside(field, position):
    x, y = position
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def _destination(game_state, action):
    x, y = game_state["self"][3]
    dx, dy = MOVE_DELTAS[action]
    return int(x + dx), int(y + dy)


def _opponent_reachability(game_state, horizon):
    """Compute the same envelope as the shared safety code with local hot loops."""
    field = game_state["field"]
    width, height = field.shape
    blocked = {tuple(position) for position, _ in game_state["bombs"]}
    reachable = {tuple(other[3]) for other in game_state["others"]}
    schedule = {}
    for time_step in range(1, horizon + 1):
        following = set(reachable)
        for x, y in reachable:
            for dx, dy in MOVES[1:]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < width and 0 <= ny < height and
                        field[nx, ny] == 0 and (nx, ny) not in blocked):
                    following.add((nx, ny))
        reachable = following
        schedule[time_step] = frozenset(reachable)
    return schedule


def _tile_open(field, position, time_step, origin, has_left_origin, bombs):
    x, y = position
    if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]) or field[position] != 0:
        return False
    for bomb_position, explosion_step, is_own_new_bomb in bombs:
        if position != bomb_position or time_step >= explosion_step:
            continue
        if (is_own_new_bomb and position == origin and not has_left_origin):
            continue
        return False
    return True


def _add_bomb_danger(danger, field, position, explosion_step):
    for tile in blast_tiles(field, position):
        for time_step in range(explosion_step, explosion_step + s.EXPLOSION_TIMER):
            danger[tile].add(time_step)


def _immediate_enemy_bombs(game_state):
    return tuple(sorted({tuple(other[3]) for other in game_state["others"]
                         if bool(other[2])}))


def _window_search(game_state, action, contested, enemy_bomb=None, legal=True):
    """Search one action while reusing the state-wide opponent envelope."""
    if not legal:
        return 0, 0
    field = game_state["field"]
    origin = tuple(game_state["self"][3])
    start = _destination(game_state, action)
    own_bomb = action == "BOMB"
    danger = build_danger_schedule(game_state, own_bomb)
    bombs = [(tuple(position), int(timer) + 1, False)
             for position, timer in game_state["bombs"]]
    if own_bomb:
        bombs.append((origin, s.BOMB_TIMER + 1, True))
    if enemy_bomb is not None:
        bombs.append((enemy_bomb, s.BOMB_TIMER + 1, False))
        # Keep the schedule mutation separate from the shared base schedule.
        danger = dict(danger)
        for tile in blast_tiles(field, enemy_bomb):
            danger.setdefault(tile, set()).update(
                range(s.BOMB_TIMER + 1,
                      s.BOMB_TIMER + 1 + s.EXPLOSION_TIMER)
            )

    left_origin = start != origin
    if 1 in danger.get(start, ()) or not _tile_open(
            field, start, 1, origin, left_origin, bombs):
        return 0, 0

    queue = deque([(start, 1, left_origin)])
    visited = {(start, 1, left_origin)}
    deepest = 1
    frontier = {start}
    while queue:
        position, time_step, has_left = queue.popleft()
        if time_step > deepest:
            deepest = time_step
            frontier = {position}
        elif time_step == deepest:
            frontier.add(position)
        if time_step >= OPPONENT_WINDOW:
            continue
        x, y = position
        next_time = time_step + 1
        for dx, dy in MOVES:
            next_position = (x + dx, y + dy)
            next_left = has_left or next_position != origin
            if not _tile_open(field, next_position, next_time, origin,
                              next_left, bombs):
                continue
            if next_time in danger.get(next_position, ()):
                continue
            if next_position in contested.get(next_time, ()):
                continue
            node = (next_position, next_time, next_left)
            if node not in visited:
                visited.add(node)
                queue.append(node)
    return deepest, len(frontier)


def opponent_window_features(game_state):
    """Preserve Agent 029's 36 values while sharing state-wide searches."""
    if game_state is None:
        return np.zeros(len(ACTIONS) * OPPONENT_FEATURES_PER_ACTION,
                        dtype=np.float32)
    field = game_state["field"]
    board_area = max(1, int(field.shape[0] * field.shape[1]))
    enemy_bombs = _immediate_enemy_bombs(game_state)
    contested = _opponent_reachability(game_state, OPPONENT_WINDOW)
    first_contested = contested.get(1, frozenset())
    legal_actions = [action_is_legal(game_state, action) for action in ACTIONS]
    values = np.empty(len(ACTIONS) * OPPONENT_FEATURES_PER_ACTION,
                      dtype=np.float32)
    cursor = 0
    origin = tuple(game_state["self"][3])
    enemy_flag = float(bool(enemy_bombs))
    for action, legal in zip(ACTIONS, legal_actions):
        if not legal:
            values[cursor:cursor + OPPONENT_FEATURES_PER_ACTION] = 0.0
            cursor += OPPONENT_FEATURES_PER_ACTION
            continue
        destination = _destination(game_state, action)
        destination_contested = float(destination != origin and
                                      destination in first_contested)
        baseline_time, baseline_frontier = _window_search(
            game_state, action, contested, legal=True,
        )
        response_stats = [
            _window_search(game_state, action, contested, bomb, legal=True)
            for bomb in enemy_bombs
        ]
        worst_time = min([baseline_time] + [item[0] for item in response_stats])
        worst_frontier = min(
            [baseline_frontier] + [item[1] for item in response_stats]
        )
        values[cursor:cursor + OPPONENT_FEATURES_PER_ACTION] = (
            1.0, enemy_flag, destination_contested,
            min(1.0, worst_time / float(OPPONENT_WINDOW)),
            min(1.0, worst_frontier / float(board_area)),
            float(bool(response_stats) and worst_time < baseline_time),
        )
        cursor += OPPONENT_FEATURES_PER_ACTION
    return values


def _open(field, point):
    x, y = point
    return (0 <= x < field.shape[0] and 0 <= y < field.shape[1] and
            field[point] == 0)


def _survivor_tiles(field, origin, danger, bombs, horizon):
    """Agent 031 endpoint search with fewer Python helper calls."""
    frontier = {(origin, False)}
    danger_get = danger.get
    width, height = field.shape
    for step in range(1, horizon + 1):
        following = set()
        for (x, y), left_origin in frontier:
            for dx, dy in MOVES:
                px, py = x + dx, y + dy
                if not (0 <= px < width and 0 <= py < height) or field[px, py] != 0:
                    continue
                if step in danger_get((px, py), ()):
                    continue
                left = left_origin or (px, py) != origin
                blocked = False
                for position, deadline in bombs:
                    if ((px, py) == position and step < deadline and
                            not ((px, py) == origin and not left)):
                        blocked = True
                        break
                if not blocked:
                    following.add(((px, py), left))
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
    width, height = field.shape
    while queue:
        x, y = queue.popleft()
        next_distance = distances[(x, y)] + 1
        for dx, dy in MOVES[1:]:
            point = x + dx, y + dy
            px, py = point
            if (0 <= px < width and 0 <= py < height and field[px, py] == 0 and
                    point not in blocked and point not in distances):
                distances[point] = next_distance
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
    planted = (build_danger_schedule(game_state, extra_bomb=True)
               if armed else baseline)
    horizon = max((max(times) for times in planted.values()), default=1) + 1
    bombs = [(tuple(pos), int(timer) + 1) for pos, timer in game_state["bombs"]]
    added_bombs = bombs + ([(origin, s.BOMB_TIMER + 1)] if armed else [])
    affected = set(blast_tiles(field, origin))
    distances = _approach_distances(game_state)
    opponents = sorted(game_state["others"], key=lambda other: (
        abs(other[3][0] - origin[0]) + abs(other[3][1] - origin[1]),
        tuple(other[3]),
    ))[:3]
    for index, opponent in enumerate(opponents):
        point = tuple(opponent[3])
        before = _survivor_tiles(field, point, baseline, bombs, horizon)
        after = _survivor_tiles(field, point, planted, added_bombs, horizon)
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
    prefix = _base_features(
        game_state, previous_action, recent_visits, steps_since_progress,
    )
    if prefix is None:
        return None
    history = short_cycle_features(action_history, action_successes,
                                   position_history)
    adversarial = opponent_window_features(game_state)
    base = np.concatenate((prefix, history, adversarial))
    if len(base) != ADVERSARIAL_FEATURE_SIZE:
        raise AssertionError(
            f"Agent 029 prefix changed: expected {ADVERSARIAL_FEATURE_SIZE}, "
            f"got {len(base)}"
        )
    return np.concatenate((base, offensive_features(game_state))).astype(
        np.float32, copy=False
    )


__all__ = [
    "ACTIONS", "FEATURE_SIZE", "OPPONENT_FEATURES_PER_ACTION",
    "OPPONENT_WINDOW", "SHORT_CYCLE_FEATURES", "offensive_features",
    "opponent_window_features", "short_cycle_features", "state_to_features",
]
