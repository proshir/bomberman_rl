"""Direct, cached implementation of Agent 039's 104-input representation.

Agent 039 computed Agent 038's complete 117-value vector and then copied the
104 retained values.  Agent 040 assembles the same compact contract directly.
The optional :class:`StateContext` is owned by the callbacks and lets the
action mask, combat features, opponent window, and route features reuse the
same state-local legality, safety, danger, and board-search results.
"""

from collections import deque
from functools import lru_cache

import numpy as np
import settings as s

from agent_code.Agent_025_combat_ddqn_short_cycle_agent.features import (
    short_cycle_features,
)
from agent_code.combat_fqi_agent.features import (
    _direction,
    bomb_value,
    count_bucket,
    crate_approach_tiles,
    danger_bucket,
    distance_bucket,
    nearest_target,
)
from agent_code.combat_fqi_history_antistag_agent.features import (
    stagnation_bucket,
)
from agent_code.combat_fqi_history_antistag_agent.safety import (
    ACTIONS,
    MOVE_DELTAS,
    action_is_legal,
    blast_tiles,
    build_danger_schedule,
    can_survive_action,
    escape_distance_after_bomb,
    safe_action_indices,
    best_survival_action_indices,
)
from agent_code.Agent_036_compact_fqi_robust_agent.features import (
    candidate_crate_tiles,
)


FEATURE_SIZE = 104
FEATURE_SCHEMA = "agent040-optimized-compact-v1"
AGENT_039_FEATURE_SIZE = FEATURE_SIZE
OPPONENT_WINDOW = s.BOMB_TIMER + 1
OPPONENT_BLOCK_SIZE = 4

# Compact layout.  The first 29 values are the combat block, followed by the
# two retained history scalars, eight non-centre topology cells, five topology
# summaries, 19 short-cycle values, one global armed-opponent flag, 24 compact
# opponent-response values, and 16 route values.
PATCH_START = 31
PATCH_OFFSETS = tuple(
    (dx, dy)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if (dx, dy) != (0, 0)
)
SHORT_CYCLE_STARTS = (44, 50)
GLOBAL_ARMED_OPPONENT_INDEX = 63
OPPONENT_BLOCK_START = 64
ROUTE_STARTS = (88, 92, 96, 100)
ROUTE_FEATURE_SIZE = 16

_MOVES = ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0))
_MOVE_ACTIONS = tuple(ACTIONS[:4])


def _field_key(field):
    array = np.asarray(field)
    return array.shape, array.dtype.str, array.tobytes()


@lru_cache(maxsize=256)
def _static_candidate_crate_tiles(shape, dtype, field_bytes):
    """Cache crate-ray candidates for repeated static board layouts."""
    field = np.frombuffer(field_bytes, dtype=np.dtype(dtype)).reshape(shape)
    return tuple(sorted(candidate_crate_tiles(field)))


def _distance_map(field, targets):
    distances = {tuple(tile): 0 for tile in targets if field[tuple(tile)] == 0}
    queue = deque(distances)
    while queue:
        x, y = queue.popleft()
        next_distance = distances[(x, y)] + 1
        for action in _MOVE_ACTIONS:
            dx, dy = MOVE_DELTAS[action]
            tile = (x + dx, y + dy)
            if (0 <= tile[0] < field.shape[0] and
                    0 <= tile[1] < field.shape[1] and
                    field[tile] == 0 and tile not in distances):
                distances[tile] = next_distance
                queue.append(tile)
    return distances


def _tile_value(field, x, y):
    if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
        return -1
    return int(field[x, y])


def _compact_topology_features(game_state):
    """Compute only Agent 039's eight patch cells and five summaries."""
    field = game_state["field"]
    x, y = map(int, game_state["self"][3])
    patch = [
        _tile_value(field, x + dx, y + dy)
        for dx, dy in PATCH_OFFSETS
    ]
    neighbours = [
        _tile_value(field, x - 1, y),
        _tile_value(field, x + 1, y),
        _tile_value(field, x, y - 1),
        _tile_value(field, x, y + 1),
    ]
    free_neighbours = sum(tile == 0 for tile in neighbours)
    crate_neighbours = sum(tile == 1 for tile in neighbours)
    radius_two = [
        _tile_value(field, x + dx, y + dy)
        for dx in range(-2, 3)
        for dy in range(-2, 3)
        if abs(dx) + abs(dy) <= 2
    ]
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
        dtype=np.float32,
    )


def _legal_actions_fast(game_state):
    """Compute immediate legality once, sharing occupancy construction."""
    field = game_state["field"]
    origin = tuple(game_state["self"][3])
    occupied = {tuple(position) for position, _ in game_state["bombs"]}
    occupied.update(tuple(other[3]) for other in game_state["others"])
    result = []
    for index, action in enumerate(ACTIONS):
        if action == "BOMB":
            legal = bool(game_state["self"][2])
        elif action == "WAIT":
            legal = True
        else:
            dx, dy = MOVE_DELTAS[action]
            destination = (origin[0] + dx, origin[1] + dy)
            legal = (
                0 <= destination[0] < field.shape[0] and
                0 <= destination[1] < field.shape[1] and
                field[destination] == 0 and destination not in occupied
            )
        if legal:
            result.append(index)
    return tuple(result)


def _tile_open(field, position, time_step, origin, has_left_origin, bombs):
    x, y = position
    if (not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]) or
            field[position] != 0):
        return False
    for bomb_position, explosion_step, is_own_new_bomb in bombs:
        if position != bomb_position or time_step >= explosion_step:
            continue
        if is_own_new_bomb and position == origin and not has_left_origin:
            continue
        return False
    return True


def _opponent_reachability(game_state, horizon):
    field = game_state["field"]
    width, height = field.shape
    blocked = {tuple(position) for position, _ in game_state["bombs"]}
    reachable = {tuple(other[3]) for other in game_state["others"]}
    schedule = {}
    for time_step in range(1, horizon + 1):
        following = set(reachable)
        for x, y in reachable:
            for action in _MOVE_ACTIONS:
                dx, dy = MOVE_DELTAS[action]
                nx, ny = x + dx, y + dy
                if (0 <= nx < width and 0 <= ny < height and
                        field[nx, ny] == 0 and (nx, ny) not in blocked):
                    following.add((nx, ny))
        reachable = following
        schedule[time_step] = frozenset(reachable)
    return schedule


def _destination(game_state, action):
    x, y = game_state["self"][3]
    dx, dy = MOVE_DELTAS[action]
    return int(x + dx), int(y + dy)


def _immediate_enemy_bombs(game_state):
    return tuple(sorted({tuple(other[3]) for other in game_state["others"]
                         if bool(other[2])}))


class StateContext:
    """Bounded-lifetime computations shared for one decision state."""

    def __init__(self, game_state):
        self.game_state = game_state
        self._legal = None
        self._safe = None
        self._candidates = None
        self._danger = {}
        self._escape_distance = None
        self._opponent = None
        self._routes = None
        self._distance_maps = {}
        self._feasible_crates = None
        self._features = {}

    def legal_indices(self):
        if self._legal is None:
            self._legal = _legal_actions_fast(self.game_state)
        return self._legal

    def is_legal(self, action):
        if isinstance(action, str):
            try:
                action = ACTIONS.index(action)
            except ValueError:
                return False
        return int(action) in self.legal_indices()

    def safe_indices(self):
        if self._safe is None:
            self._safe = tuple(safe_action_indices(self.game_state))
        return self._safe

    def candidate_indices(self):
        if self._candidates is None:
            safe = self.safe_indices()
            self._candidates = (
                safe if safe else tuple(
                    best_survival_action_indices(self.game_state)
                )
            )
        return self._candidates

    def danger(self, extra_bomb=False):
        extra_bomb = bool(extra_bomb)
        if extra_bomb not in self._danger:
            self._danger[extra_bomb] = build_danger_schedule(
                self.game_state, extra_bomb=extra_bomb
            )
        return self._danger[extra_bomb]

    def escape_distance(self):
        if self._escape_distance is None:
            self._escape_distance = escape_distance_after_bomb(
                self.game_state
            )
        return self._escape_distance

    def opponent(self):
        if self._opponent is None:
            self._opponent = _compact_opponent_features(
                self.game_state, self
            )
        return self._opponent

    def distance_map(self, targets):
        key = tuple(sorted(tuple(target) for target in targets))
        if key not in self._distance_maps:
            self._distance_maps[key] = _distance_map(
                self.game_state["field"], key
            )
        return self._distance_maps[key]

    def feasible_crates(self):
        if self._feasible_crates is not None:
            return self._feasible_crates
        game_state = self.game_state
        field = game_state["field"]
        current = tuple(game_state["self"][3])
        occupied = {tuple(other[3]) for other in game_state["others"]}
        occupied.update(tuple(position) for position, _ in game_state["bombs"])
        shape, dtype, field_bytes = _field_key(field)
        static_candidates = set(
            _static_candidate_crate_tiles(shape, dtype, field_bytes)
        )
        reachable = self.distance_map((current,))
        result = set()
        for tile in sorted(static_candidates.intersection(reachable)):
            if tile != current and tile in occupied:
                continue
            candidate = dict(game_state)
            name, score, _, _ = game_state["self"]
            candidate["self"] = (name, score, True, tile)
            if bomb_value(candidate)[0] > 0 and can_survive_action(
                    candidate, "BOMB"):
                result.add(tile)
        self._feasible_crates = frozenset(result)
        return self._feasible_crates

    def routes(self):
        if self._routes is not None:
            return self._routes
        game_state = self.game_state
        coins = tuple(tuple(coin) for coin in game_state["coins"])
        crate_targets = tuple(self.feasible_crates())
        self._routes = tuple(
            _movement_routes(game_state, coins, self) +
            _movement_routes(game_state, crate_targets, self)
        )
        return self._routes

    def feature(self, key):
        return self._features.get(key)

    def store_feature(self, key, value):
        self._features[key] = value


def _movement_routes(game_state, targets, context):
    distances = context.distance_map(targets)
    x, y = game_state["self"][3]
    legal = set(context.legal_indices())
    costs = []
    reached = []
    for index, action in enumerate(_MOVE_ACTIONS):
        dx, dy = MOVE_DELTAS[action]
        distance = distances.get((x + dx, y + dy)) if index in legal else None
        costs.append(float(distance + 1) if distance is not None else 0.0)
        reached.append(float(distance is not None))
    return costs + reached


def _window_search(game_state, action, contested, base_danger,
                   base_bombs, enemy_bomb=None, blast_cache=None):
    field = game_state["field"]
    origin = tuple(game_state["self"][3])
    start = _destination(game_state, action)
    own_bomb = action == "BOMB"
    danger = base_danger
    bombs = base_bombs
    if enemy_bomb is not None:
        danger = dict(base_danger)
        for tile in blast_cache(enemy_bomb):
            updated = set(danger.get(tile, ()))
            updated.update(range(
                s.BOMB_TIMER + 1,
                s.BOMB_TIMER + 1 + s.EXPLOSION_TIMER,
            ))
            danger[tile] = updated
        bombs = base_bombs + ((enemy_bomb, s.BOMB_TIMER + 1, False),)

    left_origin = start != origin
    if (1 in danger.get(start, ()) or not _tile_open(
            field, start, 1, origin, left_origin, bombs)):
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
        for dx, dy in _MOVES:
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


def _compact_opponent_features(game_state, context):
    field = game_state["field"]
    board_area = max(1, int(field.shape[0] * field.shape[1]))
    enemy_bombs = _immediate_enemy_bombs(game_state)
    contested = _opponent_reachability(game_state, OPPONENT_WINDOW)
    first_contested = contested.get(1, frozenset())
    legal = set(context.legal_indices())
    values = np.empty(len(ACTIONS) * OPPONENT_BLOCK_SIZE, dtype=np.float32)
    origin = tuple(game_state["self"][3])
    enemy_flag = float(bool(enemy_bombs))

    def cached_blast(position):
        key = tuple(position)
        cache = getattr(context, "_blast_cache", None)
        if cache is None:
            cache = context._blast_cache = {}
        if key not in cache:
            cache[key] = tuple(blast_tiles(field, key))
        return cache[key]

    cursor = 0
    for action_index, action in enumerate(ACTIONS):
        if action_index not in legal:
            values[cursor:cursor + 4] = 0.0
            cursor += 4
            continue
        base_danger = context.danger(action == "BOMB")
        base_bombs = tuple(
            (tuple(position), int(timer) + 1, False)
            for position, timer in game_state["bombs"]
        )
        if action == "BOMB":
            base_bombs += ((origin, s.BOMB_TIMER + 1, True),)
        destination = _destination(game_state, action)
        destination_contested = float(
            destination != origin and destination in first_contested
        )
        baseline_time, baseline_frontier = _window_search(
            game_state, action, contested, base_danger, base_bombs,
            blast_cache=cached_blast,
        )
        response_stats = [
            _window_search(
                game_state, action, contested, base_danger, base_bombs,
                enemy_bomb=bomb, blast_cache=cached_blast,
            )
            for bomb in enemy_bombs
        ]
        worst_time = min([baseline_time] + [item[0] for item in response_stats])
        worst_frontier = min(
            [baseline_frontier] + [item[1] for item in response_stats]
        )
        values[cursor:cursor + 4] = (
            destination_contested,
            min(1.0, worst_time / float(OPPONENT_WINDOW)),
            min(1.0, worst_frontier / float(board_area)),
            float(bool(response_stats) and worst_time < baseline_time),
        )
        cursor += 4
    return enemy_flag, values


def _combat_block(game_state, context):
    """Agent 039's first 29 values, with cached safety primitives."""
    field = game_state["field"]
    position = tuple(game_state["self"][3])
    safe = set(context.safe_indices())
    legal = set(context.legal_indices())
    blocked = tuple(int(index not in legal) for index in range(4))
    safe_moves = tuple(int(index in safe) for index in range(4))

    danger = context.danger(False)
    local_positions = [position]
    for action in _MOVE_ACTIONS:
        dx, dy = MOVE_DELTAS[action]
        local_positions.append((position[0] + dx, position[1] + dy))
    local_danger = []
    for tile in local_positions:
        times = danger.get(tile, ())
        local_danger.append(danger_bucket(min(times) if times else 0))

    coin, coin_distance = nearest_target(field, position, game_state["coins"])
    crate, crate_distance = nearest_target(
        field, position, crate_approach_tiles(field)
    )
    opponent, opponent_distance = _nearest_opponent(
        position, game_state["others"]
    )
    coin_direction = _direction(position, coin)
    crate_direction = _direction(position, crate)
    opponent_direction = _direction(position, opponent)
    crates_hit, opponents_hit = bomb_value(game_state)
    return np.asarray(
        blocked + safe_moves + tuple(local_danger) +
        (int(game_state["self"][2]),) + coin_direction +
        (distance_bucket(coin_distance),) + crate_direction +
        (distance_bucket(crate_distance),) + opponent_direction +
        (distance_bucket(opponent_distance),) +
        (min(crates_hit, 4), min(opponents_hit, 3),
         distance_bucket(context.escape_distance()),
         count_bucket(int(np.count_nonzero(field == 1))),
         count_bucket(len(game_state["coins"])),
         min(len(game_state["others"]), 3)),
        dtype=np.float32,
    )


def _nearest_opponent(position, others):
    if not others:
        return None, None
    locations = [tuple(other[3]) for other in others]
    target = min(
        locations,
        key=lambda point: abs(point[0] - position[0]) +
        abs(point[1] - position[1]),
    )
    return target, abs(target[0] - position[0]) + abs(target[1] - position[1])


def any_armed_opponent(game_state):
    return float(any(bool(other[2]) for other in game_state["others"]))


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0, action_history=(),
                      action_successes=(), position_history=(),
                      context=None):
    if game_state is None:
        return None
    if context is None:
        context = StateContext(game_state)
    cache_key = (
        previous_action, int(recent_visits), int(steps_since_progress),
        tuple(action_history), tuple(action_successes),
        tuple(tuple(position) for position in position_history),
    )
    cached = context.feature(cache_key)
    if cached is not None:
        return cached

    result = np.empty(FEATURE_SIZE, dtype=np.float32)
    result[0:29] = _combat_block(game_state, context)
    result[29] = min(int(recent_visits), 3)
    result[30] = stagnation_bucket(int(steps_since_progress))

    topology = _compact_topology_features(game_state)
    result[31:35] = topology[0:4]
    result[35:39] = topology[4:8]
    result[39:44] = topology[8:13]
    result[44:63] = short_cycle_features(
        action_history, action_successes, position_history
    )

    enemy_flag, opponent_values = context.opponent()
    result[63] = enemy_flag
    result[64:88] = opponent_values
    result[88:104] = context.routes()
    if result.shape != (FEATURE_SIZE,) or not np.isfinite(result).all():
        raise AssertionError("Invalid Agent 040 compact feature vector")
    context.store_feature(cache_key, result)
    return result


__all__ = [
    "ACTIONS", "AGENT_039_FEATURE_SIZE", "FEATURE_SCHEMA", "FEATURE_SIZE",
    "GLOBAL_ARMED_OPPONENT_INDEX", "OPPONENT_BLOCK_SIZE",
    "OPPONENT_BLOCK_START", "PATCH_OFFSETS", "PATCH_START",
    "ROUTE_FEATURE_SIZE", "ROUTE_STARTS", "SHORT_CYCLE_STARTS",
    "StateContext", "any_armed_opponent", "candidate_crate_tiles",
    "state_to_features",
]
