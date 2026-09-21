

                  

from collections import defaultdict, deque

import settings as s


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
MOVE_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "WAIT": (0, 0),
    "BOMB": (0, 0),
}

                                                                             
                                                                            
BOMB_ESCAPE_MARGIN = 1


def blast_tiles(field, position, power=None):
    power = s.BOMB_POWER if power is None else power
    x, y = position
    tiles = [(x, y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, power + 1):
            tile = (x + dx * distance, y + dy * distance)
            if field[tile] == -1:
                break
            tiles.append(tile)
    return tiles


def _bombs_with_times(game_state, extra_bomb=False):
    bombs = [(tuple(position), int(timer) + 1)
             for position, timer in game_state["bombs"]]
    if extra_bomb:
                                                                            
        bombs.append((tuple(game_state["self"][3]), s.BOMB_TIMER + 1))
    return bombs


def build_danger_schedule(game_state, extra_bomb=False):
    danger = defaultdict(set)
    explosion_map = game_state["explosion_map"]
    for x, y in zip(*((explosion_map > 0).nonzero())):
        for time_step in range(1, int(explosion_map[x, y]) + 1):
            danger[(int(x), int(y))].add(time_step)

    for position, explosion_step in _bombs_with_times(game_state, extra_bomb):
        for tile in blast_tiles(game_state["field"], position):
            for time_step in range(explosion_step,
                                   explosion_step + s.EXPLOSION_TIMER):
                danger[tile].add(time_step)
    return dict(danger)


def earliest_danger(game_state, position, extra_bomb=False):
    times = build_danger_schedule(game_state, extra_bomb).get(tuple(position), ())
    return min(times) if times else 0


def _inside(field, position):
    x, y = position
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def _movement_destination(game_state, action):
    x, y = game_state["self"][3]
    dx, dy = MOVE_DELTAS[action]
    return x + dx, y + dy


def action_is_legal(game_state, action):
    if action not in ACTIONS:
        return False
    if action == "BOMB":
        return bool(game_state["self"][2])
    if action == "WAIT":
        return True

    destination = _movement_destination(game_state, action)
    field = game_state["field"]
    if not _inside(field, destination) or field[destination] != 0:
        return False
    occupied = {tuple(position) for position, _ in game_state["bombs"]}
    occupied.update(tuple(other[3]) for other in game_state["others"])
    return destination not in occupied


def legal_action_indices(game_state):
    return [index for index, action in enumerate(ACTIONS)
            if action_is_legal(game_state, action)]


def bomb_is_useful(game_state):
    if not game_state["self"][2]:
        return False
    field = game_state["field"]
    affected = set(blast_tiles(field, tuple(game_state["self"][3])))
    if any(field[tile] == 1 for tile in affected):
        return True
    return any(tuple(other[3]) in affected for other in game_state["others"])


def bomb_value(game_state):
    field = game_state["field"]
    affected = set(blast_tiles(field, tuple(game_state["self"][3])))
    crates = sum(field[tile] == 1 for tile in affected)
    opponents = sum(tuple(other[3]) in affected for other in game_state["others"])
    return int(crates), int(opponents)


def _opponent_reachability(game_state, horizon):
    field = game_state["field"]
    bomb_positions = {tuple(position) for position, _ in game_state["bombs"]}
    reachable = {tuple(other[3]) for other in game_state["others"]}
    schedule = {}
    for time_step in range(1, horizon + 1):
        following = set(reachable)
        for x, y in reachable:
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                position = x + dx, y + dy
                if (_inside(field, position) and field[position] == 0 and
                        position not in bomb_positions):
                    following.add(position)
        reachable = following
        schedule[time_step] = frozenset(reachable)
    return schedule


def _tile_open_at(game_state, position, time_step, origin, left_origin,
                  bombs, opponent_reachability=None):
    field = game_state["field"]
    if not _inside(field, position) or field[position] != 0:
        return False
    for bomb_position, explosion_step in bombs:
        if position != bomb_position or time_step >= explosion_step:
            continue
                                                                              
        if position == origin and not left_origin:
            continue
        return False
    if (opponent_reachability is not None and
            position in opponent_reachability.get(time_step, ())):
                                                                             
                                                                        
        if not (time_step == 1 and position == origin and not left_origin):
            return False
    return True


def _search_after_action(game_state, action):
    extra_bomb = action == "BOMB"
    danger = build_danger_schedule(game_state, extra_bomb)
    bombs = _bombs_with_times(game_state, extra_bomb)
    origin = tuple(game_state["self"][3])
    start = _movement_destination(game_state, action)
    left_origin = start != origin

    latest_danger = max((max(times) for times in danger.values()), default=1)
    horizon = latest_danger + 1
    own_blast = (set(blast_tiles(game_state["field"], origin))
                 if extra_bomb else set())
    escape_margin = BOMB_ESCAPE_MARGIN if game_state["others"] else 0
    escape_deadline = (
        s.BOMB_TIMER + 1 - escape_margin if extra_bomb else None
    )
    opponent_reachability = (
        _opponent_reachability(game_state, escape_deadline)
        if extra_bomb and game_state["others"] else None
    )

    if 1 in danger.get(start, ()):
        return False, 0, None
    if not _tile_open_at(
            game_state, start, 1, origin, left_origin, bombs,
            opponent_reachability):
        return False, 0, None

    escaped_at = 1 if extra_bomb and start not in own_blast else None
    queue = deque([(start, 1, left_origin, escaped_at)])
    visited = {(start, 1, left_origin, escaped_at)}
    max_time = 1
    while queue:
        position, time_step, has_left, escaped_at = queue.popleft()
        max_time = max(max_time, time_step)
        if time_step >= horizon:
            if not extra_bomb or escaped_at is not None:
                return True, max_time, (
                    escaped_at - 1 if escaped_at is not None else None
                )
                                                                       
                                                                          
            continue

        x, y = position
        for dx, dy in ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)):
            next_position = (x + dx, y + dy)
            next_time = time_step + 1
            next_left = has_left or next_position != origin
            if not _tile_open_at(game_state, next_position, next_time,
                                 origin, next_left, bombs,
                                 opponent_reachability
                                 if escaped_at is None else None):
                continue
            if next_time in danger.get(next_position, ()):
                continue
            next_escaped_at = escaped_at
            if (extra_bomb and next_escaped_at is None and
                    next_position not in own_blast and
                    next_time <= escape_deadline):
                next_escaped_at = next_time
            node = (next_position, next_time, next_left, next_escaped_at)
            if node not in visited:
                visited.add(node)
                queue.append(node)
    return False, max_time, None


def can_survive_action(game_state, action):
    if not action_is_legal(game_state, action):
        return False
    survived, _, _ = _search_after_action(game_state, action)
    return survived


def action_survival_time(game_state, action):
    if not action_is_legal(game_state, action):
        return -1
    _, max_time, _ = _search_after_action(game_state, action)
    return max_time


def escape_distance_after_bomb(game_state):
    if not action_is_legal(game_state, "BOMB"):
        return 0
    survived, _, distance = _search_after_action(game_state, "BOMB")
    return int(distance or 0) if survived else 0


def safe_action_indices(game_state):
    safe = []
    for index, action in enumerate(ACTIONS):
        if action == "BOMB" and not bomb_is_useful(game_state):
            continue
        if can_survive_action(game_state, action):
            safe.append(index)
    return safe


def best_survival_action_indices(game_state):
    legal = legal_action_indices(game_state)
    if not legal:
        return [ACTIONS.index("WAIT")]
    times = [action_survival_time(game_state, ACTIONS[index]) for index in legal]
    best = max(times)
    return [index for index, time_step in zip(legal, times) if time_step == best]

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
MOVE_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "WAIT": (0, 0),
    "BOMB": (0, 0),
}

                                                                             
                                                                            
BOMB_ESCAPE_MARGIN = 1

                                                                  
                                                                           
                                          
ROBUST_REQUIRE_USEFUL_BOMB = True


def blast_tiles(field, position, power=None):
    power = s.BOMB_POWER if power is None else power
    x, y = position
    tiles = [(x, y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, power + 1):
            tile = (x + dx * distance, y + dy * distance)
            if field[tile] == -1:
                break
            tiles.append(tile)
    return tiles


def _bombs_with_times(game_state, extra_bomb=False):
    bombs = [(tuple(position), int(timer) + 1)
             for position, timer in game_state["bombs"]]
    if extra_bomb:
                                                                            
        bombs.append((tuple(game_state["self"][3]), s.BOMB_TIMER + 1))
    return bombs


def build_danger_schedule(game_state, extra_bomb=False):
    danger = defaultdict(set)
    explosion_map = game_state["explosion_map"]
    for x, y in zip(*((explosion_map > 0).nonzero())):
        for time_step in range(1, int(explosion_map[x, y]) + 1):
            danger[(int(x), int(y))].add(time_step)

    for position, explosion_step in _bombs_with_times(game_state, extra_bomb):
        for tile in blast_tiles(game_state["field"], position):
            for time_step in range(explosion_step,
                                   explosion_step + s.EXPLOSION_TIMER):
                danger[tile].add(time_step)
    return dict(danger)


def earliest_danger(game_state, position, extra_bomb=False):
    times = build_danger_schedule(game_state, extra_bomb).get(tuple(position), ())
    return min(times) if times else 0


def _inside(field, position):
    x, y = position
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def _movement_destination(game_state, action):
    x, y = game_state["self"][3]
    dx, dy = MOVE_DELTAS[action]
    return x + dx, y + dy


def action_is_legal(game_state, action):
    if action not in ACTIONS:
        return False
    if action == "BOMB":
        return bool(game_state["self"][2])
    if action == "WAIT":
        return True

    destination = _movement_destination(game_state, action)
    field = game_state["field"]
    if not _inside(field, destination) or field[destination] != 0:
        return False
    occupied = {tuple(position) for position, _ in game_state["bombs"]}
    occupied.update(tuple(other[3]) for other in game_state["others"])
    return destination not in occupied


def legal_action_indices(game_state):
    return [index for index, action in enumerate(ACTIONS)
            if action_is_legal(game_state, action)]


def bomb_is_useful(game_state):
    if not game_state["self"][2]:
        return False
    field = game_state["field"]
    affected = set(blast_tiles(field, tuple(game_state["self"][3])))
    if any(field[tile] == 1 for tile in affected):
        return True
    return any(tuple(other[3]) in affected for other in game_state["others"])


def bomb_value(game_state):
    field = game_state["field"]
    affected = set(blast_tiles(field, tuple(game_state["self"][3])))
    crates = sum(field[tile] == 1 for tile in affected)
    opponents = sum(tuple(other[3]) in affected for other in game_state["others"])
    return int(crates), int(opponents)


def _opponent_reachability(game_state, horizon):
    field = game_state["field"]
    bomb_positions = {tuple(position) for position, _ in game_state["bombs"]}
    reachable = {tuple(other[3]) for other in game_state["others"]}
    schedule = {}
    for time_step in range(1, horizon + 1):
        following = set(reachable)
        for x, y in reachable:
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                position = x + dx, y + dy
                if (_inside(field, position) and field[position] == 0 and
                        position not in bomb_positions):
                    following.add(position)
        reachable = following
        schedule[time_step] = frozenset(reachable)
    return schedule


def _tile_open_at(game_state, position, time_step, origin, left_origin,
                  bombs, opponent_reachability=None):
    field = game_state["field"]
    if not _inside(field, position) or field[position] != 0:
        return False
    for bomb_position, explosion_step in bombs:
        if position != bomb_position or time_step >= explosion_step:
            continue
                                                                              
        if position == origin and not left_origin:
            continue
        return False
    if (opponent_reachability is not None and
            position in opponent_reachability.get(time_step, ())):
                                                                             
                                                                        
        if not (time_step == 1 and position == origin and not left_origin):
            return False
    return True


def _search_after_action_details(game_state, action):
    extra_bomb = action == "BOMB"
    danger = build_danger_schedule(game_state, extra_bomb)
    bombs = _bombs_with_times(game_state, extra_bomb)
    origin = tuple(game_state["self"][3])
    start = _movement_destination(game_state, action)
    left_origin = start != origin

    latest_danger = max((max(times) for times in danger.values()), default=1)
    horizon = latest_danger + 1
    own_blast = (set(blast_tiles(game_state["field"], origin))
                 if extra_bomb else set())
    escape_margin = BOMB_ESCAPE_MARGIN if game_state["others"] else 0
    escape_deadline = (
        s.BOMB_TIMER + 1 - escape_margin if extra_bomb else None
    )
    opponent_reachability = (
        _opponent_reachability(game_state, escape_deadline)
        if extra_bomb and game_state["others"] else None
    )

    no_followups = frozenset()
    if 1 in danger.get(start, ()):
        return False, 0, None, no_followups
    if not _tile_open_at(
            game_state, start, 1, origin, left_origin, bombs,
            opponent_reachability):
        return False, 0, None, no_followups

    escaped_at = 1 if extra_bomb and start not in own_blast else None
    queue = deque([(start, 1, left_origin, escaped_at, None)])
    visited = {(start, 1, left_origin, escaped_at, None)}
    max_time = 1
    surviving_followups = set()
    first_distance = None
    while queue:
        position, time_step, has_left, escaped_at, first_followup = queue.popleft()
        max_time = max(max_time, time_step)
        if time_step >= horizon:
            if not extra_bomb:
                return True, max_time, None, no_followups
            if escaped_at is not None:
                if first_followup is not None:
                    surviving_followups.add(first_followup)
                if first_distance is None and escaped_at is not None:
                    first_distance = escaped_at - 1
                                                                             
                                                                            
                continue
                                                                       
                                                                          
            continue

        x, y = position
        for next_action in ACTIONS[:5]:
            dx, dy = MOVE_DELTAS[next_action]
            next_position = (x + dx, y + dy)
            next_time = time_step + 1
            next_left = has_left or next_position != origin
            if not _tile_open_at(game_state, next_position, next_time,
                                 origin, next_left, bombs,
                                 opponent_reachability
                                 if escaped_at is None else None):
                continue
            if next_time in danger.get(next_position, ()):
                continue
            next_escaped_at = escaped_at
            if (extra_bomb and next_escaped_at is None and
                    next_position not in own_blast and
                    next_time <= escape_deadline):
                next_escaped_at = next_time
            next_followup = (next_action if first_followup is None
                             else first_followup)
            node = (next_position, next_time, next_left, next_escaped_at,
                    next_followup)
            if node not in visited:
                visited.add(node)
                queue.append(node)
    if surviving_followups:
        return True, max_time, first_distance, frozenset(surviving_followups)
    return False, max_time, None, no_followups


def _search_after_action(game_state, action):
    survived, max_time, distance, _ = _search_after_action_details(
        game_state, action)
    return survived, max_time, distance


def surviving_followup_actions(game_state, action):
    survived, _, _, followups = _search_after_action_details(game_state, action)
    return followups if survived else frozenset()


def can_survive_action(game_state, action):
    if not action_is_legal(game_state, action):
        return False
    survived, _, _ = _search_after_action(game_state, action)
    return survived


def action_survival_time(game_state, action):
    if not action_is_legal(game_state, action):
        return -1
    _, max_time, _ = _search_after_action(game_state, action)
    return max_time


def escape_distance_after_bomb(game_state):
    if not action_is_legal(game_state, "BOMB"):
        return 0
    survived, _, distance = _search_after_action(game_state, "BOMB")
    return int(distance or 0) if survived else 0


def safe_action_indices(game_state, require_useful_bomb=ROBUST_REQUIRE_USEFUL_BOMB):
    safe = []
    for index, action in enumerate(ACTIONS):
        if (action == "BOMB" and require_useful_bomb and
                not bomb_is_useful(game_state)):
            continue
        if can_survive_action(game_state, action):
            safe.append(index)
    return safe


def best_survival_action_indices(game_state):
    legal = legal_action_indices(game_state)
    if not legal:
        return [ACTIONS.index("WAIT")]
    times = [action_survival_time(game_state, ACTIONS[index]) for index in legal]
    best = max(times)
    return [index for index, time_step in zip(legal, times) if time_step == best]


def allowed_action_indices(game_state,
                           require_useful_bomb=ROBUST_REQUIRE_USEFUL_BOMB):
    safe = safe_action_indices(game_state, require_useful_bomb)
    return safe if safe else best_survival_action_indices(game_state)


