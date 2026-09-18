"""Small, deterministic safety checks used by the combat agent.

The learner decides what is valuable.  This module only removes actions that
are physically impossible or for which we cannot find a route through the
known bomb schedule.  Keeping these rules separate makes them easy to test.
"""

# Sahand was here.

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


def blast_tiles(field, position, power=None):
    """Return the tiles hit by a bomb at ``position``.

    This follows the supplied framework: stone walls stop a blast.  The
    framework destroys crates in the blast but does not stop the ray at the
    first crate, so the safety model deliberately mirrors that behaviour.
    """
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
    """Return ``(position, explosion_step)`` pairs relative to the next move."""
    bombs = [(tuple(position), int(timer) + 1)
             for position, timer in game_state["bombs"]]
    if extra_bomb:
        # A bomb placed by the next action is counted down during that step.
        bombs.append((tuple(game_state["self"][3]), s.BOMB_TIMER + 1))
    return bombs


def build_danger_schedule(game_state, extra_bomb=False):
    """Map each tile to the future steps on which it is explosive."""
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
    """Return the first known dangerous future step, or zero if none exists."""
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
    """Check the rules visible in the current state, without future safety."""
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
    """Return all actions allowed by the immediate game rules."""
    return [index for index, action in enumerate(ACTIONS)
            if action_is_legal(game_state, action)]


def bomb_is_useful(game_state):
    """Check whether a bomb here can hit at least one crate or opponent."""
    if not game_state["self"][2]:
        return False
    field = game_state["field"]
    affected = set(blast_tiles(field, tuple(game_state["self"][3])))
    if any(field[tile] == 1 for tile in affected):
        return True
    return any(tuple(other[3]) in affected for other in game_state["others"])


def bomb_value(game_state):
    """Return the numbers of crates and opponents hit by a bomb placed here."""
    field = game_state["field"]
    affected = set(blast_tiles(field, tuple(game_state["self"][3])))
    crates = sum(field[tile] == 1 for tile in affected)
    opponents = sum(tuple(other[3]) in affected for other in game_state["others"])
    return int(crates), int(opponents)


def _tile_open_at(game_state, position, time_step, origin, left_origin,
                  bombs):
    """Check static obstacles and bombs for one node of the escape search."""
    field = game_state["field"]
    if not _inside(field, position) or field[position] != 0:
        return False
    for bomb_position, explosion_step in bombs:
        if position != bomb_position or time_step >= explosion_step:
            continue
        # An agent may remain on the bomb it just placed, but cannot re-enter.
        if position == origin and not left_origin:
            continue
        return False
    return True


def _search_after_action(game_state, action):
    """Search future position/time states after committing to one action."""
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

    if 1 in danger.get(start, ()):
        return False, 0, None

    queue = deque([(start, 1, left_origin)])
    visited = {(start, 1, left_origin)}
    max_time = 1
    escape_steps = None
    while queue:
        position, time_step, has_left = queue.popleft()
        max_time = max(max_time, time_step)
        if extra_bomb and position not in own_blast and escape_steps is None:
            escape_steps = time_step - 1
        if time_step >= horizon:
            return True, max_time, escape_steps

        x, y = position
        for dx, dy in ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)):
            next_position = (x + dx, y + dy)
            next_time = time_step + 1
            next_left = has_left or next_position != origin
            if not _tile_open_at(game_state, next_position, next_time,
                                 origin, next_left, bombs):
                continue
            if next_time in danger.get(next_position, ()):
                continue
            node = (next_position, next_time, next_left)
            if node not in visited:
                visited.add(node)
                queue.append(node)
    return False, max_time, escape_steps


def can_survive_action(game_state, action):
    """Return whether a route survives every currently predictable explosion."""
    if not action_is_legal(game_state, action):
        return False
    survived, _, _ = _search_after_action(game_state, action)
    return survived


def action_survival_time(game_state, action):
    """Return the furthest safe future step found after an action."""
    if not action_is_legal(game_state, action):
        return -1
    _, max_time, _ = _search_after_action(game_state, action)
    return max_time


def escape_distance_after_bomb(game_state):
    """Return moves needed to leave the new bomb's blast, or zero if impossible."""
    if not action_is_legal(game_state, "BOMB"):
        return 0
    survived, _, distance = _search_after_action(game_state, "BOMB")
    return int(distance or 0) if survived else 0


def safe_action_indices(game_state):
    """Return useful actions with a complete route through the known danger."""
    safe = []
    for index, action in enumerate(ACTIONS):
        if action == "BOMB" and not bomb_is_useful(game_state):
            continue
        if can_survive_action(game_state, action):
            safe.append(index)
    return safe


def best_survival_action_indices(game_state):
    """Fallback used when no action is safe for the full search horizon."""
    legal = legal_action_indices(game_state)
    if not legal:
        return [ACTIONS.index("WAIT")]
    times = [action_survival_time(game_state, ACTIONS[index]) for index in legal]
    best = max(times)
    return [index for index, time_step in zip(legal, times) if time_step == best]
