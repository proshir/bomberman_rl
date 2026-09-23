"""Agent 027 features plus compact adversarial opponent-window summaries.

The game resolves one action per agent per step.  Enumerating every opponent
action sequence would grow exponentially and would be too expensive for the
0.5-second inference budget.  Instead, for every candidate action we search
our own reachable tiles through one bomb cycle and take the worst result over
each opponent that could place a bomb immediately.  The opponent reachability
envelope also marks future contested escape tiles.  These are learned inputs,
not additional action-mask rules.
"""

from collections import defaultdict, deque

import numpy as np
import settings as s

from .dep_Agent_025_combat_ddqn_short_cycle_agent_features import (
    ACTIONS,
    FEATURE_SIZE as AGENT_027_FEATURE_SIZE,
    SHORT_CYCLE_FEATURES,
    short_cycle_features,
)
from .dep_combat_dqn_r_topology_agent_features import state_to_features as _base_features
from .dep_combat_fqi_agent_safety import (
    MOVE_DELTAS,
    _opponent_reachability,
    action_is_legal,
    blast_tiles,
    build_danger_schedule,
)


# A newly planted bomb explodes after this many future decision steps in the
# shared safety model.  It is long enough to include a complete immediate
# opponent bomb response, but remains deliberately small and CPU-friendly.
OPPONENT_WINDOW = s.BOMB_TIMER + 1
OPPONENT_FEATURES_PER_ACTION = 6
FEATURE_SIZE = AGENT_027_FEATURE_SIZE + len(ACTIONS) * OPPONENT_FEATURES_PER_ACTION


def _inside(field, position):
    x, y = position
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def _destination(game_state, action):
    x, y = game_state["self"][3]
    dx, dy = MOVE_DELTAS[action]
    return int(x + dx), int(y + dy)


def _add_bomb_danger(danger, field, position, explosion_step):
    """Add one hypothetical bomb using the same blast/timing convention."""
    for tile in blast_tiles(field, position):
        for time_step in range(explosion_step,
                               explosion_step + s.EXPLOSION_TIMER):
            danger[tile].add(time_step)


def _immediate_enemy_bombs(game_state):
    """One plausible immediate bomb placement for each armed opponent.

    An enemy which moves first can only plant on its following turn, whose
    explosion is outside this one-bomb-cycle window.  Its possible movement is
    still represented by the contested-tile envelope below.  Keeping only
    immediate placements avoids treating mutually exclusive enemy bombs as if
    they all occur together.
    """
    return tuple(sorted({tuple(other[3]) for other in game_state["others"]
                         if bool(other[2])}))


def _tile_open(field, position, time_step, origin, has_left_origin, bombs):
    if not _inside(field, position) or field[position] != 0:
        return False
    for bomb_position, explosion_step, is_own_new_bomb in bombs:
        if position != bomb_position or time_step >= explosion_step:
            continue
        # Match the existing safety search: we may stand on the bomb just
        # placed this turn, but cannot leave and later re-enter it.
        if (is_own_new_bomb and position == origin and
                not has_left_origin):
            continue
        return False
    return True


def _window_search(game_state, action, enemy_bomb=None):
    """Return deepest safe time and safe frontier size for one response case."""
    if not action_is_legal(game_state, action):
        return 0, 0

    field = game_state["field"]
    origin = tuple(game_state["self"][3])
    start = _destination(game_state, action)
    own_bomb = action == "BOMB"
    danger = defaultdict(set, build_danger_schedule(game_state, own_bomb))
    bombs = [(tuple(position), int(timer) + 1, False)
             for position, timer in game_state["bombs"]]
    if own_bomb:
        bombs.append((origin, s.BOMB_TIMER + 1, True))
    if enemy_bomb is not None:
        bombs.append((enemy_bomb, s.BOMB_TIMER + 1, False))
        _add_bomb_danger(danger, field, enemy_bomb, s.BOMB_TIMER + 1)

    contested = _opponent_reachability(game_state, OPPONENT_WINDOW)
    left_origin = start != origin
    if 1 in danger.get(start, ()) or not _tile_open(
            field, start, 1, origin, left_origin, bombs):
        return 0, 0

    # Simultaneous moves make the candidate destination contestable, but the
    # environment's random action order means a first-step overlap is not a
    # deterministic death.  Record it separately rather than inventing a veto.
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
        for dx, dy in ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)):
            next_position = (x + dx, y + dy)
            next_time = time_step + 1
            next_left = has_left or next_position != origin
            if not _tile_open(field, next_position, next_time, origin,
                              next_left, bombs):
                continue
            if next_time in danger.get(next_position, ()):
                continue
            # Unlike a known blast, a reachable opponent only makes a route
            # contested.  Excluding it after the simultaneous first step gives
            # a compact worst-case escape-envelope estimate.
            if next_position in contested.get(next_time, ()):
                continue
            node = (next_position, next_time, next_left)
            if node not in visited:
                visited.add(node)
                queue.append(node)
    return deepest, len(frontier)


def opponent_window_features(game_state):
    """Return six action-conditioned values for each of the six actions.

    Per action: immediate legality, armed-opponent flag, contestability of the
    immediate destination, worst survival horizon, worst safe-frontier size,
    and whether an immediate enemy bomb shortens that horizon.  All values are
    normalized to [0, 1] except the Boolean flags, which are already 0/1.
    """
    if game_state is None:
        return np.zeros(len(ACTIONS) * OPPONENT_FEATURES_PER_ACTION,
                        dtype=np.float32)

    field = game_state["field"]
    board_area = max(1, int(field.shape[0] * field.shape[1]))
    enemy_bombs = _immediate_enemy_bombs(game_state)
    contested = _opponent_reachability(game_state, 1).get(1, frozenset())
    values = []
    for action in ACTIONS:
        legal = action_is_legal(game_state, action)
        if not legal:
            values.extend((0.0,) * OPPONENT_FEATURES_PER_ACTION)
            continue
        destination = _destination(game_state, action)
        # Waiting/planting retains the current tile, so an opponent cannot
        # claim it before us in the same action-resolution step.
        destination_contested = int(
            destination != tuple(game_state["self"][3]) and
            destination in contested
        )
        baseline_time, baseline_frontier = _window_search(game_state, action)
        response_stats = [_window_search(game_state, action, bomb)
                          for bomb in enemy_bombs]
        worst_time = min([baseline_time] + [item[0] for item in response_stats])
        worst_frontier = min(
            [baseline_frontier] + [item[1] for item in response_stats]
        )
        bomb_trap = int(bool(response_stats) and
                        worst_time < baseline_time)
        values.extend((
            1.0,
            float(bool(enemy_bombs)),
            float(destination_contested),
            min(1.0, worst_time / float(OPPONENT_WINDOW)),
            min(1.0, worst_frontier / float(board_area)),
            float(bomb_trap),
        ))
    return np.asarray(values, dtype=np.float32)


def state_to_features(game_state, previous_action=None, recent_visits=0,
                      steps_since_progress=0, action_history=(),
                      action_successes=(), position_history=()):
    """Keep Agent 027's exact 65 inputs and append this one new feature group."""
    base = _base_features(
        game_state, previous_action, recent_visits, steps_since_progress,
    )
    if base is None:
        return None
    history = short_cycle_features(
        action_history, action_successes, position_history,
    )
    features = np.concatenate((base, history, opponent_window_features(game_state)))
    if len(features) != FEATURE_SIZE:
        raise AssertionError(
            f"Agent 029 feature size changed: expected {FEATURE_SIZE}, "
            f"got {len(features)}"
        )
    return features.astype(np.float32, copy=False)


__all__ = [
    "ACTIONS", "AGENT_027_FEATURE_SIZE", "FEATURE_SIZE",
    "OPPONENT_FEATURES_PER_ACTION", "OPPONENT_WINDOW", "SHORT_CYCLE_FEATURES",
    "opponent_window_features", "short_cycle_features", "state_to_features",
]
