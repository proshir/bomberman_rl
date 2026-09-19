"""Auditable 5,912-value rich observation contract.

The board is kept in x,y framework orientation.  ``split_features`` is the
single conversion boundary used by both replay training and deployment.
"""

from collections import deque
import numpy as np
from .config import ACTIONS, ACTION_FEATURES, BOARD_SIZE, FEATURE_SIZE, GLOBAL_FEATURES, SPATIAL_CHANNELS
from .safety import (MOVE_DELTAS, action_is_legal, bomb_value, build_danger_schedule,
                     can_survive_action, escape_distance_after_bomb, safe_action_indices)

def _free(field, tile):
    return 0 <= tile[0] < field.shape[0] and 0 <= tile[1] < field.shape[1] and field[tile] == 0

def _distance_field(field, targets):
    result, queue = {}, deque()
    for target in targets:
        target = tuple(target)
        if _free(field, target): result[target] = 0; queue.append(target)
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((0,-1),(1,0),(0,1),(-1,0)):
            nxt = x + dx, y + dy
            if _free(field, nxt) and nxt not in result:
                result[nxt] = result[(x,y)] + 1; queue.append(nxt)
    return result

def _crate_approaches(field):
    targets = set()
    for x, y in zip(*((field == 1).nonzero())):
        for dx, dy in ((0,-1),(1,0),(0,1),(-1,0)):
            tile = int(x + dx), int(y + dy)
            if _free(field, tile): targets.add(tile)
    return targets

def _component(field, start):
    distances = _distance_field(field, [start])
    return set(distances)

def _array_from_map(mapping, scale=1.0):
    out = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for (x, y), value in mapping.items():
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE: out[x, y] = min(1., float(value) / scale)
    return out

def _action_destination(state, action):
    x, y = state["self"][3]; dx, dy = MOVE_DELTAS[action]
    return x + dx, y + dy

def spatial_tensor(state, recent_visits=(), recent_trail=()):
    """Return the exact 20x17x17 spatial feature bank."""
    field = np.asarray(state["field"]); danger = build_danger_schedule(state)
    board = np.zeros((SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    board[0] = field == -1; board[1] = field == 1
    for x, y in state["coins"]: board[2, x, y] = 1
    x, y = state["self"][3]; board[3, x, y] = 1
    for other in state["others"]: board[4, other[3][0], other[3][1]] = 1
    board[5] = np.clip(np.asarray(state["explosion_map"], dtype=np.float32) / 2., 0., 1.)
    for pos, timer in state["bombs"]:
        if 0 <= timer <= 3: board[6 + int(timer), pos[0], pos[1]] = 1
    earliest = {tile: min(times) for tile, times in danger.items() if times}
    board[10] = _array_from_map(earliest, 5.); board[11] = (board[10] > 0) & (board[10] <= .8)
    component = _component(field, tuple(state["self"][3])); board[13] = _array_from_map({p: 1 for p in component})
    # A tile is marked reachable only if it is in the static component and not
    # lethal on the next action. This is intentionally conservative.
    safe_tiles = {p: 1 for p in component if 1 not in danger.get(p, ())}; board[12] = _array_from_map(safe_tiles)
    coin_dist = _distance_field(field, state["coins"]); crate_dist = _distance_field(field, _crate_approaches(field))
    board[14] = _array_from_map(coin_dist, BOARD_SIZE * 2); board[15] = _array_from_map(crate_dist, BOARD_SIZE * 2)
    for tile in component:
        crates, _ = bomb_value(state, tile); board[16, tile[0], tile[1]] = min(1., crates / 4.)
        if tile == tuple(state["self"][3]): board[17, tile[0], tile[1]] = float(can_survive_action(state, "BOMB"))
    for index, tile in enumerate(reversed(tuple(recent_visits))):
        if 0 <= tile[0] < BOARD_SIZE and 0 <= tile[1] < BOARD_SIZE: board[18, tile[0], tile[1]] += .9 ** index
    for index, tile in enumerate(reversed(tuple(recent_trail))):
        if 0 <= tile[0] < BOARD_SIZE and 0 <= tile[1] < BOARD_SIZE: board[19, tile[0], tile[1]] += .9 ** index
    return np.clip(board, 0., 1.)

def action_features(state, recent_visits=()):
    field = state["field"]; position = tuple(state["self"][3]); danger = build_danger_schedule(state)
    coin_dist = _distance_field(field, state["coins"]); crate_dist = _distance_field(field, _crate_approaches(field))
    values = []
    for action in ACTIONS:
        legal = action_is_legal(state, action); safe = legal and can_survive_action(state, action)
        dest = _action_destination(state, action)
        if not legal: values.extend([0.] * ACTION_FEATURES); continue
        earliest = min(danger.get(dest, (0,))) if danger.get(dest) else 0
        crates, opponents = bomb_value(state, position if action == "BOMB" else dest)
        visit_penalty = sum(tile == dest for tile in recent_visits) / max(1, len(recent_visits))
        coin = coin_dist.get(dest); crate = crate_dist.get(dest)
        values.extend((1., float(safe), earliest / 5., float(len(_component(field, dest))) / 289.,
                       float(escape_distance_after_bomb(state) if action == "BOMB" else 0) / 17.,
                       0. if coin is None else coin / 34., float(coin is not None),
                       0. if crate is None else crate / 34., float(crate is not None),
                       float(dest in set(state["coins"])), min(1., crates / 4.), min(1., crates / 4.),
                       float(action != "BOMB" or can_survive_action(state, "BOMB")), min(1., opponents / 3.),
                       visit_penalty, min(1., (crates * 2 + opponents * 3) / 10.)))
    return np.asarray(values, dtype=np.float32)

def global_features(state, previous_actions=(), action_successes=(), recent_visits=(), steps_since_progress=0):
    pos = tuple(state["self"][3]); field = state["field"]
    coin_dist, crate_dist = _distance_field(field, state["coins"]), _distance_field(field, _crate_approaches(field))
    opponent_dist = [abs(o[3][0]-pos[0]) + abs(o[3][1]-pos[1]) for o in state["others"]]
    base = [float(state["self"][2]), max(0., 400-int(state["step"])) / 400., state["self"][1] / 50.,
            max([o[1] for o in state["others"]], default=0) / 50.,
            (state["self"][1] - max([o[1] for o in state["others"]], default=0)) / 50.,
            len(state["others"]) / 3., len(state["coins"]) / 50., np.count_nonzero(field == 1) / 200.]
    # Previous action: six actions plus an explicit round-start value.  Each
    # of the two older actions is six-way one-hot plus its success bit.
    history = list(previous_actions)[-3:][::-1]
    immediate = [0.] * 7
    immediate[history[0] if history else 6] = 1.
    base.extend(immediate)
    for age in (1, 2):
        onehot = [0.] * 6
        if age < len(history): onehot[history[age]] = 1.
        base.extend(onehot)
        base.append(float(action_successes[-age] if len(action_successes) >= age else 0.))
    extras = [min(1., steps_since_progress / 100.), sum(p == pos for p in recent_visits) / max(1, len(recent_visits)),
              len(_component(field, pos)) / 289., len({p for p in _component(field, pos) if p not in build_danger_schedule(state)}) / 289.,
              coin_dist.get(pos, 34) / 34., crate_dist.get(pos, 34) / 34., min(opponent_dist, default=34) / 34.]
    out = np.asarray(base + extras, dtype=np.float32)
    if out.size != GLOBAL_FEATURES: raise AssertionError(f"global contract expected 36, got {out.size}")
    return out

def state_to_features(state, previous_actions=(), action_successes=(), recent_visits=(), steps_since_progress=0):
    if state is None: return None
    vector = np.concatenate((spatial_tensor(state, recent_visits, recent_visits).reshape(-1),
                             global_features(state, previous_actions, action_successes, recent_visits, steps_since_progress),
                             action_features(state, recent_visits)))
    if vector.size != FEATURE_SIZE: raise AssertionError(f"feature contract expected {FEATURE_SIZE}, got {vector.size}")
    return vector.astype(np.float32, copy=False)

def split_features(batch):
    array = np.asarray(batch, dtype=np.float32); array = array.reshape((-1, FEATURE_SIZE))
    board_end = SPATIAL_CHANNELS * BOARD_SIZE * BOARD_SIZE
    spatial = array[:, :board_end].reshape((-1, SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE))
    global_values = array[:, board_end:board_end + GLOBAL_FEATURES]
    per_action = array[:, board_end + GLOBAL_FEATURES:].reshape((-1, len(ACTIONS), ACTION_FEATURES))
    return spatial, global_values, per_action
