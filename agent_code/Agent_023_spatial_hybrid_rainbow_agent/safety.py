"""Deterministic time-expanded safety mask used for behaviour and targets."""

from collections import defaultdict, deque
import settings as s
from .config import ACTIONS

MOVE_DELTAS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1),
               "LEFT": (-1, 0), "WAIT": (0, 0), "BOMB": (0, 0)}

def _inside(field, tile):
    return 0 <= tile[0] < field.shape[0] and 0 <= tile[1] < field.shape[1]

def blast_tiles(field, position, power=None):
    power = s.BOMB_POWER if power is None else power
    out = [tuple(position)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, power + 1):
            tile = position[0] + dx * distance, position[1] + dy * distance
            if not _inside(field, tile) or field[tile] == -1:
                break
            out.append(tile)
    return out

def action_is_legal(state, action):
    if action == "BOMB": return bool(state["self"][2])
    if action == "WAIT": return True
    x, y = state["self"][3]; dx, dy = MOVE_DELTAS[action]; tile = (x + dx, y + dy)
    occupied = {tuple(pos) for pos, _ in state["bombs"]} | {tuple(o[3]) for o in state["others"]}
    return _inside(state["field"], tile) and state["field"][tile] == 0 and tile not in occupied

def bomb_value(state, position=None):
    position = tuple(state["self"][3]) if position is None else tuple(position)
    tiles = blast_tiles(state["field"], position)
    return (sum(state["field"][tile] == 1 for tile in tiles),
            sum(tuple(other[3]) in tiles for other in state["others"]))

def build_danger_schedule(state, extra_bomb=False):
    danger = defaultdict(set)
    for x, y in zip(*((state["explosion_map"] > 0).nonzero())):
        for time in range(1, int(state["explosion_map"][x, y]) + 1): danger[(int(x), int(y))].add(time)
    bombs = [(tuple(pos), int(timer) + 1) for pos, timer in state["bombs"]]
    if extra_bomb: bombs.append((tuple(state["self"][3]), s.BOMB_TIMER + 1))
    for pos, when in bombs:
        for tile in blast_tiles(state["field"], pos):
            for time in range(when, when + s.EXPLOSION_TIMER): danger[tile].add(time)
    return danger

def _survives(state, action):
    if not action_is_legal(state, action): return False, 0, set()
    field, origin = state["field"], tuple(state["self"][3])
    extra = action == "BOMB"; danger = build_danger_schedule(state, extra)
    bomb_positions = {tuple(p) for p, _ in state["bombs"]}
    if extra: bomb_positions.add(origin)
    dx, dy = MOVE_DELTAS[action]; start = (origin[0] + dx, origin[1] + dy)
    horizon = max((max(v) for v in danger.values()), default=1) + 1
    if 1 in danger.get(start, ()) or (start in bomb_positions and start != origin): return False, 0, set()
    queue, seen, reachable = deque([(start, 1, start != origin)]), {(start, 1, start != origin)}, {start}
    while queue:
        pos, time, left_origin = queue.popleft()
        if time >= horizon: return True, time, reachable
        for dx, dy in MOVE_DELTAS.values():
            nxt, nt = (pos[0] + dx, pos[1] + dy), time + 1
            if not _inside(field, nxt) or field[nxt] != 0 or nt in danger.get(nxt, ()): continue
            # A new bomb can be occupied only until the agent has left it.
            if nxt in bomb_positions and not (nxt == origin and not left_origin): continue
            node = (nxt, nt, left_origin or nxt != origin)
            if node not in seen: seen.add(node); queue.append(node); reachable.add(nxt)
    return False, 0, reachable

def can_survive_action(state, action): return _survives(state, action)[0]
def safe_action_indices(state):
    values = []
    for index, action in enumerate(ACTIONS):
        if action == "BOMB" and not any(bomb_value(state)): continue
        if can_survive_action(state, action): values.append(index)
    return values
def best_survival_action_indices(state):
    legal = [i for i, a in enumerate(ACTIONS) if action_is_legal(state, a)] or [4]
    scores = [_survives(state, ACTIONS[i])[1] for i in legal]; best = max(scores)
    return [i for i, score in zip(legal, scores) if score == best]
def escape_distance_after_bomb(state):
    if not action_is_legal(state, "BOMB"): return 0
    origin, blast = tuple(state["self"][3]), set(blast_tiles(state["field"], tuple(state["self"][3])))
    queue, seen = deque([(origin, 0)]), {origin}
    while queue:
        pos, dist = queue.popleft()
        if pos not in blast: return dist
        for dx, dy in list(MOVE_DELTAS.values())[:4]:
            nxt = pos[0] + dx, pos[1] + dy
            if _inside(state["field"], nxt) and state["field"][nxt] == 0 and nxt not in seen:
                seen.add(nxt); queue.append((nxt, dist + 1))
    return 0
