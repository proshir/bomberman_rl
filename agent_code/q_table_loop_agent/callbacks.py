"""Tabular Q-learning with legal movement action masking for coin-heaven."""

from collections import deque
from pathlib import Path
import pickle

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
MODEL_PATH = Path(__file__).resolve().parent / 'q_table.pkl'
FEATURE_MODE = 'distance'


def setup_model(self):
    """Start an empty table for training or load a frozen table for evaluation."""
    self.rng = np.random.default_rng(getattr(self, 'seed', None))
    self.model_path = MODEL_PATH if not hasattr(self, 'model_path') else Path(self.model_path)
    if self.train:
        if self.model_path.exists():
            raise FileExistsError(f'Checkpoint already exists: {self.model_path}.')
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.q_table = {}
    else:
        if not self.model_path.is_file():
            raise FileNotFoundError(f'Trained model not found: {self.model_path}')
        with open(self.model_path, 'rb') as file:
            self.q_table = pickle.load(file)


def learned_action(self, game_state: dict) -> str:
    """Explore or exploit only actions that are legal in the current state."""
    if game_state is None:
        return 'WAIT'
    state = state_to_features(game_state)
    q_values = get_q_values(self, state)
    legal = legal_action_indices(game_state)
    if self.train and self.rng.random() < self.epsilon:
        return str(self.rng.choice([ACTIONS[index] for index in legal]))
    best_value = max(q_values[index] for index in legal)
    best = [index for index in legal if q_values[index] == best_value]
    return ACTIONS[int(self.rng.choice(best))]


def legal_action_indices(game_state: dict):
    """Return movement and WAIT indices that do not enter a wall or crate."""
    field = game_state['field']
    x, y = game_state['self'][3]
    neighbours = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]
    legal = []
    for index, (nx, ny) in enumerate(neighbours):
        if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and
                field[nx, ny] == 0):
            legal.append(index)
    legal.append(ACTIONS.index('WAIT'))
    return legal


def state_to_features(game_state: dict):
    """Encode local walls, nearest-coin direction, distance, and progress."""
    if game_state is None:
        return None
    field = game_state['field']
    x, y = game_state['self'][3]
    blocked = (
        int(y - 1 < 0 or field[x, y - 1] != 0),
        int(x + 1 >= field.shape[0] or field[x + 1, y] != 0),
        int(y + 1 >= field.shape[1] or field[x, y + 1] != 0),
        int(x - 1 < 0 or field[x - 1, y] != 0),
    )
    nearest = nearest_coin(field, (x, y), game_state['coins'])
    dx, dy, distance = 0, 0, 0
    if nearest is not None:
        target, distance = nearest
        dx = int(np.sign(target[0] - x))
        dy = int(np.sign(target[1] - y))
    return blocked + (dx, dy, distance_bucket(distance),
                      remaining_coins_bucket(len(game_state['coins'])))


def nearest_coin(field, position, coins):
    """Find a nearest reachable coin and its maze distance by breadth-first search."""
    coins = set(coins)
    if not coins:
        return None
    queue = deque([(position, 0)])
    visited = {position}
    while queue:
        current, distance = queue.popleft()
        if current in coins:
            return current, distance
        x, y = current
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            neighbour = (x + dx, y + dy)
            if (0 <= neighbour[0] < field.shape[0] and
                    0 <= neighbour[1] < field.shape[1] and
                    field[neighbour] == 0 and neighbour not in visited):
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))
    return None


def distance_bucket(distance):
    """Convert maze distance to a small discrete value."""
    if distance == 0:
        return 0
    if distance == 1:
        return 1
    if distance == 2:
        return 2
    if distance <= 4:
        return 3
    if distance <= 7:
        return 4
    return 5


def remaining_coins_bucket(count):
    """Convert remaining coin count to a small progress value."""
    if count == 0:
        return 0
    if count <= 5:
        return 1
    if count <= 15:
        return 2
    if count <= 30:
        return 3
    return 4


def get_q_values(self, state):
    """Return stored values, adding unseen states only during training."""
    if state not in self.q_table:
        if self.train:
            self.q_table[state] = np.zeros(len(ACTIONS))
        else:
            return np.zeros(len(ACTIONS))
    return self.q_table[state]


# Evaluation-only intervention; the learned Q-table is never updated.
HISTORY_LENGTH = 8
REPEAT_THRESHOLD = 3


def setup(self):
    if self.train:
        raise ValueError("The loop variant evaluates frozen checkpoints only.")
    setup_model(self)
    self.loop_rng = np.random.default_rng(getattr(self, "seed", None))
    self.history = deque(maxlen=HISTORY_LENGTH)
    self.last_coins = None
    self.last_round = None
    self.loop_interventions = 0


def act(self, game_state):
    action = learned_action(self, game_state)
    if game_state is None:
        return action
    if game_state["round"] != self.last_round:
        self.history.clear()
        self.last_coins = None
        self.loop_interventions = 0
        self.last_round = game_state["round"]
    coins = tuple(sorted(game_state["coins"]))
    if coins != self.last_coins:
        self.history.clear()
        self.last_coins = coins
    position = game_state["self"][3]
    self.history.append(position)
    if self.history.count(position) < REPEAT_THRESHOLD:
        return action
    alternatives = [ACTIONS[index] for index in legal_action_indices(game_state)
                    if ACTIONS[index] not in ("WAIT", action)]
    if not alternatives:
        return action
    self.loop_interventions += 1
    self.history.clear()
    return str(self.loop_rng.choice(alternatives))
