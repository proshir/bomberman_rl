"""Plain tabular Q-learning for coin-heaven, alone, without bombs."""

from collections import deque
from pathlib import Path
import pickle

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
MODEL_PATH = Path(__file__).resolve().parent / 'q_table.pkl'


def setup(self):
    """Start a fresh training table or load a checkpoint for evaluation."""
    self.rng = np.random.default_rng(getattr(self, 'seed', None))
    self.model_path = MODEL_PATH if not hasattr(self, 'model_path') else Path(self.model_path)

    if self.train:
        if self.model_path.exists():
            raise FileExistsError(f'Checkpoint already exists: {self.model_path}. Choose a new model_path for this training run.')
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.q_table = {}
    else:
        if not self.model_path.is_file():
            raise FileNotFoundError(f'Trained model not found: {self.model_path}')
        self.logger.info(f'Loading trained model from {self.model_path}.')
        with open(self.model_path, 'rb') as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    """Explore during training; otherwise maximize Q with random tie-breaking."""
    if game_state is None:
        return 'WAIT'
    state = state_to_features(game_state)
    q_values = get_q_values(self, state)
    if self.train and self.rng.random() < self.epsilon:
        return str(self.rng.choice(ACTIONS))
    max_q = np.max(q_values)
    best_actions = [action for action, q in zip(ACTIONS, q_values) if q == max_q]
    return str(self.rng.choice(best_actions))


def state_to_features(game_state: dict):
    """Encode four blocked neighbours and the signs of the nearest coin's offset.

    Coordinates use field[x, y]. Different boards can share this compact state;
    the representation has no symmetry handling and ignores bombs and opponents.
    """
    if game_state is None:
        return None
    field = game_state['field']
    x, y = game_state['self'][3]
    up_blocked = y - 1 < 0 or field[x, y - 1] != 0
    right_blocked = x + 1 >= field.shape[0] or field[x + 1, y] != 0
    down_blocked = y + 1 >= field.shape[1] or field[x, y + 1] != 0
    left_blocked = x - 1 < 0 or field[x - 1, y] != 0

    nearest = nearest_coin(field, (x, y), game_state['coins'])
    coin_dx, coin_dy = 0, 0
    if nearest is not None:
        coin_dx = int(np.sign(nearest[0] - x))
        coin_dy = int(np.sign(nearest[1] - y))
    return (int(up_blocked), int(right_blocked), int(down_blocked),
            int(left_blocked), coin_dx, coin_dy)


def nearest_coin(field, position, coins):
    """Find the nearest reachable coin by BFS, breaking ties UP, RIGHT, DOWN, LEFT."""
    coins = set(coins)
    if not coins:
        return None
    queue = deque([position])
    visited = {position}
    while queue:
        current = queue.popleft()
        if current in coins:
            return current
        x, y = current
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            neighbour = (x + dx, y + dy)
            if (0 <= neighbour[0] < field.shape[0] and
                    0 <= neighbour[1] < field.shape[1] and
                    field[neighbour] == 0 and neighbour not in visited):
                visited.add(neighbour)
                queue.append(neighbour)
    return None


def get_q_values(self, state):
    """Return stored values, inserting unseen states only during training."""
    if state not in self.q_table:
        if self.train:
            self.q_table[state] = np.zeros(len(ACTIONS))
        else:
            return np.zeros(len(ACTIONS))
    return self.q_table[state]
