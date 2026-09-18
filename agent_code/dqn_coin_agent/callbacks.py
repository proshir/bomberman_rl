"""Inference callbacks for the compact-feature coin-navigation DQN.

Sahand was here.

The safety layer here is intentionally small because coin-heaven contains no
bombs or crates.  Movement is still masked against walls, crates, and other
agents, and evaluation always runs on CPU as required by the tournament.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .features import FEATURE_SIZE, state_to_features
from .model import QNetwork


ACTIONS = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT')
MODEL_PATH = Path(__file__).resolve().parent / 'dqn_coin.pt'


def setup(self):
    """Create a fresh training network or load a frozen CPU checkpoint."""
    seed = getattr(self, 'seed', None)
    if seed is not None:
        torch.manual_seed(int(seed))
    self.rng = np.random.default_rng(seed)
    self.model_path = Path(getattr(self, 'model_path', MODEL_PATH))
    # CUDA is useful for batched training only.  Official inference is CPU.
    requested_device = os.environ.get('DQN_DEVICE', 'auto')
    if self.train and requested_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('DQN_DEVICE=cuda but CUDA is not available.')
    use_cuda = (self.train and requested_device != 'cpu'
                and torch.cuda.is_available())
    self.device = torch.device('cuda' if use_cuda else 'cpu')
    self.model = QNetwork(FEATURE_SIZE, len(ACTIONS)).to(self.device)
    if not self.train:
        if not self.model_path.is_file():
            raise FileNotFoundError(f'Trained model not found: {self.model_path}')
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        except TypeError:  # PyTorch versions before the weights_only argument.
            checkpoint = torch.load(self.model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()


def act(self, game_state: dict) -> str:
    """Choose epsilon-greedily during training and greedily during evaluation."""
    if game_state is None:
        return 'WAIT'
    legal = legal_action_indices(game_state)
    if not legal:
        return 'WAIT'
    epsilon = float(getattr(self, 'epsilon', 0.0)) if self.train else 0.0
    if self.train and self.rng.random() < epsilon:
        return ACTIONS[int(self.rng.choice(legal))]
    state = torch.as_tensor(state_to_features(game_state), dtype=torch.float32,
                             device=self.device).unsqueeze(0)
    with torch.no_grad():
        values = self.model(state).squeeze(0).detach().cpu().numpy()
    best_value = max(float(values[index]) for index in legal)
    best = [index for index in legal if float(values[index]) == best_value]
    return ACTIONS[int(self.rng.choice(best))]


def legal_action_indices(game_state: dict) -> list[int]:
    """Return movement actions that do not enter occupied or blocked tiles."""
    field = game_state['field']
    x, y = game_state['self'][3]
    occupied = {tuple(other[3]) for other in game_state.get('others', [])}
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]
    legal = []
    for index, (nx, ny) in enumerate(directions):
        if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                and field[nx, ny] == 0 and (nx, ny) not in occupied):
            legal.append(index)
    legal.append(ACTIONS.index('WAIT'))
    return legal
