import numpy as np


def setup(self):
    np.random.seed(getattr(self, 'seed', None))


def act(agent, game_state: dict):
    raise ValueError()
