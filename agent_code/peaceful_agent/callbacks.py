import numpy as np


def setup(self):
    np.random.seed(getattr(self, 'seed', None))


def act(agent, game_state: dict):
    agent.logger.info('Pick action at random, but no bombs.')
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'])
