"""Resolved defaults shared by deployment and the training callbacks."""

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
N_ACTIONS = len(ACTIONS)
BOARD_SIZE = 17
SPATIAL_CHANNELS = 20
GLOBAL_FEATURES = 36
ACTION_FEATURES = 16
FEATURE_SIZE = SPATIAL_CHANNELS * BOARD_SIZE * BOARD_SIZE + GLOBAL_FEATURES + N_ACTIONS * ACTION_FEATURES

# Conservative CPU deployment model. Training scripts may override these in a
# resolved config, but checkpoints carry their own architecture metadata.
TRUNK_WIDTH = 96
RESIDUAL_BLOCKS = 6
QUANTILES = 51
GAMMA = 0.99
N_STEP = 3
CHECKPOINT_NAME = "spatial_hybrid_rainbow.pt"
