"""Agent 033 training constants and callbacks.

The callback implementation is reused, but the replay experiment is isolated
by changing only this package's constants in the staged runtime.
"""

from agent_code.Agent_032_combat_ddqn_optimized_features_agent import train as _base

# Proposed population experiment: same samples per environment step as 032,
# with twice the replay horizon and twice the batch size.
REPLAY_CAPACITY = 200_000
BATCH_SIZE = 256
WARMUP_TRANSITIONS = 10_000
TRAIN_EVERY = 8

# Keep the remaining hyperparameters and callbacks identical to Agent 032.
for _name in (
    "ALGORITHM", "EPSILON_DECAY_STEPS", "EPSILON_END", "EPSILON_START",
    "GAMMA", "GRADIENT_CLIP_NORM", "HIDDEN_SIZE", "LEARNING_RATE",
    "N_ACTIONS", "TARGET_UPDATE_EVERY", "COMBAT_EPSILON_START",
    "COMBAT_EPSILON_END", "COMBAT_EPSILON_DECAY_STEPS",
):
    globals()[_name] = getattr(_base, _name)

# The imported callback functions resolve their module globals at call time;
# these assignments affect only this staged population process.
_base.REPLAY_CAPACITY = REPLAY_CAPACITY
_base.BATCH_SIZE = BATCH_SIZE
_base.WARMUP_TRANSITIONS = WARMUP_TRANSITIONS
_base.TRAIN_EVERY = TRAIN_EVERY

setup_training = _base.setup_training
optimize_model = _base.optimize_model
remember = _base.remember
game_events_occurred = _base.game_events_occurred
end_of_round = _base.end_of_round

__all__ = [
    "ALGORITHM", "BATCH_SIZE", "EPSILON_DECAY_STEPS", "EPSILON_END",
    "EPSILON_START", "GAMMA", "GRADIENT_CLIP_NORM", "HIDDEN_SIZE",
    "LEARNING_RATE", "N_ACTIONS", "REPLAY_CAPACITY", "TARGET_UPDATE_EVERY",
    "TRAIN_EVERY", "WARMUP_TRANSITIONS", "setup_training", "optimize_model",
    "remember", "game_events_occurred", "end_of_round",
]
