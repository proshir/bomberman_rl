"""Reuse the repaired DQN replay, masking, and optimization callbacks."""

# Sahand was here.

from .dep_combat_dqn_agent_train import (  # noqa: F401
    end_of_round,
    game_events_occurred,
    optimize_model,
    remember,
    setup_training,
)

# Re-export the resolved recipe for experiment provenance.
ALGORITHM = "ddqn"
from .dep_combat_dqn_agent_config import (  # noqa: E402,F401
    BATCH_SIZE,
    EPSILON_DECAY_STEPS,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    GRADIENT_CLIP_NORM,
    HIDDEN_SIZE,
    LEARNING_RATE,
    N_ACTIONS,
    REPLAY_CAPACITY,
    TARGET_UPDATE_EVERY,
    TRAIN_EVERY,
    WARMUP_TRANSITIONS,
)
