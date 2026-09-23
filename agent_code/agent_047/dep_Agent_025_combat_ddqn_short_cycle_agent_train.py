"""Reuse the repaired DDQN replay and optimization recipe."""

# Sahand was here.

from .dep_combat_dqn_r_topology_agent_train import (  # noqa: F401
    ALGORITHM,
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
    end_of_round,
    game_events_occurred,
    optimize_model,
    remember,
    setup_training,
)
