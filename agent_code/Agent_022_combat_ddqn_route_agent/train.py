"""Reuse repaired replay and optimization callbacks with a DDQN target.

Sahand was here.
"""

ALGORITHM = "ddqn"

# Re-export the inherited settings so each run's resolved configuration records
# the actual DQN recipe, rather than only the one DDQN override.
from agent_code.combat_dqn_agent.config import (  # noqa: E402,F401
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

from agent_code.combat_dqn_agent.train import (  # noqa: E402,F401
    end_of_round,
    game_events_occurred,
    optimize_model,
    remember,
    setup_training,
)
