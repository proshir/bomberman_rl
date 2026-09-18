"""Reuse vanilla DQN training with the 46-dimensional topology features."""

# Sahand was here.

from agent_code.combat_dqn_agent.train import (  # noqa: F401
    end_of_round,
    game_events_occurred,
    optimize_model,
    remember,
    setup_training,
)
