"""Agent 025 DDQN optimization with scenario-balanced replay."""

from collections import defaultdict

from agent_code.combat_dqn_agent import train as _base
from agent_code.combat_dqn_r_topology_agent.train import (  # noqa: F401
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
)

from .replay import ScenarioReplayBuffer


def setup_training(self):
    self.replay_buffer = ScenarioReplayBuffer(
        REPLAY_CAPACITY, getattr(self, "seed", None)
    )
    self.replay_buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
    self.pending = None
    self.round_reward = 0.0
    self.round_steps = 0
    self.round_events = defaultdict(int)
    self.round_loss_total = 0.0
    self.round_loss_count = 0
    self.last_round_average_loss = None


__all__ = [
    "ALGORITHM", "BATCH_SIZE", "EPSILON_DECAY_STEPS", "EPSILON_END",
    "EPSILON_START", "GAMMA", "GRADIENT_CLIP_NORM", "HIDDEN_SIZE",
    "LEARNING_RATE", "N_ACTIONS", "REPLAY_CAPACITY", "TARGET_UPDATE_EVERY",
    "TRAIN_EVERY", "WARMUP_TRANSITIONS", "end_of_round",
    "game_events_occurred", "optimize_model", "remember", "setup_training",
]
