"""Agent 029 action selection with Agent 032's optimized 113-input features."""

from collections import deque
from pathlib import Path

from . import dep_combat_dqn_agent_callbacks as base
from .dep_Agent_029_combat_ddqn_adversarial_window_agent_callbacks import (
    act, next_features, progress_signature, state_key,
)
from . import dep_Agent_032_combat_ddqn_optimized_features_agent_features as features


MODEL_PATH = Path(__file__).resolve().parent / "optimized_features_checkpoint.pt"
RESUME_PATH = None


def setup(self):
    self.feature_module = features
    self.dqn_algorithm = "ddqn"
    base.MODEL_PATH = MODEL_PATH
    base.RESUME_PATH = RESUME_PATH
    base.setup(self)
    self.action_history = deque(maxlen=8)
    self.action_successes = deque(maxlen=8)
    self.last_observed_key = None
    self.last_action_state = None
    self.last_action = None


save_checkpoint = base.save_checkpoint
