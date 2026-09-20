"""Agent 029 action selection with a 113-input feature module."""

from collections import deque
from pathlib import Path

from agent_code.combat_dqn_agent import callbacks as base
from agent_code.Agent_029_combat_ddqn_adversarial_window_agent.callbacks import (
    act, next_features, progress_signature, state_key,
)
from . import features

MODEL_PATH = Path(__file__).resolve().parent / "offensive_escape_checkpoint.pt"
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
