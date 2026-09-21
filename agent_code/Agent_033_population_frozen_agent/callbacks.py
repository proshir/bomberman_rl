"""Non-training wrapper for an Agent 033 population checkpoint."""

from pathlib import Path

from agent_code.Agent_032_combat_ddqn_optimized_features_agent import callbacks as _base
from agent_code.Agent_033_population_replay_agent import features


MODEL_PATH = Path(__file__).resolve().parent / "frozen_population_checkpoint.pt"


def setup(self):
    previous_model = _base.MODEL_PATH
    previous_resume = _base.RESUME_PATH
    try:
        self.feature_module = features
        self.dqn_algorithm = "ddqn"
        _base.MODEL_PATH = MODEL_PATH
        _base.RESUME_PATH = None
        _base.setup(self)
    finally:
        _base.MODEL_PATH = previous_model
        _base.RESUME_PATH = previous_resume


act = _base.act
next_features = _base.next_features
progress_signature = _base.progress_signature
state_key = _base.state_key
