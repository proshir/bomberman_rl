"""DDQN callbacks for the target-coverage feature variant."""

# Sahand was here.

from pathlib import Path

from agent_code.combat_dqn_agent import callbacks as _base

from . import features


MODEL_PATH = Path(__file__).resolve().parent / "target_coverage_checkpoint.pt"
RESUME_PATH = None


def setup(self):
    self.feature_module = features
    self.dqn_algorithm = "ddqn"
    _base.MODEL_PATH = MODEL_PATH
    _base.RESUME_PATH = RESUME_PATH
    return _base.setup(self)


act = _base.act
next_features = _base.next_features
save_checkpoint = _base.save_checkpoint
state_key = _base.state_key

__all__ = [
    "MODEL_PATH", "RESUME_PATH", "act", "next_features", "save_checkpoint",
    "setup", "state_key",
]
