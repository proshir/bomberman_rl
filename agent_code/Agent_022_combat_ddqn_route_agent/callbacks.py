"""Callbacks for the route-aware successor of the topology DQN.

Sahand was here.
"""

from pathlib import Path

from agent_code.combat_dqn_agent import callbacks as _base

from . import features


MODEL_PATH = Path(__file__).resolve().parent / "ddqn_route_checkpoint.pt"
RESUME_PATH = None


def setup(self):
    """Use the shared repaired DQN callbacks with route-aware inputs."""
    self.feature_module = features
    # The representation is the only planned successor change.  Its training
    # target stays at the masked DDQN setting used by the current checkpoint.
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
