"""Callbacks for the topology-feature vanilla DQN ablation."""

# Sahand was here.

from agent_code.combat_dqn_agent import callbacks as _base

from . import features


MODEL_PATH = _base.MODEL_PATH


def setup(self):
    self.feature_module = features
    _base.MODEL_PATH = MODEL_PATH
    return _base.setup(self)


act = _base.act
next_features = _base.next_features
save_checkpoint = _base.save_checkpoint
state_key = _base.state_key

__all__ = [
    "MODEL_PATH", "act", "next_features", "save_checkpoint", "setup",
    "state_key",
]
