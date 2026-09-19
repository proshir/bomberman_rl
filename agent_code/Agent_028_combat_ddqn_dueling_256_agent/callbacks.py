"""Agent 027 control logic with Agent 028's dueling network."""

from pathlib import Path

from agent_code.Agent_027_combat_ddqn_short_cycle_staged_replay_agent import (
    callbacks as _agent027,
)

from .model import DuelingQNetwork


MODEL_PATH = Path(__file__).resolve().parent / "dueling_256_checkpoint.pt"
RESUME_PATH = None


def setup(self):
    # Preserve an evaluation runner's explicit --model-path while supplying a
    # unique default for standalone training.
    if not hasattr(self, "model_path"):
        self.model_path = MODEL_PATH
    self.network_class = DuelingQNetwork
    return _agent027.setup(self)


act = _agent027.act
next_features = _agent027.next_features
progress_signature = _agent027.progress_signature
save_checkpoint = _agent027.save_checkpoint
state_key = _agent027.state_key

__all__ = [
    "MODEL_PATH", "RESUME_PATH", "act", "next_features", "progress_signature",
    "save_checkpoint", "setup", "state_key",
]
