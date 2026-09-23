"""Agent 030 uses Agent 029's inference and feature contract unchanged."""

from pathlib import Path

from . import dep_Agent_029_combat_ddqn_adversarial_window_agent_callbacks as _agent029


MODEL_PATH = Path(__file__).resolve().parent / "combat_escape_replay_checkpoint.pt"
RESUME_PATH = None


def setup(self):
    _agent029.MODEL_PATH = MODEL_PATH
    _agent029.RESUME_PATH = RESUME_PATH
    _agent029.setup(self)


act = _agent029.act
next_features = _agent029.next_features
progress_signature = _agent029.progress_signature
save_checkpoint = _agent029.save_checkpoint
state_key = _agent029.state_key
