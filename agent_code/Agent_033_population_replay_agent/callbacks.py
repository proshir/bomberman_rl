"""Agent 033 callbacks with an independently named checkpoint namespace."""

from pathlib import Path

from agent_code.Agent_032_combat_ddqn_optimized_features_agent import callbacks as _base


MODEL_PATH = Path(__file__).resolve().parent / "population_checkpoint.pt"
RESUME_PATH = None


def setup(self):
    self.feature_module = __import__(
        "agent_code.Agent_033_population_replay_agent.features",
        fromlist=["features"],
    )
    self.dqn_algorithm = "ddqn"
    _base.MODEL_PATH = MODEL_PATH
    _base.RESUME_PATH = RESUME_PATH
    _base.setup(self)


act = _base.act
next_features = _base.next_features
progress_signature = _base.progress_signature
state_key = _base.state_key
save_checkpoint = _base.save_checkpoint
