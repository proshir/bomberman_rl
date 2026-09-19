"""Official-framework deployment callbacks; no imports outside this agent."""

from collections import deque
from pathlib import Path
import numpy as np
import torch
from .config import ACTIONS, CHECKPOINT_NAME
from .features import state_to_features, split_features
from .model import SpatialHybridRainbow, load_checkpoint, save_checkpoint as _save_checkpoint
from .safety import best_survival_action_indices, safe_action_indices

MODEL_PATH = Path(__file__).resolve().parent / CHECKPOINT_NAME

def setup(self):
    self.history = deque(maxlen=8); self.successes = deque(maxlen=2); self.visits = deque(maxlen=16)
    self.last_round = None; self.last_progress = None; self.progress_step = 1
    # Training uses the isolated learner GPU when available; submitted games
    # always load on CPU, preserving the official deployment constraint.
    self.device = torch.device("cuda" if self.train and torch.cuda.is_available() else "cpu")
    checkpoint = Path(getattr(self, "model_path", MODEL_PATH))
    self.model_path = checkpoint
    self.feature_cache = {}
    if not checkpoint.exists():
        if not self.train: raise FileNotFoundError(f"Agent_023 requires checkpoint {checkpoint}")
        self.policy_net = SpatialHybridRainbow().to(self.device); return
    self.policy_net, self.checkpoint = load_checkpoint(checkpoint, self.device); self.policy_net.to(self.device); self.policy_net.eval()

def _context(self, state):
    if state["round"] != self.last_round:
        self.last_round = state["round"]; self.history.clear(); self.successes.clear(); self.visits.clear(); self.last_progress = None; self.progress_step = int(state["step"])
    progress = (tuple(sorted(map(tuple, state["coins"]))), int(np.count_nonzero(state["field"] == 1)), len(state["others"]), state["self"][1])
    if progress != self.last_progress: self.last_progress = progress; self.progress_step = int(state["step"])
    vector = state_to_features(state, self.history, self.successes, self.visits, int(state["step"]) - self.progress_step)
    self.feature_cache[(state["round"], state["step"])] = vector
    return vector

def act(self, game_state):
    if game_state is None: return "WAIT"
    candidates = safe_action_indices(game_state) or best_survival_action_indices(game_state)
    vector = _context(self, game_state)
    if self.policy_net is None: choice = candidates[0]
    else:
        spatial, global_values, action_values = split_features(vector)
        with torch.inference_mode():
            q = self.policy_net.q_values(torch.from_numpy(spatial).to(self.device), torch.from_numpy(global_values).to(self.device), torch.from_numpy(action_values).to(self.device))[0].cpu().numpy()
        choice = max(candidates, key=lambda index: (q[index], -index))
    self.history.append(choice); self.successes.append(True); self.visits.append(tuple(game_state["self"][3]))
    return ACTIONS[choice]

def save_checkpoint(learner, path):
    """Runner-compatible checkpoint hook with architecture metadata."""
    _save_checkpoint(path, learner.policy_net, updates=getattr(learner, "updates", 0))
