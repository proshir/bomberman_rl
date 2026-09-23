"""DDQN callbacks with Agent 025's exact short-cycle context."""

from collections import deque
from pathlib import Path

import numpy as np
import settings as s
import torch

from . import dep_combat_dqn_agent_callbacks as _base
from .dep_combat_dqn_agent_model import DEVICE
from .dep_combat_fqi_agent_safety import action_is_legal

from . import dep_Agent_027_combat_ddqn_short_cycle_staged_replay_agent_features as features
from .dep_Agent_027_combat_ddqn_short_cycle_staged_replay_agent_features import ACTIONS


MODEL_PATH = Path(__file__).resolve().parent / "staged_replay_checkpoint.pt"
RESUME_PATH = None


def state_key(game_state):
    return int(game_state["round"]), int(game_state["step"])


def progress_signature(game_state):
    return (
        tuple(sorted(tuple(coin) for coin in game_state["coins"])),
        int(np.count_nonzero(game_state["field"] == 1)),
        len(game_state["others"]),
        int(game_state["self"][1]),
    )


def _probe_state():
    field = np.zeros((s.COLS, s.ROWS), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    return {
        "round": 1, "step": 1, "field": field, "bombs": [],
        "explosion_map": np.zeros_like(field), "coins": [(1, 1)],
        "self": ("probe", 0, True, (1, 1)), "others": [],
        "user_input": None,
    }


def _state_to_features(self, game_state, previous_action, recent_visits,
                       steps_since_progress, action_history=(),
                       action_successes=(), position_history=()):
    return self.feature_module.state_to_features(
        game_state, previous_action, recent_visits, steps_since_progress,
        action_history, action_successes, position_history,
    )


def setup(self):
    self.feature_module = features
    self.dqn_algorithm = "ddqn"
    _base.MODEL_PATH = MODEL_PATH
    _base.RESUME_PATH = RESUME_PATH
    _base.setup(self)
    self.action_history = deque(maxlen=8)
    self.action_successes = deque(maxlen=8)
    self.last_observed_key = None
    self.last_action_state = None
    self.last_action = None


def _reset_temporal_context(self, game_state):
    self.positions.clear()
    self.action_history.clear()
    self.action_successes.clear()
    self.last_observed_key = state_key(game_state)
    self.last_action_state = None
    self.last_action = None


def _advance_action_context(self, game_state):
    current_key = state_key(game_state)
    if self.last_observed_key == current_key:
        return
    if self.last_action is not None and self.last_action_state is not None:
        self.action_history.append(self.last_action)
        self.action_successes.append(
            int(action_is_legal(self.last_action_state, ACTIONS[self.last_action]))
        )
    self.last_observed_key = current_key


def _features_for_state(self, game_state):
    if game_state["round"] != self.round_id:
        self.round_id = game_state["round"]
        self.feature_cache.clear()
        self.last_progress = None
        self.last_progress_step = int(game_state["step"])
        self.previous_action = ACTIONS.index("WAIT")
        _reset_temporal_context(self, game_state)
    else:
        _advance_action_context(self, game_state)

    signature = progress_signature(game_state)
    if signature != self.last_progress:
        self.positions.clear()
        self.action_history.clear()
        self.action_successes.clear()
        self.last_progress = signature
        self.last_progress_step = int(game_state["step"])

    position = tuple(game_state["self"][3])
    feature = _state_to_features(
        self, game_state, self.previous_action, self.positions.count(position),
        int(game_state["step"]) - self.last_progress_step,
        tuple(self.action_history), tuple(self.action_successes),
        tuple(self.positions) + (position,),
    )
    self.feature_cache[state_key(game_state)] = (
        feature, tuple(self.positions), tuple(self.action_history),
        tuple(self.action_successes), signature, self.last_progress_step,
    )
    self.positions.append(position)
    return feature


def next_features(self, old_state, action, new_state):
    _, positions, actions, successes, old_progress, last_step = (
        self.feature_cache[state_key(old_state)]
    )
    positions = deque(positions, maxlen=8)
    positions.append(tuple(old_state["self"][3]))
    actions = deque(actions, maxlen=8)
    successes = deque(successes, maxlen=8)
    actions.append(ACTIONS.index(action))
    successes.append(int(action_is_legal(old_state, action)))
    if progress_signature(new_state) != old_progress:
        positions.clear()
        actions.clear()
        successes.clear()
        last_step = int(new_state["step"])
    position = tuple(new_state["self"][3])
    return _state_to_features(
        self, new_state, ACTIONS.index(action), positions.count(position),
        int(new_state["step"]) - last_step,
        tuple(actions), tuple(successes), tuple(positions) + (position,),
    )


def act(self, game_state):
    if game_state is None:
        return "WAIT"
    candidates = _base.safe_action_indices(game_state)
    if not candidates:
        candidates = _base.best_survival_action_indices(game_state)
    feature = _features_for_state(self, game_state)
    if self.train and self.rng.random() < self.epsilon:
        choice = int(self.rng.choice(candidates))
    else:
        with torch.no_grad():
            values = self.policy_net(torch.as_tensor(
                feature, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0))[0].detach().cpu().numpy()
        choice = max(candidates, key=lambda index: (values[index], -index))
    self.previous_action = choice
    self.last_action = choice
    self.last_action_state = game_state
    return ACTIONS[choice]


save_checkpoint = _base.save_checkpoint

__all__ = [
    "MODEL_PATH", "RESUME_PATH", "act", "next_features", "progress_signature",
    "save_checkpoint", "setup", "state_key",
]
