"""Agent 041 callbacks with DDQN-controlled dynamic navigation features."""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from . import dep_combat_dqn_agent_callbacks as _base
from .dep_combat_dqn_agent_model import DEVICE

from . import dep_Agent_041_dynamic_nav_ddqn_agent_features as features
from .dep_Agent_041_dynamic_nav_ddqn_agent_checkpoint import expand_agent040_checkpoint
from .dep_Agent_041_dynamic_nav_ddqn_agent_features import ACTIONS, StateContext


MODEL_PATH = Path(__file__).resolve().parent / "tournament_checkpoint.pt"
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


def _context_for_state(self, game_state):
    key = (state_key(game_state), id(game_state))
    context = self._agent041_contexts.get(key)
    if context is None:
        context = StateContext(game_state)
        self._agent041_contexts[key] = context
    return context


def _state_to_features(self, game_state, previous_action, recent_visits,
                       steps_since_progress, action_history=(),
                       action_successes=(), position_history=()):
    return self.feature_module.state_to_features(
        game_state, previous_action, recent_visits, steps_since_progress,
        action_history, action_successes, position_history,
        context=_context_for_state(self, game_state),
    )


def setup(self):
    self.feature_module = features
    self.dqn_algorithm = "ddqn"
    _base.MODEL_PATH = MODEL_PATH
    _base.RESUME_PATH = RESUME_PATH
    original_loader = _base.load_checkpoint
    if self.train and RESUME_PATH is not None:
        _base.load_checkpoint = lambda path: expand_agent040_checkpoint(
            original_loader(path)
        )
    try:
        _base.setup(self)
    finally:
        _base.load_checkpoint = original_loader
    self.feature_module = features
    self.action_history = deque(maxlen=8)
    self.action_successes = deque(maxlen=8)
    self.last_observed_key = None
    self.last_action_state = None
    self.last_action = None
    self._agent041_contexts = {}


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
        context = _context_for_state(self, self.last_action_state)
        self.action_history.append(self.last_action)
        self.action_successes.append(int(
            context.is_legal(self.last_action)
        ))
    self.last_observed_key = current_key


def _features_for_state(self, game_state):
    if game_state["round"] != self.round_id:
        self.round_id = game_state["round"]
        self.feature_cache.clear()
        self._agent041_contexts.clear()
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
    old_context = _context_for_state(self, old_state)
    actions.append(ACTIONS.index(action))
    successes.append(int(old_context.is_legal(action)))
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


def action_candidates(self, game_state):
    """Return only the existing physical/safety candidates.

    Navigation features never alter this set.  The DDQN evaluates all six
    outputs before this safety set is used for the final legal selection.
    """
    return _context_for_state(self, game_state).candidate_indices()


def action_mask(self, game_state):
    mask = np.zeros(len(ACTIONS), dtype=bool)
    mask[list(action_candidates(self, game_state))] = True
    return mask


def act(self, game_state):
    if game_state is None:
        return "WAIT"
    candidates = action_candidates(self, game_state)
    feature = _features_for_state(self, game_state)
    if self.train and self.rng.random() < self.epsilon:
        choice = int(self.rng.choice(candidates))
    else:
        with torch.inference_mode():
            values = self.policy_net(torch.as_tensor(
                feature, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0))[0].cpu().numpy()
        choice = max(candidates, key=lambda index: (values[index], -index))
    self.previous_action = choice
    self.last_action = choice
    self.last_action_state = game_state
    return ACTIONS[choice]


save_checkpoint = _base.save_checkpoint


__all__ = [
    "MODEL_PATH", "RESUME_PATH", "act", "action_candidates", "action_mask",
    "next_features", "progress_signature", "save_checkpoint", "setup",
    "state_key",
]
