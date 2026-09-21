"""Agent 042 callbacks: persistent history and task-specific exploration."""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from agent_code.combat_dqn_agent import callbacks as _base
from agent_code.combat_dqn_agent.model import DEVICE

from . import features
from .checkpoint import expand_checkpoint
from .features import ACTIONS, StateContext


MODEL_PATH = Path(__file__).resolve().parent / "tournament_checkpoint.pt"
RESUME_PATH = None

SOLO_EPSILON_START = 1.0
SOLO_EPSILON_END = 0.05
SOLO_EPSILON_DECAY_STEPS = 100_000
COMBAT_EPSILON_START = 0.30
COMBAT_EPSILON_END = 0.05
COMBAT_EPSILON_DECAY_STEPS = 100_000


def _linear_epsilon(step, start, end, decay):
    fraction = min(1.0, max(0, int(step)) / float(decay))
    return start + fraction * (end - start)


def solo_epsilon(step):
    return _linear_epsilon(
        step, SOLO_EPSILON_START, SOLO_EPSILON_END,
        SOLO_EPSILON_DECAY_STEPS,
    )


def combat_epsilon(step):
    return _linear_epsilon(
        step, COMBAT_EPSILON_START, COMBAT_EPSILON_END,
        COMBAT_EPSILON_DECAY_STEPS,
    )


def state_key(game_state):
    return int(game_state["round"]), int(game_state["step"])


def progress_signature(game_state):
    """Track only progress attributable to this agent's observation.

    The old signature reset history on global crate and opponent changes.
    Those are not proof that this agent made progress, so Agent 042 uses the
    visible coin set and own score only.  A coin removed by another player is
    still an unavoidable partial-observability ambiguity, documented for
    later analysis.
    """
    return (
        tuple(sorted(tuple(coin) for coin in game_state["coins"])),
        int(game_state["self"][1]),
    )


def _context_for_state(self, game_state):
    key = (state_key(game_state), id(game_state))
    context = self._agent042_contexts.get(key)
    if context is None:
        context = StateContext(game_state)
        self._agent042_contexts[key] = context
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
        _base.load_checkpoint = lambda path: expand_checkpoint(
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
    self._agent042_contexts = {}
    self.combat_episode = False


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
        self._agent042_contexts.clear()
        self.last_progress = None
        self.last_progress_step = int(game_state["step"])
        self.previous_action = ACTIONS.index("WAIT")
        _reset_temporal_context(self, game_state)
        self.combat_episode = False
    else:
        _advance_action_context(self, game_state)

    if game_state["others"]:
        self.combat_episode = True
    signature = progress_signature(game_state)
    if signature != self.last_progress:
        # Keep rolling position/action history.  Only the no-progress clock is
        # restarted; this prevents unrelated global events from erasing the
        # evidence needed to identify a loop.
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
    if new_state["others"]:
        self.combat_episode = True
    if progress_signature(new_state) != old_progress:
        # As above, retain the rolling loop history and only reset progress
        # timing.  This makes the transition representation match inference.
        last_step = int(new_state["step"])
    position = tuple(new_state["self"][3])
    return _state_to_features(
        self, new_state, ACTIONS.index(action), positions.count(position),
        int(new_state["step"]) - last_step,
        tuple(actions), tuple(successes), tuple(positions) + (position,),
    )


def action_candidates(self, game_state):
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
    if self.train:
        self.epsilon = (
            combat_epsilon(getattr(self, "combat_env_steps", 0))
            if self.combat_episode else
            solo_epsilon(getattr(self, "env_steps", 0))
        )
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
    "COMBAT_EPSILON_DECAY_STEPS", "COMBAT_EPSILON_END",
    "COMBAT_EPSILON_START", "MODEL_PATH", "RESUME_PATH",
    "SOLO_EPSILON_DECAY_STEPS", "SOLO_EPSILON_END", "SOLO_EPSILON_START",
    "act", "action_candidates", "action_mask", "combat_epsilon",
    "next_features", "progress_signature", "save_checkpoint", "setup",
    "solo_epsilon", "state_key",
]
