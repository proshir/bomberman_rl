"""Agent 042 DDQN training with task-specific exploration accounting."""

import events as e
import numpy as np
import settings as s
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from . import dep_Agent_030_combat_ddqn_escape_replay_agent_train as _escape
from . import dep_combat_dqn_agent_callbacks as _base_callbacks
from . import dep_combat_dqn_agent_train as _dqn
from .dep_combat_dqn_agent_model import DEVICE, dqn_targets, load_checkpoint

from . import dep_Agent_042_combat_progress_ddqn_agent_callbacks as _callbacks
from .dep_Agent_042_combat_progress_ddqn_agent_features import ACTIONS, FEATURE_SIZE
from .dep_Agent_042_combat_progress_ddqn_agent_replay import CombatEscapeReplayBuffer, ESCAPE_TAG
from .dep_Agent_042_combat_progress_ddqn_agent_symmetry import TRANSFORMS, transform_transition


ALGORITHM = _escape.ALGORITHM
BATCH_SIZE = _escape.BATCH_SIZE
EPSILON_DECAY_STEPS = _escape.EPSILON_DECAY_STEPS
EPSILON_END = _escape.EPSILON_END
EPSILON_START = _escape.EPSILON_START
GAMMA = _escape.GAMMA
GRADIENT_CLIP_NORM = _escape.GRADIENT_CLIP_NORM
HIDDEN_SIZE = _escape.HIDDEN_SIZE
LEARNING_RATE = _escape.LEARNING_RATE
N_ACTIONS = _escape.N_ACTIONS
REPLAY_CAPACITY = _escape.REPLAY_CAPACITY
TARGET_UPDATE_EVERY = _escape.TARGET_UPDATE_EVERY
TRAIN_EVERY = _escape.TRAIN_EVERY
WARMUP_TRANSITIONS = _escape.WARMUP_TRANSITIONS

COMBAT_EPSILON_START = _callbacks.COMBAT_EPSILON_START
COMBAT_EPSILON_END = _callbacks.COMBAT_EPSILON_END
COMBAT_EPSILON_DECAY_STEPS = _callbacks.COMBAT_EPSILON_DECAY_STEPS
SOLO_EPSILON_START = _callbacks.SOLO_EPSILON_START
SOLO_EPSILON_END = _callbacks.SOLO_EPSILON_END
SOLO_EPSILON_DECAY_STEPS = _callbacks.SOLO_EPSILON_DECAY_STEPS


def _combat_epsilon(step):
    return _callbacks.combat_epsilon(step)


def _solo_epsilon(step):
    return _callbacks.solo_epsilon(step)


def setup_training(self):
    _dqn.setup_training(self)
    self.replay_buffer = CombatEscapeReplayBuffer(
        REPLAY_CAPACITY, getattr(self, "seed", None), n_actions=N_ACTIONS,
        state_dim=FEATURE_SIZE,
    )
    self.replay_buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
    checkpoint = (
        load_checkpoint(_base_callbacks.RESUME_PATH)
        if _base_callbacks.RESUME_PATH is not None else {}
    )
    self.combat_env_steps = int(checkpoint.get("combat_env_steps", 0))
    self.bomb_escape_steps_remaining = 0
    self.combat_episode = False
    # Solo training uses the normal broad exploration schedule.  Combat
    # episodes switch to the dedicated counter in callbacks.act().
    self.epsilon = _solo_epsilon(getattr(self, "env_steps", 0))


def optimize_model(self):
    if len(self.replay_buffer) < max(WARMUP_TRANSITIONS, BATCH_SIZE):
        return None
    states, actions, rewards, next_states, dones, next_action_masks = (
        self.replay_buffer.sample_torch(BATCH_SIZE, DEVICE)
    )
    values = self.policy_net(states).gather(1, actions[:, None]).squeeze(1)
    targets = dqn_targets(
        self.policy_net, self.target_net, next_states, rewards, dones, GAMMA,
        next_action_masks, algorithm=getattr(self, "dqn_algorithm", ALGORITHM),
    )
    loss = F.smooth_l1_loss(values, targets)
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(self.policy_net.parameters(), GRADIENT_CLIP_NORM)
    self.optimizer.step()
    self.optimizer_steps += 1
    if self.optimizer_steps % TARGET_UPDATE_EVERY == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
    loss_value = float(loss.detach().cpu().item())
    self.round_loss_total += loss_value
    self.round_loss_count += 1
    return loss_value


def remember(self, old_state, action, new_state, events,
             next_decision_state=None):
    reward = _dqn.reward_from_transition(old_state, action, new_state, events)
    placed_combat_bomb = e.BOMB_DROPPED in events and bool(old_state["others"])
    is_combat_escape = (
        placed_combat_bomb or self.bomb_escape_steps_remaining > 0
    )
    self.replay_buffer.transition_tag = (
        ESCAPE_TAG if is_combat_escape else None
    )
    if new_state is None:
        future = None
        next_action_mask = np.zeros(N_ACTIONS, dtype=bool)
    elif next_decision_state is not None:
        cached = self.feature_cache.get(
            _callbacks.state_key(next_decision_state)
        )
        if cached is None:
            raise KeyError(
                "next decision state was not cached before its transition "
                "was committed"
            )
        future = cached[0]
        next_action_mask = _callbacks.action_mask(self, next_decision_state)
    else:
        future = _callbacks.next_features(self, old_state, action, new_state)
        next_action_mask = _callbacks.action_mask(self, new_state)

    current = self.feature_cache[_callbacks.state_key(old_state)][0]
    transform = TRANSFORMS[int(self.rng.integers(len(TRANSFORMS)))]
    current, action_index, future, next_action_mask = transform_transition(
        current, ACTIONS.index(action), future, next_action_mask, transform,
        game_state=old_state,
        next_game_state=(
            next_decision_state
            if next_decision_state is not None else new_state
        ),
    )
    self.replay_buffer.add(
        current, action_index, reward, future, new_state is None,
        next_action_mask,
    )
    if placed_combat_bomb:
        self.bomb_escape_steps_remaining = s.BOMB_TIMER
    elif self.bomb_escape_steps_remaining:
        self.bomb_escape_steps_remaining -= 1
    if new_state is None:
        self.bomb_escape_steps_remaining = 0

    self.env_steps += 1
    if getattr(self, "combat_episode", False):
        self.combat_env_steps += 1
        self.epsilon = _combat_epsilon(self.combat_env_steps)
    else:
        self.epsilon = _solo_epsilon(self.env_steps)
    if self.env_steps % TRAIN_EVERY == 0:
        optimize_model(self)
    self.round_reward += reward
    self.round_steps += 1
    for event in events:
        self.round_events[event] += 1


def game_events_occurred(self, old_game_state, self_action, new_game_state,
                         events):
    if old_game_state is None or self_action is None:
        return
    if self.pending is not None:
        remember(self, *self.pending, next_decision_state=old_game_state)
    self.pending = (old_game_state, self_action, new_game_state, list(events))


def end_of_round(self, last_game_state, last_action, events):
    if self.pending is not None:
        same_final = (
            last_game_state is not None and
            _callbacks.state_key(self.pending[0]) ==
            _callbacks.state_key(last_game_state)
        )
        if not same_final:
            if last_game_state is not None:
                remember(self, *self.pending, next_decision_state=last_game_state)
            else:
                remember(self, *self.pending)
    if last_game_state is not None and last_action is not None:
        remember(self, last_game_state, last_action, None, list(events))
    self.pending = None
    self.last_round_reward = self.round_reward
    self.last_round_steps = self.round_steps
    self.last_round_events = dict(self.round_events)
    self.last_round_average_loss = (
        self.round_loss_total / self.round_loss_count
        if self.round_loss_count else None
    )
    _base_callbacks.save_checkpoint(self, self.model_path)
    self.round_reward = 0.0
    self.round_steps = 0
    self.round_events.clear()
    self.round_loss_total = 0.0
    self.round_loss_count = 0
    self.bomb_escape_steps_remaining = 0
    self.combat_episode = False


__all__ = [
    "ALGORITHM", "BATCH_SIZE", "COMBAT_EPSILON_DECAY_STEPS",
    "COMBAT_EPSILON_END", "COMBAT_EPSILON_START", "EPSILON_DECAY_STEPS",
    "EPSILON_END", "EPSILON_START", "FEATURE_SIZE", "GAMMA",
    "GRADIENT_CLIP_NORM", "HIDDEN_SIZE", "LEARNING_RATE", "N_ACTIONS",
    "REPLAY_CAPACITY", "SOLO_EPSILON_DECAY_STEPS", "SOLO_EPSILON_END",
    "SOLO_EPSILON_START", "TARGET_UPDATE_EVERY", "TRAIN_EVERY",
    "WARMUP_TRANSITIONS", "end_of_round", "game_events_occurred",
    "optimize_model", "remember", "setup_training",
]
