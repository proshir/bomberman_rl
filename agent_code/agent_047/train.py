"""Agent 043 DDQN with bounded novelty and three-step credit assignment."""

from collections import deque

import events as e
import numpy as np
import settings as s
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from . import dep_Agent_042_combat_progress_ddqn_agent_train as _base_train
from . import dep_combat_dqn_agent_callbacks as _base_callbacks
from . import dep_combat_dqn_agent_train as _dqn
from .dep_combat_dqn_agent_model import DEVICE, dqn_targets, load_checkpoint

from . import callbacks as _callbacks
from .features import ACTIONS, FEATURE_SIZE, StateContext
from .replay import CombatEscapeReplayBuffer, ESCAPE_TAG
from .symmetry import TRANSFORMS, transform_action, transform_features, transform_mask


ALGORITHM = _base_train.ALGORITHM
BATCH_SIZE = _base_train.BATCH_SIZE
EPSILON_DECAY_STEPS = _base_train.EPSILON_DECAY_STEPS
EPSILON_END = _base_train.EPSILON_END
EPSILON_START = _base_train.EPSILON_START
GAMMA = _base_train.GAMMA
GRADIENT_CLIP_NORM = _base_train.GRADIENT_CLIP_NORM
HIDDEN_SIZE = _base_train.HIDDEN_SIZE
LEARNING_RATE = _base_train.LEARNING_RATE
N_ACTIONS = _base_train.N_ACTIONS
REPLAY_CAPACITY = _base_train.REPLAY_CAPACITY
TARGET_UPDATE_EVERY = _base_train.TARGET_UPDATE_EVERY
TRAIN_EVERY = _base_train.TRAIN_EVERY
WARMUP_TRANSITIONS = _base_train.WARMUP_TRANSITIONS

COMBAT_EPSILON_START = _base_train.COMBAT_EPSILON_START
COMBAT_EPSILON_END = _base_train.COMBAT_EPSILON_END
COMBAT_EPSILON_DECAY_STEPS = _base_train.COMBAT_EPSILON_DECAY_STEPS
SOLO_EPSILON_START = _base_train.SOLO_EPSILON_START
SOLO_EPSILON_END = _base_train.SOLO_EPSILON_END
SOLO_EPSILON_DECAY_STEPS = _base_train.SOLO_EPSILON_DECAY_STEPS

N_STEP_RETURN = 3
GAMMA_N = GAMMA ** N_STEP_RETURN
POTENTIAL_SHAPING_SCALE = 0.05


def _combat_epsilon(step):
    return _callbacks.combat_epsilon(step)


def _solo_epsilon(step):
    return _callbacks.solo_epsilon(step)


def _combat_potential(game_state):
    """Bounded nearest-opponent potential for potential-based shaping."""
    if game_state is None or not game_state["others"]:
        return 0.0
    context = StateContext(game_state)
    distances = context.distance_map(
        tuple(tuple(other[3]) for other in game_state["others"])
    )
    distance = distances.get(tuple(game_state["self"][3]))
    if distance is None:
        return 0.0
    board_scale = max(1, max(int(game_state["field"].shape[0]),
                             int(game_state["field"].shape[1])) - 1)
    return -min(1.0, float(distance) / float(board_scale))


def reward_from_transition(old_state, action, new_state, events):
    """Agent 042 reward plus bounded potential-based combat shaping."""
    reward = _dqn.reward_from_transition(old_state, action, new_state, events)
    old_potential = _combat_potential(old_state)
    new_potential = _combat_potential(new_state)
    reward += POTENTIAL_SHAPING_SCALE * (
        GAMMA * new_potential - old_potential
    )
    return reward


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
    self.n_step_queue = deque()
    self.epsilon = _solo_epsilon(getattr(self, "env_steps", 0))


def optimize_model(self):
    if len(self.replay_buffer) < max(WARMUP_TRANSITIONS, BATCH_SIZE):
        return None
    states, actions, rewards, next_states, dones, next_action_masks = (
        self.replay_buffer.sample_torch(BATCH_SIZE, DEVICE)
    )
    values = self.policy_net(states).gather(1, actions[:, None]).squeeze(1)
    targets = dqn_targets(
        self.policy_net, self.target_net, next_states, rewards, dones,
        GAMMA_N, next_action_masks,
        algorithm=getattr(self, "dqn_algorithm", ALGORITHM),
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


def _flush_n_step(self, force=False):
    """Commit complete three-step returns, or terminal partial returns."""
    while self.n_step_queue and (force or len(self.n_step_queue) >= N_STEP_RETURN):
        span = min(N_STEP_RETURN, len(self.n_step_queue))
        rows = list(self.n_step_queue)[:span]
        total = 0.0
        future = None
        next_action_mask = np.zeros(N_ACTIONS, dtype=bool)
        done = False
        last_row = rows[-1]
        for index, row in enumerate(rows):
            total += (GAMMA ** index) * float(row[2])
            if row[4]:
                done = True
                future = None
                next_action_mask = np.zeros(N_ACTIONS, dtype=bool)
                last_row = row
                break
            if index == span - 1:
                future = row[3]
                next_action_mask = row[5]

        transform = TRANSFORMS[int(self.rng.integers(len(TRANSFORMS)))]
        first = rows[0]
        current = transform_features(first[0], transform, first[6])
        action_index = transform_action(first[1], transform)
        if future is not None:
            future = transform_features(future, transform, last_row[7])
            next_action_mask = transform_mask(next_action_mask, transform)
        else:
            next_action_mask = np.zeros(N_ACTIONS, dtype=bool)
        self.replay_buffer.transition_tag = (
            ESCAPE_TAG if any(row[8] for row in rows) else None
        )
        self.replay_buffer.add(
            current, action_index, total, future, done, next_action_mask,
        )
        self.n_step_queue.popleft()
        if not force and len(self.n_step_queue) < N_STEP_RETURN:
            break


def remember(self, old_state, action, new_state, events,
             next_decision_state=None):
    reward = reward_from_transition(old_state, action, new_state, events)
    placed_combat_bomb = e.BOMB_DROPPED in events and bool(old_state["others"])
    is_combat_escape = (
        placed_combat_bomb or self.bomb_escape_steps_remaining > 0
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
        future = _callbacks.next_features(
            self, old_state, action, new_state, events=events
        )
        next_action_mask = _callbacks.action_mask(self, new_state)

    current = self.feature_cache[_callbacks.state_key(old_state)][0]
    next_game_state = (
        next_decision_state if next_decision_state is not None else new_state
    )
    self.n_step_queue.append((
        current, ACTIONS.index(action), reward, future, new_state is None,
        next_action_mask, old_state, next_game_state, is_combat_escape,
    ))
    _flush_n_step(self)

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
    _callbacks.record_progress_events(self, new_game_state, events)
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
    _flush_n_step(self, force=True)
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
    "EPSILON_END", "EPSILON_START", "FEATURE_SIZE", "GAMMA", "GAMMA_N",
    "GRADIENT_CLIP_NORM", "HIDDEN_SIZE", "LEARNING_RATE", "N_ACTIONS",
    "N_STEP_RETURN", "POTENTIAL_SHAPING_SCALE", "REPLAY_CAPACITY",
    "SOLO_EPSILON_DECAY_STEPS", "SOLO_EPSILON_END", "SOLO_EPSILON_START",
    "TARGET_UPDATE_EVERY", "TRAIN_EVERY", "WARMUP_TRANSITIONS",
    "end_of_round", "game_events_occurred", "optimize_model", "remember",
    "reward_from_transition", "setup_training",
]
