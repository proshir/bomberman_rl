"""Training callbacks for the minimal vanilla DQN."""

# Sahand was here.

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

import events as e
from agent_code.combat_fqi_history_antistag_agent.train import reward_from_transition
from .callbacks import next_features, save_checkpoint, state_key
from .config import (
    BATCH_SIZE, EPSILON_DECAY_STEPS, EPSILON_END, EPSILON_START, GAMMA,
    GRADIENT_CLIP_NORM, REPLAY_CAPACITY, TARGET_UPDATE_EVERY, TRAIN_EVERY,
    WARMUP_TRANSITIONS,
)
from .features import ACTIONS
from .model import DEVICE, vanilla_targets
from .replay import ReplayBuffer
from .safety import best_survival_action_indices, safe_action_indices


def setup_training(self):
    self.replay_buffer = ReplayBuffer(REPLAY_CAPACITY, getattr(self, "seed", None))
    self.pending = None
    self.round_reward = 0.0
    self.round_steps = 0
    self.round_events = defaultdict(int)
    self.round_loss_total = 0.0
    self.round_loss_count = 0
    self.last_round_average_loss = None


def _epsilon(step):
    fraction = min(1.0, step / EPSILON_DECAY_STEPS)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def optimize_model(self):
    if len(self.replay_buffer) < max(WARMUP_TRANSITIONS, BATCH_SIZE):
        return None
    batch = self.replay_buffer.sample(BATCH_SIZE)
    states = torch.as_tensor(batch.states, dtype=torch.float32, device=DEVICE)
    actions = torch.as_tensor(batch.actions, dtype=torch.long, device=DEVICE)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=DEVICE)
    next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=DEVICE)
    dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=DEVICE)
    next_action_masks = torch.as_tensor(
        batch.next_action_masks, dtype=torch.bool, device=DEVICE
    )
    values = self.policy_net(states).gather(1, actions[:, None]).squeeze(1)
    targets = vanilla_targets(
        self.target_net, next_states, rewards, dones, GAMMA, next_action_masks
    )
    loss = F.smooth_l1_loss(values, targets)
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(self.policy_net.parameters(), GRADIENT_CLIP_NORM)
    self.optimizer.step()
    self.optimizer_steps += 1
    if self.optimizer_steps % TARGET_UPDATE_EVERY == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
    self.round_loss_total += float(loss.detach().cpu().item())
    self.round_loss_count += 1
    return float(loss.detach().cpu().item())


def _candidate_mask(game_state):
    """Encode the exact action candidates used by ``callbacks.act``."""
    candidates = safe_action_indices(game_state)
    if not candidates:
        candidates = best_survival_action_indices(game_state)
    mask = np.zeros(len(ACTIONS), dtype=bool)
    mask[candidates] = True
    return mask


def remember(self, old_state, action, new_state, events,
             next_decision_state=None):
    reward = reward_from_transition(old_state, action, new_state, events)
    if new_state is None:
        future = None
        next_action_mask = np.zeros(len(ACTIONS), dtype=bool)
    elif next_decision_state is not None:
        # ``old_game_state`` in the following callback is the exact state that
        # was passed to the next act() call.  Its feature cache therefore has
        # the same step/history context as inference.
        cached = self.feature_cache.get(state_key(next_decision_state))
        if cached is None:
            raise KeyError(
                "next decision state was not cached before its transition "
                "was committed"
            )
        future = cached[0]
        next_action_mask = _candidate_mask(next_decision_state)
    else:
        # Compatibility fallback for direct callers that do not provide the
        # following action-time state.  Normal framework callbacks use the
        # exact cached state path above.
        future = next_features(self, old_state, action, new_state)
        next_action_mask = _candidate_mask(new_state)
    self.replay_buffer.add(
        self.feature_cache[state_key(old_state)][0], ACTIONS.index(action),
        reward, future, new_state is None, next_action_mask,
    )
    self.env_steps += 1
    self.epsilon = _epsilon(self.env_steps)
    if self.env_steps % TRAIN_EVERY == 0:
        optimize_model(self)
    self.round_reward += reward
    self.round_steps += 1
    for event in events:
        self.round_events[event] += 1


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action is None:
        return
    if self.pending is not None:
        remember(self, *self.pending, next_decision_state=old_game_state)
    self.pending = (old_game_state, self_action, new_game_state, list(events))


def end_of_round(self, last_game_state, last_action, events):
    if self.pending is not None:
        pending_state = self.pending[0]
        pending_is_final = (
            last_game_state is not None and state_key(pending_state) == state_key(last_game_state)
        )
        if not pending_is_final:
            if last_game_state is not None:
                remember(self, *self.pending,
                         next_decision_state=last_game_state)
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
    save_checkpoint(self, self.model_path)
    self.round_reward = 0.0
    self.round_steps = 0
    self.round_events.clear()
    self.round_loss_total = 0.0
    self.round_loss_count = 0
