"""Double-DQN training callbacks for the combat agent."""

# Sahand was here.

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

import events as e
from agent_code.combat_fqi_history_antistag_agent.train import (
    reward_from_transition,
)
from .callbacks import next_features, save_checkpoint, state_key
from .config import (
    BATCH_SIZE,
    EPSILON_DECAY_STEPS,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    GRADIENT_CLIP_NORM,
    REPLAY_CAPACITY,
    TARGET_UPDATE_EVERY,
    TRAIN_EVERY,
    WARMUP_TRANSITIONS,
)
from .features import ACTIONS
from .model import DEVICE, double_dqn_targets
from .replay import ReplayBuffer
from .safety import best_survival_action_indices, safe_action_indices


def setup_training(self):
    """Initialize replay, counters, and episode metrics."""
    self.replay_buffer = ReplayBuffer(
        capacity=REPLAY_CAPACITY,
        seed=getattr(self, "seed", None),
    )
    self.pending = None
    self.round_reward = 0.0
    self.round_steps = 0
    self.round_events = defaultdict(int)
    self.round_loss_total = 0.0
    self.round_loss_count = 0
    self.last_round_average_loss = None


def _epsilon_for_step(step):
    fraction = min(1.0, float(step) / EPSILON_DECAY_STEPS)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def optimize_model(self):
    """Perform one random mini-batch Double-DQN update."""
    if len(self.replay_buffer) < max(WARMUP_TRANSITIONS, BATCH_SIZE):
        return None
    batch = self.replay_buffer.sample(BATCH_SIZE)
    states = torch.as_tensor(batch.states, dtype=torch.float32, device=DEVICE)
    actions = torch.as_tensor(batch.actions, dtype=torch.long, device=DEVICE)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=DEVICE)
    next_states = torch.as_tensor(batch.next_states, dtype=torch.float32,
                                  device=DEVICE)
    dones = torch.as_tensor(batch.dones, dtype=torch.bool, device=DEVICE)
    next_masks = torch.as_tensor(batch.next_action_masks, dtype=torch.bool,
                                 device=DEVICE)

    self.policy_net.train()
    values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    targets = double_dqn_targets(
        self.policy_net, self.target_net, next_states, rewards, dones,
        next_masks, gamma=GAMMA,
    )
    loss = F.smooth_l1_loss(values, targets)
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(self.policy_net.parameters(), GRADIENT_CLIP_NORM)
    self.optimizer.step()
    self.optimizer_steps += 1
    if self.optimizer_steps % TARGET_UPDATE_EVERY == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
    self.last_loss = float(loss.detach().cpu().item())
    self.round_loss_total = getattr(self, "round_loss_total", 0.0) + self.last_loss
    self.round_loss_count = getattr(self, "round_loss_count", 0) + 1
    return self.last_loss


def _next_action_mask(new_state):
    if new_state is None:
        return np.zeros(len(ACTIONS), dtype=bool)
    candidates = safe_action_indices(new_state)
    if not candidates:
        candidates = best_survival_action_indices(new_state)
    mask = np.zeros(len(ACTIONS), dtype=bool)
    mask[candidates] = True
    return mask


def remember(self, old_state, action, new_state, events):
    """Store one transition and train only after replay warmup."""
    reward = reward_from_transition(old_state, action, new_state, events)
    future = None
    if new_state is not None:
        future = next_features(self, old_state, action, new_state)
    self.replay_buffer.add(
        self.feature_cache[state_key(old_state)][0],
        ACTIONS.index(action),
        reward,
        future,
        new_state is None,
        _next_action_mask(new_state),
    )
    self.env_steps += 1
    self.epsilon = _epsilon_for_step(self.env_steps)
    if self.env_steps % TRAIN_EVERY == 0:
        optimize_model(self)
    self.round_reward += reward
    self.round_steps += 1
    for event in events:
        self.round_events[event] += 1


def game_events_occurred(self, old_game_state, self_action,
                         new_game_state, events):
    """Delay one transition so the final callback can mark it terminal."""
    if old_game_state is None or self_action is None:
        return
    if self.pending is not None:
        remember(self, *self.pending)
    self.pending = (old_game_state, self_action, new_game_state, list(events))


def end_of_round(self, last_game_state, last_action, events):
    """Store the terminal transition and persist the complete checkpoint."""
    if self.pending is not None:
        pending_state = self.pending[0]
        pending_is_final = (
            last_game_state is not None and
            state_key(pending_state) == state_key(last_game_state)
        )
        if not pending_is_final:
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
