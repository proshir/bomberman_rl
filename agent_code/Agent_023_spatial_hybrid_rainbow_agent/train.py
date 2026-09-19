"""Framework training hooks: masked Double-DQN quantile updates with PER.

This is intentionally a small online fallback. Large actor/learner runs use
the same contracts but live under ``src/`` so they are never packaged for the
official single-process agent.
"""

import numpy as np
import torch
from torch.nn import functional as F
import events as e
from .callbacks import save_checkpoint
from .config import ACTIONS, GAMMA
from .features import split_features
from .model import SpatialHybridRainbow
from .safety import best_survival_action_indices, safe_action_indices

def setup_training(self):
    self.target_net = SpatialHybridRainbow().to(self.device); self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
    self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=2e-4, weight_decay=1e-5)
    self.replay, self.priorities, self.pending = [], [], None; self.updates = 0
    self.epsilon = 0.0; self.env_steps = 0; self.optimizer_steps = 0
    self.last_round_reward = 0.0; self.last_round_average_loss = None

def _reward(events):
    return 1.0 * events.count(e.COIN_COLLECTED) + .25 * events.count(e.CRATE_DESTROYED) + 5.0 * events.count(e.KILLED_OPPONENT) - 5.0 * events.count(e.KILLED_SELF) - .2 * events.count(e.INVALID_ACTION)

def _mask(state):
    indices = safe_action_indices(state) or best_survival_action_indices(state)
    out = np.zeros(6, dtype=bool); out[indices] = True; return out

def _add(self, state, action, reward, next_state, done):
    current = self.feature_cache.get((state["round"], state["step"]))
    future = None if next_state is None else self.feature_cache.get((next_state["round"], next_state["step"]))
    if current is None or (next_state is not None and future is None): return
    self.replay.append((current, ACTIONS.index(action), reward, future, done, np.zeros(6, bool) if done else _mask(next_state)))
    self.priorities.append(max(self.priorities, default=1.0))
    if len(self.replay) > 200_000: self.replay.pop(0); self.priorities.pop(0)

def _update(self, batch_size=64):
    if len(self.replay) < 4096: return
    p = np.asarray(self.priorities, dtype=float) ** .6; p /= p.sum(); ids = np.random.choice(len(self.replay), batch_size, p=p)
    rows = [self.replay[i] for i in ids]; states, actions, rewards, futures, dones, masks = zip(*rows)
    s, g, a = (torch.from_numpy(x).to(self.device) for x in split_features(np.stack(states)))
    ns, ng, na = (torch.from_numpy(x).to(self.device) for x in split_features(np.stack([np.zeros_like(states[0]) if x is None else x for x in futures])))
    actions, rewards, dones, masks = torch.tensor(actions, device=self.device), torch.tensor(rewards, device=self.device), torch.tensor(dones, dtype=torch.bool, device=self.device), torch.tensor(np.stack(masks), device=self.device)
    quantiles = self.policy_net(s, g, a); chosen = quantiles[torch.arange(batch_size), actions]
    with torch.no_grad():
        online_q = self.policy_net.q_values(ns, ng, na).masked_fill(~masks, -torch.inf); selected = online_q.argmax(1)
        target = self.target_net(ns, ng, na)[torch.arange(batch_size), selected]; target[dones] = 0.; target = rewards[:, None] + GAMMA * target
    delta = target[:, None, :] - chosen[:, :, None]
    tau = (torch.arange(chosen.shape[1], dtype=torch.float32, device=self.device) + .5) / chosen.shape[1]
    chosen_pairs = chosen[:, :, None].expand_as(delta)
    target_pairs = target[:, None, :].expand_as(delta)
    loss_per = (torch.abs(tau[None,:,None] - (delta.detach() < 0).float()) * F.smooth_l1_loss(chosen_pairs, target_pairs, reduction="none")).mean((1,2))
    weights = torch.tensor((len(self.replay) * p[ids]) ** -.4, dtype=torch.float32, device=self.device)
    weights /= weights.max(); loss = (weights * loss_per).mean()
    self.optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.); self.optimizer.step()
    for index, value in zip(ids, loss_per.detach().cpu().numpy()): self.priorities[index] = float(value) + 1e-4
    self.updates += 1; self.optimizer_steps = self.updates; self.last_round_average_loss = float(loss.detach().cpu())
    if self.updates % 500 == 0: self.target_net.load_state_dict(self.policy_net.state_dict())

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action is None: return
    self.env_steps += 1
    # Commit the preceding transition only once this state has been passed to
    # ``act`` and therefore has its exact action-time context in feature_cache.
    if self.pending is not None:
        _add(self, *self.pending, next_state=old_game_state, done=False); _update(self)
    self.pending = (old_game_state, self_action, _reward(list(events)))

def end_of_round(self, last_game_state, last_action, events):
    if self.pending is not None:
        _add(self, *self.pending, next_state=last_game_state, done=False)
    if last_game_state is not None and last_action is not None: _add(self, last_game_state, last_action, _reward(list(events)), None, True)
    self.pending = None
    self.last_round_reward = _reward(list(events))
    save_checkpoint(self, self.model_path)
