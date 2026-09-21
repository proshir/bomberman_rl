from collections import deque
from pathlib import Path
import copy
import os

import numpy as np
import settings as s
import torch
from torch import nn, optim

from . import features
from .features import ACTIONS, StateContext, AGENT_040_FEATURE_SIZE, FEATURE_SIZE


MODEL_PATH = Path(__file__).resolve().parent / "tournament_checkpoint.pt"
RESUME_PATH = None


ALGORITHM = os.environ.get("BOMBERMAN_DQN_ALGORITHM", "dqn")

GAMMA = 0.99
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
REPLAY_CAPACITY = 100_000
WARMUP_TRANSITIONS = 5_000
TRAIN_EVERY = 4
TARGET_UPDATE_EVERY = 1_000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 100_000
GRADIENT_CLIP_NORM = 10.0
HIDDEN_SIZE = 128
N_ACTIONS = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions=N_ACTIONS):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_actions = int(n_actions)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, self.n_actions),
        )

    def forward(self, states):
        return self.network(states)


def dqn_targets(policy_net, target_net, next_states, rewards, dones, gamma,
                next_action_masks, algorithm="dqn"):
    algorithm = algorithm.strip().lower()
    if algorithm not in {"dqn", "ddqn", "double_dqn"}:
        raise ValueError(
            "algorithm must be 'dqn', 'ddqn', or 'double_dqn', "
            f"not {algorithm!r}"
        )
    next_action_masks = next_action_masks.to(dtype=torch.bool)
    with torch.no_grad():
        target_next_q = target_net(next_states)
        selection_q = (
            target_next_q
            if algorithm == "dqn"
            else policy_net(next_states)
        )
        masked_selection_q = selection_q.masked_fill(
            ~next_action_masks, -torch.inf
        )
        best_actions = masked_selection_q.argmax(dim=1, keepdim=True)
        next_values = target_next_q.gather(1, best_actions).squeeze(1)
        has_candidates = next_action_masks.any(dim=1)
        bootstrap = torch.where(
            (dones > 0) | ~has_candidates,
            torch.zeros_like(next_values),
            next_values,
        )
        return rewards + float(gamma) * (1.0 - dones) * bootstrap


def vanilla_targets(target_net, next_states, rewards, dones, gamma,
                    next_action_masks):
    return dqn_targets(
        target_net, target_net, next_states, rewards, dones, gamma,
        next_action_masks, algorithm="dqn",
    )


def save_checkpoint(learner, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "combat_vanilla_dqn_v1",
        "input_dim": learner.policy_net.input_dim,
        "n_actions": learner.policy_net.n_actions,
        "policy_state_dict": learner.policy_net.state_dict(),
        "target_state_dict": learner.target_net.state_dict(),
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "epsilon": float(learner.epsilon),
        "env_steps": int(learner.env_steps),
        "optimizer_steps": int(learner.optimizer_steps),
    }
    if hasattr(learner, "combat_env_steps"):
        checkpoint["combat_env_steps"] = int(learner.combat_env_steps)
    torch.save(checkpoint, path)


def load_checkpoint(path):
    return torch.load(Path(path), map_location=DEVICE, weights_only=False)


_save_checkpoint = save_checkpoint
_load_checkpoint = load_checkpoint

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
                        steps_since_progress):
    return self.feature_module.state_to_features(
        game_state, previous_action, recent_visits, steps_since_progress
    )


def _base_setup(self):
    self.rng = np.random.default_rng(getattr(self, "seed", None))
    self.model_path = Path(getattr(self, "model_path", MODEL_PATH))
    self.feature_cache = {}
    self.round_id = None
    self.last_progress = None
    self.last_progress_step = 0
    self.previous_action = ACTIONS.index("WAIT")
    self.positions = deque(maxlen=8)
                                                                         
                                                                          
    if not hasattr(self, "dqn_algorithm"):
        self.dqn_algorithm = ALGORITHM
    if not hasattr(self, "feature_module"):
        self.feature_module = features
    network_class = getattr(self, "network_class", QNetwork)
    self.policy_net = network_class(
        len(self.feature_module.state_to_features(_probe_state())), N_ACTIONS
    ).to(DEVICE)
    self.target_net = network_class(
        self.policy_net.input_dim, N_ACTIONS
    ).to(DEVICE)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
    self.epsilon = 1.0
    self.env_steps = 0
    self.optimizer_steps = 0
    if self.train:
        resume_path = Path(RESUME_PATH) if RESUME_PATH is not None else None
        if resume_path is not None:
            checkpoint = _load_checkpoint(resume_path)
            if int(checkpoint["input_dim"]) != self.policy_net.input_dim:
                raise ValueError("DQN checkpoint feature dimension does not match.")
            if int(checkpoint.get("n_actions", N_ACTIONS)) != N_ACTIONS:
                raise ValueError("DQN checkpoint action dimension does not match.")
            self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = float(checkpoint.get("epsilon", EPSILON_END))
            self.env_steps = int(checkpoint.get("env_steps", 0))
            self.optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
        elif self.model_path.exists():
            raise FileExistsError(f"Checkpoint already exists: {self.model_path}")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint = _load_checkpoint(self.model_path)
        if int(checkpoint["input_dim"]) != self.policy_net.input_dim:
            raise ValueError("DQN checkpoint feature dimension does not match.")
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.epsilon = float(checkpoint.get("epsilon", EPSILON_END))
        self.env_steps = int(checkpoint.get("env_steps", 0))
        self.optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
        self.policy_net.eval()




def expand_agent040_checkpoint(checkpoint):
    old_dim = int(checkpoint["input_dim"])
    if old_dim == FEATURE_SIZE:
        return checkpoint
    if old_dim != AGENT_040_FEATURE_SIZE:
        raise ValueError(
            f"Agent 041 accepts {AGENT_040_FEATURE_SIZE}- or "
            f"{FEATURE_SIZE}-input checkpoints, got {old_dim}"
        )

    result = copy.deepcopy(checkpoint)
    added = FEATURE_SIZE - old_dim

    def pad(tensor):
        if tensor.ndim != 2 or tensor.shape[1] != old_dim:
            raise ValueError(
                "Unexpected Agent040 first-layer checkpoint shape: "
                f"{tuple(tensor.shape)}"
            )
        return torch.cat(
            (tensor, tensor.new_zeros((tensor.shape[0], added))), dim=1
        )

    first_weight = "network.0.weight"
    old_shape = result["policy_state_dict"][first_weight].shape
    for name in ("policy_state_dict", "target_state_dict"):
        result[name][first_weight] = pad(result[name][first_weight])

    optimizer = result.get("optimizer_state_dict", {})
    groups = optimizer.get("param_groups", [])
    if groups and groups[0].get("params"):
        first_parameter = groups[0]["params"][0]
        state = optimizer.get("state", {}).get(first_parameter, {})
        for name, value in list(state.items()):
            if torch.is_tensor(value) and value.shape == old_shape:
                state[name] = pad(value)

    result["input_dim"] = FEATURE_SIZE
    result["migration"] = (
        f"{old_dim}-to-{FEATURE_SIZE} zero-padded dynamic navigation columns"
    )
    return result


__all__ = ["expand_agent040_checkpoint"]

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
    _base_setup(self)
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


save_checkpoint = _save_checkpoint


__all__ = [
    "MODEL_PATH", "RESUME_PATH", "act", "action_candidates", "action_mask",
    "next_features", "progress_signature", "save_checkpoint", "setup",
    "state_key",
]


