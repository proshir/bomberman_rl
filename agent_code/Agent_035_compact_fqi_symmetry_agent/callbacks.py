"""Cached decisions and frozen inference for corrected compact FQI."""

import pickle
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np

from .features import FEATURE_SCHEMA, FEATURE_SIZE, HISTORY_LENGTH, state_to_features
from .safety import ACTIONS, allowed_action_indices


MODEL_PATH = Path(__file__).resolve().parent / "trees.pkl"
RESUME_PATH = None


def state_key(game_state):
    return int(game_state["round"]), int(game_state["step"])


def progress_signature(game_state):
    return (int(game_state["self"][1]),
            int(np.count_nonzero(game_state["field"] == 1)))


def predict_values(trees, states):
    states = np.asarray(states, dtype=np.float32).reshape(-1, FEATURE_SIZE)
    values = np.zeros((len(states), len(ACTIONS)), dtype=float)
    for index, tree in enumerate(trees):
        if tree is not None:
            values[:, index] = tree.predict(states)
    return values


def _read_checkpoint(path):
    from .train import CHECKPOINT_SCHEMAS
    with Path(path).open("rb") as stream:
        checkpoint = pickle.load(stream)
    if (not isinstance(checkpoint, dict) or
            checkpoint.get("schemas") != CHECKPOINT_SCHEMAS or
            len(checkpoint.get("trees", ())) != len(ACTIONS)):
        raise ValueError("Incompatible Agent 035 checkpoint")
    return checkpoint


def setup(self):
    self.rng = np.random.default_rng(getattr(self, "seed", None))
    self.model_path = Path(getattr(self, "model_path", MODEL_PATH))
    resume_path = getattr(self, "resume_path", RESUME_PATH)
    self.resume_payload = None
    if self.train:
        if resume_path is not None:
            checkpoint = _read_checkpoint(resume_path)
            if "training_state" not in checkpoint:
                raise ValueError("Agent 035 resume requires a full training checkpoint")
            self.trees = checkpoint["trees"]
            self.resume_payload = checkpoint["training_state"]
            self.rng.bit_generator.state = self.resume_payload["rng_state"]
        else:
            if self.model_path.exists():
                raise FileExistsError(f"Checkpoint already exists: {self.model_path}")
            self.trees = [None] * len(ACTIONS)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        self.trees = _read_checkpoint(self.model_path)["trees"]
    self.round_id = None
    self.positions = deque(maxlen=HISTORY_LENGTH)
    self.progress_step = 0
    self.last_progress = None
    self.previous_action = None
    self.observations = {}
    restored = self.resume_payload or {}
    self.feature_seconds = float(restored.get("feature_seconds", 0.0))
    self.safety_seconds = float(restored.get("safety_seconds", 0.0))
    self.action_seconds = float(restored.get("action_seconds", 0.0))
    self.max_action_seconds = float(restored.get("max_action_seconds", 0.0))
    self.action_count = int(restored.get("action_count", 0))


def save_checkpoint(self, path):
    from .train import CHECKPOINT_SCHEMAS
    payload = {"schemas": CHECKPOINT_SCHEMAS, "trees": self.trees}
    if self.train and hasattr(self, "transitions"):
        payload["training_state"] = {
            "transitions": self.transitions,
            "epsilon": self.epsilon,
            "completed_rounds": self.completed_rounds,
            "env_steps": self.env_steps,
            "rng_state": self.rng.bit_generator.state,
            "feature_seconds": self.feature_seconds,
            "safety_seconds": self.safety_seconds,
            "action_seconds": self.action_seconds,
            "max_action_seconds": self.max_action_seconds,
            "action_count": self.action_count,
            "fit_seconds": self.fit_seconds,
            "last_fit_records": self.last_fit_records,
            "last_fit_tag_counts": self.last_fit_tag_counts,
            "last_action_support": self.last_action_support,
        }
    with Path(path).open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def observe(self, game_state):
    key = state_key(game_state)
    if game_state["round"] != self.round_id:
        self.round_id = game_state["round"]
        self.positions.clear()
        self.progress_step = int(game_state["step"])
        self.last_progress = None
        self.previous_action = None
        self.observations.clear()
    progress = progress_signature(game_state)
    if progress != self.last_progress:
        self.positions.clear()
        self.progress_step = int(game_state["step"])
        self.last_progress = progress
    if key not in self.observations:
        position = tuple(game_state["self"][3])
        started = perf_counter()
        features = state_to_features(
            game_state, self.previous_action, self.positions.count(position),
            int(game_state["step"]) - self.progress_step,
        )
        self.feature_seconds += perf_counter() - started
        started = perf_counter()
        allowed = np.zeros(len(ACTIONS), dtype=bool)
        allowed[allowed_action_indices(game_state)] = True
        self.safety_seconds += perf_counter() - started
        self.observations[key] = (features, allowed)
        self.positions.append(position)
    return self.observations[key]


def act(self, game_state):
    if game_state is None:
        return "WAIT"
    started = perf_counter()
    features, allowed = observe(self, game_state)
    candidates = np.flatnonzero(allowed)
    if self.train and self.rng.random() < self.epsilon:
        choice = int(self.rng.choice(candidates))
    else:
        values = predict_values(self.trees, [features])[0]
        choice = int(candidates[np.argmax(values[candidates])])
    self.previous_action = choice
    elapsed = perf_counter() - started
    self.action_seconds += elapsed
    self.max_action_seconds = max(self.max_action_seconds, elapsed)
    self.action_count += 1
    return ACTIONS[choice]
