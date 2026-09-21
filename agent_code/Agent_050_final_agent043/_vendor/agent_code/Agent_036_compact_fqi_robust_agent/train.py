"""Scenario-balanced, symmetry-augmented fitted Q iteration for Agent 036."""

from collections import Counter, deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.tree import DecisionTreeRegressor

import events as e
from .callbacks import (next_decision_key, next_decision_state,
                        note_progress_events, observe, predict_values,
                        save_checkpoint, state_key)
from .features import FEATURE_SCHEMA
from .safety import ACTIONS, bomb_is_useful, earliest_danger
from .symmetry import SYMMETRY_SCHEMA, TRANSFORMS, transform_transition


DISCOUNT = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
BUFFER_SIZE_PER_TAG = 40000
MAX_FIT_RECORDS = 24000
FIT_EVERY_ROUNDS = 20
FIT_ITERATIONS = 5
MAX_DEPTH = 8
MIN_SAMPLES_LEAF = 5
SAFETY_SCHEMA = "combat-fqi-time-safety-v1"
REWARD_SCHEMA = "combat-fqi-reward-terminal-once-v2"
HISTORY_SCHEMA = "previous-eight-personal-progress-v3"
CHECKPOINT_SCHEMAS = (FEATURE_SCHEMA, SAFETY_SCHEMA, REWARD_SCHEMA,
                      HISTORY_SCHEMA, SYMMETRY_SCHEMA, "full-training-state-v3")


@dataclass(frozen=True)
class Transition:
    features: np.ndarray
    action: int
    reward: float
    next_features: np.ndarray | None
    terminal: bool
    allowed: np.ndarray
    next_allowed: np.ndarray
    events: tuple
    scenario: str
    lineup: tuple
    round_id: int
    board_seed: int | None
    step: int
    schemas: tuple


class TaggedReplay:
    def __init__(self, capacity_per_tag):
        self.capacity_per_tag = capacity_per_tag
        self.by_tag = {}
        self.latest = None

    def append(self, record):
        tag = record.scenario, record.lineup
        self.by_tag.setdefault(tag, deque(maxlen=self.capacity_per_tag)).append(record)
        self.latest = record

    def __iter__(self):
        for rows in self.by_tag.values():
            yield from rows

    def __len__(self):
        return sum(map(len, self.by_tag.values()))

    def __getitem__(self, index):
        if index == -1 and self.latest is not None:
            return self.latest
        return list(self)[index]


def setup_training(self):
    restored = self.resume_payload
    self.epsilon = restored["epsilon"] if restored else EPSILON_START
    self.transitions = (restored["transitions"] if restored else
                        TaggedReplay(BUFFER_SIZE_PER_TAG))
    self.completed_rounds = restored["completed_rounds"] if restored else 0
    self.env_steps = int(restored.get("env_steps", 0)) if restored else 0
    self.pending = None
    self.round_reward = 0.0
    self.round_steps = 0
    self.round_events = Counter()
    self.fit_seconds = (float(restored.get("fit_seconds", 0.0))
                        if restored else 0.0)
    self.last_fit_records = (int(restored.get("last_fit_records", 0))
                             if restored else 0)
    self.last_fit_tag_counts = (dict(restored.get("last_fit_tag_counts", {}))
                                if restored else {})
    self.last_action_support = (dict(restored.get("last_action_support", {}))
                                if restored else {})
    self.last_available_tag_counts = (
        dict(restored.get("last_available_tag_counts", {}))
        if restored else {}
    )
    self.last_effective_scenario_weights = (
        dict(restored.get("last_effective_scenario_weights", {}))
        if restored else {}
    )
    self.training_config = (
        dict(restored.get("training_config", {})) if restored else {}
    )
    restored_weights = (
        restored.get("replay_weights") if restored else None
    )
    self.replay_weights = (
        None if restored_weights is None else dict(restored_weights)
    )


def reward_from_transition(old_state, action, new_state, events):
    reward = -0.01
    reward += events.count(e.COIN_COLLECTED) * 1.0
    reward += events.count(e.KILLED_OPPONENT) * 5.0
    reward += events.count(e.CRATE_DESTROYED) * 0.2
    reward += events.count(e.COIN_FOUND) * 0.2
    reward += events.count(e.SURVIVED_ROUND) * 0.5
    reward -= events.count(e.INVALID_ACTION) * 1.0
    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        reward -= 5.0
    if action == "BOMB" and not bomb_is_useful(old_state):
        reward -= 0.1
    if new_state is not None:
        old_time = earliest_danger(old_state, old_state["self"][3])
        new_time = earliest_danger(new_state, new_state["self"][3])
        if old_time and (not new_time or new_time > old_time):
            reward += 0.1
    return reward


def remember(self, old_state, action, new_state, events, terminal=False):
    features, allowed = self.observations[state_key(old_state)]
    action_index = ACTIONS.index(action)
    if not allowed[action_index]:
        raise AssertionError("Executed action absent from decision-time allowed set")
    if terminal or new_state is None:
        next_features = None
        next_allowed = np.zeros(len(ACTIONS), dtype=bool)
    else:
        next_key = next_decision_key(new_state)
        next_features, next_allowed = self.observations[next_key]
        if not next_allowed.any():
            raise AssertionError("Nonterminal next-state allowed set is empty")
    reward = reward_from_transition(old_state, action, new_state, events)
    self.transitions.append(Transition(
        features.copy(), action_index, reward,
        None if next_features is None else next_features.copy(),
        terminal or new_state is None, allowed.copy(), next_allowed.copy(),
        tuple(events), str(getattr(self, "scenario_tag", "unspecified")),
        tuple(getattr(self, "lineup_tag", ())), int(old_state["round"]),
        getattr(self, "board_seed", None), int(old_state["step"]),
        CHECKPOINT_SCHEMAS,
    ))
    self.round_reward += reward
    self.round_steps += 1
    self.round_events.update(events)


def game_events_occurred(self, old_game_state, self_action,
                         new_game_state, events):
    if old_game_state is None or self_action is None:
        return
    if self.pending is not None:
        remember(self, *self.pending)
    if new_game_state is not None:
        self.previous_action = ACTIONS.index(self_action)
        next_decision = next_decision_state(new_game_state)
        note_progress_events(self, events, next_decision["step"])
        observe(self, next_decision)
    self.pending = (old_game_state, self_action, new_game_state, list(events))


def _tag_weight(tag, weights, lineups_per_scenario):
    scenario, lineup = tag
    if weights is None:
        return 1.0 / len(lineups_per_scenario[scenario])
    full_tag = scenario + "|" + ",".join(lineup)
    if full_tag in weights:
        return float(weights[full_tag])
    return float(weights.get(scenario, 0.0)) / len(lineups_per_scenario[scenario])


def balanced_records(transitions, weights=None, max_records=MAX_FIT_RECORDS,
                     return_metadata=False):
    """Select unique records while preserving weighted scenario mass.

    A rare tag is capped at its available records instead of setting the
    global sample size to the rarest tag.  Per-record weights then give each
    selected tag its requested effective share.  The returned metadata makes
    both the available and selected counts auditable.
    """
    groups = {}
    for record in transitions:
        groups.setdefault((record.scenario, record.lineup), []).append(record)
    if not groups:
        empty = ([], np.asarray([], dtype=float), {
            "available": {}, "selected": {}, "effective": {},
        })
        return empty if return_metadata else []
    lineups_per_scenario = {}
    for scenario, lineup in groups:
        lineups_per_scenario.setdefault(scenario, set()).add(lineup)
    tag_weights = {
        tag: _tag_weight(tag, weights, lineups_per_scenario)
        for tag in groups
    }
    tag_weights = {
        tag: weight for tag, weight in tag_weights.items() if weight > 0
    }
    if not tag_weights:
        raise ValueError("No positive replay weight for an available tag")

    available = {tag: len(groups[tag]) for tag in tag_weights}
    budget = min(int(max_records), sum(available.values()))
    weight_total = float(sum(tag_weights.values()))
    fractions = {
        tag: float(tag_weights[tag]) / weight_total for tag in tag_weights
    }
    ideal = {tag: budget * fractions[tag] for tag in tag_weights}
    selected = {
        tag: min(available[tag], int(np.floor(ideal[tag])))
        for tag in tag_weights
    }
    # Redistribute unused budget to the largest unmet target.  This fills the
    # rest of a large group when a new scenario has only a small buffer.
    while sum(selected.values()) < budget:
        candidates = [tag for tag in tag_weights if selected[tag] < available[tag]]
        if not candidates:
            break
        tag = max(candidates, key=lambda item: (
            ideal[item] - selected[item], fractions[item], str(item),
        ))
        selected[tag] += 1

    result = []
    result_weights = []
    for tag in sorted(tag_weights):
        rows = groups[tag]
        count = selected[tag]
        if count <= 0:
            continue
        indices = np.linspace(0, len(rows) - 1, count, dtype=int)
        result.extend(rows[index] for index in indices)
        result_weights.extend(
            [float(tag_weights[tag]) / count] * count
        )

    effective = Counter()
    for record, sample_weight in zip(result, result_weights):
        effective[record.scenario] += sample_weight
    if weight_total:
        effective = {
            scenario: value / weight_total
            for scenario, value in effective.items()
        }
    metadata = {
        "available": available,
        "selected": {
            tag: selected[tag] for tag in sorted(selected)
            if selected[tag] > 0
        },
        "effective": effective,
    }
    if return_metadata:
        return result, np.asarray(result_weights, dtype=float), metadata
    return result


def fitted_targets(records, trees):
    targets = np.asarray([record.reward for record in records], dtype=float)
    continuing = [index for index, row in enumerate(records) if not row.terminal]
    if continuing:
        states = np.stack([records[index].next_features for index in continuing])
        masks = np.stack([records[index].next_allowed for index in continuing])
        if not masks.any(axis=1).all():
            raise AssertionError("Nonterminal backup has an empty allowed set")
        values = predict_values(trees, states)
        values[~masks] = -np.inf
        targets[continuing] += DISCOUNT * values.max(axis=1)
    return targets


def fit_trees(self):
    started = perf_counter()
    records, sample_weights, metadata = balanced_records(
        self.transitions, self.replay_weights, return_metadata=True,
    )
    if not records:
        return
    self.last_fit_records = len(records)
    self.last_fit_tag_counts = dict(Counter((row.scenario, row.lineup)
                                            for row in records))
    self.last_available_tag_counts = metadata["available"]
    self.last_effective_scenario_weights = metadata["effective"]
    self.last_action_support = {
        "allowed": [sum(bool(row.allowed[index]) for row in records)
                    for index in range(len(ACTIONS))],
        "executed": [sum(row.action == index for row in records)
                     for index in range(len(ACTIONS))],
    }
    augmented = [transform_transition(record, transform)
                 for transform in TRANSFORMS for record in records]
    augmented_weights = np.tile(sample_weights, len(TRANSFORMS))
    states = np.stack([row.features for row in augmented])
    actions = np.asarray([row.action for row in augmented])
    for _ in range(FIT_ITERATIONS):
        targets = fitted_targets(augmented, self.trees)
        replacements = list(self.trees)
        for action in range(len(ACTIONS)):
            selected = actions == action
            if selected.any():
                model = DecisionTreeRegressor(
                    max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
                    random_state=int(getattr(self, "seed", 0) or 0),
                )
                model.fit(
                    states[selected], targets[selected],
                    sample_weight=augmented_weights[selected],
                )
                replacements[action] = model
        self.trees = replacements
    self.fit_seconds += perf_counter() - started


def end_of_round(self, last_game_state, last_action, events):
    if self.pending is not None:
        old_state, action, new_state, prior_events = self.pending
        if (last_game_state is not None and
                state_key(old_state) == state_key(last_game_state)):
            # The engine's final event list already includes prior_events.
            remember(self, old_state, action, None, list(events), True)
            last_game_state = None
        else:
            remember(self, old_state, action, new_state, prior_events)
    if last_game_state is not None and last_action is not None:
        remember(self, last_game_state, last_action, None, list(events), True)
    self.pending = None
    self.completed_rounds += 1
    if self.completed_rounds % FIT_EVERY_ROUNDS == 0:
        fit_trees(self)
    self.last_round_reward = self.round_reward
    self.last_round_steps = self.round_steps
    self.last_round_events = dict(self.round_events)
    self.env_steps += self.round_steps
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    if self.completed_rounds % FIT_EVERY_ROUNDS == 0:
        save_checkpoint(self, self.model_path)
    self.round_reward = 0.0
    self.round_steps = 0
    self.round_events.clear()
