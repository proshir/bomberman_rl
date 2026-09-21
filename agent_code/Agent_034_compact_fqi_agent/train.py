"""Scenario-tagged, frozen-target fitted Q iteration for Agent 034."""

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
from sklearn.tree import DecisionTreeRegressor

import events as e
from agent_code.combat_fqi_agent.safety import bomb_is_useful, earliest_danger
from .callbacks import observe, predict_values, save_checkpoint, state_key
from .features import FEATURE_SCHEMA
from .safety import ACTIONS


DISCOUNT = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
BUFFER_SIZE_PER_TAG = 40000
FIT_EVERY_ROUNDS = 20
FIT_ITERATIONS = 5
MAX_DEPTH = 8
MIN_SAMPLES_LEAF = 5
SAFETY_SCHEMA = "combat-fqi-time-safety-v1"
REWARD_SCHEMA = "combat-fqi-reward-v1"
HISTORY_SCHEMA = "previous-eight-score-crate-progress-v2"


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
    """Keep a bounded, independent history for every scenario and lineup."""

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
    self.epsilon = EPSILON_START
    self.transitions = TaggedReplay(BUFFER_SIZE_PER_TAG)
    self.pending = None
    self.completed_rounds = 0
    self.round_reward = 0.0
    self.round_steps = 0
    self.round_events = Counter()


def reward_from_transition(old_state, action, new_state, events):
    """Keep the audited combat FQI reward accounting for a controlled bridge."""
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
    if not allowed[ACTIONS.index(action)]:
        raise AssertionError("Executed action absent from decision-time allowed set")
    if terminal or new_state is None:
        next_features = None
        next_allowed = np.zeros(len(ACTIONS), dtype=bool)
    else:
        # The engine reports the post-action state with the same step number
        # as the pre-action decision. The following decision is step + 1.
        next_key = (int(new_state["round"]), int(new_state["step"]) + 1)
        next_features, next_allowed = self.observations[next_key]
        if not next_allowed.any():
            raise AssertionError("Nonterminal next-state allowed set is empty")
    reward = reward_from_transition(old_state, action, new_state, events)
    self.transitions.append(Transition(
        features.copy(), ACTIONS.index(action), reward,
        None if next_features is None else next_features.copy(),
        terminal or new_state is None, allowed.copy(), next_allowed.copy(),
        tuple(events), str(getattr(self, "scenario_tag", "unspecified")),
        tuple(getattr(self, "lineup_tag", ())), int(old_state["round"]),
        getattr(self, "board_seed", None), int(old_state["step"]),
        (FEATURE_SCHEMA, SAFETY_SCHEMA, REWARD_SCHEMA, HISTORY_SCHEMA),
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
        next_decision_state = dict(new_game_state)
        next_decision_state["step"] = int(new_game_state["step"]) + 1
        observe(self, next_decision_state)
    self.pending = (old_game_state, self_action, new_game_state, list(events))


def balanced_records(transitions):
    """Retain all tagged data and fit on equal sized scenario samples."""
    groups = {}
    for record in transitions:
        groups.setdefault((record.scenario, record.lineup), []).append(record)
    if len(groups) <= 1:
        return list(transitions)
    count = min(map(len, groups.values()))
    # Deterministic evenly spaced sampling prevents a large later stage from
    # swamping an earlier solo task without duplicating individual records.
    result = []
    for tag in sorted(groups):
        rows = groups[tag]
        result.extend(rows[index] for index in np.linspace(
            0, len(rows) - 1, count, dtype=int))
    return result


def fitted_targets(records, trees):
    rewards = np.asarray([record.reward for record in records], dtype=float)
    targets = rewards.copy()
    continuing = [index for index, row in enumerate(records) if not row.terminal]
    if continuing:
        next_states = np.stack([records[index].next_features for index in continuing])
        masks = np.stack([records[index].next_allowed for index in continuing])
        if not masks.any(axis=1).all():
            raise AssertionError("Nonterminal backup has an empty allowed set")
        values = predict_values(trees, next_states)
        values[~masks] = -np.inf
        targets[continuing] += DISCOUNT * values.max(axis=1)
    return targets


def fit_trees(self):
    records = balanced_records(self.transitions)
    if not records:
        return
    states = np.stack([row.features for row in records])
    actions = np.asarray([row.action for row in records])
    for _ in range(FIT_ITERATIONS):
        targets = fitted_targets(records, self.trees)
        replacements = list(self.trees)
        for action in range(len(ACTIONS)):
            selected = actions == action
            if selected.any():
                model = DecisionTreeRegressor(
                    max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
                    random_state=int(getattr(self, "seed", 0) or 0),
                )
                model.fit(states[selected], targets[selected])
                replacements[action] = model
        self.trees = replacements


def end_of_round(self, last_game_state, last_action, events):
    if self.pending is not None:
        old_state, action, new_state, prior_events = self.pending
        if (last_game_state is not None and
                state_key(old_state) == state_key(last_game_state)):
            remember(self, old_state, action, None, prior_events + list(events), True)
            last_game_state = None
        else:
            remember(self, old_state, action, new_state, prior_events)
    if last_game_state is not None and last_action is not None:
        remember(self, last_game_state, last_action, None, list(events), True)
    self.pending = None
    self.completed_rounds += 1
    if self.completed_rounds % FIT_EVERY_ROUNDS == 0:
        fit_trees(self)
        save_checkpoint(self, self.model_path)
    self.last_round_reward = self.round_reward
    self.last_round_steps = self.round_steps
    self.last_round_events = dict(self.round_events)
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.round_reward = 0.0
    self.round_steps = 0
    self.round_events.clear()
