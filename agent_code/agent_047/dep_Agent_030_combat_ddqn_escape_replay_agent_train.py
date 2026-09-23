"""Continue Agent 029 with combat exploration and bomb-escape replay."""

from .dep_Agent_027_combat_ddqn_short_cycle_staged_replay_agent_train import *  # noqa: F401,F403
from . import dep_combat_dqn_agent_train as _base
from . import dep_combat_dqn_agent_callbacks as _base_callbacks
from .dep_combat_dqn_agent_model import load_checkpoint
from .dep_combat_dqn_agent_callbacks import state_key
import events as e
import settings as s

from .dep_Agent_030_combat_ddqn_escape_replay_agent_replay import CombatEscapeReplayBuffer, ESCAPE_TAG


COMBAT_EPSILON_START = 0.20
COMBAT_EPSILON_END = 0.05
COMBAT_EPSILON_DECAY_STEPS = 60_000


def _combat_epsilon(step):
    fraction = min(1.0, step / COMBAT_EPSILON_DECAY_STEPS)
    return COMBAT_EPSILON_START + fraction * (
        COMBAT_EPSILON_END - COMBAT_EPSILON_START
    )


def setup_training(self):
    _base.setup_training(self)
    self.replay_buffer = CombatEscapeReplayBuffer(
        REPLAY_CAPACITY, getattr(self, "seed", None)
    )
    self.replay_buffer.set_context("coin-heaven", {"coin-heaven": 1.0})
    checkpoint = (load_checkpoint(_base_callbacks.RESUME_PATH)
                  if _base_callbacks.RESUME_PATH is not None else {})
    self.combat_env_steps = int(checkpoint.get("combat_env_steps", 0))
    self.bomb_escape_steps_remaining = 0
    self.epsilon = _combat_epsilon(self.combat_env_steps)


def remember(self, old_state, action, new_state, events,
             next_decision_state=None):
    # Include the placement move and every following decision through the
    # explosion, so both successful exits and blocked/self-destructive exits
    # receive explicit replay coverage.
    placed_combat_bomb = e.BOMB_DROPPED in events and bool(old_state["others"])
    is_combat_escape = placed_combat_bomb or self.bomb_escape_steps_remaining > 0
    self.replay_buffer.transition_tag = ESCAPE_TAG if is_combat_escape else None
    _base.remember(self, old_state, action, new_state, events,
                   next_decision_state=next_decision_state)

    if placed_combat_bomb:
        self.bomb_escape_steps_remaining = s.BOMB_TIMER
    elif self.bomb_escape_steps_remaining:
        self.bomb_escape_steps_remaining -= 1
    if new_state is None:
        self.bomb_escape_steps_remaining = 0

    self.combat_env_steps += 1
    self.epsilon = _combat_epsilon(self.combat_env_steps)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    # Imported functions retain their defining module's globals. Dispatch to
    # this module's remember explicitly, including the delayed transition.
    if old_game_state is None or self_action is None:
        return
    if self.pending is not None:
        remember(self, *self.pending, next_decision_state=old_game_state)
    self.pending = (old_game_state, self_action, new_game_state, list(events))


def end_of_round(self, last_game_state, last_action, events):
    if self.pending is not None:
        same_final = (last_game_state is not None and
                      state_key(self.pending[0]) == state_key(last_game_state))
        if not same_final:
            remember(self, *self.pending, next_decision_state=last_game_state)
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
    _base.save_checkpoint(self, self.model_path)
    self.round_reward = 0.0
    self.round_steps = 0
    self.round_events.clear()
    self.round_loss_total = 0.0
    self.round_loss_count = 0
    self.bomb_escape_steps_remaining = 0
