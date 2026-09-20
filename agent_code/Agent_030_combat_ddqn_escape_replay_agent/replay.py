"""Scenario replay with dedicated samples for combat bomb-escape sequences."""

from agent_code.Agent_027_combat_ddqn_short_cycle_staged_replay_agent.replay import (
    ScenarioReplayBuffer,
)


ESCAPE_TAG = "combat_escape"
ESCAPE_FRACTION = 0.20


class CombatEscapeReplayBuffer(ScenarioReplayBuffer):
    """Reserve replay capacity for moves from a combat bomb to its outcome."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transition_tag = None

    def set_context(self, tag, weights):
        super().set_context(
            tag,
            {name: (1.0 - ESCAPE_FRACTION) * weight
             for name, weight in weights.items()} | {ESCAPE_TAG: ESCAPE_FRACTION},
        )

    def add(self, *args, **kwargs):
        original_tag = self.current_tag
        if self.transition_tag is not None:
            self.current_tag = self.transition_tag
        try:
            super().add(*args, **kwargs)
        finally:
            self.current_tag = original_tag
            self.transition_tag = None
