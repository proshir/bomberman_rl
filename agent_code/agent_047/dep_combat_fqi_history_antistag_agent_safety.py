"""Shared deterministic bomb-safety rules for the combat FQI agents.

The safety implementation is deliberately shared with the original combat
agent.  This keeps the controlled experiment focused on the new history
features and prevents two copies of safety-critical code from drifting.
"""

# Sahand was here.

from .dep_combat_fqi_agent_safety import (  # noqa: F401
    ACTIONS,
    MOVE_DELTAS,
    action_is_legal,
    action_survival_time,
    best_survival_action_indices,
    blast_tiles,
    bomb_is_useful,
    bomb_value,
    build_danger_schedule,
    can_survive_action,
    earliest_danger,
    escape_distance_after_bomb,
    legal_action_indices,
    safe_action_indices,
)

__all__ = [
    "ACTIONS", "MOVE_DELTAS", "action_is_legal", "action_survival_time",
    "best_survival_action_indices", "blast_tiles", "bomb_is_useful",
    "bomb_value", "build_danger_schedule", "can_survive_action",
    "earliest_danger", "escape_distance_after_bomb",
    "legal_action_indices", "safe_action_indices",
]
