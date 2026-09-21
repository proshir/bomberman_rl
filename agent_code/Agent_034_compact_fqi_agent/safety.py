"""Engine-aligned, time-indexed safety rules shared with the audited FQI agent."""

from agent_code.combat_fqi_agent.safety import (  # noqa: F401
    ACTIONS, MOVE_DELTAS, action_is_legal, action_survival_time,
    best_survival_action_indices, blast_tiles, bomb_value,
    build_danger_schedule, can_survive_action, legal_action_indices,
    safe_action_indices,
)


def allowed_action_indices(game_state):
    """Return the actual policy/backup set, including the emergency fallback."""
    safe = safe_action_indices(game_state)
    return safe if safe else best_survival_action_indices(game_state)
