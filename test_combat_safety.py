"""Focused checks for the rules that keep the combat agent alive."""

# Sahand was here.

import unittest

import numpy as np

from agent_code.combat_fqi_agent.safety import (
    ACTIONS,
    action_is_legal,
    blast_tiles,
    build_danger_schedule,
    can_survive_action,
    safe_action_indices,
)


def make_state(field, position=(3, 3), bombs=(), others=(), bomb_available=True,
               explosion_map=None):
    if explosion_map is None:
        explosion_map = np.zeros_like(field)
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "bombs": list(bombs),
        "explosion_map": explosion_map,
        "coins": [],
        "self": ("combat", 0, bomb_available, position),
        "others": list(others),
        "user_input": None,
    }


def open_field(size=7):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    return field


class CombatSafetyTest(unittest.TestCase):
    def test_blast_stops_at_stone_wall(self):
        field = open_field()
        field[4, 3] = -1
        tiles = blast_tiles(field, (3, 3))
        self.assertNotIn((4, 3), tiles)
        self.assertNotIn((5, 3), tiles)

    def test_blast_matches_framework_and_continues_through_crate(self):
        field = open_field()
        field[4, 3] = 1
        tiles = blast_tiles(field, (3, 3))
        self.assertIn((4, 3), tiles)
        self.assertIn((5, 3), tiles)

    def test_current_explosion_is_immediately_dangerous(self):
        field = open_field()
        explosion = np.zeros_like(field)
        explosion[3, 3] = 1
        state = make_state(field, explosion_map=explosion)
        self.assertIn(1, build_danger_schedule(state)[(3, 3)])
        self.assertNotIn(ACTIONS.index("WAIT"), safe_action_indices(state))

    def test_rejects_bomb_when_there_is_no_escape(self):
        field = np.full((7, 7), -1, dtype=int)
        field[3, 3] = 0
        state = make_state(field)
        self.assertFalse(can_survive_action(state, "BOMB"))

    def test_accepts_useful_bomb_with_escape_route(self):
        field = open_field()
        field[5, 3] = 1
        state = make_state(field)
        self.assertTrue(can_survive_action(state, "BOMB"))
        self.assertIn(ACTIONS.index("BOMB"), safe_action_indices(state))

    def test_bomb_and_opponent_tiles_are_not_legal_destinations(self):
        field = open_field()
        state = make_state(
            field,
            bombs=[((3, 2), 3)],
            others=[("enemy", 0, True, (4, 3))],
        )
        self.assertFalse(action_is_legal(state, "UP"))
        self.assertFalse(action_is_legal(state, "RIGHT"))

    def test_cannot_bomb_without_bomb_available(self):
        state = make_state(open_field(), bomb_available=False)
        self.assertFalse(action_is_legal(state, "BOMB"))

    def test_useless_bomb_is_not_offered_to_the_learner(self):
        state = make_state(open_field())
        self.assertNotIn(ACTIONS.index("BOMB"), safe_action_indices(state))


if __name__ == "__main__":
    unittest.main()
