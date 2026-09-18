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

    def test_rejects_bomb_when_opponent_can_claim_only_escape(self):
        # This models the observed Classic failures: the static corridor is
        # escapable, but an opponent can enter its only exit before detonation.
        field = np.full((7, 7), -1, dtype=int)
        for position in ((3, 1), (3, 2), (3, 3), (4, 3), (5, 3)):
            field[position] = 0
        field[3, 4] = 1
        state = make_state(
            field, position=(3, 1),
            others=[("enemy", 0, True, (5, 3))],
        )
        self.assertFalse(can_survive_action(state, "BOMB"))
        self.assertNotIn(ACTIONS.index("BOMB"), safe_action_indices(state))

    def test_accepts_bomb_when_one_escape_stays_clear_of_opponent(self):
        field = open_field(9)
        field[5, 4] = 1
        state = make_state(
            field, position=(4, 4),
            others=[("enemy", 0, True, (7, 7))],
        )
        self.assertTrue(can_survive_action(state, "BOMB"))

    def test_rejects_bomb_without_one_step_escape_margin(self):
        # Four moves reach safety on the detonation step.  That was accepted
        # before, but in combat leaves no allowance for a delayed move.
        field = np.full((7, 7), -1, dtype=int)
        for position in ((3, 1), (3, 2), (3, 3), (3, 4), (4, 4), (1, 5)):
            field[position] = 0
        field[2, 1] = 1
        state = make_state(
            field, position=(3, 1),
            others=[("enemy", 0, True, (1, 5))],
        )
        self.assertFalse(can_survive_action(state, "BOMB"))
        self.assertNotIn(ACTIONS.index("BOMB"), safe_action_indices(state))

    def test_solo_bomb_keeps_last_step_escape_route(self):
        # Solo Loot Crate has no collision uncertainty, so retain the original
        # ability to leave the blast on the final movement opportunity.
        field = np.full((7, 7), -1, dtype=int)
        for position in ((3, 1), (3, 2), (3, 3), (3, 4), (4, 4)):
            field[position] = 0
        field[2, 1] = 1
        state = make_state(field, position=(3, 1))
        self.assertTrue(can_survive_action(state, "BOMB"))

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
