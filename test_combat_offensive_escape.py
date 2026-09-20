"""Regressions for real callback dispatch, escape geometry, and warm starts."""

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import events as e
from test_combat_adversarial_window import open_state
from agent_code.Agent_029_combat_ddqn_adversarial_window_agent import callbacks as old
from agent_code.Agent_030_combat_ddqn_escape_replay_agent import callbacks as control
from agent_code.Agent_030_combat_ddqn_escape_replay_agent import train
from agent_code.Agent_031_combat_ddqn_offensive_escape_agent import callbacks as candidate
from agent_code.Agent_031_combat_ddqn_offensive_escape_agent.features import (
    offensive_features, state_to_features, _survivor_tiles,
)
from agent_code.combat_dqn_agent.model import DEVICE, QNetwork, load_checkpoint, save_checkpoint
from prepare_offensive_checkpoint import expand_checkpoint


class OffensiveTests(unittest.TestCase):
    def setUp(self):
        from agent_code.combat_dqn_agent import callbacks as base
        for module in (old, control, candidate, base):
            for name in ("MODEL_PATH", "RESUME_PATH"):
                patcher = patch.object(module, name, getattr(module, name))
                patcher.start()
                self.addCleanup(patcher.stop)

    def test_trap_and_open_escape_are_distinguished(self):
        enemy = ("enemy", 0, True, (9, 7))
        state = open_state((enemy,))
        open_features = offensive_features(state).reshape(3, 4)[0]
        state["field"][:] = -1
        state["field"][7:10, 7] = 0
        trapped = offensive_features(state).reshape(3, 4)[0]
        self.assertEqual(open_features[0], 1)
        self.assertEqual(open_features[2], 0)
        self.assertEqual(trapped[1], 1)
        self.assertEqual(trapped[2], 1)
        self.assertEqual(trapped[3], 1)

    def test_prefix_missing_opponents_and_permutation(self):
        from agent_code.Agent_029_combat_ddqn_adversarial_window_agent.features import state_to_features as base
        state = open_state((("a", 0, True, (10, 7)), ("b", 0, True, (3, 3))))
        result = state_to_features(state)
        self.assertEqual(result.shape, (113,))
        np.testing.assert_array_equal(result[:101], base(state))
        state["others"].reverse()
        np.testing.assert_array_equal(state_to_features(state), result)
        self.assertTrue(np.all(result[-4:] == 0))
        self.assertTrue(np.all(offensive_features(open_state()) == 0))
        state["self"] = ("learner", 0, False, (7, 7))
        self.assertTrue(np.all(offensive_features(state).reshape(3, 4)[:, :3] == 0))

    def test_search_checks_lingering_flames(self):
        field = np.full((7, 7), -1)
        field[3, 3] = 0
        self.assertEqual(_survivor_tiles(field, (3, 3), {(3, 3): {3, 4}}, [], 5), set())

    def test_callbacks_commit_escape_and_terminal_once_and_persist_epsilon(self):
        for callbacks, dim in ((control, 101), (candidate, 113)):
            with self.subTest(dim=dim), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pkl"
                learner = SimpleNamespace(train=True, seed=0, model_path=path,
                                          logger=logging.getLogger("test"))
                with patch.object(callbacks, "RESUME_PATH", None):
                    callbacks.setup(learner)
                    train.setup_training(learner)
                learner.env_steps = 200_000  # reproduces the old reset bug
                states = []
                for step in range(1, 7):
                    state = open_state((("enemy", 0, True, (10, 7)),))
                    state["step"] = step
                    if step >= 5:
                        state["others"] = []  # still retain escape outcome
                    states.append(state)
                    old._features_for_state(learner, state)
                for index in range(5):
                    train.game_events_occurred(
                        learner, states[index], "BOMB" if index == 0 else "WAIT",
                        states[index + 1], [e.BOMB_DROPPED] if index == 0 else [e.WAITED],
                    )
                train.end_of_round(learner, states[4], "WAIT", [e.KILLED_SELF, e.GOT_KILLED])
                self.assertEqual(len(learner.replay_buffer), 5)
                self.assertEqual(learner.replay_buffer.tags, ["combat_escape"] * 5)
                self.assertEqual(learner.combat_env_steps, 5)
                self.assertGreater(learner.epsilon, .19)
                final = learner.replay_buffer.storage[-1]
                self.assertTrue(final[4])
                self.assertFalse(final[5].any())
                self.assertEqual(final[0].shape, (dim,))
                self.assertEqual(load_checkpoint(path)["combat_env_steps"], 5)
                restored = SimpleNamespace(train=True, seed=0, model_path=Path(directory)/"restored.pkl")
                with patch.object(callbacks, "RESUME_PATH", path):
                    callbacks.setup(restored)
                    train.setup_training(restored)
                self.assertEqual(restored.combat_env_steps, 5)
                self.assertEqual(restored.epsilon, learner.epsilon)

    def test_migration_preserves_q_values_and_optimizer_moments(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = SimpleNamespace(train=True, seed=0, model_path=Path(directory)/"source.pkl")
            with patch.object(old, "RESUME_PATH", None):
                old.setup(learner)
            x = torch.randn(12, 101, device=DEVICE)
            learner.policy_net(x).sum().backward()
            learner.optimizer.step()  # create nonzero Adam moments
            save_checkpoint(learner, learner.model_path)
            source = load_checkpoint(learner.model_path)
            migrated = expand_checkpoint(source)
            extended = QNetwork(113).to(DEVICE)
            extended.load_state_dict(migrated["policy_state_dict"])
            torch.testing.assert_close(
                extended(torch.cat((x, torch.randn(12, 12, device=DEVICE)), 1)),
                learner.policy_net(x),
            )
            optimizer = torch.optim.Adam(extended.parameters())
            optimizer.load_state_dict(migrated["optimizer_state_dict"])
            first_id = source["optimizer_state_dict"]["param_groups"][0]["params"][0]
            old_moment = source["optimizer_state_dict"]["state"][first_id]["exp_avg"]
            new_moment = optimizer.state[next(extended.parameters())]["exp_avg"]
            torch.testing.assert_close(new_moment[:, :101], old_moment)
            self.assertEqual(torch.count_nonzero(new_moment[:, 101:]).item(), 0)
            extended(torch.randn(4, 113, device=DEVICE)).sum().backward()
            optimizer.step()  # catches incorrectly sized optimizer tensors


if __name__ == "__main__":
    unittest.main()
