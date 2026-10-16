import unittest
import sys
from pathlib import Path

import numpy as np

WM_ROOT = Path(__file__).resolve().parents[1] / "wm"
if str(WM_ROOT) not in sys.path:
    sys.path.insert(0, str(WM_ROOT))

from domain.minigrid.minigrid_custom_env import INTERNAL_NOOP_ACTION, CustomMiniGridEnv
from domain.minigrid.action_codec import MODEL_ACTION_COUNT, compact_to_native
from minigrid.core.actions import Actions
from minigrid.core.world_object import Door, Key
from minigrid.wrappers import FullyObsWrapper


LAYOUT = "WWWWW\nWSEGW\nWWWWW"
COLORS = "WWWWW\nWEEGW\nWWWWW"
KEY_LAYOUT = "WWWWW\nWSKGW\nWWWWW"
KEY_COLORS = "WWWWW\nWYYGW\nWWWWW"
ALL_ACTIONS = tuple(Actions(i) for i in range(6))


def make_env(max_steps=20, **kwargs):
    return CustomMiniGridEnv(
        layout_str=LAYOUT,
        color_str=COLORS,
        agent_start_dir=0,
        max_steps=max_steps,
        **kwargs,
    )


def state_snapshot(env):
    carrying = env.carrying
    carrying_value = None if carrying is None else (carrying.type, carrying.color)
    return (
        tuple(env.agent_pos),
        int(env.agent_dir),
        env.grid.encode().copy(),
        carrying,
        carrying_value,
    )


class StochasticMiniGridEnvTests(unittest.TestCase):
    def test_default_is_deterministic_and_info_uses_new_probability_name(self):
        env = make_env()
        obs, _ = env.reset(seed=7)
        next_obs, reward, terminated, truncated, info = env.step(Actions.forward)

        self.assertEqual(tuple(env.agent_pos), (2, 1))
        self.assertEqual(obs["image"].shape, next_obs["image"].shape)
        self.assertFalse(info["stochastic_enabled"])
        self.assertFalse(info["action_failed"])
        self.assertEqual(info["requested_action"], int(Actions.forward))
        self.assertEqual(info["executed_action"], int(Actions.forward))
        self.assertEqual(info["action_failure_prob"], 0.2)
        self.assertNotIn("move_failure_prob", info)
        self.assertIsInstance(reward, (int, float, np.number))
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_probability_extremes_cover_all_six_native_actions(self):
        for action in ALL_ACTIONS:
            with self.subTest(action=action):
                env = make_env(stochastic_enabled=True, action_failure_prob=1.0)
                env.reset(seed=1)
                before = state_snapshot(env)
                _, reward, terminated, truncated, info = env.step(action)
                after = state_snapshot(env)
                self.assertEqual(after[:2], before[:2])
                np.testing.assert_array_equal(after[2], before[2])
                self.assertIs(after[3], before[3])
                self.assertEqual(after[4], before[4])
                self.assertEqual(env.step_count, 1)
                self.assertTrue(info["action_failed"])
                self.assertEqual(info["requested_action"], int(action))
                self.assertEqual(info["executed_action"], INTERNAL_NOOP_ACTION)
                self.assertEqual(reward, 0)
                self.assertFalse(terminated)
                self.assertFalse(truncated)

            env = make_env(stochastic_enabled=True, action_failure_prob=0.0)
            env.reset(seed=1)
            _, _, _, _, info = env.step(action)
            self.assertFalse(info["action_failed"])
            self.assertEqual(info["executed_action"], int(action))

            env = make_env(stochastic_enabled=False)
            env.reset(seed=1)
            _, _, _, _, info = env.step(action)
            self.assertFalse(info["action_failed"])
            self.assertEqual(info["executed_action"], int(action))
            self.assertFalse(info["stochastic_enabled"])

    def test_probability_one_does_not_change_carried_object_or_world(self):
        env = make_env(
            stochastic_enabled=True,
            action_failure_prob=1.0,
            initial_carrying_key_color="yellow",
        )
        env.reset(seed=3)
        before = state_snapshot(env)
        env.step(Actions.drop)
        after = state_snapshot(env)
        np.testing.assert_array_equal(after[2], before[2])
        self.assertEqual(after[:2], before[:2])
        self.assertIs(after[3], before[3])
        self.assertEqual(after[4], before[4])

    def test_valid_pickup_drop_toggle_open_close_and_unlock_execute_at_zero(self):
        env = CustomMiniGridEnv(
            layout_str=KEY_LAYOUT,
            color_str=KEY_COLORS,
            agent_start_dir=0,
            stochastic_enabled=True,
            action_failure_prob=0.0,
            max_steps=20,
        )
        env.reset(seed=2)
        key_pos = (2, 1)
        env.step(Actions.pickup)
        self.assertIsInstance(env.carrying, Key)
        self.assertIsNone(env.grid.get(*key_pos))

        env.step(Actions.drop)
        self.assertIsNone(env.carrying)
        self.assertIsInstance(env.grid.get(*key_pos), Key)

        env.grid.set(*key_pos, Door("yellow", is_locked=False, is_open=False))
        env.step(Actions.toggle)
        door = env.grid.get(*key_pos)
        self.assertTrue(door.is_open)
        env.step(Actions.toggle)
        self.assertFalse(door.is_open)

        env.carrying = Key("yellow")
        key = env.carrying
        env.grid.set(*key_pos, Door("yellow", is_locked=True, is_open=False))
        env.step(Actions.toggle)
        door = env.grid.get(*key_pos)
        self.assertFalse(door.is_locked)
        self.assertTrue(door.is_open)
        self.assertIs(env.carrying, key)

    def test_valid_interactions_preserve_state_on_failure(self):
        for interaction in ("pickup", "drop", "open", "close", "unlock"):
            with self.subTest(interaction=interaction):
                env = make_env(stochastic_enabled=True, action_failure_prob=1.0)
                self.addCleanup(env.close)
                env.reset(seed=2)
                action = Actions.toggle
                if interaction == "pickup":
                    env.grid.set(2, 1, Key("yellow"))
                    action = Actions.pickup
                elif interaction == "drop":
                    env.carrying = Key("yellow")
                    action = Actions.drop
                else:
                    env.grid.set(2, 1, Door(
                        "yellow", is_open=interaction == "close", is_locked=interaction == "unlock"))
                    if interaction == "unlock":
                        env.carrying = Key("yellow")
                before = state_snapshot(env)
                _, reward, _, _, info = env.step(action)
                after = state_snapshot(env)
                self.assertEqual(after[:2], before[:2])
                np.testing.assert_array_equal(after[2], before[2])
                self.assertIs(after[3], before[3])
                self.assertEqual(after[4], before[4])
                self.assertTrue(info["action_failed"])
                self.assertEqual(reward, 0)

    def test_ineffective_interactions_do_not_report_failure_at_zero(self):
        env = make_env(stochastic_enabled=True, action_failure_prob=0.0)
        env.reset(seed=4)
        for action in (Actions.pickup, Actions.drop, Actions.toggle):
            with self.subTest(action=action):
                _, _, _, _, info = env.step(action)
                self.assertFalse(info["action_failed"])
                self.assertEqual(info["executed_action"], int(action))

        env.grid.set(2, 1, Door("yellow", is_locked=True, is_open=False))
        _, _, _, _, info = env.step(Actions.toggle)
        self.assertFalse(info["action_failed"])
        self.assertEqual(info["executed_action"], int(Actions.toggle))
        self.assertTrue(env.grid.get(2, 1).is_locked)

    def test_enabled_draws_once_per_valid_action_and_disabled_draws_none(self):
        seed = 91
        for enabled, expected_draws in ((False, 0), (True, 6)):
            with self.subTest(enabled=enabled):
                env = make_env(stochastic_enabled=enabled, action_failure_prob=0.0)
                env.reset(seed=seed)
                for action in ALL_ACTIONS:
                    env.step(action)
                expected_rng = np.random.default_rng(seed)
                expected_rng.random(expected_draws)
                self.assertEqual(env.np_random.random(), expected_rng.random())

    def test_seed_reproduces_mixed_six_action_failure_trace(self):
        actions = ALL_ACTIONS * 4

        def rollout():
            env = make_env(stochastic_enabled=True, action_failure_prob=0.35)
            obs, _ = env.reset(seed=123)
            trace = [(obs["image"].copy(), state_snapshot(env))]
            failures = []
            for action in actions:
                obs, _, _, _, info = env.step(action)
                trace.append((obs["image"].copy(), state_snapshot(env)))
                failures.append(bool(info["action_failed"]))
            return trace, failures

        first, first_failures = rollout()
        second, second_failures = rollout()
        self.assertEqual(first_failures, second_failures)
        for (first_image, first_state), (second_image, second_state) in zip(first, second):
            np.testing.assert_array_equal(first_image, second_image)
            self.assertEqual(first_state[:2], second_state[:2])
            np.testing.assert_array_equal(first_state[2], second_state[2])
            self.assertEqual(first_state[4], second_state[4])

    def test_probability_one_noop_truncates_at_max_steps(self):
        env = make_env(max_steps=2, stochastic_enabled=True, action_failure_prob=1.0)
        env.reset(seed=0)
        _, reward, terminated, truncated, _ = env.step(Actions.forward)
        self.assertEqual(reward, 0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        _, reward, terminated, truncated, info = env.step(Actions.toggle)
        self.assertEqual(reward, 0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["action_failed"])
        self.assertEqual(env.step_count, 2)

    def test_uniform_only_randomizes_position(self):
        positions = set()
        for seed in range(10):
            env = make_env(replace_start_with_empty=True)
            env.reset(seed=seed)
            positions.add(tuple(env.agent_pos))
        self.assertGreater(len(positions), 1)

        env = make_env()
        for seed in range(5):
            env.reset(seed=seed)
            self.assertEqual(tuple(env.agent_pos), (1, 1))

    def test_invalid_probability_and_legacy_field(self):
        for value in (-0.01, 1.01, float("nan"), float("inf"), "not-a-number", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    make_env(action_failure_prob=value)

        with self.assertRaisesRegex(ValueError, "action_failure_prob"):
            make_env(move_failure_prob=0.1)

    def test_non_uniform_layout_requires_a_start_marker(self):
        with self.assertRaises(ValueError):
            CustomMiniGridEnv(
                layout_str="WWWWW\nWEEGW\nWWWWW",
                color_str="WWWWW\nWEEGW\nWWWWW",
                agent_start_dir=0,
            ).reset(seed=0)

    def test_wrapper_and_compact_action_contract_are_unchanged(self):
        env = FullyObsWrapper(
            make_env(stochastic_enabled=True, action_failure_prob=1.0)
        )
        observation, _ = env.reset(seed=0)
        self.assertEqual(observation["image"].shape, (5, 3, 3))
        self.assertEqual(env.action_space.n, MODEL_ACTION_COUNT)
        with self.assertRaises(ValueError):
            env.step(6)
        self.assertEqual(MODEL_ACTION_COUNT, 6)
        self.assertEqual(compact_to_native(2), int(Actions.forward))
        self.assertEqual(compact_to_native(4), int(Actions.toggle))
        self.assertEqual(compact_to_native(5), int(Actions.drop))

    def test_each_action_empirical_failure_rate(self):
        for action in ALL_ACTIONS:
            with self.subTest(action=action):
                env = make_env(max_steps=1001, stochastic_enabled=True, action_failure_prob=0.2)
                self.addCleanup(env.close)
                env.reset(seed=123)
                env.agent_dir = 2  # Forward faces a wall, so sampling cannot terminate the episode.
                failures = []
                for _ in range(1000):
                    _, _, _, _, info = env.step(action)
                    failures.append(info["action_failed"])
                self.assertAlmostEqual(float(np.mean(failures)), 0.2, delta=0.05)


if __name__ == "__main__":
    unittest.main()
