import unittest
import sys
from pathlib import Path

import numpy as np

WM_ROOT = Path(__file__).resolve().parents[1] / "wm"
if str(WM_ROOT) not in sys.path:
    sys.path.insert(0, str(WM_ROOT))

from domain.minigrid.minigrid_custom_env import (
    INTERNAL_NOOP_ACTION,
    CustomMiniGridEnv,
)
from domain.minigrid.action_codec import MODEL_ACTION_COUNT, compact_to_native
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper


LAYOUT = "WWWWW\nWSEGW\nWWWWW"
COLORS = "WWWWW\nWEEGW\nWWWWW"
KEY_LAYOUT = "WWWWW\nWSKGW\nWWWWW"
KEY_COLORS = "WWWWW\nWYYGW\nWWWWW"


def make_env(**kwargs):
    return CustomMiniGridEnv(
        layout_str=LAYOUT,
        color_str=COLORS,
        agent_start_dir=0,
        max_steps=20,
        **kwargs,
    )


class StochasticMiniGridEnvTests(unittest.TestCase):
    def test_default_is_deterministic_and_info_is_stable(self):
        env = make_env()
        obs, _ = env.reset(seed=7)
        next_obs, reward, terminated, truncated, info = env.step(Actions.forward)

        self.assertEqual(tuple(env.agent_pos), (2, 1))
        self.assertEqual(obs["image"].shape, next_obs["image"].shape)
        self.assertFalse(info["stochastic_enabled"])
        self.assertFalse(info["action_failed"])
        self.assertEqual(info["requested_action"], int(Actions.forward))
        self.assertEqual(info["executed_action"], int(Actions.forward))
        self.assertIsInstance(reward, (int, float, np.number))
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_probability_extremes(self):
        for action in (Actions.left, Actions.right, Actions.forward):
            env = make_env(stochastic_enabled=True, move_failure_prob=1.0)
            env.reset(seed=1)
            before = (tuple(env.agent_pos), int(env.agent_dir), env.step_count)
            _, _, _, _, info = env.step(action)
            self.assertEqual(tuple(env.agent_pos), before[0])
            self.assertEqual(int(env.agent_dir), before[1])
            self.assertEqual(env.step_count, before[2] + 1)
            self.assertTrue(info["action_failed"])
            self.assertEqual(info["executed_action"], INTERNAL_NOOP_ACTION)

        env = make_env(stochastic_enabled=True, move_failure_prob=0.0)
        env.reset(seed=1)
        _, _, _, _, info = env.step(Actions.forward)
        self.assertEqual(tuple(env.agent_pos), (2, 1))
        self.assertFalse(info["action_failed"])

    def test_interaction_actions_are_not_dropped(self):
        env = CustomMiniGridEnv(
            layout_str=KEY_LAYOUT,
            color_str=KEY_COLORS,
            agent_start_dir=0,
            stochastic_enabled=True,
            move_failure_prob=1.0,
            max_steps=20,
        )
        env.reset(seed=2)
        env.step(Actions.pickup)
        self.assertIsNotNone(env.carrying)

    def test_seed_reproduces_initial_state_and_failure_sequence(self):
        def rollout():
            env = make_env(stochastic_enabled=True, move_failure_prob=0.35)
            obs, _ = env.reset(seed=123)
            trace = [(obs["image"].copy(), tuple(env.agent_pos), int(env.agent_dir))]
            failures = []
            for _ in range(20):
                obs, _, _, _, info = env.step(Actions.forward)
                trace.append(
                    (obs["image"].copy(), tuple(env.agent_pos), int(env.agent_dir))
                )
                failures.append(bool(info["action_failed"]))
            return trace, failures

        first, first_failures = rollout()
        second, second_failures = rollout()
        self.assertEqual(first_failures, second_failures)
        for first_item, second_item in zip(first, second):
            np.testing.assert_array_equal(first_item[0], second_item[0])
            self.assertEqual(first_item[1:], second_item[1:])

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

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            make_env(move_failure_prob=-0.01)
        with self.assertRaises(ValueError):
            make_env(move_failure_prob=1.01)

    def test_non_uniform_layout_requires_a_start_marker(self):
        with self.assertRaises(ValueError):
            CustomMiniGridEnv(
                layout_str="WWWWW\nWEEGW\nWWWWW",
                color_str="WWWWW\nWEEGW\nWWWWW",
                agent_start_dir=0,
            ).reset(seed=0)

    def test_wrapper_and_compact_action_contract_are_unchanged(self):
        env = FullyObsWrapper(make_env(stochastic_enabled=True, move_failure_prob=1.0))
        observation, _ = env.reset(seed=0)
        self.assertEqual(observation["image"].shape, (5, 3, 3))
        self.assertEqual(env.action_space.n, MODEL_ACTION_COUNT)
        with self.assertRaises(ValueError):
            env.step(6)
        self.assertEqual(MODEL_ACTION_COUNT, 6)
        self.assertEqual(compact_to_native(2), int(Actions.forward))

    def test_empirical_failure_rate(self):
        env = make_env(stochastic_enabled=True, move_failure_prob=0.25)
        failures = []
        for seed in range(1000):
            env.reset(seed=seed)
            _, _, _, _, info = env.step(Actions.forward)
            failures.append(info["action_failed"])
        self.assertAlmostEqual(float(np.mean(failures)), 0.25, delta=0.06)


if __name__ == "__main__":
    unittest.main()
