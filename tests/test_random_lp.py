import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir

# Use the entry point's existing path/environment setup.
from trainer.mac_wm_learning import ROOT_DIR
from generator.generator_interface import GeneratorInterface
from generator.random_generator_agent import RandomGeneratorAgent
from generator.random_lp import RandomLPEnvironmentPool
from trainer.common.run_resume import config_digest, generator_state, restore_generator, replay_state, restore_replay


class RandomLPPoolTests(unittest.TestCase):
    def pool(self, **kwargs):
        return RandomLPEnvironmentPool(**dict(
            dict(capacity=4, new_fraction=.5, uniform_mix=.2, seed=7), **kwargs,
        ))

    @staticmethod
    def candidates(offset=0):
        return [{"obj": np.array([i + offset])} for i in range(4)]

    def test_positive_lp_probabilities_and_without_replacement(self):
        pool = self.pool()
        pool.mix(self.candidates(), 0)
        pool.update([1, 2, 3, 4], [2, 2, 2, 1], [True] * 4)
        # Positive weights are [0, 0, 1, 3]; negative LP is not absolute LP.
        rng = Mock()
        rng.choice.side_effect = [3, 2]
        pool.rng = rng
        batch = pool.mix(self.candidates(10), 1)
        np.testing.assert_allclose(rng.choice.call_args_list[0].kwargs["p"], [.05, .05, .25, .65])
        np.testing.assert_allclose(rng.choice.call_args_list[1].kwargs["p"], [1/15, 1/15, 13/15])
        self.assertEqual([int(item["obj"][0]) for item in batch], [10, 11, 3, 2])

    def test_nonpositive_lp_uses_uniform_sampling(self):
        pool = self.pool()
        pool.mix(self.candidates(), 0)
        pool.update([0] * 4, [1, 0, 1, 0], [True] * 4)
        pool.rng = Mock()
        pool.rng.choice.side_effect = [0, 0]
        pool.mix(self.candidates(10), 1)
        np.testing.assert_allclose(pool.rng.choice.call_args_list[0].kwargs["p"], [.25] * 4)

    def test_fifo_invalid_probes_and_score_refresh(self):
        pool = self.pool()
        pool.mix(self.candidates(), 0)
        pool.update([1] * 4, [0] * 4, [True] * 4)
        pool.rng = Mock()
        pool.rng.choice.side_effect = [3, 2]
        pool.mix(self.candidates(10), 1)
        rows = pool.update([2, 2, np.nan, 1], [1, 1, np.nan, 3], [True, True, False, True])
        self.assertEqual([entry["id"] for entry in pool.entries], [2, 3, 4, 5])
        self.assertEqual([entry["lp"] for entry in pool.entries], [-2, 0, 1, 1])
        self.assertEqual(rows[2]["score_age"], 1)
        self.assertIsNone(rows[2]["learning_progress"])
        self.assertEqual(rows[3]["learning_progress"], -2)

    def test_resume_preserves_selection_and_candidate_arrays(self):
        pool = self.pool()
        pool.mix(self.candidates(), 0)
        pool.update([1, 2, 3, 4], [0] * 4, [True] * 4)
        state = pool.state_dict()
        restored = self.pool()
        restored.load_state_dict(state)
        first, second = pool.mix(self.candidates(10), 1), restored.mix(self.candidates(10), 1)
        for actual, expected in zip(first, second):
            np.testing.assert_array_equal(actual["obj"], expected["obj"])
        self.assertEqual(pool.pending[1], restored.pending[1])
        first[-1]["obj"][:] = -1
        self.assertTrue(all(int(entry["candidate"]["obj"][0]) >= 0 for entry in pool.entries))

    def test_incomplete_or_nonfinite_scores_fail(self):
        pool = self.pool()
        pool.mix(self.candidates(), 0)
        with self.assertRaisesRegex(RuntimeError, "scored"):
            pool.mix(self.candidates(), 1)
        with self.assertRaisesRegex(RuntimeError, "after scoring"):
            pool.state_dict()
        with self.assertRaisesRegex(ValueError, "finite"):
            pool.update([np.nan] * 4, [0] * 4, [True] * 4)

    def test_no_event_probe_retains_environment_with_unavailable_lp(self):
        pool = self.pool()
        pool.mix(self.candidates(), 0)
        rows = pool.update([np.nan] * 4, [np.nan] * 4, [True] * 4, scorable=[False] * 4)
        self.assertEqual(len(pool.entries), 4)
        self.assertEqual([entry["lp"] for entry in pool.entries], [0.] * 4)
        self.assertTrue(all(row["valid"] and not row["lp_valid"] for row in rows))
        self.assertTrue(all(row["learning_progress"] is None for row in rows))


class RandomLPWiringTests(unittest.TestCase):
    @staticmethod
    def config(domain):
        with initialize_config_dir(config_dir=ROOT_DIR + "/trainer/conf", version_base=None):
            return compose(config_name="config_random_lp", overrides=[
                f"domain={domain}", "seed=0", f"domains.{domain}.generator_batch_size=4",
                "domains.minigrid.reward.diversity_mode=action_hamming",
            ])

    def interface(self, domain):
        wm = Mock()
        wm.encode_map_features.side_effect = lambda maps: torch.ones(len(maps), 4)
        wm.calc_minigrid_changed_focal_loss.return_value = 2.0
        wm.calc_minigrid_probe_metrics.return_value = {"changed_focal_loss": .5}

        def crafter_losses(trajectories, **kwargs):
            values = [2.0 if kwargs["return_components"] else .5] * len(trajectories)
            if kwargs["return_components"]:
                return values, [{"layout": 1.0, "inventory": 1.0,
                                 "layout_changed_count": 1, "inventory_changed_count": 1}] * len(values)
            return values

        wm.calc_crafter_changed_focal_losses.side_effect = crafter_losses
        with patch("generator.generator_interface.GeneratorPPO", side_effect=AssertionError("PPO constructed")):
            interface = GeneratorInterface(wm, "cpu", self.config(domain), agent_type="random_lp")
        self.assertIsInstance(interface.ppo, RandomGeneratorAgent)
        return interface

    def test_both_domains_recollect_selected_layouts_and_keep_probe_subset(self):
        for domain in ("minigrid", "crafter"):
            with self.subTest(domain=domain):
                interface = self.interface(domain)
                trajectory = {"obs": torch.zeros(10, 3, 8, 8), "act": torch.zeros(10)}
                rollout = Mock(return_value=(trajectory, None, 1., True, 1., 1., 0, 10., 1.))
                interface._rollout_combined = rollout
                interface.ppo.save_buffer = Mock(side_effect=AssertionError("PPO buffer used"))
                interface.step(None, iteration=0)
                pending = interface._pending_minigrid_round if domain == "minigrid" else interface._pending_crafter_round
                self.assertEqual(len(pending["probe_trajectories"][0]["obs"]), 4)
                probe = pending["probe_trajectories"]
                if domain == "minigrid":
                    interface.finalize_minigrid_rewards()
                    self.assertIs(interface.wm.calc_minigrid_probe_metrics.call_args_list[0].args[0], probe[0])
                else:
                    interface.finalize_crafter_learning_progress(apply_rewards=True)
                    self.assertIs(interface.wm.calc_crafter_changed_focal_losses.call_args.args[0][0], probe[0])
                self.assertEqual([entry["lp"] for entry in interface.random_lp_pool.entries], [1.5] * 4)
                first_entries = {entry["id"]: entry["candidate"] for entry in interface.random_lp_pool.entries}
                rollout.reset_mock()
                interface.step(None, iteration=1)
                records = interface.random_lp_pool.pending[1]
                self.assertEqual(rollout.call_count, 4)
                self.assertEqual([row["is_replay"] for row in records], [False, False, True, True])
                for call, row in zip(rollout.call_args_list[2:], records[2:]):
                    stored = first_entries[row["environment_id"]]
                    np.testing.assert_array_equal(call.args[0], stored["obj"])
                    np.testing.assert_array_equal(call.args[1], stored["stats"])
                    np.testing.assert_array_equal(call.kwargs["color_np"], stored["color"])
                    np.testing.assert_array_equal(call.kwargs["state_np"], stored["state"])
                if domain == "minigrid":
                    interface.finalize_minigrid_rewards()
                else:
                    interface.finalize_crafter_learning_progress(apply_rewards=True)
                self.assertEqual(interface.update(iteration=1), (0., 0., 0.))
                rows = interface.last_minigrid_map_diagnostics if domain == "minigrid" else interface.last_crafter_map_diagnostics
                self.assertEqual(len(rows), 4)
                self.assertEqual(rows[-1]["score_age"], 1)

    def test_crafter_resume_includes_pool_without_ppo_state(self):
        interface = self.interface("crafter")
        pool = interface.random_lp_pool
        pool.mix(RandomLPPoolTests.candidates(), 0)
        pool.update([1] * 4, [0] * 4, [True] * 4)
        state = generator_state(interface, include_policy=True)
        self.assertNotIn("ppo", state)
        restored = self.interface("crafter")
        restore_generator(restored, state, include_policy=True)
        original_batch = pool.mix(RandomLPPoolTests.candidates(10), 1)
        restored_batch = restored.random_lp_pool.mix(RandomLPPoolTests.candidates(10), 1)
        for actual, expected in zip(original_batch, restored_batch):
            np.testing.assert_array_equal(actual["obj"], expected["obj"])
        del state["random_lp"]
        with self.assertRaisesRegex(ValueError, "saved environment pool"):
            restore_generator(restored, state, include_policy=True)

    def test_minigrid_resume_preserves_pool_and_novelty_archives(self):
        interface = self.interface("minigrid")
        interface.random_lp_pool.mix(RandomLPPoolTests.candidates(), 0)
        interface.random_lp_pool.update([1] * 4, [0] * 4, [True] * 4)
        interface.map_archive = [np.array([7])]
        interface.combination_archive = [{"objects": np.array([3])}]
        state = generator_state(interface, include_policy=True)
        restored = self.interface("minigrid")
        restore_generator(restored, state, include_policy=True)
        np.testing.assert_array_equal(restored.map_archive[0], [7])
        np.testing.assert_array_equal(restored.combination_archive[0]["objects"], [3])
        original = interface.random_lp_pool.mix(RandomLPPoolTests.candidates(10), 1)
        continued = restored.random_lp_pool.mix(RandomLPPoolTests.candidates(10), 1)
        for left, right in zip(original, continued):
            np.testing.assert_array_equal(left["obj"], right["obj"])

    def test_reservoir_resume_preserves_admission_and_sampling_rng(self):
        from modelBased.continue_learning.reservoir_buffer import ReservoirReplayBuffer
        import copy
        original = ReservoirReplayBuffer(4, seed=3)
        original.update_combined({"obs": np.arange(20).reshape(10, 2)})
        restored = ReservoirReplayBuffer(4, seed=99)
        restore_replay(restored, copy.deepcopy(replay_state(original)))
        more = {"obs": np.arange(20, 60).reshape(20, 2)}
        original.update_combined(more)
        restored.update_combined(more)
        self.assertEqual(original.num_seen, restored.num_seen)
        np.testing.assert_array_equal([x["obs"] for x in original.buffer], [x["obs"] for x in restored.buffer])
        np.testing.assert_array_equal(original._rng.integers(100, size=10), restored._rng.integers(100, size=10))

    def test_crafter_no_event_probes_are_not_numerical_failures(self):
        interface = self.interface("crafter")
        def no_events(trajectories, **kwargs):
            losses = [float("nan")] * len(trajectories)
            components = [{"layout_changed_count": 0, "inventory_changed_count": 0}] * len(trajectories)
            return (losses, components) if kwargs["return_components"] else losses
        interface.wm.calc_crafter_changed_focal_losses.side_effect = no_events
        trajectory = {"obs": torch.zeros(10, 3, 8, 8)}
        interface._rollout_combined = Mock(return_value=(trajectory, None, 0., True, 0., 0., 0, 10., 1.))
        interface.step(None, iteration=0)
        interface.finalize_crafter_learning_progress()
        self.assertEqual(len(interface.random_lp_pool.entries), 4)
        self.assertEqual(interface.last_crafter_metrics["paired_probe_count"], 0)
        self.assertEqual(interface.last_crafter_metrics["valid_probe_count"], 4)
        self.assertTrue(all(not row["lp_valid"] for row in interface.last_crafter_map_diagnostics))

    def test_preset_isolates_artifacts_and_rejects_invalid_protocol(self):
        for domain in ("minigrid", "crafter"):
            cfg = self.config(domain)
            for path in (cfg.mac_results_dir, cfg.mac_temp_data_dir, cfg.attention_model.model_save_path,
                         cfg.attention_model.checkpoint_dir, cfg.domains.minigrid.rmax_like.checkpoint_path):
                self.assertTrue(str(path).startswith(str(cfg.random_lp_run_dir)))
            self.assertEqual(cfg.generator_agent.total_iterations, 50)
        cfg = self.config("minigrid")
        cfg.generator_agent.warmup_iterations = 10
        with self.assertRaisesRegex(ValueError, "warmup_iterations=0"):
            GeneratorInterface(Mock(), "cpu", cfg, agent_type="random_lp")

    def test_resume_digest_ignores_collection_filenames_but_checks_protocol(self):
        cfg = self.config("crafter")
        before = config_digest(cfg)
        cfg.env.collect.data_save_path = "/tmp/iteration0_batch1.npz"
        cfg.env.collect.visualize_filename = "iteration0_batch1.png"
        cfg.env.collect.env_visualize_filename = "iteration0_batch1_env.png"
        self.assertEqual(config_digest(cfg), before)
        cfg.generator_agent.random_lp.uniform_mix = .3
        self.assertNotEqual(config_digest(cfg), before)


if __name__ == "__main__":
    unittest.main()
