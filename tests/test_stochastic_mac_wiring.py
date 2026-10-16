import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from minigrid.core.actions import Actions

from tests.test_stochastic_minigrid_env import LAYOUT, COLORS, CustomMiniGridEnv
from domain.minigrid.action_codec import compact_to_native
from modelBased.common.artifacts import dataset_matches, dataset_metadata
from modelBased.world_model.AttentionWM_support import AttentionModule
from modelBased.world_model.crafter_dynamics import categorical_effect_target
from trainer.common.utils import collect_data_general


class StochasticMACWiringTests(unittest.TestCase):
    def test_planner_real_environments_use_shared_stochastic_configuration(self):
        from modelBased.policy_training.planners import mcts_planner, mpc_planner

        class EnvironmentChecked(Exception):
            pass

        for mode, module, planner_name in (
            ('mpc', mpc_planner, 'WorldModelMPC'),
            ('mcts', mcts_planner, 'WorldModelMCTS'),
        ):
            for enabled in (False, True):
                with self.subTest(mode=mode, enabled=enabled), tempfile.TemporaryDirectory() as temp:
                    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'wm/modelBased/config/config.yaml')
                    cfg.domain = 'minigrid'
                    cfg.domains.minigrid.stochastic.enabled = enabled
                    cfg.domains.minigrid.stochastic.action_failure_prob = 1.0
                    cfg.PPO.train_in_real_env = False
                    cfg.PPO.wm_control_mode = mode
                    cfg.PPO.checkpoint_path_wm = None
                    cfg.PPO[mode].output_dir = temp
                    cfg.PPO[mode].episodes = 1

                    def check_env(**kwargs):
                        kwargs.pop('txt_file_path')
                        env = CustomMiniGridEnv(layout_str=LAYOUT, color_str=COLORS, **kwargs)
                        try:
                            env.reset(seed=11)
                            _, _, _, _, info = env.step(Actions.toggle)
                            self.assertEqual(info['action_failed'], enabled)
                            self.assertEqual(env.action_failure_prob, 1.0)
                        finally:
                            env.close()
                        raise EnvironmentChecked

                    with patch.object(module, '_load_world_model', return_value=(Mock(), Path(temp) / 'wm.ckpt')), \
                         patch.object(module, planner_name), \
                         patch.object(module, 'CustomMiniGridEnv', side_effect=check_env):
                        with self.assertRaises(EnvironmentChecked):
                            getattr(module, f'run_online_{mode}')(cfg)

    def test_pipeline_forwards_the_shared_probability(self):
        from run_pipeline import domain_runtime_overrides

        cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'wm/modelBased/config/config.yaml')
        cfg.domain = 'minigrid'
        cfg.domains.minigrid.stochastic.enabled = True
        cfg.domains.minigrid.stochastic.action_failure_prob = .35
        overrides = domain_runtime_overrides(cfg, 'minigrid')
        self.assertIn('domains.minigrid.stochastic.enabled=true', overrides)
        self.assertIn('domains.minigrid.stochastic.action_failure_prob=0.35', overrides)
        cfg.domains.minigrid.stochastic.move_failure_prob = .7
        with self.assertRaisesRegex(ValueError, 'action_failure_prob'):
            domain_runtime_overrides(cfg, 'minigrid')

    def test_tuple_collection_uses_the_same_domain_switch_as_the_wm(self):
        config_path = Path(__file__).resolve().parents[1] / 'trainer/conf/config_mac.yaml'
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temp:
                cfg = OmegaConf.load(config_path)
                cfg.domain = 'minigrid'
                cfg.domains.minigrid.stochastic.enabled = enabled
                cfg.domains.minigrid.stochastic.action_failure_prob = 1.0
                cfg.env.collect.data_folder = temp
                cfg.env.collect.visualize_save_path = temp
                cfg.env.collect.data_type = 'random'

                def inspect_collection(*, env, **kwargs):
                    try:
                        self.assertEqual(env.unwrapped.stochastic_enabled, enabled)
                        self.assertEqual(env.unwrapped.action_failure_prob, 1.0)
                        self.assertEqual(cfg.attention_model.stochastic_enabled, enabled)
                        self.assertFalse(cfg.attention_model.transition_balanced_sampling)
                        env.reset(seed=7)
                        _, _, _, _, info = env.step(Actions.forward)
                        self.assertEqual(info['action_failed'], enabled)
                        self.assertEqual(info['requested_action'], int(Actions.forward))
                    finally:
                        env.close()

                with patch('trainer.common.utils.Support') as support:
                    support.return_value.collect_data_trainer.side_effect = inspect_collection
                    collect_data_general(cfg, (LAYOUT, COLORS), save_name='stochastic_wiring')
                    support.return_value.collect_data_trainer.assert_called_once()

    def test_real_collection_preserves_all_actions_and_latent_training_labels(self):
        class CyclingPolicy:
            expects_raw_obs = True

            def __init__(self):
                self.step = 0

            def select_action(self, obs):
                action = self.step % 6
                self.step += 1
                return action

        def seeded_env(*args, **kwargs):
            env = CustomMiniGridEnv(*args, **kwargs)
            env.reset(seed=123)
            return env

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = 'WWWWW\nWSEEW\nWEEGW\nWEEEW\nWWWWW'
            colors = 'WWWWW\nWEEEW\nWEEGW\nWEEEW\nWWWWW'
            layout_path = root / 'layout.txt'
            layout_path.write_text(layout + '\n\n' + colors)
            cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'trainer/conf/config_mac.yaml')
            cfg.domain = 'minigrid'
            cfg.domains.minigrid.stochastic.enabled = True
            cfg.domains.minigrid.layout_path = str(layout_path)
            cfg.env.collect.data_folder = temp
            cfg.env.collect.visualize_save_path = temp
            cfg.env.collect.data_type = 'random'
            cfg.env.collect.episodes = 1
            cfg.env.collect.mini_dataset_size = 48
            cfg.env.collect.save_env_visualize = False
            cfg.env.collect.save_coverage_visualize = False
            cfg.env.save_visualized_img = False
            with patch('trainer.common.utils.CustomMiniGridEnv', side_effect=seeded_env):
                path = collect_data_general(
                    cfg, (layout, colors), save_name='all_actions_smoke',
                    maximum_dataset_size=48, max_steps=100, policy=CyclingPolicy())
            self.assertTrue(dataset_matches(path, cfg))
            self.assertEqual(dataset_metadata(path)['stochastic']['kernel'], 'all_actions_noop_v1')
            with np.load(path, allow_pickle=True) as data:
                actions = data['c'].reshape(-1)
                records = data['f'].reshape(-1)
                np.testing.assert_array_equal(actions, np.tile(np.arange(6), 8))
                for action, record in zip(actions, records):
                    self.assertEqual(record['requested_action'], compact_to_native(action))
                    self.assertEqual(record['executed_action'],
                                     -1 if record['action_failed'] else compact_to_native(action))
                labels = torch.tensor([record['action_failed'] for record in records], dtype=torch.long)
                self.assertEqual(set(labels.tolist()), {0, 1})
                state = torch.as_tensor(data['a'], dtype=torch.long)
                next_state = torch.as_tensor(data['b'], dtype=torch.long)
                inventory = torch.tensor([r['current_carrying_token'] for r in records])
                next_inventory = torch.tensor([r['next_carrying_token'] for r in records])

            module = AttentionModule(
                data_type='discrete', grid_shape=(3, 5, 5), mask_size=5,
                embed_dim=16, num_heads=1, env_type='minigrid',
                minigrid_transition_mode='effect', stochastic_model='latent_v2')
            output = module(
                state, torch.as_tensor(actions), None, inv=inventory,
                return_distribution=True, next_state=next_state,
                next_inventory=next_inventory, latent_labels=labels, sample_mode='mode')
            torch.testing.assert_close(output['latent_sample'][:, 0], labels)
            q_log = F.log_softmax(output['posterior_logits'], dim=-1)
            kl = (q_log.exp() * (q_log - F.log_softmax(output['prior_logits'], dim=-1))).sum(-1).mean()
            loss = (
                F.cross_entropy(output['state_logits'][:, :12],
                                categorical_effect_target(state[:, 0], next_state[:, 0]))
                + F.cross_entropy(output['inventory_logits'],
                                  categorical_effect_target(inventory, next_inventory))
                + .1 * kl
                + F.cross_entropy(output['prior_logits'][:, 0], labels)
                + F.cross_entropy(output['posterior_logits'][:, 0], labels))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for parameter in (module.prior_head[0].weight, module.posterior_head[0].weight,
                              module.latent_embedding):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    unittest.main()
