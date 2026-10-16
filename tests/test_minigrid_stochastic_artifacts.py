import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from tests.test_stochastic_minigrid_env import LAYOUT
from domain.minigrid.minigrid_support import stochastic_env_kwargs
from modelBased.common.artifacts import dataset_matches, identity_from_config


class StochasticDatasetIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        layout = self.root / 'layout.txt'
        layout.write_text(LAYOUT)
        self.cfg = OmegaConf.create({
            'domain': 'minigrid',
            'domains': {'minigrid': {
                'layout_path': str(layout), 'task_name': 'test',
                'stochastic': {'enabled': True, 'action_failure_prob': .2},
            }},
        })
        self.path = self.root / 'data.npz'

    def save(self, metadata):
        np.savez(self.path, metadata=np.asarray(json.dumps(metadata), dtype=object))

    def test_active_kernel_requires_exact_stochastic_identity(self):
        metadata = identity_from_config(self.cfg)
        self.assertEqual(metadata['stochastic'], {
            'enabled': True, 'kernel': 'all_actions_noop_v1', 'action_failure_prob': .2,
        })
        self.save(metadata)
        self.assertTrue(dataset_matches(self.path, self.cfg))
        for recorded in (None, {'enabled': False},
                         {'enabled': True, 'move_failure_prob': .2},
                         {'enabled': True, 'action_failure_prob': .2},
                         {**metadata['stochastic'], 'action_failure_prob': .4},
                         {**metadata['stochastic'], 'kernel': 'navigation_noop_v1'}):
            with self.subTest(recorded=recorded):
                candidate = copy.deepcopy(metadata)
                if recorded is None:
                    candidate.pop('stochastic')
                else:
                    candidate['stochastic'] = recorded
                self.save(candidate)
                self.assertFalse(dataset_matches(self.path, self.cfg))

    def test_disabled_kernel_retains_legacy_deterministic_data(self):
        self.cfg.domains.minigrid.stochastic.enabled = False
        metadata = identity_from_config(self.cfg)
        for recorded in (None, {'enabled': False},
                         {'enabled': False, 'move_failure_prob': .2}):
            with self.subTest(recorded=recorded):
                candidate = copy.deepcopy(metadata)
                if recorded is None:
                    candidate.pop('stochastic')
                else:
                    candidate['stochastic'] = recorded
                self.save(candidate)
                self.assertTrue(dataset_matches(self.path, self.cfg))
        metadata['stochastic'] = {
            'enabled': True, 'kernel': 'all_actions_noop_v1', 'action_failure_prob': .2,
        }
        self.save(metadata)
        self.assertFalse(dataset_matches(self.path, self.cfg))

    def test_relocated_layout_still_checks_kernel_and_other_semantics(self):
        metadata = identity_from_config(self.cfg)
        metadata['env_path'] = metadata['layout_path'] = '/old/workspace/layout.txt'
        self.save(metadata)
        self.assertTrue(dataset_matches(self.path, self.cfg))
        metadata['stochastic']['action_failure_prob'] = .3
        self.save(metadata)
        self.assertFalse(dataset_matches(self.path, self.cfg))
        self.cfg.domains.minigrid.stochastic.enabled = False
        metadata = identity_from_config(self.cfg)
        metadata['inventory_encoding'] = 'different'
        self.save(metadata)
        self.assertFalse(dataset_matches(self.path, self.cfg))

    def test_old_config_field_is_rejected_at_constructor_wiring_and_identity(self):
        self.cfg.domains.minigrid.stochastic.move_failure_prob = .7
        for consumer in (stochastic_env_kwargs, identity_from_config):
            with self.subTest(consumer=consumer.__name__):
                with self.assertRaisesRegex(ValueError, 'action_failure_prob'):
                    consumer(self.cfg)

    def test_stochastic_training_cannot_bypass_missing_identity_with_legacy_flag(self):
        import torch
        from modelBased.world_model import AttentionWM_training

        cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'trainer/conf/config_mac.yaml')
        cfg.domains.minigrid.stochastic.enabled = True
        cfg.attention_model.allow_legacy_dataset = True
        cfg.attention_model.data_dir = str(self.path)
        cfg.attention_model.model_save_path = str(self.root / 'model.ckpt')
        np.savez(self.path, a=np.zeros((1, 3, 5, 5)))
        for model_enabled in (True, False):
            with self.subTest(model_enabled=model_enabled):
                cfg.attention_model.stochastic_enabled = model_enabled
                with self.assertRaisesRegex(RuntimeError, 'stochastic kernel'):
                    AttentionWM_training.run(cfg, net=torch.nn.Linear(1, 1))


if __name__ == '__main__':
    unittest.main()
