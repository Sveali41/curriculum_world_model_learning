import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf
from minigrid.core.actions import Actions

from tests.test_stochastic_minigrid_env import LAYOUT, COLORS
from trainer.common.utils import collect_data_general


class StochasticMACWiringTests(unittest.TestCase):
    def test_tuple_collection_uses_the_same_domain_switch_as_the_wm(self):
        config_path = Path(__file__).resolve().parents[1] / 'trainer/conf/config_mac.yaml'
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temp:
                cfg = OmegaConf.load(config_path)
                cfg.domain = 'minigrid'
                cfg.domains.minigrid.stochastic.enabled = enabled
                cfg.domains.minigrid.stochastic.move_failure_prob = 1.0
                cfg.env.collect.data_folder = temp
                cfg.env.collect.visualize_save_path = temp
                cfg.env.collect.data_type = 'random'

                def inspect_collection(*, env, **kwargs):
                    try:
                        self.assertEqual(env.unwrapped.stochastic_enabled, enabled)
                        self.assertEqual(env.unwrapped.move_failure_prob, 1.0)
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


if __name__ == '__main__':
    unittest.main()
