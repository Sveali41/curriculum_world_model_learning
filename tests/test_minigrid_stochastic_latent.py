import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch


WM_ROOT = Path(__file__).resolve().parents[1] / "wm"
if str(WM_ROOT) not in sys.path:
    sys.path.insert(0, str(WM_ROOT))

from modelBased.data.datamodule import WMRLDataset
from modelBased.world_model.AttentionWM_support import AttentionModule


class MiniGridStochasticLatentTests(unittest.TestCase):
    def _module(self, stochastic_outcome):
        return AttentionModule(
            data_type="discrete",
            grid_shape=(3, 5, 5),
            mask_size=5,
            embed_dim=16,
            num_heads=1,
            env_type="minigrid",
            minigrid_transition_mode="effect",
            stochastic_outcome=stochastic_outcome,
        )

    def _inputs(self):
        state = torch.zeros(3, 3, 5, 5, dtype=torch.long)
        state[:, 0, 2, 2] = 10
        action = torch.tensor([0, 1, 2], dtype=torch.long)
        inventory = torch.zeros(3, dtype=torch.long)
        return state, action, inventory

    def test_disabled_forward_contract_is_unchanged(self):
        module = self._module(stochastic_outcome=False)
        state, action, inventory = self._inputs()
        result = module(state, action, None, inv=inventory)
        self.assertEqual(len(result), 3)
        with self.assertRaises(ValueError):
            module(state, action, None, inv=inventory, return_outcome=True)

    def test_enabled_outcome_logits_are_a_binary_distribution(self):
        module = self._module(stochastic_outcome=True)
        state, action, inventory = self._inputs()
        state_logits, _, inventory_logits, outcome_logits = module(
            state, action, None, inv=inventory, return_outcome=True
        )
        self.assertEqual(tuple(state_logits.shape), (3, 24, 5, 5))
        self.assertEqual(tuple(inventory_logits.shape), (3, 8))
        self.assertEqual(tuple(outcome_logits.shape), (3, 2))
        torch.testing.assert_close(
            torch.softmax(outcome_logits, dim=-1).sum(dim=-1), torch.ones(3)
        )
        self.assertIn("stochastic_outcome", module.checkpoint_contract)

    def test_normalized_info_keeps_action_failed(self):
        normalized = WMRLDataset._normalize_minigrid_info_for_batch(np.asarray([
            {"action_failed": True, "current_carrying_token": 0, "next_carrying_token": 0},
            {"action_failed": False, "current_carrying_token": 2, "next_carrying_token": 2},
        ], dtype=object))
        self.assertTrue(normalized[0]["action_failed"])
        self.assertFalse(normalized[1]["action_failed"])
        self.assertIsInstance(normalized[0]["action_failed"], bool)

    def test_enabled_dataset_rejects_missing_outcome_labels(self):
        obs = np.zeros((2, 3, 5, 5), dtype=np.int64)
        obs[:, 0, 2, 2] = 10
        hparams = SimpleNamespace(
            obs_norm_values=1,
            action_norm_values=6,
            attention_mask_size=5,
            env_type="minigrid",
            data_type="discrete",
            frame_stack=1,
            replay_frac=0.0,
            stochastic_latent={"enabled": True},
        )
        raw = {
            "a": obs,
            "b": obs.copy(),
            "c": np.asarray([0, 1], dtype=np.int64),
            "f": np.asarray([
                {"current_carrying_token": 0, "next_carrying_token": 0},
                {"current_carrying_token": 0, "next_carrying_token": 0},
            ], dtype=object),
        }
        with self.assertRaisesRegex(ValueError, "action_failed"):
            WMRLDataset(raw, hparams)


if __name__ == "__main__":
    unittest.main()
