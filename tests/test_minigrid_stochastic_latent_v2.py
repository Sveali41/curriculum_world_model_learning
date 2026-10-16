import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


WM_ROOT = Path(__file__).resolve().parents[1] / "wm"
if str(WM_ROOT) not in sys.path:
    sys.path.insert(0, str(WM_ROOT))

from modelBased.world_model.AttentionWM_support import AttentionModule
from modelBased.world_model.AttentionWM import AttentionWorldModel


class MiniGridStochasticLatentV2Tests(unittest.TestCase):
    def _module(self, factors=1, classes=2):
        return AttentionModule(
            data_type="discrete", grid_shape=(3, 5, 5), mask_size=5,
            embed_dim=16, num_heads=1, env_type="minigrid",
            minigrid_transition_mode="effect", stochastic_model="latent_v2",
            latent_num_factors=factors, latent_num_classes=classes,
        )

    @staticmethod
    def _inputs(batch_size=3):
        state = torch.zeros(batch_size, 3, 5, 5, dtype=torch.long)
        state[:, 0, 2, 2] = 10
        next_state = state.clone()
        next_state[0, 0, 2, 2] = 0
        next_state[0, 0, 2, 3] = 10
        action = torch.arange(batch_size, dtype=torch.long) % 3
        return state, next_state, action, torch.zeros(batch_size, dtype=torch.long)

    def test_distribution_shapes_and_contract(self):
        module = self._module(factors=2, classes=3)
        state, next_state, action, inv = self._inputs()
        output = module(
            state, action, None, inv=inv, return_distribution=True,
            next_state=next_state, latent_labels=None, sample_mode="mode",
        )
        self.assertEqual(tuple(output["state_logits"].shape), (3, 24, 5, 5))
        self.assertEqual(tuple(output["inventory_logits"].shape), (3, 8))
        self.assertEqual(tuple(output["prior_logits"].shape), (3, 2, 3))
        self.assertEqual(tuple(output["posterior_logits"].shape), (3, 2, 3))
        self.assertEqual(tuple(output["latent_sample"].shape), (3, 2))
        self.assertIn("stochastic_latent", module.checkpoint_contract)

    def test_prior_sampling_is_seeded_and_does_not_need_next_state(self):
        module = self._module().eval()
        state, _, action, inv = self._inputs()
        first = module(
            state, action, None, inv=inv, return_distribution=True,
            sample_mode="sample", generator=torch.Generator().manual_seed(123),
        )
        second = module(
            state, action, None, inv=inv, return_distribution=True,
            sample_mode="sample", generator=torch.Generator().manual_seed(123),
        )
        self.assertIsNone(first["posterior_logits"])
        torch.testing.assert_close(first["latent_sample"], second["latent_sample"])

    def test_labels_condition_factor_zero_and_loss_backpropagates(self):
        module = self._module().train()
        state, next_state, action, inv = self._inputs()
        labels = torch.tensor([1, 0, 1], dtype=torch.long)
        output = module(
            state, action, None, inv=inv, return_distribution=True,
            next_state=next_state, latent_labels=labels, sample_mode="mode",
        )
        torch.testing.assert_close(output["latent_sample"][:, 0], labels)
        q_log = F.log_softmax(output["posterior_logits"], dim=-1)
        q = q_log.exp()
        kl = (q * (q_log - F.log_softmax(output["prior_logits"], dim=-1))).sum(-1).mean()
        loss = output["state_logits"].mean() + output["inventory_logits"].mean() + kl
        loss.backward()
        self.assertIsNotNone(module.prior_head[0].weight.grad)
        self.assertIsNotNone(module.posterior_head[0].weight.grad)
        self.assertIsNotNone(module.latent_embedding.grad)

    def test_plain_forward_contract_stays_three_values(self):
        module = self._module()
        state, _, action, inv = self._inputs()
        self.assertEqual(len(module(state, action, None, inv=inv)), 3)

    def test_domain_switch_selects_the_wm_mode(self):
        config_path = Path(__file__).resolve().parents[1] / "trainer" / "conf" / "config_mac.yaml"
        cfg = OmegaConf.load(config_path)
        self.assertEqual(AttentionWorldModel(cfg.attention_model).stochastic_model, "none")
        # A legacy per-block selector cannot override the resolved domain flag.
        cfg.attention_model.stochastic_model = "outcome_v1"
        self.assertEqual(AttentionWorldModel(cfg.attention_model).stochastic_model, "none")
        cfg.domains.minigrid.stochastic.enabled = True
        wm = AttentionWorldModel(cfg.attention_model)
        self.assertEqual(wm.stochastic_model, "latent_v2")
        self.assertTrue(wm.stochastic_latent_v2_enabled)


if __name__ == "__main__":
    unittest.main()
