"""Train a comparison WM without letting it affect MAC's data collection."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import open_dict

from modelBased.world_model import AttentionWM_training
from modelBased.world_model.AttentionWM import AttentionWorldModel
from trainer.common.run_resume import restore_rng, rng_state


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


class CrafterCNNComparison:
    def __init__(self, cfg, reference, device):
        self.checkpoint = Path(cfg.attention_model.model_save_path).with_name("cnn_wm.ckpt")
        self.device = device
        self.old_params = self.fisher = None
        model_cfg = deepcopy(cfg)
        with open_dict(model_cfg.attention_model):
            model_cfg.attention_model.crafter_spatial_backbone = "cnn"
        with preserve_rng():
            self.net = AttentionWorldModel(model_cfg.attention_model).to(device)

        # All unchanged weights (including prediction heads) start identically.
        reference_weights = reference.state_dict()
        weights = self.net.state_dict()
        weights.update({key: reference_weights[key] for key in weights if key in reference_weights})
        self.net.load_state_dict(weights)
        for label, model in (("AC-AWM", reference), ("CNN-WM", self.net)):
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[Crafter backbone] {label}: {count:,} trainable parameters")

    def update(self, cfg, replay_data, training_rng):
        train_cfg = deepcopy(cfg)
        with open_dict(train_cfg.attention_model):
            train_cfg.attention_model.crafter_spatial_backbone = "cnn"
        train_cfg.attention_model.model_save_path = str(self.checkpoint)
        train_cfg.attention_model.checkpoint_dir = str(self.checkpoint.parent / "cnn_lightning")
        # Rewind training RNG for the paired update, then return it to MAC.
        with preserve_rng():
            restore_rng(training_rng)
            result, self.fisher, _ = AttentionWM_training.train_api(
                train_cfg, self.net, self.old_params, self.fisher, replay_data=replay_data,
            )
            self.old_params = result["old_params"]
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            self.net = AttentionWorldModel(train_cfg.attention_model).to(self.device)
            self.net.load_state_dict(state["state_dict"])

    def state_dict(self):
        return {"wm": self.net.state_dict(), "old_params": self.old_params, "fisher": self.fisher}

    def load_state_dict(self, state):
        self.net.load_state_dict(state["wm"])
        self.old_params, self.fisher = state["old_params"], state["fisher"]
