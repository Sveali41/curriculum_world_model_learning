"""WAKER-inspired parameterized uncertainty sampling for Crafter."""

import numpy as np
import torch
import torch.nn.functional as F

from generator.crafter_env_designer import CRAFTER_OBJ_MAP
from generator.generator_interface import GeneratorInterface
from modelBased.common import utils as wm_utils
from modelBased.world_model.crafter_dynamics import get_crafter_agent_position


OBJECT_NAMES = (
    "tree", "stone", "coal", "iron", "diamond",
    "water", "table", "furnace", "plant", "cow",
)
# DR actions 0 and 1 both produce an empty inventory, so theta has five
# distinct inventory conditions: empty and progression stages 1 through 4.
NUM_SETTINGS = 3 ** len(OBJECT_NAMES) * 5


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class CrafterPUSGenerator(GeneratorInterface):
    def __init__(self, world_model, device, cfg):
        if str(cfg.domain) != "crafter":
            raise ValueError("Crafter PUS requires domain=crafter")
        if str(cfg.env.collect.data_type).lower() != "random":
            raise ValueError("Crafter PUS requires env.collect.data_type=random")
        super().__init__(world_model, device, cfg, agent_type="random")
        self.rng = np.random.default_rng(int(cfg.seed))
        self.settings = range(NUM_SETTINGS)
        self.uniform_fraction = float(cfg.pus.uniform_fraction)
        self.ema_alpha = float(cfg.pus.ema_alpha)
        self.reservoir_size = int(cfg.pus.reservoir_per_theta)
        self.history_train_samples = int(cfg.pus.history_train_samples)
        self.predictor_batch_size = int(cfg.pus.predictor_batch_size)
        if not 0 <= self.uniform_fraction <= 1:
            raise ValueError("pus.uniform_fraction must be between 0 and 1")
        if not 0 <= self.ema_alpha < 1:
            raise ValueError("pus.ema_alpha must be at least 0 and less than 1")
        if min(self.reservoir_size, self.history_train_samples, self.predictor_batch_size) < 1:
            raise ValueError("PUS sample and batch sizes must be positive")

        latent_dim = int(cfg.attention_model.embed_dim)
        action_dim = int(cfg.domains.crafter.action_norm)
        self.action_dim = action_dim
        input_dim = latent_dim + action_dim + 16
        output_dim = latent_dim + 16
        self.heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, latent_dim * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(latent_dim * 2, output_dim),
            )
            for _ in range(int(cfg.pus.num_models))
        ]).to(device)
        self.predictor_optimizer = torch.optim.Adam(
            self.heads.parameters(), lr=float(cfg.p2e.disag_lr)
        )
        self.reservoir = {}
        self.seen = {}
        self.scores = {}
        self.selection_rows = []

    def _sample_uniform_setting(self):
        counts = tuple(int(value) for value in self.rng.integers(0, 3, size=len(OBJECT_NAMES)))
        return (*counts, int(self.rng.integers(0, 5)))

    def _select_settings(self, iteration):
        if iteration == 0:
            return [
                (self._sample_uniform_setting(), "uniform", 1 / NUM_SETTINGS)
                for _ in range(self.batch_size)
            ]
        observed = tuple(self.scores)
        if not observed:
            raise RuntimeError("Crafter PUS has no scored setting after the first WM update")
        scores = np.asarray([self.scores[theta] for theta in observed])
        if not np.isfinite(scores).all():
            raise RuntimeError("Crafter PUS uncertainty scores must be finite")
        std = float(scores.std())
        logits = (scores - scores.mean()) / std if std > 1e-12 else np.zeros_like(scores)
        weights = np.exp(logits - logits.max())
        weighted_probabilities = weights / weights.sum()
        probabilities = dict(zip(observed, weighted_probabilities))
        selected = []
        for _ in range(self.batch_size):
            if self.rng.random() < self.uniform_fraction:
                theta = self._sample_uniform_setting()
                mode = "uniform"
            else:
                theta = observed[int(self.rng.choice(len(observed), p=weighted_probabilities))]
                mode = "uncertainty"
            probability = (
                self.uniform_fraction / NUM_SETTINGS
                + (1 - self.uniform_fraction) * probabilities.get(theta, 0.0)
            )
            selected.append((theta, mode, float(probability)))
        return selected

    def generate_map(self, theta):
        theta = tuple(int(value) for value in theta)
        if len(theta) != len(OBJECT_NAMES) + 1 or any(
            count not in (0, 1, 2) for count in theta[:-1]
        ) or theta[-1] not in range(5):
            raise ValueError(f"Invalid Crafter PUS setting: {theta}")
        grid = self.seeder.generate()
        interior = grid[1:-1, 1:-1]
        interior[interior != CRAFTER_OBJ_MAP["agent"]] = CRAFTER_OBJ_MAP["grass"]
        positions = np.argwhere(grid == CRAFTER_OBJ_MAP["grass"])
        if sum(theta[:-1]) > len(positions):
            raise ValueError(f"Crafter PUS setting {theta} exceeds editable cells")
        positions = positions[self.rng.permutation(len(positions))]
        offset = 0
        for name, count in zip(OBJECT_NAMES, theta[:-1]):
            for row, col in positions[offset:offset + count]:
                grid[row, col] = CRAFTER_OBJ_MAP[name]
            offset += count
        stats = self._default_stats()
        if theta[-1]:
            stats[4:16] = self._sample_crafter_stage_inventory(theta[-1])
        return grid, stats

    def step(self, old_params, iteration=0):
        selected = self._select_settings(iteration)
        self.selection_rows = []
        trajectories = []
        for env_index, (theta, mode, probability) in enumerate(selected):
            grid, stats = self.generate_map(theta)
            result = self._rollout_combined(
                grid, stats, iteration, env_index, old_params=old_params,
                evaluate_wm=False,
            )
            trajectory = result[0]
            if not trajectory or any(
                trajectory.get(key) is None for key in ("obs", "obs_next", "act", "inv", "inv_next")
            ):
                raise RuntimeError(
                    f"Crafter PUS rollout failed at iteration {iteration + 1}, "
                    f"map {env_index}, theta={theta}"
                )
            trajectory["theta"] = theta
            trajectories.append(trajectory)
            self.selection_rows.append({
                "Seed": int(self.cfg.seed), "Iter": iteration + 1, "Map": env_index,
                **{f"n_{name}": count for name, count in zip(OBJECT_NAMES, theta[:-1])},
                "inventory_stage": theta[-1],
                "Selection_Mode": mode, "Selection_Probability": probability,
                "Uncertainty": self.scores.get(theta, float("nan")),
                "Transitions": len(trajectory["obs"]),
            })
        return (
            None, None, float("nan"), float("nan"), float("nan"),
            float("nan"), trajectories, len(trajectories), 0.0,
        )

    def _encode(self, observations):
        features = []
        was_training = self.wm.training
        self.wm.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(observations), self.predictor_batch_size):
                    obs = torch.as_tensor(
                        observations[start:start + self.predictor_batch_size],
                        device=self.device,
                    )
                    positions = get_crafter_agent_position(obs)
                    masked = wm_utils.extract_masked_state_torch(
                        obs, int(self.cfg.attention_model.attention_mask_size),
                        positions, pad_value=0,
                    )
                    features.append(self.wm.encode(masked.long()).mean(dim=1))
        finally:
            self.wm.train(was_training)
        return torch.cat(features, dim=0)

    def _predictor_inputs(self, observations, actions, inventory):
        features = self._encode(observations)
        actions = torch.as_tensor(actions, device=self.device).long().reshape(-1)
        inventory = torch.as_tensor(inventory, device=self.device).float() / 9.0
        return torch.cat(
            (features, F.one_hot(actions, num_classes=self.action_dim).float(), inventory),
            dim=-1,
        )

    def _update_reservoir(self, trajectories):
        for trajectory in trajectories:
            theta = tuple(trajectory["theta"])
            values = tuple(_numpy(trajectory[key]) for key in (
                "obs", "act", "obs_next", "inv", "inv_next",
            ))
            bucket = self.reservoir.setdefault(theta, [])
            for entry in zip(*values):
                self.seen[theta] = self.seen.get(theta, 0) + 1
                stored = tuple(np.asarray(value).copy() for value in entry)
                if len(bucket) < self.reservoir_size:
                    bucket.append(stored)
                else:
                    slot = int(self.rng.integers(self.seen[theta]))
                    if slot < self.reservoir_size:
                        bucket[slot] = stored

    def update_uncertainty(self, current_batch, trajectories):
        keys = ("obs", "act", "obs_next", "inv", "inv_next")
        if any(current_batch.get(key) is None for key in keys):
            raise ValueError("Crafter PUS requires layout, action, and inventory transitions")
        prior = [entry for bucket in self.reservoir.values() for entry in bucket]
        if len(prior) > self.history_train_samples:
            indices = self.rng.choice(len(prior), self.history_train_samples, replace=False)
            prior = [prior[int(index)] for index in indices]
        values = [_numpy(current_batch[key]) for key in keys]
        if prior:
            values = [
                np.concatenate((value, np.stack([entry[index] for entry in prior])))
                for index, value in enumerate(values)
            ]
        obs, actions, next_obs, inventory, next_inventory = values
        inputs = self._predictor_inputs(obs, actions, inventory)
        targets = torch.cat((
            self._encode(next_obs),
            torch.as_tensor(next_inventory, device=self.device).float() / 9.0,
        ), dim=-1)
        order = self.rng.permutation(len(actions))
        losses = []
        for start in range(0, len(order), self.predictor_batch_size):
            indices = order[start:start + self.predictor_batch_size]
            batch_input = inputs[indices]
            batch_target = targets[indices]
            loss = sum(F.mse_loss(head(batch_input), batch_target) for head in self.heads)
            self.predictor_optimizer.zero_grad()
            loss.backward()
            self.predictor_optimizer.step()
            losses.append(float(loss.detach()) / len(self.heads))
        self._update_reservoir(trajectories)

        score_entries = [entry for bucket in self.reservoir.values() for entry in bucket]
        score_inputs = self._predictor_inputs(
            np.stack([entry[0] for entry in score_entries]),
            np.asarray([entry[1] for entry in score_entries]),
            np.stack([entry[3] for entry in score_entries]),
        )
        with torch.no_grad():
            predictions = torch.stack([head(score_inputs) for head in self.heads])
            disagreement = predictions.var(dim=0, unbiased=False).mean(dim=-1)
        offset = 0
        for theta, bucket in self.reservoir.items():
            current = float(disagreement[offset:offset + len(bucket)].mean())
            previous = self.scores.get(theta)
            self.scores[theta] = (
                current if previous is None
                else self.ema_alpha * previous + (1 - self.ema_alpha) * current
            )
            offset += len(bucket)
        return float(np.mean(losses))
