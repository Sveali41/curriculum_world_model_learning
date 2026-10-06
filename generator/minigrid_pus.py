"""Parameterized uncertainty sampling for MiniGrid world-model training."""

from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from domain.minigrid import minigrid_support as minigrid_utils
from generator.generator_interface import GeneratorInterface
from trainer.p2e_baseline import P2E_Ensemble


OBJECT_NAMES = ("door", "key", "lava", "wall")
COLOR_NAMES = ("yellow", "red", "blue", "green")
COLOR_COUNT_VECTORS = tuple(
    counts for counts in product((0, 1, 2), repeat=4) if sum(counts) <= 2
)
# 15 door-color counts x 15 key-color counts x 3 lava counts x 3 wall counts.
PUS_SETTINGS = tuple(
    (*door_counts, *key_counts, lava_count, wall_count)
    for door_counts in COLOR_COUNT_VECTORS
    for key_counts in COLOR_COUNT_VECTORS
    for lava_count in (0, 2, 4)
    for wall_count in (0, 4, 8)
)


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class MiniGridPUSGenerator(GeneratorInterface):
    def __init__(self, world_model, device, cfg):
        if str(cfg.domain) != "minigrid":
            raise ValueError("PUS is defined only for MiniGrid")
        if str(cfg.domains.minigrid.exploration_policy).lower() != "random":
            raise ValueError("PUS requires domains.minigrid.exploration_policy=random")
        if str(cfg.env.collect.data_type).lower() != "random":
            raise ValueError("PUS requires env.collect.data_type=random")
        super().__init__(world_model, device, cfg, agent_type="random")
        self.rng = np.random.default_rng(int(cfg.seed))
        self.settings = PUS_SETTINGS
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
        self.ensemble = P2E_Ensemble(cfg, num_models=int(cfg.pus.num_models))
        self.reservoir = {theta: [] for theta in self.settings}
        self.seen = {theta: 0 for theta in self.settings}
        self.scores = {}
        self.selection_rows = []
        self.action_ids = {
            name: [
                action_id for action_id, value in self.ACTION_TABLE.items()
                if value is not None and value[0] == name
            ]
            for name in OBJECT_NAMES
        }
        if any(not ids for ids in self.action_ids.values()):
            raise ValueError("MiniGrid action table cannot generate every PUS object type")
        self.color_action_ids = {
            (name, color_name): next(
                action_id for action_id, value in self.ACTION_TABLE.items()
                if value is not None and value[:2] == (name, color_name)
            )
            for name in ("door", "key")
            for color_name in COLOR_NAMES
        }

    def _sample_uniform_setting(self):
        return self.settings[int(self.rng.integers(len(self.settings)))]

    def _select_settings(self, iteration):
        if iteration == 0:
            return [
                (self._sample_uniform_setting(), "uniform", 1 / len(self.settings))
                for _ in range(self.batch_size)
            ]

        observed = tuple(self.scores)
        if not observed:
            raise RuntimeError("PUS has no scored setting after the first WM update")
        scores = np.asarray([self.scores[setting] for setting in observed])
        if not np.isfinite(scores).all():
            raise RuntimeError("PUS uncertainty scores must be finite")
        std = float(scores.std())
        logits = (scores - scores.mean()) / std if std > 1e-12 else np.zeros_like(scores)
        weights = np.exp(logits - logits.max())
        weighted_probabilities = weights / weights.sum()
        weighted_by_setting = dict(zip(observed, weighted_probabilities))
        selected = []
        for _ in range(self.batch_size):
            if self.rng.random() < self.uniform_fraction:
                setting = self._sample_uniform_setting()
                mode = "uniform"
            else:
                index = int(self.rng.choice(len(observed), p=weighted_probabilities))
                setting = observed[index]
                mode = "uncertainty"
            probability = (
                self.uniform_fraction / len(self.settings)
                + (1 - self.uniform_fraction) * weighted_by_setting.get(setting, 0.0)
            )
            selected.append((setting, mode, float(probability)))
        return selected

    def generate_map(self, theta, iteration, env_index):
        theta = tuple(int(value) for value in theta)
        if theta not in self.reservoir:
            raise ValueError(f"Unknown PUS parameter setting: {theta}")
        base = self._minigrid_empty_base(iteration, env_index)
        positions = np.argwhere(base == self.OBJ_EMPTY)
        if sum(theta) > len(positions):
            raise ValueError(f"PUS setting {theta} exceeds {len(positions)} editable cells")
        planned_actions = [
            self.color_action_ids[(name, color_name)]
            for name, color_counts in (("door", theta[:4]), ("key", theta[4:8]))
            for color_name, count in zip(COLOR_NAMES, color_counts)
            for _ in range(count)
        ]
        planned_actions += [self.action_ids["lava"][0]] * theta[8]
        planned_actions += [self.action_ids["wall"][0]] * theta[9]
        for _ in range(100):
            shuffled = positions[self.rng.permutation(len(positions))]
            actions = np.zeros_like(base)
            for (row, col), action_id in zip(shuffled, planned_actions):
                actions[row, col] = action_id
            obj, color, state = self._apply_action(base, actions)
            colors_match = all(
                np.count_nonzero(
                    (obj == OBJECT_TO_IDX[name]) & (color == COLOR_TO_IDX[color_name])
                ) == count
                for name, color_counts in (("door", theta[:4]), ("key", theta[4:8]))
                for color_name, count in zip(COLOR_NAMES, color_counts)
            )
            terrain_matches = all(
                np.count_nonzero(obj == OBJECT_TO_IDX[name])
                == np.count_nonzero(base == OBJECT_TO_IDX[name]) + count
                for name, count in (("lava", theta[8]), ("wall", theta[9]))
            )
            if colors_match and terrain_matches:
                return obj, color, state, self._default_stats()
        raise RuntimeError(f"Could not realize PUS setting {theta} in 100 map samples")

    def step(self, old_params, iteration=0):
        selected = self._select_settings(iteration)
        maps, colors, states, stats = [], [], [], []
        for env_index, (theta, _, _) in enumerate(selected):
            obj, color, state, stat = self.generate_map(theta, iteration, env_index)
            maps.append(obj)
            colors.append(color)
            states.append(state)
            stats.append(stat)
        self._record_generated_minigrid_batch(maps, colors, states, stats, iteration)

        trajectories = []
        self.selection_rows = []
        solved_count = 0
        distances = []
        for env_index, (theta, mode, probability) in enumerate(selected):
            result = self._rollout_combined(
                maps[env_index], stats[env_index], iteration, env_index,
                color_np=colors[env_index], state_np=states[env_index],
                evaluate_wm=False,
            )
            trajectory = result[0]
            if not trajectory or "obs" not in trajectory:
                raise RuntimeError(
                    f"PUS rollout failed at iteration {iteration + 1}, map {env_index}, theta={theta}"
                )
            trajectory["theta"] = theta
            trajectories.append(trajectory)
            distance = float(result[8]) if len(result) > 8 else 0.0
            if distance > 0:
                solved_count += 1
                distances.append(distance)
            colored_objects = np.isin(
                maps[env_index], (OBJECT_TO_IDX["door"], OBJECT_TO_IDX["key"])
            )
            color_counts = {
                f"{name}_{color_name}": int(np.count_nonzero(
                    (maps[env_index] == OBJECT_TO_IDX[name])
                    & (colors[env_index] == COLOR_TO_IDX[color_name])
                ))
                for name in ("door", "key")
                for color_name in ("yellow", "red", "blue", "green")
            }
            self.selection_rows.append({
                "Seed": int(self.cfg.seed), "Iter": iteration + 1,
                "Map": env_index, "n_door": sum(theta[:4]), "n_key": sum(theta[4:8]),
                "n_lava": theta[8], "n_wall": theta[9],
                "Selection_Mode": mode, "Selection_Probability": probability,
                "Uncertainty": self.scores.get(theta, float("nan")),
                "Transitions": len(trajectory["obs"]),
                "Unique_Key_Door_Colors": int(np.unique(colors[env_index][colored_objects]).size),
                **color_counts,
            })
        self.last_minigrid_metrics = {
            name: float("nan") for name in (
                "Pre_Changed_Focal_Loss", "Post_Changed_Focal_Loss",
                "Learning_Progress", "Difficulty_Rank", "Learning_Progress_Rank",
            )
        }
        return (
            None, None, float("nan"), float("nan"), float("nan"),
            float("nan"), trajectories, solved_count,
            float(np.mean(distances)) if distances else 0.0,
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
                    positions = minigrid_utils.get_agent_position(obs, player_id=self.OBJ_START)
                    masked = minigrid_utils.extract_masked_state(
                        obs, int(self.cfg.attention_model.attention_mask_size), positions
                    )
                    features.append(self.wm.encode(masked.long()).mean(dim=1).to(self.ensemble.device))
        finally:
            self.wm.train(was_training)
        return torch.cat(features, dim=0)

    def _update_reservoir(self, trajectories):
        for trajectory in trajectories:
            theta = tuple(trajectory["theta"])
            obs = _numpy(trajectory["obs"])
            actions = _numpy(trajectory["act"]).reshape(-1)
            next_obs = _numpy(trajectory["obs_next"])
            bucket = self.reservoir[theta]
            for index in range(len(actions)):
                self.seen[theta] += 1
                entry = (obs[index].copy(), int(actions[index]), next_obs[index].copy())
                if len(bucket) < self.reservoir_size:
                    bucket.append(entry)
                else:
                    slot = int(self.rng.integers(self.seen[theta]))
                    if slot < self.reservoir_size:
                        bucket[slot] = entry

    def update_uncertainty(self, current_batch, trajectories):
        prior = [entry for theta in self.settings for entry in self.reservoir[theta]]
        if len(prior) > self.history_train_samples:
            indices = self.rng.choice(len(prior), self.history_train_samples, replace=False)
            prior = [prior[int(index)] for index in indices]
        obs = _numpy(current_batch["obs"])
        actions = _numpy(current_batch["act"]).reshape(-1)
        next_obs = _numpy(current_batch["obs_next"])
        if prior:
            obs = np.concatenate((obs, np.stack([item[0] for item in prior])))
            actions = np.concatenate((actions, np.asarray([item[1] for item in prior])))
            next_obs = np.concatenate((next_obs, np.stack([item[2] for item in prior])))
        features = self._encode(obs)
        next_features = self._encode(next_obs)
        order = self.rng.permutation(len(actions))
        losses = []
        for start in range(0, len(order), self.predictor_batch_size):
            indices = order[start:start + self.predictor_batch_size]
            losses.append(self.ensemble.train_step(
                features[indices], actions[indices], next_features[indices]
            ))
        self._update_reservoir(trajectories)

        score_obs, score_actions, spans = [], [], {}
        for theta in self.settings:
            bucket = self.reservoir[theta]
            if bucket:
                spans[theta] = (len(score_obs), len(score_obs) + len(bucket))
                score_obs.extend(item[0] for item in bucket)
                score_actions.extend(item[1] for item in bucket)
        score_features = self._encode(np.stack(score_obs))
        score_actions = torch.as_tensor(score_actions, device=self.ensemble.device)
        with torch.no_grad():
            inputs = torch.cat((
                score_features,
                F.one_hot(score_actions.long(), num_classes=self.ensemble.action_dim).float(),
            ), dim=-1)
            predictions = torch.stack([head(inputs) for head in self.ensemble.heads])
            disagreement = predictions.var(dim=0, unbiased=False).mean(dim=-1)
        for theta, (start, end) in spans.items():
            current = float(disagreement[start:end].mean())
            previous = self.scores.get(theta)
            self.scores[theta] = (
                current if previous is None
                else self.ema_alpha * previous + (1 - self.ema_alpha) * current
            )
        return float(np.mean(losses))
