import torch
import numpy as np
import torch.nn.functional as F
import os
import random
import math
from collections import deque

from generator.generator_agent import GeneratorPPO
from generator.random_generator_agent import RandomGeneratorAgent
from generator.reward_system import DiversityModule, check_solvability
from generator.crafter_env_designer import CrafterPCGSeeder, CRAFTER_ACTION_MAP, CRAFTER_OBJ_MAP
from generator.bipedal_env_designer import BipedalPCGSeeder, ACTION_TABLE_BIPEDAL
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from modelBased.common.support import Support
from trainer.common.utils import extract_loss_map_over_validations, collect_data_general

ACTION_TABLE_MINIGRID = {
    0: None,  # No-op
    1: ("key", "yellow"),
    2: ("key", "red"),
    3: ("key", "blue"),
    4: ("key", "green"),
    5: ("door", "yellow"),
    6: ("door", "red"),
    7: ("door", "blue"),
    8: ("door", "green"),
    9: ("wall", None),
    10: ("lava", None),
}

LOCKED_DOOR_ACTIONS_MINIGRID = {
    11: ("door", "yellow", STATE_TO_IDX["locked"]),
    12: ("door", "red", STATE_TO_IDX["locked"]),
    13: ("door", "blue", STATE_TO_IDX["locked"]),
    14: ("door", "green", STATE_TO_IDX["locked"]),
}

# MiniGrid has one carried-object slot. These generator actions map to the
# colour-aware WM inventory tokens (empty=0, colour index + 1).
MINIGRID_INVENTORY_ACTION_TO_TOKEN = np.asarray([
    0,
    COLOR_TO_IDX["yellow"] + 1,
    COLOR_TO_IDX["red"] + 1,
    COLOR_TO_IDX["blue"] + 1,
    COLOR_TO_IDX["green"] + 1,
], dtype=np.int64)


def minigrid_action_table(enable_locked_doors):
    table = dict(ACTION_TABLE_MINIGRID)
    if enable_locked_doors:
        table.update(LOCKED_DOOR_ACTIONS_MINIGRID)
    return table

class GeneratorInterface:
    def __init__(self, world_model, device, cfg, agent_type='ppo'):
        self.device = device
        self.cfg = cfg
        self.agent_type = agent_type
        self.ablation_type = getattr(getattr(cfg, "ablation", None), "type", "none")
        hparams = cfg.generator_agent
        self.batch_size = hparams.batch_size
        self.support = Support(cfg)
        self.wm = world_model
        self.debug_mode = bool(getattr(cfg.attention_model, "debug_mode", False))
        self.use_elites = getattr(hparams, "use_elites", True)
        
        self.use_elites = getattr(hparams, "use_elites", True)
        
        # Check environment type
        self.is_crafter = (getattr(cfg.attention_model, "env_type", "") == "crafter")
        self.is_bipedal = (getattr(cfg.attention_model, "env_type", "") == "bipedalwalker")
        self.is_minigrid = not self.is_crafter and not self.is_bipedal
        minigrid_domain_cfg = getattr(getattr(cfg, "domains", None), "minigrid", None)
        self.minigrid_rmax_explorer = None
        if self.is_minigrid:
            exploration_policy = str(
                getattr(minigrid_domain_cfg, "exploration_policy", "random")
            ).lower()
            if exploration_policy not in {"random", "rmax"}:
                raise ValueError(
                    "domains.minigrid.exploration_policy must be 'random' or 'rmax'"
                )
            if exploration_policy == "rmax":
                from modelBased.exploration.minigrid_rmax import MiniGridRMaxExplorer

                self.minigrid_rmax_explorer = MiniGridRMaxExplorer(cfg)
                if bool(getattr(minigrid_domain_cfg.rmax_like, "resume", False)):
                    self.minigrid_rmax_explorer.load_checkpoint()

        if self.is_crafter:
            # Crafter: use custom seeder, actions, and map sizes
            # `map_height` now refers only to the physical layout.
            self.map_height = hparams.map_height
            self.map_width = hparams.map_width
            self.seeder = CrafterPCGSeeder(height=self.map_height, width=self.map_width)
            self.ACTION_TABLE = {
                0: None, 
                1: CRAFTER_OBJ_MAP['grass'],     # allows erasing
                2: CRAFTER_OBJ_MAP['tree'], 
                3: CRAFTER_OBJ_MAP['stone'], 
                4: CRAFTER_OBJ_MAP['coal'], 
                5: CRAFTER_OBJ_MAP['iron'], 
                6: CRAFTER_OBJ_MAP['diamond'],
                7: CRAFTER_OBJ_MAP['water'], 
                8: CRAFTER_OBJ_MAP['table'], 
                9: CRAFTER_OBJ_MAP['furnace'],
                10: CRAFTER_OBJ_MAP['plant'],
                11: CRAFTER_OBJ_MAP['cow'],
            }
        elif self.is_bipedal:
            self.map_height = 1
            self.map_width = int(getattr(hparams, "map_width", 5))  # Default 5 slots
            bipedal_domain_cfg = getattr(getattr(cfg, "domains", None), "bipedalwalker", None)
            self.active_bipedal_width = int(
                getattr(bipedal_domain_cfg, "active_w", self.map_width)
            )
            self.seeder = BipedalPCGSeeder(width=self.map_width)
            self.ACTION_TABLE = ACTION_TABLE_BIPEDAL
        else:
            # MiniGrid environments are built from a canonical empty canvas so
            # the generated layout is attributable to the policy action.
            self.map_height = hparams.map_height
            self.map_width = hparams.map_width
            self.ACTION_TABLE = minigrid_action_table(
                bool(getattr(minigrid_domain_cfg, "locked_doors", False))
            )
            # Keep start/goal fixed from base map, editor edits around them.
            self.minigrid_anchor_start = True
            self.minigrid_goal_fallback = False

        edit_group_sizes = None
        if self.is_minigrid:
            group_counts = {}
            for action_id, action_value in self.ACTION_TABLE.items():
                if action_id == 0:
                    continue
                family = action_value[0] if isinstance(action_value, tuple) else action_value
                group_counts[family] = group_counts.get(family, 0) + 1
            edit_group_sizes = [
                group_counts[self.ACTION_TABLE[action_id][0]]
                for action_id in range(1, len(self.ACTION_TABLE))
            ]

        if agent_type == 'random':
            self.ppo = RandomGeneratorAgent(
                num_actions=len(self.ACTION_TABLE),
                device=device,
                edit_action_group_sizes=edit_group_sizes,
                env_type=(
                    "bipedalwalker" if self.is_bipedal
                    else "crafter" if self.is_crafter
                    else "minigrid"
                ),
            )
        else:
            ppo_cfg = getattr(cfg, "PPO", None)
            self.ppo = GeneratorPPO(
                context_dim=hparams.context_dim,
                num_actions=len(self.ACTION_TABLE),
                his_emb_dim=hparams.his_emb_dim,
                top_k_features=hparams.ctx_top_k_features,
                ablation_type=self.cfg.ablation.type,
                env_type=("bipedalwalker" if self.is_bipedal else ("crafter" if self.is_crafter else "minigrid")),
                lr_actor=float(getattr(ppo_cfg, "lr_actor", 1e-4)) if ppo_cfg is not None else 1e-4,
                lr_critic=float(getattr(ppo_cfg, "lr_critic", 3e-4)) if ppo_cfg is not None else 3e-4,
                gamma=float(getattr(ppo_cfg, "gamma", 0.99)) if ppo_cfg is not None else 0.99,
                K_epochs=int(getattr(ppo_cfg, "K_epochs", 10)) if ppo_cfg is not None else 10,
                eps_clip=float(getattr(ppo_cfg, "eps_clip", 0.2)) if ppo_cfg is not None else 0.2,
                entropy_coef=float(getattr(ppo_cfg, "entropy_coef", 0.05)) if ppo_cfg is not None else 0.05,
                entropy_coef_start=float(getattr(ppo_cfg, "entropy_coef_start", getattr(ppo_cfg, "entropy_coef", 0.05))) if ppo_cfg is not None else 0.05,
                entropy_coef_end=float(getattr(ppo_cfg, "entropy_coef_end", getattr(ppo_cfg, "entropy_coef", 0.05))) if ppo_cfg is not None else 0.05,
                entropy_anneal_iters=int(getattr(ppo_cfg, "entropy_anneal_iters", 0)) if ppo_cfg is not None else 0,
                buffer_window_rounds=int(getattr(ppo_cfg, "buffer_window_rounds", 1)) if ppo_cfg is not None else 1,
                initial_edit_ratio=float(hparams.max_edits_layout),
                initial_inventory_edit_ratio=float(hparams.max_edits_inventory),
                edit_action_group_sizes=edit_group_sizes,
            )
        
        self.div_k = hparams.div_k
        minigrid_reward_cfg = self._get_minigrid_reward_cfg()
        self.diversity_mode = str(
            getattr(minigrid_reward_cfg, "diversity_mode", "action_hamming")
        ).lower()
        valid_diversity_modes = {"random_feature_knn", "action_hamming"}
        if self.is_minigrid and self.diversity_mode not in valid_diversity_modes:
            raise ValueError(
                "domains.minigrid.reward.diversity_mode must be one of "
                f"{sorted(valid_diversity_modes)}, got {self.diversity_mode!r}"
            )
        env_type = (
            "crafter" if self.is_crafter
            else "bipedalwalker" if self.is_bipedal
            else "minigrid"
        )
        use_random_feature_diversity = (
            not self.is_minigrid or self.diversity_mode == "random_feature_knn"
        )
        self.diversity = None
        if use_random_feature_diversity:
            diversity_kwargs = {
                "input_h": self.map_height,
                "input_w": self.map_width,
                "k": self.div_k,
                "max_archive_size": int(getattr(hparams, "map_archive_size", 1000)),
                "device": self.device,
                "env_type": env_type,
            }
            if self.is_minigrid:
                cuda_devices = []
                diversity_device = torch.device(self.device)
                if diversity_device.type == "cuda":
                    cuda_devices = [
                        diversity_device.index
                        if diversity_device.index is not None
                        else torch.cuda.current_device()
                    ]
                # Keep the fixed random encoder from shifting PPO's sampling
                # stream, so same-seed diversity ablations are paired.
                with torch.random.fork_rng(devices=cuda_devices):
                    self.diversity = DiversityModule(**diversity_kwargs)
            else:
                self.diversity = DiversityModule(**diversity_kwargs)
        self.map_archive = [] if self.is_minigrid else None
        self.combination_archive = [] if self.is_minigrid else None
        self.map_archive_size = int(getattr(hparams, "map_archive_size", 1000))
        self.map_knn_k = max(1, int(getattr(hparams, "map_knn_k", 10)))
        self.map_batch_archive_mix = float(
            np.clip(getattr(hparams, "map_batch_archive_mix", 0.5), 0.0, 1.0)
        )
        self.latent_kernel_temperature = max(
            float(getattr(hparams, "latent_kernel_temperature", 0.2)), 1e-6
        )
        self.combination_archive_size = int(
            getattr(hparams, "combination_archive_size", 1000)
        )
        self.combination_knn_k = max(
            1, int(getattr(hparams, "combination_knn_k", 10))
        )
        self.combination_batch_archive_mix = float(np.clip(
            getattr(hparams, "combination_batch_archive_mix", 0.5), 0.0, 1.0
        ))
        self.max_edits_layout = hparams.max_edits_layout
        self.max_edits_inventory = hparams.max_edits_inventory
        
        if self.is_bipedal:
            self.OBJ_START = 0
            self.OBJ_GOAL = 999
            self.OBJ_EMPTY = 0
        else:
            self.OBJ_START = OBJECT_TO_IDX["agent"] if not self.is_crafter else CRAFTER_OBJ_MAP['agent']
            self.OBJ_GOAL = OBJECT_TO_IDX["goal"] if not self.is_crafter else 999 # No goal in Crafter
            self.OBJ_EMPTY = OBJECT_TO_IDX["empty"] if not self.is_crafter else CRAFTER_OBJ_MAP['grass']

        self.elite_buffer = [] 
        self.max_elites = max(1, self.batch_size // 2)
        self.prev_data = None
        self.crafter_reward_cfg = self._get_crafter_reward_cfg()
        self.bipedal_reward_cfg = self._get_bipedal_reward_cfg()
        self.minigrid_reward_cfg = minigrid_reward_cfg
        minigrid_domain_cfg = getattr(getattr(cfg, "domains", None), "minigrid", None)
        lp_probe_budget = int(
            getattr(minigrid_domain_cfg, "learning_progress_probe_budget", 1000)
        )
        self.minigrid_lp_probe_size = max(
            1, int(math.ceil(lp_probe_budget / max(self.batch_size, 1)))
        )
        self.last_minigrid_metrics = {}
        self.last_generated_minigrid_batch = None
        self._pending_minigrid_round = None
        self.bipedal_history_len = int(self._get_bipedal_history_len())
        self.bipedal_history = deque(maxlen=self.bipedal_history_len)
        self._last_bipedal_memory = (
            np.zeros(26, dtype=np.float32) if self.is_bipedal else np.zeros(16, dtype=np.float32)
        )

    def _get_crafter_reward_cfg(self):
        if not self.is_crafter:
            return {}
        domains_cfg = getattr(self.cfg, "domains", None)
        if domains_cfg is None:
            return {}
        crafter_cfg = getattr(domains_cfg, "crafter", None)
        if crafter_cfg is None:
            return {}
        reward_cfg = getattr(crafter_cfg, "reward", None)
        return reward_cfg if reward_cfg is not None else {}

    def _get_bipedal_reward_cfg(self):
        if not self.is_bipedal:
            return {}
        domains_cfg = getattr(self.cfg, "domains", None)
        if domains_cfg is None:
            return {}
        bipedal_cfg = getattr(domains_cfg, "bipedalwalker", None)
        if bipedal_cfg is None:
            return {}
        reward_cfg = getattr(bipedal_cfg, "reward", None)
        return reward_cfg if reward_cfg is not None else {}

    def _get_minigrid_reward_cfg(self):
        if self.is_crafter or self.is_bipedal:
            return {}
        domains_cfg = getattr(self.cfg, "domains", None)
        if domains_cfg is None:
            return {}
        minigrid_cfg = getattr(domains_cfg, "minigrid", None)
        if minigrid_cfg is None:
            return {}
        reward_cfg = getattr(minigrid_cfg, "reward", None)
        return reward_cfg if reward_cfg is not None else {}

    def _get_bipedal_history_len(self):
        if not self.is_bipedal:
            return getattr(self.cfg.generator_agent, "history_len", 5)
        domains_cfg = getattr(self.cfg, "domains", None)
        if domains_cfg is not None:
            bipedal_cfg = getattr(domains_cfg, "bipedalwalker", None)
            if bipedal_cfg is not None and getattr(bipedal_cfg, "history_len", None) is not None:
                return getattr(bipedal_cfg, "history_len")
        return getattr(self.cfg.generator_agent, "history_len", 5)

    def _get_diversity_reward(self, map_tensor, inventory_vec=None):
        """Compute diversity safely for discrete MiniGrid maps.

        ``DiversityModule`` one-hot encodes object and colour IDs on CUDA.
        Validate the IDs on CPU first so an invalid generator proposal produces
        a diagnostic and a zero diversity reward instead of an unrecoverable
        device-side assertion.
        """
        if self.diversity is None:
            return 0.0
        if not self.is_crafter and not self.is_bipedal:
            if torch.is_tensor(map_tensor):
                map_np = map_tensor.detach().cpu().numpy()
            else:
                map_np = np.asarray(map_tensor)
            if map_np.ndim == 4:
                map_np = map_np[0]
            if map_np.ndim != 3 or map_np.shape[0] < 2:
                print(
                    f"[Diversity] Invalid MiniGrid map shape {map_np.shape}; "
                    "using diversity reward 0."
                )
                return 0.0

            obj_ids = map_np[0]
            color_ids = map_np[1]
            finite = np.isfinite(obj_ids).all() and np.isfinite(color_ids).all()
            integral = (
                np.equal(obj_ids, np.floor(obj_ids)).all()
                and np.equal(color_ids, np.floor(color_ids)).all()
            )
            obj_min, obj_max = float(np.min(obj_ids)), float(np.max(obj_ids))
            color_min, color_max = float(np.min(color_ids)), float(np.max(color_ids))
            valid = (
                finite
                and integral
                and obj_min >= 0
                and obj_max < 11
                and color_min >= 0
                and color_max < 6
            )
            if not valid:
                print(
                    "[Diversity] Invalid MiniGrid IDs: "
                    f"object=[{obj_min}, {obj_max}], color=[{color_min}, {color_max}]. "
                    "Using diversity reward 0 for this proposal."
                )
                return 0.0

        return float(
            self.diversity.get_reward(
                torch.as_tensor(map_tensor, device=self.device).unsqueeze(0),
                inventory_vec=inventory_vec,
            )
        )

    def _materialize_candidates(self, base_ids, actions, stats_actions, base_stats, mask):
        """Apply a sampled edit batch once, before rollout scoring."""
        base_ids = np.asarray(base_ids)
        actions = np.asarray(actions)
        stats_actions = None if stats_actions is None else np.asarray(stats_actions)
        base_stats = np.asarray(base_stats)
        edit_mask = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
        maps_obj, maps_color, maps_state, maps_stats = [], [], [], []
        for i in range(len(base_ids)):
            obj, color, state = self._apply_action(
                base_ids[i], actions[i], mask=edit_mask[i, 0]
            )
            stats = base_stats[i].copy()
            current_stats = stats_actions[i] if stats_actions is not None else None
            if self.is_crafter and getattr(self.cfg.generator_agent, "random_stats_actions", False):
                current_stats = (np.random.rand(32) < 0.15).astype(np.int64)
            if self.is_crafter and current_stats is not None:
                for k_idx, value in enumerate(current_stats):
                    if value == 1:
                        slot = k_idx % 16
                        stats[slot] += 1 if k_idx < 16 else 5
            elif self.is_minigrid and current_stats is not None:
                inventory_action = int(np.asarray(current_stats).reshape(-1)[0])
                if not 0 <= inventory_action < len(MINIGRID_INVENTORY_ACTION_TO_TOKEN):
                    raise ValueError(
                        f"Invalid MiniGrid inventory action {inventory_action}; expected 0..4"
                    )
                stats[0] = MINIGRID_INVENTORY_ACTION_TO_TOKEN[inventory_action]
            maps_obj.append(obj)
            maps_color.append(color)
            maps_state.append(state)
            maps_stats.append(stats)
        return maps_obj, maps_color, maps_state, maps_stats

    def _record_generated_minigrid_batch(
        self, object_maps, color_maps, state_maps, stats_batch, iteration
    ):
        if not self.is_minigrid:
            return
        self.last_generated_minigrid_batch = {
            "object_maps": np.stack(object_maps).astype(np.int64, copy=True),
            "color_maps": np.stack(color_maps).astype(np.int64, copy=True),
            "state_maps": np.stack(state_maps).astype(np.int64, copy=True),
            "inventory_tokens": np.asarray(stats_batch, dtype=np.int64)[:, 0].copy(),
            "iteration": int(iteration),
        }

    @staticmethod
    def _object_distance_metrics(grid, excluded_ids):
        """Mean pair/nearest Manhattan distances for non-background objects."""
        positions = np.argwhere(~np.isin(np.asarray(grid), list(excluded_ids)))
        if len(positions) < 2:
            return 0.0, 0.0
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.abs(deltas).sum(axis=-1)
        upper = distances[np.triu_indices(len(positions), k=1)]
        nearest = np.min(np.where(np.eye(len(positions), dtype=bool), np.inf, distances), axis=1)
        return float(upper.mean()), float(nearest.mean())

    def _selected_edit_pair_distance(self, action_mask):
        if torch.is_tensor(action_mask):
            mask = action_mask.detach().cpu().numpy()
        else:
            mask = np.asarray(action_mask)
        if mask.ndim == 4:
            mask = mask[:, 0]
        values = []
        for sample in mask:
            positions = np.argwhere(sample > 0)
            if len(positions) >= 2:
                distances = np.abs(positions[:, None, :] - positions[None, :, :]).sum(axis=-1)
                values.append(float(distances[np.triu_indices(len(positions), k=1)].mean()))
            else:
                values.append(0.0)
        return float(np.mean(values)) if values else 0.0

    def _record_minigrid_metrics(
        self, final_maps, map_novelties, combination_novelties, batch_logdet,
        random_feature_novelties=None,
        topk_action_mask=None, editable_mask=None, reward_components=None,
        pre_changed_focal_losses=None, difficulty_ranks=None,
        post_changed_focal_losses=None, learning_progress_values=None,
        learning_progress_ranks=None,
        novelty_ranks=None, batch_nearest_hamming=0.0,
        archive_nearest_hamming=0.0, novelty_distance_std=0.0,
    ):
        if not self.is_minigrid:
            return
        excluded = {
            self.OBJ_EMPTY,
            self.OBJ_START,
            self.OBJ_GOAL,
            OBJECT_TO_IDX.get("wall", 1),
            OBJECT_TO_IDX.get("floor", 2),
        }
        object_metrics = [self._object_distance_metrics(grid, excluded) for grid in final_maps]
        goal_positions = {
            tuple(position)
            for grid in final_maps
            for position in np.argwhere(np.asarray(grid) == self.OBJ_GOAL)
        }
        component_means = {}
        if reward_components:
            keys = (
                "learning_progress", "combination_novelty", "random_feature_novelty",
                "reward_learning_progress", "reward_combination_novelty",
                "reward_random_feature_novelty", "total",
            )
            column_names = {
                "learning_progress": "Learning_Progress",
                "combination_novelty": "Combination_Novelty",
                "random_feature_novelty": "Random_Feature_Novelty",
                "reward_learning_progress": "Reward_Learning_Progress",
                "reward_combination_novelty": "Reward_Combination_Novelty",
                "reward_random_feature_novelty": "Reward_Random_Feature_Novelty",
                "total": "Final_Generator_Reward",
            }
            component_means = {
                column_names[key]: float(np.mean([item[key] for item in reward_components]))
                for key in keys
            }
        mean_edit_rate = 0.0
        if topk_action_mask is not None and editable_mask is not None:
            selected = torch.as_tensor(topk_action_mask).detach().cpu().bool()
            editable = torch.as_tensor(editable_mask).detach().cpu().bool()
            if editable.ndim == 4:
                editable = editable[:, 0]
            rates = selected.sum(dim=(1, 2)).float() / editable.sum(dim=(1, 2)).clamp_min(1).float()
            mean_edit_rate = float(rates.mean())
        self.last_minigrid_metrics = {
            "Map_Novelty": float(np.mean(map_novelties)) if len(map_novelties) else 0.0,
            "Combination_Novelty": (
                float(np.mean(combination_novelties))
                if combination_novelties is not None and len(combination_novelties) > 0
                else 0.0
            ),
            "Random_Feature_Novelty": (
                float(np.mean(random_feature_novelties))
                if random_feature_novelties is not None and len(random_feature_novelties) > 0
                else 0.0
            ),
            "Pre_Changed_Focal_Loss": (
                float(np.mean(pre_changed_focal_losses))
                if pre_changed_focal_losses is not None and len(pre_changed_focal_losses) > 0
                else 0.0
            ),
            "Post_Changed_Focal_Loss": (
                float(np.mean(post_changed_focal_losses))
                if post_changed_focal_losses is not None and len(post_changed_focal_losses) > 0
                else 0.0
            ),
            "Learning_Progress": (
                float(np.mean(learning_progress_values))
                if learning_progress_values is not None and len(learning_progress_values) > 0
                else 0.0
            ),
            "Difficulty_Rank": (
                float(np.mean(difficulty_ranks))
                if difficulty_ranks is not None and len(difficulty_ranks) > 0 else 0.0
            ),
            "Learning_Progress_Rank": (
                float(np.mean(learning_progress_ranks))
                if learning_progress_ranks is not None and len(learning_progress_ranks) > 0 else 0.0
            ),
            "Novelty_Rank": (
                float(np.mean(novelty_ranks))
                if novelty_ranks is not None and len(novelty_ranks) > 0 else 0.0
            ),
            "Batch_Nearest_Hamming": float(batch_nearest_hamming),
            "Archive_Nearest_Hamming": float(archive_nearest_hamming),
            "Novelty_Distance_Std": float(novelty_distance_std),
            "Latent_Batch_LogDet": float(batch_logdet),
            "Mean_Object_Pair_Distance": float(np.mean([m[0] for m in object_metrics])) if object_metrics else 0.0,
            "Mean_Nearest_Object_Distance": float(np.mean([m[1] for m in object_metrics])) if object_metrics else 0.0,
            "Selected_Edit_Pair_Distance": self._selected_edit_pair_distance(topk_action_mask) if topk_action_mask is not None else 0.0,
            "Mean_Edit_Rate": mean_edit_rate,
            "Unique_Goal_Positions": len(goal_positions),
            **component_means,
        }

    def _get_warmup_iterations(self):
        raw = self.cfg.generator_agent.get("warmup_iterations", 0)
        if raw is None:
            print("[Config Warning] generator_agent.warmup_iterations is None, fallback to 0.")
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            print(f"[Config Warning] generator_agent.warmup_iterations={raw} is invalid, fallback to 0.")
            return 0

    def sync_world_model(self, state_dict):
        """Update the internal world model instance with new weights while clearing stale Hooks."""
        if state_dict is not None:
             # Critical: Clear hooks to prevent ReferenceError in state_dict()
             if hasattr(self.wm, "_state_dict_hooks"):
                 self.wm._state_dict_hooks.clear()
             if hasattr(self.wm, "_parameters"):
                 for p_name, p in self.wm._parameters.items():
                     if p is not None and hasattr(p, "_hooks"):
                         p._hooks.clear()
             
             self.wm.load_state_dict(state_dict)

    def clear_runtime_buffers(self):
        """
        Clear runtime curriculum caches at warmup boundary so adversarial phase
        starts from a clean state driven by cfg.generator_agent.warmup_iterations.
        """
        if hasattr(self.diversity, "archive"):
            self.diversity.archive.clear()
        if self.is_minigrid:
            self.map_archive.clear()
            self.combination_archive.clear()
            self.last_minigrid_metrics = {}
            self._pending_minigrid_round = None
        self.elite_buffer.clear()
        self.prev_data = None

        if self.is_bipedal:
            self.bipedal_history.clear()
            self._last_bipedal_memory = np.zeros(26, dtype=np.float32)

        if hasattr(self.ppo, "clear_buffer"):
            self.ppo.clear_buffer()

    def _zero_context(self, B, H, W):
        # (Map features, Inventory Heatmap)
        feedback_channels = 2 if not self.is_crafter and not self.is_bipedal else 1
        map_h = torch.zeros((B, feedback_channels, H, W), device=self.device)
        if self.is_crafter:
            stats_h = torch.zeros((B, 16), device=self.device)
        elif self.is_bipedal:
            stats_h = torch.zeros((B, 26), device=self.device)
        else:
            stats_h = None
        return (torch.zeros((B, 3, H, W), device=self.device), map_h, stats_h)

    def _minigrid_map_novelty(self, maps):
        """Return per-map predictive-latent novelty and batch log-determinant."""
        if not self.is_minigrid or not maps:
            return np.zeros(0, dtype=np.float32), 0.0

        map_batch = torch.as_tensor(
            np.stack(maps), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            embeddings = F.normalize(
                self.wm.encode_map_features(map_batch), p=2, dim=1
            )
            batch_similarity = embeddings @ embeddings.t()

        batch_scores = np.zeros(len(maps), dtype=np.float32)
        if len(maps) > 1:
            k = min(self.map_knn_k, len(maps) - 1)
            for idx in range(len(maps)):
                similarities = batch_similarity[idx].clone()
                similarities[idx] = -1.0
                nearest = torch.topk(similarities, k=k, largest=True).values
                batch_scores[idx] = float(
                    torch.clamp((1.0 - nearest) / 2.0, 0.0, 1.0).mean()
                )

        archive_scores = np.zeros(len(maps), dtype=np.float32)
        if self.map_archive:
            archive_batch = torch.as_tensor(
                np.stack(self.map_archive),
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                archive_embeddings = F.normalize(
                    self.wm.encode_map_features(archive_batch), p=2, dim=1
                )
                archive_similarity = embeddings @ archive_embeddings.t()
            k = min(self.map_knn_k, archive_similarity.size(1))
            nearest = torch.topk(archive_similarity, k=k, dim=1).values
            archive_scores = torch.clamp(
                ((1.0 - nearest) / 2.0).mean(dim=1), 0.0, 1.0
            ).cpu().numpy().astype(np.float32)

        if self.map_archive:
            novelty = (
                self.map_batch_archive_mix * batch_scores
                + (1.0 - self.map_batch_archive_mix) * archive_scores
            )
        else:
            novelty = batch_scores

        for map_array in maps:
            self.map_archive.append(np.asarray(map_array, dtype=np.int64).copy())
        if len(self.map_archive) > self.map_archive_size:
            del self.map_archive[: len(self.map_archive) - self.map_archive_size]

        distances = (1.0 - batch_similarity).clamp(0.0, 2.0)
        kernel = torch.exp(-distances.pow(2) / self.latent_kernel_temperature)
        sign, logdet = torch.linalg.slogdet(
            kernel + torch.eye(len(maps), device=kernel.device) * 1e-6
        )
        batch_logdet = float(logdet) if float(sign) > 0 else -float("inf")
        return novelty.astype(np.float32), batch_logdet

    @staticmethod
    def _masked_hamming_distance(
        action_a, editable_a, action_b, editable_b,
        inventory_a=None, inventory_b=None,
    ):
        """Compare map edits and, when supplied, the one-slot inventory edit."""
        action_a = np.asarray(action_a)
        action_b = np.asarray(action_b)
        valid = np.asarray(editable_a, dtype=bool) & np.asarray(editable_b, dtype=bool)
        if not np.any(valid):
            map_mismatches = 0
            compared = 0
        else:
            map_mismatches = int(np.count_nonzero(action_a[valid] != action_b[valid]))
            compared = int(np.count_nonzero(valid))
        if inventory_a is not None and inventory_b is not None:
            map_mismatches += int(
                int(np.asarray(inventory_a).reshape(-1)[0])
                != int(np.asarray(inventory_b).reshape(-1)[0])
            )
            compared += 1
        return float(map_mismatches / compared) if compared else 0.0

    def _minigrid_combination_novelty_batch(
        self, action_matrices, editable_masks, inventory_actions=None
    ):
        """Score categorical action matrices before appending them to the FIFO archive."""
        if not self.is_minigrid:
            return np.zeros(len(action_matrices), dtype=np.float32), 0.0, 0.0, 0.0

        batch_distances = [[] for _ in action_matrices]
        for i in range(len(action_matrices)):
            for j in range(i + 1, len(action_matrices)):
                distance = self._masked_hamming_distance(
                    action_matrices[i], editable_masks[i],
                    action_matrices[j], editable_masks[j],
                    None if inventory_actions is None else inventory_actions[i],
                    None if inventory_actions is None else inventory_actions[j],
                )
                batch_distances[i].append(distance)
                batch_distances[j].append(distance)

        batch_nearest = np.asarray([
            float(np.mean(sorted(distances)[:self.combination_knn_k]))
            if distances else 0.0
            for distances in batch_distances
        ], dtype=np.float32)

        archive_nearest = np.zeros(len(action_matrices), dtype=np.float32)
        all_archive_distances = []
        if self.combination_archive:
            for i, (action, editable) in enumerate(zip(action_matrices, editable_masks)):
                distances = [
                    self._masked_hamming_distance(
                        action,
                        editable,
                        item["action"],
                        item["editable"],
                        None if inventory_actions is None else inventory_actions[i],
                        item.get("inventory"),
                    )
                    for item in self.combination_archive
                ]
                all_archive_distances.extend(distances)
                if distances:
                    archive_nearest[i] = float(
                        np.mean(sorted(distances)[:self.combination_knn_k])
                    )

        if self.combination_archive:
            scores = (
                self.combination_batch_archive_mix * batch_nearest
                + (1.0 - self.combination_batch_archive_mix) * archive_nearest
            )
        else:
            scores = batch_nearest

        # Update only after every current sample has been scored.
        for index, (action, editable) in enumerate(zip(action_matrices, editable_masks)):
            self.combination_archive.append({
                "action": np.asarray(action, dtype=np.int64).copy(),
                "editable": np.asarray(editable, dtype=bool).copy(),
                "inventory": (
                    None
                    if inventory_actions is None
                    else int(np.asarray(inventory_actions[index]).reshape(-1)[0])
                ),
            })
        if len(self.combination_archive) > self.combination_archive_size:
            del self.combination_archive[:-self.combination_archive_size]

        distance_values = [value for values in batch_distances for value in values]
        return (
            np.clip(scores, 0.0, 1.0).astype(np.float32),
            float(np.mean(batch_nearest)) if len(batch_nearest) else 0.0,
            float(np.mean(archive_nearest)) if self.combination_archive and all_archive_distances else 0.0,
            float(np.std(distance_values)) if distance_values else 0.0,
        )

    @staticmethod
    def _tie_aware_percentile_rank(values, valid_mask=None):
        """Return average percentile ranks, with larger values ranked higher."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        valid = np.ones(len(values), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        ranks = np.zeros(len(values), dtype=np.float32)
        indices = np.flatnonzero(valid & np.isfinite(values))
        if len(indices) <= 1:
            return ranks
        valid_values = values[indices]
        for position, index in enumerate(indices):
            value = valid_values[position]
            less = np.sum(valid_values < value)
            equal = np.sum(valid_values == value)
            ranks[index] = (less + 0.5 * (equal - 1)) / float(len(indices) - 1)
        return np.clip(ranks, 0.0, 1.0)

    @staticmethod
    def _minigrid_reward_components(
        learning_progress,
        combination_novelty,
        random_feature_novelty=0.0,
        diversity_mode="action_hamming",
        learning_progress_weight=1.0,
        diversity_weight=1.0,
    ):
        learning_progress = float(learning_progress)
        combination_novelty = float(np.clip(combination_novelty, 0.0, 1.0))
        random_feature_novelty = float(np.clip(random_feature_novelty, 0.0, 1.0))
        learning_progress_weight = max(float(learning_progress_weight), 0.0)
        diversity_weight = max(float(diversity_weight), 0.0)
        reward_learning_progress = learning_progress_weight * learning_progress
        if diversity_mode == "random_feature_knn":
            reward_combination_novelty = 0.0
            reward_random_feature_novelty = (
                diversity_weight * random_feature_novelty
            )
        else:
            reward_combination_novelty = (
                diversity_weight * combination_novelty
            )
            reward_random_feature_novelty = 0.0
        return {
            "learning_progress": learning_progress,
            "combination_novelty": combination_novelty,
            "random_feature_novelty": random_feature_novelty,
            "reward_learning_progress": reward_learning_progress,
            "reward_combination_novelty": reward_combination_novelty,
            "reward_random_feature_novelty": reward_random_feature_novelty,
            "total": (
                reward_learning_progress
                + reward_combination_novelty
                + reward_random_feature_novelty
            ),
        }

    def _evaluate_minigrid_changed_focal_losses(self, trajectories, valid, phase):
        """Evaluate changed-effect focal loss on held-out per-layout probes."""
        losses = np.zeros(self.batch_size, dtype=np.float32)
        for index, trajectory in enumerate(trajectories):
            if not valid[index] or not trajectory:
                continue
            try:
                losses[index] = float(
                    self.wm.calc_minigrid_changed_focal_loss(trajectory)
                )
            except Exception as exc:
                print(
                    f"[GeneratorInterface] {phase} loss evaluation failed for env "
                    f"{index}: {exc}"
                )
        return losses

    def _evaluate_minigrid_post_update_probes(self, trajectories, valid):
        """Evaluate post-update probe loss and spatial history feedback."""
        losses = np.zeros(self.batch_size, dtype=np.float32)
        error_maps = np.zeros(
            (self.batch_size, self.map_height, self.map_width), dtype=np.float32
        )
        coverage_maps = np.zeros_like(error_maps)
        for index, trajectory in enumerate(trajectories):
            if not valid[index] or not trajectory:
                continue
            try:
                metrics = self.wm.calc_minigrid_probe_metrics(
                    trajectory, include_spatial=True
                )
                losses[index] = float(metrics["changed_focal_loss"])
                error_maps[index] = np.asarray(
                    metrics["error_map"], dtype=np.float32
                )
                coverage_maps[index] = np.asarray(
                    metrics["coverage_map"], dtype=np.float32
                )
            except Exception as exc:
                print(
                    "[GeneratorInterface] Post-update probe evaluation failed for "
                    f"env {index}: {exc}"
                )
        return losses, error_maps, coverage_maps

    def _set_minigrid_post_update_history(
        self, history_maps, error_maps, coverage_maps
    ):
        if self.agent_type == "random":
            return
        maps_tensor = torch.as_tensor(
            np.asarray(history_maps), dtype=torch.float32, device=self.device
        )
        feedback = np.stack([error_maps, coverage_maps], axis=1)
        feedback_tensor = torch.as_tensor(
            feedback, dtype=torch.float32, device=self.device
        )
        self.prev_data = (maps_tensor, feedback_tensor)

    def finalize_minigrid_rewards(self):
        """Apply held-out learning progress plus diversity to the pending PPO round."""
        pending = self._pending_minigrid_round
        if not self.is_minigrid or pending is None:
            return

        valid = np.asarray(pending["valid"], dtype=bool)
        pre_losses = np.asarray(pending["pre_changed_focal_losses"], dtype=np.float32)
        post_losses, post_error_maps, post_coverage_maps = (
            self._evaluate_minigrid_post_update_probes(
                pending["probe_trajectories"], valid
            )
        )
        self._set_minigrid_post_update_history(
            pending["history_maps"], post_error_maps, post_coverage_maps
        )
        learning_progress = pre_losses - post_losses
        difficulty_ranks = self._tie_aware_percentile_rank(pre_losses, valid)
        learning_progress_ranks = self._tie_aware_percentile_rank(
            learning_progress, valid
        )
        combination_values = np.asarray(
            pending["combination_novelties"], dtype=np.float32
        )
        random_feature_values = np.asarray(
            pending.get("random_feature_novelties", np.zeros(self.batch_size)),
            dtype=np.float32,
        )
        diversity_mode = str(
            getattr(self, "diversity_mode", "action_hamming")
        ).lower()
        learning_progress_weight = float(
            getattr(self.minigrid_reward_cfg, "learning_progress", 1.0)
        )
        if diversity_mode == "random_feature_knn":
            novelty_values = random_feature_values
            diversity_weight = float(
                getattr(self.minigrid_reward_cfg, "random_feature_knn", 1.0)
            )
        else:
            novelty_values = combination_values
            diversity_weight = float(
                getattr(self.minigrid_reward_cfg, "combination_novelty", 1.0)
            )
        if self.ablation_type == "no_diversity" or diversity_weight <= 0.0:
            novelty_ranks = np.zeros(self.batch_size, dtype=np.float32)
            diversity_weight = 0.0
        else:
            novelty_ranks = self._tie_aware_percentile_rank(novelty_values, valid)

        reward_components = []
        rewards = []
        for index in range(self.batch_size):
            if not valid[index]:
                reward_components.append({
                    "learning_progress": float(learning_progress[index]),
                    "combination_novelty": float(combination_values[index]),
                    "random_feature_novelty": float(random_feature_values[index]),
                    "reward_learning_progress": 0.0,
                    "reward_combination_novelty": 0.0,
                    "reward_random_feature_novelty": 0.0,
                    "total": -5.0,
                })
                rewards.append(-5.0)
                continue
            component = self._minigrid_reward_components(
                learning_progress[index], combination_values[index],
                random_feature_novelty=random_feature_values[index],
                diversity_mode=diversity_mode,
                learning_progress_weight=learning_progress_weight,
                diversity_weight=diversity_weight,
            )
            reward_components.append(component)
            rewards.append(float(component["total"]))

        buffer_rewards = getattr(self.ppo, "buffer", {}).get("reward", [])
        if buffer_rewards and len(buffer_rewards) >= self.batch_size:
            start = len(buffer_rewards) - self.batch_size
            buffer_rewards[start:start + self.batch_size] = rewards

        self._record_minigrid_metrics(
            pending["final_maps"], pending["map_novelties"], combination_values,
            pending["batch_logdet"],
            random_feature_novelties=random_feature_values,
            topk_action_mask=pending["topk_action_mask"],
            editable_mask=pending["editable_mask"], reward_components=reward_components,
            pre_changed_focal_losses=pre_losses, difficulty_ranks=difficulty_ranks,
            post_changed_focal_losses=post_losses,
            learning_progress_values=learning_progress,
            learning_progress_ranks=learning_progress_ranks,
            novelty_ranks=novelty_ranks,
            batch_nearest_hamming=pending["batch_nearest_hamming"],
            archive_nearest_hamming=pending["archive_nearest_hamming"],
            novelty_distance_std=pending["novelty_distance_std"],
        )
        self._pending_minigrid_round = None

    def _normalize_base_map(self, grid):
        """
        Normalize environment layouts to a single-sample `[H, W]` integer array.
        Some generators, notably the bipedal seeder, return extra batch/channel dims.
        """
        grid_np = np.asarray(grid)

        if grid_np.shape == (self.map_height, self.map_width):
            return grid_np

        if grid_np.ndim >= 2 and grid_np.shape[-2:] == (self.map_height, self.map_width):
            leading = int(np.prod(grid_np.shape[:-2]))
            if leading == 1:
                return grid_np.reshape(self.map_height, self.map_width)

        raise ValueError(
            f"Unexpected base map shape {grid_np.shape}; expected (*, {self.map_height}, {self.map_width})"
        )

    def _anchor_minigrid_start(self, grid_np):
        """
        Put one start tile on the base map when missing.
        """
        if self.is_crafter or self.is_bipedal:
            return grid_np
        if not getattr(self, "minigrid_anchor_start", True):
            return grid_np

        out = np.array(grid_np, copy=True)
        if np.any(out == self.OBJ_START):
            return out

        wall_id = OBJECT_TO_IDX["wall"]
        lava_id = OBJECT_TO_IDX["lava"]
        empty_id = self.OBJ_EMPTY
        h, w = out.shape

        candidates = np.argwhere(out == empty_id)
        if len(candidates) == 0:
            candidates = np.argwhere((out != wall_id) & (out != lava_id))
        if len(candidates) == 0:
            return out

        # Prefer boundary-adjacent free cells for stable starts.
        edge = []
        for y, x in candidates:
            if y == 1 or x == 1 or y == h - 2 or x == w - 2:
                edge.append((y, x))
        pick_pool = edge if len(edge) > 0 else [tuple(p) for p in candidates]
        sy, sx = random.choice(pick_pool)
        out[sy, sx] = self.OBJ_START
        return out

    def _anchor_minigrid_terminals(self, grid_np):
        """
        Ensure MiniGrid base map has exactly one start and one reachable goal.
        Extra starts/goals are removed.
        """
        if self.is_crafter or self.is_bipedal:
            return grid_np

        out = np.array(grid_np, copy=True)
        out = self._anchor_minigrid_start(out)

        wall_id = OBJECT_TO_IDX["wall"]
        lava_id = OBJECT_TO_IDX["lava"]
        h, w = out.shape

        # Keep exactly one start.
        start_positions = np.argwhere(out == self.OBJ_START)
        if len(start_positions) == 0:
            return out
        sy, sx = tuple(start_positions[0])
        for py, px in start_positions[1:]:
            out[py, px] = self.OBJ_EMPTY
        out[sy, sx] = self.OBJ_START

        # BFS distance map from start through passable cells.
        dist = np.full((h, w), -1, dtype=np.int64)
        q = deque([(sy, sx)])
        dist[sy, sx] = 0
        while q:
            y, x = q.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if dist[ny, nx] >= 0:
                    continue
                if out[ny, nx] in (wall_id, lava_id):
                    continue
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))

        # Keep one existing reachable goal if possible.
        goal_positions = np.argwhere(out == self.OBJ_GOAL)
        keep_goal = None
        for gy, gx in goal_positions:
            if dist[gy, gx] > 0:
                keep_goal = (int(gy), int(gx))
                break

        # Otherwise place goal on farthest reachable non-start tile.
        if keep_goal is None:
            candidates = np.argwhere(dist > 0)
            if len(candidates) > 0:
                dvals = dist[candidates[:, 0], candidates[:, 1]]
                gy, gx = candidates[int(np.argmax(dvals))]
                keep_goal = (int(gy), int(gx))
            else:
                # Degenerate case: start is isolated. Open one neighbor and place goal.
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = sy + dy, sx + dx
                    if 0 < ny < h - 1 and 0 < nx < w - 1:
                        out[ny, nx] = self.OBJ_EMPTY
                        keep_goal = (ny, nx)
                        break

        # Remove all previous goals and set exactly one.
        for gy, gx in goal_positions:
            out[gy, gx] = self.OBJ_EMPTY
        if keep_goal is not None:
            out[keep_goal[0], keep_goal[1]] = self.OBJ_GOAL

        return out

    def _build_bipedal_history_feature(self):
        if not self.is_bipedal or len(self.bipedal_history) == 0:
            return np.zeros(26, dtype=np.float32)
        records = list(self.bipedal_history)
        action_hist = np.mean(np.stack([rec["action_hist"] for rec in records], axis=0), axis=0)
        # combined_errors is 20-dim: [Physical(10) | Semantic(10)]
        # Use defensive padding in case older 10-dim records are still in history
        all_errs = [rec["token_errors"] for rec in records]
        # Ensure all records being stacked are 20-dim
        padded_errs = []
        for e in all_errs:
            if e.size < 20:
                e = np.pad(e, (0, 20 - e.size))
            padded_errs.append(e[:20])
        
        combined_errors = np.mean(np.stack(padded_errs, axis=0), axis=0)
        
        feat = np.zeros(26, dtype=np.float32)
        # feat[0:10] -> Original Physical Token Errors (Hull, Legs, Join, Lidar)
        # feat[10:20] -> Semantic Obstacle Errors (Action Type MSE Attribution)
        # feat[20:26] -> Grouped Action History (Frequencies)
        feat[0:10] = combined_errors[0:10]
        feat[10:20] = combined_errors[10:20]
        feat[20:26] = action_hist[:6]
        return feat

    def _update_bipedal_history(self, final_maps, token_errors):
        if not self.is_bipedal or len(final_maps) == 0:
            return

        hist_bins = np.zeros(6, dtype=np.float32)
        active_width = min(self.active_bipedal_width, self.map_width)
        for env_map in final_maps:
            flat = np.asarray(env_map, dtype=np.int64).reshape(-1)[:active_width]
            counts = np.bincount(flat.clip(min=0), minlength=len(self.ACTION_TABLE))
            hist_bins[0] += counts[1:4].sum()   # stump family
            hist_bins[1] += counts[4:6].sum()   # pit family
            hist_bins[2] += counts[6:9].sum()   # stairs family
            hist_bins[3] += counts[9] if len(counts) > 9 else 0.0  # rough terrain
            hist_bins[4] += counts[0]           # safe ground / no-op
            hist_bins[5] += float(np.count_nonzero(flat))

        norm = max(float(len(final_maps) * active_width), 1.0)
        hist_bins /= norm
        token_errors = np.asarray(token_errors, dtype=np.float32).reshape(-1)
        # Handle 20-dim (Phys + Semantic)
        if token_errors.size < 20:
            token_errors = np.pad(token_errors, (0, 20 - token_errors.size))
        elif token_errors.size > 20:
            token_errors = token_errors[:20]

        record = {
            "token_errors": token_errors,
            "action_hist": hist_bins,
        }
        self.bipedal_history.append(record)
        self._last_bipedal_memory = self._build_bipedal_history_feature()

    def _default_stats(self):
        if self.is_minigrid:
            return np.zeros(1, dtype=np.int64)
        default_stats = np.zeros(16, dtype=np.float32)
        default_stats[0] = 9.0
        default_stats[1] = 9.0
        default_stats[2] = 9.0
        default_stats[3] = 9.0
        return default_stats

    def _minigrid_empty_base(self, iteration=0, env_index=0):
        if not self.is_minigrid:
            raise RuntimeError("Canonical empty base is only defined for MiniGrid")
        if self.map_height < 4 or self.map_width < 4:
            raise ValueError("MiniGrid canonical base requires height and width >= 4")

        grid = np.full(
            (self.map_height, self.map_width),
            self.OBJ_EMPTY,
            dtype=np.int64,
        )
        wall = OBJECT_TO_IDX["wall"]
        grid[0, :] = wall
        grid[-1, :] = wall
        grid[:, 0] = wall
        grid[:, -1] = wall
        start = (1, 1)
        grid[start] = self.OBJ_START

        candidates = [
            (row, col)
            for row in range(1, self.map_height - 1)
            for col in range(1, self.map_width - 1)
            if (row, col) != start
        ]
        experiment_seed = int(getattr(getattr(self, "cfg", None), "seed", 0))
        goal_rng = np.random.default_rng(np.random.SeedSequence((
            experiment_seed,
            int(iteration),
            int(env_index),
        )))
        goal = candidates[int(goal_rng.integers(len(candidates)))]
        grid[goal] = self.OBJ_GOAL
        return grid

    def _generate_random_base_batch(self, iteration=0):
        base_maps = []
        base_stats = []
        for env_index in range(self.batch_size):
            if self.is_bipedal:
                batch_grid, _ = self.seeder.generate(1)
                grid = batch_grid[0]
            elif self.is_minigrid:
                grid = self._minigrid_empty_base(iteration, env_index)
            else:
                grid = self.seeder.generate()
            norm_grid = self._normalize_base_map(grid)
            norm_grid = self._anchor_minigrid_terminals(norm_grid)
            base_maps.append(norm_grid)
            base_stats.append(self._default_stats())
        return base_maps, base_stats

    def _step_random(self, old_params, iteration=0):
        base_maps, base_stats = self._generate_random_base_batch(iteration)
        base_ids = torch.from_numpy(np.stack(base_maps)).to(self.device).long()
        B, H, W = base_ids.shape
        zeros = torch.zeros((B, H, W), device=self.device, dtype=torch.float32)
        curr_map = torch.stack([base_ids.float(), zeros, zeros], dim=1)
        mask = self._immutable_mask(base_ids)

        actions, stats_actions, _, _, topk_action_mask, _, _, _ = self.ppo.select_action(
            curr_map, None, mask, self.max_edits_layout, self.max_edits_inventory
        )

        warmup_iters = self._get_warmup_iterations()
        is_warmup = (iteration < warmup_iters)
        valid_trajs = []
        raw_scalar_losses, div_rewards = [], []
        raw_ce_losses, raw_inv_losses = [], []
        solved_count = 0
        bfs_count = 0
        total_bfs_dist = 0.0

        base_ids_np = base_ids.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        stats_actions_np = stats_actions.detach().cpu().numpy() if stats_actions is not None else None
        base_stats_np = np.stack(base_stats)
        final_maps_obj, final_maps_color, final_maps_state, final_stats_batch = self._materialize_candidates(
            base_ids_np, actions_np, stats_actions_np, base_stats_np, mask
        )
        self._record_generated_minigrid_batch(
            final_maps_obj, final_maps_color, final_maps_state,
            final_stats_batch, iteration,
        )
        final_maps_3ch = [
            np.stack([obj, color, state], axis=0)
            for obj, color, state in zip(
                final_maps_obj, final_maps_color, final_maps_state
            )
        ]
        map_novelties, batch_logdet = (
            self._minigrid_map_novelty(final_maps_3ch)
            if self.is_minigrid else (np.zeros(self.batch_size, dtype=np.float32), 0.0)
        )
        editable_masks_np = (mask[:, 0] < 0.5).detach().cpu().numpy()
        combination_novelties, batch_nearest_hamming, archive_nearest_hamming, novelty_distance_std = (
            self._minigrid_combination_novelty_batch(
                actions_np, editable_masks_np, stats_actions_np
            )
            if self.is_minigrid else (np.zeros(self.batch_size, dtype=np.float32), 0.0, 0.0, 0.0)
        )
        random_feature_novelties = np.zeros(self.batch_size, dtype=np.float32)
        valid_flags = []
        round_trajectories = []
        probe_trajectories = []

        for i in range(self.batch_size):
            final_map_obj = final_maps_obj[i]
            final_map_col = final_maps_color[i]
            final_map_state = final_maps_state[i]
            final_stats = final_stats_batch[i]

            final_map_3ch = final_maps_3ch[i]
            res_rollout = self._rollout_combined(
                final_map_obj, final_stats, iteration, i, old_params=old_params,
                color_np=final_map_col, state_np=final_map_state,
                evaluate_wm=not is_warmup,
            )
            traj, errors, raw_loss_val, solved = res_rollout[0], res_rollout[1], res_rollout[2], res_rollout[3]
            round_trajectories.append(traj)
            if self.is_minigrid and traj and "obs" in traj:
                probe_traj = self._rollout_combined(
                    final_map_obj,
                    final_stats,
                    iteration,
                    f"{i}_lp_probe",
                    old_params=old_params,
                    color_np=final_map_col,
                    state_np=final_map_state,
                    evaluate_wm=False,
                    maximum_dataset_size=self.minigrid_lp_probe_size,
                )[0]
            else:
                probe_traj = {}
            probe_trajectories.append(probe_traj)
            errors = self._normalize_rollout_errors(errors)

            t_loss_batch = res_rollout[4] if len(res_rollout) > 4 else raw_loss_val
            i_loss_batch = res_rollout[5] if len(res_rollout) > 5 else 0.0
            inv_changed_slots = res_rollout[6] if len(res_rollout) > 6 else 0

            if self.is_crafter:
                is_connected, conn_stats = self.seeder.check_connectivity(final_map_obj)
                if is_connected:
                    solved_count += 1
                    total_bfs_dist += conn_stats.get('max_dist', 0)
                    bfs_count += 1
            elif self.is_bipedal:
                if solved:
                    solved_count += 1
                    total_bfs_dist += res_rollout[8] if len(res_rollout) > 8 else 1.0
                    bfs_count += 1
            else:
                shortest_dist = res_rollout[8] if len(res_rollout) > 8 else 0.0
                if shortest_dist > 0:
                    total_bfs_dist += shortest_dist
                    bfs_count += 1
                if shortest_dist > 0:
                    solved_count += 1

            if self.is_minigrid:
                if self.diversity_mode == "random_feature_knn":
                    random_feature_novelties[i] = self._get_diversity_reward(
                        final_map_3ch
                    )
                    r_div = float(random_feature_novelties[i])
                else:
                    r_div = float(combination_novelties[i])
            else:
                r_div = self._get_diversity_reward(
                    final_map_3ch, inventory_vec=final_stats
                )
            div_rewards.append(r_div)
            if self.is_minigrid:
                reward = 0.0  # DR has no PPO update; diagnostics are finalized later.
            else:
                reward = self._calculate_reward(
                    raw_loss_val, r_div, is_warmup,
                    inv_diversity=inv_changed_slots, ce_loss=t_loss_batch,
                    inv_loss=i_loss_batch,
                )

            if not traj or 'obs' not in traj:
                reward = -5.0
                valid_flags.append(False)
            else:
                valid_flags.append(True)
            raw_scalar_losses.append(raw_loss_val)
            raw_ce_losses.append(t_loss_batch)
            raw_inv_losses.append(i_loss_batch)

            if traj and 'obs' in traj:
                valid_trajs.append(traj)

        avg_bfs = total_bfs_dist / max(1, bfs_count) if bfs_count > 0 else 0.0
        mean_raw_loss = np.mean(raw_scalar_losses) if len(raw_scalar_losses) > 0 else 0.0
        mean_ce_loss = np.mean(raw_ce_losses) if len(raw_ce_losses) > 0 else 0.0
        mean_inv_loss = np.mean(raw_inv_losses) if len(raw_inv_losses) > 0 else 0.0
        mean_div_reward = np.mean(div_rewards) if len(div_rewards) > 0 else 0.0
        if self.is_minigrid:
            probe_valid_flags = [
                valid and bool(probe) and "obs" in probe
                for valid, probe in zip(valid_flags, probe_trajectories)
            ]
            pre_changed_focal_losses = self._evaluate_minigrid_changed_focal_losses(
                probe_trajectories, probe_valid_flags, phase="Pre-update probe"
            )
            self._record_minigrid_metrics(
                final_maps_obj, map_novelties, combination_novelties, batch_logdet,
                random_feature_novelties=random_feature_novelties,
                topk_action_mask=topk_action_mask, editable_mask=mask < 0.5,
                batch_nearest_hamming=batch_nearest_hamming,
                archive_nearest_hamming=archive_nearest_hamming,
                novelty_distance_std=novelty_distance_std,
            )
            self._pending_minigrid_round = {
                "probe_trajectories": probe_trajectories,
                "valid": probe_valid_flags,
                "pre_changed_focal_losses": pre_changed_focal_losses,
                "final_maps": final_maps_obj,
                "history_maps": final_maps_3ch,
                "map_novelties": map_novelties,
                "combination_novelties": combination_novelties,
                "random_feature_novelties": random_feature_novelties,
                "batch_logdet": batch_logdet,
                "topk_action_mask": topk_action_mask,
                "editable_mask": mask < 0.5,
                "batch_nearest_hamming": batch_nearest_hamming,
                "archive_nearest_hamming": archive_nearest_hamming,
                "novelty_distance_std": novelty_distance_std,
            }

        self.prev_data = None
        return None, None, mean_raw_loss, mean_ce_loss, mean_inv_loss, mean_div_reward, valid_trajs, solved_count, avg_bfs

    def _step_policy(self, old_params, iteration=0):
        base_maps = []
        base_stats = []
        if self.is_bipedal:
            context_heats_terrain_src = self._normalize_rollout_errors(None, self.map_height, self.map_width)["terrain"]
        else:
            context_heats_terrain_src = None

        context_maps, context_heats_terrain, context_heats_stats = [], [], []
        shared_bipedal_memory = self._build_bipedal_history_feature() if self.is_bipedal else None

        warmup_iters = self._get_warmup_iterations()
        is_warmup = (iteration < warmup_iters)
        num_random = self.batch_size

        # Unpack `prev_data` across environments with different tuple layouts.
        p_maps, p_terrain, p_stats = None, None, None
        if self.prev_data is not None:
            if len(self.prev_data) == 3:
                p_maps, p_terrain, p_stats = self.prev_data
            else:
                p_maps, p_terrain = self.prev_data
                p_stats = None

        for r_idx in range(num_random):
            if self.is_bipedal:
                batch_grid, _ = self.seeder.generate(1)
                grid = batch_grid[0]
            elif self.is_minigrid:
                grid = self._minigrid_empty_base(iteration, r_idx)
            else:
                grid = self.seeder.generate()

            grid = self._normalize_base_map(grid)
            grid = self._anchor_minigrid_terminals(grid)
            base_maps.append(grid)
            base_stats.append(self._default_stats())

            zm, zh, _ = self._zero_context(1, self.map_height, self.map_width)
            
            # Forward context adaptively across environments.
            if p_maps is not None and not is_warmup:
                idx = r_idx % p_maps.shape[0] if p_maps.shape[0] > 0 else 0
                context_maps.append(p_maps[idx:idx+1])
                context_heats_terrain.append(p_terrain[idx:idx+1])
                
                # Handle Stats/Inventory Heatmap
                if p_stats is not None:
                    curr_p_stats = p_stats[idx:idx+1]
                    if self.is_bipedal:
                        # BUGFIX(bipedal-history): Merge online errors with long-term action history
                        merged_stats = curr_p_stats.clone()
                        if shared_bipedal_memory is not None:
                            merged_stats[:, 20:26] = torch.tensor(
                                shared_bipedal_memory[20:26],
                                dtype=torch.float32,
                                device=self.device,
                            ).reshape(1, 6)
                        context_heats_stats.append(merged_stats)
                    else:
                        context_heats_stats.append(curr_p_stats)
                else:
                    # Fallback for envs without stats (MiniGrid)
                    context_heats_stats.append(torch.zeros((1, 16), device=self.device))
            else:
                # First iteration: fall back to zero context
                context_maps.append(zm)
                context_heats_terrain.append(zh)
                context_heats_stats.append(torch.zeros((1, 26 if self.is_bipedal else 16), device=self.device))

        base_ids = torch.from_numpy(np.stack(base_maps)).to(self.device).long()
        B, H, W = base_ids.shape
        zeros = torch.zeros((B, H, W), device=self.device, dtype=torch.float32)
        ppo_input_context = (
            torch.cat(context_maps),
            torch.cat(context_heats_terrain),
            torch.cat(context_heats_stats) if (self.is_crafter or self.is_bipedal) else None,
        )
        # Keep discrete map channels discrete. Terrain heat belongs in the
        # history/context stream, not in the color embedding index channel.
        curr_map = torch.stack([base_ids.float(), zeros, zeros], dim=1)
        mask = self._immutable_mask(base_ids)

        (
            actions, stats_actions, logp, values, topk_action_mask,
            location_order, topk_stats_action_mask, _,
        ) = self.ppo.select_action(
            curr_map, ppo_input_context, mask, self.max_edits_layout, self.max_edits_inventory
        )

        valid_trajs = []
        next_maps, next_heats_terrain, next_heats_stats = [], [], []
        raw_scalar_losses, div_rewards = [], []
        raw_ce_losses, raw_inv_losses = [], []
        final_maps_for_history = []
        bipedal_token_error_records = []
        all_avg_ep_lens = []
        solved_count = 0
        bfs_count = 0
        total_bfs_dist = 0.0

        base_ids_np = base_ids.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        stats_actions_np = stats_actions.detach().cpu().numpy() if stats_actions is not None else None
        base_stats_np = np.stack(base_stats)
        final_maps_obj, final_maps_color, final_maps_state, final_stats_batch = self._materialize_candidates(
            base_ids_np, actions_np, stats_actions_np, base_stats_np, mask
        )
        self._record_generated_minigrid_batch(
            final_maps_obj, final_maps_color, final_maps_state,
            final_stats_batch, iteration,
        )
        final_maps_3ch = [
            np.stack([obj, color, state], axis=0)
            for obj, color, state in zip(
                final_maps_obj, final_maps_color, final_maps_state
            )
        ]
        map_novelties, batch_logdet = (
            self._minigrid_map_novelty(final_maps_3ch)
            if self.is_minigrid else (np.zeros(self.batch_size, dtype=np.float32), 0.0)
        )
        editable_masks_np = (mask[:, 0] < 0.5).detach().cpu().numpy()
        combination_novelties, batch_nearest_hamming, archive_nearest_hamming, novelty_distance_std = (
            self._minigrid_combination_novelty_batch(
                actions_np, editable_masks_np, stats_actions_np
            )
            if self.is_minigrid else (np.zeros(self.batch_size, dtype=np.float32), 0.0, 0.0, 0.0)
        )
        random_feature_novelties = np.zeros(self.batch_size, dtype=np.float32)
        valid_flags = []
        round_trajectories = []
        probe_trajectories = []

        for i in range(self.batch_size):
            final_map_obj = final_maps_obj[i]
            final_map_col = final_maps_color[i]
            final_map_state = final_maps_state[i]
            final_stats = final_stats_batch[i]

            final_map_3ch = final_maps_3ch[i]
            res_rollout = self._rollout_combined(
                final_map_obj, final_stats, iteration, i, old_params=old_params,
                color_np=final_map_col, state_np=final_map_state,
                evaluate_wm=not is_warmup,
            )
            traj, errors, raw_loss_val, solved = res_rollout[0], res_rollout[1], res_rollout[2], res_rollout[3]
            round_trajectories.append(traj)
            if self.is_minigrid and traj and "obs" in traj:
                probe_traj = self._rollout_combined(
                    final_map_obj,
                    final_stats,
                    iteration,
                    f"{i}_lp_probe",
                    old_params=old_params,
                    color_np=final_map_col,
                    state_np=final_map_state,
                    evaluate_wm=False,
                    maximum_dataset_size=self.minigrid_lp_probe_size,
                )[0]
            else:
                probe_traj = {}
            probe_trajectories.append(probe_traj)
            errors = self._normalize_rollout_errors(errors)
            t_loss_batch = res_rollout[4] if len(res_rollout) > 4 else raw_loss_val
            i_loss_batch = res_rollout[5] if len(res_rollout) > 5 else 0.0
            inv_changed_slots = res_rollout[6] if len(res_rollout) > 6 else 0
            avg_ep_len = res_rollout[7] if len(res_rollout) > 7 else 200.0
            all_avg_ep_lens.append(avg_ep_len)

            if self.is_crafter:
                is_connected, conn_stats = self.seeder.check_connectivity(final_map_obj)
                if is_connected:
                    solved_count += 1
                    total_bfs_dist += conn_stats.get('max_dist', 0)
                    bfs_count += 1
            elif self.is_bipedal:
                if solved:
                    solved_count += 1
                    total_bfs_dist += res_rollout[8] if len(res_rollout) > 8 else 1.0
                    bfs_count += 1
            else:
                shortest_dist = res_rollout[8] if len(res_rollout) > 8 else 0.0
                if shortest_dist > 0:
                    total_bfs_dist += shortest_dist
                    bfs_count += 1
                if shortest_dist > 0:
                    solved_count += 1

            if self.is_minigrid:
                if self.diversity_mode == "random_feature_knn":
                    random_feature_novelties[i] = self._get_diversity_reward(
                        final_map_3ch
                    )
                    r_div = float(random_feature_novelties[i])
                else:
                    r_div = float(combination_novelties[i])
            else:
                r_div = self._get_diversity_reward(
                    final_map_3ch, inventory_vec=final_stats
                )
            div_rewards.append(r_div)
            if self.is_minigrid:
                reward = 0.0  # Finalized after this round's WM update.
            else:
                reward = self._calculate_reward(
                    raw_loss_val, r_div, is_warmup,
                    inv_diversity=inv_changed_slots, ce_loss=t_loss_batch,
                    inv_loss=i_loss_batch, avg_ep_len=avg_ep_len,
                )

            if not traj or 'obs' not in traj:
                reward = -5.0
                valid_flags.append(False)
            else:
                valid_flags.append(True)
            final_maps_for_history.append(final_map_obj.copy())

            cm = ppo_input_context[0][i:i+1]
            cht = ppo_input_context[1][i:i+1]
            if ppo_input_context[2] is None:
                prev_data_i = (cm, cht)
            else:
                chs = ppo_input_context[2][i:i+1]
                prev_data_i = (cm, cht, chs)

            raw_scalar_losses.append(raw_loss_val)
            raw_ce_losses.append(t_loss_batch)
            raw_inv_losses.append(i_loss_batch)
            self.ppo.save_buffer(
                curr_map[i:i+1],
                prev_data_i,
                mask[i:i+1],
                actions[i:i+1],
                (
                    stats_actions[i:i+1]
                    if self.is_crafter or self.is_minigrid
                    else torch.zeros((1, 32), device=self.device, dtype=torch.long)
                ),
                logp[i:i+1],
                values[i:i+1],
                reward,
                topk_action_mask[i:i+1],
                location_order[i:i+1],
                topk_stats_action_mask[i:i+1],
            )

            if traj and 'obs' in traj:
                valid_trajs.append(traj)
                # MiniGrid history is built from the held-out probe only after
                # the WM update, so the pre-update rollout heat is not retained.
                if self.is_minigrid:
                    continue
                next_maps.append(self._map_to_tensor(final_map_3ch))
                terrain_feedback = np.asarray(errors['terrain'], dtype=np.float32)
                if not self.is_crafter and not self.is_bipedal:
                    terrain_feedback = np.stack(
                        [terrain_feedback, np.asarray(errors['coverage'], dtype=np.float32)],
                        axis=0,
                    )
                else:
                    terrain_feedback = terrain_feedback[None, ...]
                next_heats_terrain.append(
                    torch.tensor(
                        terrain_feedback,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)
                )
                if self.debug_mode and (not self.is_bipedal):
                    h_map = errors['terrain']
                    if h_map.max() > 1e-6:
                        y, x = np.unravel_index(np.argmax(h_map), h_map.shape)
                        # MiniGrid final_map_obj is [H, W]
                        obj_id = int(final_map_obj[y, x])
                        # Look up object name
                        obj_name = [k for k, v in self.support.cfg.training_generator.map_element.items() if v == obj_id]
                        obj_name = obj_name[0] if obj_name else f"ID({obj_id})"
                        print(f"[Feedback-Debug] Iter {iteration} Env {i}: Max Error {h_map.max():.4f} at ({y},{x}) which is {obj_name}")
                    else:
                        print(f"[Feedback-Debug] Iter {iteration} Env {i}: Heatmap is zero.")

                if self.is_bipedal:
                    # Combine semantic obstacle usage with physical error signals.
                    
                    # 1. Semantic (10-dim)
                    sem_err = np.zeros(10, dtype=np.float32)
                    actions_np = actions.detach().cpu().numpy()
                    
                    # Only consider actions within the editable `active_width`.
                    active_w = getattr(self, "active_bipedal_width", 6)
                    current_actions = actions_np[i].flatten()[:active_w]
                    used_actions = np.unique(current_actions)
                    
                    # Ensure raw_loss_val is the valid scalar loss (0.3435 etc.)
                    base_loss = float(raw_loss_val) if raw_loss_val > 0 else 0.0
                    
                    for ua in used_actions:
                        if 0 < ua < 10:  # Ignore 0 (Grass) so it doesn't soak up the error
                            sem_err[ua] = base_loss
                    
                    # 2. Physical (10-dim)
                    # Handle cases where errors['inventory'] might be empty or wrong size
                    raw_phys = np.asarray(errors["inventory"], dtype=np.float32).reshape(-1)
                    phys_err = np.zeros(10, dtype=np.float32)
                    if raw_phys.size > 0:
                        n_copy = min(raw_phys.size, 10)
                        phys_err[:n_copy] = raw_phys[:n_copy]
                    
                    # 3. Combine to 20-dim combined error record
                    combined_err = np.concatenate([phys_err, sem_err])
                    bipedal_token_error_records.append(combined_err)
                    
                    # 4. Online Context (26-dim: [Phys(10) | Sem(10) | Stats(6)])
                    # We reuse the builds stats from current history feature for the padding part
                    # but typically just zero it for the online step unless we have rolling stats.
                    # For stability, carry over deque-based action-history stats for the last 6 dims.
                    online_feat_26 = np.zeros(26, dtype=np.float32)
                    online_feat_26[0:20] = combined_err
                    if shared_bipedal_memory is not None:
                        online_feat_26[20:26] = shared_bipedal_memory[20:26]
                    next_heats_stats.append(torch.tensor(online_feat_26, device=self.device).unsqueeze(0))
                else:
                    next_heats_stats.append(torch.tensor(errors['inventory'], device=self.device).unsqueeze(0))

        if len(next_maps) > 0:
            self.prev_data = (
                torch.cat(next_maps),
                torch.cat(next_heats_terrain),
                torch.cat(next_heats_stats),
            )

        avg_bfs = total_bfs_dist / max(1, bfs_count) if bfs_count > 0 else 0.0
        mean_raw_loss = np.mean(raw_scalar_losses) if len(raw_scalar_losses) > 0 else 0.0
        mean_ce_loss = np.mean(raw_ce_losses) if len(raw_ce_losses) > 0 else 0.0
        mean_inv_loss = np.mean(raw_inv_losses) if len(raw_inv_losses) > 0 else 0.0
        mean_div_reward = np.mean(div_rewards) if len(div_rewards) > 0 else 0.0
        if self.is_minigrid:
            probe_valid_flags = [
                valid and bool(probe) and "obs" in probe
                for valid, probe in zip(valid_flags, probe_trajectories)
            ]
            pre_changed_focal_losses = self._evaluate_minigrid_changed_focal_losses(
                probe_trajectories, probe_valid_flags, phase="Pre-update probe"
            )
            self._record_minigrid_metrics(
                final_maps_obj, map_novelties, combination_novelties, batch_logdet,
                random_feature_novelties=random_feature_novelties,
                topk_action_mask=topk_action_mask, editable_mask=mask < 0.5,
                batch_nearest_hamming=batch_nearest_hamming,
                archive_nearest_hamming=archive_nearest_hamming,
                novelty_distance_std=novelty_distance_std,
            )
            self._pending_minigrid_round = {
                "probe_trajectories": probe_trajectories,
                "valid": probe_valid_flags,
                "pre_changed_focal_losses": pre_changed_focal_losses,
                "final_maps": final_maps_obj,
                "history_maps": final_maps_3ch,
                "map_novelties": map_novelties,
                "combination_novelties": combination_novelties,
                "random_feature_novelties": random_feature_novelties,
                "batch_logdet": batch_logdet,
                "topk_action_mask": topk_action_mask,
                "editable_mask": mask < 0.5,
                "batch_nearest_hamming": batch_nearest_hamming,
                "archive_nearest_hamming": archive_nearest_hamming,
                "novelty_distance_std": novelty_distance_std,
            }

        if self.is_bipedal:
            # Debug Print: See if generator is actually doing anything
            # action sum > 0 means it placed at least one non-zero obstacle
            mean_token_errors = (
                np.mean(np.stack(bipedal_token_error_records, axis=0), axis=0).astype(np.float32)
                if len(bipedal_token_error_records) > 0
                else np.zeros(20, dtype=np.float32)
            )
            self._update_bipedal_history(final_maps_for_history, token_errors=mean_token_errors)

        mean_avg_ep_len = float(np.mean(all_avg_ep_lens)) if len(all_avg_ep_lens) > 0 else 0.0

        return None, None, mean_raw_loss, mean_ce_loss, mean_inv_loss, mean_div_reward, valid_trajs, solved_count, avg_bfs, mean_avg_ep_len

    def step(self, old_params, iteration=0):
        explorer = self.minigrid_rmax_explorer
        if explorer is not None:
            explorer.begin_iteration()

        if self.agent_type == 'random':
            result = self._step_random(old_params, iteration=iteration)
        else:
            result = self._step_policy(old_params, iteration=iteration)

        if explorer is not None:
            update_metrics = explorer.end_iteration()
            checkpoint_every = int(
                getattr(
                    self.cfg.domains.minigrid.rmax_like,
                    "checkpoint_every_iterations",
                    1,
                )
            )
            if (
                checkpoint_every > 0
                and explorer.completed_iteration % checkpoint_every == 0
            ):
                explorer.save_checkpoint()
            print(
                "[MiniGrid RMax iteration] "
                f"updated={bool(update_metrics.get('updated', False))} "
                f"sequences={int(update_metrics.get('sequence_count', 0))} "
                f"transitions={int(update_metrics.get('transition_count', 0))} "
                f"ppo_optimizer_steps={int(update_metrics.get('optimizer_steps', 0))} "
                f"ppo_approx_kl={float(update_metrics.get('approx_kl', 0.0)):.6f} "
                f"ppo_clip_fraction={float(update_metrics.get('clip_fraction', 0.0)):.3f} "
                f"ppo_entropy={float(update_metrics.get('entropy', 0.0)):.3f} "
                f"ppo_updates={explorer.update_count} "
                f"ride_updates={int(update_metrics.get('ride', {}).get('update_count', 0))} "
                f"ride_transitions={int(update_metrics.get('ride', {}).get('transition_count', 0))} "
                f"ride_sampled={int(update_metrics.get('ride', {}).get('sampled_transition_count', 0))} "
                f"ride_action_counts={update_metrics.get('ride', {}).get('per_action_counts', [])} "
                f"ride_inverse_accuracy={float(update_metrics.get('ride', {}).get('inverse_accuracy', 0.0)):.3f} "
                f"ride_per_action_inverse_accuracy={update_metrics.get('ride', {}).get('per_action_inverse_accuracy', [])} "
                f"ride_forward_loss={float(update_metrics.get('ride', {}).get('forward_loss', 0.0)):.6f} "
                f"ride_inverse_loss={float(update_metrics.get('ride', {}).get('inverse_loss', 0.0)):.6f}"
            )
        return result

    def _rollout_combined(
        self, map_np, stats_np, iter, idx, old_params=None, color_np=None,
        state_np=None, evaluate_wm=True, maximum_dataset_size=None,
    ):
        import traceback

        env_type = str(getattr(self.cfg.attention_model, "env_type", "")).lower()
        try:
             # Use the modernized support.interpret_env
             # Note: For MiniGrid, stats_np is ignored by its version of support/interpret
             if env_type == "crafter":
                 interpreted = self.support.interpret_env(map_np, inventory_vec=stats_np)
                 env_source = interpreted[0] if isinstance(interpreted, tuple) else interpreted
             elif env_type == "minigrid":
                 interpreted = self.support.interpret_env(
                     map_np, color_array=color_np, state_array=state_np
                 )
                 if isinstance(interpreted, tuple) and len(interpreted) == 2:
                     # collect_data_general expects (layout_str, color_str) for MiniGrid.
                     env_source = interpreted
                 else:
                     raise ValueError(
                         f"MiniGrid interpret_env must return (layout_str, color_str), got: {type(interpreted)}"
                     )
             elif env_type == "bipedalwalker":
                 interpreted = self.support.interpret_env(map_np)
                 env_source = interpreted[0] if isinstance(interpreted, tuple) else interpreted
             else:
                 raise ValueError(f"Unsupported env_type for rollout routing: {env_type}")

             save_name = f'UED_Dual_iter{iter}_b{idx}'
             original_maximum_dataset_size = getattr(
                 self.support.cfg.env.collect, "maximum_dataset_size", None
             )
             original_data_type = str(self.support.cfg.env.collect.data_type)
             original_save_coverage = bool(
                 getattr(
                     self.support.cfg.env.collect,
                     "save_coverage_visualize",
                     False,
                 )
             )
             original_visualize_save_path = str(
                 self.support.cfg.env.collect.visualize_save_path
             )
             try:
                exploration_policy = self.minigrid_rmax_explorer
                if exploration_policy is not None:
                    exploration_policy.set_training(bool(evaluate_wm))
                    self.support.cfg.env.collect.data_type = "rmax"
                    save_minienv_coverage = bool(
                        getattr(
                            self.cfg.domains.minigrid.rmax_like,
                            "save_minienv_coverage",
                            False,
                        )
                    )
                    self.support.cfg.env.collect.save_coverage_visualize = (
                        save_minienv_coverage and bool(evaluate_wm)
                    )
                    self.support.cfg.env.collect.visualize_save_path = os.path.join(
                        original_visualize_save_path,
                        "minienv_coverage",
                        str(self.agent_type),
                        f"seed{int(getattr(self.cfg, 'seed', 0))}",
                    )
                save_path = collect_data_general(
                    self.support.cfg,
                    env_source=env_source,
                    save_name=save_name,
                    maximum_dataset_size=maximum_dataset_size,
                    recollect_data=True,
                    initial_carrying_token=(
                        int(np.asarray(stats_np).reshape(-1)[0])
                        if env_type == "minigrid"
                        else None
                    ),
                    policy=exploration_policy,
                    intrinsic_reward_fn=(
                        exploration_policy.intrinsic_reward
                        if exploration_policy is not None and evaluate_wm
                        else None
                    ),
                    # The WM dataset keeps native environment rewards; RIDE
                    # reward is used only to train the exploration policy.
                    store_intrinsic_reward=False,
                )
                if exploration_policy is not None and evaluate_wm:
                    print(
                        "[MiniGrid RMax] "
                        f"unique_semantic_states={exploration_policy.unique_semantic_states} "
                        f"unique_positions={exploration_policy.episodic_visited_positions} "
                        f"ppo_updates={exploration_policy.update_count}"
                    )
                    metrics = exploration_policy.rollout_metrics
                    print(
                        "[MiniGrid RMax metrics] "
                        f"rewarded_rate={metrics['rewarded_transition_rate']:.3f} "
                        f"no_change_rate={metrics['semantic_no_change_rate']:.3f} "
                        f"movement_rate={metrics['movement_rate']:.3f} "
                        f"impact_by_action={metrics['ride_impact_by_action']} "
                        f"reward_by_action={metrics['intrinsic_reward_by_action']} "
                        f"pickup={int(metrics['pickup_successes'])} "
                        f"toggle={int(metrics['toggle_successes'])} "
                        f"drop={int(metrics['drop_successes'])}"
                    )
             finally:
                 self.support.cfg.env.collect.maximum_dataset_size = (
                     original_maximum_dataset_size
                 )
                 self.support.cfg.env.collect.data_type = original_data_type
                 self.support.cfg.env.collect.save_coverage_visualize = (
                     original_save_coverage
                 )
                 self.support.cfg.env.collect.visualize_save_path = (
                     original_visualize_save_path
                 )
             if not os.path.exists(save_path):
                 print(f"[GeneratorInterface] Missing rollout file after collection: {save_path}")
                 return {}, {}, 0.0, False, 0.0, 0.0, 0

             # Load trajectory first. Even if validation later fails, keep the rollout.
             task_npz = np.load(save_path, allow_pickle=True)
             traj = {
                 'obs': torch.tensor(task_npz['a'], device=self.device),
                 'obs_next': torch.tensor(task_npz['b'], device=self.device),
                 'act': torch.tensor(task_npz['c'], device=self.device),
                 'done': task_npz['e'] if 'e' in task_npz else None,
                 'info': task_npz['f'] if 'f' in task_npz else None,
                 'inv': torch.tensor(task_npz['g'], device=self.device) if 'g' in task_npz else None,
                 'inv_next': torch.tensor(task_npz['h'], device=self.device) if 'h' in task_npz else None
             }
             solved = np.any((task_npz['e']) & (task_npz['d'] > 0))
             
             # Compute average episode length for the survival reward.
             done_flags = task_npz['e'].astype(bool).flatten()
             num_dones = max(1, int(done_flags.sum()))
             total_steps = len(done_flags)
             avg_ep_len = float(total_steps) / num_dones
             
             inv_changed_slots = 0
             if 'g' in task_npz and 'h' in task_npz:
                 inv_arr = task_npz['g'].astype(np.float32)
                 inv_next_arr = task_npz['h'].astype(np.float32)
                 delta = inv_next_arr - inv_arr
                 inv_changed_slots = int(np.any(delta != 0, axis=0).sum())

             shortest_dist = 0.0
             if env_type == "minigrid":
                 _, shortest_dist = check_solvability(map_np)

             error_dict = {
                 "terrain": np.zeros((self.map_height, self.map_width)),
                 "coverage": np.zeros((self.map_height, self.map_width)),
                 "inventory": np.zeros(16),
             }
             mean_loss = 0.0
             mean_aux_metric = 0.0
             mean_inv_loss = 0.0
             try:
                 if not evaluate_wm:
                     return (
                         traj, error_dict, 0.0, solved, 0.0, 0.0,
                         inv_changed_slots, avg_ep_len, shortest_dist,
                     )
                 # Extract Dual-Head Error Signal
                 v_times = getattr(self.cfg.attention_model, "valid_times", 1)
                 res_eval = extract_loss_map_over_validations(self.cfg, self.wm, old_params, save_path, valid_times=v_times)
                 error_dict, loss_list = res_eval[0], res_eval[1]
                 aux_metric_list, inv_losses = res_eval[2], res_eval[3]
                 mean_loss = float(np.mean(loss_list)) if len(loss_list) > 0 else 0.0
                 mean_aux_metric = float(np.mean(aux_metric_list)) if len(aux_metric_list) > 0 else 0.0
                 mean_inv_loss = float(np.mean(inv_losses)) if len(inv_losses) > 0 else 0.0
             except Exception as e:
                 print(f"[GeneratorInterface] Validation failed for {save_name}: {e}")
                 traceback.print_exc()

             return traj, error_dict, mean_loss, solved, mean_aux_metric, mean_inv_loss, inv_changed_slots, avg_ep_len, shortest_dist
        except Exception as e:
             print(f"[GeneratorInterface] Rollout failed for env_type={env_type}, iter={iter}, idx={idx}: {e}")
             traceback.print_exc()
             return {}, {
                 "terrain": np.zeros((self.map_height, self.map_width)),
                 "coverage": np.zeros((self.map_height, self.map_width)),
                 "inventory": np.zeros(16),
             }, 0.0, False, 0.0, 0.0, 0, 0.0, 0.0

    def _ensure_minigrid_goal(self, obj, mask=None):
        """
        DR's MiniGrid seeder creates only wall/empty canvases, while `_apply_action`
        guarantees an agent but does not otherwise create a goal. Add one on a
        reachable empty tile so generated environments can terminate successfully.
        """
        if self.is_crafter or self.is_bipedal:
            return obj
        if np.any(obj == self.OBJ_GOAL):
            return obj

        agent_positions = np.argwhere(obj == self.OBJ_START)
        if len(agent_positions) == 0:
            return obj

        wall_id = OBJECT_TO_IDX["wall"]
        lava_id = OBJECT_TO_IDX["lava"]
        empty_id = self.OBJ_EMPTY
        start = tuple(agent_positions[0])
        h, w = obj.shape

        dist = np.full((h, w), -1, dtype=np.int64)
        queue = deque([start])
        dist[start] = 0
        while queue:
            y, x = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if dist[ny, nx] >= 0:
                    continue
                if obj[ny, nx] in (wall_id, lava_id):
                    continue
                dist[ny, nx] = dist[y, x] + 1
                queue.append((ny, nx))

        candidates = np.argwhere((obj == empty_id) & (dist > 0))
        if len(candidates) == 0:
            candidates = np.argwhere((obj == empty_id) & (np.arange(h)[:, None] > 0))
        if len(candidates) == 0:
            candidates = np.argwhere((obj != wall_id) & (obj != lava_id) & (obj != self.OBJ_START))
        if len(candidates) == 0:
            return obj

        candidate_dists = dist[candidates[:, 0], candidates[:, 1]]
        # Prefer farthest reachable tile; if all are unreachable, fall back to the first candidate.
        best_idx = int(np.argmax(candidate_dists))
        gy, gx = candidates[best_idx]
        obj[gy, gx] = self.OBJ_GOAL
        return obj

    def _enforce_single_minigrid_goal(self, obj):
        """
        Keep at most one goal tile for MiniGrid maps.
        Extra goals are converted to empty.
        """
        if self.is_crafter or self.is_bipedal:
            return obj
        goal_positions = np.argwhere(obj == self.OBJ_GOAL)
        if len(goal_positions) <= 1:
            return obj
        out = np.array(obj, copy=True)
        # Keep the first goal and clear the rest.
        for gy, gx in goal_positions[1:]:
            out[gy, gx] = self.OBJ_EMPTY
        return out

    def _apply_action(self, base_map, act, mask=None):
        # Local terrain modification logic (Spatial Head)
        obj = base_map.copy()
        color = np.zeros_like(obj)
        state = np.zeros_like(obj)
        H, W = obj.shape
        
        # Track limits for restricted items
        restricted_limits = {}
        counts = {}
        if self.is_crafter:
            restricted_limits = {
                CRAFTER_OBJ_MAP['diamond']: 1,
                CRAFTER_OBJ_MAP['table']: 1,
                CRAFTER_OBJ_MAP['furnace']: 1
            }
            # Initialize counts with existing items on base map
            for r_id in restricted_limits:
                counts[r_id] = np.sum(obj == r_id)
        elif (not self.is_crafter) and (not self.is_bipedal):
            # MiniGrid terminals are fixed by base map + immutable mask.
            restricted_limits = {}

        for i in range(H):
            for j in range(W):
                # Skip if immutable (mask is 1.0 for boundaries)
                if mask is not None and mask[i, j] > 0:
                    continue
                try:
                    a = int(act[i, j])
                except (TypeError, ValueError):
                    continue
                if a == 0: continue
                
                if self.is_bipedal:
                    obj[i, j] = a
                    continue
                    
                val = self.ACTION_TABLE.get(a)
                
                if val is not None: 
                    color_name = None
                    door_state = STATE_TO_IDX["closed"]
                    # MiniGrid action table may encode (object_name, color_name).
                    # Preserve both object and color so generated DR maps match
                    # target color distributions.
                    if isinstance(val, tuple):
                        obj_name = val[0] if len(val) > 0 else None
                        color_name = val[1] if len(val) > 1 else None
                        door_state = int(val[2]) if len(val) > 2 else door_state
                        if isinstance(obj_name, str):
                            val = OBJECT_TO_IDX.get(obj_name, None)
                        elif obj_name is None:
                            val = None
                        else:
                            try:
                                val = int(obj_name)
                            except (TypeError, ValueError):
                                val = None
                        if val is None:
                            continue

                    # Enforce restriction limits
                    if self.is_crafter and val in restricted_limits:
                        if counts[val] >= restricted_limits[val]:
                            continue # Ignore this action, limit reached
                        counts[val] += 1
                    
                    obj[i, j] = val
                    if color_name is not None:
                        color[i, j] = COLOR_TO_IDX.get(color_name, 0)
                    if self.is_minigrid and val == OBJECT_TO_IDX["door"]:
                        state[i, j] = door_state

                    # No hard rollback on unsolvable edits:
                    # MiniGrid agent learns solvability through reward signal.
        
        # Mandatory Agent Placement Fallback (Ensures environment can always start)
        if not self.is_bipedal and not np.any(obj == self.OBJ_START):
            candidate_positions = []
            grass_positions = []
            for r in range(H):
                for c in range(W):
                    if mask is not None and mask[r, c] > 0: continue
                    candidate_positions.append((r, c))
                    if obj[r, c] == self.OBJ_EMPTY: # self.OBJ_EMPTY is 'grass' for Crafter
                        grass_positions.append((r, c))
            
            # Prioritize grass, fallback to any non-boundary tile
            pick_list = grass_positions if len(grass_positions) > 0 else candidate_positions
            if len(pick_list) > 0:
                spawn_r, spawn_c = random.choice(pick_list)
                obj[spawn_r, spawn_c] = self.OBJ_START

        if (not self.is_crafter) and (not self.is_bipedal):
            # Safety net for legacy maps entering this path.
            obj = self._anchor_minigrid_terminals(obj)
                
        return obj, color, state

    def _normalize_rollout_errors(self, err_dict, H=None, W=None):
        if H is None: H = self.map_height
        if W is None: W = self.map_width
        inventory = np.zeros(16, dtype=np.float32)
        coverage = np.zeros((H, W), dtype=np.float32)

        def _normalize_terrain_array(arr):
            terrain = np.asarray(arr, dtype=np.float32)
            if terrain.shape == (H, W):
                return terrain
            if terrain.size == H * W:
                return terrain.reshape(H, W)
            return np.zeros((H, W), dtype=np.float32)
        
        if self.is_bipedal:
            # --- [Bipedal Custom Logic] ---
            # Instead of a misleading semantic-to-spatial heatmap, 
            # we provide a clean "Activity Mask" (1.0 where editing is allowed).
            # This forces the CNN to use 'ctx' for semantic decisions (what to place)
            # while knowing exactly where its 'workspace' is.
            heatmap = np.zeros((H, W), dtype=np.float32)
            active_w = min(getattr(self, "active_bipedal_width", W), W)
            heatmap[:, :active_w] = 1.0
            if isinstance(err_dict, dict) and "inventory" in err_dict:
                inventory = np.asarray(err_dict["inventory"], dtype=np.float32).reshape(-1)
                if inventory.size < 16:
                    inventory = np.pad(inventory, (0, 16 - inventory.size))
                elif inventory.size > 16:
                    inventory = inventory[:16]
            return {"terrain": heatmap, "coverage": coverage, "inventory": inventory}

        # Original Spatial logic for Minigrid/Crafter
        if isinstance(err_dict, np.ndarray):
            # Already spatial (e.g. from Crafter validation heatmap)
            return {
                "terrain": _normalize_terrain_array(err_dict),
                "coverage": coverage,
                "inventory": inventory,
            }
        if isinstance(err_dict, dict):
            terrain = _normalize_terrain_array(err_dict.get("terrain", np.zeros((H, W), dtype=np.float32)))
            coverage = _normalize_terrain_array(
                err_dict.get("coverage", np.zeros((H, W), dtype=np.float32))
            )
            inv = err_dict.get("inventory", inventory)
            inv = np.asarray(inv, dtype=np.float32).reshape(-1)
            if inv.size < 16:
                inv = np.pad(inv, (0, 16 - inv.size))
            elif inv.size > 16:
                inv = inv[:16]
            return {"terrain": terrain, "coverage": coverage, "inventory": inv}
        return {
            "terrain": np.zeros((H, W), dtype=np.float32),
            "coverage": coverage,
            "inventory": inventory,
        }

    def _calculate_reward(
        self,
        raw_loss,
        div_score,
        is_warmup,
        inv_diversity=0,
        ce_loss=None,
        inv_loss=None,
        avg_ep_len=200.0,
    ):
        if is_warmup and self.is_crafter:
            reward_cfg = self.crafter_reward_cfg
            w_div = float(getattr(reward_cfg, "div", getattr(self.cfg.generator_agent, "reward_w_div", 3.0)))
            w_inv_change = float(getattr(reward_cfg, "inv_change", getattr(self.cfg.generator_agent, "reward_w_inv_change", 0.0)))
            inv_change_norm_slots = float(getattr(reward_cfg, "inv_change_norm_slots", getattr(self.cfg.generator_agent, "inv_change_norm_slots", 8.0)))
            bias = float(getattr(reward_cfg, "bias", getattr(self.cfg.generator_agent, "reward_bias", 1.0)))
            reward_clip = float(getattr(reward_cfg, "clip", getattr(self.cfg.generator_agent, "reward_clip", 50.0)))

            inv_changed = max(float(inv_diversity), 0.0)
            inv_norm = max(inv_change_norm_slots, 1e-6)
            inv_change_bonus = min(inv_changed / inv_norm, 1.0)

            # `no_diversity` ablation disables diversity and inventory-change rewards.
            if self.ablation_type == "no_diversity":
                div_score = 0.0
                inv_change_bonus = 0.0

            reward = (
                w_div * float(div_score)
                + w_inv_change * inv_change_bonus
                + bias
            )
            return float(np.clip(reward, -reward_clip, reward_clip))

        if is_warmup and self.is_bipedal:
            w_survival = float(getattr(self.bipedal_reward_cfg, "survival", 0.08))
            return 1.0 + div_score * 5.0 + w_survival * avg_ep_len
        
        if self.is_bipedal:
            # For bipedal, validation returns:
            # - inv_loss slot: contact BCE (lower is better)
            # We intentionally do NOT use contact accuracy here because it
            # saturates early and is not a reliable training signal.
            reward_cfg = self.bipedal_reward_cfg
            w_bce = float(getattr(reward_cfg, "contact_bce", getattr(self.cfg.generator_agent, "reward_w_inv", 3.0)))
            w_div = float(getattr(reward_cfg, "div", getattr(self.cfg.generator_agent, "reward_w_div", 3.0)))
            w_total = float(getattr(reward_cfg, "total_loss", getattr(self.cfg.generator_agent, "reward_w_total", 0.0)))
            bias = float(getattr(reward_cfg, "bias", getattr(self.cfg.generator_agent, "reward_bias", 2.0)))
            reward_clip = float(getattr(reward_cfg, "clip", getattr(self.cfg.generator_agent, "reward_clip", 100.0)))

            contact_bce = float(inv_loss) if inv_loss is not None else 0.0
            total_loss = float(raw_loss)

            # Use log-scaled MSE to preserve reward resolution at small error values.
            mse_reward_term = float(np.log10(total_loss * 50.0 + 1.0))

            # Add survival as an independent reward term based on episode length.
            survival_ratio = avg_ep_len
            w_survival = float(getattr(reward_cfg, "survival", 3.0))

            reward = (
                + w_bce * contact_bce
                + w_total * mse_reward_term
                + w_div * float(div_score)
                + w_survival * survival_ratio
                + bias
            )
            reward = float(np.clip(reward, -reward_clip, reward_clip))

            return reward

        # Crafter: No BFS, Pure adversarial reward loop. 
        # Any environment is a valid challenge for the World Model.
        if self.is_crafter:
            reward_cfg = self.crafter_reward_cfg
            w_ce = float(getattr(reward_cfg, "ce", getattr(self.cfg.generator_agent, "reward_w_ce", 5.0)))
            w_inv = float(getattr(reward_cfg, "inv", getattr(self.cfg.generator_agent, "reward_w_inv", 5.0)))
            w_div = float(getattr(reward_cfg, "div", getattr(self.cfg.generator_agent, "reward_w_div", 3.0)))
            w_inv_change = float(getattr(reward_cfg, "inv_change", getattr(self.cfg.generator_agent, "reward_w_inv_change", 0.0)))
            inv_change_norm_slots = float(getattr(reward_cfg, "inv_change_norm_slots", getattr(self.cfg.generator_agent, "inv_change_norm_slots", 8.0)))
            bias = float(getattr(reward_cfg, "bias", getattr(self.cfg.generator_agent, "reward_bias", 2.0)))
            reward_clip = float(getattr(reward_cfg, "clip", getattr(self.cfg.generator_agent, "reward_clip", 100.0)))

            ce_term = float(ce_loss) if ce_loss is not None else float(raw_loss)
            inv_term = float(inv_loss) if inv_loss is not None else 0.0
            inv_changed = max(float(inv_diversity), 0.0)
            inv_norm = max(inv_change_norm_slots, 1e-6)
            inv_change_bonus = min(inv_changed / inv_norm, 1.0)

                # `no_diversity` ablation disables diversity and inventory-change rewards.
            if self.ablation_type == "no_diversity":
                div_score = 0.0
                inv_change_bonus = 0.0
            reward = (
                w_ce * ce_term
                + w_inv * inv_term
                + w_div * float(div_score)
                + w_inv_change * inv_change_bonus
                + bias
            )
            reward = float(np.clip(reward, -reward_clip, reward_clip))

            return reward
            
        raise RuntimeError("_calculate_reward is only used for Crafter and BipedalWalker")

    def _immutable_mask(self, ids):
        mask = torch.zeros_like(ids, dtype=torch.float32)
        if self.is_bipedal:
            active_width = min(getattr(self, "active_bipedal_width", ids.shape[-1]), ids.shape[-1])
            if active_width < ids.shape[-1]:
                mask[:, :, active_width:] = 1.0
            return mask.unsqueeze(1)
        # 1. Protect Boundary Water
        mask[:, 0, :] = 1.0; mask[:, -1, :] = 1.0; mask[:, :, 0] = 1.0; mask[:, :, -1] = 1.0
        # 2. Protect Agent's Position (Don't erase/move the player once placed)
        mask[ids == self.OBJ_START] = 1.0
        # 3. For MiniGrid keep goal fixed on base map as well.
        if (not self.is_crafter) and (not self.is_bipedal):
            mask[ids == self.OBJ_GOAL] = 1.0
        return mask.unsqueeze(1)

    def _map_to_tensor(self, m):
        return torch.tensor(m, device=self.device).float().unsqueeze(0)

    def update(self, iteration=None):
        loss, ent = self.ppo.update(iteration=iteration)
        return loss, ent, self.ppo.last_mean_reward
