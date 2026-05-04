import torch
import numpy as np
import torch.nn.functional as F
import os
import random
from collections import deque

from generator.generator_agent import GeneratorPPO
from generator.random_generator_agent import RandomGeneratorAgent
from generator.reward_system import DiversityModule, check_solvability
from generator.minigrid_env_designer import PCGSeeder as MinigridPCGSeeder, task_placer as minigrid_task_placer
from generator.crafter_env_designer import CrafterPCGSeeder, CRAFTER_ACTION_MAP, CRAFTER_OBJ_MAP
from generator.bipedal_env_designer import BipedalPCGSeeder, ACTION_TABLE_BIPEDAL
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
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
    11: ("empty", None),
    12: ("goal", None),
}

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

        if self.is_crafter:
            # Crafter: use custom seeder, actions, and map sizes
            # NOTE: map_height now ONLY includes the physical layout (No more +1 hack!)
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
            # MiniGrid: use standard seeder
            self.map_height = hparams.map_height
            self.map_width = hparams.map_width
            self.seeder = MinigridPCGSeeder(height=self.map_height, width=self.map_width)
            self.ACTION_TABLE = ACTION_TABLE_MINIGRID
            # Keep start/goal fixed from base map, editor edits around them.
            self.minigrid_anchor_start = True
            self.minigrid_goal_fallback = False

        if agent_type == 'random':
            self.ppo = RandomGeneratorAgent(num_actions=len(self.ACTION_TABLE), device=device)
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
            )
        
        self.div_k = hparams.div_k
        self.diversity = DiversityModule(
            self.map_height, self.map_width, self.div_k, 
            env_type=('crafter' if self.is_crafter else ('bipedalwalker' if self.is_bipedal else 'minigrid'))
        )
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
        self.minigrid_reward_cfg = self._get_minigrid_reward_cfg()
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
        self.elite_buffer.clear()
        self.prev_data = None

        if self.is_bipedal:
            self.bipedal_history.clear()
            self._last_bipedal_memory = np.zeros(26, dtype=np.float32)

        if hasattr(self.ppo, "clear_buffer"):
            self.ppo.clear_buffer()

    def _zero_context(self, B, H, W):
        # (Map features, Inventory Heatmap)
        # Spatial heat map: [B, 1, H, W]
        map_h = torch.zeros((B, 1, H, W), device=self.device)
        if self.is_crafter:
            stats_h = torch.zeros((B, 16), device=self.device)
        elif self.is_bipedal:
            stats_h = torch.zeros((B, 26), device=self.device)
        else:
            stats_h = None
        return (torch.zeros((B, 3, H, W), device=self.device), map_h, stats_h)

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
        default_stats = np.zeros(16, dtype=np.float32)
        default_stats[0] = 9.0
        default_stats[1] = 9.0
        default_stats[2] = 9.0
        default_stats[3] = 9.0
        return default_stats

    def _generate_random_base_batch(self):
        base_maps = []
        base_stats = []
        for _ in range(self.batch_size):
            if self.is_bipedal:
                batch_grid, _ = self.seeder.generate(1)
                grid = batch_grid[0]
            else:
                grid = self.seeder.generate()
            norm_grid = self._normalize_base_map(grid)
            norm_grid = self._anchor_minigrid_terminals(norm_grid)
            base_maps.append(norm_grid)
            base_stats.append(self._default_stats())
        return base_maps, base_stats

    def _step_random(self, old_params, iteration=0):
        base_maps, base_stats = self._generate_random_base_batch()
        base_ids = torch.from_numpy(np.stack(base_maps)).to(self.device).long()
        B, H, W = base_ids.shape
        zeros = torch.zeros((B, H, W), device=self.device, dtype=torch.float32)
        curr_map = torch.stack([base_ids.float(), zeros, zeros], dim=1)
        mask = self._immutable_mask(base_ids)

        actions, stats_actions, _, _, _, _, _ = self.ppo.select_action(
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

        for i in range(self.batch_size):
            final_map_obj, final_map_col = self._apply_action(base_ids_np[i], actions_np[i], mask=mask[i, 0].cpu().numpy())
            final_stats = base_stats_np[i].copy()

            if self.is_crafter and getattr(self.cfg.generator_agent, "random_stats_actions", False):
                current_stats_act = (np.random.rand(32) < 0.15).astype(np.int64)
            else:
                current_stats_act = stats_actions_np[i] if stats_actions_np is not None else None

            if self.is_crafter and current_stats_act is not None:
                for k_idx in range(32):
                    if current_stats_act[k_idx] == 1:
                        slot = k_idx % 16
                        if k_idx < 16:
                            final_stats[slot] += 1
                        else:
                            final_stats[slot] += 5

            final_map_2ch = np.stack([final_map_obj, final_map_col], axis=0)
            res_rollout = self._rollout_combined(final_map_obj, final_stats, iteration, i, old_params=old_params, color_np=final_map_col)
            traj, errors, raw_loss_val, solved = res_rollout[0], res_rollout[1], res_rollout[2], res_rollout[3]
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
                if solved:
                    solved_count += 1

            is_connected_final = is_connected if self.is_crafter else solved
            mg_reach_ratio, mg_norm_dist = (0.0, 0.0)
            if (not self.is_crafter) and (not self.is_bipedal):
                mg_reach_ratio, mg_norm_dist = self._minigrid_path_metrics(final_map_obj)
            r_div = self.diversity.get_reward(torch.tensor(final_map_2ch).unsqueeze(0).to(self.device), inventory_vec=final_stats)
            div_rewards.append(r_div)
            reward = self._calculate_reward(
                raw_loss_val,
                r_div,
                is_connected_final,
                is_warmup,
                inv_diversity=inv_changed_slots,
                ce_loss=t_loss_batch,
                inv_loss=i_loss_batch,
                minigrid_reach_ratio=mg_reach_ratio,
                minigrid_norm_dist=mg_norm_dist,
            )

            if not traj or 'obs' not in traj:
                reward = -5.0
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

        # [NEW] Robust prev_data unpacking for all envs (MiniGrid: 2 elements, others: 3)
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
            else:
                grid = self.seeder.generate()

            grid = self._normalize_base_map(grid)
            grid = self._anchor_minigrid_terminals(grid)
            base_maps.append(grid)
            base_stats.append(self._default_stats())

            zm, zh, _ = self._zero_context(1, self.map_height, self.map_width)
            
            # [MODIFIED] Universal adaptive context forwarding for all environments
            if p_maps is not None:
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

        actions, stats_actions, logp, values, topk_action_mask, topk_stats_action_mask, _ = self.ppo.select_action(
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

        for i in range(self.batch_size):
            final_map_obj, final_map_col = self._apply_action(base_ids_np[i], actions_np[i], mask=mask[i, 0].cpu().numpy())
            final_stats = base_stats_np[i].copy()

            if self.is_crafter and getattr(self.cfg.generator_agent, "random_stats_actions", False):
                current_stats_act = (np.random.rand(32) < 0.15).astype(np.int64)
            else:
                current_stats_act = stats_actions_np[i] if stats_actions_np is not None else None

            if self.is_crafter and current_stats_act is not None:
                for k_idx in range(32):
                    if current_stats_act[k_idx] == 1:
                        slot = k_idx % 16
                        if k_idx < 16:
                            final_stats[slot] += 1
                        else:
                            final_stats[slot] += 5

            final_map_3ch = np.stack([final_map_obj, final_map_col, np.zeros_like(final_map_obj)], axis=0)
            res_rollout = self._rollout_combined(final_map_obj, final_stats, iteration, i, old_params=old_params, color_np=final_map_col)
            traj, errors, raw_loss_val, solved = res_rollout[0], res_rollout[1], res_rollout[2], res_rollout[3]
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
                if solved:
                    solved_count += 1

            is_connected_final = is_connected if self.is_crafter else solved
            mg_reach_ratio, mg_norm_dist = (0.0, 0.0)
            if (not self.is_crafter) and (not self.is_bipedal):
                mg_reach_ratio, mg_norm_dist = self._minigrid_path_metrics(final_map_obj)
            r_div = self.diversity.get_reward(torch.tensor(final_map_3ch).unsqueeze(0).to(self.device), inventory_vec=final_stats)
            div_rewards.append(r_div)
            reward = self._calculate_reward(
                raw_loss_val,
                r_div,
                is_connected_final,
                is_warmup,
                inv_diversity=inv_changed_slots,
                ce_loss=t_loss_batch,
                inv_loss=i_loss_batch,
                avg_ep_len=avg_ep_len,
                minigrid_reach_ratio=mg_reach_ratio,
                minigrid_norm_dist=mg_norm_dist,
            )

            if not traj or 'obs' not in traj:
                reward = -5.0
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
                stats_actions[i:i+1] if self.is_crafter else torch.zeros((1, 32), device=self.device, dtype=torch.long),
                logp[i:i+1],
                values[i:i+1],
                reward,
                topk_action_mask[i:i+1],
                topk_stats_action_mask[i:i+1],
            )

            if traj and 'obs' in traj:
                valid_trajs.append(traj)
                next_maps.append(self._map_to_tensor(final_map_3ch))
                next_heats_terrain.append(torch.tensor(errors['terrain'], dtype=torch.float32, device=self.device).view(1, 1, self.map_height, self.map_width))
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
                    # [MODIFIED] Semantic + Physical Error Attribution
                    # Now we combine which obstacle types were present AND the physical failures
                    
                    # 1. Semantic (10-dim)
                    sem_err = np.zeros(10, dtype=np.float32)
                    actions_np = actions.detach().cpu().numpy()
                    
                    # [BUG FIX] Only consider actions within the `active_width` (e.g. first 6 slots).
                    # Otherwise, Grass (0) from the 18 static uneditable slots is ALWAYS included!
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
            if self.is_crafter or self.is_bipedal:
                self.prev_data = (torch.cat(next_maps), torch.cat(next_heats_terrain), torch.cat(next_heats_stats))
            else:
                self.prev_data = (torch.cat(next_maps), torch.cat(next_heats_terrain))

        avg_bfs = total_bfs_dist / max(1, bfs_count) if bfs_count > 0 else 0.0
        mean_raw_loss = np.mean(raw_scalar_losses) if len(raw_scalar_losses) > 0 else 0.0
        mean_ce_loss = np.mean(raw_ce_losses) if len(raw_ce_losses) > 0 else 0.0
        mean_inv_loss = np.mean(raw_inv_losses) if len(raw_inv_losses) > 0 else 0.0
        mean_div_reward = np.mean(div_rewards) if len(div_rewards) > 0 else 0.0

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
        if self.agent_type == 'random':
            return self._step_random(old_params, iteration=iteration)
        return self._step_policy(old_params, iteration=iteration)

    def _rollout_combined(self, map_np, stats_np, iter, idx, old_params=None, color_np=None):
        import traceback

        env_type = str(getattr(self.cfg.attention_model, "env_type", "")).lower()
        try:
             # Use the modernized support.interpret_env
             # Note: For MiniGrid, stats_np is ignored by its version of support/interpret
             if env_type == "crafter":
                 interpreted = self.support.interpret_env(map_np, inventory_vec=stats_np)
                 env_source = interpreted[0] if isinstance(interpreted, tuple) else interpreted
             elif env_type == "minigrid":
                 interpreted = self.support.interpret_env(map_np, color_array=color_np)
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
             save_path = collect_data_general(self.support.cfg, env_source=env_source, save_name=save_name, recollect_data=True)
             if not os.path.exists(save_path):
                 print(f"[GeneratorInterface] Missing rollout file after collection: {save_path}")
                 return {}, {}, 0.0, False, 0.0, 0.0, 0

             # Load trajectory first. Even if validation later fails, keep the rollout.
             task_npz = np.load(save_path, allow_pickle=True)
             traj = {
                 'obs': torch.tensor(task_npz['a'], device=self.device),
                 'obs_next': torch.tensor(task_npz['b'], device=self.device),
                 'act': torch.tensor(task_npz['c'], device=self.device),
                 'info': task_npz['f'] if 'f' in task_npz else None,
                 'inv': torch.tensor(task_npz['g'], device=self.device) if 'g' in task_npz else None,
                 'inv_next': torch.tensor(task_npz['h'], device=self.device) if 'h' in task_npz else None
             }
             solved = np.any((task_npz['e']) & (task_npz['d'] > 0))
             
             # [NEW] Compute average episode length for survival reward
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

             error_dict = {"terrain": np.zeros((self.map_height, self.map_width)), "inventory": np.zeros(16)}
             mean_loss = 0.0
             mean_aux_metric = 0.0
             mean_inv_loss = 0.0
             try:
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

             # [NEW] Compute theoretical shortest path for MiniGrid reporting
             shortest_dist = 0.0
             if env_type == "minigrid":
                 _, shortest_dist = check_solvability(map_np)

             return traj, error_dict, mean_loss, solved, mean_aux_metric, mean_inv_loss, inv_changed_slots, avg_ep_len, shortest_dist
        except Exception as e:
             print(f"[GeneratorInterface] Rollout failed for env_type={env_type}, iter={iter}, idx={idx}: {e}")
             traceback.print_exc()
             return {}, {"terrain": np.zeros((self.map_height, self.map_width)), "inventory": np.zeros(16)}, 0.0, False, 0.0, 0.0, 0, 0.0

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
                    # MiniGrid action table may encode (object_name, color_name).
                    # Preserve both object and color so generated DR maps match
                    # target color distributions.
                    if isinstance(val, tuple):
                        obj_name = val[0] if len(val) > 0 else None
                        color_name = val[1] if len(val) > 1 else None
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

                    if (not self.is_crafter) and (not self.is_bipedal) and int(val) == self.OBJ_GOAL:
                        # Goal is fixed by the base map; editor cannot relocate it.
                        continue

                    # Enforce restriction limits
                    if self.is_crafter and val in restricted_limits:
                        if counts[val] >= restricted_limits[val]:
                            continue # Ignore this action, limit reached
                        counts[val] += 1
                    
                    obj[i, j] = val
                    if color_name is not None:
                        color[i, j] = COLOR_TO_IDX.get(color_name, 0)

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
                
        return obj, color

    def _normalize_rollout_errors(self, err_dict, H=None, W=None):
        if H is None: H = self.map_height
        if W is None: W = self.map_width
        inventory = np.zeros(16, dtype=np.float32)

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
            return {"terrain": heatmap, "inventory": inventory}

        # Original Spatial logic for Minigrid/Crafter
        if isinstance(err_dict, np.ndarray):
            # Already spatial (e.g. from Crafter validation heatmap)
            return {"terrain": _normalize_terrain_array(err_dict), "inventory": inventory}
        if isinstance(err_dict, dict):
            terrain = _normalize_terrain_array(err_dict.get("terrain", np.zeros((H, W), dtype=np.float32)))
            inv = err_dict.get("inventory", inventory)
            inv = np.asarray(inv, dtype=np.float32).reshape(-1)
            if inv.size < 16:
                inv = np.pad(inv, (0, 16 - inv.size))
            elif inv.size > 16:
                inv = inv[:16]
            return {"terrain": terrain, "inventory": inv}
        return {"terrain": np.zeros((H, W), dtype=np.float32), "inventory": inventory}

    def _minigrid_path_metrics(self, grid_obj_np):
        """
        Continuous MiniGrid difficulty signals (no hard solved gate):
        - reach_ratio: reachable free cells / all free cells
        - norm_dist: shortest path to goal normalized by (H+W-2), 0 if unreachable
        """
        if self.is_crafter or self.is_bipedal:
            return 0.0, 0.0

        obj = np.asarray(grid_obj_np)
        if obj.ndim != 2:
            return 0.0, 0.0

        H, W = obj.shape
        wall_id = OBJECT_TO_IDX["wall"]
        lava_id = OBJECT_TO_IDX["lava"]
        start_positions = np.argwhere(obj == self.OBJ_START)
        if len(start_positions) == 0:
            return 0.0, 0.0
        sy, sx = map(int, start_positions[0])

        passable = (obj != wall_id) & (obj != lava_id)
        passable_count = int(passable.sum())
        if passable_count <= 0:
            return 0.0, 0.0

        dist = np.full((H, W), -1, dtype=np.int64)
        q = deque([(sy, sx)])
        dist[sy, sx] = 0

        while q:
            y, x = q.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if dist[ny, nx] >= 0:
                    continue
                if not passable[ny, nx]:
                    continue
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))

        reachable_count = int((dist >= 0).sum())
        reach_ratio = float(reachable_count / max(passable_count, 1))

        goal_positions = np.argwhere(obj == self.OBJ_GOAL)
        shortest_dist = 0.0
        if len(goal_positions) > 0:
            gy, gx = map(int, goal_positions[0])
            d = int(dist[gy, gx])
            if d > 0:
                shortest_dist = float(d)

        dist_norm_denom = float(max(H + W - 2, 1))
        norm_dist = float(np.clip(shortest_dist / dist_norm_denom, 0.0, 1.0))
        return reach_ratio, norm_dist

    def _calculate_reward(
        self,
        raw_loss,
        div_score,
        solved,
        is_warmup,
        inv_diversity=0,
        ce_loss=None,
        inv_loss=None,
        avg_ep_len=200.0,
        minigrid_reach_ratio=0.0,
        minigrid_norm_dist=0.0,
    ):
        if is_warmup:
            if self.is_crafter:
                reward_cfg = self.crafter_reward_cfg
                w_div = float(getattr(reward_cfg, "div", getattr(self.cfg.generator_agent, "reward_w_div", 3.0)))
                w_inv_change = float(getattr(reward_cfg, "inv_change", getattr(self.cfg.generator_agent, "reward_w_inv_change", 0.0)))
                inv_change_norm_slots = float(getattr(reward_cfg, "inv_change_norm_slots", getattr(self.cfg.generator_agent, "inv_change_norm_slots", 8.0)))
                bias = float(getattr(reward_cfg, "bias", getattr(self.cfg.generator_agent, "reward_bias", 1.0)))
                reward_clip = float(getattr(reward_cfg, "clip", getattr(self.cfg.generator_agent, "reward_clip", 50.0)))

                inv_changed = max(float(inv_diversity), 0.0)
                inv_norm = max(inv_change_norm_slots, 1e-6)
                inv_change_bonus = min(inv_changed / inv_norm, 1.0)

                # [ABLATION] no_diversity: zero out diversity + inv_change rewards
                if self.ablation_type == "no_diversity":
                    div_score = 0.0
                    inv_change_bonus = 0.0

                reward = (
                    w_div * float(div_score)
                    + w_inv_change * inv_change_bonus
                    + bias
                )
                return float(np.clip(reward, -reward_clip, reward_clip))

            w_survival = float(getattr(self.bipedal_reward_cfg, "survival", 0.08)) if self.is_bipedal else 0.0
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

            # [MODIFIED] Using Log-Scale for MSE to maintain reward signal at fine-grained scales.
            mse_reward_term = float(np.log10(total_loss * 50.0 + 1.0))

            # [NEW] Survival as an independent additive reward term.
            # Directly use avg episode length as the reward signal: longer = better.
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

            # [ABLATION] no_diversity: zero out diversity + inv_change rewards
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
            
        # Minigrid: continuous reward without hard solved gate.
        # Use WM difficulty + diversity + topology quality signals.
        reward_cfg = self.minigrid_reward_cfg
        w_loss = float(getattr(reward_cfg, "loss", 10.0))
        w_div = float(getattr(reward_cfg, "div", 2.0))
        w_reach = float(getattr(reward_cfg, "reach", 3.0))
        w_dist = float(getattr(reward_cfg, "dist", 1.0))
        bias = float(getattr(reward_cfg, "bias", -1.0))
        reward_clip = float(getattr(reward_cfg, "clip", 50.0))

        reward = (
            w_loss * float(raw_loss)
            + w_div * float(div_score)
            + w_reach * float(np.clip(minigrid_reach_ratio, 0.0, 1.0))
            + w_dist * float(np.clip(minigrid_norm_dist, 0.0, 1.0))
            + bias
        )
        return float(np.clip(reward, -reward_clip, reward_clip))

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
