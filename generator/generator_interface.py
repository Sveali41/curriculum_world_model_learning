import torch
import numpy as np
import torch.nn.functional as F
import os
import random

from generator.generator_agent import GeneratorPPO
from generator.random_generator_agent import RandomGeneratorAgent
from generator.reward_system import DiversityModule, check_solvability
from generator.minigrid_env_designer import PCGSeeder as MinigridPCGSeeder, task_placer as minigrid_task_placer
from generator.crafter_env_designer import CrafterPCGSeeder, CRAFTER_ACTION_MAP, CRAFTER_OBJ_MAP
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from modelBased.common.support import Support
from trainer.common.utils import extract_loss_map_over_validations, collect_data_general

ACTION_TABLE_MINIGRID = {
    0: None,  # No-op
    1: ("key", "yellow"),
    2: ("key", "red"),
    3: ("key", "blue"),
    4: ("door", "yellow"),
    5: ("door", "red"),
    6: ("door", "blue"),
    7: ("lava", None),
    8: ("empty", None),
}

class GeneratorInterface:
    def __init__(self, world_model, device, cfg, agent_type='ppo'):
        self.device = device
        self.cfg = cfg
        hparams = cfg.generator_agent
        self.batch_size = hparams.batch_size
        self.support = Support(cfg)
        self.wm = world_model
        self.use_elites = getattr(hparams, "use_elites", True)
        
        # Check if environment is Crafter
        self.is_crafter = (getattr(cfg.attention_model, "env_type", "") == "crafter")

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
                10: CRAFTER_OBJ_MAP['plant']
            }
        else:
            # MiniGrid: use standard seeder
            self.map_height = hparams.map_height
            self.map_width = hparams.map_width
            self.seeder = MinigridPCGSeeder(height=self.map_height, width=self.map_width)
            self.ACTION_TABLE = ACTION_TABLE_MINIGRID

        if agent_type == 'random':
            self.ppo = RandomGeneratorAgent(num_actions=len(self.ACTION_TABLE), device=device)
        else:
            self.ppo = GeneratorPPO(
                context_dim=hparams.context_dim,
                num_actions=len(self.ACTION_TABLE),
                his_emb_dim=hparams.his_emb_dim,
                top_k_features=hparams.ctx_top_k_features,
                ablation_type=self.cfg.ablation.type 
            )
        
        self.div_k = hparams.div_k
        self.diversity = DiversityModule(
            self.map_height, self.map_width, self.div_k, 
            env_type=('crafter' if self.is_crafter else 'minigrid')
        )
        self.max_edits_layout = hparams.max_edits_layout
        self.max_edits_inventory = hparams.max_edits_inventory
        
        self.OBJ_START = OBJECT_TO_IDX["agent"] if not self.is_crafter else CRAFTER_OBJ_MAP['agent']
        self.OBJ_GOAL = OBJECT_TO_IDX["goal"] if not self.is_crafter else 999 # No goal in Crafter
        self.OBJ_EMPTY = OBJECT_TO_IDX["empty"] if not self.is_crafter else CRAFTER_OBJ_MAP['grass']

        self.elite_buffer = [] 
        self.max_elites = max(1, self.batch_size // 2)
        self.prev_data = None

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

    def _zero_context(self, B, H, W):
        # (Map features, Inventory Heatmap)
        # Spatial heat map: [B, 1, H, W]
        # Stats heat map: [B, 16] - We now handle this separately for Crafter
        map_h = torch.zeros((B, 1, H, W), device=self.device)
        if self.is_crafter:
            stats_h = torch.zeros((B, 16), device=self.device)
        else:
            stats_h = None
        return (torch.zeros((B, 3, H, W), device=self.device), map_h, stats_h)

    def step(self, old_params, iteration=0):
        # 1. Prepare Base Maps & Stats & Context
        base_maps = []
        base_stats = []
        context_maps, context_heats_terrain, context_heats_stats = [], [], []
        
        warmup_iters = self.cfg.generator_agent.get("warmup_iterations", 0)
        is_warmup = (iteration < warmup_iters)

        num_elites = 0
        if self.use_elites and (not is_warmup) and len(self.elite_buffer) > 0:
            num_elites = min(len(self.elite_buffer), self.max_elites)
        
        num_random = self.batch_size - num_elites

        # A) Load Elites (Legacy supporting map only for now)
        for i in range(num_elites):
            obj_map_np, h_dict, stats_np = self.elite_buffer[i]
            base_maps.append(obj_map_np)
            base_stats.append(stats_np)
            
            zm, zh, zhs = self._zero_context(1, self.map_height, self.map_width)
            m_3ch = zm.clone()
            m_3ch[0, 0] = torch.tensor(obj_map_np, device=self.device)
            context_maps.append(m_3ch)
            context_heats_terrain.append(torch.tensor(h_dict['terrain']).to(self.device))
            context_heats_stats.append(torch.tensor(h_dict['inventory']).to(self.device).reshape(1, 16))
        
        # B) Load PCG
        for _ in range(num_random):
            grid = self.seeder.generate()
            base_maps.append(grid)
            default_stats = np.zeros(16)
            default_stats[0] = 9.0 # health
            default_stats[1] = 9.0 # food
            default_stats[2] = 9.0 # drink
            default_stats[3] = 9.0 # energy
            base_stats.append(default_stats)
            
            zm, zh, zhs = self._zero_context(1, self.map_height, self.map_width)
            context_maps.append(zm)
            context_heats_terrain.append(zh)
            context_heats_stats.append(torch.zeros((1, 16), device=self.device))

        # 2. Finalize Input Tensors
        base_ids = torch.from_numpy(np.stack(base_maps)).to(self.device).long()
        B, H, W = base_ids.shape
        zeros = torch.zeros((B, H, W), device=self.device, dtype=torch.float32)
        
        # Context packing
        # For Crafter, we pass Map and Stats heat independently
        ppo_input_context = (
            torch.cat(context_maps), 
            torch.cat(context_heats_terrain), 
            torch.cat(context_heats_stats) if self.is_crafter else None
        )

        # Inject Terrain Error Heatmap DIRECTLY into the spatial CNN channels!
        # Channel 0: Structural Layout
        # Channel 1: Terrain Error Heatmap (Spatially aligns with features!)
        # Channel 2: Blank (Padding to 3 channels)
        heat_terrain_spatial = ppo_input_context[1].squeeze(1).float() # [B, H, W]
        curr_map = torch.stack([base_ids.float(), heat_terrain_spatial, zeros], dim=1) 
        
        mask = self._immutable_mask(base_ids)


        # 3. select dual actions
        actions, stats_actions, logp, values, topk_action_mask, _ = self.ppo.select_action(
            curr_map, ppo_input_context, mask, self.max_edits_layout, self.max_edits_inventory
        )

        # 2. Sequential Evaluation for Rewards
        valid_trajs = []
        next_maps, next_heats_terrain, next_heats_stats = [], [], []
        raw_scalar_losses, div_rewards = [], []
        raw_ce_losses, raw_inv_losses = [], []
        solved_count = 0
        total_bfs_dist = 0.0
        
        # np versions for easy access
        base_ids_np = base_ids.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        stats_actions_np = stats_actions.detach().cpu().numpy() if stats_actions is not None else None
        base_stats_np = np.stack(base_stats)

        for i in range(self.batch_size):
            # Apply Terrain Actions
            final_map_obj, _ = self._apply_action(base_ids_np[i], actions_np[i], mask=mask[i, 0].cpu().numpy())
            # Apply Stats Actions (32 Piano Keys: 16 Slots * 2 Action Rows)
            # Row 0 (0-15): Increment by 1
            # Row 1 (16-31): Increment by 5
            final_stats = base_stats_np[i].copy()
            
            # --- [NEW] Check for Randomized Stats Actions ---
            # If enabled, replace PPO's output with a random bitmask (e.g. 15% probability of a button being pressed)
            if self.is_crafter and getattr(self.cfg.generator_agent, "random_stats_actions", False):
                # Sample 32 bits from Bernoulli(p=0.15)
                current_stats_act = (np.random.rand(32) < 0.15).astype(np.int64)
            else:
                current_stats_act = stats_actions_np[i] if stats_actions_np is not None else None

            if self.is_crafter and current_stats_act is not None:
                # current_stats_act is [32] bits
                for k_idx in range(32):
                    if current_stats_act[k_idx] == 1:
                        slot = k_idx % 16
                        if k_idx < 16:
                             final_stats[slot] += 1
                        else:
                             final_stats[slot] += 5
            
            # Prepare for rollout
            final_map_2ch = np.stack([final_map_obj, np.zeros_like(final_map_obj)], axis=0)
            res_rollout = self._rollout_combined(final_map_obj, final_stats, iteration, i, old_params=old_params)
            traj, errors, raw_loss_val, solved = res_rollout[0], res_rollout[1], res_rollout[2], res_rollout[3]
            
            # Extract sub-losses if available (for 6-column CSV)
            t_loss_batch = res_rollout[4] if len(res_rollout) > 4 else raw_loss_val
            i_loss_batch = res_rollout[5] if len(res_rollout) > 5 else 0.0
            # [NEW] Extract Inventory Transition Diversity
            inv_changed_slots = res_rollout[6] if len(res_rollout) > 6 else 0

            # --- Crafter Specific Solvability Logic ---
            # Basically, check if player is not blocked by water walls. Use the seeder's BFS.
            if self.is_crafter:
                 is_connected, conn_stats = self.seeder.check_connectivity(final_map_obj)
                 if is_connected: 
                     solved_count += 1
                     total_bfs_dist += conn_stats.get('max_dist', 0)
            else:
                 if solved: 
                     solved_count += 1
                     total_bfs_dist += 1.0 # Minigrid placeholder
            
            # Reward Logic
            # For Crafter, 'solved' is redefined as 'is_connected' (BFS reachability)
            is_connected_final = is_connected if self.is_crafter else solved
            r_div = self.diversity.get_reward(torch.tensor(final_map_2ch).unsqueeze(0).to(self.device), inventory_vec=final_stats)
            div_rewards.append(r_div)
            reward = self._calculate_reward(raw_loss_val, r_div, is_connected_final, is_warmup, inv_diversity=inv_changed_slots)
            
            if not traj or 'obs' not in traj:
                reward = -5.0 # Basic failure penalty
            
            # Provide the correct error pattern context slices instead of None!
            # Otherwise PPO update computes gradients on ZERO context!
            cm = ppo_input_context[0][i:i+1]
            cht = ppo_input_context[1][i:i+1]
            chs = ppo_input_context[2][i:i+1] if self.is_crafter else torch.zeros((1, 16), device=self.device)
            prev_data_i = (cm, cht, chs)
            
            raw_scalar_losses.append(raw_loss_val)
            raw_ce_losses.append(t_loss_batch)
            raw_inv_losses.append(i_loss_batch)
            self.ppo.save_buffer(
                curr_map[i:i+1],
                prev_data_i,
                mask[i:i+1],
                actions[i:i+1],
                stats_actions[i:i+1] if self.is_crafter else torch.zeros((1, 16), device=self.device),
                logp[i:i+1],
                values[i:i+1],
                reward,
                topk_action_mask[i:i+1]
            )
            
            if traj and 'obs' in traj:
                valid_trajs.append(traj)
                next_maps.append(self._map_to_tensor(final_map_2ch))
                next_heats_terrain.append(torch.tensor(errors['terrain']).unsqueeze(0).unsqueeze(0))
                next_heats_stats.append(torch.tensor(errors['inventory']).unsqueeze(0))

        # Update Memory
        if len(next_maps) > 0:
            self.prev_data = (torch.cat(next_maps), torch.cat(next_heats_terrain), torch.cat(next_heats_stats))

        # Return real count; avg_bfs is derived from BFS max depth
        avg_bfs = total_bfs_dist / max(1, solved_count) if solved_count > 0 else 0.0
        
        # Clean up NaNs
        mean_raw_loss = np.mean(raw_scalar_losses) if len(raw_scalar_losses) > 0 else 0.0
        mean_ce_loss = np.mean(raw_ce_losses) if len(raw_ce_losses) > 0 else 0.0
        mean_inv_loss = np.mean(raw_inv_losses) if len(raw_inv_losses) > 0 else 0.0
        mean_div_reward = np.mean(div_rewards) if len(div_rewards) > 0 else 0.0

        return None, None, mean_raw_loss, mean_ce_loss, mean_inv_loss, mean_div_reward, valid_trajs, solved_count, avg_bfs

    def _rollout_combined(self, map_np, stats_np, iter, idx, old_params=None):
        try:
             # Use the modernized support.interpret_env
             # Note: For MiniGrid, stats_np is ignored by its version of support/interpret
             env_source, _ = self.support.interpret_env(map_np, self.cfg, inventory_vec=stats_np)
             save_name = f'UED_Dual_iter{iter}_b{idx}'
             save_path = collect_data_general(self.support.cfg, env_source=env_source, save_name=save_name, recollect_data=True)
             
             if not os.path.exists(save_path): return {}, {}, 0.0, False
             
             # Extract Dual-Head Error Signal
             v_times = getattr(self.cfg.attention_model, "valid_times", 1)
             res_eval = extract_loss_map_over_validations(self.cfg, self.wm, old_params, save_path, valid_times=v_times)
             error_dict, loss_list = res_eval[0], res_eval[1]
             terrain_losses, inv_losses = res_eval[2], res_eval[3]

             # Load trajectory (a=obs, b=obs_next, c=act, f=info)
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
             
             # [NEW] Compute Inventory Transition Diversity
             # Count how many distinct inventory slots changed at least once during the rollout
             inv_changed_slots = 0
             if 'g' in task_npz and 'h' in task_npz:
                 inv_arr      = task_npz['g'].astype(np.float32)  # [T, 16]
                 inv_next_arr = task_npz['h'].astype(np.float32)  # [T, 16]
                 delta = inv_next_arr - inv_arr
                 inv_changed_slots = int(np.any(delta != 0, axis=0).sum())  # 0-16
             
             return traj, error_dict, np.mean(loss_list), solved, np.mean(terrain_losses), np.mean(inv_losses), inv_changed_slots
        except Exception as e:
             print(f"[GeneratorInterface] Rollout failed: {e}")
             return {}, {"terrain": np.zeros((self.map_height, self.map_width)), "inventory": np.zeros(16)}, 0.0, False, 0.0, 0.0, 0

    def _apply_action(self, base_map, act, mask=None):
        # Local terrain modification logic (Spatial Head)
        obj = base_map.copy()
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

        for i in range(H):
            for j in range(W):
                # Skip if immutable (mask is 1.0 for boundaries)
                if mask is not None and mask[i, j] > 0:
                    continue
                a = act[i, j]
                if a == 0: continue
                val = self.ACTION_TABLE.get(a)
                
                if val is not None: 
                    # Enforce restriction limits
                    if self.is_crafter and val in restricted_limits:
                        if counts[val] >= restricted_limits[val]:
                            continue # Ignore this action, limit reached
                        counts[val] += 1
                    
                    obj[i, j] = val
        
        # Mandatory Agent Placement Fallback (Ensures environment can always start)
        if not np.any(obj == self.OBJ_START):
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
                
        return obj, np.zeros_like(obj)

    def _calculate_reward(self, raw_loss, div_score, solved, is_warmup, inv_diversity=0):
        if is_warmup: return 1.0 + div_score * 5.0
        
        # Crafter: No BFS, Pure adversarial reward loop. 
        # Any environment is a valid challenge for the World Model.
        if self.is_crafter:
            # [MODIFIED] Only provide bonus if stats actions are NOT randomized
            # This allows PPO to focus 100% on challenging terrain (raw_loss) during randomized experiments.
            
            print(f"[Reward] raw_loss={raw_loss:.3f} | div={div_score:.3f} | inv_slots_changed={inv_diversity}")
            return raw_loss * 5.0 + 3.0 * div_score + 2.0
            
        # Minigrid: Strict solvability required (Walls are immutable obstacles).
        # Regret-based reward for curricular difficulty
        # If not connected/solved, penalize heavily to avoid exploiting "stuck" states
        if not solved: return -5.0 + 2.0 * div_score
        
        # PPO requires smooth advantage curves.
        reward_loss = raw_loss * 10.0
        return reward_loss + 2.0 * div_score + 10.0 # Solved bonus is implicit

    def _immutable_mask(self, ids):
        mask = torch.zeros_like(ids, dtype=torch.float32)
        # 1. Protect Boundary Water
        mask[:, 0, :] = 1.0; mask[:, -1, :] = 1.0; mask[:, :, 0] = 1.0; mask[:, :, -1] = 1.0
        # 2. Protect Agent's Position (Don't erase/move the player once placed)
        mask[ids == self.OBJ_START] = 1.0
        return mask.unsqueeze(1)

    def _map_to_tensor(self, m):
        return torch.tensor(m, device=self.device).float().unsqueeze(0)

    def update(self):
        loss, ent = self.ppo.update()
        return loss, ent, self.ppo.last_mean_reward

