import torch
import numpy as np
import torch.nn.functional as F
import os
import random

from generator.generator_agent import GeneratorPPO
from generator.random_generator_agent import RandomGeneratorAgent
from generator.reward_system import DiversityModule, check_solvability
from generator.env_designer import PCGSeeder, task_placer
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from modelBased.common.support import Support
from trainer.common.utils import extract_loss_map_over_validations, collect_data_general

ACTION_TABLE = {
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
        
        if agent_type == 'random':
            print("[Generator] Using Random Agent")
            self.ppo = RandomGeneratorAgent(num_actions=len(ACTION_TABLE), device=device)
        else:
            self.ppo = GeneratorPPO(
                context_dim=hparams.context_dim,
                num_actions=len(ACTION_TABLE),
                his_emb_dim=hparams.his_emb_dim,
                top_k_features=hparams.ctx_top_k_features,
                ablation_type=self.cfg.ablation.type 
            )
        
        self.map_height = hparams.map_height
        self.map_width = hparams.map_width
        self.div_k = hparams.div_k
        self.diversity = DiversityModule(self.map_height, self.map_width, self.div_k)
        self.max_edits = hparams.max_edits
        self.seeder = PCGSeeder(height=self.map_height, width=self.map_width)
        
        self.OBJ_START = OBJECT_TO_IDX["agent"]
        self.OBJ_GOAL = OBJECT_TO_IDX["goal"]
        self.OBJ_EMPTY = OBJECT_TO_IDX["empty"]

        self.elite_buffer = [] 
        self.max_elites = max(1, self.batch_size // 2)
        self.prev_data = None

    def sync_world_model(self, state_dict):
        self.wm.load_state_dict(state_dict)

    def step(self, iteration=0):
        # 1. Prepare Base Maps & Aligned Context (Adversarial Inheritance)
        base_maps = []
        context_maps = []
        context_heats = []
        
        warmup_iters = self.cfg.generator_agent.get("warmup_iterations", 0)
        is_warmup = (iteration < warmup_iters)

        num_elites = 0
        if (
            self.use_elites
            and (not is_warmup)
            and len(self.elite_buffer) > 0
        ):
            num_elites = min(len(self.elite_buffer), self.max_elites)

        
        num_random = self.batch_size - num_elites

        # A) Load Elites
        for i in range(num_elites):
            obj_map_np, h_tensor, _ = self.elite_buffer[i]
            base_maps.append(obj_map_np)
            
            zm, zh = self._zero_context(1, self.map_height, self.map_width)
            m_3ch = zm.clone()
            m_3ch[0, 0] = torch.tensor(obj_map_np, device=self.device)
            context_maps.append(m_3ch)
            context_heats.append(h_tensor.to(self.device))
        
        # B) Load PCG
        for _ in range(num_random):
            z = np.random.randint(0, 1e6)
            
            # [RESTORE DIVERSITY] Randomly choose between pure_random and blob
            struct_mode = random.choice(["pure_random", "blob"])
            self.seeder.structure_mode = struct_mode
            
            grid = self.seeder.generate(z=z)
            min_dim = min(self.map_height, self.map_width)
            adaptive_ratio = 0.85 if min_dim <= 10 else 0.4
            grid, _ = task_placer(grid, min_dist_ratio=adaptive_ratio)
            
            base_maps.append(grid)
            zm, zh = self._zero_context(1, self.map_height, self.map_width)
            context_maps.append(zm)
            context_heats.append(zh)

        # 2. Finalize Input Tensors
        base_ids = torch.from_numpy(np.stack(base_maps)).to(self.device).long()
        B, H, W = base_ids.shape
        zeros = torch.zeros((B, H, W), device=self.device, dtype=torch.long)
        curr_map = torch.stack([base_ids, zeros, zeros], dim=1).float() 
        
        mask = self._immutable_mask(base_ids)
        ppo_input_context = (torch.cat(context_maps), torch.cat(context_heats))

        if hasattr(self.cfg, 'ablation') and self.cfg.ablation.type == 'no_history':
            ppo_input_context = self._zero_context(B, H, W)

        # 3. select actions
        actions, logp, values, topk_action_mask, _ = self.ppo.select_action(
            curr_map, ppo_input_context, mask, max_edits=self.max_edits
        )

        # 4. Rollout & Reward Collection
        next_maps, next_heats = [], []
        valid_trajs, raw_scalar_losses, div_rewards, bfs_stats = [], [], [], []
        clutter_counts = []

        base_ids_np = base_ids.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        solvable_count = 0

        for i in range(self.batch_size):
            obj_map, color_map = self._apply_action(base_ids_np[i], actions_np[i])
            final_map = np.stack([obj_map, color_map], axis=0)
            
            is_solvable, bfs_dist = check_solvability(obj_map)
            if is_solvable:
                solvable_count += 1
                bfs_stats.append(bfs_dist) 
            else:
                bfs_stats.append(0)
            
            traj, heat, raw_loss_val, solved = self._rollout_env(final_map, iteration=iteration, batch_idx=i)
            
            # [MOVED] Calculate diversity BEFORE checking validity, so we log it even for failed envs
            r_div = self.diversity.get_reward(torch.tensor(final_map).unsqueeze(0).to(self.device))
            r_div_val = r_div.item() if hasattr(r_div, 'item') else r_div
            div_rewards.append(r_div_val)
            
            # [ABLATION]
            r_div_weight_factor = 1.0
            if hasattr(self.cfg, 'ablation') and self.cfg.ablation.type == 'no_diversity':
                r_div_weight_factor = 0.0
            
            if not traj or 'obs' not in traj:
                fail_reward = -5.0 if not is_solvable else -1.0
                fail_reward = -5.0 if not is_solvable else -1.0
                self.ppo.save_buffer(
                    curr_map[i:i+1],
                    (ppo_input_context[0][i:i+1], ppo_input_context[1][i:i+1]),
                    mask[i:i+1],
                    actions[i:i+1],
                    logp[i:i+1],
                    values[i:i+1],
                    fail_reward,
                    topk_action_mask[i:i+1]
                )
                continue

            valid_trajs.append(traj)
            next_maps.append(self._map_to_tensor(final_map))
            next_heats.append(heat)
            


            # [DEBUG]
            if iteration % 1 == 0:
                 print(f"[DEBUG] Iter {iteration} Batch {i}: ArchiveSize={len(self.diversity.archive)}, r_div_val={r_div_val:.6f}")

            scalar_loss = raw_loss_val
            if scalar_loss <= 0.5:
                reward_loss = np.exp(scalar_loss * 10.0) * 10.0 
            else:
                reward_loss = 1480.0 + (scalar_loss - 0.5) * 1000.0
            
            # [CRITICAL] Cap the reward. We don't want 20,000 pts 
            # overwhelming our density penalty.
            reward_loss = min(1000.0, reward_loss)
            
            if hasattr(self.cfg, 'ablation') and self.cfg.ablation.type == 'no_learning_progress':
                reward_loss = 0.0
            
            r_len = bfs_dist * 0.5 
            
            # [FIX] Compute object density
            clutter_mask = np.isin(final_map[0], [4, 5, 9])
            clutter_count = np.sum(clutter_mask)
            clutter_counts.append(clutter_count)

            # [REBALANCED] Link clutter threshold to the actual max_edits in config.
            edit_ratio = getattr(self.cfg.generator_agent, "max_edits", 0.06)
            clutter_threshold = int(self.map_height * self.map_width * edit_ratio)
            
            # [REBALANCED] Penalty: -50.0 per excess item.
            r_density = -50.0 * max(0, clutter_count - clutter_threshold)

            if not is_solvable:
                reward = -5.0 + r_density
            elif solved:
                reward = reward_loss + 5.0 * r_div_val * r_div_weight_factor + r_len + 0.01 + r_density
            else:
                reward = (reward_loss * 0.8) + 2.5 * r_div_val * r_div_weight_factor + r_len + 0.005 + r_density

            # [FIX] Apply density penalty to warmup too! 
            # Otherwise generator learns "flooding" as a good strategy for diversity/BFS.
            if iteration < warmup_iters:
                reward = 1.0 + (bfs_dist * 0.2) + (r_div_val * 5.0 * r_div_weight_factor) + r_density

            raw_scalar_losses.append(raw_loss_val) 
            self.ppo.save_buffer(
                curr_map[i:i+1],
                (ppo_input_context[0][i:i+1], ppo_input_context[1][i:i+1]),
                mask[i:i+1],
                actions[i:i+1],
                logp[i:i+1],
                values[i:i+1],
                reward,
                topk_action_mask[i:i+1]
            )

        if len(next_maps) > 0:
            self.prev_data = (torch.cat(next_maps), torch.cat(next_heats))
            if not is_warmup:
                elite_candidates = []
                for i in range(len(next_maps)):
                    map_np = next_maps[i][0, 0].cpu().numpy()
                    heat_i = next_heats[i].detach().cpu()
                    loss_i = raw_scalar_losses[i]
                    clutter_i = clutter_counts[i]   

                    edit_ratio = getattr(self.cfg.generator_agent, "max_edits", 0.06)
                    threshold = int(self.map_height * self.map_width * edit_ratio)

                    if clutter_i <= threshold:
                        elite_candidates.append(
                            (map_np, heat_i, loss_i)
                        )
                    else:
                        # If the champion is too cluttered, it's not a champion, it's a bug.
                        pass

                elite_candidates.sort(key=lambda x: x[2], reverse=True)
                self.elite_buffer = elite_candidates[:self.max_elites]
                if len(self.elite_buffer) > 0:
                    print(f"[Generator] Evolution Memory Updated ({num_elites} inherited, Top Sparse Loss: {self.elite_buffer[0][2]:.4f})")

        mean_raw_loss = np.mean(raw_scalar_losses) if raw_scalar_losses else 0.0
        mean_div_score = np.mean(div_rewards) * 5.0 if div_rewards else 0.0
        avg_bfs_dist = np.mean(bfs_stats) if bfs_stats else 0.0

        return (
            None, # next_maps placeholder
            None, # next_heats placeholder
            mean_raw_loss,
            mean_div_score,
            valid_trajs,
            solvable_count,
            avg_bfs_dist
        )

    def update(self):
        loss, entropy = self.ppo.update()
        return loss, entropy, self.ppo.last_mean_reward

    def _immutable_mask(self, ids):
        # 0.0 means mutable, 1.0 means immutable
        # [FIX] Allow editing EVERYTHING except Start/Goal/Outer Walls.
        # This gives the agent the power to "Delete" or "Change" existing objects in Elites.
        mask = torch.zeros_like(ids, dtype=torch.float32)
        
        # 1. Start and Goal are immutable
        mask[ids == self.OBJ_START] = 1.0
        mask[ids == self.OBJ_GOAL] = 1.0
        
        # 2. Outer walls (boundary) are immutable 
        # Detect boundary positions [B, H, W]
        H, W = ids.shape[-2:]
        mask[:, 0, :] = 1.0
        mask[:, -1, :] = 1.0
        mask[:, :, 0] = 1.0
        mask[:, :, -1] = 1.0
        
        # 3. Existing Walls inside remain immutable to keep connectivity stable?
        # Actually, let's allow editing walls inside too, to allow "Digging".
        # But we'll keep the boundary strictly closed.

        return mask.unsqueeze(1)

    def _apply_action(self, base_obj_map, act):
        H, W = base_obj_map.shape
        obj = base_obj_map.copy()
        
        MAX_OBJ_ID = max(OBJECT_TO_IDX.values())
        default_color_map = np.zeros(MAX_OBJ_ID + 1, dtype=np.int64)
        default_color_map[OBJECT_TO_IDX["wall"]] = COLOR_TO_IDX["grey"]
        default_color_map[OBJECT_TO_IDX["door"]] = COLOR_TO_IDX["yellow"]
        default_color_map[OBJECT_TO_IDX["key"]] = COLOR_TO_IDX["yellow"]
        default_color_map[OBJECT_TO_IDX["ball"]] = COLOR_TO_IDX["red"]
        default_color_map[OBJECT_TO_IDX["box"]] = COLOR_TO_IDX["yellow"]
        default_color_map[OBJECT_TO_IDX["goal"]] = COLOR_TO_IDX["green"]
        default_color_map[OBJECT_TO_IDX["lava"]] = COLOR_TO_IDX["red"]
        
        color = default_color_map[np.clip(obj, 0, MAX_OBJ_ID)]
        immutable = (obj == self.OBJ_START) | (obj == self.OBJ_GOAL)

        for i in range(H):
            for j in range(W):
                if immutable[i, j]: continue
                a = act[i, j]
                if a == 0: continue
                obj_type, color_name = ACTION_TABLE[a]
                if obj_type == "key":
                    obj[i, j] = OBJECT_TO_IDX["key"]; color[i, j] = COLOR_TO_IDX[color_name]
                elif obj_type == "door":
                    obj[i, j] = OBJECT_TO_IDX["door"]; color[i, j] = COLOR_TO_IDX[color_name]
                elif obj_type == "lava":
                    obj[i, j] = OBJECT_TO_IDX["lava"]
                elif obj_type == "empty":
                    obj[i, j] = OBJECT_TO_IDX["empty"]; color[i, j] = 0
        return obj, color

    def _zero_context(self, B, H, W):
        return (torch.zeros((B, 3, H, W), device=self.device), torch.zeros((B, 1, H, W), device=self.device))

    def _map_to_tensor(self, m):
        t = torch.tensor(m, device=self.device).float()
        state_channel = torch.zeros_like(t[0:1])
        return torch.cat([t, state_channel], dim=0).unsqueeze(0)

    def _rollout_env(self, map_obj, iteration=0, batch_idx=0):
        obj_map, color_map = map_obj
        map_tensor = torch.tensor(obj_map, dtype=torch.long, device=self.device)
        color_tensor = torch.tensor(color_map, dtype=torch.long, device=self.device)
        try:
            obj_str, color_str = self.support.interpret_env(map_tensor.cpu(), color_array=color_tensor.cpu())
            env_str = (obj_str, color_str)
        except: return {}, None, 0.0, False

        old_episodes = self.support.cfg.env.collect.episodes
        try:
            save_name = f'UED_temp_iter{iteration}_b{batch_idx}'
            self.support.cfg.env.collect.episodes = 1 
            save_path = collect_data_general(self.support.cfg, env_source=env_str, save_name=save_name, max_steps=1000, recollect_data=True)
            
            if os.path.exists(save_path):
                task_npz = np.load(save_path, allow_pickle=True)
                traj_data = {'obs': torch.tensor(task_npz['a'], device=self.device), 'obs_next': torch.tensor(task_npz['b'], device=self.device),
                             'act': torch.tensor(task_npz['c'], device=self.device), 'info': task_npz['f'] if 'f' in task_npz else None}
                solved = np.any((task_npz['e']) & (task_npz['d'] > 0))
            else: return {}, None, 0.0, False

            avg_loss_map, loss_list = extract_loss_map_over_validations(self.cfg, net=self.wm, old_params=None, data_dir=save_path, valid_times=1)
            scalar_loss = np.mean(loss_list) if loss_list else 0.0
            scaled_h = np.log(avg_loss_map + 1e-8)
            heat = torch.tensor(scaled_h, device=self.device).unsqueeze(0).unsqueeze(0)
        except:
            traj_data, heat, scalar_loss, solved = {}, torch.zeros((1, 1, self.map_height, self.map_width), device=self.device), 0.0, False
        finally:
            self.support.cfg.env.collect.episodes = old_episodes
        return traj_data, heat, scalar_loss, solved
