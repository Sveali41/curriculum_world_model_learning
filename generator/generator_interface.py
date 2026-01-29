import torch
import numpy as np
import torch.nn.functional as F

from generator.generator_agent import GeneratorPPO
from generator.random_generator_agent import RandomGeneratorAgent # [NEW]
from generator.reward_system import DiversityModule, check_solvability, calculate_lp_reward
from generator.env_designer import PCGSeeder, task_placer
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

from modelBased.common.support import Support
import copy
from modelBased.common.utils import TRAINER_PATH
from trainer.common.utils import extract_loss_map_over_validations, collect_data_general
import os




# Generator action vocabulary (8 actions)

ACTION_TABLE = {
    0: None,  # No-op
    1: ("key", "red"),
    2: ("key", "green"),
    3: ("key", "blue"),
    4: ("door", "red"),    # open door
    5: ("door", "green"),
    6: ("door", "blue"),
    7: ("lava", None),
}



class GeneratorInterface:
    def __init__(self, world_model, device, cfg, agent_type='ppo'):
        self.device = device
        self.cfg = cfg
        hparams = cfg.generator_agent
        self.batch_size = hparams.batch_size
        self.support = Support(cfg)
        self.wm = world_model
        
        # [MODIFIED] Switch between PPO and Random Agent
        if agent_type == 'random':
             print("[Generator] Using Random Agent (Domain Randomization Baseline)")
             self.ppo = RandomGeneratorAgent(num_actions=len(ACTION_TABLE), device=device)
        else:
             self.ppo = GeneratorPPO(context_dim=hparams.context_dim,   
                num_actions=len(ACTION_TABLE),
                his_emb_dim=hparams.his_emb_dim,
                # ratio=hparams.ratio, # removed
                top_k_features=hparams.ctx_top_k_features,
            )
        self.map_height = hparams.map_height
        self.map_width = hparams.map_width
        self.div_k = hparams.div_k
        self.diversity = DiversityModule(self.map_height, self.map_width, self.div_k)
        self.max_edits = hparams.max_edits
        self.seeder = PCGSeeder(height=self.map_height, width=self.map_width)
        self.prev_data = None
        self.OBJ_START = OBJECT_TO_IDX["agent"]
        self.OBJ_GOAL = OBJECT_TO_IDX["goal"]
        self.OBJ_EMPTY = OBJECT_TO_IDX["empty"]

    # ------------------------------------------------------------
    def sync_world_model(self, state_dict):
        self.wm.load_state_dict(state_dict)

    # ------------------------------------------------------------
    def step(self, iteration=0):
            base_maps = []

            # 1. 生成基础地图 (PCG)
            for _ in range(self.batch_size):
                z = np.random.randint(0, 1e6)
                grid = self.seeder.generate(z=z)
                
                # [MODIFIED] Adaptive task placement based on map size
                # Small maps (<10) need strict long-distance (0.7) to avoid trivial paths.
                # Large maps can be more relaxed (0.4).
                min_dim = min(self.map_height, self.map_width)
                adaptive_ratio = 0.85 if min_dim <= 10 else 0.4
                
                grid, _ = task_placer(grid, min_dist_ratio=adaptive_ratio)
                base_maps.append(grid)

            # 优化：先转 numpy 再转 tensor，避免 UserWarning
            # 优化：先转 numpy 再转 tensor
            base_ids = torch.from_numpy(np.stack(base_maps)).to(self.device).long()
            
            # 手动构建 [B, 3, H, W] 的输入张量
            # Channel 0: Object IDs (真实数据)
            # Channel 1: Color IDs (默认为 0)
            # Channel 2: State IDs (默认为 0)
            B, H, W = base_ids.shape
            zeros = torch.zeros((B, H, W), device=self.device, dtype=torch.long)
            
            curr_map = torch.stack([base_ids, zeros, zeros], dim=1).float() 
            # shape: (batch_size, 3, map_height, map_width)

            mask = self._immutable_mask(base_ids)

            if self.prev_data is None:
                self.prev_data = self._zero_context(curr_map.size(0), self.map_height, self.map_width)

            # [ABLATION] w/o History
            ppo_input_context = self.prev_data
            if hasattr(self.cfg, 'ablation') and self.cfg.ablation.type == 'no_history':
                 ppo_input_context = self._zero_context(curr_map.size(0), self.map_height, self.map_width)
                 # print("[Ablation] History/Context disabled.")

            # 2. 生成器选择动作 (关键：获取采样时的 topk_action_mask)
            # 假设 select_action 返回: actions, logp, values, topk_action_mask, global_ctx
            actions, logp, values, topk_action_mask, _ = self.ppo.select_action(
                curr_map, ppo_input_context, mask, max_edits=self.max_edits
            )

            next_maps, next_heats = [], []
            valid_trajs = []
            raw_scalar_losses = []
            raw_scalar_losses = []
            div_rewards = []
            bfs_stats = [] # [NEW] Track path lengths

            base_ids_np = base_ids.detach().cpu().numpy()
            actions_np = actions.detach().cpu().numpy()

            # [NEW] Track solvability rate
            solvable_count = 0

            for i in range(self.batch_size):
                # 3. 应用编辑动作
                obj_map, color_map = self._apply_action(
                    base_ids_np[i],
                    actions_np[i],
                )
                final_map = np.stack([obj_map, color_map], axis=0)

                # --- 第一层过滤：物理连通性 (Hard Filter) ---
                is_solvable, bfs_dist = check_solvability(obj_map)
                if is_solvable:
                    solvable_count += 1
                    bfs_stats.append(bfs_dist) # [NEW] Only track valid path lengths
                else:
                    bfs_stats.append(0)
                
                # [REVERTED] User Request: Record Unsolvable Data for Physics Learning.
                # We still penalize the Generator (-5.0) later in the reward section.
                # if not is_solvable:
                #     print(f"[Warning] Trajectory {i} is unsolvable (disconnected). Applying Penalty but collecting data.")
                
                # 4. 执行 Rollout 采集数据 (包含智能体在该图上的预测误差 heat 和是否解决 solved)
                traj, heat, scalar_loss, solved = self._rollout_env(final_map, iteration=iteration, batch_idx=i)
                
                # 检查 traj 是否有效
                if not traj or 'obs' not in traj:
                    print(f"[Warning] Trajectory {i} is empty, skipping.")
                    # [MODIFIED] 如果是物理不可解导致的失败，直接给重罚 -5.0，而不是默认的 -1.0
                    
                    # [ABLATION] w/o Validity Reward (Penalty)
                    if hasattr(self.cfg, 'ablation') and self.cfg.ablation.type == 'no_validity_reward':
                        fail_reward = 0.0 # Standard failure penalty (like in DR)
                        # No print needed
                    else:
                        # Standard UED: Strong Penalty
                        fail_reward = -5.0 if not is_solvable else -1.0
                        if not is_solvable:
                            print(f"  -> Applied Penalty -5.0 (Unsolvable)")
                    
                    self._save(i, fail_reward, curr_map, mask, actions, logp, values, topk_action_mask)
                    continue

                # --- 关键逻辑：分层奖励与数据保留 ---
                # 只要通过了连通性检查，就收集数据用于 World Model 训练
                valid_trajs.append(traj)
                next_maps.append(self._map_to_tensor(final_map))
                next_heats.append(heat)
                # 计算多样性奖励
                r_div = self.diversity.get_reward(
                    torch.tensor(final_map).unsqueeze(0).to(self.device)
                )
                
                # [ABLATION] w/o Diversity
                # Fix: We must LOG the real diversity score (r_div) to see mode collapse,
                # but we must ZERO out the reward given to the agent.
                r_div_for_reward = r_div.item() if hasattr(r_div, 'item') else r_div
                
                if hasattr(self.cfg, 'ablation') and self.cfg.ablation.type == 'no_diversity':
                     r_div_for_reward = 0.0
                
                div_rewards.append(r_div) # Log actual score

                # 5. 奖励计算逻辑
                raw_scalar_losses.append(scalar_loss) # Log raw loss
                
                # [MODIFIED] Balanced Difficulty Tuning
                # 1. Reduced Multiplier 10000 -> 2000: Align with EWC strength to prevent gradient explosion.
                scalar_loss = (scalar_loss * 10000.0) 

                # [ABLATION] No Learning Progress (No Loss Reward)
                if hasattr(self.cfg, 'ablation') and self.cfg.ablation.type == 'no_learning_progress':
                     scalar_loss = 0.0
                
                # [NEW] Solution Length Reward (Complexity Bonus)
                # Reduced to 0.1 so it doesn't overpower the Loss Reward.
                # We want High_Loss + Solvable, not just Long_Path + Easy.
                
                # Case 1 (Current): Loss=4.5, BFS=16 -> Total=20.5 (Prefers Easy Long)
                # Case 2 (Hard): Loss=5.7, BFS=12 -> Total=17.7
                
                # If BFS*0.1:
                # Case 1: 4.5 + 1.6 = 6.1
                # Case 2: 5.7 + 1.2 = 6.9 (Prefers Hard!)
                r_len = bfs_dist * 0.1 

                if hasattr(self.cfg, 'ablation') and self.cfg.ablation.type == 'no_diversity':
                     r_len = 0.0 
                
                if not is_solvable:
                     # [CRITICAL] 物理不可达：强制重罚
                     # 无论 Loss 多高，只要不可达就是失败的设计。
                     if hasattr(self.cfg, 'ablation') and self.cfg.ablation.type == 'no_validity_reward':
                         # [ABLATION] Disable strong penalty
                         reward = 0.0
                     else:
                         reward = -5.0
                elif solved:
                    # 智能体跑通了：全额奖励
                    # [Diversity Adjustment] 10.0 -> 5.0. 让多样性彻底成为辅助信号。
                    reward = scalar_loss + 5.0 * r_div_for_reward + r_len + 1.0 
                else:
                    # 连通但未跑通：
                    # [Diversity Adjustment] 5.0 -> 2.5.
                    reward = (scalar_loss * 0.8) + 2.5 * r_div_for_reward + r_len + 0.5 
                    # print(f"[UED] Iter {iteration} Env {i}: Solvable but not solved. LP: {scalar_loss:.4f}")

                # [NEW] Warmup Logic override
                # If in warmup phase, ignore WM loss and focus PURELY on structural validity & complexity
                warmup_iters = self.cfg.generator_agent.get("warmup_iterations", 0)
                if iteration < warmup_iters:
                     # Warmup Reward:
                     # 1. Base: +1.0 for being Solvable (vs -5.0 for fail)
                     # 2. Complexity: + BFS * 0.2 (Encourage long paths) -> ZERO if no_diversity
                     # 3. Diversity: + Div * 0.5 (Encourage variety) -> ZERO if no_diversity
                     # 4. WM Loss: IGNORED (0.0)
                     
                     r_bfs_warmup = bfs_dist * 0.2
                     if hasattr(self.cfg, 'ablation') and self.cfg.ablation.type == 'no_diversity':
                         r_bfs_warmup = 0.0
                     
                     reward = 1.0 + r_bfs_warmup + (r_div_for_reward * 5.0)

                # 保存到生成器的 PPO Buffer
                self._save(i, reward, curr_map, mask, actions, logp, values, topk_action_mask)

            # 6. 为下一轮准备历史上下文
            if len(next_maps) > 0:
                self.prev_data = (
                    torch.cat(next_maps),
                    torch.cat(next_heats),
                )
             
            # Calculate mean raw scalar loss for logging
            # Calculate mean raw scalar loss for logging
            mean_raw_loss = np.mean(raw_scalar_losses) if raw_scalar_losses else 0.0
            
            # [LOGGING] Calculate mean diversity reward (Weighted)
            # Replaced 10.0 with 5.0 to match the new reward formula.
            raw_div_score = np.mean([r.item() if hasattr(r, 'item') else r for r in div_rewards]) if div_rewards else 0.0
            mean_div_score = raw_div_score * 5.0
            
            # [NEW] Calculate avg BFS distance
            avg_bfs_dist = np.mean(bfs_stats) if bfs_stats else 0.0

            # [FIX] Handle empty next_maps (all rollouts failed)
            if len(next_maps) == 0:
                print("[Generator] Critical Warning: All environments failed to generate valid data!")
                # Return dummy tensors to prevent crash, downstream task should handle empty batch
                dummy_map = torch.zeros(0, 3, self.map_height, self.map_width, device=self.device)
                dummy_heat = torch.zeros(0, 1, self.map_height, self.map_width, device=self.device)
                return (
                    dummy_map,
                    dummy_heat,
                    0.0,
                    0.0,
                    [],
                    0,
                    0.0
                )

            return (
                torch.stack(next_maps).to(self.device),
                torch.stack(next_heats).to(self.device),
                mean_raw_loss, # Changed from mean_valid_loss to mean_raw_loss to match existing variable
                mean_div_score,
                valid_trajs,
                solvable_count, # [MODIFIED] Rate -> Count
                avg_bfs_dist    # [NEW] Avg Path Length
            )
    def update(self):
        loss, entropy = self.ppo.update()
        return loss, entropy, self.ppo.last_mean_reward

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _immutable_mask(self, ids):
        """
        Create an immutable mask for generator editing.

        Immutable positions (mask = 1.0):
        - start
        - goal
        - non-empty cells (e.g. wall)

        Mutable positions (mask = 0.0):
        - empty cells

        ids:    [B, H, W] object id map
        return: [B, 1, H, W] float tensor (1.0 = immutable)
        """
        mask = torch.ones_like(ids, dtype=torch.float32)

        # Only empty cells are editable
        mask[ids == self.OBJ_EMPTY] = 0.0

        # Explicitly protect start & goal (redundant but safe)
        mask[ids == self.OBJ_START] = 1.0
        mask[ids == self.OBJ_GOAL] = 1.0

        return mask.unsqueeze(1)


    def _apply_action(self, base_obj_map, act):
        H, W = base_obj_map.shape

        # 1) 复制 object map
        obj = base_obj_map.copy()

        # 2) 初始化 color (Fix: use default colors based on object type)
        # 默认 Minigrid 颜色映射:
        # red=0, green=1, blue=2, purple=3, yellow=4, grey=5
        # Objects: Wall=2, Floor=1, Door=4, Key=5, Ball=6, Box=7, Goal=8, Lava=9
        
        MAX_OBJ_ID = max(OBJECT_TO_IDX.values())

        # 建立默认颜色映射表 (Vectorized lookup)
        # 默认为 0 (red) for unknown
        default_color_map = np.zeros(MAX_OBJ_ID + 1, dtype=np.int64)
        
        # Configure defaults
        default_color_map[OBJECT_TO_IDX["wall"]] = COLOR_TO_IDX["grey"]    # Wall -> Grey
        default_color_map[OBJECT_TO_IDX["floor"]] = COLOR_TO_IDX["grey"]   # Floor -> Grey (or whatever)
        default_color_map[OBJECT_TO_IDX["door"]] = COLOR_TO_IDX["yellow"]  # Door -> Yellow
        default_color_map[OBJECT_TO_IDX["key"]] = COLOR_TO_IDX["yellow"]   # Key -> Yellow
        default_color_map[OBJECT_TO_IDX["ball"]] = COLOR_TO_IDX["red"]     # Ball -> Red
        default_color_map[OBJECT_TO_IDX["box"]] = COLOR_TO_IDX["yellow"]   # Box -> Yellow
        default_color_map[OBJECT_TO_IDX["goal"]] = COLOR_TO_IDX["green"]   # Goal -> Green
        default_color_map[OBJECT_TO_IDX["lava"]] = COLOR_TO_IDX["red"]     # Lava -> Red
        
        # Apply mapping
        # obj 是 [H, W] 的 object indices
        # color 变为对应的默认颜色
        # valid indices check
        safe_obj = obj.copy()
        safe_obj[safe_obj > MAX_OBJ_ID] = 0
        color = default_color_map[safe_obj]

        # Ensure correct type
        color = color.astype(np.int64)
        immutable = (obj == self.OBJ_START) | (obj == self.OBJ_GOAL)

        for i in range(H):
            for j in range(W):
                if immutable[i, j]:
                    continue

                a = act[i, j]
                if a == 0:
                    continue  # No-op

                obj_type, color_name = ACTION_TABLE[a]

                if obj_type == "key":
                    obj[i, j] = OBJECT_TO_IDX["key"]
                    color[i, j] = COLOR_TO_IDX[color_name]

                elif obj_type == "door":
                    obj[i, j] = OBJECT_TO_IDX["door"]
                    color[i, j] = COLOR_TO_IDX[color_name]
                    

                elif obj_type == "lava":
                    obj[i, j] = OBJECT_TO_IDX["lava"]

        return obj, color

    def _save(self, i, r, maps, masks, acts, lps, vals, topk_masks):
        pm, ph = self.prev_data
        self.ppo.save_buffer(
            maps[i : i + 1],
            (pm[i : i + 1], ph[i : i + 1]),
            masks[i : i + 1],
            acts[i : i + 1],
            lps[i : i + 1],
            vals[i : i + 1],
            r,
            topk_masks[i : i + 1],
        )

    def _zero_context(self, B, H, W):
        return (
            torch.zeros((B, 3, H, W), device=self.device),
            torch.zeros((B, 1, H, W), device=self.device)
        )

    def _map_to_tensor(self, m):
        # m is numpy (2, H, W) -> [Obj, Color]
        # We need (1, 3, H, W) -> [Obj, Color, State]
        t = torch.tensor(m, device=self.device).float() # (2, H, W)
        state_channel = torch.zeros_like(t[0:1]) # (1, H, W)
        return torch.cat([t, state_channel], dim=0).unsqueeze(0) # (1, 3, H, W)

    def _rollout_env(self, map_obj, iteration=0, batch_idx=0):
            """
            Rollout the agent in the generated environment.
            Returns:
                traj (dict): collected trajectory
                heat (Tensor): error heatmap [1, 1, H, W]
                scalar_loss (float): mean loss
                solved (bool): whether solved
            """
            # 1. 包装环境
            obj_map, color_map = map_obj
            map_tensor = torch.tensor(obj_map, dtype=torch.long, device=self.device)
            color_tensor = torch.tensor(color_map, dtype=torch.long, device=self.device)

            try:
                obj_str, color_str = self.support.interpret_env(map_tensor.cpu(), color_array=color_tensor.cpu())
                env_str = (obj_str, color_str)
            except Exception as e:
                print(f"Error wrapping env: {e}")
                return {}, None, None, False # 返回空字典而不是空列表，保持一致性

            # 2. 配置环境采样次数
            old_path = self.support.cfg.env.collect.data_save_path
            old_episodes = self.support.cfg.env.collect.episodes
            # self.support.cfg.env.collect.episodes = 100 # removed hardcode 

            try:
                # 3. 收集数据 (即使没解开，轨迹也会存入 npz)
                save_name = f'UED_temp_data_path_iter{iteration}_batch{batch_idx}'
                
                # 3. 收集数据 (即使没解开，轨迹也会存入 npz)
                save_name = f'UED_temp_data_path_iter{iteration}_batch{batch_idx}'
                
                # [关键修复]：限制单次 Rollout 只跑 1 个 Episode
                # 目的：防止 Agent 在同一个地图里反复刷分，导致 Buffer 被重复数据填满。
                # 我们希望 Generator 多生成不同的地图，而不是在一个地图里跑几万步。
                self.support.cfg.env.collect.episodes = 1 
                
                save_path = collect_data_general(
                    self.support.cfg,
                    env_source=env_str,
                    save_name=save_name,
                    max_steps=1000,             # 单个 Episode 最多 2000 步 (防止死循环)
                    maximum_dataset_size=self.support.cfg.env.collect.mini_dataset_size*2,  # 硬顶，超过就停
                    recollect_data=True 
                )
                
                # 4. 加载数据
                if os.path.exists(save_path):
                    task_npz = np.load(save_path, allow_pickle=True)
                    traj_data = {
                        'obs': torch.tensor(task_npz['a'], device=self.device),
                        'obs_next': torch.tensor(task_npz['b'], device=self.device),
                        'act': torch.tensor(task_npz['c'], device=self.device),
                        'info': task_npz['f'] if 'f' in task_npz else None
                    }
                    
                    rew_np = task_npz['d']
                    done_np = task_npz['e']
                    # solved 仅作为元数据保留，用于观察 Agent 表现
                    solved = np.any((done_np) & (rew_np > 0))
                else:
                    print(f"Error: Rollout data file {save_name} not found.")
                    return {}, None, None, False

                # 5. 计算 Heat Map (WM 在这些轨迹上的表现)
                # 注意：即使 solved=False，这些轨迹对 WM 依然有极高的学习价值
                avg_loss_map, loss_list = extract_loss_map_over_validations(
                    self.cfg,
                    net=self.wm,
                    old_params=None, 
                    data_dir=save_path,
                    valid_times=1
                )
                scalar_loss = np.mean(loss_list) if loss_list else 0.0
                
                # [MODIFIED] Log-Scale Heatmap to make small errors visible to Generator
                # MSE is typically 1e-3 to 1e-4, which is invisible to NN compared to integer inputs.
                # Log scale brings it to range [-9, 0], providing distinct gradients.
                scaled_loss_map = np.log(avg_loss_map + 1e-8)
                heat = torch.tensor(scaled_loss_map, device=self.device).unsqueeze(0).unsqueeze(0)

            except Exception as e:
                print(f"Error in rollout/heat computation: {e}")
                traj_data = {}
                heat = torch.zeros((1, 1, self.map_height, self.map_width), device=self.device)
                solved = False
                scalar_loss = 0.0
            finally:
                # 恢复原始配置
                self.support.cfg.env.collect.data_save_path = old_path
                self.support.cfg.env.collect.episodes = old_episodes

            return traj_data, heat, scalar_loss, solved
