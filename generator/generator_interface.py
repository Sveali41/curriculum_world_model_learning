import torch
import numpy as np
# === 直接导入你写好的模块 ===
from generator_agent import GeneratorPPO
from pcg_seeder import RuleBasedSeeder
from reward_system import DiversityModule, check_solvability, calculate_lp_reward
from env_designer import apply_action_flattened, get_immutable_mask
import torch.nn.functional as F

class GeneratorInterface:
    def __init__(self, world_model, device, batch_size=32):
        self.device = device
        self.batch_size = batch_size
        self.wm = world_model
        
        # === 1. 调用已经写好的类 ===
        self.ppo_agent = GeneratorPPO().to(device) # 参数在 GeneratorPPO 内部默认或传入
        self.seeder = RuleBasedSeeder(device=device)
        self.diversity = DiversityModule()
        
        # 历史状态缓存
        self.prev_data = None 

    def step(self):
        """
        执行一整套流程：生成 -> 评估 -> 存数据
        返回: valid_trajectories (给 WM 训练用)
        """
        # --- A. 调用 Seeder 生成底稿 ---
        base_map_ids, _ = self.seeder.generate_batch(self.batch_size)
        
        # --- B. 数据格式转换 (ID -> OneHot) ---
        map_onehot = F.one_hot(base_map_ids, num_classes=11).permute(0, 3, 1, 2).float()
        mask = get_immutable_mask(base_map_ids)
        
        # --- C. 调用 Agent 决策 ---
        # 这里的 select_action 内部已经封装了 History Encoder 的调用逻辑
        actions, logprobs, values, global_ctx = self.ppo_agent.select_action(
            map_onehot, self.prev_data, mask
        )
        
        # --- D. 执行动作 & 算分 ---
        valid_trajs = []
        next_history_list = [] # 用来收集 (Map, Heatmap, Attn)
        
        for i in range(self.batch_size):
            # 1. 物理应用 (调用 env_designer)
            final_obj, final_col = apply_action_flattened(
                base_map_ids[i].cpu().numpy(), 
                np.zeros_like(base_map_ids[i].cpu().numpy()), # 默认颜色
                actions[i].cpu().numpy()
            )
            
            # 2. 连通性检查 (调用 reward_system)
            if not check_solvability(final_obj):
                # 存负分
                self._save_to_ppo(i, -5.0, map_onehot, mask, actions, logprobs, values)
                continue # 跳过，不给 WM
            
            # 3. 跑环境 (Rollout) -> 得到 trajectory, heatmap, attn
            # 这一步需要你写一个辅助函数把 map 变成 minigrid env 跑一下
            traj, heat, attn = self._rollout_env(final_obj, final_col)
            
            # 4. 算分 (调用 reward_system)
            r_lp = calculate_lp_reward(self.wm, traj, lr=1e-3)
            r_div = self.diversity.get_reward(torch.tensor(final_obj).unsqueeze(0))
            total_r = r_lp + 0.1 * r_div
            
            # 5. 存正分
            self._save_to_ppo(i, total_r, map_onehot, mask, actions, logprobs, values)
            
            # 6. 收集有效数据
            valid_trajs.append(traj)
            
            # 7. 收集历史 (Map, Heat, Attn) 用于下一轮
            # 注意要转成 tensor
            next_history_list.append((
                torch.tensor(final_obj).to(self.device), 
                heat.to(self.device), 
                attn.to(self.device)
            ))

        # --- E. 更新历史上下文 ---
        if len(next_history_list) > 0:
            # 堆叠 valid 的数据作为下一轮的 history
            # 解包并 Stack
            maps, heats, attns = zip(*next_history_list)
            # 这里需要处理一下维度，确保是 [B, C, H, W]
            # 简单起见，如果 valid 数量不够 batch size，可以 repeat 或者只用部分
            self.prev_data = (torch.stack(maps), torch.stack(heats), torch.stack(attns))
            
        return valid_trajs

    def update(self):
        """调用 PPO 更新"""
        self.ppo_agent.update()

    # --- 内部辅助 ---
    def _save_to_ppo(self, idx, reward, maps, masks, acts, lps, vals):
        """简单的封装，调用 agent.save_buffer"""
        self.ppo_agent.save_buffer(
            maps[idx:idx+1], self.prev_data, masks[idx:idx+1],
            acts[idx:idx+1], lps[idx:idx+1], vals[idx:idx+1], reward
        )

    def _rollout_env(self, obj, col):
        """
        这里是你连接 Minigrid 的胶水代码
        你需要把 obj/col 变成 env，跑几步，拿到数据
        """
        # ... 你的环境运行逻辑 ...
        return traj, heatmap, attn