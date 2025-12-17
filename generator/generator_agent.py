import torch
import torch.nn as nn
import torch.optim as optim
from generator_network import MapEditorActorCritic
from history_encoder import HistoryEncoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GeneratorPPO:
    def __init__(self, 
                 lr_actor=1e-4, 
                 lr_critic=3e-4, 
                 gamma=0.99, 
                 K_epochs=4, 
                 eps_clip=0.2,
                 context_dim=64): # 新增 context维度
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # === 1. 初始化双网络 ===
        # A. 历史编码器 (The Historian)
        self.encoder = HistoryEncoder(context_dim=context_dim).to(device)
        
        # B. 策略网络 (The Editor) - 注意这里不再需要 num_actions 参数如果网络里写死了，或者根据需要传入
        self.policy = MapEditorActorCritic(context_dim=context_dim, num_actions=11).to(device)
        
        # === 2. 联合优化器 ===
        # 我们希望 PPO 的梯度能同时更新 Generator 和 HistoryEncoder
        # 这样 Encoder 就会学习"提取什么样的历史特征能帮 Generator 拿高分"
        self.optimizer = optim.Adam([
            # Encoder 参数
            {'params': self.encoder.parameters(), 'lr': lr_actor},
            # Policy 参数 (包含 backbone, heads, embeddings)
            {'params': self.policy.stem.parameters(), 'lr': lr_actor},
            {'params': self.policy.res_blocks.parameters(), 'lr': lr_actor},
            {'params': self.policy.emb_obj.parameters(), 'lr': lr_actor},
            {'params': self.policy.emb_color.parameters(), 'lr': lr_actor},
            {'params': self.policy.emb_state.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        # 旧策略 (用于 PPO Ratio 计算)
        self.policy_old = MapEditorActorCritic(context_dim=context_dim, num_actions=11).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 我们不需要 encoder_old，因为 encoder 只是特征提取器，它的输出被视为 state 的一部分
        
        self.MseLoss = nn.MSELoss()

        # === 3. 初始化 Buffer (存上一轮历史 + 当前轮动作) ===
        self.buffer = {
            'curr_map': [],      # [B, 3, 15, 15] 当前底图
            
            # --- 历史数据 (用于重算 Context) ---
            'prev_map': [],      # [B, 3, 15, 15]
            'prev_heatmap': [],  # [B, 1, 15, 15]
            'prev_attn': [],     # [B, 1, 15, 15]
            
            'mask': [],          # [B, 15, 15]
            'actions': [],       # [B, 15, 15]
            'logprobs': [],      # [B]
            'rewards': [],       # [B]
            'state_values': []   # [B]
        }

    def get_global_context(self, prev_map, prev_heatmap, prev_attn):
        """
        辅助函数：计算全局 Context Vector
        """
        # 1. 编码所有历史样本 -> [B, 64]
        batch_ctx = self.encoder(prev_map, prev_heatmap, prev_attn)
        # 2. 聚合 (Mean) -> [1, 64]
        global_ctx = batch_ctx.mean(dim=0, keepdim=True)
        return global_ctx

    def select_action(self, curr_map, prev_data, action_mask, max_edits=5):
        """
        Rollout 阶段
        prev_data: tuple (prev_map, prev_heatmap, prev_attn)
        """
        with torch.no_grad():
            # 1. 计算 Context (使用当前 Encoder)
            # 如果是第一轮(冷启动)，给全0 context
            if prev_data is None:
                batch_size = curr_map.shape[0]
                global_ctx = torch.zeros(1, 64).to(device)
            else:
                global_ctx = self.get_global_context(*prev_data)
            
            # 2. 广播 Context 到当前 Batch Size: [1, 64] -> [B, 64]
            # 注意：act 函数内部会负责把它广播成 [B, 64, H, W]
            batch_size = curr_map.shape[0]
            ctx_expanded = global_ctx.repeat(batch_size, 1)

            # 3. 策略决策
            action, action_logprob, state_val = self.policy_old.act(
                curr_map, ctx_expanded, action_mask, max_edits
            )
        
        # 空间 logprob 求和
        action_logprob_sum = action_logprob.sum(dim=(1, 2))
        
        return action, action_logprob_sum, state_val, global_ctx

    def save_buffer(self, curr_map, prev_data, mask, action, logprob, value, reward):
        """
        存数据。注意 prev_data 需要解包存储。
        """
        self.buffer['curr_map'].append(curr_map.cpu())
        
        if prev_data is not None:
            self.buffer['prev_map'].append(prev_data[0].cpu())
            self.buffer['prev_heatmap'].append(prev_data[1].cpu())
            self.buffer['prev_attn'].append(prev_data[2].cpu())
        else:
            # 冷启动时的占位符 (全0)
            dummy_shape = curr_map.shape
            dummy_heat = torch.zeros((dummy_shape[0], 1, dummy_shape[2], dummy_shape[3]))
            self.buffer['prev_map'].append(torch.zeros_like(curr_map).cpu())
            self.buffer['prev_heatmap'].append(dummy_heat)
            self.buffer['prev_attn'].append(dummy_heat)

        self.buffer['mask'].append(mask.cpu())
        self.buffer['actions'].append(action.cpu())
        self.buffer['logprobs'].append(logprob.cpu())
        self.buffer['state_values'].append(value.cpu())
        self.buffer['rewards'].append(reward)

    def clear_buffer(self):
        for k in self.buffer:
            self.buffer[k] = []

    def update(self):
        """
        PPO Update (包含 Encoder 更新)
        """
        # 1. 准备数据 Tensor
        rewards = torch.tensor(self.buffer['rewards'], dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_curr_maps = torch.cat(self.buffer['curr_map']).to(device)
        old_masks = torch.cat(self.buffer['mask']).to(device)
        old_actions = torch.cat(self.buffer['actions']).to(device)
        old_logprobs = torch.cat(self.buffer['logprobs']).to(device)
        old_state_values = torch.cat(self.buffer['state_values']).to(device).squeeze()
        
        # 历史数据
        old_prev_maps = torch.cat(self.buffer['prev_map']).to(device)
        old_prev_heatmaps = torch.cat(self.buffer['prev_heatmap']).to(device)
        old_prev_attns = torch.cat(self.buffer['prev_attn']).to(device)

        # 2. 计算 Advantage
        advantages = rewards - old_state_values.detach()

        # 3. 训练循环
        for _ in range(self.K_epochs):
            # --- 关键：重新计算 Context ---
            # 这一步必须在梯度带(Gradient Tape)里做，这样梯度才能流回 self.encoder
            # 1. 编码历史
            batch_ctx = self.encoder(old_prev_maps, old_prev_heatmaps, old_prev_attns)
            
            # 2. 聚合 (Mean)
            # 这里我们假设 buffer 里的是同一批次的数据，可以直接 mean
            # 如果 buffer 包含多个批次，逻辑会更复杂(需要按批次 mean)，但对于 on-policy 只要 buffer 是一轮生成的通常没问题
            global_ctx = batch_ctx.mean(dim=0, keepdim=True)
            
            # 3. 广播
            batch_size = old_curr_maps.shape[0]
            ctx_expanded = global_ctx.repeat(batch_size, 1)

            # --- 评估策略 ---
            # 使用重新计算的 ctx_expanded
            logprobs_map, state_values, dist_entropy = self.policy.evaluate(
                old_curr_maps, ctx_expanded, old_actions, old_masks
            )
            
            logprobs = logprobs_map.sum(dim=(1, 2))
            state_values = torch.squeeze(state_values)
            dist_entropy = dist_entropy.mean()

            # --- PPO Loss ---
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # 这里的 Loss 会同时更新 Policy 和 Encoder
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # --- Update ---
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 4. 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 5. 清空
        self.clear_buffer()

    def save(self, checkpoint_path):
        # 保存两个网络
        state = {
            'policy': self.policy.state_dict(),
            'encoder': self.encoder.state_dict()
        }
        torch.save(state, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.policy_old.load_state_dict(checkpoint['policy']) 
        self.encoder.load_state_dict(checkpoint['encoder'])