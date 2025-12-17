import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ResBlock(nn.Module):
    """(保持不变) 标准残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))

class MapEditorActorCritic(nn.Module):
    def __init__(self, 
                 num_actions=11, 
                 hidden_dim=64,
                 max_obj_id=15, 
                 max_color_id=6, 
                 max_state_id=3,
                 context_dim=64): # <--- [修改点 1] 新增 context_dim 参数
        super().__init__()

        # === 1. Embedding Layers (保持不变) ===
        self.emb_dim_obj = 16
        self.emb_dim_color = 8
        self.emb_dim_state = 4
        
        self.emb_obj = nn.Embedding(max_obj_id + 1, self.emb_dim_obj)
        self.emb_color = nn.Embedding(max_color_id + 1, self.emb_dim_color)
        self.emb_state = nn.Embedding(max_state_id + 1, self.emb_dim_state)

        # === [修改点 2] 计算总输入通道数 ===
        # 原来: Embeddings + Heatmap(1) + Coords(2)
        # 现在: Embeddings + Context(64) + Coords(2)
        # 我们去掉了 Heatmap，因为用 Context 代替了它
        total_in_channels = (self.emb_dim_obj + self.emb_dim_color + self.emb_dim_state) + context_dim + 2
        
        # === 3. Backbone (ResNet) (保持不变) ===
        self.stem = nn.Sequential(
            nn.Conv2d(total_in_channels, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim)
        )

        # === 4. Actor Head (保持不变) ===
        self.actor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, num_actions, 1) 
        )

        # === 5. Critic Head (保持不变) ===
        self.critic = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Flatten(),
            nn.Linear(15 * 15, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_coordinate_channels(self, batch_size, h, w, device):
        """生成相对坐标通道"""
        xx = torch.arange(w, device=device).view(1, 1, 1, w).repeat(batch_size, 1, h, 1)
        yy = torch.arange(h, device=device).view(1, 1, h, 1).repeat(batch, 1, 1, w)
        xx = xx / (w - 1) * 2 - 1
        yy = yy / (h - 1) * 2 - 1
        return xx, yy

    def forward_features(self, map_vec, context_vec): # <--- [修改点 3] 输入变了
        """
        map_vec: [B, 3, H, W] (Base Map)
        context_vec: [B, context_dim] (History Context, 1D Vector)
        """
        B, _, H, W = map_vec.shape
        
        # 1. 提取 Map Embeddings
        feat_obj = self.emb_obj(map_vec[:, 0].long()).permute(0, 3, 1, 2)
        feat_col = self.emb_color(map_vec[:, 1].long()).permute(0, 3, 1, 2)
        feat_sta = self.emb_state(map_vec[:, 2].long()).permute(0, 3, 1, 2)
        
        # 2. 生成坐标
        xx, yy = self.get_coordinate_channels(B, H, W, map_vec.device)
        
        # === [修改点 4] 处理 Context Vector (广播/Tiling) ===
        # context_vec 是 [B, 64]
        # 我们要把它变成 [B, 64, H, W]，即让每个像素点都拥有这个 context 信息
        context_tiled = context_vec.view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        # 3. 拼接所有特征
        # [Obj, Col, Sta, Context, X, Y]
        x = torch.cat([feat_obj, feat_col, feat_sta, context_tiled, xx, yy], dim=1)
        
        # 4. 通过骨干网络
        x = self.stem(x)
        x = self.res_blocks(x)
        return x

    def act(self, map_vec, context_vec, action_mask=None, max_edits=5): # <--- [修改点 5] 参数变了
        """
        采样动作
        """
        features = self.forward_features(map_vec, context_vec) # 传 context
        logits = self.actor(features) 
        
        # --- Safety Masking & Top-K Logic (保持不变) ---
        if action_mask is not None:
            mask_expanded = action_mask.unsqueeze(1)
            logits[:, 0, :, :].masked_fill_(mask_expanded.squeeze(1), 1e9)
            logits[:, 1:, :, :].masked_fill_(mask_expanded, -1e9)

        probs = F.softmax(logits, dim=1)
        prob_change = 1.0 - probs[:, 0, :, :]
        
        B, H, W = prob_change.shape
        flat_probs = prob_change.view(B, -1)
        topk_values, _ = torch.topk(flat_probs, k=max_edits, dim=1)
        threshold = topk_values[:, -1].view(B, 1, 1)
        
        topk_mask = prob_change >= threshold
        
        logits[:, 1:, :, :].masked_fill_((~topk_mask).unsqueeze(1), -1e9)
        logits[:, 0, :, :].masked_fill_((~topk_mask).unsqueeze(1), 1e9)

        # --- Sampling ---
        logits = logits.permute(0, 2, 3, 1)
        dist = Categorical(logits=logits)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(features)
        
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, map_vec, context_vec, action, action_mask=None): # <--- [修改点 6] 参数变了
        """计算LogProb (用于 PPO Update)"""
        features = self.forward_features(map_vec, context_vec) # 传 context
        logits = self.actor(features)

        if action_mask is not None:
            mask_expanded = action_mask.unsqueeze(1)
            logits[:, 0, :, :].masked_fill_(mask_expanded.squeeze(1), 1e9)
            logits[:, 1:, :, :].masked_fill_(mask_expanded, -1e9)

        logits = logits.permute(0, 2, 3, 1)
        dist = Categorical(logits=logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)

        return action_logprobs, state_values, dist_entropy