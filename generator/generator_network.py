import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ResBlock(nn.Module):
    """标准残差块"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))


class MapEditorActorCritic(nn.Module):
    def __init__(
        self,
        num_actions=11,
        hidden_dim=64,
        max_obj_id=15,
        max_color_id=6,
        max_state_id=3,
        context_dim=64,
    ):
        super().__init__()

        # === 1. Embedding Layers ===
        self.emb_dim_obj = 16
        self.emb_dim_color = 8
        self.emb_dim_state = 4

        self.emb_obj = nn.Embedding(max_obj_id + 1, self.emb_dim_obj)
        self.emb_color = nn.Embedding(max_color_id + 1, self.emb_dim_color)
        self.emb_state = nn.Embedding(max_state_id + 1, self.emb_dim_state)

        # === 2. 输入通道数 ===
        # Embeddings + Context(context_dim) + Coords(2)
        total_in_channels = (self.emb_dim_obj + self.emb_dim_color + self.emb_dim_state) + context_dim + 2

        # === 3. Backbone (ResNet) ===
        self.stem = nn.Sequential(
            nn.Conv2d(total_in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )

        # === 4. Actor Head ===
        self.actor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_actions, 1),
        )

        # === 5. Critic Head (修复：不再硬编码 15*15) ===
        self.critic = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_coordinate_channels(self, batch_size: int, h: int, w: int, device):
        """    
        Create normalized x/y coordinate channels (CoordConv) in [-1, 1]
        to provide explicit spatial position information to the network.
        """
        # shape: [B,1,H,W]
        xx = torch.arange(w, device=device).view(1, 1, 1, w).repeat(batch_size, 1, h, 1)
        yy = torch.arange(h, device=device).view(1, 1, h, 1).repeat(batch_size, 1, 1, w)

        # 避免 w/h 为 1 时除零
        w_denom = max(w - 1, 1)
        h_denom = max(h - 1, 1)

        xx = xx / w_denom * 2 - 1
        yy = yy / h_denom * 2 - 1
        return xx, yy

    def forward_features(self, base_map_vec, context_vec):
        """
  
        Encode the base map and history-conditioned context into
        a spatial feature map for per-cell editing decisions.

        base_map_vec:     [B, 3, H, W] (Long)
        context_vec: [B, context_dim] (Float)
        """
        B, _, H, W = base_map_vec.shape

        # 1) Map embeddings
        feat_obj = self.emb_obj(base_map_vec[:, 0].long()).permute(0, 3, 1, 2)     # [B, 16, H, W]
        feat_col = self.emb_color(base_map_vec[:, 1].long()).permute(0, 3, 1, 2)   # [B,  8, H, W]
        feat_sta = self.emb_state(base_map_vec[:, 2].long()).permute(0, 3, 1, 2)   # [B,  4, H, W]

        # 2) Coords
        xx, yy = self.get_coordinate_channels(B, H, W, base_map_vec.device)

        # 3) Context broadcast: [B, C] -> [B, C, H, W]
        context_tiled = context_vec.view(B, -1, 1, 1).expand(-1, -1, H, W)

        # 4) Concat
        x = torch.cat([feat_obj, feat_col, feat_sta, context_tiled, xx, yy], dim=1)

        # 5) Backbone
        x = self.stem(x)
        x = self.res_blocks(x)
        return x

    @torch.no_grad()
    def act(self, map_vec, context_vec, action_mask=None, max_edits=0.4):
        """
        采样动作
        action_mask: [B, H, W] 的 bool mask，True 表示该位置不可编辑
        """
        features = self.forward_features(map_vec, context_vec)
        logits = self.actor(features)  # [B, A, H, W]

        # --- Safety Masking ---
        action_mask = action_mask.bool() # [B,1,H,W]
        mask_hw = action_mask.squeeze(1)
        if action_mask is not None:
            logits[:, 0, :, :].masked_fill_(mask_hw, 1e9)       # No-op 强制
            logits[:, 1:, :, :].masked_fill_(action_mask, -1e9)   # 其他动作禁止

        probs = F.softmax(logits, dim=1)
        prob_change = 1.0 - probs[:, 0, :, :]  # [B,H,W]

        # --- Top-K edits ---
        B, H, W = prob_change.shape
        flat_probs = prob_change.view(B, -1)
        num_cells = H * W
        k = int(max(1, round(max_edits * num_cells)))
        k = min(k, num_cells)
        topk_values, _ = torch.topk(flat_probs, k=k, dim=1)
        threshold = topk_values[:, -1].view(B, 1, 1)

        topk_mask = prob_change >= threshold  # [B,H,W]

        # 对非 topk 的位置：禁止修改动作，强制 No-op
        logits[:, 1:, :, :].masked_fill_((~topk_mask).unsqueeze(1), -1e9)
        logits[:, 0, :, :].masked_fill_(~topk_mask, 1e9)  # 修复：不 unsqueeze

        # --- Sampling ---
        logits_hw = logits.permute(0, 2, 3, 1)  # [B,H,W,A]
        dist = Categorical(logits=logits_hw)

        action = dist.sample()                 # [B,H,W]
        action_logprob = dist.log_prob(action) # [B,H,W]
        state_val = self.critic(features)      # [B,1]

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, map_vec, context_vec, action, action_mask=None):
        """计算 LogProb / Value / Entropy (用于 PPO update)"""
        features = self.forward_features(map_vec, context_vec)
        logits = self.actor(features)  # [B,A,H,W]

        if action_mask is not None:
            mask_expanded = action_mask.unsqueeze(1)  # [B,1,H,W]
            logits[:, 0, :, :].masked_fill_(action_mask, 1e9)
            logits[:, 1:, :, :].masked_fill_(mask_expanded, -1e9)

        logits_hw = logits.permute(0, 2, 3, 1)  # [B,H,W,A]
        dist = Categorical(logits=logits_hw)

        action_logprobs = dist.log_prob(action)  # [B,H,W]
        dist_entropy = dist.entropy()            # [B,H,W]
        state_values = self.critic(features)     # [B,1]

        return action_logprobs, state_values, dist_entropy
