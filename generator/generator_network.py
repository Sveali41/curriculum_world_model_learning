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
        
        # 优化：重置 Actor 最后一层的权重为 0.01（近乎 0）
        # 这样初始 Logits 接近 0，使得初始策略接近“均匀随机分布”
        # 确保所有动作（门、钥匙、岩浆）在最开始都有相同的概率被采样到
        last_layer = self.actor[-1]
        nn.init.orthogonal_(last_layer.weight, gain=0.01)
        nn.init.constant_(last_layer.bias, 0)

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

        base_map_vec:     [B, NUM_CLASSES, H, W] (Long)
        context_vec: [B, context_dim] (Float)
        """
        B, _, H, W = base_map_vec.shape

        # 1) Map embeddings
        feat_obj = self.emb_obj(base_map_vec[:, 0].long()).permute(0, 3, 1, 2)     # [B, H, W]
        feat_col = self.emb_color(base_map_vec[:, 1].long()).permute(0, 3, 1, 2)   # [B, H, W]
        feat_sta = self.emb_state(base_map_vec[:, 2].long()).permute(0, 3, 1, 2)   # [B, H, W]

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


    def _get_topk_mask(self, logits, max_edits):
        """
        Helper to compute the Top-K mask based on logits and max_edits ratio.
        Shared by act() and evaluate().
        """
        probs = F.softmax(logits, dim=1)
        prob_change = 1.0 - probs[:, 0, :, :]  # [B,H,W]

        B, H, W = prob_change.shape
        flat_probs = prob_change.view(B, -1)
        num_cells = H * W
        
        # Calculate k
        k = int(max(1, round(max_edits * num_cells)))
        k = min(k, num_cells)
        
        # Find Top-K threshold
        topk_values, _ = torch.topk(flat_probs, k=k, dim=1)
        threshold = topk_values[:, -1].view(B, 1, 1)
        
        # Determine mask
        topk_mask = prob_change >= threshold  # [B,H,W]
        return topk_mask

    @torch.no_grad()
    def act(self, map_vec, context_vec, action_mask=None, max_edits=0.4):
        """
        采样动作
        map_vec: [B, num_classes_obj, H, W] 
        action_mask: [B, H, W] 的 bool mask，True 表示该位置不可编辑
        """
        # Cache max_edits for use in evaluate/update phase
        self._cached_max_edits = max_edits

        features = self.forward_features(map_vec, context_vec)
        logits = self.actor(features)  # [B, A, H, W]

        # --- Stability Clamp ---
        logits = torch.clamp(logits, min=-100, max=100)

        # --- Safety Masking ---
        if action_mask is not None:
            action_mask = action_mask.bool() # [B,1,H,W] or [B,H,W]
            if action_mask.dim() == 3:
                mask_hw = action_mask
                mask_others = action_mask.unsqueeze(1)
            else: # dim == 4
                mask_hw = action_mask.squeeze(1)
                mask_others = action_mask

            # 强制 No-op: 将 No-op 的 logit 设为极大
            logits[:, 0, :, :].masked_fill_(mask_hw, 1e9)       
            # 禁止其他动作: 将其他动作的 logit 设为极小
            logits[:, 1:, :, :].masked_fill_(mask_others, -1e9)   

        # --- Top-K edits logic ---
        topk_mask = self._get_topk_mask(logits, max_edits)

        # 对非 topk 的位置：
        # 1. 强制 No-op (logits[0] -> 1e9)
        logits[:, 0, :, :].masked_fill_(~topk_mask, 1e9)
        # 2. 禁止修改动作 (logits[1:] -> -1e9)
        logits[:, 1:, :, :].masked_fill_((~topk_mask).unsqueeze(1), -1e9)

        # --- Sampling ---
        logits_hw = logits.permute(0, 2, 3, 1)  # [B,H,W,A]
        dist = Categorical(logits=logits_hw)

        action = dist.sample()                 # [B,H,W]
        action_logprob = dist.log_prob(action) # [B,H,W]
        state_val = self.critic(features)      # [B,1]

        return action.detach(), action_logprob.detach(), state_val.detach(), topk_mask.detach()

    def evaluate(self, map_vec, context_vec, action, action_mask=None, max_edits=None, target_topk_mask=None):
        """
        计算 LogProb / Value / Entropy
        target_topk_mask: [B, H, W] 的 bool mask。
              如果提供，则强制使用该 mask（通常来自 Buffer），防止重新计算导致 Top-K 集合变化引发 NaN。
        """
        # Attempt to use cached max_edits from act() if available
        if max_edits is None:
            max_edits = getattr(self, '_cached_max_edits', 0.4)

        features = self.forward_features(map_vec, context_vec)
        logits = self.actor(features)  # [B, A, H, W]

        # --- Stability Clamp ---
        logits = torch.clamp(logits, min=-100, max=100)

        # --- Safety Masking ---
        if action_mask is not None:
            mask_bool = action_mask.to(torch.bool)
            if mask_bool.dim() == 3:
                m3d = mask_bool
                m4d = mask_bool.unsqueeze(1)
            else:
                m3d = mask_bool.squeeze(1)
                m4d = mask_bool
            
            # 使用足够大的值进行 Mask，而不是之前的 100
            FILL_VAL = 1e9
            logits[:, 0, :, :].masked_fill_(m3d, FILL_VAL)
            logits[:, 1:, :, :].masked_fill_(m4d, -FILL_VAL)

        # --- Top-K edits logic ---
        if target_topk_mask is not None:
            topk_mask = target_topk_mask
        else:
            topk_mask = self._get_topk_mask(logits, max_edits)

        # Apply Top-K Mask
        logits[:, 0, :, :].masked_fill_(~topk_mask, 1e9)
        logits[:, 1:, :, :].masked_fill_((~topk_mask).unsqueeze(1), -1e9)

        # --- Distribution ---
        logits_hw = logits.permute(0, 2, 3, 1)  # [B, H, W, A]
        dist = Categorical(logits=logits_hw)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)

        # --- Safety Checks ---
        if torch.isnan(action_logprobs).any():
             # 兜底：如果仍有 NaN，设为 0（虽然这不应该发生）
            action_logprobs = torch.nan_to_num(action_logprobs, 0.0)
        if torch.isnan(dist_entropy).any():
            dist_entropy = torch.nan_to_num(dist_entropy, 0.0)
            
        return action_logprobs, state_values, dist_entropy