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
        ablation_type="none",
    ):
        super().__init__()

        # === 1. Embedding Layers ===
        self.emb_dim_obj = 16
        self.emb_dim_color = 8
        self.emb_dim_state = 4

        self.emb_obj = nn.Embedding(max_obj_id + 1, self.emb_dim_obj)
        self.emb_color = nn.Embedding(max_color_id + 1, self.emb_dim_color)
        self.emb_state = nn.Embedding(max_state_id + 1, self.emb_dim_state)
        self.ablation_type = ablation_type

        # === 2. Input Channels ===
        base_in_channels = (self.emb_dim_obj + self.emb_dim_color + self.emb_dim_state) + 2  
        if self.ablation_type == "no_history":
            total_in_channels = base_in_channels 
        else:
            total_in_channels = base_in_channels + context_dim

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

        # === 4. Dual Heads ===
        self.num_actions = num_actions
        # A. Terrain Head (Spatial)
        self.actor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_actions, 1),
        )

        # B. Stats Head (32 Buttons Config: 2 rows x 16 slots)
        self.num_stats_slots = 32 
        self.num_stats_actions = 2 # (0: Off, 1: On)
        self.stats_actor = nn.Sequential(
            nn.Linear(hidden_dim + 16, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.num_stats_slots * self.num_stats_actions)
        )

        # === 5. Critic Head ===
        self.critic = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.apply(self._init_weights)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0)
        self.actor[-1].bias.data[0] = 0.0

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_coordinate_channels(self, batch_size: int, h: int, w: int, device):
        xx = torch.arange(w, device=device).view(1, 1, 1, w).repeat(batch_size, 1, h, 1)
        yy = torch.arange(h, device=device).view(1, 1, h, 1).repeat(batch_size, 1, 1, w)
        w_denom, h_denom = max(w - 1, 1), max(h - 1, 1)
        return xx / w_denom * 2 - 1, yy / h_denom * 2 - 1

    def forward_features(self, base_map_vec, context_vec):
        B, _, H, W = base_map_vec.shape
        feat_obj = self.emb_obj(base_map_vec[:, 0].long()).permute(0, 3, 1, 2)
        feat_col = self.emb_color(base_map_vec[:, 1].long()).permute(0, 3, 1, 2)
        feat_sta = self.emb_state(base_map_vec[:, 2].long()).permute(0, 3, 1, 2)
        xx, yy = self.get_coordinate_channels(B, H, W, base_map_vec.device)
        
        if self.ablation_type == "no_history" or context_vec is None:
            x = torch.cat([feat_obj, feat_col, feat_sta, xx, yy], dim=1)
        else:
            context_tiled = context_vec.view(B, -1, 1, 1).expand(-1, -1, H, W)
            x = torch.cat([feat_obj, feat_col, feat_sta, context_tiled, xx, yy], dim=1)
        
        x = self.stem(x)
        x = self.res_blocks(x)
        return x

    def _get_topk_mask(self, logits, max_edits, action_mask=None):
        probs = F.softmax(logits, dim=1)
        prob_change = 1.0 - probs[:, 0, :, :]
        
        # KEY BUG FIX: Force masked pixels (like boundary walls) to have 0 prob of change 
        # so they don't consume the top-k budget when logits are uniformly initialized!
        if action_mask is not None:
             prob_change = prob_change.masked_fill(action_mask.squeeze(1) > 0.5, -1.0)
             
        B, H, W = prob_change.shape
        flat_probs = prob_change.view(B, -1)
        num_cells = (action_mask < 0.5).sum(dim=(1, 2, 3)).float().mean().item() if action_mask is not None else H*W
        k = max(1, min(int(round(max_edits * num_cells)), H * W))
        
        # When flat_probs are equal (uniform), torch.topk returns the first k elements
        # Without the mask fill above, it always picked the top-left corner (the walls!)
        topk_vals, topk_indices = torch.topk(flat_probs, k=k, dim=1)
        flat_mask = torch.zeros_like(flat_probs, dtype=torch.bool)
        flat_mask.scatter_(1, topk_indices, True)
        return flat_mask.view(B, H, W)

    def _get_stats_topk_mask(self, logits_stats, max_stats_edit_ratio):
        B, N, _ = logits_stats.shape
        prob_click = torch.softmax(logits_stats, dim=-1)[:, :, 1]
        
        # Calculate dynamic k from ratio (e.g. 0.1 * 32 slots = 3.2 -> 3)
        k = max(1, min(int(round(max_stats_edit_ratio * N)), N))
        
        _, topk_indices = torch.topk(prob_click, k=k, dim=-1)
        mask = torch.zeros_like(prob_click, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)
        return mask

    def act(self, map_vec, context_vec, action_mask=None, max_edits=0.4, max_stats_edit_ratio=0.1, stats_heat=None):
        features = self.forward_features(map_vec, context_vec)
        B, _, H, W = map_vec.shape

        # A. Terrain Sampling
        logits = self.actor(features)
        topk_mask = self._get_topk_mask(logits, max_edits, action_mask)
        logits[:, 0, :, :].masked_fill_(~topk_mask, 1e9)
        logits[:, 1:, :, :].masked_fill_(~topk_mask.unsqueeze(1), -1e9)
        dist = Categorical(logits=logits.permute(0, 2, 3, 1))
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # B. Stats (32 Piano Buttons) Sampling
        global_vec = F.adaptive_avg_pool2d(features, (1, 1)).view(B, -1)
        if stats_heat is None:
            stats_heat = torch.zeros(B, 16, device=map_vec.device)
        global_vec = torch.cat([global_vec, stats_heat], dim=1)
        
        logits_stats = self.stats_actor(global_vec).view(B, self.num_stats_slots, self.num_stats_actions)
        topk_stats_mask = self._get_stats_topk_mask(logits_stats, max_stats_edit_ratio)
        logits_stats[:, :, 0].masked_fill_(~topk_stats_mask, 1e9)
        logits_stats[:, :, 1].masked_fill_(~topk_stats_mask, -1e9)
        stats_dist = Categorical(logits=logits_stats)
        stats_action = stats_dist.sample()
        stats_logprob = stats_dist.log_prob(stats_action).sum(dim=-1)

        value = self.critic(features)
        return action.detach(), stats_action.detach(), action_logprob.detach(), stats_logprob.detach(), value.detach(), topk_mask.detach()

    def evaluate(self, map_vec, context_vec, action_tuple, action_mask=None, target_topk_mask=None, stats_heat=None):
        terrain_action, stats_action = action_tuple
        features = self.forward_features(map_vec, context_vec)
        B, _, H, W = map_vec.shape

        # A. Terrain Eval
        logits = self.actor(features)
        if target_topk_mask is not None:
             logits[:, 0, :, :].masked_fill_(~target_topk_mask, 1e9)
             logits[:, 1:, :, :].masked_fill_(~target_topk_mask.unsqueeze(1), -1e9)
        dist = Categorical(logits=logits.permute(0, 2, 3, 1))
        action_logprobs = dist.log_prob(terrain_action)
        dist_entropy = dist.entropy().mean()

        # B. Stats Eval
        global_vec = F.adaptive_avg_pool2d(features, (1, 1)).view(B, -1)
        if stats_heat is None:
            stats_heat = torch.zeros(B, 16, device=map_vec.device)
        global_vec = torch.cat([global_vec, stats_heat], dim=1)
        
        logits_stats = self.stats_actor(global_vec).view(B, self.num_stats_slots, self.num_stats_actions)
        stats_dist = Categorical(logits=logits_stats)
        stats_logprobs = stats_dist.log_prob(stats_action).sum(dim=-1)
        stats_entropy = stats_dist.entropy().mean()

        value = self.critic(features)
        total_entropy = dist_entropy + stats_entropy
        return action_logprobs, stats_logprobs, value, total_entropy
