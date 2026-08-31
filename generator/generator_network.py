import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from generator.crafter_env_designer import CRAFTER_OBJ_MAP


class ResBlock(nn.Module):
    """Standard residual block."""
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
        env_type="minigrid",
        spatial_dpp_sigma=1.5,
    ):
        super().__init__()
        self.env_type = str(env_type).lower()
        self.is_bipedal = ("bipedal" in self.env_type)
        self.is_crafter = ("crafter" in self.env_type)
        self.is_minigrid = (self.env_type == "minigrid")

        if self.is_crafter:
            max_obj_id = max(CRAFTER_OBJ_MAP.values())
            # Crafter generator currently uses object ids plus placeholder zero
            # channels for color/state.
            max_color_id = 0
            max_state_id = 0

        # === 1. Embedding Layers ===
        self.emb_dim_obj = 16
        self.emb_dim_color = 8
        self.emb_dim_state = 4

        self.emb_obj = nn.Embedding(max_obj_id + 1, self.emb_dim_obj)
        self.emb_color = nn.Embedding(max_color_id + 1, self.emb_dim_color)
        self.emb_state = nn.Embedding(max_state_id + 1, self.emb_dim_state)
        
        # If bipedal, channel 1 (terrain error) is continuous. 
        # We adjust in_channels accordingly.
        actual_col_dim = 1 if self.is_bipedal else self.emb_dim_color
        actual_sta_dim = 1 if self.is_bipedal else self.emb_dim_state
        
        self.ablation_type = ablation_type
        self.spatial_dpp_sigma = max(float(spatial_dpp_sigma), 1e-3)

        # === 2. Input Channels ===
        base_in_channels = (self.emb_dim_obj + actual_col_dim + actual_sta_dim) + 2  
        if self.ablation_type == "no_history" or self.is_minigrid:
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

        # MiniGrid position quality is computed only from the current base map.
        # Semantic history is fused afterward as a residual on edit-type logits,
        # so a previous failure can request "what" to revisit without anchoring
        # the edit to the previous absolute coordinate.
        if self.is_minigrid and self.ablation_type != "no_history":
            self.history_fusion = nn.Sequential(
                nn.Conv2d(hidden_dim + context_dim, hidden_dim, 1),
                nn.ReLU(inplace=True),
                ResBlock(hidden_dim),
            )
            self.history_type_actor = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, num_actions - 1, 1),
            )
        else:
            self.history_fusion = None
            self.history_type_actor = None

        # B. Stats Head (32 Buttons Config: 2 rows x 16 slots)
        self.num_stats_slots = 32 
        self.num_stats_actions = 2 # (0: Off, 1: On)
        
        # Bipedal uses 26-dim stats heat, MiniGrid/Crafter still use 16
        stats_in_dim = 26 if self.is_bipedal else 16
        self.stats_actor = nn.Sequential(
            nn.Linear(hidden_dim + stats_in_dim, 256),
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
        if self.history_type_actor is not None:
            nn.init.orthogonal_(self.history_type_actor[-1].weight, gain=0.01)
            nn.init.constant_(self.history_type_actor[-1].bias, 0)

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
        """
        Compresses spatial map features and global context vectors into a unified unified representation 
        using feature concatenation and residual processing.
        base_map_vec: (B, 3, H, W)
        context_vec: (B, context_dim)
        return: (B, hidden_dim, H, W)
        """
        B, _, H, W = base_map_vec.shape
        feat_obj = self.emb_obj(base_map_vec[:, 0].long()).permute(0, 3, 1, 2)
        
        if self.is_bipedal:
            # Channel 1: Raw Terrain Error Heatmap
            # Channel 2: Zeros (or other continuous signal if added)
            feat_col = base_map_vec[:, 1:2]
            feat_sta = base_map_vec[:, 2:3]
        else:
            feat_col = self.emb_color(base_map_vec[:, 1].long()).permute(0, 3, 1, 2)
            feat_sta = self.emb_state(base_map_vec[:, 2].long()).permute(0, 3, 1, 2)
        xx, yy = self.get_coordinate_channels(B, H, W, base_map_vec.device)
        
        if self.is_minigrid or self.ablation_type == "no_history" or context_vec is None:
            x = torch.cat([feat_obj, feat_col, feat_sta, xx, yy], dim=1)
        else:
            context_tiled = context_vec.view(B, -1, 1, 1).expand(-1, -1, H, W)
            x = torch.cat([feat_obj, feat_col, feat_sta, context_tiled, xx, yy], dim=1)
        
        x = self.stem(x)
        x = self.res_blocks(x)
        return x

    def _minigrid_logits(self, features, context_vec):
        """Return history-free placement logits and semantic type logits."""
        base_logits = self.actor(features)
        if self.history_fusion is None or context_vec is None:
            return base_logits, base_logits, features

        B, _, H, W = features.shape
        context_tiled = context_vec.view(B, -1, 1, 1).expand(-1, -1, H, W)
        conditioned_features = self.history_fusion(
            torch.cat([features, context_tiled], dim=1)
        )
        logits = base_logits.clone()
        logits[:, 1:] = logits[:, 1:] + self.history_type_actor(conditioned_features)
        return base_logits, logits, conditioned_features

    def _get_topk_mask(self, logits, max_edits, action_mask=None):
        if self.is_crafter or self.is_bipedal:
            return self._get_legacy_topk_mask(logits, max_edits, action_mask)

        probs = F.softmax(logits, dim=1)
        prob_change = 1.0 - probs[:, 0, :, :]
        
        # KEY BUG FIX: Force masked pixels (like boundary walls) to have 0 prob of change 
        # so they don't consume the top-k budget when logits are uniformly initialized!
        if action_mask is not None:
             prob_change = prob_change.masked_fill(action_mask.squeeze(1) > 0.5, -1.0)
             
        B, H, W = prob_change.shape
        flat_mask = torch.zeros_like(prob_change, dtype=torch.bool).view(B, -1)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=logits.device, dtype=logits.dtype),
            torch.arange(W, device=logits.device, dtype=logits.dtype),
            indexing="ij",
        )
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
        sigma_sq = self.spatial_dpp_sigma ** 2

        for batch_idx in range(B):
            editable = torch.ones((H, W), dtype=torch.bool, device=logits.device)
            editable[[0, -1], :] = False
            editable[:, [0, -1]] = False
            if action_mask is not None:
                editable &= action_mask[batch_idx, 0] < 0.5
            candidate_indices = torch.where(editable.flatten())[0]
            if candidate_indices.numel() == 0:
                continue

            k = min(
                max(0, int(round(float(max_edits) * candidate_indices.numel()))),
                candidate_indices.numel(),
            )
            if k == 0:
                continue
            quality = prob_change[batch_idx].flatten()[candidate_indices].clamp_min(1e-6)
            candidate_coords = coords[candidate_indices]
            distances = torch.cdist(candidate_coords, candidate_coords).pow(2)
            similarity = torch.exp(-distances / (2.0 * sigma_sq))
            quality_root = quality.sqrt()
            kernel = quality_root[:, None] * similarity * quality_root[None, :]

            selected = []
            remaining = list(range(candidate_indices.numel()))
            for _ in range(k):
                best_local = None
                best_logdet = -float("inf")
                for local_idx in remaining:
                    trial = selected + [local_idx]
                    sub_kernel = kernel[trial][:, trial]
                    sub_kernel = sub_kernel + torch.eye(
                        len(trial), device=kernel.device, dtype=kernel.dtype
                    ) * 1e-6
                    sign, logdet = torch.linalg.slogdet(sub_kernel)
                    score = float(logdet) if float(sign) > 0 else -float("inf")
                    if score > best_logdet:
                        best_local = local_idx
                        best_logdet = score
                if best_local is None:
                    best_local = max(
                        remaining,
                        key=lambda idx: float(quality[idx]),
                    )
                selected.append(best_local)
                remaining.remove(best_local)

            flat_mask[batch_idx, candidate_indices[selected]] = True
        return flat_mask.view(B, H, W)

    def _get_legacy_topk_mask(self, logits, max_edits, action_mask=None):
        probs = F.softmax(logits, dim=1)
        prob_change = 1.0 - probs[:, 0, :, :]
        if action_mask is not None:
            prob_change = prob_change.masked_fill(
                action_mask.squeeze(1) > 0.5, -1.0
            )
        B, H, W = prob_change.shape
        flat_probs = prob_change.view(B, -1)
        if action_mask is not None:
            num_cells = (action_mask < 0.5).sum(dim=(1, 2, 3))
        else:
            num_cells = torch.full(
                (B,), H * W, device=logits.device, dtype=torch.long
            )
        flat_mask = torch.zeros_like(flat_probs, dtype=torch.bool)
        for batch_idx in range(B):
            editable_count = int(num_cells[batch_idx].item())
            k = max(1, min(int(round(float(max_edits) * editable_count)), H * W))
            _, topk_indices = torch.topk(flat_probs[batch_idx], k=k)
            flat_mask[batch_idx, topk_indices] = True
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
        if self.is_minigrid:
            placement_logits, logits, critic_features = self._minigrid_logits(
                features, context_vec
            )
        else:
            placement_logits = logits = self.actor(features)
            critic_features = features
        topk_mask = self._get_topk_mask(placement_logits, max_edits, action_mask)
        logits[:, 0, :, :].masked_fill_(~topk_mask, 1e9)
        logits[:, 1:, :, :].masked_fill_(~topk_mask.unsqueeze(1), -1e9)
        dist = Categorical(logits=logits.permute(0, 2, 3, 1))
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # B. Stats (32 Piano Buttons) Sampling
        # Only Crafter edits inventory statistics. MiniGrid and Bipedal do not
        # consume stats actions, so their policy likelihood must not contain an
        # unrelated 32-action branch.
        if not self.is_crafter:
            stats_action = torch.zeros(
                (B, self.num_stats_slots), device=map_vec.device, dtype=torch.long
            )
            stats_logprob = torch.zeros(B, device=map_vec.device, dtype=logits.dtype)
            topk_stats_mask = torch.zeros(
                (B, self.num_stats_slots), device=map_vec.device, dtype=torch.bool
            )
        else:
            # [Fix 1] Use MaxPool instead of AvgPool to capture the presence of sparse objects
            global_vec = F.adaptive_max_pool2d(features, (1, 1)).view(B, -1)
            if stats_heat is None:
                sh_dim = 26 if self.is_bipedal else 16
                stats_heat = torch.zeros(B, sh_dim, device=map_vec.device)
            global_vec = torch.cat([global_vec, stats_heat], dim=1)
            
            logits_stats = self.stats_actor(global_vec).view(B, self.num_stats_slots, self.num_stats_actions)
            topk_stats_mask = self._get_stats_topk_mask(logits_stats, max_stats_edit_ratio)
            # [Fix 2] Remove Hard Masking on logits to fix the DEAD GRADIENT issue.
            # Let PPO learn natively without 1e9 jumping discontinuities.
            # logits_stats[:, :, 0].masked_fill_(~topk_stats_mask, 1e9)
            # logits_stats[:, :, 1].masked_fill_(~topk_stats_mask, -1e9)
            stats_dist = Categorical(logits=logits_stats)
            stats_action = stats_dist.sample()
            stats_logprob = stats_dist.log_prob(stats_action).sum(dim=-1)

        value = self.critic(critic_features)
        return (
            action.detach(),
            stats_action.detach(),
            action_logprob.detach(),
            stats_logprob.detach(),
            value.detach(),
            topk_mask.detach(),
            topk_stats_mask.detach(),
        )

    def evaluate(
        self,
        map_vec,
        context_vec,
        action_tuple,
        action_mask=None,
        target_topk_mask=None,
        target_stats_topk_mask=None,
        stats_heat=None,
    ):
        terrain_action, stats_action = action_tuple
        features = self.forward_features(map_vec, context_vec)
        B, _, H, W = map_vec.shape

        # A. Terrain Eval
        if self.is_minigrid:
            _, logits, critic_features = self._minigrid_logits(features, context_vec)
        else:
            logits = self.actor(features)
            critic_features = features
        if target_topk_mask is not None:
             logits[:, 0, :, :].masked_fill_(~target_topk_mask, 1e9)
             logits[:, 1:, :, :].masked_fill_(~target_topk_mask.unsqueeze(1), -1e9)
        dist = Categorical(logits=logits.permute(0, 2, 3, 1))
        action_logprobs = dist.log_prob(terrain_action)
        if self.is_minigrid and target_topk_mask is not None:
            selected = target_topk_mask.to(dtype=logits.dtype)
            dist_entropy = (dist.entropy() * selected).sum() / selected.sum().clamp_min(1.0)
        else:
            dist_entropy = dist.entropy().mean()

        # B. Stats Eval
        if not self.is_crafter:
            stats_logprobs = torch.zeros(B, device=map_vec.device, dtype=logits.dtype)
            stats_entropy = torch.zeros((), device=map_vec.device, dtype=logits.dtype)
        else:
            # [Fix 1] MaxPool matching the inference code above
            global_vec = F.adaptive_max_pool2d(features, (1, 1)).view(B, -1)
            if stats_heat is None:
                stats_heat_dim = 26 if self.is_bipedal else 16
                stats_heat = torch.zeros(B, stats_heat_dim, device=map_vec.device)
            global_vec = torch.cat([global_vec, stats_heat], dim=1)
            
            logits_stats = self.stats_actor(global_vec).view(B, self.num_stats_slots, self.num_stats_actions)
            # [Fix 2] Removed evaluation target hard masking matching inference
            # if target_stats_topk_mask is not None:
            #     logits_stats[:, :, 0].masked_fill_(~target_stats_topk_mask, 1e9)
            #     logits_stats[:, :, 1].masked_fill_(~target_stats_topk_mask, -1e9)
            stats_dist = Categorical(logits=logits_stats)
            stats_logprobs = stats_dist.log_prob(stats_action).sum(dim=-1)
            stats_entropy = stats_dist.entropy().mean()

        value = self.critic(critic_features)
        total_entropy = dist_entropy + stats_entropy
        return action_logprobs, stats_logprobs, value, total_entropy
