import torch
import torch.nn as nn
import torch.nn.functional as F
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

from generator.crafter_env_designer import CRAFTER_OBJ_MAP

class HistoryEncoder(nn.Module):
    '''
    History encoder that maps past state grids, error heatmaps, and
    auxiliary features into a global context vector for the generator.

    Design:
    1. Embed object, color, and state IDs.
    2. Extract spatial features with a CNN backbone.
    3. Use adaptive max pooling to capture the most salient local features.
    4. Project the combined representation into the global context space.
    '''

    def __init__(self, context_dim=64, emb_dim=16, env_type="minigrid"):
        super().__init__()
        self.env_type = env_type

        if self.env_type == "bipedalwalker":
            self.layout_emb = nn.Embedding(10, emb_dim)
            in_channels = emb_dim + 1 + 2
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.local_fc = nn.Sequential(
                nn.Linear(64, 64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
            )
            self.global_fc = nn.Sequential(
                nn.Linear(26, 64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
            )
            self.fc = nn.Sequential(
                nn.Linear(64 + 64, 128),
                nn.LayerNorm(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, context_dim),
                nn.ReLU(),
            )
            return

        # 1. Embedding layers
        if self.env_type == "crafter":
            max_object_id = max(CRAFTER_OBJ_MAP.values())
            max_color_id = 0
            max_cell_state_id = 0
        else:
            max_object_id = max(OBJECT_TO_IDX.values())
            max_color_id = max(COLOR_TO_IDX.values())
            max_cell_state_id = max(STATE_TO_IDX.values())
        
        self.emb_object = nn.Embedding(max_object_id + 1, emb_dim)
        self.emb_color = nn.Embedding(max_color_id + 1, emb_dim)
        self.emb_cell_state = nn.Embedding(max_cell_state_id + 1, emb_dim)

        # 2. Channel count: embeddings + heatmap + CoordConv channels
        in_channels = emb_dim * 3 + 1 + 2

        # 3. CNN Backbone
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 4. Adaptive max pooling emphasizes the most salient local failures.
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        # 5. Output projection with fused inventory error features.
        self.fc = nn.Sequential(
            nn.Linear(64 + 16, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, context_dim),
            nn.ReLU()
        )

    def _add_coords(self, x):
        B, _, H, W = x.shape
        yy = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, xx, yy], dim=1)

    def forward(self, state_grid, error_heatmap, stats_error=None):
        """
        state_grid: [B, 3, H, W]
        error_heatmap: [B, 1, H, W]
        stats_error: [B, 16] (Inventory errors)
        """
        B, _, H, W = state_grid.shape

        if self.env_type == "bipedalwalker":
            # For BipedalWalker, the active layout replaces the usual spatial heatmap.
            # `stats_error[:, 0:10]` tracks physical error and `stats_error[:, 10:20]`
            # tracks terrain-related semantic error.
            layout_ids = state_grid[:, 0].long().clamp_min(0).clamp_max(9)
            feat_layout = self.layout_emb(layout_ids).permute(0, 3, 1, 2)
            x = torch.cat([feat_layout, error_heatmap], dim=1)
            x = self._add_coords(x)
            x = self.net(x)
            spatial_features = self.pool(x).flatten(1)
            if stats_error is None:
                stats_error = torch.zeros(B, 26, device=state_grid.device)
            local_ctx = self.local_fc(spatial_features)
            global_ctx = self.global_fc(stats_error)
            combined = torch.cat([local_ctx, global_ctx], dim=1)
            return self.fc(combined)
        
        # 1. Embedding lookup
        feat_obj = self.emb_object(state_grid[:, 0].long()).permute(0, 3, 1, 2)
        feat_col = self.emb_color(state_grid[:, 1].long()).permute(0, 3, 1, 2)
        feat_sta = self.emb_cell_state(state_grid[:, 2].long()).permute(0, 3, 1, 2)

        # 2. Concatenate spatial features
        x = torch.cat([feat_obj, feat_col, feat_sta, error_heatmap], dim=1)
        x = self._add_coords(x)

        # 3. Extract CNN features and pool
        x = self.net(x)
        spatial_features = self.pool(x).flatten(1) # [B, 64]

        # 4. Fuse inventory-error features
        if stats_error is None:
            stats_error = torch.zeros(B, 16, device=state_grid.device)
        
        # Concat spatial fail patterns + stats fail patterns
        combined = torch.cat([spatial_features, stats_error], dim=1) # [B, 80]

        # 5. Project into the global context space
        context = self.fc(combined) # [B, context_dim]
        return context
