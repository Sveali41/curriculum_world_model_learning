import torch
import torch.nn as nn
import torch.nn.functional as F
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

from generator.crafter_env_designer import CRAFTER_OBJ_MAP

class HistoryEncoder(nn.Module):
    '''
    历史状态编码器：将历史状态网格及其对应的错误热图和注意力图编码为全局上下文向量
    用于指导生成器的决策过程
    设计思路：
    1. Embedding 层：分别对对象ID、颜色ID和状态ID
         进行嵌入，捕捉离散特征的语义信息
    2. CNN Backbone：多层卷积网络提取空间特征
    3. 池化层：使用自适应最大池化捕捉局部最显著特征
    4. 输出映射：两层全连接网络，结合 LayerNorm 和 ReLU 激活，
         映射到全局上下文空间，确保非负特征以便 Max-Pooling 逻辑    
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

        # 1. Embedding 层
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

        # 2. 通道计算：Embeddings(16*3) + Heatmap(1) + Attention(1) + CoordConv(2) = 52
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

        # 4. 池化：改用 AdaptiveMaxPool2d 能更敏锐地捕捉“局部最显著的故障特征”
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        # 5. 输出映射：增加 16 维背包误差特征的融合空间
        # 输入：空间特征(64) + 背包误差(16) = 80
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
            # in bipedal, we don't use the spatial error heatmap, 
            # instead we use active map to replace it
            # stats error: stats_error[:, 0:10] is physical error.
            # stats_error[:, 10:20] is terrain score (semantic error).
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
        
        # 1. Embedding 处理
        feat_obj = self.emb_object(state_grid[:, 0].long()).permute(0, 3, 1, 2)
        feat_col = self.emb_color(state_grid[:, 1].long()).permute(0, 3, 1, 2)
        feat_sta = self.emb_cell_state(state_grid[:, 2].long()).permute(0, 3, 1, 2)

        # 2. 拼接空间特征
        x = torch.cat([feat_obj, feat_col, feat_sta, error_heatmap], dim=1)
        x = self._add_coords(x)

        # 3. 提取 CNN 特征并池化
        x = self.net(x)
        spatial_features = self.pool(x).flatten(1) # [B, 64]

        # 4. 融合背包误差特征
        if stats_error is None:
            stats_error = torch.zeros(B, 16, device=state_grid.device)
        
        # Concat spatial fail patterns + stats fail patterns
        combined = torch.cat([spatial_features, stats_error], dim=1) # [B, 80]

        # 5. 映射到 Global Context 空间
        context = self.fc(combined) # [B, context_dim]
        return context
