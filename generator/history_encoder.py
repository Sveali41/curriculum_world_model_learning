import torch
import torch.nn as nn
import torch.nn.functional as F
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

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

    def __init__(self, context_dim=64, emb_dim=16):
        super().__init__()

        # 1. Embedding 层
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

        # 5. 输出映射：加入 LayerNorm 和两层 MLP
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),   # 保证不同样本间特征量级可比，对后续 Max 聚合至关重要
            nn.ReLU(inplace=True),
            nn.Linear(64, context_dim),
            nn.ReLU()           # 核心：确保非负，用于 Max-Pooling 并集逻辑
        )

    def _add_coords(self, x):
        B, _, H, W = x.shape
        yy = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, xx, yy], dim=1)

    def forward(self, state_grid, error_heatmap):
        # Embedding 处理
        feat_obj = self.emb_object(state_grid[:, 0].long()).permute(0, 3, 1, 2)
        feat_col = self.emb_color(state_grid[:, 1].long()).permute(0, 3, 1, 2)
        feat_sta = self.emb_cell_state(state_grid[:, 2].long()).permute(0, 3, 1, 2)

        # 拼接特征
        x = torch.cat([feat_obj, feat_col, feat_sta, error_heatmap], dim=1)
        x = self._add_coords(x)

        # 提取空间特征
        x = self.net(x)
        x = self.pool(x).flatten(1) # [B, 64]

        # 映射到 Context 空间
        context = self.fc(x) # [B, context_dim]
        return context