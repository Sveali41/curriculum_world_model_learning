import torch
import torch.nn as nn
import torch.nn.functional as F
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX



class HistoryEncoder(nn.Module):
    """
    Encodes historical environment state, prediction error, and attention maps
    into a fixed-dimensional context vector for history-conditioned generation.
    It's the hostorical information so everything should be in one batch and compacted into one vector.
    """
    def __init__(
        self,
        context_dim=64,    # 输出向量维度
        emb_dim=16,
    ):
        super().__init__()

        # === 1. Embedding Layers ===
        # grid cell 的语义嵌入
        max_object_id = max(OBJECT_TO_IDX.values())
        max_color_id = max(COLOR_TO_IDX.values())
        max_cell_state_id = max(STATE_TO_IDX.values())
        self.emb_dim = emb_dim
        self.emb_object = nn.Embedding(max_object_id + 1, emb_dim)
        self.emb_color = nn.Embedding(max_color_id + 1, emb_dim)
        self.emb_cell_state = nn.Embedding(max_cell_state_id + 1, emb_dim)

        # === 2. 输入通道数 ===
        # object + color + cell_state + error_heatmap + attention
        in_channels = emb_dim * 3 + 1 + 1

        # === 3. CNN Backbone ===
        # 提取局部“失败模式 / 结构关联”
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # === 4. Global Pooling ===
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # === 5. 输出映射 ===
        self.fc = nn.Linear(64, context_dim)

    def forward(self, state_grid, error_heatmap, attention_map):
        """
        输入是上一轮 batch 的历史信息

        state_grid:      [B, 3, H, W] (Long)
            channel 0: object_id
            channel 1: color_id
            channel 2: cell_state_id

        error_heatmap:   [B, 1, H, W] (Float)
        attention_map:   [B, 1, H, W] (Float)
        """

        # === 1. Embedding ===
        feat_object = self.emb_object(state_grid[:, 0].long()).permute(0, 3, 1, 2)
        feat_color = self.emb_color(state_grid[:, 1].long()).permute(0, 3, 1, 2)
        feat_cell_state = self.emb_cell_state(state_grid[:, 2].long()).permute(0, 3, 1, 2)

        # === 2. 拼接空间特征 ===
        x = torch.cat(
            [
                feat_object,
                feat_color,
                feat_cell_state,
                error_heatmap,
                attention_map,
            ],
            dim=1,
        )

        # === 3. 卷积特征提取 ===
        x = self.net(x)

        # === 4. 全局池化 ===
        x = self.pool(x)      # [B, 64, 1, 1]
        x = x.flatten(1)      # [B, 64]

        # === 5. 输出 context 向量 ===
        context = self.fc(x)  # [B, context_dim]

        return context
