import torch
import torch.nn as nn
import torch.nn.functional as F

class HistoryEncoder(nn.Module):
    def __init__(self, 
                 context_dim=64,    # 输出向量的维度
                 max_obj_id=15, 
                 max_color_id=6, 
                 max_state_id=3):
        super().__init__()

        # === 1. Embedding Layers ===
        # 将离散的地图 ID 转换为特征向量
        self.emb_dim = 16
        self.emb_obj = nn.Embedding(max_obj_id + 1, self.emb_dim)
        self.emb_col = nn.Embedding(max_color_id + 1, self.emb_dim)
        self.emb_sta = nn.Embedding(max_state_id + 1, self.emb_dim)

        # === 2. 计算输入通道数 ===
        # MapEmbeddings(16*3) + Heatmap(1) + Attention(1)
        in_channels = (self.emb_dim * 3) + 1 + 1
        
        # === 3. CNN Backbone ===
        # 目标：提取"错误模式" (例如: 红色门+高误差)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            # 这里的层数不用太深，主要是提取局部关联
        )

        # === 4. 压缩层 (Global Pooling) ===
        # 关键步骤：把 HxW 的空间信息拍扁
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # === 5. 输出映射 ===
        self.fc = nn.Linear(64, context_dim)

    def forward(self, map_vec, heatmap, attention):
        """
        输入是上一轮 Batch 的数据
        map_vec:   [B, 3, H, W] (Long)
        heatmap:   [B, 1, H, W] (Float)
        attention: [B, 1, H, W] (Float)
        """
        # 1. Embedding
        feat_obj = self.emb_obj(map_vec[:, 0].long()).permute(0, 3, 1, 2)
        feat_col = self.emb_col(map_vec[:, 1].long()).permute(0, 3, 1, 2)
        feat_sta = self.emb_state(map_vec[:, 2].long()).permute(0, 3, 1, 2)
        
        # 2. Concat (绑定空间关系)
        x = torch.cat([feat_obj, feat_col, feat_sta, heatmap, attention], dim=1)
        
        # 3. 卷积提取
        x = self.net(x)
        
        # 4. 全局池化 [B, 64, H, W] -> [B, 64, 1, 1]
        x = self.pool(x)
        x = x.flatten(1) # [B, 64]
        
        # 5. 线性层调整
        context = self.fc(x) # [B, context_dim]
        
        return context