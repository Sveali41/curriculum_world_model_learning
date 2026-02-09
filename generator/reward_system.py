'''
A module implementing reward systems for environment generation.
Reward = Validity + Diversity + Learning Progress
- Validity: Check if the generated environment is solvable.
- Diversity: Reward based on novelty using Random Network Distillation (RND) and an archive of past environments.
- Learning Progress: Reward based on the improvement in the world model's prediction loss.
'''

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from minigrid.core.constants import OBJECT_TO_IDX
from torch.nn import functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. 连通性检查 (Validity - BFS)
# ==========================================
from collections import deque
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX


def check_solvability(grid_obj_np):
    """
    Check whether there exists a safe path from start to goal using BFS.
    A valid path:
      - does NOT pass through walls
      - does NOT step on lava
      - doors and keys are treated as passable

    grid_obj_np: np.ndarray of shape [H, W], containing object IDs
    return: bool
    """

    H, W = grid_obj_np.shape

    # Object IDs
    WALL  = OBJECT_TO_IDX["wall"]
    LAVA  = OBJECT_TO_IDX["lava"]
    START = OBJECT_TO_IDX["agent"]
    GOAL  = OBJECT_TO_IDX["goal"]

    # ------------------------------------------------
    # 1. Find start position automatically
    # ------------------------------------------------
    start_positions = np.argwhere(grid_obj_np == START)
    if len(start_positions) == 0:
        return False  # no start → invalid map

    start_pos = tuple(start_positions[0])  # (row, col)

    # ------------------------------------------------
    # 2. BFS
    # ------------------------------------------------
    # ------------------------------------------------
    # 2. BFS
    # ------------------------------------------------
    # Queue stores: (row, col, distance)
    queue = deque([(start_pos[0], start_pos[1], 0)])
    visited = set([start_pos])
    max_dist = 0

    while queue:
        r, c, dist = queue.popleft()
        max_dist = max(max_dist, dist)

        # reached goal
        if grid_obj_np[r, c] == GOAL:
            # Found shortest path to goal
            return True, dist

        # 4-neighborhood
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < H and 0 <= nc < W:
                if (nr, nc) in visited:
                    continue

                cell = grid_obj_np[nr, nc]

                # block walls & lava
                if cell == WALL or cell == LAVA:
                    continue

                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))

    return False, 0


# ==========================================
# 2. 多样性打分 (Diversity - RND + Archive)
# ==========================================
class DiversityModule(nn.Module):
    def __init__(self, input_h=15, input_w=15, k=10, max_archive_size=1000, device='cuda'):
        super().__init__()
        self.k = k
        self.max_size = max_archive_size
        self.archive = [] 
        self.device = device
        
        # === 1. 定义 One-Hot 的类别数 ===
        # Minigrid 通常 Object ID 最大约 11-13，Color ID 最大约 6
        # 根据你的 tensor 数据，至少要有 11 (因为看到了 ID 10)
        self.num_obj_types = 11 
        self.num_colors = 6
        
        # 输入通道数 = 物体类别数 + 颜色类别数
        input_channels = self.num_obj_types + self.num_colors 
        
        # === 2. 定义编码器 CNN ===
        self.encoder = nn.Sequential(
            # 输入: [1, 17, H, W]
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            # 展平后维度: 32 * H * W
            nn.Linear(32 * input_h * input_w, 64) 
        ).to(device)

        # [NEW] Orthogonal Initialization for better feature extraction sensitivity
        for m in self.encoder:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _preprocess(self, map_tensor):
        """
        将 [2, H, W] 的 ID 地图转换为 [1, 17, H, W] 的 One-Hot 特征图
        """
        # 1. 增加 Batch 维度: [2, H, W] -> [1, 2, H, W]
        x = map_tensor 
        
        # 2. 分离通道
        obj_ids = x[:, 0, :, :].long() # [1, H, W]
        col_ids = x[:, 1, :, :].long() # [1, H, W]
        
        # 3. One-Hot 编码
        # [FIX] Mask out Agent (ID=10) to Empty/Floor (ID=1)
        # Avoid calculating diversity based on random agent spawn position
        obj_ids_clean = obj_ids.clone()
        obj_ids_clean[obj_ids_clean == 10] = 1

        # obj_ids -> [1, H, W, 11] -> permute -> [1, 11, H, W]
        obj_oh = F.one_hot(obj_ids_clean, num_classes=self.num_obj_types).permute(0, 3, 1, 2).float()
        
        # col_ids -> [1, H, W, 6] -> permute -> [1, 6, H, W]
        col_oh = F.one_hot(col_ids, num_classes=self.num_colors).permute(0, 3, 1, 2).float()
        
        # 4. 拼接: [1, 11+6, H, W]
        return torch.cat([obj_oh, col_oh], dim=1)

    def get_reward(self, map_vec_tensor):
        """
        输入: map_vec_tensor [2, H, W] (单个地图)
        输出: float (标量奖励)
        """
        # 确保数据在正确的设备上
        map_vec_tensor = map_vec_tensor.to(self.device)
        
        with torch.no_grad():
            # 1. 预处理 (One-Hot + Unsqueeze)
            x = self._preprocess(map_vec_tensor) # [1, 17, H, W]
            
            # 2. 编码提取特征
            emb_raw = self.encoder(x) # [1, 64]
            # [NEW] Feature Normalization (Unit Norm)
            # This ensures Euclidean distance is bounded [0, 2] and highly sensitive to direction
            norm = torch.norm(emb_raw, p=2, dim=1, keepdim=True)
            emb = (emb_raw / (norm + 1e-8)).cpu().numpy().flatten() # [64]
            
        # 3. KNN 距离计算 (新颖性)
        if len(self.archive) == 0:
             # Archive is empty (first step), so no novelty can be computed relative to history.
             # Return 0.0. The archive will be populated immediately after this.
             reward = 0.0
        else:
            archive_matrix = np.stack(self.archive)
            # 计算当前 emb 与 archive 中所有点的欧氏距离
            dists = np.linalg.norm(archive_matrix - emb, axis=1)
            
            # Use dynamic K if archive is small
            # This solves the "Cold Start" problem where first K samples got 0 reward
            current_k = min(len(self.archive), self.k)
            
            # 取最近的 k 个
            dists.sort()
            nearest_k = dists[:current_k]
            
            # 平均距离越大，说明越新颖
            reward = np.mean(nearest_k)
            
        # 4. 更新档案 (FIFO)
        self.archive.append(emb)
        if len(self.archive) > self.max_size:
            self.archive.pop(0)
            
        return float(reward)

def calculate_lp_reward(world_model, trajectory_data, lr=1e-3):
    """
    计算 Head-only Learning Progress (LP).

    核心思想：
    - 只对预测头（head / decoder 等）做一次“影子更新”
    - 测量 loss_before - loss_after 作为学习潜力
    - 所有参数在函数结束时都会被完整恢复，不污染主训练
    """

    import torch

    # ============================================================
    # 1. Snapshot：只保存【参数数值】，不使用 state_dict()
    #    （避免 Lightning / ShardedTensor 的 hook 问题）
    # ============================================================
    original_params = {
        name: param.detach().clone()
        for name, param in world_model.named_parameters()
    }

    # ============================================================
    # 2. 冻结 Backbone，只解冻 Head
    # ============================================================
    params_to_update = []

    for name, param in world_model.named_parameters():
        if any(key in name for key in ['head', 'decoder', 'predictor', 'fc_out']):
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    # 兜底：如果没找到 head（名字不匹配）
    if len(params_to_update) == 0:
        print("[LP Warning] No head parameters found, fallback to all parameters.")
        for param in world_model.parameters():
            param.requires_grad = True
            params_to_update.append(param)

    # ============================================================
    # 3. 临时优化器（Shadow Optimizer）
    #    单步 SGD，避免 optimizer state snapshot
    # ============================================================
    temp_optimizer = torch.optim.SGD(params_to_update, lr=lr)

    # ============================================================
    # 4. Loss Before（更新前）
    # ============================================================
    loss_before = world_model.calc_loss(trajectory_data)

    # ============================================================
    # 5. Shadow Update（只更新 head）
    # ============================================================
    temp_optimizer.zero_grad(set_to_none=True)
    loss_before.backward()
    temp_optimizer.step()

    # ============================================================
    # 6. Loss After（更新后）
    # ============================================================
    with torch.no_grad():
        loss_after = world_model.calc_loss(trajectory_data)

    # ============================================================
    # 7. Restore：恢复所有参数数值 + requires_grad 状态
    # ============================================================
    with torch.no_grad():
        for name, param in world_model.named_parameters():
            param.copy_(original_params[name])

    for param in world_model.parameters():
        param.requires_grad = True

    # ============================================================
    # 8. LP Reward
    # ============================================================
    lp_reward = loss_before.item() - loss_after.item()

    # 过滤数值抖动导致的极小负值
    return max(0.0, lp_reward)
