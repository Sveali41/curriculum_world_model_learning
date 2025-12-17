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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. 连通性检查 (Validity - BFS)
# ==========================================
def check_solvability(grid_obj_np, start_pos=(1, 1), goal_id=8):
    """
    使用 BFS 检查是否存在一条【安全】的路径到达终点
    即：不穿墙，也不踩岩浆
    """
    height, width = grid_obj_np.shape
    
    # 获取 ID 常量
    WALL = OBJECT_TO_IDX['wall']
    LAVA = OBJECT_TO_IDX['lava']  # <--- 新增：定义岩浆 ID
    
    queue = deque([start_pos])
    visited = set([start_pos])
    
    while queue:
        r, c = queue.popleft()
        
        # 找到终点
        if grid_obj_np[r, c] == goal_id:
            return True
            
        # 上下左右
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < height and 0 <= nc < width:
                if (nr, nc) not in visited:
                    cell_type = grid_obj_np[nr, nc]
                    
                    # === 关键修改 ===
                    # 只有当它既不是墙，也不是岩浆时，才允许通过
                    # (门 Door 和 钥匙 Key 依然视为通路)
                    if cell_type != WALL and cell_type != LAVA:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
                        
    return False

# ==========================================
# 2. 多样性打分 (Diversity - RND + Archive)
# ==========================================
class DiversityModule:
    def __init__(self, archive_size=1000, k=10, input_channels=3):
        # 1. 固定随机 CNN (Random Encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64) # 假设输入 15x15，池化后约 7x7
        ).to(device).eval()
        
        # 冻结参数 (这是一把固定的尺子)
        for p in self.encoder.parameters():
            p.requires_grad = False
            
        self.archive = [] # 存 Embedding (numpy arrays)
        self.max_size = archive_size
        self.k = k

    def get_reward(self, map_vec_tensor):
        """
        map_vec_tensor: [1, 3, 15, 15] (Minigrid Vectorized Obs)
        返回: float reward
        """
        with torch.no_grad():
            # 提取特征 [64]
            emb = self.encoder(map_vec_tensor.float()).cpu().numpy().flatten()
        
        # 如果档案库太小，先给 0 分
        if len(self.archive) < self.k:
            reward = 0.0
        else:
            # 计算与 Archive 中所有点的距离
            # archive_matrix: [N, 64]
            archive_matrix = np.stack(self.archive)
            dists = np.linalg.norm(archive_matrix - emb, axis=1)
            
            # 找到最近的 k 个邻居 (KNN)
            dists.sort()
            nearest_k = dists[:self.k]
            
            # 距离越远，分数越高 (新颖性)
            reward = np.mean(nearest_k)
            
        # 更新档案库 (FIFO)
        self.archive.append(emb)
        if len(self.archive) > self.max_size:
            self.archive.pop(0)
            
        return reward

def calculate_lp_reward(world_model, trajectory_data, lr=1e-3):
    """
    计算 Head-only Learning Progress.
    只更新模型的 '预测头' 部分进行影子测试，以此估算学习潜力。
    """
    
    # === 1. 保存当前状态 (Snapshot) ===
    # 我们不仅要回滚参数，还要回滚 requires_grad 的状态
    original_state = {k: v.clone() for k, v in world_model.state_dict().items()}
    
    # === 2. 冻结 Backbone，只解冻 Head (关键步骤) ===
    params_to_update = []
    
    for name, param in world_model.named_parameters():
        # 这里需要根据你 World Model 的实际层名称修改关键字
        # 通常预测头会包含 'head', 'decoder', 'predictor', 'fc' 等字眼
        # 而卷积层通常叫 'encoder', 'cnn', 'backbone'
        if any(key in name for key in ['head', 'decoder', 'predictor', 'fc_out']):
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
            
    # 如果没找到任何 Head 参数（防止名字不匹配导致空列表）
    if len(params_to_update) == 0:
        print("Warning: No head parameters found for Shadow Update! Check layer names.")
        # 降级方案：更新所有参数 (慢，但安全)
        for param in world_model.parameters():
            param.requires_grad = True
            params_to_update.append(param)

    # === 3. 创建临时优化器 ===
    # 对于单步更新，SGD 比 Adam 更轻量且足够有效
    temp_optimizer = torch.optim.SGD(params_to_update, lr=lr)

    # === 4. 摸底 (Loss Before) ===
    # 此时 backward 只会计算 Head 的梯度，非常快
    loss_before = world_model.calc_loss(trajectory_data)
    
    # === 5. 影子更新 (Shadow Update) ===
    temp_optimizer.zero_grad()
    loss_before.backward()
    temp_optimizer.step()
    
    # === 6. 复试 (Loss After) ===
    with torch.no_grad():
        loss_after = world_model.calc_loss(trajectory_data)
        
    # === 7. 恢复现场 (Restore) ===
    # A. 恢复参数数值
    world_model.load_state_dict(original_state)
    # B. 恢复所有参数为可训练状态 (为接下来的主循环训练做准备)
    for param in world_model.parameters():
        param.requires_grad = True
    
    lp_reward = loss_before.item() - loss_after.item()
    
    # 过滤掉极小的负值 (误差波动)
    return max(0.0, lp_reward)