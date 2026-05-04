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
# 1. Solvability check (Validity - BFS)
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
# 2. Diversity scoring (RND + archive)
# ==========================================
class DiversityModule(nn.Module):
    def __init__(self, input_h=15, input_w=15, k=10, max_archive_size=1000, device=None, env_type='minigrid'):
        super().__init__()
        self.k = k
        self.max_size = max_archive_size
        self.archive = []
        self.env_type = env_type
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # === 1. Class definitions and feature heads ===
        if self.env_type == 'crafter':
            self.num_obj_types = 25 # Crafter map elements (up to 16, leaving room)
            self.num_colors = 5    # Crafter player directions (0..4)
            self.inv_size = 16     # Crafter inventory stats
            # Inventory Encoder (Stats Path)
            self.inv_encoder = nn.Sequential(
                nn.Linear(self.inv_size, 16),
                nn.ReLU()
            ).to(self.device)
            # Joint Embedding Dim: 64 (Map) + 16 (Inv) = 80
            self.joint_dim = 64 + 16
        elif self.env_type == 'bipedalwalker':
            self.num_obj_types = 10
            self.num_colors = 0
            self.inv_encoder = None
            self.joint_dim = 64
        else:
            # MiniGrid
            self.num_obj_types = 11 
            self.num_colors = 6
            self.inv_encoder = None
            self.joint_dim = 64

        # Input channels = object classes + color/direction classes.
        input_channels = self.num_obj_types + self.num_colors 
        
        # === 2. Map encoder (CNN or MLP) ===
        if self.env_type == 'bipedalwalker':
            # Bipedal is just 1x5 with 10 one-hot classes = 50 dims. MLP is enough.
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_channels * input_h * input_w, 32),
                nn.ReLU(),
                nn.Linear(32, 64)
            ).to(self.device)
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * input_h * input_w, 64) 
            ).to(self.device)

        for m in self.encoder:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _preprocess(self, map_tensor):
        """
        [1, 2, H, W] -> [1, Oh_Obj+Oh_Col, H, W]
        """
        x = map_tensor 
        obj_ids = x[:, 0, :, :].long()
        col_ids = x[:, 1, :, :].long()
        
        # Mask Agent to avoid calculating diversity based on random spawn position
        obj_ids_clean = obj_ids.clone()
        if self.env_type == 'crafter':
            obj_ids_clean[obj_ids_clean == 9] = 0 # Crafter Agent (9) -> Grass (0)
        elif self.env_type != 'bipedalwalker':
            obj_ids_clean[obj_ids_clean == 10] = 1 # MiniGrid Agent (10) -> Empty (1)

        # One-hot encoding. `F.one_hot` returns `[B, H, W, N]`.
        obj_oh = F.one_hot(obj_ids_clean, num_classes=self.num_obj_types).permute(0, 3, 1, 2).float()
        
        if self.env_type == 'bipedalwalker':
            return obj_oh
            
        col_oh = F.one_hot(col_ids, num_classes=self.num_colors).permute(0, 3, 1, 2).float()
        
        return torch.cat([obj_oh, col_oh], dim=1)

    def get_reward(self, map_vec_tensor, inventory_vec=None):
        """
        Inputs:
            map_vec_tensor: [1, 2, H, W] 
            inventory_vec: [1, 16] (Numpy or Tensor)
        Output: float
        """
        map_vec_tensor = map_vec_tensor.to(self.device)
        
        with torch.no_grad():
            # 1. Map feature [1, 64]
            x = self._preprocess(map_vec_tensor)
            emb_map = self.encoder(x) # [1, 64]
            
            # 2. Inventory feature [1, 16] (optional)
            if self.env_type == 'crafter' and inventory_vec is not None:
                if not isinstance(inventory_vec, torch.Tensor):
                    inventory_vec = torch.from_numpy(inventory_vec).float().to(self.device)
                if inventory_vec.dim() == 1:
                    inventory_vec = inventory_vec.unsqueeze(0)
                emb_inv = self.inv_encoder(inventory_vec) # [1, 16]
                emb_raw = torch.cat([emb_map, emb_inv], dim=1) # [1, 80]
            else:
                emb_raw = emb_map # [1, 64]
            
            # 3. Normalize the joint embedding on the unit hypersphere.
            norm = torch.norm(emb_raw, p=2, dim=1, keepdim=True)
            emb = (emb_raw / (norm + 1e-8)).cpu().numpy().flatten()
            
        # 4. KNN distance for novelty estimation
        if len(self.archive) == 0:
             reward = 0.0
        else:
            archive_matrix = np.stack(self.archive)
            dists = np.linalg.norm(archive_matrix - emb, axis=1)
            current_k = min(len(self.archive), self.k)
            dists.sort()
            nearest_k = dists[:current_k]
            reward = np.mean(nearest_k)
            
        # 5. Update the archive (FIFO)
        self.archive.append(emb)
        if len(self.archive) > self.max_size:
            self.archive.pop(0)
            
        return float(reward)

def calculate_lp_reward(world_model, trajectory_data, lr=1e-3):
    """
    Compute head-only learning progress (LP).

    Core idea:
    - apply a temporary update only to the prediction head
    - measure `loss_before - loss_after` as the learning signal
    - restore all parameters before returning so the main model is unchanged
    """

    import torch

    # ============================================================
    # 1. Snapshot parameter values directly instead of using `state_dict()`.
    # ============================================================
    original_params = {
        name: param.detach().clone()
        for name, param in world_model.named_parameters()
    }

    # ============================================================
    # 2. Freeze the backbone and update only the head.
    # ============================================================
    params_to_update = []

    for name, param in world_model.named_parameters():
        if any(key in name for key in ['head', 'decoder', 'predictor', 'fc_out']):
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    # Fallback in case no head parameters are matched.
    if len(params_to_update) == 0:
        print("[LP Warning] No head parameters found, fallback to all parameters.")
        for param in world_model.parameters():
            param.requires_grad = True
            params_to_update.append(param)

    # ============================================================
    # 3. Temporary optimizer with a single SGD step.
    # ============================================================
    temp_optimizer = torch.optim.SGD(params_to_update, lr=lr)

    # ============================================================
    # 4. Loss before the temporary update
    # ============================================================
    loss_before = world_model.calc_loss(trajectory_data)

    # ============================================================
    # 5. Temporary head-only update
    # ============================================================
    temp_optimizer.zero_grad(set_to_none=True)
    loss_before.backward()
    temp_optimizer.step()

    # ============================================================
    # 6. Loss after the temporary update
    # ============================================================
    with torch.no_grad():
        loss_after = world_model.calc_loss(trajectory_data)

    # ============================================================
    # 7. Restore parameter values and `requires_grad` flags
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

    # Clamp tiny negative values caused by numerical noise.
    return max(0.0, lp_reward)
