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
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX
from torch.nn import functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Solvability check (Validity - BFS)
# ==========================================
from collections import deque
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX


def check_solvability(grid_obj_np, color_np=None, state_np=None, inventory_token=0):
    """
    Check whether a MiniGrid map has a safe path from start to goal.

    When color/state maps are supplied, BFS tracks the set of key colours
    reachable by the agent and blocks locked doors unless the matching key is
    available.  The legacy object-only call keeps the original geometric BFS
    for non-MiniGrid callers.
    """
    obj = np.asarray(grid_obj_np)
    H, W = obj.shape
    WALL, LAVA = OBJECT_TO_IDX["wall"], OBJECT_TO_IDX["lava"]
    START, GOAL = OBJECT_TO_IDX["agent"], OBJECT_TO_IDX["goal"]
    starts = np.argwhere(obj == START)
    if len(starts) == 0:
        return False, 0
    start = tuple(int(value) for value in starts[0])

    # Without semantic channels, preserve the original geometry-only check.
    semantic = color_np is not None or state_np is not None
    if not semantic:
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            (r, c), dist = queue.popleft()
            if obj[r, c] == GOAL:
                return True, dist
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in visited or obj[nr, nc] in (WALL, LAVA):
                    continue
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
        return False, 0

    if color_np is None or state_np is None:
        raise ValueError("MiniGrid semantic solvability requires both color_np and state_np")
    colors, states = np.asarray(color_np), np.asarray(state_np)
    if colors.shape != obj.shape or states.shape != obj.shape:
        raise ValueError(
            f"MiniGrid solvability channels must match object map shape {obj.shape}, "
            f"got color={colors.shape}, state={states.shape}"
        )
    owned = frozenset({int(inventory_token) - 1}) if int(inventory_token) > 0 else frozenset()
    queue = deque([(start, owned, 0)])
    visited = {(start, owned)}
    while queue:
        (r, c), owned, dist = queue.popleft()
        if obj[r, c] == GOAL:
            return True, dist
        # Treat reaching a key as the corresponding pickup being available.
        # This is a reachability oracle, so it does not require an exact
        # orientation/action sequence to stand next to the key.
        next_owned = owned
        if obj[r, c] == OBJECT_TO_IDX["key"]:
            next_owned = frozenset(set(owned) | {int(colors[r, c])})
        if next_owned != owned and ((r, c), next_owned) not in visited:
            visited.add(((r, c), next_owned))
            queue.appendleft(((r, c), next_owned, dist))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            position = (nr, nc)
            cell = int(obj[nr, nc])
            if cell in (WALL, LAVA):
                continue
            if (
                cell == OBJECT_TO_IDX["door"]
                and int(states[nr, nc]) == STATE_TO_IDX["locked"]
                and int(colors[nr, nc]) not in next_owned
            ):
                continue
            key = (position, next_owned)
            if key not in visited:
                visited.add(key)
                queue.append((position, next_owned, dist + 1))
    return False, 0


# ==========================================
# 2. Diversity scoring (RND + archive)
# ==========================================
class DiversityModule(nn.Module):
    def __init__(self, input_h=15, input_w=15, k=10, max_archive_size=1000, device=None, env_type='minigrid'):
        super().__init__()
        self.k = k
        self.max_size = max_archive_size
        self.env_type = env_type
        self.last_components = {}
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # === 1. Class definitions and feature heads ===
        if self.env_type == 'crafter':
            self.num_obj_types = 25 # Crafter map elements (up to 16, leaving room)
            self.num_colors = 5    # Crafter player directions (0..4)
            self.inv_size = 16     # Crafter inventory stats
            self.inv_encoder = None
            self.joint_dim = 64
            self.archive = []
        elif self.env_type == 'bipedalwalker':
            self.num_obj_types = 10
            self.num_colors = 0
            self.inv_encoder = None
            self.joint_dim = 64
            self.archive = []
        else:
            # MiniGrid
            self.num_obj_types = 11 
            self.num_colors = 6
            self.inv_encoder = None
            self.joint_dim = 64
            self.archive = []

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

        if self.env_type == 'crafter':
            with torch.no_grad():
                zero_edit = torch.zeros((1, 2, input_h, input_w), dtype=torch.long, device=self.device)
                background = self.encoder(self._one_hot_crafter(zero_edit))
            self.register_buffer("_crafter_zero_edit_embedding", background, persistent=False)

    def _preprocess(self, map_tensor):
        """
        [1, 2, H, W] -> [1, Oh_Obj+Oh_Col, H, W]
        """
        x = map_tensor 
        obj_ids = x[:, 0, :, :].long()
        col_ids = x[:, 1, :, :].long()
        
        obj_ids_clean = obj_ids.clone()
        if self.env_type == 'crafter':
            # Crafter preprocessing excludes the agent from the layout edit.
            raise RuntimeError("Crafter preprocessing requires _crafter_reward")
        elif self.env_type != 'bipedalwalker':
            obj_ids_clean[obj_ids_clean == 10] = 1 # MiniGrid Agent (10) -> Empty (1)

        # One-hot encoding. `F.one_hot` returns `[B, H, W, N]`.
        obj_oh = F.one_hot(obj_ids_clean, num_classes=self.num_obj_types).permute(0, 3, 1, 2).float()
        
        if self.env_type == 'bipedalwalker':
            return obj_oh
            
        col_oh = F.one_hot(col_ids, num_classes=self.num_colors).permute(0, 3, 1, 2).float()
        
        return torch.cat([obj_oh, col_oh], dim=1)

    def _append_fifo(self, archive, representation):
        archive.append(representation)
        if len(archive) > self.max_size:
            archive.pop(0)

    def _crafter_reward(self, map_vec_tensor, inventory_vec, start_position=None, stage_token=None):
        if map_vec_tensor.ndim != 4 or map_vec_tensor.shape[0] != 1 or map_vec_tensor.shape[1] < 2:
            raise ValueError("Crafter diversity expects map tensor [1, >=2, H, W]")
        if inventory_vec is None and stage_token is None:
            raise ValueError("Crafter diversity requires a stage token or a 16-dimensional inventory vector")

        obj_ids = map_vec_tensor[0, 0].long()
        direction_ids = map_vec_tensor[0, 1].long()
        agent_positions = torch.nonzero(obj_ids == 13, as_tuple=False)
        if start_position is None:
            if len(agent_positions) != 1:
                raise ValueError(
                    f"Crafter diversity requires exactly one agent (ID 13), found {len(agent_positions)}"
                )
            y, x = (int(agent_positions[0, 0]), int(agent_positions[0, 1]))
        else:
            y, x = map(int, start_position)
            if len(agent_positions):
                raise ValueError("Crafter effective edit map must exclude the agent")
        height, width = obj_ids.shape
        if not (0 <= y < height and 0 <= x < width):
            raise ValueError(f"Crafter start position {(y, x)} is outside the map")
        map_obj = obj_ids.clone()
        map_dir = direction_ids.clone()
        if start_position is None:
            map_obj[y, x] = 2  # Legacy full-map diversity path.
            map_dir[y, x] = 0
        map_input = torch.stack([map_obj, map_dir]).unsqueeze(0)

        if stage_token is None:
            if not isinstance(inventory_vec, torch.Tensor):
                inventory_vec = torch.as_tensor(inventory_vec, dtype=torch.float32)
            inventory_vec = inventory_vec.to(self.device, dtype=torch.float32).reshape(-1)
            if inventory_vec.numel() != self.inv_size:
                raise ValueError(
                    f"Crafter diversity requires inventory with {self.inv_size} values, got {inventory_vec.numel()}"
                )
            inventory_representation = torch.clamp(inventory_vec, 0.0, 9.0).div(9.0).cpu().numpy()
        else:
            token = int(stage_token)
            if not 0 <= token <= 5:
                raise ValueError(f"Crafter stage token must be 0..5, got {token}")
            # KEEP and STAGE_0 both represent the empty item inventory.
            inventory_representation = np.eye(5, dtype=np.float32)[max(token - 1, 0)]

        with torch.no_grad():
            emb_map = self.encoder(self._one_hot_crafter(map_input))
            edit_embedding = emb_map - self._crafter_zero_edit_embedding
            map_representation = F.normalize(edit_embedding, p=2, dim=1).cpu().numpy().flatten()
        # Keep the no-edit map distinct while giving every map block unit norm.
        map_representation = np.append(
            map_representation, float(not torch.any(map_input))
        ).astype(np.float32)

        # One representation for the layout edit and inventory stage pair.
        # The random agent start and sampled item quantities do not enter it.
        joint_representation = np.concatenate(
            [map_representation, inventory_representation]
        ).astype(np.float32)
        joint_representation /= max(float(np.linalg.norm(joint_representation)), 1e-8)
        if not self.archive:
            total = 0.0
        else:
            joint_distances = np.linalg.norm(
                np.stack(self.archive) - joint_representation, axis=1
            )
            total = float(np.mean(np.sort(joint_distances)[:min(len(joint_distances), self.k)]))
        self.last_components = {"total_novelty": total}
        self._append_fifo(self.archive, joint_representation)
        return float(total)

    def _one_hot_crafter(self, map_tensor):
        obj_oh = F.one_hot(map_tensor[:, 0].long(), num_classes=self.num_obj_types).permute(0, 3, 1, 2).float()
        direction_oh = F.one_hot(map_tensor[:, 1].long(), num_classes=self.num_colors).permute(0, 3, 1, 2).float()
        return torch.cat([obj_oh, direction_oh], dim=1)

    def get_reward(self, map_vec_tensor, inventory_vec=None, start_position=None, stage_token=None):
        """
        Inputs:
            map_vec_tensor: [1, 2, H, W]
            inventory_vec: optional 16-value vector for legacy callers
            stage_token: Crafter inventory stage action (0..5)
        Output: novelty distance to the nearest archived environments
        """
        map_vec_tensor = map_vec_tensor.to(self.device)
        if self.env_type == 'crafter':
            return self._crafter_reward(map_vec_tensor, inventory_vec, start_position, stage_token)
        
        with torch.no_grad():
            # 1. Map feature [1, 64]
            x = self._preprocess(map_vec_tensor)
            emb_map = self.encoder(x) # [1, 64]
            
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
