import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PLAYER_ID = 13  # Updated: player ID in new CustomCrafterEnv mapping (was 10)


def _to_nchw(obs: np.ndarray) -> np.ndarray:
    """Convert observations to (N, C, H, W)."""
    if obs.ndim != 4:
        raise ValueError(f"Expected 4D observations, got shape {obs.shape}")

    # Already (N, C, H, W)
    # Check if channel is obviously dim 1 (e.g. 2 or 3 channels vs H/W > 3, or explicitly fewer channels than width)
    if obs.shape[1] <= 3 or obs.shape[1] < obs.shape[-1]:
        return obs

    # Usually Crafter saved as (N, H, W, C)
    if obs.shape[-1] <= 8:
        return np.moveaxis(obs, -1, 1)

    raise ValueError(f"Cannot infer channel axis for shape {obs.shape}")


def extract_player_positions(obs: np.ndarray, player_id: int = PLAYER_ID) -> np.ndarray:
    """
    Return array of (y, x) positions for each frame.
    If player not found in a frame, returns (-1, -1) for that frame.
    """
    obs_nchw = _to_nchw(obs)
    obj_map = obs_nchw[:, 0, :, :]  # object-id channel

    positions = np.full((obj_map.shape[0], 2), -1, dtype=np.int32)
    for i in range(obj_map.shape[0]):
        hits = np.argwhere(obj_map[i] == player_id)
        if len(hits) > 0:
            positions[i] = hits[0]  # (y, x)

    return positions


def plot_crafter_coverage_from_npz(data_path: str, save_path: str | None = None, title: str = "Crafter Exploration Coverage"):
    data = np.load(data_path, allow_pickle=True)

    if "a" not in data:
        raise KeyError(f"Dataset {data_path} has no key 'a' for observations")

    obs = data["a"]
    obs_nchw = _to_nchw(obs)
    h, w = obs_nchw.shape[2], obs_nchw.shape[3]

    positions = extract_player_positions(obs)
    heatmap = np.zeros((h, w), dtype=np.int64)

    for y, x in positions:
        if 0 <= y < h and 0 <= x < w:
            heatmap[y, x] += 1

    plt.figure(figsize=(7, 7))
    plt.imshow(heatmap, cmap="viridis", origin="upper")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Visit Count")

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Coverage saved to {out}")
    else:
        plt.show()

    plt.close()


# ============================================================
#  World Model Support Functions for Crafter (Classification)
# ============================================================
import torch
import torch.nn.functional as F

# CustomCrafterEnv Object IDs (matching original Crafter mat_ids + shifted entities):
# 0=None/empty, 1=water, 2=grass, 3=stone, 4=path, 5=sand, 6=tree,
# 7=lava, 8=coal, 9=iron, 10=diamond, 11=table, 12=furnace,
# 13=Player, 14=Cow, 15=Zombie, 16=Skeleton, 17=Arrow, 18=Plant, 19=Fence
# Total: 20 classes (0-19)
# Direction IDs: 0=none, 1=up, 2=down, 3=left, 4=right  →  5 classes
CRAFTER_OBJ_CLASSES = 20
CRAFTER_DIR_CLASSES = 5
CRAFTER_ACTION_NAMES = [
    'noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do', 'sleep',
    'place_stone', 'place_table', 'place_furnace', 'place_plant',
    'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe',
    'make_wood_sword', 'make_stone_sword', 'make_iron_sword'
]
_CLAMP_DEBUG_PRINTED = False


def crafter_clamp_targets(obj_true: torch.Tensor, dir_true: torch.Tensor):
    """
    Clamp object and direction targets to valid class ranges.
    Prints a one-time debug message if out-of-range values are detected.
    """
    global _CLAMP_DEBUG_PRINTED
    if not _CLAMP_DEBUG_PRINTED:
        obj_max = int(obj_true.max().item())
        dir_max = int(dir_true.max().item())
        obj_min = int(obj_true.min().item())
        dir_min = int(dir_true.min().item())
        if obj_max >= CRAFTER_OBJ_CLASSES or dir_max >= CRAFTER_DIR_CLASSES or obj_min < 0 or dir_min < 0:
            print(f"[CrafterClamp] WARNING: out-of-range targets detected! "
                  f"obj=[{obj_min},{obj_max}] (valid 0-{CRAFTER_OBJ_CLASSES-1}), "
                  f"dir=[{dir_min},{dir_max}] (valid 0-{CRAFTER_DIR_CLASSES-1}). Clamping.")
        else:
            print(f"[CrafterClamp] Targets in range: obj=[{obj_min},{obj_max}], dir=[{dir_min},{dir_max}]")
        _CLAMP_DEBUG_PRINTED = True
    obj_true = obj_true.clamp(0, CRAFTER_OBJ_CLASSES - 1)
    dir_true = dir_true.clamp(0, CRAFTER_DIR_CLASSES - 1)
    return obj_true, dir_true


def crafter_classification_loss(
    next_pred: torch.Tensor,
    next_true: torch.Tensor,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Compute per-location CrossEntropy loss for Crafter classification output.

    Args:
        next_pred: (B, 22, H, W) logits — first 17 = obj, last 5 = dir
        next_true: (B, 2, H, W)  ground truth — ch0=obj ID, ch1=dir ID
        reduction: 'none' => (B,H,W), 'mean' => scalar

    Returns:
        loss map or scalar depending on reduction
    """
    obj_logits = next_pred[:, :CRAFTER_OBJ_CLASSES]   # (B, 17, H, W)
    dir_logits = next_pred[:, CRAFTER_OBJ_CLASSES:]   # (B, 5, H, W)

    obj_true = next_true[:, 0].long()
    dir_true = next_true[:, 1].long()

    obj_true, dir_true = crafter_clamp_targets(obj_true, dir_true)

    loss_obj = F.cross_entropy(obj_logits, obj_true, reduction=reduction)
    loss_dir = F.cross_entropy(dir_logits, dir_true, reduction=reduction)

    total = loss_obj + loss_dir  # (B, H, W) or scalar

    if reduction == "mean":
        return total
    return total  # (B, H, W) per-location loss


def crafter_reconstruct_from_logits(next_pred: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct integer observation channels from logits using argmax.
    Used for pseudo-MSE metric logging.

    Args:
        next_pred: (B, 22, H, W)

    Returns:
        (B, 2, H, W) float tensor: [obj_id, dir_id]
    """
    obj_pred = next_pred[:, :CRAFTER_OBJ_CLASSES].argmax(dim=1).float()  # (B, H, W)
    dir_pred = next_pred[:, CRAFTER_OBJ_CLASSES:].argmax(dim=1).float()  # (B, H, W)
    return torch.stack([obj_pred, dir_pred], dim=1)  # (B, 2, H, W)
    
def visualize_crafter_wm(
    obs_masked: torch.Tensor, 
    obs_next_masked: torch.Tensor, 
    obs_pred_logits: torch.Tensor,
    action: int,
    step: int,
    save_dir: str = "trainer/logs/wm_visual",
    full_map_size: tuple = (64, 64),
    agent_pos: tuple = (32, 32),
    inv: np.ndarray = None,
    inv_next: np.ndarray = None
):
    """
    Visualize WM prediction for Crafter with 5 panels.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle batch dimension if present
    if obs_masked.ndim == 4: obs_masked = obs_masked[0]
    if obs_next_masked.ndim == 4: obs_next_masked = obs_next_masked[0]
    if obs_pred_logits.ndim == 4: obs_pred_logits = obs_pred_logits[0]
        
    # 1. 提取预测 ID 和 不确定性
    obj_logits = obs_pred_logits[:CRAFTER_OBJ_CLASSES]  
    obj_probs = F.softmax(obj_logits, dim=0)
    obj_pred = obj_probs.argmax(dim=0).detach().cpu().numpy()
    confidence = obj_probs.max(dim=0)[0].detach().cpu().numpy()
    uncertainty = 1.0 - confidence
    
    # 2. 获取真实观测
    obj_curr = obs_masked[0].detach().cpu().numpy()
    obj_next = obs_next_masked[0].detach().cpu().numpy()
    
    # 3. 计算 Error Map (必须在裁剪前，基于完整预测和真实值计算)
    error_map = (obj_pred != obj_next).astype(np.float32)

    # --- 改进：基于地图尺寸和 Agent 位置进行精确裁剪 ---
    mask_size = obs_masked.shape[-1]
    half = mask_size // 2
    # 确保坐标是整数
    ay, ax = int(agent_pos[0]), int(agent_pos[1])
    H, W = int(full_map_size[0]), int(full_map_size[1])
    
    y_start = half - ay
    y_end = y_start + H
    x_start = half - ax
    x_end = x_start + W
    
    # 确保切片不越界
    y_start, y_end = max(0, y_start), min(mask_size, y_end)
    x_start, x_end = max(0, x_start), min(mask_size, x_end)

    obj_curr = obj_curr[y_start:y_end, x_start:x_end]
    obj_pred = obj_pred[y_start:y_end, x_start:x_end]
    obj_next = obj_next[y_start:y_end, x_start:x_end]
    uncertainty = uncertainty[y_start:y_end, x_start:x_end]
    error_map = error_map[y_start:y_end, x_start:x_end]
    
    h_crop, w_crop = obj_curr.shape
        
    # --- 改进：定义符合直觉的 Crafter 颜色映射 ---
    # 0=Empty, 1=Water, 2=Grass, 3=Stone, 4=Path, 5=Sand, 6=Tree, 7=Lava, 8=Coal, 9=Iron, 10=Diamond, 11=Table, 12=Furnace
    # 13=Player, 14=Cow, 15=Zombie, 16=Skeleton, 17=Arrow, 18=Plant, 19=Fence
    from matplotlib.colors import ListedColormap
    colors = [
        '#000000', # 0: None (Black)
        '#1E90FF', # 1: Water (Dodger Blue)
        '#32CD32', # 2: Grass (Lime Green) - 改为绿色
        '#888888', # 3: Stone (Grey)
        '#964B00', # 4: Path (Brown)
        '#FFFFBB', # 5: Sand (Yellowish)
        '#006400', # 6: Tree (Dark Green)
        '#FFA500', # 7: Lava (Orange-Yellow) - 改为橙黄色，与 Agent 区分
        '#333333', # 8: Coal (Grey/Black)
        '#CCCCCC', # 9: Iron (Light Grey)
        '#00FFFF', # 10: Diamond (Cyan)
        '#774400', # 11: Table (Wood)
        '#331100', # 12: Furnace (Darker)
        '#FF0000', # 13: Player (Red)
        '#FFFFFF', # 14: Cow (White)
        '#9400D3', # 15: Zombie (Dark Violet) - 改为紫色，避开草地绿色
        '#EEEEEE', # 16: Skeleton (Bone)
        '#FFFF00', # 17: Arrow (Yellow)
        '#ADFF2F', # 18: Plant (Green Yellow)
        '#442200', # 19: Fence (Wood)
    ]
    # 如果类别数超过颜色数，补齐
    while len(colors) < CRAFTER_OBJ_CLASSES: colors.append('#333333')
    cmap = ListedColormap(colors)
    
    # 创建 2x3 的布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    
    # Panel 0: Current State
    axes[0].imshow(obj_curr, cmap=cmap, vmin=0, vmax=CRAFTER_OBJ_CLASSES-1)
    axes[0].set_title(f"Current Observed (T)")
    
    # Panel 1: Prediction (Argmax)
    act_name = CRAFTER_ACTION_NAMES[action] if action < len(CRAFTER_ACTION_NAMES) else f"act_{action}"
    axes[1].imshow(obj_pred, cmap=cmap, vmin=0, vmax=CRAFTER_OBJ_CLASSES-1)
    axes[1].set_title(f"WM Prediction (T+1) | {act_name}")
    
    # Panel 2: Ground Truth Next
    axes[2].imshow(obj_next, cmap=cmap, vmin=0, vmax=CRAFTER_OBJ_CLASSES-1)
    axes[2].set_title(f"True Observed (T+1)")
    
    # Panel 3: Error Map (Correctness)
    # 使用 binary 映射，黄色代表错，紫色代表对
    im3 = axes[3].imshow(error_map, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title(f"Error Map (Yellow=Wrong)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Panel 4: Uncertainty Heatmap
    im4 = axes[4].imshow(uncertainty, cmap="hot", vmin=0, vmax=1.0)
    axes[4].set_title(f"Uncertainty (Entropy/Confidence)")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    
    # Panel 5: Legend & Detail Stats (Text)
    axes[5].axis('off')
    mean_uncer = uncertainty.mean()
    total_errors = int(error_map.sum())
    acc = (1.0 - (total_errors / (h_crop * w_crop))) * 100 if (h_crop * w_crop) > 0 else 0
    
    inv_text = ""
    if inv is not None:
        inv_labels = ['health','food','drink','energy','wood','stone','coal','iron','diamond','sapling','w_pick','s_pick','i_pick','w_sword','s_sword','i_sword']
        inv_text = "Inventory (Now -> Next):\n"
        for i, val in enumerate(inv):
            if i >= len(inv_labels): break
            next_val = inv_next[i] if inv_next is not None else val
            if val > 0 or next_val > 0 or i < 4:
                diff_sym = "↑" if next_val > val else ("↓" if next_val < val else " ")
                inv_text += f"{inv_labels[i][:7]:7s}:{int(val):1d}->{int(next_val):1d}{diff_sym} "
                if i % 2 == 1: inv_text += "\n"

    stats_text = (
        f"Step: {step} | Act: {action} ({act_name})\n"
        f"Acc: {acc:.1f}% | Unc: {mean_uncer:.3f}\n"
        f"Mismatch: {total_errors}\n"
        f"{inv_text}"
    )
    axes[5].text(0.0, 1.0, stats_text, fontsize=11, family='monospace', verticalalignment='top')
    
    # --- Add Color Legend (More compact) ---
    import matplotlib.patches as mpatches
    legend_items = [
        ("Water", colors[1]), ("Grass", colors[2]), ("Stone", colors[3]),
        ("Path", colors[4]), ("Tree", colors[6]), ("Lava", colors[7]),
        ("Table", colors[11]), ("Player", colors[13]), ("Zomb", colors[15])
    ]
    for i, (label, color) in enumerate(legend_items):
        y_pos = 0.35 - (i // 3) * 0.1
        x_pos = (i % 3) * 0.33
        axes[5].add_patch(mpatches.Rectangle((x_pos, y_pos), 0.05, 0.08, color=color, transform=axes[5].transAxes))
        axes[5].text(x_pos + 0.07, y_pos + 0.02, label, fontsize=10, transform=axes[5].transAxes)
    
    # 给所有格子加网格线，清晰展示 layout
    for i in range(5):
        axes[i].set_xticks(np.arange(-.5, w_crop, 1), minor=True)
        axes[i].set_yticks(np.arange(-.5, h_crop, 1), minor=True)
        axes[i].grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
        # 隐藏刻度和标签，不使用 labelsize=0 以免触发警告
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].tick_params(which='both', size=0)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"step_{step:06d}.png")
    plt.savefig(save_path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Crafter NPZ exploration coverage.")
    parser.add_argument("data_path", help="Path to collected .npz dataset")
    parser.add_argument("--save", dest="save_path", default=None, help="Output image path (png)")
    parser.add_argument("--title", default="Crafter Exploration Coverage", help="Plot title")
    args = parser.parse_args()

    plot_crafter_coverage_from_npz(
        data_path=args.data_path,
        save_path=args.save_path,
        title=args.title,
    )


if __name__ == "__main__":
    main()

