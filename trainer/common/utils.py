import gc
import os
import random
import numpy as np
import torch
from modelBased.common.utils import TRAINER_PATH
from domain.minigrid.minigrid_support import extract_unique_patches, generate_minitasks_until_covered
from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from modelBased.data.data_collect import visualize_agent_coverage, visualize_saved_dataset
from modelBased.common.support import Support
from minigrid.wrappers import FullyObsWrapper
import csv
from modelBased.world_model import AttentionWM_training



def set_seed(seed: int):
    """Fix all random sources to ensure full reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Random seed fixed to {seed}]")


def count_data_in_dataset(file_name):
    """
    Input: dataset filename, e.g. `only_lava_minitask_test.npz`
    Output: number of samples, i.e. `data['a'].shape[0]`
    """
    data_path = TRAINER_PATH / 'data' / file_name
    if not os.path.exists(data_path):
        print(f"[Error] File not found: {data_path}")
        return None

    try:
        data = np.load(data_path, allow_pickle=True)
        num_samples = data['a'].shape[0]
        print(f"{file_name}: {num_samples} samples")
        return num_samples
    except Exception as e:
        print(f"[Error] Failed to read {file_name}: {e}")
        return None

def split_targets_into_minitasks(
    target_task_input, patch_size=3, patches_per_minitask=4, trials=200, add_agent_start=False
):
    """
    Extract patches from one or multiple target tasks, remove duplicates,
    and generate minitasks multiple times to return the smallest minitask set.

    Args:
        target_task_input: str or List[str]
        patch_size: int
        patches_per_minitask: int
        trials: how many times to run the generation (default=10)

    Returns:
        List[str]: the smallest minitask set found across trials
    """

    # --------- Normalize input to list ---------
    if isinstance(target_task_input, str):
        target_files = [target_task_input]
    elif isinstance(target_task_input, (list, tuple)):
        target_files = list(target_task_input)
    else:
        raise ValueError("target_task_input must be a str or list[str]")

    # print(f"[Patch Collect] Processing {len(target_files)} target tasks...")

    # --------- Collect patches from all targets ---------
    all_patches = set()

    for file in target_files:
        env = CustomMiniGridEnv(
            txt_file_path=TRAINER_PATH / "level" / file,
            custom_mission="Find the key and open the door.",
            max_steps=5000,
            render_mode=None
        )
        env.reset()
        layout_str = env.layout_str
        patches = extract_unique_patches(layout_str, patch_size)
        all_patches.update(patches)

    # print(f"[Patch Collect] Total unique patches across all targets = {len(all_patches)}")

    all_patches_list = list(all_patches)

    # ======================================================
    # Run multiple times and pick the smallest minitask set
    # ======================================================
    best_minitasks = None
    best_size = float("inf")

    # print(f"[Minitasks] Running {trials} trials to find minimal cover...")

    for t in range(trials):
        minitasks = generate_minitasks_until_covered(
            all_patches_list,
            patch_size,
            patches_per_minitask=patches_per_minitask,
            add_agent_start=add_agent_start
        )

        size = len(minitasks)
        # print(f"  Trial {t+1}/{trials}: {size} minitasks")

        if size < best_size:
            best_size = size
            best_minitasks = minitasks

    print(f"[Minitasks] Best result: {best_size} minitasks (out of {trials} trials)")
    return best_minitasks


def collect_data_general(
    cfg,
    env_source,
    save_name: str,
    max_steps: int = 10000,
    maximum_dataset_size: int = None,
    recollect_data: bool = False
):
    """
    General environment data-collection function.

    env_source can be:
        - str (ending with .txt): path to MiniGrid layout file
        - tuple(layout_str, color_str): minitask strings
    
    save_name: file prefix to save data, e.g. "lava_minitask"
    """
    if maximum_dataset_size is not None:
        cfg.env.collect.maximum_dataset_size = maximum_dataset_size
    support = Support(cfg)

    # -----------------------------
    # 0. Print environment info.
    # -----------------------------
    env_type = getattr(cfg.attention_model, "env_type", "")
    is_crafter = (env_type == "crafter")
    is_bipedal = (env_type == "bipedalwalker")

    # Sync env_type to cfg.env so _finalize_and_save can dispatch visualization correctly
    if env_type and not getattr(cfg.env, "env_type", ""):
        from omegaconf import open_dict
        with open_dict(cfg.env):
            cfg.env.env_type = env_type
    
    # -----------------------------
    # 1. Build environment
    # -----------------------------
    if is_crafter and isinstance(env_source, str):
        if env_source.endswith(".txt"):
            # From text file path
            env = support.wrap_env_from_text(env_source, max_steps=max_steps)
        else:
            # Multi-line string with layout + stats
            print(f"\n[Environment] Generating Crafter Task:\n{env_source}")
            from domain.crafter.crafter_custom_env import CustomCrafterEnv as CrafterEnv
            env = CrafterEnv(layout_str=env_source, max_steps=max_steps)

    elif not is_crafter and isinstance(env_source, (str, os.PathLike)) and str(env_source).endswith(".txt"):
        # MiniGrid/Bipedal from text file (resolved by Support)
        env = support.wrap_env_from_text(env_source, max_steps=max_steps)

    elif is_bipedal and isinstance(env_source, str):
        # Bipedal from layout string (e.g., "G20 S3 P4 T2")
        from domain.bipedalwalker.custom_bipedal_env import CustomBipedalEnv
        data_type = str(getattr(cfg.env.collect, "data_type", "")).lower()
        if getattr(cfg.env, "visualize", False):
            render_mode = "human"
        elif getattr(cfg.env.collect, "save_env_visualize", False) and data_type == "random":
            render_mode = "rgb_array"
        else:
            render_mode = None
        env = CustomBipedalEnv(render_mode=render_mode)
        env.set_custom_layout_from_str(env_source)
        cfg.env.collect.current_layout_str = env_source

    elif isinstance(env_source, tuple) and len(env_source) == 2:
        # MiniGrid from minitask strings (layout_str, color_str)
        layout_str, color_str = env_source
        render_mode = "human" if getattr(cfg.env, "visualize", False) else None
        env = FullyObsWrapper(CustomMiniGridEnv(
            layout_str=layout_str,
            color_str=color_str,
            custom_mission="Learn minitask",
            render_mode=render_mode,
            max_steps=max_steps,
        ))
    elif hasattr(env_source, "reset") and hasattr(env_source, "step"):
        # Accept pre-built env objects, which is how the bipedal UED rollout path
        # currently passes generator-produced tasks through Support.interpret_env.
        env = env_source
    else:
        # Fallback for other cases (e.g. direct env objects if supported later)
        if is_crafter:
             raise ValueError("For Crafter UED, env_source must be a .txt path or a layout string.")
        if is_bipedal:
             raise ValueError("For BipedalWalker UED, env_source must be a .txt path or a layout string.")
        raise ValueError("env_source must be a .txt filepath or (layout_str, color_str) tuple")

    if is_bipedal and hasattr(env, "layout_str"):
        cfg.env.collect.current_layout_str = env.layout_str

    # -----------------------------
    # 2. Set dataset save paths
    # -----------------------------
    from pathlib import Path
    
    data_save_dir = Path(getattr(cfg.env.collect, "data_folder", str(TRAINER_PATH / "data")))
    os.makedirs(data_save_dir, exist_ok=True)
    
    explore_type = cfg.env.collect.data_type  # random / uniform
    save_path = data_save_dir / f"{save_name}_test_{explore_type}.npz"

    cfg.env.collect.data_save_path = str(save_path)
    
    vis_dir = Path(getattr(cfg.env.collect, "visualize_save_path", str(TRAINER_PATH / "logs" / "dataset_visualization")))
    os.makedirs(vis_dir, exist_ok=True)
    cfg.env.collect.visualize_save_path = str(vis_dir)
    
    cfg.env.collect.visualize_filename = f"{save_name}_{explore_type}.png"
    cfg.env.collect.env_visualize_filename = f"collect_{save_name}_env.png"
    if not recollect_data and os.path.exists(save_path):
        print(f"[Data Collection] Skipped: {save_name} already exists → {save_path}")
        return save_path 
    # -----------------------------
    # 3. Delete old dataset file
    # -----------------------------
    support.del_env_data_file()

    # -----------------------------
    # 4. Run actual data collection
    # -----------------------------
    support.collect_data_trainer(
        env=env,
        wandb_run=None,
        validate=False,
        save_img=cfg.env.get('save_visualized_img', False),
        log_name=f"collect_{save_name}",
        max_steps=None,  # already set in env
    )

    print("Data collection complete!")
    return save_path

def create_data_subsets(dataset_npz, interval_size):
    """
    Shuffle and split dataset_npz into multiple subsets of size interval_size.
    Return a list of dict subsets: [{a,b,c,f}, ...]
    """

    obs_all = dataset_npz["a"]
    next_all = dataset_npz["b"]
    act_all = dataset_npz["c"]
    info_all = dataset_npz["f"] if "f" in dataset_npz else None
    inv_all = dataset_npz["g"] if "g" in dataset_npz else None
    inv_next_all = dataset_npz["h"] if "h" in dataset_npz else None

    total = len(obs_all)
    if interval_size is None:
        return [{
            "a": obs_all,
            "b": next_all,
            "c": act_all,
            "f": info_all,
            "g": inv_all,
            "h": inv_next_all,
        }]

    # ---- Shuffle ----
    indices = np.arange(total)
    np.random.shuffle(indices)

    obs_all = obs_all[indices]
    next_all = next_all[indices]
    act_all = act_all[indices]
    if info_all is not None:
        info_all = info_all[indices]
    if inv_all is not None:
        inv_all = inv_all[indices]
    if inv_next_all is not None:
        inv_next_all = inv_next_all[indices]

    # ---- Split into subsets ----
    subsets = []
    num_rounds = int(np.ceil(total / interval_size))

    for i in range(num_rounds):
        start = i * interval_size
        end = min((i + 1) * interval_size, total)

        subset = {
            "a": obs_all[start:end],
            "b": next_all[start:end],
            "c": act_all[start:end],
            "f": info_all[start:end] if info_all is not None else None,
            "g": inv_all[start:end] if inv_all is not None else None,
            "h": inv_next_all[start:end] if inv_next_all is not None else None,
        }

        subsets.append(subset)

    return subsets

def train_wm_with_subsets(
    cfg,
    net,
    subsets,
    fisher_buffer,
    temp_dir,
    num_iterations,
    old_params,
    fisher,
    current_sample_ratio,
    fisher_buffer_elements_ratio
):
    """
    Train WM on multiple subsets with Fisher-based replay.
    Keeps old_params/fisher across phases.
    Returns:
        old_params, fisher, total_transitions_used
    """

    phase_transitions_used = 0  

    for it in range(num_iterations):

        # -------- pick subset --------
        idx = it if it < len(subsets) else np.random.randint(len(subsets))
        subset = subsets[idx]

        # Count how many transitions used in this iteration
        transitions_this_iter = subset["a"].shape[0]
        phase_transitions_used += transitions_this_iter   # Accumulate transitions consumed in this phase.

        # ---- write subset to temp npz ----
        temp_path = os.path.join(temp_dir, f"subset_{idx}.npz")
        np.savez_compressed(temp_path, **subset)
        cfg.attention_model.data_dir = temp_path

        # ---- Prepare replay data ----
        replay_data = fisher_buffer.export_dict() if len(fisher_buffer) > 0 else None
        print(f"Using replay data with {len(fisher_buffer)} samples.")

        # ---- Train WM ----
        cfg.attention_model.freeze_weight = False
        old_params, fisher, net = AttentionWM_training.train_api(
            cfg,
            net,
            old_params,
            fisher,
            replay_data=replay_data
        )

        # ---- Update fisher buffer ----
        samples = {
            'obs': subset['a'],
            'obs_next': subset['b'],
            'act': subset['c'],
            'info': subset['f'],
            'inv': subset['g'],
            'inv_next': subset['h']
        }

        fisher_buffer.update_combined(samples, current_sample_ratio, fisher_buffer_elements_ratio)

        print(f"[WM] Iter {it+1}/{num_iterations} using subset {idx} "
              f"({transitions_this_iter} transitions)")

    # print(f"[WM] Total transitions used in this phase: {total_transitions_used}")

    return net, old_params, fisher, phase_transitions_used    

def validate_on_target_task(cfg, net, old_params, data_save_dir, target_file, phase_name="validation", VALID_TIMES=1):
    """
    Run WM validation on the fixed target task, return avg loss.
    Returns: (avg_mse_loss, avg_weighted_loss)
    """

    prev_freeze_weight = cfg.attention_model.freeze_weight
    prev_keep_cell_loss = cfg.attention_model.keep_cell_loss
    prev_data_dir = cfg.attention_model.data_dir

    cfg.attention_model.freeze_weight = True
    # Optional switch: default off for lightweight target validation.
    cfg.attention_model.keep_cell_loss = bool(
        getattr(cfg.attention_model, "target_validation_keep_cell_loss", False)
    )
    cfg.attention_model.data_dir = os.path.join(data_save_dir, target_file)

    losses = []
    inv_losses = []
    terrain_losses = []
    contact_accs = []
    contact_bces = []
    is_bipedal = (getattr(cfg.attention_model, "env_type", "") == "bipedalwalker")
    is_crafter = (getattr(cfg.attention_model, "env_type", "") == "crafter")

    try:
        for v in range(VALID_TIMES):
            # train_api in validation mode returns a dict where "avg_val_loss" holds the Lightning metrics
            val_res, _, model = AttentionWM_training.train_api(cfg, net, old_params, None)
            
            actual_val_out = val_res.get("avg_val_loss", {})
            
            if isinstance(actual_val_out, list) and len(actual_val_out) > 0:
                 metrics = actual_val_out[0]
            elif isinstance(actual_val_out, dict):
                 metrics = actual_val_out
            else:
                 metrics = {}

            # Map Crafter-specific classification CE metrics
            main_loss = float(metrics.get('avg_val_loss_wm', metrics.get('best_loss', 0.0)))
            t_loss = float(metrics.get('val/terrain_loss', metrics.get('val/ce_loss', main_loss)))
            i_loss = float(metrics.get('val/inventory_loss', metrics.get('val/inv_loss', 0.0)))
            contact_acc = float(metrics.get('val/contact_acc', 0.0))
            contact_bce = float(metrics.get('val/contact_bce', 0.0))
            
            losses.append(main_loss)
            terrain_losses.append(t_loss)
            inv_losses.append(i_loss)
            contact_accs.append(contact_acc)
            contact_bces.append(contact_bce)

            del model
            torch.cuda.empty_cache()
            gc.collect()
    finally:
        cfg.attention_model.freeze_weight = prev_freeze_weight
        cfg.attention_model.keep_cell_loss = prev_keep_cell_loss
        cfg.attention_model.data_dir = prev_data_dir

    result = {
        'avg_val_loss_wm': float(np.mean(losses)),
    }
    if is_crafter:
        result['terrain_loss'] = float(np.mean(terrain_losses))
        result['inventory_loss'] = float(np.mean(inv_losses))
    if is_bipedal:
        result['contact_acc'] = float(np.mean(contact_accs))
        result['contact_bce'] = float(np.mean(contact_bces))
    return result


def validate_on_all_targets(
    cfg,
    net,
    data_save_dir,
    target_names,
    val_suffix,
    phase_name="validation",
    VALID_TIMES=1,
    disable_wandb=True,
):
    """
    Unified multi-target validation helper used by baseline/P2E/DR.
    `target_names` may be bare task names or `.txt` names; `val_suffix` is appended to the base name.
    Returns aggregated means plus per-target results.
    """
    old_use_wandb = getattr(cfg.attention_model, "use_wandb", False)
    if disable_wandb:
        cfg.attention_model.use_wandb = False

    losses = []
    inv_losses = []
    terrain_losses = []
    contact_accs = []
    contact_bces = []
    per_target = {}
    valid_count = 0
    is_bipedal = (getattr(cfg.attention_model, "env_type", "") == "bipedalwalker")
    is_crafter = (getattr(cfg.attention_model, "env_type", "") == "crafter")

    try:
        for task_name in target_names:
            task_base = task_name[:-4] if str(task_name).endswith(".txt") else str(task_name)
            val_file = f"{task_base}{val_suffix}"
            val_path = os.path.join(data_save_dir, val_file)
            if not os.path.exists(val_path):
                continue

            res = validate_on_target_task(
                cfg,
                net,
                None,
                data_save_dir,
                val_file,
                phase_name=phase_name,
                VALID_TIMES=VALID_TIMES,
            )
            if not res:
                continue

            l_val = float(res.get("avg_val_loss_wm", 0.0))
            ce_or_terrain = float(res.get("terrain_loss", 0.0))
            inv_val = float(res.get("inventory_loss", 0.0))
            c_acc = float(res.get("contact_acc", 0.0))
            c_bce = float(res.get("contact_bce", 0.0))

            losses.append(l_val)
            terrain_losses.append(ce_or_terrain)
            inv_losses.append(inv_val)
            contact_accs.append(c_acc)
            contact_bces.append(c_bce)
            per_target[task_base] = {
                "avg_val_loss_wm": l_val,
                "terrain_loss": ce_or_terrain,
                "inventory_loss": inv_val,
                "contact_acc": c_acc,
                "contact_bce": c_bce,
            }
            valid_count += 1
    finally:
        if disable_wandb:
            cfg.attention_model.use_wandb = old_use_wandb

    result = {
        "valid_count": valid_count,
        "avg_val_loss_wm": float(np.mean(losses)) if valid_count > 0 else 0.0,
        "per_target": per_target,
    }
    if is_crafter:
        result["terrain_loss"] = float(np.mean(terrain_losses)) if valid_count > 0 else 0.0
        result["inventory_loss"] = float(np.mean(inv_losses)) if valid_count > 0 else 0.0
    elif is_bipedal:
        result["contact_acc"] = float(np.mean(contact_accs)) if valid_count > 0 else 0.0
        result["contact_bce"] = float(np.mean(contact_bces)) if valid_count > 0 else 0.0
    else:
        result["terrain_loss"] = float(np.mean(terrain_losses)) if valid_count > 0 else 0.0
        result["inventory_loss"] = float(np.mean(inv_losses)) if valid_count > 0 else 0.0
    return result


def plot_loss_heatmap(
    loss_map: np.ndarray,
    save_path: str,
    phase_name: str = "",
    target_file: str = "",
    cmap: str = "viridis_r"
):
    """
    Plot and save a heatmap of the loss map.

    Args:
        loss_map (np.ndarray): 2D array representing loss over grid.
        save_path (str): path to save the heatmap PNG.
        phase_name (str): title / phase name to show on the plot.
        cmap (str): Matplotlib colormap.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    plt.imshow(loss_map, cmap=cmap, interpolation="nearest")

    plt.colorbar(label="Average Loss Value")

    title = f"Average Loss Map Heatmap ({phase_name}) - {target_file}" if phase_name else "Loss Map Heatmap"
    plt.title(title)

    plt.xlabel("X Position (columns)")
    plt.ylabel("Y Position (rows)")

    # ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Heatmap Saved] {save_path}")

def save_validation_csv(csv_path, seed, mode, phase_name, transitions, loss):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['seed', 'mode', 'phase', 'transitions', 'avg_target_loss']
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'seed': seed,
            'mode': mode,
            'phase': phase_name,
            'transitions': transitions,
            'avg_target_loss': loss,
        })

def extract_loss_map_over_validations(
    cfg,
    net,
    old_params,
    data_dir: str,
    valid_times: int = 10
):
    """
    Run WM validation multiple times and accumulate + average loss maps.

    Args:
        cfg: hydra config
        old_params: parameters of the world model (for validation)
        data_dir (str): path to target npz file
        valid_times (int): number of validation rounds

    Returns:
        avg_loss_map (np.ndarray): averaged 2D loss map
        avg_losses (List[float]): list of scalar avg_val_loss_wm for each run
    """

    import numpy as np
    import gc
    import torch
    from modelBased.world_model import AttentionWM_training

    prev_freeze_weight = cfg.attention_model.freeze_weight
    prev_keep_cell_loss = cfg.attention_model.keep_cell_loss
    prev_data_dir = cfg.attention_model.data_dir

    # Enable world-model validation mode while keeping per-cell loss for heatmaps.
    cfg.attention_model.freeze_weight = True
    cfg.attention_model.keep_cell_loss = True
    cfg.attention_model.data_dir = data_dir

    sum_map = None
    loss_list = []
    terrain_loss_list = []
    inv_loss_list = []
    contact_acc_list = []
    contact_bce_list = []
    inv_vectors = []
    bipedal_token_vectors = []

    env_type = getattr(cfg.attention_model, "env_type", "")
    is_crafter = (env_type == "crafter")
    is_bipedal = (env_type == "bipedalwalker")

    try:
        for _ in range(valid_times):
            # Run one validation pass.
            val_result, _, model = AttentionWM_training.train_api(cfg, net, old_params, None)
            loss_map = model.loss_map_result  # (H,W) array

            if sum_map is None:
                sum_map = np.array(loss_map, dtype=np.float32)
            else:
                sum_map += loss_map

            # train_api returns a dict with "avg_val_loss" containing the actual lightning output
            # e.g., result["avg_val_loss"] = [{'avg_val_loss_wm': 1.2, 'val/terrain_loss': 0.8...}]
            actual_val_out = val_result.get("avg_val_loss", {})
            
            # Robust result parsing for both list and dict formats from Lightning's trainer.validate
            if isinstance(actual_val_out, list) and len(actual_val_out) > 0:
                 metrics = actual_val_out[0]
            elif isinstance(actual_val_out, dict):
                 metrics = actual_val_out
            else:
                 metrics = {}
            
            # Primary metrics: Classification cross-entropy
            total_l = float(metrics.get('avg_val_loss_wm', metrics.get('best_loss', 0.0)))
            t_l = float(metrics.get('val/terrain_loss', metrics.get('val/ce_loss', total_l)))
            i_l = float(metrics.get('val/inventory_loss', metrics.get('val/inv_loss', 0.0)))
            c_acc = float(metrics.get('val/contact_acc', 0.0))
            c_bce = float(metrics.get('val/contact_bce', 0.0))
            
            loss_list.append(total_l)
            terrain_loss_list.append(t_l)
            inv_loss_list.append(i_l)
            contact_acc_list.append(c_acc)
            contact_bce_list.append(c_bce)
            inv_vec = getattr(model, "inventory_loss_vector_result", None)
            if inv_vec is not None:
                inv_vectors.append(np.asarray(inv_vec, dtype=np.float32))
            if is_bipedal:
                token_metric_keys = [
                    "val/token_hull_pose_mse",
                    "val/token_hull_vel_mse",
                    "val/token_leg1_hip_mse",
                    "val/token_leg1_knee_mse",
                    "val/token_leg1_contact_bce",
                    "val/token_leg2_hip_mse",
                    "val/token_leg2_knee_mse",
                    "val/token_leg2_contact_bce",
                    "val/token_lidar_near_mse",
                    "val/token_lidar_far_mse",
                ]
                token_vector = np.array(
                    [float(metrics.get(key, 0.0)) for key in token_metric_keys],
                    dtype=np.float32,
                )
                bipedal_token_vectors.append(token_vector)

            # Cleanup
            del model
            torch.cuda.empty_cache()
            gc.collect()
    finally:
        cfg.attention_model.freeze_weight = prev_freeze_weight
        cfg.attention_model.keep_cell_loss = prev_keep_cell_loss
        cfg.attention_model.data_dir = prev_data_dir

    # Compute the mean loss map.
    avg_loss_map = sum_map / valid_times

    if is_crafter:
        # Use slot-wise inventory error vector if available.
        # Fallback: keep legacy scalar->vector behavior for compatibility.
        if len(inv_vectors) > 0:
            inventory_pattern = np.mean(np.stack(inv_vectors, axis=0), axis=0).astype(np.float32)
        else:
            avg_inv_total = np.mean(inv_loss_list)
            inventory_pattern = np.full(16, avg_inv_total / 10.0, dtype=np.float32)
        
        return {"terrain": avg_loss_map, "inventory": inventory_pattern}, loss_list, terrain_loss_list, inv_loss_list

    if is_bipedal:
        if len(bipedal_token_vectors) > 0:
            token_error_vector = np.mean(np.stack(bipedal_token_vectors, axis=0), axis=0).astype(np.float32)
        else:
            token_error_vector = np.zeros(10, dtype=np.float32)
        return {"terrain": avg_loss_map, "inventory": token_error_vector}, loss_list, contact_acc_list, contact_bce_list
    
    return avg_loss_map, loss_list, [], []

def convert_trajectories_to_batch(trajectories):
    """
    convert_trajectories_to_batch: make a list of trajectories into a dict of numpy arrays
    Input:
      - List of trajectories (legacy): each traj is List[(state, action, reward, next_state, done, info)]
      - List of dicts (new): each dict has 'obs', 'act', 'obs_next', 'info' as tensors/arrays
    
    Output:
      - dict with keys: 'obs', 'obs_next', 'act', 'info' (numpy arrays)
    """
    # 1. New Format: List of Dicts (already batched per trajectory)
    if len(trajectories) > 0 and isinstance(trajectories[0], dict):
        # trajectories is a list of dicts. We need to concatenate them.
        # Assuming each dict contains tensors/arrays for a single trajectory.
        
        obs_list = []
        act_list = []
        next_obs_list = []
        rew_list = []
        done_list = []
        info_list = []
        inv_list = []
        inv_next_list = []
        
        has_info = 'info' in trajectories[0] and trajectories[0]['info'] is not None
        has_inv = 'inv' in trajectories[0] and trajectories[0]['inv'] is not None
        has_inv_next = 'inv_next' in trajectories[0] and trajectories[0]['inv_next'] is not None

        for traj in trajectories:
            def to_numpy(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                return np.array(x)

            obs_list.append(to_numpy(traj['obs']))
            act_list.append(to_numpy(traj['act']))
            next_obs_list.append(to_numpy(traj['obs_next']))
            if 'rew' in traj: rew_list.append(to_numpy(traj['rew']))
            if 'done' in traj: done_list.append(to_numpy(traj['done']))
            if has_info:
                info_list.append(to_numpy(traj['info']))
            if has_inv:
                inv_list.append(to_numpy(traj['inv']))
            if has_inv_next:
                inv_next_list.append(to_numpy(traj['inv_next']))

        obs = np.concatenate(obs_list, axis=0)
        obs_next = np.concatenate(next_obs_list, axis=0)
        act = np.concatenate(act_list, axis=0)
        rew = np.concatenate(rew_list, axis=0) if rew_list else None
        done = np.concatenate(done_list, axis=0) if done_list else None
        info = np.concatenate(info_list, axis=0) if has_info else None
        inv = np.concatenate(inv_list, axis=0) if has_inv else None
        inv_next = np.concatenate(inv_next_list, axis=0) if has_inv_next else None

        # Keep both legacy training keys (a-h) and readable aliases for callers
        # such as fisher buffer utilities that still expect obs/act naming.
        return {
            'a': obs,
            'b': obs_next,
            'c': act,
            'd': rew,
            'e': done,
            'f': info,
            'g': inv,
            'h': inv_next,
            'obs': obs,
            'obs_next': obs_next,
            'act': act,
            'rew': rew,
            'done': done,
            'info': info,
            'inv': inv,
            'inv_next': inv_next,
        }

    # 2. Legacy Format: List of Lists of Tuples
    obs_list, act_list, next_obs_list, rew_list, done_list = [], [], [], [], []
    
    for traj in trajectories:
        for step in traj:
            state, action, reward, next_state, done, info = step
            obs_list.append(np.array(state))
            act_list.append(np.array(action))
            next_obs_list.append(np.array(next_state))
            rew_list.append(reward)
            done_list.append(done)
            
    obs = np.array(obs_list)
    obs_next = np.array(next_obs_list)
    act = np.array(act_list)
    rew = np.array(rew_list)
    done = np.array(done_list)

    return {
        'a': obs,
        'b': obs_next,
        'c': act,
        'd': rew,
        'e': done,
        'f': None,
        'g': None,
        'h': None,
        'obs': obs,
        'obs_next': obs_next,
        'act': act,
        'rew': rew,
        'done': done,
        'info': None,
        'inv': None,
        'inv_next': None,
    }
