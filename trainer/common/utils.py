import gc
import os
import random
import numpy as np
import torch
from modelBased.common.utils import TRAINER_PATH, extract_unique_patches, generate_minitasks_until_covered
from domain.minigrid_custom_env import CustomMiniGridEnv
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
    输入: data 文件名（例如 'only_lava_minitask_test.npz'）
    输出: 样本数量（data['a'].shape[0]）
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
    cfg.env.collect.maximum_dataset_size = maximum_dataset_size
    support = Support(cfg)

    # -----------------------------
    # 1. Build environment
    # -----------------------------
    if isinstance(env_source, (str, os.PathLike)) and str(env_source).endswith(".txt"):
        # From text file
        env = support.wrap_env_from_text(env_source, max_steps=max_steps)

    elif isinstance(env_source, tuple) and len(env_source) == 2:
        # From minitask strings
        layout_str, color_str = env_source

        env = FullyObsWrapper(CustomMiniGridEnv(
            layout_str=layout_str,
            color_str=color_str,
            custom_mission="Learn minitask",
            render_mode=None,
            max_steps=max_steps,
        ))
    else:
        raise ValueError("env_source must be a .txt filepath or (layout_str, color_str) tuple")

    # -----------------------------
    # 2. Set dataset save paths
    # -----------------------------
    data_save_dir = TRAINER_PATH / "data"
    explore_type = cfg.env.collect.data_type  # random / uniform
    save_path = data_save_dir / f"{save_name}_test_{explore_type}.npz"

    cfg.env.collect.data_save_path = str(save_path)
    cfg.env.collect.visualize_save_path = TRAINER_PATH / "logs" / "dataset_visualization"
    cfg.env.collect.visualize_filename = f"{save_name}_{explore_type}.png"
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
        save_img=False,
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

    total = len(obs_all)
    if interval_size is None:
        return [{
            "a": obs_all,
            "b": next_all,
            "c": act_all,
            "f": info_all,
        }]

    # ---- Shuffle ----
    indices = np.arange(total)
    np.random.shuffle(indices)

    obs_all = obs_all[indices]
    next_all = next_all[indices]
    act_all = act_all[indices]
    if info_all is not None:
        info_all = info_all[indices]

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
        phase_transitions_used += transitions_this_iter   # <--- 统计累计使用 transitions

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
            'info': subset['f']
        }

        fisher_buffer.update_combined(samples, current_sample_ratio, fisher_buffer_elements_ratio)

        print(f"[WM] Iter {it+1}/{num_iterations} using subset {idx} "
              f"({transitions_this_iter} transitions)")

    # print(f"[WM] Total transitions used in this phase: {total_transitions_used}")

    return net, old_params, fisher, phase_transitions_used    

def validate_on_target_task(cfg, net, old_params, data_save_dir, target_file, phase_name, VALID_TIMES=1):
    """
    Run WM validation on the fixed target task, return avg loss.
    Save no heatmap here (can add if needed).
    """

    cfg.attention_model.freeze_weight = True
    cfg.attention_model.keep_cell_loss = True
    cfg.attention_model.data_dir = os.path.join(data_save_dir, target_file)

    losses = []

    for v in range(VALID_TIMES):
        val_result, _, model = AttentionWM_training.train_api(cfg, net, old_params, None)
        loss_val = float(val_result[0]['avg_val_loss_wm'])
        losses.append(loss_val)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    cfg.attention_model.keep_cell_loss = False

    avg_loss = float(np.mean(losses))
    return avg_loss


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

    # Set WM to validation mode
    cfg.attention_model.freeze_weight = True
    cfg.attention_model.keep_cell_loss = True
    cfg.attention_model.data_dir = data_dir

    sum_map = None
    loss_list = []

    for _ in range(valid_times):
        # Run one validation
        val_result, model = AttentionWM_training.train_api(cfg, net, old_params, None)
        loss_map = model.loss_map_result  # (H,W) array

        # Accumulate map
        if sum_map is None:
            sum_map = np.array(loss_map, dtype=np.float32)
        else:
            sum_map += loss_map

        # Record scalar loss
        loss_val = float(val_result[0]['avg_val_loss_wm'])
        loss_list.append(loss_val)

        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Disable special val mode
    cfg.attention_model.keep_cell_loss = False

    # Compute average loss map
    avg_loss_map = sum_map / valid_times

    return avg_loss_map, loss_list

def convert_trajectories_to_batch(trajectories):
    """
    convert_trajectories_to_batch: make a list of trajectories into a dict of numpy arrays
    Each trajectory is a list of (state, action, reward, next_state, done, info) tuples.
    The output dict has keys 'a' (obs), 'b' (next_obs), 'c' (action).   
    """
    obs_list, act_list, next_obs_list = [], [], []
    
    for traj in trajectories:
        for step in traj:
            state, action, reward, next_state, done, info = step
            obs_list.append(np.array(state))
            act_list.append(np.array(action))
            next_obs_list.append(np.array(next_state))
            
    return {
        'a': np.array(obs_list),     
        'b': np.array(next_obs_list), 
        'c': np.array(act_list),      
    }